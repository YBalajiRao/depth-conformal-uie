"""
Per-image metrics matrix for SS-PGN (or any method's saved outputs) on a split manifest.

Modes
  model mode   --ckpt <run dir | last.pt>   runs SS-PGN (PadCropWrapper, no resizing) and also
               computes the reliability metrics from its NIG output.
  output mode  --pred-dir <folder>          scores pre-computed outputs of another method (files
               named like the unit's filename). Same metric code, so baselines are scored
               identically; no reliability metrics.

Metrics per image
  Full-reference (enhanced vs reference), on the 8-bit output, i.e. exactly what a saved PNG holds:
    PSNR (RGB, peak 255); SSIM (skimage, RGB, Gaussian window sigma=1.5, Wang et al. constants);
    LPIPS (lpips v0.1, AlexNet, inputs scaled to [-1,1]).
  No-reference (src/validation/uw_metrics.py): UIQM (+UICM/UISM/UIConM), UCIQE. Also computed for
    the raw input and the reference with --score-input-ref (useful context rows).
  Reliability (model mode), on the float prediction mu:
    Spearman(sigma, |Y - J_hat|) over pixels x channels (200k-element seeded subsample);
    AUSE-MAE / AUSE-RMSE (normalised; Ilg et al. 2018);
    with --bounds: calibrated half-width q_hat*sigma, the pixel coverage fraction, and the image-level
    event s_i <= q_hat (the quantity the conformal guarantee is about).
    Spearman and AUSE are invariant to the positive scaling q_hat, so calibration does not change them.

Test-set discipline
  --split test requires --final and appends an entry to data/calibration/test_access_log.jsonl
  (time, checkpoint SHA-256, command line). Use --split val during development.

Outputs (<out>/): per_image.csv, summary.json (mean/std/n overall and per dataset), console matrix.

  python src/validation/evaluate_metrics.py --ckpt runs/sspgn/<run> --split val --bounds data/calibration/calib_bounds.json
  python src/validation/evaluate_metrics.py --ckpt runs/sspgn/<run> --split test --final --bounds ...
  python src/validation/evaluate_metrics.py --pred-dir outputs/PSPL_uwcnn/test --split test --final --name PSPL-UWCNN
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/validation"))
from calibrate_uncertainty import (read_manifest, load_model, pick_device, predictive_std, sparsification,  # noqa: E402
                                   ause, spearman, file_sha256)
from uw_metrics import uiqm, uciqe  # noqa: E402
sys.path.insert(0, str(ROOT / "src/utils"))
from pad_crop_wrapper import PadCropWrapper  # noqa: E402


def load_u8(p) -> np.ndarray:
    from PIL import Image
    with Image.open(p) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)


def psnr_u8(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def ssim_u8(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=2, data_range=255, gaussian_weights=True,
                                       sigma=1.5, use_sample_covariance=False))


class LPIPSMetric:
    def __init__(self, device):
        import lpips
        self.net = lpips.LPIPS(net="alex", version="0.1", verbose=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, a, b):
        t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(self.device)
        return float(self.net(t(a), t(b)).item())


def nr_metrics(img, prefix=""):
    q, c = uiqm(img), uciqe(img)
    return {f"{prefix}uiqm": q["uiqm"], f"{prefix}uicm": q["uicm"], f"{prefix}uism": q["uism"],
            f"{prefix}uiconm": q["uiconm"], f"{prefix}uciqe": c["uciqe"]}


def log_test_access(args, ckpt_sha):
    p = ROOT / "data/calibration/test_access_log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "ckpt_sha256": ckpt_sha,
                            "pred_dir": args.pred_dir, "argv": sys.argv}) + "\n")
    n = sum(1 for _ in open(p, encoding="utf-8"))
    print(f"[test] access #{n} logged to {p}")
    if n > 1:
        print("[test] NOTE: the test split has been evaluated before; report every access, or pre-register which one counts.")


def summarise(rows, keys):
    def stats(rs):
        out = {}
        for k in keys:
            v = np.array([r[k] for r in rs if k in r and r[k] is not None and np.isfinite(r[k])], dtype=np.float64)
            out[k] = {"mean": float(v.mean()) if v.size else None, "std": float(v.std(ddof=1)) if v.size > 1 else None,
                      "n": int(v.size)}
        return out
    res = {"ALL": stats(rows)}
    for d in sorted({r["dataset"] for r in rows}):
        res[d] = stats([r for r in rows if r["dataset"] == d])
    return res


def print_matrix(summary, keys, title):
    groups = list(summary)
    print(f"\n{title}")
    print(f"  {'metric':22s}" + "".join(f"{g:>20s}" for g in groups))
    for k in keys:
        cells = []
        for g in groups:
            s = summary[g][k]
            cells.append("-" if s["mean"] is None else f"{s['mean']:.4f}+-{(s['std'] or 0):.4f}")
        print(f"  {k:22s}" + "".join(f"{c:>20s}" for c in cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="run dir (-> last.pt) or checkpoint file")
    src.add_argument("--pred-dir", help="folder with another method's outputs, named by unit filename")
    ap.add_argument("--name", default=None, help="method name for the report")
    ap.add_argument("--split-dir", default=str(ROOT / "data/splits/lsui_uieb_all_d6_seed0"))
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--final", action="store_true", help="required for --split test")
    ap.add_argument("--bounds", default=None, help="calib_bounds.json (model mode)")
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--var", default=None, help="override variance kind (default: the one in --bounds, else total)")
    ap.add_argument("--score-input-ref", action="store_true", help="also UIQM/UCIQE of raw input and reference")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.split == "test" and not args.final:
        raise RuntimeError("--split test requires --final (test-set discipline). Use --split val while developing.")
    device = pick_device(args.device)
    manifest = Path(args.split_dir) / f"{args.split}_manifest.txt"
    rows_in = read_manifest(manifest, allow_test=args.split == "test")
    if args.max_images > 0:
        rows_in = rows_in[:args.max_images]

    model = info = bounds = q_hat = None
    ckpt_sha = None
    var_kind = args.var or "total"
    if args.ckpt:
        ckpt = Path(args.ckpt)
        ckpt = ckpt / "last.pt" if ckpt.is_dir() else ckpt
        ckpt_sha = file_sha256(ckpt)
        model, info = load_model(ckpt, device)
        name = args.name or f"SS-PGN[{info['arch']}]@{info['step']}"
        if args.bounds:
            bounds = json.loads(Path(args.bounds).read_text(encoding="utf-8"))
            if bounds["ckpt_sha256"] != ckpt_sha:
                raise RuntimeError(f"bounds were calibrated for checkpoint {bounds['ckpt_sha256'][:12]} "
                                   f"but --ckpt is {ckpt_sha[:12]}; recalibrate.")
            var_kind = args.var or bounds["var"]
            if var_kind != bounds["var"]:
                raise RuntimeError(f"--var {var_kind} differs from calibrated variance kind {bounds['var']}")
            try:
                q_hat = float(bounds["q_hat"][str(args.gamma)][str(args.alpha)])
            except KeyError:
                raise RuntimeError(f"no q_hat for gamma={args.gamma}, alpha={args.alpha} in {args.bounds}")
            if not math.isfinite(q_hat):
                raise RuntimeError(f"q_hat is infinite for gamma={args.gamma}, alpha={args.alpha} (too few calibration images)")
        wrapped = PadCropWrapper(model, model.cfg.multiple, mode="reflect")
    else:
        name = args.name or Path(args.pred_dir).name
        if not Path(args.pred_dir).is_dir():
            raise FileNotFoundError(args.pred_dir)
    if args.split == "test":
        log_test_access(args, ckpt_sha)

    lp = None if args.no_lpips else LPIPSMetric(device)
    out = Path(args.out) if args.out else ROOT / "reports/evaluation" / f"{name.replace('/', '_')}_{args.split}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[eval] {name} on {args.split} ({len(rows_in)} images) device={device} var={var_kind} "
          f"q_hat={q_hat if q_hat is not None else '-'} -> {out}")

    recs, t0, missing = [], time.time(), 0
    for i, r in enumerate(rows_in):
        y8 = load_u8(r["reference"])
        rec = {"dataset": r["dataset"], "unit": r["unit"], "H": y8.shape[0], "W": y8.shape[1]}
        if model is not None:
            x = torch.from_numpy(load_u8(r["input"])).permute(2, 0, 1)[None].float().div(255).to(device)
            if tuple(x.shape[-2:]) != y8.shape[:2]:
                raise ValueError(f"{r['unit']}: input/reference size mismatch")
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    o = wrapped(x)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); print(f"  [skip OOM] {r['unit']}"); continue
            nig = o["nig"].float()
            assert nig.shape == (1, 12, *y8.shape[:2]), (r["unit"], tuple(nig.shape))
            mu = nig[:, 0:3]
            pred8 = (mu.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
            y = torch.from_numpy(y8).permute(2, 0, 1)[None].float().div(255).to(device)
            sig = predictive_std(nig[:, 3:6], nig[:, 6:9], nig[:, 9:12], var_kind).flatten().cpu()
            err = (mu - y).abs().flatten().cpu()
            sp = sparsification(err, sig)
            rec.update(spearman_sigma_err=spearman(sig, err), ause_mae=ause(sp, "mae"), ause_rmse=ause(sp, "rmse"),
                       sigma_median=float(sig.median()))
            if q_hat is not None:
                rr = err / sig
                s_i = float(torch.quantile(rr.double(), args.gamma)) if rr.numel() <= 16_000_000 \
                    else float(np.quantile(rr.numpy(), args.gamma))
                rec.update(halfwidth_median=float(q_hat * sig.median()),
                           pixel_coverage=float((rr <= q_hat).double().mean()),
                           event_covered=float(s_i <= q_hat))
        else:
            fn = Path(r["reference"]).name
            cands = [Path(args.pred_dir) / fn] + list(Path(args.pred_dir).glob(Path(fn).stem + ".*"))
            p = next((c for c in cands if c.is_file()), None)
            if p is None:
                missing += 1; continue
            pred8 = load_u8(p)
            if pred8.shape != y8.shape:
                raise ValueError(f"{r['unit']}: prediction {pred8.shape} != reference {y8.shape} "
                                 "(resized outputs cannot be scored pixel-wise; regenerate at native size)")
        rec.update(psnr=psnr_u8(pred8, y8), ssim=ssim_u8(pred8, y8))
        if lp is not None:
            rec["lpips"] = lp(pred8, y8)
        rec.update(nr_metrics(pred8))
        if args.score_input_ref:
            rec.update(nr_metrics(load_u8(r["input"]), "input_"))
            rec.update(nr_metrics(y8, "ref_"))
        recs.append(rec)
        if (i + 1) % 25 == 0 or i + 1 == len(rows_in):
            print(f"  [{i + 1}/{len(rows_in)}] {time.time() - t0:.0f}s")

    if not recs:
        raise RuntimeError("no images scored")
    keys = [k for k in ["psnr", "ssim", "lpips", "uiqm", "uciqe", "spearman_sigma_err", "ause_mae", "ause_rmse",
                        "pixel_coverage", "event_covered", "halfwidth_median",
                        "input_uiqm", "input_uciqe", "ref_uiqm", "ref_uciqe"] if k in recs[0]]
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader(); w.writerows(recs)
    summary = summarise(recs, keys)
    meta = {"method": name, "split": args.split, "manifest": str(manifest), "n_scored": len(recs),
            "n_missing_predictions": missing, "model": info, "ckpt_sha256": ckpt_sha, "var": var_kind,
            "gamma": args.gamma if q_hat is not None else None, "alpha": args.alpha if q_hat is not None else None,
            "q_hat": q_hat, "metric_notes": {
                "fr": "PSNR/SSIM/LPIPS on 8-bit outputs at native resolution",
                "nr": "UIQM = FUnIE-GAN port (variant funie, blue weight 0.144); UCIQE = JOU-UIP port",
                "event_covered": f"mean = empirical P[>= {args.gamma} of pixels covered]; target >= {1 - args.alpha}"
                if q_hat is not None else None}}
    (out / "summary.json").write_text(json.dumps({"meta": meta, "summary": summary}, indent=2), encoding="utf-8")
    print_matrix(summary, keys, f"METRICS MATRIX: {name} on {args.split} (mean+-std over images)")
    if missing:
        print(f"WARNING: {missing} predictions missing in {args.pred_dir}")
    print(f"[out] {out}")


if __name__ == "__main__":
    main()

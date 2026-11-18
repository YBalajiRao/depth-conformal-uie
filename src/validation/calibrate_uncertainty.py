"""
Split-conformal calibration of SS-PGN per-pixel uncertainty (formulation: §6.1 of
reports/research_formulation/SS-PGN_gaps_objectives_methodology_2026-11-06.md).

Pipeline
  1. Load a checkpoint (best.pt / last.pt) and rebuild its architecture from the stored config.
  2. Run full-resolution inference (PadCropWrapper, no resizing) on the held-out images of a
     manifest, extracting the 12-channel NIG tensor [mu | nu | alpha | beta].
  3. Per pixel and channel, the uncalibrated predictive std sigma_hat is taken from the
     Student-t predictive (Amini et al. 2020):
         total      Var = beta (1 + nu) / (nu (alpha - 1))      (default)
         aleatoric  Var = beta / (alpha - 1)
         epistemic  Var = beta / (nu (alpha - 1))
     Normalised residual r = |y - mu| / sigma_hat.
  4. Image-level score  s_i(gamma) = Q_gamma({r_i(p, c)})  (quantile over all pixels and channels).
  5. Split by duplicate component into a CALIBRATION part and a VERIFICATION part. Then
         q_hat = s_(ceil((n+1)(1-alpha)))  over the calibration images,
     and the guarantee (exchangeable images, fixed model) is
         P[ at least a fraction gamma of test pixels satisfy |y - mu| <= q_hat * sigma_hat ] >= 1 - alpha.
     It is checked empirically on the verification part, for a gamma x alpha grid.
  6. Reports: coverage matrices (image-level event rate, mean pixel coverage, interval width),
     per-dataset breakdown, sparsification curves and AUSE (MAE and RMSE; Ilg et al., ECCV 2018),
     and Spearman(sigma, |err|).

Validity notes, also written to the report
  * If the checkpoint was SELECTED on this same split (best.pt chosen by val PSNR), the
    calibration scores are not independent of the model and the finite-sample guarantee is
    only approximate. For the paper, use a checkpoint chosen by a fixed rule (e.g. last.pt at
    the final step), or a split never used for selection.
  * Calibration and verification are split by duplicate COMPONENT (from the manifest), so no
    near-duplicate image straddles the two parts.
  * Exchangeability is assumed between calibration and verification images. Under dataset shift
    (e.g. calibrate on LSUI, test on UIEB) coverage is measured, not guaranteed.
  * The test manifest is read only with --eval-test (final evaluation): then all
    validation images calibrate and the test images verify.

Bounds file (consumed by src/validation/evaluate_metrics.py)
  q_hat for every (gamma, alpha) is computed on ALL validation images and written to
  data/calibration/calib_bounds.json (any previous file is kept as a timestamped backup). The file
  records the checkpoint's SHA-256, step and arch, so evaluate_metrics.py refuses bounds that were
  made for a different model. best*.pt is refused unless --allow-best (selection bias, see above).

Per-image statistics are cached (npz), so re-calibrating with other gamma/alpha needs no new inference:
  python src/validation/calibrate_uncertainty.py --ckpt runs/sspgn/<run>          # uses <run>/last.pt
  python src/validation/calibrate_uncertainty.py --ckpt ... --from-cache <out>/per_image.npz --gammas 0.5,0.9
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for sub in ("src/models", "src/training", "src/utils"):
    sys.path.insert(0, str(ROOT / sub))
from pad_crop_wrapper import PadCropWrapper  # noqa: E402

Q_GRID = np.round(np.linspace(0.0, 1.0, 1001), 3)        # stored quantile levels of r per image
SPARS_FRACS = np.round(np.linspace(0.0, 0.99, 100), 2)    # fraction of pixels removed
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")   # numpy 2 renamed trapz


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def read_manifest(path: Path, allow_test: bool):
    if "test" in path.name.lower() and not allow_test:
        raise RuntimeError(f"{path.name}: test manifest requires --eval-test")
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 5:
            raise ValueError(f"{path.name}:{ln}: expected 5 fields, got {len(parts)}")
        parts[3:5] = [str(Path(p) if Path(p).is_absolute() else Path(__file__).resolve().parents[2] / p) for p in parts[3:5]]
        rows.append(dict(zip(("dataset", "unit", "component", "input", "reference"), parts)))
    return rows


def load_model(ckpt_path: Path, device):
    from train import build_model           # same constructor table as training
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    arch = cfg.get("args", {}).get("arch", "baseline")
    model = build_model(arch, cfg.get("model_cfg"))
    model.load_state_dict(ck["model"], strict=True)
    info = {"arch": arch, "step": ck.get("step"), "best_psnr": ck.get("best_psnr"),
            "compose": model.cfg.compose, "n_params": model.n_params}
    return model.to(device).eval(), info


def to_tensor(p):
    from PIL import Image
    with Image.open(p) as im:
        a = np.array(im.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(a).permute(2, 0, 1).float().div_(255.0)[None]


def pick_device(req: str):
    if req != "auto":
        return torch.device(req)
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free / 2 ** 30 >= 3.0:
            return torch.device("cuda")
        print(f"[device] only {free / 2 ** 30:.1f} GiB GPU memory free (training running?) -> using CPU")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# Per-image statistics
# --------------------------------------------------------------------------
def predictive_std(nu, alpha, beta, kind):
    if kind == "total":
        var = beta * (1 + nu) / (nu * (alpha - 1))
    elif kind == "aleatoric":
        var = beta / (alpha - 1)
    elif kind == "epistemic":
        var = beta / (nu * (alpha - 1))
    else:
        raise ValueError(kind)
    return var.clamp_min(1e-12).sqrt()


def sparsification(err: torch.Tensor, unc: torch.Tensor):
    """Remaining MAE / RMSE after removing the top-f fraction of elements ranked by `unc`
    (model curve) and by `err` itself (oracle). Returns dict of curves on SPARS_FRACS."""
    n = err.numel()
    out = {}
    for name, key in (("model", unc), ("oracle", err)):
        order = torch.argsort(key, descending=True)
        e = err[order].double()
        tail_abs = torch.flip(torch.cumsum(torch.flip(e, [0]), 0), [0])          # sum of e[k:]
        tail_sq = torch.flip(torch.cumsum(torch.flip(e * e, [0]), 0), [0])
        ks = torch.as_tensor((SPARS_FRACS * n).astype(np.int64)).clamp(max=n - 1)
        remain = (n - ks).double()
        out[f"mae_{name}"] = (tail_abs[ks] / remain).numpy()
        out[f"rmse_{name}"] = torch.sqrt(tail_sq[ks] / remain).numpy()
    return out


def ause(curves, metric):
    m, o = curves[f"{metric}_model"], curves[f"{metric}_oracle"]
    m0 = m[0] if m[0] > 0 else 1.0
    return float(_trapz((m - o) / m0, SPARS_FRACS))      # normalised by the full-image error


def spearman(a: torch.Tensor, b: torch.Tensor, max_n=200_000, seed=0):
    n = a.numel()
    if n > max_n:
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:max_n]
        a, b = a[idx], b[idx]
    ra = torch.argsort(torch.argsort(a)).double()
    rb = torch.argsort(torch.argsort(b)).double()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(1e-12))


@torch.no_grad()
def infer_stats(model, rows, device, var_kind, max_images=0):
    wrapped = PadCropWrapper(model, model.cfg.multiple, mode="reflect")
    rows = rows if max_images <= 0 else rows[:max_images]
    recs, t0 = [], time.time()
    for i, r in enumerate(rows):
        x, y = to_tensor(r["input"]).to(device), to_tensor(r["reference"]).to(device)
        if x.shape != y.shape:
            raise ValueError(f"{r['unit']}: input {tuple(x.shape)} != reference {tuple(y.shape)}")
        try:
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                out = wrapped(x)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  [skip OOM] {r['unit']} {tuple(x.shape[-2:])}")
            continue
        nig = out["nig"].float()
        assert nig.shape == (1, 12, *x.shape[-2:]), (r["unit"], tuple(nig.shape))
        mu, nu, alpha, beta = nig[:, 0:3], nig[:, 3:6], nig[:, 6:9], nig[:, 9:12]
        assert bool((nu > 0).all() and (alpha > 1).all() and (beta > 0).all()), f"{r['unit']}: invalid NIG params"
        sig = predictive_std(nu, alpha, beta, var_kind).flatten().cpu()
        err = (mu - y).abs().flatten().cpu()
        rr = err / sig
        qs = torch.quantile(rr.double(), torch.as_tensor(Q_GRID, dtype=torch.float64)).numpy() \
            if rr.numel() <= 16_000_000 else np.quantile(rr.numpy(), Q_GRID)
        sp = sparsification(err, sig)
        mse = float((mu.clamp(0, 1) - y).pow(2).mean())
        recs.append({"unit": r["unit"], "dataset": r["dataset"], "component": r["component"],
                     "H": x.shape[-2], "W": x.shape[-1], "r_quantiles": qs,
                     "sigma_quantiles": np.quantile(sig.numpy(), [0.1, 0.5, 0.9]),
                     "ause_mae": ause(sp, "mae"), "ause_rmse": ause(sp, "rmse"),
                     "spars": sp, "spearman": spearman(sig, err),
                     "psnr": 10 * math.log10(1 / max(mse, 1e-12))})
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print(f"  [{i + 1}/{len(rows)}] {time.time() - t0:.0f}s")
    return recs


# --------------------------------------------------------------------------
# Conformal calibration
# --------------------------------------------------------------------------
def s_at(rec, gamma):
    k = int(round(gamma * 1000))
    if abs(Q_GRID[k] - gamma) > 1e-9:
        raise ValueError(f"gamma={gamma} not on the 0.001 grid")
    return float(rec["r_quantiles"][k])


def pixel_coverage(rec, q):
    """Fraction of elements with r <= q (from the 0.001 quantile grid; error <= 0.001)."""
    return float(np.interp(q, rec["r_quantiles"], Q_GRID, left=0.0, right=1.0))


def conformal_qhat(scores, alpha):
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")                     # too few calibration images for this alpha
    return float(np.sort(scores)[k - 1])


def stratified_subset(rows, n, seed):
    """Identical selection to train.py --val-subset (same seed -> same val_id)."""
    import random
    if n >= len(rows):
        return rows
    by = {}
    for r in rows:
        by.setdefault(r["dataset"], []).append(r)
    rng = random.Random(seed)
    sub = []
    for d in sorted(by):
        k = max(1, round(n * len(by[d]) / len(rows)))
        sub += rng.sample(sorted(by[d], key=lambda r: r["unit"]), min(k, len(by[d])))
    return sorted(sub, key=lambda r: r["unit"])[:n]


def wait_for_final(ckpt: Path, poll: int) -> Path:
    """Poll until ckpt's step equals the run's configured iters. Copies are not made here; the
    checkpoint is written once at the final step by train.py (step == iters triggers a final save)."""
    while True:
        try:
            meta = torch.load(ckpt, map_location="cpu", weights_only=False)
            st, tot = meta.get("step"), meta.get("cfg", {}).get("args", {}).get("iters")
            if tot and st == tot:
                print(f"[wait] {ckpt.name} reached final step {st}", flush=True)
                time.sleep(30)                                # let the writer finish
                return ckpt
            print(f"[wait] {ckpt.name} at step {st} of {tot}; next check in {poll}s", flush=True)
        except (FileNotFoundError, RuntimeError, EOFError) as e:
            print(f"[wait] {ckpt} not readable yet ({type(e).__name__})", flush=True)
        time.sleep(poll)


def file_sha256(path: Path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def split_by_component(recs, frac_cal, seed):
    def h(c):
        return int(hashlib.sha1(f"{seed}:{c}".encode()).hexdigest(), 16) / 16 ** 40
    cal = [r for r in recs if h(r["component"]) < frac_cal]
    ver = [r for r in recs if h(r["component"]) >= frac_cal]
    assert not ({r["component"] for r in cal} & {r["component"] for r in ver})
    return cal, ver


def coverage_table(cal, ver, gammas, alphas):
    rows = []
    for g in gammas:
        sc = np.array([s_at(r, g) for r in cal])
        for a in alphas:
            q = conformal_qhat(sc, a)
            ev = np.array([s_at(r, g) <= q for r in ver])
            pc = np.array([pixel_coverage(r, q) for r in ver]) if math.isfinite(q) else np.ones(len(ver))
            n = len(cal)
            rows.append({"gamma": g, "alpha": a, "n_cal": n, "n_ver": len(ver), "q_hat": q,
                         "target_event_rate": 1 - a,
                         "upper_bound_event_rate": min(1.0, 1 - a + 1 / (n + 1)),
                         "event_rate": float(ev.mean()) if len(ev) else float("nan"),
                         "event_rate_se": float(ev.std(ddof=1) / math.sqrt(len(ev))) if len(ev) > 1 else float("nan"),
                         "mean_pixel_coverage": float(pc.mean()) if len(pc) else float("nan"),
                         "min_pixel_coverage": float(pc.min()) if len(pc) else float("nan"),
                         "by_dataset": {d: float(np.mean([s_at(r, g) <= q for r in ver if r["dataset"] == d]))
                                        for d in sorted({r["dataset"] for r in ver})}})
    return rows


def print_matrix(rows, gammas, alphas, key, title, fmt="{:.3f}"):
    print(f"\n{title}")
    print("  gamma \\ alpha " + "".join(f"{a:>10}" for a in alphas))
    for g in gammas:
        vals = [next(r[key] for r in rows if r["gamma"] == g and r["alpha"] == a) for a in alphas]
        print(f"  {g:<14}" + "".join(f"{fmt.format(v):>10}" for v in vals))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="checkpoint file or run directory (-> last.pt)")
    ap.add_argument("--allow-best", action="store_true", help="permit best*.pt (selected on val; biased)")
    ap.add_argument("--wait-final", action="store_true",
                    help="block until <run>/last.pt reaches the run's final step (cfg iters), then calibrate")
    ap.add_argument("--poll", type=int, default=300, help="seconds between checks with --wait-final")
    ap.add_argument("--allow-nonfinal", action="store_true",
                    help="permit a checkpoint before the final step (diagnostics only; bounds are tied to that exact file)")
    ap.add_argument("--val-subset", type=int, default=0,
                    help="calibrate on the same fixed stratified subset train.py used (e.g. 100, seed --seed). "
                         "0 (default) = all val images: more calibration points, still never used to pick last.pt")
    ap.add_argument("--bounds", default=str(ROOT / "data/calibration/calib_bounds.json"))
    ap.add_argument("--split-dir", default=str(ROOT / "data/splits/lsui_uieb_all_d6_seed0"))
    ap.add_argument("--var", choices=["total", "aleatoric", "epistemic"], default="total")
    ap.add_argument("--gammas", default="0.5,0.8,0.9,0.95")
    ap.add_argument("--alphas", default="0.2,0.1,0.05")
    ap.add_argument("--cal-frac", type=float, default=0.5, help="fraction of val components used to calibrate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-images", type=int, default=0, help="debug: limit images per manifest")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-test", action="store_true", help="FINAL ONLY: calibrate on all val, verify on test")
    ap.add_argument("--from-cache", default=None, help="per_image.npz from a previous run (skips inference)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    gammas = [float(g) for g in args.gammas.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]

    ckpt = Path(args.ckpt)
    if ckpt.is_dir():
        ckpt = ckpt / "last.pt"
    if args.wait_final:
        ckpt = wait_for_final(ckpt, args.poll)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    if not args.from_cache:
        meta = torch.load(ckpt, map_location="cpu", weights_only=False)
        st, tot = meta.get("step"), meta.get("cfg", {}).get("args", {}).get("iters")
        if tot and st is not None and st < tot and not args.allow_nonfinal:
            raise RuntimeError(f"{ckpt} is at step {st} of {tot}. Calibrate the FINAL model (use --wait-final), "
                               "or pass --allow-nonfinal for diagnostics.")
        del meta
    if ckpt.name.startswith("best") and not args.allow_best:
        raise RuntimeError(f"{ckpt.name} was selected on the validation split; calibrating on the same split "
                           "biases q_hat. Use last.pt (or pass --allow-best for diagnostics only).")
    out = Path(args.out) if args.out else ckpt.parent / f"calibration_{args.var}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    warnings = []
    if ckpt.name.startswith("best"):
        warnings.append("Checkpoint selected on the validation split (best.pt): calibration on val is not "
                        "independent of model selection; guarantee is approximate. Use last.pt for the paper.")

    split = Path(args.split_dir)
    if args.from_cache:
        z = np.load(args.from_cache, allow_pickle=True)
        val_recs, test_recs, info = list(z["val"]), list(z["test"]), z["info"].item()
        print(f"[cache] {len(val_recs)} val, {len(test_recs)} test records from {args.from_cache}")
    else:
        device = pick_device(args.device)
        model, info = load_model(ckpt, device)
        print(f"[model] {info} on {device}")
        val_rows = read_manifest(split / "val_manifest.txt", allow_test=False)
        if args.val_subset > 0:
            val_rows = stratified_subset(val_rows, args.val_subset, args.seed)
        vid = hashlib.sha1(chr(10).join(sorted(r["unit"] for r in val_rows)).encode()).hexdigest()[:16]
        print(f"[infer] val: {len(val_rows)} images (val_id {vid})")
        val_recs = infer_stats(model, val_rows, device, args.var, args.max_images)
        test_recs = []
        if args.eval_test:
            test_rows = read_manifest(split / "test_manifest.txt", allow_test=True)
            print(f"[infer] TEST: {len(test_rows)} images (final evaluation)")
            test_recs = infer_stats(model, test_rows, device, args.var, args.max_images)
        np.savez_compressed(out / "per_image.npz", val=np.array(val_recs, dtype=object),
                            test=np.array(test_recs, dtype=object), info=np.array(info, dtype=object))

    if args.eval_test:
        if not test_recs:
            raise RuntimeError("--eval-test but no test records (cache built without --eval-test?)")
        cal, ver, mode = val_recs, test_recs, "calibrate=val (all), verify=test"
    else:
        cal, ver = split_by_component(val_recs, args.cal_frac, args.seed)
        mode = f"calibrate/verify = val split by component ({args.cal_frac:.0%}/{1 - args.cal_frac:.0%}, seed {args.seed})"
    if len(cal) < 20 or len(ver) < 20:
        warnings.append(f"small sets: n_cal={len(cal)}, n_ver={len(ver)}; coverage estimates are noisy")

    # ---- bounds on ALL validation images (what the test evaluation will use) ----
    if not args.eval_test:
        ck_sha = file_sha256(ckpt)
        ck_meta = torch.load(ckpt, map_location="cpu", weights_only=False)
        step, total = ck_meta.get("step"), ck_meta.get("cfg", {}).get("args", {}).get("iters")
        if total and step is not None and step < total:
            warnings.append(f"{ckpt.name} is from step {step} of {total}: not the final model. "
                            "Recalibrate after training ends; bounds are tied to this exact checkpoint.")
        bounds = {"created": datetime.now().isoformat(timespec="seconds"), "ckpt": str(ckpt),
                  "ckpt_sha256": ck_sha, "ckpt_step": step, "ckpt_total_steps": total, "model": info,
                  "var": args.var, "score": "s_i = Q_gamma(|y - mu| / sigma_hat) over all pixels and channels",
                  "n_cal": len(val_recs), "calibration_manifest": str(split / "val_manifest.txt"),
                  "calibration_manifest_sha256": file_sha256(split / "val_manifest.txt"),
                  "q_hat": {f"{g}": {f"{a}": conformal_qhat(np.array([s_at(r, g) for r in val_recs]), a)
                                     for a in alphas} for g in gammas},
                  "guarantee": "P[fraction of covered test pixels >= gamma] >= 1 - alpha, if test images are "
                               "exchangeable with the calibration images and the model is fixed",
                  "warnings": warnings}
        bpath = Path(args.bounds)
        bpath.parent.mkdir(parents=True, exist_ok=True)
        if bpath.exists():
            bpath.rename(bpath.with_name(f"{bpath.stem}_{datetime.now():%Y%m%d_%H%M%S}.bak.json"))
        bpath.write_text(json.dumps(bounds, indent=2, default=float), encoding="utf-8")
        (out / "calib_bounds.json").write_text(json.dumps(bounds, indent=2, default=float), encoding="utf-8")
        print(f"\nq_hat on ALL validation images (n={len(val_recs)}), written to {bpath}:")
        for g in gammas:
            print(f"  gamma={g:<5} " + "  ".join(f"alpha={a}: {bounds['q_hat'][str(g)][str(a)]:.3f}" for a in alphas))

    rows = coverage_table(cal, ver, gammas, alphas)
    print(f"\n[mode] {mode}   var={args.var}   n_cal={len(cal)} n_ver={len(ver)}")
    print_matrix(rows, gammas, alphas, "event_rate",
                 "IMAGE-LEVEL EVENT RATE on verification  P[>= gamma of pixels covered]  (target >= 1-alpha)")
    print_matrix(rows, gammas, alphas, "target_event_rate", "TARGET 1 - alpha")
    print_matrix(rows, gammas, alphas, "mean_pixel_coverage", "MEAN PIXEL COVERAGE on verification")
    print_matrix(rows, gammas, alphas, "q_hat", "q_hat (interval half-width = q_hat * sigma_hat)", "{:.2f}")

    allr = cal + ver
    spars = {k: np.mean([r["spars"][k] for r in ver], axis=0) for k in ver[0]["spars"]}
    summary = {
        "ckpt": str(ckpt), "model": info, "mode": mode, "var": args.var,
        "n_cal": len(cal), "n_ver": len(ver), "coverage": rows,
        "ause_mae_ver": float(np.mean([r["ause_mae"] for r in ver])),
        "ause_rmse_ver": float(np.mean([r["ause_rmse"] for r in ver])),
        "spearman_sigma_err_ver": float(np.mean([r["spearman"] for r in ver])),
        "psnr_ver": float(np.mean([r["psnr"] for r in ver])),
        "per_dataset": {d: {"n": sum(r["dataset"] == d for r in allr),
                            "ause_rmse": float(np.mean([r["ause_rmse"] for r in ver if r["dataset"] == d] or [np.nan])),
                            "spearman": float(np.mean([r["spearman"] for r in ver if r["dataset"] == d] or [np.nan]))}
                        for d in sorted({r["dataset"] for r in allr})},
        "sparsification_curves_ver": {"fractions": SPARS_FRACS.tolist(), **{k: v.tolist() for k, v in spars.items()}},
        "warnings": warnings,
    }
    print(f"\nAUSE (verification, normalised): MAE={summary['ause_mae_ver']:.4f}  RMSE={summary['ause_rmse_ver']:.4f}"
          f"   Spearman(sigma,|err|)={summary['spearman_sigma_err_ver']:.3f}   PSNR={summary['psnr_ver']:.2f}")
    print("SPARSIFICATION (remaining RMSE after removing top-f most uncertain | oracle):")
    for f in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9):
        k = int(round(f * 100))
        print(f"  f={f:<5} model={spars['rmse_model'][k]:.4f}  oracle={spars['rmse_oracle'][k]:.4f}")
    for w in warnings:
        print(f"WARNING: {w}")
    (out / "calibration_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"[out] {out}")


if __name__ == "__main__":
    main()

"""
Phase E zero-shot evaluation (PI decisions 2026-11-13 / 2026-11-17). EXPLORATORY; analysis fixed by this commit, before
any Phase E output exists. Neither set was seen in training by any evaluated model (overlap audit 2026-11-13, bound below).

Sets
  dark : EUVP Paired/underwater_dark, trainA (input) / trainB (reference), 5550 pairs -> PSNR8, SSIM8, LPIPS(alex)
  c60  : UIEB challenge_60, 60 inputs, no reference          -> UIQM, UCIQE (src/validation/uw_metrics, as Phase B)
  (U45 not on disk on 2026-11-17 -> not evaluated.)
Models
  P1b_s0 (locked flagship after Gate G2), and the three retrained baselines fiveaplus / ushape / wweuie (last.pt, step
  100000, SHA-256 pinned). Spectroformer's released file is the 'model_without_CA' ablation (not the published model);
  it is NOT included pending a PI decision.
P1b depth on new images is computed exactly as src/data/precompute_depth.py did for training: Depth-Anything-V2-Small
@5426e4f (weights SHA-256 pinned), fp16 CUDA with fp32 fallback, P1/P99 normalisation, stored precision float16.
P1b diagnostics (per image): mean predictive std (aleatoric / epistemic, NIG) and ordered-DGT beta_D (R, G, B); the
same diagnostics on the 517 validation images (precomputed depth) as the in-domain reference.
Statistics: paired bootstrap of P1b - baseline per image (B = 2000, seed 20261107), percentile 95% CI, d_z.
Parity: per model, 5 dark images CUDA vs CPU, |dPSNR8| < 0.01 dB, with the shared +/-8 checkerboard failure control.
Precision fp32; CUDA allocator capped (--vram-frac, default 0.50); OOM images fall back to CPU (counted).
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata as md
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "src/data"), str(ROOT / "src/validation"), str(ROOT / "baselines/adapters")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import common  # noqa: E402
from common import (git_state, new_outdir, write_json, environment, Timer, val_rows, load_u8, u8_to_tensor,  # noqa: E402
                    to_u8_tensor, psnr_u8, ssim_u8, predictive_std, sha256_file, B_BOOT, BOOT_SEED)
from s1_eval517 import parity_delta, parity_fails, control_trips, PARITY_TOL_DB  # noqa: E402
from p1_eval517 import DepthRunner, DEPTH  # noqa: E402
from train import DepthStore  # noqa: E402
import precompute_depth as PD  # noqa: E402
from eval517_baseline import BaselineRunner  # noqa: E402
from uw_metrics import uiqm, uciqe  # noqa: E402

E = ROOT / "data/raw/EUVP/Paired/underwater_dark"
C60 = ROOT / "data/raw/UIEB/challenge_60"
AUDIT = ROOT / "reports/integrity_audit/overlap_audit_20261113.json"
P1B = ("P1b_s0", ROOT / "runs/phase1/P1b_v2_dgt_ordered/20261110_093229/last.pt",
       "5b22ea7c217b09b96ad2b6309b21b5cbca906444e2f442dd9732965ec9db52c0")
BASELINES = {"fiveaplus": ("runs/baselines/fiveaplus/20261113_170747/last.pt",
                           "61cf5174bbdee938d35fecbf9a5be72457f8a7c55784643689f961f558cb96a1"),
             "ushape": ("runs/baselines/ushape/20261116_094852/last.pt",
                        "8057943ef123268fda1cdeef3858461561daf2eeeb7851f1b5af1cca4a771ac7"),
             "wweuie": ("runs/baselines/wweuie/20261116_222308/last.pt",
                        "c0e9261627943589e9bad0353ac0e7cbf0af664520788b21d188c66ed96856c4")}
LPIPS_PIN = {"package": "lpips", "version": "0.1.4", "net": "alex",
             "lin_weights_sha256": "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0",
             "alexnet_backbone": "alexnet-owt-7be5be79.pth",
             "alexnet_sha256": "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"}
IMG = (".jpg", ".jpeg", ".png", ".bmp")


def check_audit():
    a = json.loads(AUDIT.read_text(encoding="utf-8"))["sets"]
    for k in ("EUVP_underwater_dark", "UIEB_challenge_60"):
        if a[k]["verified_duplicates_pix_ge35dB"] != 0 or a[k]["verified_near_pix_ge25dB"] != 0:
            raise SystemExit(f"[abort] overlap audit: {k} not clean {a[k]}")
    return {k: a[k] for k in ("EUVP_underwater_dark", "UIEB_challenge_60")}


def pin_lpips(dev):
    import lpips
    import os
    got = {"version": md.version("lpips"),
           "lin_weights_sha256": sha256_file(Path(os.path.dirname(lpips.__file__)) / "weights/v0.1/alex.pth"),
           "alexnet_sha256": sha256_file(Path(torch.hub.get_dir()) / "checkpoints" / LPIPS_PIN["alexnet_backbone"])}
    for k, v in got.items():
        if v != LPIPS_PIN[k]:
            raise SystemExit(f"[abort] LPIPS pin mismatch {k}: {v} != {LPIPS_PIN[k]}")
    return lpips.LPIPS(net="alex", verbose=False).to(dev).eval()


def dark_pairs():
    A, B = E / "trainA", E / "trainB"
    ins = sorted(p for p in A.iterdir() if p.suffix.lower() in IMG)
    pairs = [(p, B / p.name) for p in ins]
    missing = [p.name for p, q in pairs if not q.exists()]
    if missing or len(pairs) != 5550:
        raise SystemExit(f"[abort] dark pairs {len(pairs)} missing refs {missing[:5]}")
    return pairs


class P1bRunner:
    """P1b with depth computed on the fly exactly as precompute_depth (fp16 CUDA, fp32 fallback, float16 storage)."""

    def __init__(self, dev):
        common.CKPTS[P1B[0]] = (P1B[1], P1B[2], 100000)
        self.run = DepthRunner(P1B[0])
        self.proc, self.dm, self.dm16, self.dmeta = PD.load_depth_model(dev, fp16=True)
        self.dev = dev

    def depth(self, x8):
        d, prec = PD.disparity(self.proc, self.dm, self.dm16, x8, self.dev)
        z, _, degen = PD.normalise(d)
        return torch.from_numpy(z.astype(np.float16).astype(np.float32))[None], prec, degen

    def __call__(self, x8, z=None, force_device=None):
        if z is None:
            z, _, _ = self.depth(x8)
        o, dv = self.run(u8_to_tensor(x8), z[None], force_device=force_device)
        return o, dv


def p1b_diag(o):
    al = predictive_std(o["nu"], o["alpha"], o["beta"], "aleatoric")
    ep = predictive_std(o["nu"], o["alpha"], o["beta"], "epistemic")
    r = {"std_aleatoric": float(al.mean()), "std_epistemic": float(ep.mean())}
    if "beta_D" in o and torch.is_tensor(o["beta_D"]):
        b = o["beta_D"].reshape(-1)[:3].tolist()
        r.update(beta_D_R=b[0], beta_D_G=b[1], beta_D_B=b[2])
    return r


def parity(name, call, xs_ys):
    recs, ctl = [], False
    for x8, y8 in xs_ys:
        oc, dc = call(x8, None)
        if dc != "cuda":
            continue
        op, _ = call(x8, "cpu")
        pc, pp = to_u8_tensor(oc), to_u8_tensor(op)
        recs.append(parity_delta(pc, pp, y8))
        if not ctl:
            if not control_trips(pc, pp, y8):
                raise SystemExit(f"[abort] {name}: parity failure control did not fail")
            ctl = True
    if len(recs) < 3 or parity_fails(recs):
        raise SystemExit(f"[HALT] {name}: parity failed or too few CUDA images {recs}")
    return {"n": len(recs), "max_abs_d_psnr8": max(recs), "control_tripped": ctl}


def boot(d, B=B_BOOT, seed=BOOT_SEED):
    d = np.asarray(d, dtype=np.float64); rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(B, len(d)))].mean(1)
    sd = d.std(ddof=1)
    return {"est": float(d.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5)),
            "d_z": float(d.mean() / sd) if sd > 0 else float("nan"), "n": int(len(d)),
            "ci_excludes_0": bool(np.percentile(m, 2.5) > 0 or np.percentile(m, 97.5) < 0)}


def save(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vram-frac", type=float, default=0.50)
    ap.add_argument("--limit", type=int, default=0, help="dry run on the first N dark images (0 = all)")
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(a.vram_frac, 0)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    tm, commit = Timer(), git_state()
    audit = check_audit()
    out = new_outdir("PHASE_E" + ("_DRYRUN" if a.limit else ""))
    print(f"[phase_e] commit {commit} out {out} vram_frac {a.vram_frac}", flush=True)
    lp = pin_lpips(dev)
    pairs = dark_pairs()[: a.limit or None]
    c60 = sorted(p for p in C60.iterdir() if p.suffix.lower() in IMG)
    if len(c60) != 60:
        raise SystemExit(f"[abort] C60 has {len(c60)} images")
    if a.limit:
        c60 = c60[:4]

    p1b = P1bRunner(dev)
    runners = {"P1b_s0": lambda x8, fd: (lambda o: (o[0]["J"], o[1]))(p1b(x8, force_device=fd))}
    for n, (ck, sh) in BASELINES.items():
        br = BaselineRunner(n, str(ROOT / ck), sh)
        runners[n] = (lambda br: lambda x8, fd: br(u8_to_tensor(x8), force_device=fd))(br)

    par_imgs = [(load_u8(p), load_u8(q)) for p, q in pairs[:6]]
    parity_res = {n: parity(n, f, par_imgs) for n, f in runners.items()}
    print("[parity]", parity_res, flush=True)

    with torch.no_grad():
        def lp_score(p8, y8):
            t = lambda u: (torch.from_numpy(u).permute(2, 0, 1)[None].float().div(127.5) - 1.0).to(dev)  # noqa: E731
            return float(lp(t(p8), t(y8)))

        dark = {n: [] for n in runners}; c6 = {n: [] for n in runners}; depth_prec = {}
        for i, (p, q) in enumerate(pairs):
            x8, y8 = load_u8(p), load_u8(q)
            z, prec, degen = p1b.depth(x8); depth_prec[prec] = depth_prec.get(prec, 0) + 1
            for n, f in runners.items():
                if n == "P1b_s0":
                    o, dv = p1b(x8, z=z); J = o["J"]; extra = p1b_diag(o)
                else:
                    J, dv = f(x8, None); extra = {}
                p8 = to_u8_tensor(J)
                dark[n].append({"image": p.name, "device": dv, "psnr8": psnr_u8(p8, y8), "ssim8": ssim_u8(p8, y8),
                                "lpips": lp_score(p8, y8), **({"depth_degenerate": int(degen)} if n == "P1b_s0" else {}),
                                **extra})
            if (i + 1) % 250 == 0:
                print(f"[dark] {i + 1}/{len(pairs)}  P1b mean PSNR8 {np.mean([r['psnr8'] for r in dark['P1b_s0']]):.3f}", flush=True)
        for p in c60:
            x8 = load_u8(p)
            c6.setdefault("input", []).append({"image": p.name, **{k: v for k, v in (("uiqm", uiqm(x8)["uiqm"]), ("uciqe", uciqe(x8)["uciqe"]))}})
            z, prec, _ = p1b.depth(x8)
            for n, f in runners.items():
                if n == "P1b_s0":
                    o, dv = p1b(x8, z=z); J = o["J"]; extra = p1b_diag(o)
                else:
                    J, dv = f(x8, None); extra = {}
                p8 = to_u8_tensor(J)
                c6[n].append({"image": p.name, "device": dv, "uiqm": uiqm(p8)["uiqm"], "uciqe": uciqe(p8)["uciqe"], **extra})
        print("[c60] done", flush=True)

        vrows = val_rows()[: (8 if a.limit else None)]; rows4 = [(r["dataset"], r["unit"], r["input"], r["reference"]) for r in vrows]
        store = DepthStore(DEPTH, rows4, check_sha=True); vdiag = []
        for r in vrows:
            x8 = load_u8(r["input"]); z = store.load(r["unit"], x8.shape[:2])
            o, dv = p1b(x8, z=z); vdiag.append({"unit": r["unit"], "dataset": r["dataset"], "device": dv, **p1b_diag(o)})
        print("[val517 in-domain uncertainty reference] done", flush=True)

    for n in runners:
        save(out / f"dark_{n}.csv", dark[n]); save(out / f"c60_{n}.csv", c6[n])
    save(out / "c60_input.csv", c6["input"]); save(out / "val517_P1b_diag.csv", vdiag)

    absd = {n: {m: float(np.mean([r[m] for r in dark[n]])) for m in ("psnr8", "ssim8", "lpips")} for n in runners}
    absc = {n: {m: float(np.mean([r[m] for r in c6[n]])) for m in ("uiqm", "uciqe")} for n in list(runners) + ["input"]}
    paired = {}
    for n in BASELINES:
        paired[n] = {"dark": {m: boot([a_[m] - b_[m] for a_, b_ in zip(dark["P1b_s0"], dark[n])]) for m in ("psnr8", "ssim8", "lpips")},
                     "c60": {m: boot([a_[m] - b_[m] for a_, b_ in zip(c6["P1b_s0"], c6[n])]) for m in ("uiqm", "uciqe")}}
    unc = {s: {k: float(np.mean([r[k] for r in rows])) for k in ("std_aleatoric", "std_epistemic")}
           for s, rows in (("val517", vdiag), ("dark", dark["P1b_s0"]), ("c60", c6["P1b_s0"]))}
    for s in ("dark", "c60"):
        unc[s]["epistemic_vs_val517"] = boot([r["std_epistemic"] for r in (dark["P1b_s0"] if s == "dark" else c6["P1b_s0"])]
                                             )["est"] / unc["val517"]["std_epistemic"]
    summary = {"commit": commit, "exploratory": True, "overlap_audit": audit, "lpips_pin": LPIPS_PIN,
               "depth_model": PD.MODEL_ID, "depth_meta": p1b.dmeta, "depth_precision_counts": depth_prec,
               "checkpoints": {"P1b_s0": P1B[2], **{n: s for n, (_, s) in BASELINES.items()}},
               "n": {"dark": len(pairs), "c60": len(c60), "val517": len(vdiag)}, "parity": parity_res,
               "absolute_dark": absd, "absolute_c60": absc, "paired_P1b_minus": paired, "p1b_uncertainty": unc,
               "spectroformer": "not evaluated: released checkpoint is model_without_CA ablation (epoch 37); PI decision pending",
               "u45": "not on disk (2026-11-17)", "env": environment(), "runtime": tm.done()}
    write_json(out / "summary.json", summary)
    print("[phase_e] done", json.dumps({"absolute_dark": absd, "absolute_c60": absc}, indent=1), flush=True)


if __name__ == "__main__":
    main()

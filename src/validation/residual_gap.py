"""
Residual generalization-gap diagnostic (CPU only; read-only; changes no model, calibration or gate).

    Delta_residual = E_val[ |Y - J_hat| ] - E_train[ |Y - J_hat| ]

Per image, MAE = mean over all pixels and channels of |Y - J_hat| at native resolution (PadCropWrapper,
no resizing; J_hat unclamped, float32). Reported for train and val, overall and for LSUI and UIEB
separately, with percentile bootstrap 95% CIs (image-level resampling, done independently within
train and val; B = 2000). Image counts are reported for every cell.

Interpretation (also written into the output): a positive Delta means the restorer's residuals are
larger on images it did not train on. This is a residual generalization gap, NOT proof of
overfitting: train and val differ in scene content and source composition (the per-dataset split
reduces but does not remove this), and image-level bootstrap ignores the residual dependence
inside duplicate components (most components are singletons).

Data roles: val references are used only to MEASURE residuals here. Nothing is fitted or selected.
The test split is never read.

  python src/validation/residual_gap.py --ckpt runs/sspgn/20261106_173805/last.pt              # final model
  python src/validation/residual_gap.py --ckpt ... --n-train 517 --seed 0                       # train sample size
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")                # CPU only

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/validation"))
sys.path.insert(0, str(ROOT / "src/utils"))
from calibrate_uncertainty import read_manifest, load_model, to_tensor, file_sha256  # noqa: E402
from pad_crop_wrapper import PadCropWrapper  # noqa: E402


def stratified_sample(rows, n, seed):
    if n <= 0 or n >= len(rows):
        return list(rows)
    by = {}
    for r in rows:
        by.setdefault(r["dataset"], []).append(r)
    rng = random.Random(seed)
    out = []
    for d in sorted(by):
        k = max(1, round(n * len(by[d]) / len(rows)))
        out += rng.sample(sorted(by[d], key=lambda r: r["unit"]), min(k, len(by[d])))
    return sorted(out, key=lambda r: r["unit"])


def bootstrap_mean_ci(x, B, rng):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    idx = rng.integers(0, x.size, size=(B, x.size))
    m = x[idx].mean(1)
    return float(x.mean()), (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def bootstrap_diff_ci(a, b, B, rng):
    """CI for mean(a) - mean(b), resampling a and b independently."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    da = a[rng.integers(0, a.size, size=(B, a.size))].mean(1)
    db = b[rng.integers(0, b.size, size=(B, b.size))].mean(1)
    d = da - db
    return float(a.mean() - b.mean()), (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


@torch.no_grad()
def per_image_mae(model, rows, tag):
    wrapped = PadCropWrapper(model, model.cfg.multiple, mode="reflect")
    out, t0 = [], time.time()
    for i, r in enumerate(rows):
        x, y = to_tensor(r["input"]), to_tensor(r["reference"])
        if x.shape != y.shape:
            raise ValueError(f"{r['unit']}: input/reference size mismatch")
        J = wrapped(x)["J"].float()
        assert J.shape == y.shape
        out.append({"split": tag, "dataset": r["dataset"], "unit": r["unit"], "component": r["component"],
                    "H": y.shape[-2], "W": y.shape[-1], "mae": float((y - J).abs().mean())})
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            print(f"  [{tag}] {i + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--allow-nonfinal", action="store_true", help="diagnostics on a non-final checkpoint (tests)")
    ap.add_argument("--split-dir", default=str(ROOT / "data/splits/lsui_uieb_all_d6_seed0"))
    ap.add_argument("--n-train", type=int, default=517, help="stratified train sample (0 = all 4135; slow on CPU)")
    ap.add_argument("--n-val", type=int, default=0, help="0 = all 517")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)

    ckpt = Path(args.ckpt)
    meta = torch.load(ckpt, map_location="cpu", weights_only=False)
    step, total = meta.get("step"), meta.get("cfg", {}).get("args", {}).get("iters")
    if total and step != total and not args.allow_nonfinal:
        raise RuntimeError(f"{ckpt} is at step {step} of {total}; run on the final model or pass --allow-nonfinal")
    del meta
    model, info = load_model(ckpt, torch.device("cpu"))
    split = Path(args.split_dir)
    tr = stratified_sample(read_manifest(split / "train_manifest.txt", allow_test=False), args.n_train, args.seed)
    va = stratified_sample(read_manifest(split / "val_manifest.txt", allow_test=False), args.n_val, args.seed)
    out = Path(args.out) if args.out else ROOT / "reports/diagnostics" / f"residual_gap_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[gap] ckpt step {step}/{total} sha {file_sha256(ckpt)[:12]} | train {len(tr)} | val {len(va)} -> {out}", flush=True)

    recs = per_image_mae(model, tr, "train") + per_image_mae(model, va, "val")
    with open(out / "per_image_mae.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(recs)

    rng = np.random.default_rng(args.seed)
    table = {}
    for group in ["ALL", "LSUI", "UIEB"]:
        sel = lambda s: [r["mae"] for r in recs if r["split"] == s and (group == "ALL" or r["dataset"] == group)]
        a_tr, a_va = sel("train"), sel("val")
        m_tr, ci_tr = bootstrap_mean_ci(a_tr, args.B, rng)
        m_va, ci_va = bootstrap_mean_ci(a_va, args.B, rng)
        d, ci_d = bootstrap_diff_ci(a_va, a_tr, args.B, rng)
        table[group] = {"n_train": len(a_tr), "n_val": len(a_va), "mae_train": m_tr, "mae_train_ci95": ci_tr,
                        "mae_val": m_va, "mae_val_ci95": ci_va, "delta_val_minus_train": d, "delta_ci95": ci_d,
                        "relative_gap": d / m_tr if m_tr else float("nan")}
    summary = {"ckpt": str(ckpt), "ckpt_sha256": file_sha256(ckpt), "step": step, "total": total,
               "seed": args.seed, "B": args.B, "table": table,
               "definition": "Delta = E_val|Y-J_hat| - E_train|Y-J_hat|, per-image MAE over pixels x channels, native resolution",
               "interpretation": "residual generalization gap; not proof of overfitting (scene/source composition may "
                                 "confound; image-level bootstrap ignores within-component dependence)"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nRESIDUAL GENERALIZATION GAP  (per-image MAE of |Y - J_hat|, 95% bootstrap CI)")
    print(f"  {'group':6s} {'n_tr':>5s} {'MAE train':>24s} {'n_val':>6s} {'MAE val':>24s} {'Delta (val-train)':>28s} {'rel':>7s}")
    for g, t in table.items():
        f = lambda m, ci: f"{m:.4f} [{ci[0]:.4f},{ci[1]:.4f}]"
        print(f"  {g:6s} {t['n_train']:>5d} {f(t['mae_train'], t['mae_train_ci95']):>24s} {t['n_val']:>6d} "
              f"{f(t['mae_val'], t['mae_val_ci95']):>24s} {f(t['delta_val_minus_train'], t['delta_ci95']):>28s} "
              f"{t['relative_gap']:>+7.1%}")
    print(f"[out] {out}")
    return summary


if __name__ == "__main__":
    main()

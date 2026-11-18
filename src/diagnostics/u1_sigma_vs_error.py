"""
U1: per-image sigma-hat vs per-image error and vs gauge fraction G (CPU only; no inference; reads existing files).

Inputs (no model is run):
  runs/sspgn/20261106_173805/calibration_total_20261107_030031/per_image.npz   (calibration cache, last.pt, CPU fp32)
    sigma_q50 = median over all pixels x channels of sigma_hat = sqrt(total NIG variance)   [0,1] intensity units
                (calibrate_uncertainty.py:189,198; var=total per calib_bounds.json)
    rmse      = 10 ** (-psnr / 20), psnr = 10 log10(1 / MSE(clamp(mu,0,1), y)) in float      [0,1] units  (:195,201)
  runs/diagnostics/T5_G_R517_20261107_040554/per_image_last.csv
    g         = 1 - MSE(clip(A* P), Y) / MSE(P, Y), per-channel affine oracle (t5_g_r517.py:59)  dimensionless
Statistics: Spearman rho (scipy), n, two-sided p, percentile bootstrap 95% CI (B=2000, seed 20261107), per dataset;
pooled values are secondary. Image-level association only: says nothing about spatial localization of sigma_hat.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from common import ROOT, B_BOOT, BOOT_SEED, new_outdir, write_json, environment, Timer  # noqa: E402

NPZ = ROOT / "runs/sspgn/20261106_173805/calibration_total_20261107_030031/per_image.npz"
CSV = ROOT / "runs/diagnostics/T5_G_R517_20261107_040554/per_image_last.csv"


def rho_ci(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = spearmanr(x, y)
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(x), (B_BOOT, len(x)))
    bs = np.array([spearmanr(x[i], y[i]).statistic for i in idx])
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(len(x)),
            "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]}


def main():
    tm = Timer()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    V = list(np.load(NPZ, allow_pickle=True)["val"])
    G = {r["unit"]: r for r in csv.DictReader(open(CSV, encoding="utf-8"))}
    units = [r["unit"] for r in V]
    assert len(units) == len(set(units)) == len(G) == 517 and set(units) == set(G), "ID mismatch"
    rows = []
    for r in V:
        c = G[r["unit"]]
        assert c["dataset"] == r["dataset"] and int(c["H"]) == r["H"] and int(c["W"]) == r["W"]
        q10, q50, q90 = (float(v) for v in r["sigma_quantiles"])
        rows.append({"unit": r["unit"], "dataset": r["dataset"], "sigma_q10": q10, "sigma_q50": q50, "sigma_q90": q90,
                     "rmse": 10 ** (-float(r["psnr"]) / 20), "g": float(c["g"])})
    out = new_outdir("U1")
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    res = {}
    for grp in ("LSUI", "UIEB", "ALL (secondary)"):
        R = rows if grp.startswith("ALL") else [r for r in rows if r["dataset"] == grp]
        col = lambda k: [r[k] for r in R]
        res[grp] = {"sigma_q50_vs_rmse": rho_ci(col("sigma_q50"), col("rmse")),
                    "sigma_q50_vs_g": rho_ci(col("sigma_q50"), col("g")),
                    "sigma_q90_vs_rmse": rho_ci(col("sigma_q90"), col("rmse")),
                    "rmse_vs_g": rho_ci(col("rmse"), col("g"))}
    write_json(out / "summary.json", {"diagnostic": "U1", "status": "EXPLORATORY (no pre-registered criterion)",
                                      "results": res, "labels": "validation, not held-out; image-level only",
                                      "bootstrap": {"B": B_BOOT, "seed": BOOT_SEED, "type": "percentile 95%"}})
    write_json(out / "meta.json", {"head": head, "working_tree_dirty": bool(dirty), "inputs": [str(NPZ), str(CSV)],
                                   "env": environment(), **tm.done(), "inference": "none"})
    for grp, v in res.items():
        for k, s in v.items():
            print(f"{grp:16s} {k:20s} rho={s['rho']:+.3f} CI[{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}] n={s['n']} p={s['p']:.2e}")
    print("->", out)


if __name__ == "__main__":
    main()

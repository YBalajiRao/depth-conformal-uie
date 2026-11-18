"""
F3 analysis (CPU; report section S). Calibrates on the full validation collection only, evaluates on test and dark.
Output: runs/f3/F3_RESULTS_<ts>.json
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import norm, spearmanr

ROOT = Path(__file__).resolve().parents[2]
F3 = ROOT / "runs/f3"
LEVELS = (0.80, 0.90, 0.95)
B, SEED = 2000, 20261107


def latest_full(split, n_expected):
    for d in sorted(F3.glob(f"{split}_*"), reverse=True):
        m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if m["n"] == n_expected:
            return d, np.load(d / "samples.npz")
    raise SystemExit(f"no full {split} collection with n={n_expected}")


def conformal_q(scores, level):
    s = np.sort(scores.ravel()); n = len(s)
    k = min(int(np.ceil((n + 1) * level)) - 1, n - 1)
    return float(s[k])


def evaluate(r, s, half):
    """r, s: (N_img, 3, K); half: half-width array broadcastable to r. Returns coverage and sharpness stats."""
    cov = (r <= half)
    per_img = cov.reshape(cov.shape[0], -1).mean(1)
    return {"coverage": float(cov.mean()), "per_image_mean": float(per_img.mean()),
            "per_image_p05": float(np.percentile(per_img, 5)), "half_width_x255": float(np.broadcast_to(half, r.shape).mean() * 255)}


def boot(d):
    d = np.asarray(d, dtype=np.float64); rng = np.random.default_rng(SEED)
    m = d[rng.integers(0, len(d), size=(B, len(d)))].mean(1); lo, hi = np.percentile(m, [2.5, 97.5])
    return {"est": float(d.mean()), "lo": float(lo), "hi": float(hi), "ci_excludes_0": bool(lo > 0 or hi < 0), "n": len(d)}


def main():
    sets = {"val": latest_full("val", 517), "test": latest_full("test", 517), "dark": latest_full("dark", 5550)}
    res = {"sources": {k: str(v[0]) for k, v in sets.items()}, "levels": LEVELS, "predictors": {}}
    for pred in ("1", "E"):
        name = {"1": "P1b_s0", "E": "E3_ensemble"}[pred]
        rv, sv = sets["val"][1]["r" + pred], sets["val"][1]["s" + pred]
        P = {"calibration": {}, "eval": {}}
        for lv in LEVELS:
            qa = conformal_q(rv / np.maximum(sv, 1e-8), lv)          # adaptive: |r| / sigma
            qr = conformal_q(rv, lv)                                 # non-adaptive constant width
            z = float(norm.ppf(0.5 + lv / 2))                        # uncalibrated NIG Gaussian
            P["calibration"][str(lv)] = {"q_adaptive": qa, "q_constant_x255": qr * 255, "z_gaussian": z}
            for sn in ("val", "test", "dark"):
                r, s = sets[sn][1]["r" + pred], sets[sn][1]["s" + pred]
                P["eval"].setdefault(sn, {})[str(lv)] = {
                    "adaptive_conformal": evaluate(r, s, qa * s), "nig_gaussian_uncalibrated": evaluate(r, s, z * s),
                    "constant_conformal": evaluate(r, s, np.full((1, 1, 1), qr))}
        for sn in ("val", "test", "dark"):
            r, s = sets[sn][1]["r" + pred], sets[sn][1]["s" + pred]
            sub = np.random.default_rng(0).choice(r.size, size=min(2_000_000, r.size), replace=False)
            P["eval"][sn]["spearman_sigma_vs_abs_residual"] = float(spearmanr(s.ravel()[sub], r.ravel()[sub]).statistic)
        res["predictors"][name] = P
    acc = {}
    for sn in ("val", "dark", "test"):
        rows = list(csv.DictReader(open(Path(res["sources"][sn]) / "per_image.csv", encoding="utf-8")))
        acc[sn + ("_POSTHOC" if sn == "test" else "")] = {
            "P1b_s0_psnr8": float(np.mean([float(r["psnr8_s0"]) for r in rows])),
            "E3_psnr8": float(np.mean([float(r["psnr8_E3"]) for r in rows])),
            "E3_minus_s0_psnr8": boot([float(r["psnr8_E3"]) - float(r["psnr8_s0"]) for r in rows]),
            "E3_minus_s0_ssim8": boot([float(r["ssim8_E3"]) - float(r["ssim8_s0"]) for r in rows])}
    res["ensemble_accuracy"] = acc
    out = F3 / f"F3_RESULTS_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8"); print("wrote", out)
    for name, P in res["predictors"].items():
        print(f"\n== {name}")
        for sn in ("val", "test", "dark"):
            for lv in LEVELS:
                e = P["eval"][sn][str(lv)]
                print(f"  {sn:4s} {lv:.2f}: adaptive cov {e['adaptive_conformal']['coverage']:.4f} hw {e['adaptive_conformal']['half_width_x255']:.2f} | "
                      f"NIG-uncal cov {e['nig_gaussian_uncalibrated']['coverage']:.4f} | constant cov {e['constant_conformal']['coverage']:.4f} hw {e['constant_conformal']['half_width_x255']:.2f} | per-img p05 {e['adaptive_conformal']['per_image_p05']:.3f}")
            print(f"  {sn:4s} spearman(sigma,|r|) = {P['eval'][sn]['spearman_sigma_vs_abs_residual']:.3f}")
    print("\nensemble accuracy:", json.dumps(acc, indent=1))


if __name__ == "__main__":
    sys.exit(main())

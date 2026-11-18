"""
Target-domain conformal recalibration on EUVP underwater_dark (EXPLORATORY; added 2026-11-18 after F3, not part of the
section-S pre-registration). CPU only, from the saved F3 samples (runs/f3/dark_*/samples.npz; 1024 px/image).

Split: the 5,550 dark images are partitioned with numpy.random.default_rng(20261118) into 500 calibration images (~9%)
and 5,050 evaluation images. Conformal quantiles (score |r|/sigma, and constant width |r|) are fitted on the 500-image
slice and evaluated on the disjoint 5,050-image slice, for P1b_s0 and the 3-seed ensemble, at 80/90/95%.
For reference, the validation-calibrated quantiles (section S) are re-evaluated on the same 5,050 images.
Output: runs/f3/F3_TARGET_RECAL_<ts>.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from f3_analyze import latest_full, conformal_q, evaluate, LEVELS  # noqa: E402


def main():
    _, val = latest_full("val", 517); dpath, dark = latest_full("dark", 5550)
    idx = np.random.default_rng(20261118).permutation(5550); cal, ev = np.sort(idx[:500]), np.sort(idx[500:])
    res = {"source_dark": str(dpath), "n_cal": len(cal), "n_eval": len(ev), "split_seed": 20261118, "exploratory": True,
           "predictors": {}}
    for pred, name in (("1", "P1b_s0"), ("E", "E3_ensemble")):
        r, s = dark["r" + pred], dark["s" + pred]; rv, sv = val["r" + pred], val["s" + pred]
        out = {}
        for lv in LEVELS:
            q_t = conformal_q(r[cal] / np.maximum(s[cal], 1e-8), lv); qc_t = conformal_q(r[cal], lv)
            q_v = conformal_q(rv / np.maximum(sv, 1e-8), lv)
            out[str(lv)] = {"target_recalibrated_adaptive": evaluate(r[ev], s[ev], q_t * s[ev]),
                            "target_recalibrated_constant": evaluate(r[ev], s[ev], np.full((1, 1, 1), qc_t)),
                            "validation_calibrated_adaptive": evaluate(r[ev], s[ev], q_v * s[ev]),
                            "q_target": q_t, "q_validation": q_v}
        res["predictors"][name] = out
    o = ROOT / "runs/f3" / f"F3_TARGET_RECAL_{datetime.now():%Y%m%d_%H%M%S}.json"
    o.write_text(json.dumps(res, indent=1), encoding="utf-8"); print("wrote", o)
    for name, P in res["predictors"].items():
        for lv in LEVELS:
            e = P[str(lv)]
            print(f"{name:12s} {lv:.2f}: val-cal cov {e['validation_calibrated_adaptive']['coverage']:.4f} | target-recal cov "
                  f"{e['target_recalibrated_adaptive']['coverage']:.4f} hw {e['target_recalibrated_adaptive']['half_width_x255']:.2f} "
                  f"| target const cov {e['target_recalibrated_constant']['coverage']:.4f} hw {e['target_recalibrated_constant']['half_width_x255']:.2f} "
                  f"| worst5% {e['target_recalibrated_adaptive']['per_image_p05']:.3f}")


if __name__ == "__main__":
    main()

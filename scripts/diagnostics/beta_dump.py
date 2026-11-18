"""
Per-image DGT coefficients of P1b_s0 (beta^D, beta^B, B_inf) on the Phase E sets and the 517 validation images.
EXPLORATORY add-on to Phase E (whose summary recorded beta^D only). Reuses the committed Phase E P1b runner
(src/diagnostics/phase_e_eval.py: same checkpoint SHA, same on-the-fly depth as precompute_depth). Test split untouched.
Output: runs/diagnostics/PHASE_E_BETA_<ts>/beta_<set>.csv and summary.json
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/diagnostics")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import phase_e_eval as PE  # noqa: E402
from common import val_rows, load_u8  # noqa: E402
from train import DepthStore  # noqa: E402

torch.cuda.set_per_process_memory_fraction(0.50, 0)
out = ROOT / "runs/diagnostics" / f"PHASE_E_BETA_{datetime.now():%Y%m%d_%H%M%S}"; out.mkdir(parents=True)
p1b = PE.P1bRunner(torch.device("cuda"))
KEYS = ("beta_D", "beta_B", "B_inf")


def rec(name, o):
    r = {"image": name}
    for k in KEYS:
        for c, v in zip("RGB", o[k].reshape(-1)[:3].tolist()):
            r[f"{k}_{c}"] = v
    return r


sets = {}
with torch.no_grad():
    sets["dark"] = [rec(p.name, p1b(load_u8(p))[0]) for p, _ in PE.dark_pairs()]
    sets["c60"] = [rec(p.name, p1b(load_u8(p))[0]) for p in sorted(x for x in PE.C60.iterdir() if x.suffix.lower() in PE.IMG)]
    vr = val_rows(); store = DepthStore(PE.DEPTH, [(r["dataset"], r["unit"], r["input"], r["reference"]) for r in vr], check_sha=True)
    sets["val517"] = [rec(r["unit"], p1b(x8, z=store.load(r["unit"], x8.shape[:2]))[0]) for r in vr for x8 in [load_u8(r["input"])]]
summ = {"checkpoint_sha256": PE.P1B[2], "phase_e_commit_runner": "src/diagnostics/phase_e_eval.py @ 658cfaf"}
for s, rows in sets.items():
    with open(out / f"beta_{s}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    st = {"n": len(rows)}
    for k in ("beta_D", "beta_B"):
        b = np.array([[r[f"{k}_{c}"] for c in "RGB"] for r in rows])
        st[k] = {"mean": b.mean(0).tolist(), "sd": b.std(0).tolist(), "min": b.min(0).tolist(), "max": b.max(0).tolist(),
                 "gap_RG_median": float(np.median(b[:, 0] - b[:, 1])), "gap_RG_p95": float(np.percentile(b[:, 0] - b[:, 1], 95)),
                 "gap_GB_median": float(np.median(b[:, 1] - b[:, 2])), "gap_GB_p95": float(np.percentile(b[:, 1] - b[:, 2], 95)),
                 "rel_spread_RB_median": float(np.median((b[:, 0] - b[:, 2]) / b[:, 2])),
                 "rel_spread_RB_p95": float(np.percentile((b[:, 0] - b[:, 2]) / b[:, 2], 95)),
                 "frac_rel_spread_gt_5pct": float(np.mean((b[:, 0] - b[:, 2]) / b[:, 2] > 0.05))}
    summ[s] = st
(out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
print(json.dumps(summ, indent=1))

"""
Apply the pre-registered Stage-1 interpretation rules (execution report §D) to the S1_EVAL517 summary.

Reads  <eval_dir>/summary.json (written by s1_eval517.py). Writes <eval_dir>/LABELS.json and
<out_md> (markdown tables for the report). No inference, no GPU. All numbers are copied from summary.json
fields; nothing is recomputed except the rule decisions.

Rules (§D, on paired dPSNR8 = arm - matched control, per dataset LSUI and UIEB):
  OBVIOUS_FAILURE       : on either dataset, est < -1.0 dB and the 95% CI excludes 0
  CANDIDATE_EFFECT      : on at least one dataset the 95% CI excludes 0  (-> would go to Stage 2)
  REDUNDANT_PROVISIONAL : |est| < 0.2 dB on both datasets and both CIs include 0
  INCONCLUSIVE          : none of the above applies (not a §D category; stated as such)
Every label is EXPLORATORY: single seed (0), validation split (used repeatedly; not held-out).
Matched controls: A2 is built on v1 -> control v1_retrain (A1); A3/A4 are built on v2 -> control v2_full.
The §D wording "each arm minus A1" is also reported for A3/A4 (secondary, not matched).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PRIMARY = [("v2_full", "v1_retrain", "v2 conditioning vs matched v1 (A1)"),
           ("A2_no_physics", "v1_retrain", "physics pathway removed vs A1"),
           ("A3_v2_no_Lphys", "v2_full", "L_phys removed vs v2 (matched base)"),
           ("A4_v2_no_NIG", "v2_full", "NIG loss removed vs v2 (matched base)")]
SECONDARY = [("A3_v2_no_Lphys", "v1_retrain", "§D wording: arm - A1 (bases differ)"),
             ("A4_v2_no_NIG", "v1_retrain", "§D wording: arm - A1 (bases differ)"),
             ("v1_retrain", "v1_orig", "run-to-run reference: A1 vs A0 (A0 resumed, code differs; not matched)")]
DS = ("LSUI", "UIEB")


def excl0(c):
    return c["lo"] > 0 or c["hi"] < 0


def label(r):
    if any(r[d]["est"] < -1.0 and excl0(r[d]) for d in DS):
        return "OBVIOUS_FAILURE"
    if any(excl0(r[d]) for d in DS):
        return "CANDIDATE_EFFECT"
    if all(abs(r[d]["est"]) < 0.2 and not excl0(r[d]) for d in DS):
        return "REDUNDANT_PROVISIONAL"
    return "INCONCLUSIVE"


def ci(c, nd=3, sign=True):
    f = f"{{:{'+' if sign else ''}.{nd}f}}"
    return f"{f.format(c['est'])} [{f.format(c['lo'])}, {f.format(c['hi'])}]"


def main(eval_dir: Path, out_md: Path):
    s = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    P, A = s["paired"], s["absolute"]
    labels = {}
    for group, items in (("primary", PRIMARY), ("secondary", SECONDARY)):
        for a, b, what in items:
            r = P[f"{a} - {b}"]["d_psnr8"]
            labels[f"{a} - {b}"] = {"group": group, "what": what, "label": label(r),
                                    "d_psnr8": {g: r[g] for g in ("LSUI", "UIEB", "ALL")},
                                    "d_ssim8_ALL": P[f"{a} - {b}"]["d_ssim8"]["ALL"]}
    out = {"status": "EXPLORATORY: single seed (0); 517-image validation split, used repeatedly, NOT held-out; "
                     "no claim of component contribution (Stage-2 criterion, §D, not evaluated)",
           "rules": "execution report §D; see s1_labels.py docstring",
           "sigma_labels": {k: A[k].get("sigma_status", "n/a") for k in A},
           "comparisons": labels}
    (eval_dir / "LABELS.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    L = ["| Model | PSNR₈ ALL (n=517) | LSUI (n=428) | UIEB (n=89) | SSIM₈ ALL | σ̂ Spearman ALL | σ̂ AUSE-MAE ALL | devices |",
         "|---|---|---|---|---|---|---|---|"]
    for k, a in A.items():
        sp = ci(a["sigma_spearman_mean"]["ALL"], 3, False) if "sigma_spearman_mean" in a else "n/a"
        au = f"{a['sigma_ause_mae_mean']['ALL']['est']:.4f}" if "sigma_ause_mae_mean" in a else "n/a"
        if "UNTRAINED" in a.get("sigma_status", ""):
            sp, au = sp + " **(untrained)**", au + " **(untrained)**"
        L.append(f"| {k} | {ci(a['psnr8_mean']['ALL'], 3, False)} | {ci(a['psnr8_mean']['LSUI'], 3, False)} | "
                 f"{ci(a['psnr8_mean']['UIEB'], 3, False)} | {ci(a['ssim8_mean']['ALL'], 4, False)} | {sp} | {au} | "
                 f"{a['device_counts']} |")
    L += ["", "| Comparison (a − b) | ΔPSNR₈ LSUI | ΔPSNR₈ UIEB | ΔPSNR₈ ALL | d_z ALL | ΔSSIM₈ ALL | §D label |",
          "|---|---|---|---|---|---|---|"]
    for name, v in labels.items():
        d = v["d_psnr8"]
        L.append(f"| {name} ({v['group']}) | {ci(d['LSUI'])} | {ci(d['UIEB'])} | {ci(d['ALL'])} | "
                 f"{d['ALL']['d_z']:+.2f} | {ci(v['d_ssim8_ALL'], 4)} | {v['label']} |")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"[s1_labels] wrote {eval_dir / 'LABELS.json'} and {out_md}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: s1_labels.py <S1_EVAL517 dir> <out.md>")
    main(Path(sys.argv[1]), Path(sys.argv[2]))

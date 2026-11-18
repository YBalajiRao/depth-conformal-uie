"""
P1c checkpoint (v2 + ordered DGT + decoder FiLM): 517-image validation evaluation with paired bootstrap against
P1b (single-variable base: FiLM is the only difference), S1 v2_full and S1 v1_retrain.

Same protocol and code as p1b_eval517.py (evaluate(), parity gate, depth as aligned aux input, n == 517,
B = 2000, seed 20261107). Comparators are read from their pinned per-image records (SHA-256 + image order checked):
P1b from P1B_EVAL517_20261110_184113; v2_full / v1_retrain from S1_EVAL517_20261109_165139.
EXPLORATORY: single seed; validation reused for decisions; not held-out.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import common  # noqa: E402
from common import (git_state, new_outdir, write_json, environment, Timer, val_rows, boot_groups, by_dataset,  # noqa: E402
                    mean_stat, pooled_psnr_stat, fmt_ci, sha256_file, B_BOOT, BOOT_SEED, ROOT)
from s1_eval517 import parity_fails, dz, save_csv, PARITY_TOL_DB, N_EXPECTED  # noqa: E402
from p1_eval517 import DepthRunner, parity_gate, DEPTH  # noqa: E402
from p1b_eval517 import evaluate  # noqa: E402
from train import DepthStore  # noqa: E402

KEY = "P1c_v2_dgt_ordered_film"
common.CKPTS[KEY] = (ROOT / "runs/phase1/P1c_v2_dgt_ordered_film/20261110_185628/last.pt",
                     "d3a69d62dcaafb62bc2cfb2b64d0641443a6e94e82a31b596c8e8b3a0abe58be", 100000)
REFS = {  # name: (csv path, sha256)
    "P1b": (ROOT / "runs/diagnostics/P1B_EVAL517_20261110_184113/per_image_P1b_v2_dgt_ordered.csv",
            "c16fd969710197074a2faa0f7fda403c88c213a3929e1be6437bc1f66637dccc"),
    "v2_full": (ROOT / "runs/diagnostics/S1_EVAL517_20261109_165139/per_image_v2_full.csv",
                "6388aa038b1193a00bc8a39e28c965635de6be36184b6e65ff833909e5766f39"),
    "v1_retrain": (ROOT / "runs/diagnostics/S1_EVAL517_20261109_165139/per_image_v1_retrain.csv",
                   "8af46e99895daab1e57065f9952fb07838cece2b1ae43ef60b5e7a0901a2949a"),
}


def load_ref(name, rows):
    p, sha = REFS[name]
    got = sha256_file(p)
    if got != sha:
        raise SystemExit(f"[abort] {p.name} SHA-256 {got} != pinned {sha}")
    with open(p, newline="", encoding="utf-8") as f:
        recs = [{k: (float(v) if k in ("mse8", "psnr8", "ssim8") else v) for k, v in r.items()} for r in csv.DictReader(f)]
    if len(recs) != 517 or [r["unit"] for r in recs] != [r["unit"] for r in rows]:
        raise SystemExit(f"[abort] {p.name}: image order/count differs")
    return recs


def main():
    commit = git_state()
    tm = Timer()
    torch.set_num_threads(12)
    out = new_outdir("P1C_EVAL517")
    rows = val_rows()
    counts = {d: sum(r["dataset"] == d for r in rows) for d in N_EXPECTED}
    if counts != N_EXPECTED:
        raise SystemExit(f"[abort] counts {counts}")
    store = DepthStore(DEPTH, [(r["dataset"], r["unit"], r["input"], r["reference"]) for r in rows], check_sha=True)
    refs = {k: load_ref(k, rows) for k in REFS}
    print(f"[p1c_eval517] commit {commit} out {out} images {counts}; refs pinned", flush=True)

    run = DepthRunner(KEY)
    par, ctl = parity_gate(run, rows, store)
    mx = max(r["abs_d_psnr8"] for r in par)
    write_json(out / "parity.json", {"tolerance_db": PARITY_TOL_DB, "max_abs_d_psnr8": mx,
                                     "failure_control_tripped": ctl, "images": par})
    print(f"[parity] max |dPSNR8| = {mx:.6f} dB over {len(par)} images", flush=True)
    if parity_fails([r["abs_d_psnr8"] for r in par]):
        raise SystemExit("[HALT] parity gate failed")
    import p1b_eval517
    p1b_eval517.KEY = KEY                                   # progress label only
    recs = evaluate(run, rows, store)
    if len(recs) != 517:
        raise SystemExit(f"[abort] n={len(recs)}")
    meta_run = run.meta()
    save_csv(out / f"per_image_{KEY}.csv", recs)

    absolute = {"psnr8_mean": boot_groups(by_dataset(recs, "psnr8"), mean_stat),
                "psnr8_pooled": boot_groups(by_dataset(recs, "mse8"), pooled_psnr_stat),
                "ssim8_mean": boot_groups(by_dataset(recs, "ssim8"), mean_stat),
                "sigma_spearman_mean": boot_groups(by_dataset(recs, "sigma_spearman"), mean_stat),
                "device_counts": meta_run["device_counts"],
                "telemetry_mean": {k: float(np.mean([r[k] for r in recs])) for k in
                                   ["mean_tD", "mean_tB"] + [f"{n}_{c}" for n in ("beta_D", "beta_B") for c in "RGB"]}}
    paired = {}
    for k in REFS:
        tmp = [{"dataset": r["dataset"], "d_psnr8": r["psnr8"] - q["psnr8"], "d_ssim8": r["ssim8"] - q["ssim8"]}
               for r, q in zip(recs, refs[k])]
        res = {"role": "single-variable base" if k == "P1b" else "baseline"}
        for m in ("d_psnr8", "d_ssim8"):
            g = by_dataset(tmp, m)
            ci = boot_groups(g, mean_stat)
            dzs = {d: dz(v[0]) for d, v in g.items()}
            dzs["ALL"] = dz([t[m] for t in tmp])
            res[m] = {grp: {**c, "d_z": dzs[grp], "ci_excludes_0": bool(c["lo"] > 0 or c["hi"] < 0),
                            "frac_images_a_better": float(np.mean([t[m] > 0 for t in tmp
                                                                    if grp == "ALL" or t["dataset"] == grp]))}
                      for grp, c in ci.items()}
        paired[f"P1c - {k}"] = res

    write_json(out / "summary.json", {"absolute": absolute, "paired": paired,
                                      "refs": {k: {"file": str(v[0]), "sha256": v[1]} for k, v in REFS.items()},
                                      "labels": "validation (517), not held-out; single seed (0); exploratory",
                                      "bootstrap": {"B": B_BOOT, "seed": BOOT_SEED,
                                                    "type": "paired percentile 95%; ALL stratified by dataset"}})
    write_json(out / "meta.json", {"commit": commit, "model": meta_run, "env": environment(), **tm.done(),
                                   "images": [r["unit"] for r in rows], "counts": counts, "depth_dir": str(DEPTH)})
    a = absolute
    print("\n== P1c absolute (mean, 95% CI) ==")
    for g in ("LSUI", "UIEB", "ALL"):
        print(f"{g:4s} PSNR {fmt_ci(a['psnr8_mean'][g])}  SSIM {fmt_ci(a['ssim8_mean'][g], 4)}")
    print(f"devices {a['device_counts']}")
    print("\n== paired (P1c - b), 95% CI, d_z ==")
    for name, r in paired.items():
        for m in ("d_psnr8", "d_ssim8"):
            nd = 3 if m == "d_psnr8" else 4
            s = "  ".join(f"{g} {c['est']:+.{nd}f} [{c['lo']:+.{nd}f},{c['hi']:+.{nd}f}] dz={c['d_z']:+.2f}"
                          for g, c in r[m].items())
            print(f"{name:20s} {m}  {s}")
    print(f"\n[p1c_eval517] done -> {out}  {tm.done()}", flush=True)


if __name__ == "__main__":
    main()

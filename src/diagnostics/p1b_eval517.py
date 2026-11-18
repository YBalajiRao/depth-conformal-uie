"""
P1b checkpoint (v2 + ordered DGT, direct supervision): 517-image validation evaluation with paired bootstrap
against the pinned Stage-1 per-image records (v1_retrain, v2_full primary; A4 secondary).

Same protocol and machinery as p1_eval517.py (fp32, full resolution, depth as aligned aux input, CUDA with
CPU-on-OOM fallback, 10-image CUDA/CPU parity gate with planted-fault control, n == 517 asserted, B = 2000,
seed 20261107). Differs only in the per-image record: no J_struct / CSA outputs (the model has no CSA).
EXPLORATORY: single seed; validation reused for decisions; not held-out.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import common  # noqa: E402
from common import (git_state, new_outdir, write_json, environment, Timer, val_rows, load_u8, u8_to_tensor,  # noqa: E402
                    to_u8_tensor, mse_u8, psnr_u8, ssim_u8, predictive_std, sparsification, ause, spearman,
                    boot_groups, by_dataset, mean_stat, pooled_psnr_stat, fmt_ci, B_BOOT, BOOT_SEED, ROOT)
from s1_eval517 import parity_fails, dz, save_csv, PARITY_TOL_DB, N_EXPECTED  # noqa: E402
from p1_eval517 import DepthRunner, parity_gate, load_ref, REFS, DEPTH  # noqa: E402
from train import DepthStore  # noqa: E402  (src/training on sys.path via p1_eval517)

KEY = "P1b_v2_dgt_ordered"
common.CKPTS[KEY] = (ROOT / "runs/phase1/P1b_v2_dgt_ordered/20261110_093229/last.pt",
                     "5b22ea7c217b09b96ad2b6309b21b5cbca906444e2f442dd9732965ec9db52c0", 100000)


def evaluate(run, rows, store):
    recs = []
    for i, r in enumerate(rows):
        x8, y8 = load_u8(r["input"]), load_u8(r["reference"])
        if x8.shape != y8.shape:
            raise ValueError(f"{r['unit']}: size mismatch")
        x, z = u8_to_tensor(x8), store.load(r["unit"], x8.shape[:2])[None]
        o, dev = run(x, z)
        p8 = to_u8_tensor(o["J"])
        rec = {"unit": r["unit"], "dataset": r["dataset"], "H": y8.shape[0], "W": y8.shape[1], "device": dev,
               "precision": "fp32", "mse8": mse_u8(p8, y8), "psnr8": psnr_u8(p8, y8), "ssim8": ssim_u8(p8, y8),
               "mean_tD": float(o["tD"].mean()), "mean_tB": float(o["tB"].mean())}
        for nm in ("beta_D", "beta_B"):
            for ci, cn in enumerate("RGB"):
                rec[f"{nm}_{cn}"] = float(o[nm][0, ci])
        if not (np.isfinite(rec["psnr8"]) and np.isfinite(rec["ssim8"])):
            raise SystemExit(f"[abort] non-finite metric for {r['unit']}")
        nig = o["nig"]
        err = (o["J"] - u8_to_tensor(y8)).abs().flatten()
        sig = predictive_std(nig[:, 3:6], nig[:, 6:9], nig[:, 9:12], "total").flatten()
        rec["sigma_spearman"] = spearman(sig, err)
        rec["sigma_ause_mae"] = ause(sparsification(err, sig), "mae")
        recs.append(rec)
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            print(f"  [{KEY}] {i + 1}/{len(rows)} devices={run.device_counts}", flush=True)
    return recs


def main():
    commit = git_state()
    tm = Timer()
    torch.set_num_threads(12)
    out = new_outdir("P1B_EVAL517")
    rows = val_rows()
    counts = {d: sum(r["dataset"] == d for r in rows) for d in N_EXPECTED}
    if counts != N_EXPECTED:
        raise SystemExit(f"[abort] counts {counts}")
    store = DepthStore(DEPTH, [(r["dataset"], r["unit"], r["input"], r["reference"]) for r in rows], check_sha=True)
    refs = {k: load_ref(k, rows) for k in REFS}
    print(f"[p1b_eval517] commit {commit} out {out} images {counts}; refs pinned", flush=True)

    run = DepthRunner(KEY)
    par, ctl = parity_gate(run, rows, store)
    mx = max(r["abs_d_psnr8"] for r in par)
    write_json(out / "parity.json", {"tolerance_db": PARITY_TOL_DB, "max_abs_d_psnr8": mx,
                                     "failure_control_tripped": ctl, "images": par})
    print(f"[parity] max |dPSNR8| = {mx:.6f} dB over {len(par)} images", flush=True)
    if parity_fails([r["abs_d_psnr8"] for r in par]):
        raise SystemExit("[HALT] parity gate failed")
    recs = evaluate(run, rows, store)
    if len(recs) != 517:
        raise SystemExit(f"[abort] n={len(recs)}")
    meta_run = run.meta()
    save_csv(out / f"per_image_{KEY}.csv", recs)

    absolute = {"psnr8_mean": boot_groups(by_dataset(recs, "psnr8"), mean_stat),
                "psnr8_pooled": boot_groups(by_dataset(recs, "mse8"), pooled_psnr_stat),
                "ssim8_mean": boot_groups(by_dataset(recs, "ssim8"), mean_stat),
                "sigma_spearman_mean": boot_groups(by_dataset(recs, "sigma_spearman"), mean_stat),
                "sigma_ause_mae_mean": boot_groups(by_dataset(recs, "sigma_ause_mae"), mean_stat),
                "device_counts": meta_run["device_counts"],
                "telemetry_mean": {k: float(np.mean([r[k] for r in recs])) for k in
                                   ["mean_tD", "mean_tB"] + [f"{n}_{c}" for n in ("beta_D", "beta_B") for c in "RGB"]}}
    paired = {}
    for k, (_, _, prim) in REFS.items():
        tmp = [{"dataset": r["dataset"], "d_psnr8": r["psnr8"] - q["psnr8"], "d_ssim8": r["ssim8"] - q["ssim8"]}
               for r, q in zip(recs, refs[k])]
        res = {"role": "primary" if prim else "secondary"}
        for m in ("d_psnr8", "d_ssim8"):
            g = by_dataset(tmp, m)
            ci = boot_groups(g, mean_stat)
            dzs = {d: dz(v[0]) for d, v in g.items()}
            dzs["ALL"] = dz([t[m] for t in tmp])
            res[m] = {grp: {**c, "d_z": dzs[grp], "ci_excludes_0": bool(c["lo"] > 0 or c["hi"] < 0),
                            "frac_images_a_better": float(np.mean([t[m] > 0 for t in tmp
                                                                    if grp == "ALL" or t["dataset"] == grp]))}
                      for grp, c in ci.items()}
        paired[f"P1b - {k}"] = res

    write_json(out / "summary.json", {"absolute": absolute, "paired": paired,
                                      "refs": {k: {"file": v[0], "sha256": v[1]} for k, v in REFS.items()},
                                      "labels": "validation (517), not held-out; single seed (0); exploratory",
                                      "bootstrap": {"B": B_BOOT, "seed": BOOT_SEED,
                                                    "type": "paired percentile 95%; ALL stratified by dataset"}})
    write_json(out / "meta.json", {"commit": commit, "model": meta_run, "env": environment(), **tm.done(),
                                   "images": [r["unit"] for r in rows], "counts": counts, "depth_dir": str(DEPTH)})
    a = absolute
    print("\n== P1b absolute (mean, 95% CI) ==")
    for g in ("LSUI", "UIEB", "ALL"):
        print(f"{g:4s} PSNR {fmt_ci(a['psnr8_mean'][g])}  SSIM {fmt_ci(a['ssim8_mean'][g], 4)}")
    print(f"devices {a['device_counts']}")
    print("\n== paired (P1b - b), 95% CI, d_z ==")
    for name, r in paired.items():
        for m in ("d_psnr8", "d_ssim8"):
            nd = 3 if m == "d_psnr8" else 4
            s = "  ".join(f"{g} {c['est']:+.{nd}f} [{c['lo']:+.{nd}f},{c['hi']:+.{nd}f}] dz={c['d_z']:+.2f}"
                          for g, c in r[m].items())
            print(f"{name:22s} {m}  {s}")
    print(f"\n[p1b_eval517] done -> {out}  {tm.done()}", flush=True)


if __name__ == "__main__":
    main()

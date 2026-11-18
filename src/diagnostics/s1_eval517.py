"""
Stage-1 517-image validation evaluation (6 checkpoints: original v1 + Stage-1 runs 1-5).

Validation split only (428 LSUI + 89 UIEB), not held-out. No test split, no training, no checkpoint writes.

Protocol (approved 2026-11-09):
  * Each last.pt is pinned to its SHA-256 and stored step 100000; abort on mismatch.
  * Inference: fp32, no autocast, eval, no_grad, full native resolution, PadCropWrapper(reflect, multiple 16),
    deterministic cuDNN. CUDA first; on CUDA OOM the same image is re-run full-image on CPU fp32 (common.Runner).
    No tiling. Device used is recorded per image.
  * Parity gate (before any full pass): for every model, 10 validation images (5 LSUI + 5 UIEB, drawn with
    seed 20261107 from images that ran on CUDA) are forwarded on CUDA and on CPU. Hard halt if any
    |PSNR8_cuda - PSNR8_cpu| >= 0.01 dB. A planted fault (1/255 offset on the CPU output of one image) must
    trip the same comparison function, otherwise the script halts (failure control).
  * n == 517 per model is asserted; an image that fails on both devices raises (no silent skip).
  * Paired per-image bootstrap, B = 2000, seed 20261107 (common.boot_groups), per dataset + stratified ALL.
    Effect size: paired d_z = mean(delta) / sd(delta, ddof=1).
  * Uncertainty (sigma from the NIG head): reported per model; for A4 (le = 0) the head received no
    training signal and its numbers are labelled UNTRAINED.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import common  # noqa: E402
from common import (ROOT, Runner, git_state, new_outdir, write_json, environment, Timer, val_rows, load_u8,  # noqa: E402
                    u8_to_tensor, to_u8_tensor, mse_u8, psnr_u8, ssim_u8, predictive_std, sparsification, ause,
                    spearman, boot_groups, by_dataset, mean_stat, pooled_psnr_stat, load_model, fmt_ci,
                    B_BOOT, BOOT_SEED)

S1 = ROOT / "runs/ablation/stage1"
STEP = 100000
MODELS = {  # key: (path, sha256, label)
    "v1_orig": (ROOT / "runs/sspgn/20261106_173805/last.pt",
                "be3c23cf157295ffcae738cb3468c79b09611ad04b6c0ee7bc1003895f81499a", "original v1 (pre-Stage-1)"),
    "v1_retrain": (S1 / "S1_v1_retrain/20261107_102639/last.pt",
                   "42e1b674cb864362275f3f04df061f6c07b815252df738db63993da7b986a624", "Run 1 S1_v1_retrain"),
    "v2_full": (S1 / "S1_v2_full/20261107_193709/last.pt",
                "8a50ea4dc3dd43c2dd23af343df531df4ca5185ff30ad0955dddb5dad60b2eaa", "Run 2 S1_v2_full"),
    "A2_no_physics": (S1 / "S1_A2_no_physics/20261108_084156/last.pt",
                      "62f7604c78145a959e67532049b71502dd7bf71e595c72e67b78b2752179ac99", "Run 3 S1_A2_no_physics"),
    "A3_v2_no_Lphys": (S1 / "S1_A3_v2_no_Lphys/20261108_184308/last.pt",
                       "91a4fdeaa3ea973b3d06716542e7886582da7d2466d9ba001650a4c505d334fd", "Run 4 S1_A3_v2_no_Lphys"),
    "A4_v2_no_NIG": (S1 / "S1_A4_v2_no_NIG/20261109_074648/last.pt",
                     "f70fb83c123eb56b867f50d099c2bb7362a3e5e6d87c91a572e24481507b02c4", "Run 5 S1_A4_v2_no_NIG"),
}
UNTRAINED_SIGMA = {"A4_v2_no_NIG"}
COMPARISONS = [  # (a, b): delta = a - b
    ("v2_full", "v1_retrain"), ("v2_full", "v1_orig"), ("v1_retrain", "v1_orig"),
    ("A2_no_physics", "v2_full"), ("A3_v2_no_Lphys", "v2_full"), ("A4_v2_no_NIG", "v2_full"),
    ("A2_no_physics", "v1_retrain"), ("A3_v2_no_Lphys", "v1_retrain"), ("A4_v2_no_NIG", "v1_retrain"),
    ("A2_no_physics", "v1_orig"), ("A3_v2_no_Lphys", "v1_orig"), ("A4_v2_no_NIG", "v1_orig"),
]
N_EXPECTED = {"LSUI": 428, "UIEB": 89}
PARITY_PER_DS, PARITY_TOL_DB = 5, 0.01

# register pinned checkpoints so common.Runner / verify_ckpt verify SHA-256 and stored step
for _k, (_p, _s, _l) in MODELS.items():
    common.CKPTS[_k] = (_p, _s, STEP)


def dz(d):
    d = np.asarray(d, np.float64)
    s = float(np.std(d, ddof=1))
    return float(np.mean(d) / s) if s > 0 else float("nan")


# ----------------------------------------------------------------------------- parity gate
def parity_delta(p8_a, p8_b, y8):
    return abs(psnr_u8(p8_a, y8) - psnr_u8(p8_b, y8))


def parity_fails(deltas):
    return any((not np.isfinite(d)) or d >= PARITY_TOL_DB for d in deltas)


# Failure control (revised 2026-11-17). The former +1 grey-level offset could cancel against a model's own mean bias:
# P1e_s0 on LSUI:3142 has mean(output - ref) = -0.566 levels, so +1 moved PSNR8 by only 0.004 dB (< tol) although
# GPU-vs-CPU parity was 0.000004 dB. A +/-8 checkerboard cannot cancel (it raises MSE by ~64 for any bias); it moved
# PSNR8 by 1.14 dB on that image. The parity tolerance itself (PARITY_TOL_DB) is unchanged.
PLANT_LEVELS = 8


def planted_fault(p8):
    s = ((np.indices(p8.shape[:2]).sum(0) % 2) * 2 - 1)[..., None] * PLANT_LEVELS
    return np.clip(p8.astype(np.int16) + s, 0, 255).astype(np.uint8)


def control_trips(pc, pp, y8):
    """True iff the planted fault on the CPU output trips the same PSNR8 parity comparison AND is visible as a
    max absolute pixel difference of at least PLANT_LEVELS - 1 (allowing the 1-level CUDA/CPU rounding)."""
    bad = planted_fault(pp)
    max_abs = int(np.abs(pc.astype(np.int16) - bad.astype(np.int16)).max())
    return parity_fails([parity_delta(pc, bad, y8)]) and max_abs >= PLANT_LEVELS - 1


def parity_gate(run: Runner, rows):
    rng = np.random.default_rng(BOOT_SEED)
    recs, planted_checked = [], False
    for ds in ("LSUI", "UIEB"):
        cand = [r for r in rows if r["dataset"] == ds]
        order = rng.permutation(len(cand))
        got = 0
        for j in order:
            if got == PARITY_PER_DS:
                break
            r = cand[j]
            x8, y8 = load_u8(r["input"]), load_u8(r["reference"])
            x = u8_to_tensor(x8)
            oc, devc = run(x)
            if devc != "cuda":          # did not fit on GPU: not eligible for parity, draw next
                continue
            op, devp = run(x, force_device="cpu")
            assert devp == "cpu"
            pc, pp = to_u8_tensor(oc["J"]), to_u8_tensor(op["J"])
            d = parity_delta(pc, pp, y8)
            recs.append({"unit": r["unit"], "dataset": ds, "psnr8_cuda": psnr_u8(pc, y8),
                         "psnr8_cpu": psnr_u8(pp, y8), "abs_d_psnr8": d,
                         "max_abs_mu": float((oc["J"] - op["J"]).abs().max()),
                         "u8_pixels_differ": int((pc != pp).any(-1).sum())})
            if not planted_checked:     # failure control: a +/-8 checkerboard must trip the same check
                if not control_trips(pc, pp, y8):
                    raise SystemExit("[abort] parity failure control did not fail; gate has no power")
                planted_checked = True
            got += 1
        if got < PARITY_PER_DS:
            raise SystemExit(f"[abort] only {got} {ds} images fit on CUDA for parity")
    return recs, planted_checked


# ----------------------------------------------------------------------------- full pass
def evaluate(run: Runner, rows):
    recs = []
    for i, r in enumerate(rows):
        x8, y8 = load_u8(r["input"]), load_u8(r["reference"])
        if x8.shape != y8.shape:
            raise ValueError(f"{r['unit']}: input/reference size mismatch")
        x = u8_to_tensor(x8)
        o, dev = run(x)
        mu = o["J"]
        p8 = to_u8_tensor(mu)
        rec = {"unit": r["unit"], "dataset": r["dataset"], "H": y8.shape[0], "W": y8.shape[1], "device": dev,
               "precision": "fp32", "mse8": mse_u8(p8, y8), "psnr8": psnr_u8(p8, y8), "ssim8": ssim_u8(p8, y8)}
        if not np.isfinite(rec["psnr8"]) or not np.isfinite(rec["ssim8"]):
            raise SystemExit(f"[abort] non-finite metric for {r['unit']} ({run.key})")
        nig = o.get("nig")
        if nig is not None:
            err = (mu - u8_to_tensor(y8)).abs().flatten()
            sig = predictive_std(nig[:, 3:6], nig[:, 6:9], nig[:, 9:12], "total").flatten()
            rec["sigma_spearman"] = spearman(sig, err)
            rec["sigma_ause_mae"] = ause(sparsification(err, sig), "mae")
        recs.append(rec)
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            print(f"  [{run.key}] {i + 1}/{len(rows)}  devices={run.device_counts}", flush=True)
    return recs


def save_csv(path, recs):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)


def main():
    commit = git_state()
    tm = Timer()
    torch.set_num_threads(12)
    out = new_outdir("S1_EVAL517")
    rows = val_rows()
    counts = {d: sum(r["dataset"] == d for r in rows) for d in N_EXPECTED}
    if counts != N_EXPECTED:
        raise SystemExit(f"[abort] validation counts {counts} != {N_EXPECTED}")
    print(f"[s1_eval517] commit {commit}  out {out}  images {counts}", flush=True)

    # ---- parity gate for every model, before any full pass
    parity, metas = {}, {}
    for k in MODELS:
        run = Runner(k)
        recs, ctl = parity_gate(run, rows)
        parity[k] = {"images": recs, "max_abs_d_psnr8": max(r["abs_d_psnr8"] for r in recs),
                     "failure_control_tripped": ctl}
        print(f"[parity] {k}: max |dPSNR8| = {parity[k]['max_abs_d_psnr8']:.6f} dB over {len(recs)} images", flush=True)
        del run; torch.cuda.empty_cache()
        if parity_fails([r["abs_d_psnr8"] for r in recs]):
            write_json(out / "parity.json", parity)
            raise SystemExit(f"[HALT] parity gate failed for {k} (tolerance {PARITY_TOL_DB} dB)")
    write_json(out / "parity.json", {"tolerance_db": PARITY_TOL_DB, "models": parity})
    print("[parity] PASS for all models", flush=True)

    # ---- full pass
    per = {}
    for k in MODELS:
        run = Runner(k)
        recs = evaluate(run, rows)
        if len(recs) != 517 or [r["unit"] for r in recs] != [r["unit"] for r in rows]:
            raise SystemExit(f"[abort] {k}: n={len(recs)} != 517 or order mismatch")
        metas[k] = run.meta()
        save_csv(out / f"per_image_{k}.csv", recs)
        per[k] = recs
        del run; torch.cuda.empty_cache()

    # ---- summaries
    absolute = {}
    for k, recs in per.items():
        a = {"label": MODELS[k][2],
             "psnr8_mean": boot_groups(by_dataset(recs, "psnr8"), mean_stat),
             "psnr8_pooled": boot_groups(by_dataset(recs, "mse8"), pooled_psnr_stat),
             "ssim8_mean": boot_groups(by_dataset(recs, "ssim8"), mean_stat),
             "device_counts": metas[k]["device_counts"]}
        if "sigma_spearman" in recs[0]:
            a["sigma_spearman_mean"] = boot_groups(by_dataset(recs, "sigma_spearman"), mean_stat)
            a["sigma_ause_mae_mean"] = boot_groups(by_dataset(recs, "sigma_ause_mae"), mean_stat)
            a["sigma_status"] = "UNTRAINED (le = 0; NIG head received no training signal)" \
                if k in UNTRAINED_SIGMA else "trained (le = 0.01)"
        absolute[k] = a

    paired = {}
    for a_k, b_k in COMPARISONS:
        A, B = per[a_k], per[b_k]
        assert [r["unit"] for r in A] == [r["unit"] for r in B]
        tmp = [{"dataset": ra["dataset"], "d_psnr8": ra["psnr8"] - rb["psnr8"],
                "d_ssim8": ra["ssim8"] - rb["ssim8"]} for ra, rb in zip(A, B)]
        res = {}
        for m in ("d_psnr8", "d_ssim8"):
            g = by_dataset(tmp, m)
            ci = boot_groups(g, mean_stat)
            dzs = {d: dz(v[0]) for d, v in g.items()}
            dzs["ALL"] = dz([t[m] for t in tmp])
            res[m] = {grp: {**c, "d_z": dzs[grp], "ci_excludes_0": bool(c["lo"] > 0 or c["hi"] < 0),
                            "frac_images_a_better": float(np.mean([t[m] > 0 for t in tmp
                                                                    if grp == "ALL" or t["dataset"] == grp]))}
                      for grp, c in ci.items()}
        paired[f"{a_k} - {b_k}"] = res

    write_json(out / "summary.json", {"absolute": absolute, "paired": paired,
                                      "labels": "validation (517), not held-out; single seed (0); exploratory",
                                      "bootstrap": {"B": B_BOOT, "seed": BOOT_SEED,
                                                    "type": "paired percentile 95%; ALL stratified by dataset"}})
    write_json(out / "meta.json", {"commit": commit, "models": metas, "env": environment(), **tm.done(),
                                   "images": [r["unit"] for r in rows], "counts": counts})

    # ---- console table
    print("\n== absolute (mean PSNR8 / SSIM8, 95% CI) ==", flush=True)
    for k, a in absolute.items():
        print(f"{k:16s} PSNR ALL {fmt_ci(a['psnr8_mean']['ALL'])}  LSUI {fmt_ci(a['psnr8_mean']['LSUI'])}  "
              f"UIEB {fmt_ci(a['psnr8_mean']['UIEB'])}  SSIM ALL {fmt_ci(a['ssim8_mean']['ALL'], 4)}  "
              f"dev {a['device_counts']}", flush=True)
    print("\n== paired dPSNR8 (a - b), 95% CI, d_z ==", flush=True)
    for name, r in paired.items():
        s = "  ".join(f"{g} {c['est']:+.3f} [{c['lo']:+.3f},{c['hi']:+.3f}] dz={c['d_z']:+.2f}"
                      for g, c in r["d_psnr8"].items())
        print(f"{name:34s} {s}", flush=True)
    print(f"\n[s1_eval517] done -> {out}  {tm.done()}", flush=True)


if __name__ == "__main__":
    main()

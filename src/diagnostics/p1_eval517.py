"""
Phase-1 checkpoint: 517-image validation evaluation with paired bootstrap against Stage-1 models.

Validation split only (428 LSUI + 89 UIEB), not held-out. No test split, no training, no checkpoint writes.

Protocol: identical to s1_eval517.py (fp32, no autocast, full native resolution, PadCropWrapper reflect x16,
deterministic cuDNN, CUDA with full-image CPU fp32 fallback on OOM, n == 517 asserted, 10-image CUDA/CPU
parity gate with |dPSNR8| < 0.01 dB and a planted-fault control), with one addition: the precomputed depth
map (DepthStore, bound to each input's SHA-256) is passed as an aligned auxiliary input and padded exactly like
the image.

Stage-1 comparators are NOT re-run: their per-image records from S1_EVAL517_20261109_165139 (same metric code,
same protocol) are read, SHA-256-pinned, and checked for identical image order. Pairing is per image.
Paired bootstrap B = 2000, seed 20261107 (common.boot_groups); effect size d_z = mean(delta)/sd(delta).
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
from common import (ROOT, Runner, git_state, new_outdir, write_json, environment, Timer, val_rows, load_u8,  # noqa: E402
                    u8_to_tensor, to_u8_tensor, mse_u8, psnr_u8, ssim_u8, predictive_std, sparsification, ause,
                    spearman, boot_groups, by_dataset, mean_stat, pooled_psnr_stat, fmt_ci, sha256_file,
                    B_BOOT, BOOT_SEED)
from s1_eval517 import (parity_delta, parity_fails, control_trips, dz, save_csv, PARITY_PER_DS,  # noqa: E402
                        PARITY_TOL_DB, N_EXPECTED)
sys.path.insert(0, str(ROOT / "src/training"))
from train import DepthStore  # noqa: E402

KEY = "P1_dgt_csa_aligned"
P1 = (ROOT / "runs/phase1/P1_v2_dgt_csa_aligned/20261109_192224/last.pt",
      "c7ed099309f0941f831c1318ff182143be74694cf93104c0ec2a80dbe4a20e2c", 100000)
common.CKPTS[KEY] = P1
DEPTH = ROOT / "data/depth/da2_small_5426e4f/lsui_uieb_all_d6_seed0"
S1DIR = ROOT / "runs/diagnostics/S1_EVAL517_20261109_165139"
REFS = {  # key: (per-image csv, sha256, primary?)
    "v1_retrain": ("per_image_v1_retrain.csv", "8af46e99895daab1e57065f9952fb07838cece2b1ae43ef60b5e7a0901a2949a", True),
    "v2_full": ("per_image_v2_full.csv", "6388aa038b1193a00bc8a39e28c965635de6be36184b6e65ff833909e5766f39", True),
    "A4_v2_no_NIG": ("per_image_A4_v2_no_NIG.csv", "31ca512bccc95035c2bbaeffeb9a4c30e3b17323140d37002fd7ca12fb3405f1", False),
}


class DepthRunner(Runner):
    """common.Runner with the depth map passed as an aligned aux input (same CUDA -> CPU-on-OOM logic)."""

    @torch.no_grad()
    def __call__(self, x, z, force_device=None):
        dev = force_device or self.device.type
        if dev == "cuda":
            try:
                o = self.wrapped(x.to(self.device), z.to(self.device))
                o = {k: v.float().cpu() if torch.is_tensor(v) else v for k, v in o.items()}
                self.device_counts["cuda"] = self.device_counts.get("cuda", 0) + 1
                return o, "cuda"
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
        o = self._cpu_wrapped()(x.cpu(), z.cpu())
        o = {k: v.float() if torch.is_tensor(v) else v for k, v in o.items()}
        self.device_counts["cpu"] = self.device_counts.get("cpu", 0) + 1
        return o, "cpu"


def parity_gate(run, rows, store):
    rng = np.random.default_rng(BOOT_SEED)
    recs, ctl = [], False
    for ds in ("LSUI", "UIEB"):
        cand = [r for r in rows if r["dataset"] == ds]
        got = 0
        for j in rng.permutation(len(cand)):
            if got == PARITY_PER_DS:
                break
            r = cand[j]
            x8, y8 = load_u8(r["input"]), load_u8(r["reference"])
            x, z = u8_to_tensor(x8), store.load(r["unit"], x8.shape[:2])[None]
            oc, dc = run(x, z)
            if dc != "cuda":
                continue
            op, _ = run(x, z, force_device="cpu")
            pc, pp = to_u8_tensor(oc["J"]), to_u8_tensor(op["J"])
            d = parity_delta(pc, pp, y8)
            recs.append({"unit": r["unit"], "dataset": ds, "abs_d_psnr8": d,
                         "max_abs_mu": float((oc["J"] - op["J"]).abs().max())})
            if not ctl:                 # failure control: +/-8 checkerboard (s1_eval517.control_trips)
                if not control_trips(pc, pp, y8):
                    raise SystemExit("[abort] parity failure control did not fail")
                ctl = True
            got += 1
        if got < PARITY_PER_DS:
            raise SystemExit(f"[abort] only {got} {ds} images fit on CUDA for parity")
    return recs, ctl


def evaluate(run, rows, store):
    recs = []
    for i, r in enumerate(rows):
        x8, y8 = load_u8(r["input"]), load_u8(r["reference"])
        if x8.shape != y8.shape:
            raise ValueError(f"{r['unit']}: size mismatch")
        x, z = u8_to_tensor(x8), store.load(r["unit"], x8.shape[:2])[None]
        o, dev = run(x, z)
        p8, s8 = to_u8_tensor(o["J"]), to_u8_tensor(o["J_struct"])
        rec = {"unit": r["unit"], "dataset": r["dataset"], "H": y8.shape[0], "W": y8.shape[1], "device": dev,
               "precision": "fp32", "mse8": mse_u8(p8, y8), "psnr8": psnr_u8(p8, y8), "ssim8": ssim_u8(p8, y8),
               "mse8_struct": mse_u8(s8, y8), "psnr8_struct": psnr_u8(s8, y8),
               "mean_tD": float(o["tD"].mean()), "mean_tB": float(o["tB"].mean())}
        for nm in ("beta_D", "beta_B"):
            for ci, cn in enumerate("RGB"):
                rec[f"{nm}_{cn}"] = float(o[nm][0, ci])
        rec["csa_MI"] = float((o["csa_M"][0] - torch.eye(3)).abs().mean())
        if not (np.isfinite(rec["psnr8"]) and np.isfinite(rec["ssim8"])):
            raise SystemExit(f"[abort] non-finite metric for {r['unit']}")
        nig = o["nig"]                                    # NIG mean = J_final (Phase-1 design)
        err = (o["J"] - u8_to_tensor(y8)).abs().flatten()
        sig = predictive_std(nig[:, 3:6], nig[:, 6:9], nig[:, 9:12], "total").flatten()
        rec["sigma_spearman"] = spearman(sig, err)
        rec["sigma_ause_mae"] = ause(sparsification(err, sig), "mae")
        recs.append(rec)
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            print(f"  [{KEY}] {i + 1}/{len(rows)} devices={run.device_counts}", flush=True)
    return recs


def load_ref(name, rows):
    fn, sha, _ = REFS[name]
    p = S1DIR / fn
    got = sha256_file(p)
    if got != sha:
        raise SystemExit(f"[abort] {p.name} SHA-256 {got} != pinned {sha}")
    with open(p, newline="", encoding="utf-8") as f:
        recs = [{k: (float(v) if k in ("mse8", "psnr8", "ssim8") else v) for k, v in r.items()} for r in csv.DictReader(f)]
    if [r["unit"] for r in recs] != [r["unit"] for r in rows] or len(recs) != 517:
        raise SystemExit(f"[abort] {p.name}: image order/count differs from the validation manifest")
    return recs


def main():
    commit = git_state()
    tm = Timer()
    torch.set_num_threads(12)
    out = new_outdir("P1_EVAL517")
    rows = val_rows()
    counts = {d: sum(r["dataset"] == d for r in rows) for d in N_EXPECTED}
    if counts != N_EXPECTED:
        raise SystemExit(f"[abort] counts {counts}")
    store = DepthStore(DEPTH, [(r["dataset"], r["unit"], r["input"], r["reference"]) for r in rows], check_sha=True)
    refs = {k: load_ref(k, rows) for k in REFS}
    print(f"[p1_eval517] commit {commit} out {out} images {counts}; refs pinned", flush=True)

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

    absolute = {"P1": {"psnr8_mean": boot_groups(by_dataset(recs, "psnr8"), mean_stat),
                       "psnr8_pooled": boot_groups(by_dataset(recs, "mse8"), pooled_psnr_stat),
                       "ssim8_mean": boot_groups(by_dataset(recs, "ssim8"), mean_stat),
                       "psnr8_struct_mean": boot_groups(by_dataset(recs, "psnr8_struct"), mean_stat),
                       "sigma_spearman_mean": boot_groups(by_dataset(recs, "sigma_spearman"), mean_stat),
                       "sigma_ause_mae_mean": boot_groups(by_dataset(recs, "sigma_ause_mae"), mean_stat),
                       "device_counts": meta_run["device_counts"],
                       "telemetry_mean": {k: float(np.mean([r[k] for r in recs])) for k in
                                          ["mean_tD", "mean_tB", "csa_MI"] +
                                          [f"{n}_{c}" for n in ("beta_D", "beta_B") for c in "RGB"]}}}
    paired = {}

    def pair(name, a_vals, b_vals, key_ssim=None):
        tmp = [{"dataset": r["dataset"], "d_psnr8": a - b} for r, a, b in zip(recs, a_vals, b_vals)]
        if key_ssim is not None:
            for t, (sa, sb) in zip(tmp, key_ssim):
                t["d_ssim8"] = sa - sb
        res = {}
        for m in [k for k in ("d_psnr8", "d_ssim8") if k in tmp[0]]:
            g = by_dataset(tmp, m)
            ci = boot_groups(g, mean_stat)
            dzs = {d: dz(v[0]) for d, v in g.items()}
            dzs["ALL"] = dz([t[m] for t in tmp])
            res[m] = {grp: {**c, "d_z": dzs[grp], "ci_excludes_0": bool(c["lo"] > 0 or c["hi"] < 0),
                            "frac_images_a_better": float(np.mean([t[m] > 0 for t in tmp
                                                                    if grp == "ALL" or t["dataset"] == grp]))}
                      for grp, c in ci.items()}
        paired[name] = res

    for k, (_, _, prim) in REFS.items():
        pair(f"P1 - {k}", [r["psnr8"] for r in recs], [r["psnr8"] for r in refs[k]],
             [(r["ssim8"], q["ssim8"]) for r, q in zip(recs, refs[k])])
        paired[f"P1 - {k}"]["role"] = "primary" if prim else "secondary"
    pair("P1 J_final - P1 J_struct (CSA effect)", [r["psnr8"] for r in recs], [r["psnr8_struct"] for r in recs])
    paired["P1 J_final - P1 J_struct (CSA effect)"]["role"] = "internal"

    write_json(out / "summary.json", {"absolute": absolute, "paired": paired,
                                      "refs": {k: {"file": str(S1DIR / v[0]), "sha256": v[1]} for k, v in REFS.items()},
                                      "labels": "validation (517), not held-out; single seed (0); exploratory",
                                      "bootstrap": {"B": B_BOOT, "seed": BOOT_SEED,
                                                    "type": "paired percentile 95%; ALL stratified by dataset"}})
    write_json(out / "meta.json", {"commit": commit, "model": meta_run, "env": environment(), **tm.done(),
                                   "images": [r["unit"] for r in rows], "counts": counts, "depth_dir": str(DEPTH)})
    a = absolute["P1"]
    print("\n== P1 absolute (mean PSNR8 / SSIM8, 95% CI) ==")
    print(f"PSNR ALL {fmt_ci(a['psnr8_mean']['ALL'])}  LSUI {fmt_ci(a['psnr8_mean']['LSUI'])}  "
          f"UIEB {fmt_ci(a['psnr8_mean']['UIEB'])}  SSIM ALL {fmt_ci(a['ssim8_mean']['ALL'], 4)}  "
          f"J_struct ALL {fmt_ci(a['psnr8_struct_mean']['ALL'])}  dev {a['device_counts']}")
    print("\n== paired dPSNR8 (a - b), 95% CI, d_z ==")
    for name, r in paired.items():
        s = "  ".join(f"{g} {c['est']:+.3f} [{c['lo']:+.3f},{c['hi']:+.3f}] dz={c['d_z']:+.2f}"
                      for g, c in r["d_psnr8"].items())
        print(f"{name:40s} {s}")
    print(f"\n[p1_eval517] done -> {out}  {tm.done()}", flush=True)


if __name__ == "__main__":
    main()

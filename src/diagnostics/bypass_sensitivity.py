"""
Physics-path bypass diagnostic (PI-approved 2026-11-10). Evaluation only: no training, no checkpoint writes.

Question: does the restoration output depend on the physics inputs? For each model, every one of the 517
validation images is restored natively and under perturbations of the physics input; per-image paired deltas
(perturbed - native) of PSNR8 are bootstrapped (B = 2000, seed 20261107, per dataset + stratified ALL), and the
mean absolute change of the 8-bit output |dJ| is recorded (target-independent sensitivity).

P1 (DGT; runs/phase1/.../last.pt): depth input z~ perturbed
    const05 : z~ = 0.5 everywhere
    shuffle : z~ of another validation image (fixed derangement, seed 20261107), bilinearly resized
    vflip   : z~ flipped vertically
S1 v2_full: the field-branch t_D, t_B at H/16 (Pi and high-band gain) perturbed by a forward hook on
    bottleneck[0].phys_out (pre-sigmoid; channels per colour = (tD, tB, B_low)):
    const05 : t_D = t_B = 0.5 (pre-activation -2, since t = sigmoid(raw + 2))
    perm    : spatial positions of the t_D, t_B maps permuted (fixed seed)
    vflip   : t_D, t_B maps flipped vertically
    extreme : t_D = t_B = 0.05 (power check: a large out-of-distribution change)
In compose='direct' (mu = x + R) these inputs reach mu only through the bottleneck conditioning (and, for P1,
the CSA input statistics are unaffected), so the deltas measure the restoration path's use of the physics.

Self-checks (abort on failure): native per-image PSNR8 equals the pinned earlier evaluation CSVs (max |d| < 1e-6);
each perturbation measurably changes the fields it targets (checked on the first image).
Inference protocol as s1_eval517/p1_eval517 (fp32, full resolution, CUDA with CPU-on-OOM fallback).
EXPLORATORY: single seed; validation, not held-out.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import common  # noqa: E402
from common import (ROOT, Runner, git_state, new_outdir, write_json, environment, Timer, val_rows, load_u8,  # noqa: E402
                    u8_to_tensor, to_u8_tensor, mse_u8, psnr_u8, boot_groups, by_dataset, mean_stat, sha256_file,
                    B_BOOT, BOOT_SEED)
from s1_eval517 import dz, save_csv  # noqa: E402  (also registers the Stage-1 checkpoints in common.CKPTS)
from p1_eval517 import DepthRunner, KEY as P1KEY, DEPTH  # noqa: E402  (registers the P1 checkpoint)
sys.path.insert(0, str(ROOT / "src/training"))
from train import DepthStore  # noqa: E402

REF_CSV = {
    P1KEY: (ROOT / "runs/diagnostics/P1_EVAL517_20261110_081052/per_image_P1_dgt_csa_aligned.csv", None),
    "v2_full": (ROOT / "runs/diagnostics/S1_EVAL517_20261109_165139/per_image_v2_full.csv",
                "6388aa038b1193a00bc8a39e28c965635de6be36184b6e65ff833909e5766f39"),
}


# ----------------------------------------------------------------------------- v2 field hook
class FieldHook:
    """Forward hook on SS-PGB.phys_out; mode None = no-op. Channels (N, 9, h, w) grouped (colour, {tD,tB,B})."""

    def __init__(self):
        self.mode, self.fired = None, 0

    def __call__(self, module, inp, out):
        if self.mode is None:
            return out
        n, c, h, w = out.shape
        o = out.view(n, 3, 3, h, w).clone()
        tt = o[:, :, 0:2]                                           # (n, 3, 2, h, w) tD, tB pre-activation
        if self.mode == "const05":
            tt = torch.full_like(tt, -2.0)                          # sigmoid(-2 + 2) = 0.5
        elif self.mode == "extreme":
            tt = torch.full_like(tt, float(np.log(0.05 / 0.95)) - 2.0)   # sigmoid(. + 2) = 0.05
        elif self.mode == "vflip":
            tt = torch.flip(tt, (-2,))
        elif self.mode == "perm":
            g = torch.Generator().manual_seed(BOOT_SEED)
            idx = torch.randperm(h * w, generator=g).to(tt.device)
            tt = tt.flatten(-2)[..., idx].view_as(tt)
        else:
            raise ValueError(self.mode)
        o[:, :, 0:2] = tt
        self.fired += 1
        return o.view(n, c, h, w)


class HookedRunner(Runner):
    def __init__(self, key, hook):
        super().__init__(key)
        self.hook = hook
        self.model.bottleneck[0].phys_out.register_forward_hook(hook)

    def _cpu_wrapped(self):
        new = self._cpu is None
        w = super()._cpu_wrapped()
        if new:
            self._cpu[0].bottleneck[0].phys_out.register_forward_hook(self.hook)
        return w


# ----------------------------------------------------------------------------- helpers
def ref_psnr(key, rows):
    p, sha = REF_CSV[key]
    if sha is not None and sha256_file(p) != sha:
        raise SystemExit(f"[abort] {p.name} SHA-256 mismatch")
    with open(p, newline="", encoding="utf-8") as f:
        recs = list(csv.DictReader(f))
    if [r["unit"] for r in recs] != [r["unit"] for r in rows]:
        raise SystemExit(f"[abort] {p.name} image order differs")
    return np.array([float(r["psnr8"]) for r in recs])


def derangement(n):
    rng = np.random.default_rng(BOOT_SEED)
    p = rng.permutation(n)
    return np.roll(p, 1)[np.argsort(p)]                             # maps i -> another index, no fixed points


def summarise(rows, native, pert):
    out = {}
    for cond, recs in pert.items():
        tmp = [{"dataset": r["dataset"], "d_psnr8": p["psnr8"] - n["psnr8"], "mad8": p["mad8"]}
               for r, n, p in zip(rows, native, recs)]
        g = by_dataset(tmp, "d_psnr8")
        ci = boot_groups(g, mean_stat)
        dzs = {d: dz(v[0]) for d, v in g.items()}
        dzs["ALL"] = dz([t["d_psnr8"] for t in tmp])
        out[cond] = {"d_psnr8": {grp: {**c, "d_z": dzs[grp],
                                       "within_pm_0p10": bool(-0.10 <= c["lo"] and c["hi"] <= 0.10)}
                                 for grp, c in ci.items()},
                     "mean_abs_dJ8": boot_groups(by_dataset(tmp, "mad8"), mean_stat)}
    return out


# ----------------------------------------------------------------------------- passes
def run_p1(rows, store, out):
    run = DepthRunner(P1KEY)
    perm = derangement(len(rows))
    conds = ("native", "const05", "shuffle", "vflip")
    res = {c: [] for c in conds}
    for i, r in enumerate(rows):
        x8, y8 = load_u8(r["input"]), load_u8(r["reference"])
        x = u8_to_tensor(x8)
        z = store.load(r["unit"], x8.shape[:2])[None]
        o_r = rows[perm[i]]
        z_other = store.load(o_r["unit"], load_u8(o_r["input"]).shape[:2])[None]
        zs = {"native": z, "const05": torch.full_like(z, 0.5),
              "shuffle": F.interpolate(z_other, size=x8.shape[:2], mode="bilinear", align_corners=False),
              "vflip": torch.flip(z, (-2,))}
        p_nat = None
        for c in conds:
            o, dev = run(x, zs[c])
            p8 = to_u8_tensor(o["J"])
            if c == "native":
                p_nat = p8
            if i == 0 and c != "native":                            # the perturbation must reach the fields
                o_n, _ = run(x, zs["native"])
                if torch.equal(o["tD"], o_n["tD"]):
                    raise SystemExit(f"[abort] P1 {c}: tD unchanged by the perturbation")
            res[c].append({"unit": r["unit"], "dataset": r["dataset"], "device": dev,
                           "psnr8": psnr_u8(p8, y8), "mse8": mse_u8(p8, y8),
                           "mad8": float(np.abs(p8.astype(np.int16) - p_nat.astype(np.int16)).mean()),
                           "mean_tD": float(o["tD"].mean())})
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            print(f"  [P1] {i + 1}/{len(rows)} devices={run.device_counts}", flush=True)
    for c in conds:
        save_csv(out / f"per_image_P1_{c}.csv", res[c])
    return res, run.meta(), {"shuffle_partner": [rows[j]["unit"] for j in perm]}


def run_v2(rows, out):
    hook = FieldHook()
    run = HookedRunner("v2_full", hook)
    conds = ("native", "const05", "perm", "vflip", "extreme")
    res = {c: [] for c in conds}
    for i, r in enumerate(rows):
        x8, y8 = load_u8(r["input"]), load_u8(r["reference"])
        x = u8_to_tensor(x8)
        p_nat, tD_nat = None, None
        for c in conds:
            hook.mode = None if c == "native" else c
            o, dev = run(x)
            p8 = to_u8_tensor(o["J"])
            if c == "native":
                p_nat, tD_nat = p8, o["tD"]
            elif i == 0:
                if torch.equal(o["tD"], tD_nat):
                    raise SystemExit(f"[abort] v2 {c}: hook did not change tD")
                if c == "const05" and not torch.allclose(o["tD"], torch.full_like(o["tD"], 0.5), atol=1e-6):
                    raise SystemExit("[abort] v2 const05: tD != 0.5")
            res[c].append({"unit": r["unit"], "dataset": r["dataset"], "device": dev,
                           "psnr8": psnr_u8(p8, y8), "mse8": mse_u8(p8, y8),
                           "mad8": float(np.abs(p8.astype(np.int16) - p_nat.astype(np.int16)).mean()),
                           "mean_tD": float(o["tD"].mean())})
        hook.mode = None
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            print(f"  [v2] {i + 1}/{len(rows)} devices={run.device_counts} hook_fired={hook.fired}", flush=True)
    for c in conds:
        save_csv(out / f"per_image_v2_{c}.csv", res[c])
    return res, run.meta()


def main():
    commit = git_state()
    tm = Timer()
    torch.set_num_threads(12)
    out = new_outdir("BYPASS517")
    rows = val_rows()
    if len(rows) != 517:
        raise SystemExit("[abort] n != 517")
    store = DepthStore(DEPTH, [(r["dataset"], r["unit"], r["input"], r["reference"]) for r in rows], check_sha=True)
    print(f"[bypass] commit {commit} out {out}", flush=True)

    summary, meta = {}, {}
    p1, meta["P1"], extra = run_p1(rows, store, out)
    v2, meta["v2_full"] = run_v2(rows, out)
    for key, res, name in ((P1KEY, p1, "P1"), ("v2_full", v2, "v2_full")):
        nat = np.array([r["psnr8"] for r in res["native"]])
        dmax = float(np.abs(nat - ref_psnr(key, rows)).max())
        if dmax > 1e-6:
            raise SystemExit(f"[abort] {name} native PSNR8 differs from the earlier evaluation by {dmax:.2e}")
        if len(nat) != 517:
            raise SystemExit(f"[abort] {name} n={len(nat)}")
        summary[name] = {"native_reproduces_earlier_eval_max_abs_d": dmax,
                         "conditions": summarise(rows, res["native"],
                                                 {c: v for c, v in res.items() if c != "native"})}
    write_json(out / "summary.json", {**summary, "bypass_band_db": [-0.10, 0.10],
                                      "labels": "validation (517), not held-out; single seed; exploratory",
                                      "bootstrap": {"B": B_BOOT, "seed": BOOT_SEED, "type": "paired percentile 95%"}})
    write_json(out / "meta.json", {"commit": commit, "models": meta, "env": environment(), **tm.done(), **extra})
    print("\n== paired dPSNR8 (perturbed - native), 95% CI;  mean |dJ| (8-bit levels) ==")
    for name, s in summary.items():
        for c, r in s["conditions"].items():
            d = r["d_psnr8"]
            print(f"{name:8s} {c:8s} " + "  ".join(f"{g} {d[g]['est']:+.3f} [{d[g]['lo']:+.3f},{d[g]['hi']:+.3f}]"
                                               for g in ("LSUI", "UIEB", "ALL"))
                  + f"   |dJ| {r['mean_abs_dJ8']['ALL']['est']:.3f}")
    print(f"\n[bypass] done -> {out}  {tm.done()}", flush=True)


if __name__ == "__main__":
    main()

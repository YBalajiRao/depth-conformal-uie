"""
Phase F: the single test-split evaluation (pre-registered in report section Q, committed before the test split is read).

  python src/diagnostics/phase_f_test.py --split val  --limit 6          # dry run on VALIDATION rows (code check)
  python src/diagnostics/phase_f_test.py --split test --confirm-single-opening

Guards: test requires --confirm-single-opening; a lock file runs/phase_f/TEST_OPENED.json is written before the first
test image is read and its presence aborts any later test run; src/ must be clean (common.git_state).
Fail-closed preprocessing check (before any test image): on 8 validation images the on-the-fly depth and P1e scene
descriptor must reproduce the stored training-time values (depth max |d| <= 2e-3, PCA-16 max |d| <= 5e-2).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "src/data"), str(ROOT / "src/validation"), str(ROOT / "baselines/adapters")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
import common  # noqa: E402
from common import (git_state, write_json, environment, Timer, load_u8, u8_to_tensor, to_u8_tensor, psnr_u8,  # noqa: E402
                    ssim_u8, predictive_std, sha256_file, B_BOOT, BOOT_SEED, Runner)
from s1_eval517 import parity_delta, parity_fails, control_trips  # noqa: E402
from p1_eval517 import DepthRunner, DEPTH  # noqa: E402
from phase2_eval517 import SCENE  # noqa: E402
from train import DepthStore, SceneStore  # noqa: E402
import precompute_depth as PD  # noqa: E402
import precompute_scene_features as PSF  # noqa: E402
from eval517_baseline import BaselineRunner  # noqa: E402
from uw_metrics import uiqm, uciqe  # noqa: E402
from overlap_audit import hashes, nearest, pix_psnr  # noqa: E402
from phase_e_eval import pin_lpips  # noqa: E402

SPLIT_DIR = ROOT / "data/splits/lsui_uieb_all_d6_seed0"
LOCK = ROOT / "runs/phase_f/TEST_OPENED.json"
PCA = SCENE / "pca16_train.npz"
R = "runs/"
MODELS = {   # key: (kind, checkpoint, sha256)   -- frozen list, report section Q
    "P1b_s0": ("dgt", R + "phase1/P1b_v2_dgt_ordered/20261110_093229/last.pt", "5b22ea7c217b09b96ad2b6309b21b5cbca906444e2f442dd9732965ec9db52c0"),
    "P1b_s1": ("dgt", R + "phase2/P1b_s1/20261111_070454/last.pt", "140f0e4f6cf60bbf7e7f186acae6f10fa3923f617654ebefd8bd97db4d888f88"),
    "P1b_s2": ("dgt", R + "phase2/P1b_s2/20261112_082323/last.pt", "684b5a324ab71e1a0d004168af563c03e436f96c56c3401bfd9ed9b489956430"),
    "v2_s0": ("plain", R + "ablation/stage1/S1_v2_full/20261107_193709/last.pt", "8a50ea4dc3dd43c2dd23af343df531df4ca5185ff30ad0955dddb5dad60b2eaa"),
    "v2_s1": ("plain", R + "phase2/v2_s1/20261111_161156/last.pt", "9030037adcb124706f1577bb8c194f5ad3358ff2373e6c2c99858a0604875dc0"),
    "v2_s2": ("plain", R + "phase2/v2_s2/20261112_174530/last.pt", "0549d7445b07629e412cd0dbbcc95ac5ca1eadba1deb98e7351e3eba72d23816"),
    "P1e_s0": ("scene", R + "phase1/P1e_scene_pca16/20261117_061952/last.pt", "f0ac978c371abcb870b5fbb88c31b47897ca33c95ceb395b03a3417e9a6fc3d4"),
    "v1_retrain": ("plain", R + "ablation/stage1/S1_v1_retrain/20261107_102639/last.pt", "42e1b674cb864362275f3f04df061f6c07b815252df738db63993da7b986a624"),
    "A2_no_physics": ("plain", R + "ablation/stage1/S1_A2_no_physics/20261108_084156/last.pt", "62f7604c78145a959e67532049b71502dd7bf71e595c72e67b78b2752179ac99"),
    "A3_v2_no_Lphys": ("plain", R + "ablation/stage1/S1_A3_v2_no_Lphys/20261108_184308/last.pt", "91a4fdeaa3ea973b3d06716542e7886582da7d2466d9ba001650a4c505d334fd"),
    "A4_v2_no_NIG": ("plain", R + "ablation/stage1/S1_A4_v2_no_NIG/20261109_074648/last.pt", "f70fb83c123eb56b867f50d099c2bb7362a3e5e6d87c91a572e24481507b02c4"),
    "fiveaplus": ("base", R + "baselines/fiveaplus/20261113_170747/last.pt", "61cf5174bbdee938d35fecbf9a5be72457f8a7c55784643689f961f558cb96a1"),
    "ushape": ("base", R + "baselines/ushape/20261116_094852/last.pt", "8057943ef123268fda1cdeef3858461561daf2eeeb7851f1b5af1cca4a771ac7"),
    "wweuie": ("base", R + "baselines/wweuie/20261116_222308/last.pt", "c0e9261627943589e9bad0353ac0e7cbf0af664520788b21d188c66ed96856c4"),
}
GRID_MODELS = ["P1b_s0", "v2_s0", "P1e_s0", "wweuie", "ushape", "fiveaplus"]


def read_rows(name, allow_test):
    path = SPLIT_DIR / f"{name}_manifest.txt"
    if "test" in name and not allow_test:
        raise SystemExit("[abort] test manifest requested without the single-opening guard")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        ds, uid, _cid, inp, ref = line.split("\t")
        rows.append({"dataset": ds, "unit": uid, "input": inp, "reference": ref})
    return rows, sha256_file(path)


class Preproc:
    """Depth (precompute_depth) and PCA-16 scene descriptor (precompute_scene_features + training-only PCA)."""

    def __init__(self, dev):
        self.dev = dev
        self.proc, self.dm, self.dm16, self.meta = PD.load_depth_model(dev, fp16=True)
        with np.load(PCA, allow_pickle=False) as d:
            self.mean, self.comps, self.scale = (torch.from_numpy(d[k].astype(np.float32)) for k in ("mean", "components", "scale"))

    def depth(self, x8):
        d, _ = PD.disparity(self.proc, self.dm, self.dm16, x8, self.dev)
        z, _, degen = PD.normalise(d)
        return torch.from_numpy(z.astype(np.float16).astype(np.float32))[None], degen

    @torch.inference_mode()
    def scene(self, x8):
        g, _, _ = PSF.descriptor(self.proc, self.dm16, x8, self.dev)
        return ((torch.from_numpy(g) - self.mean) @ self.comps.T) / self.scale


def preproc_check(pp, n=8):
    rows, _ = read_rows("val", False)
    rows = rows[:n]
    r4 = [(r["dataset"], r["unit"], r["input"], r["reference"]) for r in rows]
    ds, ss = DepthStore(DEPTH, r4, check_sha=True), SceneStore(SCENE, r4, check_sha=True, pca_file=str(PCA))
    dz, dg = 0.0, 0.0
    for r in rows:
        x8 = load_u8(r["input"])
        dz = max(dz, float((pp.depth(x8)[0] - ds.load(r["unit"], x8.shape[:2])).abs().max()))
        dg = max(dg, float((pp.scene(x8) - ss.load(r["unit"])).abs().max()))
    ok = dz <= 2e-3 and dg <= 5e-2
    print(f"[preproc check] depth max|d| {dz:.2e}  pca16 max|d| {dg:.2e}  -> {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise SystemExit("[abort] on-the-fly preprocessing does not reproduce training-time values")
    return {"n": n, "depth_max_abs": dz, "pca16_max_abs": dg}


class SceneRunner(DepthRunner):
    @torch.no_grad()
    def __call__(self, x, z, g, force_device=None):
        dev = force_device or self.device.type
        if dev == "cuda":
            try:
                o = self.wrapped(x.to(self.device), z.to(self.device), g.to(self.device))
                o = {k: v.float().cpu() if torch.is_tensor(v) else v for k, v in o.items()}
                self.device_counts["cuda"] = self.device_counts.get("cuda", 0) + 1
                return o, "cuda"
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
        o = self._cpu_wrapped()(x.cpu(), z.cpu(), g)
        o = {k: v.float() if torch.is_tensor(v) else v for k, v in o.items()}
        self.device_counts["cpu"] = self.device_counts.get("cpu", 0) + 1
        return o, "cpu"


def build_runners():
    run = {}
    for k, (kind, ck, sh) in MODELS.items():
        if kind == "base":
            run[k] = BaselineRunner(k, str(ROOT / ck), sh)
            continue
        common.CKPTS[k] = (ROOT / ck, sh, 100000)
        run[k] = {"dgt": DepthRunner, "scene": SceneRunner, "plain": Runner}[kind](k)
    return run


def forward(k, run, x8, z, g, force_device=None):
    """-> (J tensor (1,3,H,W), device, NIG outputs or None)."""
    kind = MODELS[k][0]
    x = u8_to_tensor(x8)
    if kind == "base":
        J, dv = run(x, force_device=force_device); return J, dv, None
    if kind == "plain":
        o, dv = run(x, force_device=force_device)
    elif kind == "dgt":
        o, dv = run(x, z[None], force_device=force_device)
    else:
        o, dv = run(x, z[None], g[None], force_device=force_device)
    return o["J"], dv, o


def dedup_flags(test_rows):
    tr, _ = read_rows("train", False)
    tr_paths = [r["input"] for r in tr]
    H_tr = hashes(tr_paths)
    lsui = [i for i, r in enumerate(test_rows) if r["dataset"] == "LSUI"]
    d, j = nearest(hashes([test_rows[i]["input"] for i in lsui]), H_tr)
    flags = {}
    for n, i in enumerate(lsui):
        dup = bool(d[n] <= 8 and pix_psnr(test_rows[i]["input"], tr_paths[j[n]]) >= 25)
        flags[test_rows[i]["unit"]] = {"near_dup": dup, "hamming": int(d[n]), "train_neighbour": tr[j[n]]["unit"]}
    return flags


def boot_groups(vals, groups, B=B_BOOT, seed=BOOT_SEED):
    """vals: per-image values; groups: list of group labels. Stratified bootstrap of the n-weighted mean."""
    v, gl = np.asarray(vals, dtype=np.float64), np.asarray(groups)
    rng = np.random.default_rng(seed)
    parts = [v[gl == u] for u in sorted(set(groups))]
    bs = np.zeros(B)
    for p in parts:
        bs += p[rng.integers(0, len(p), size=(B, len(p)))].sum(1)
    bs /= len(v)
    sd = v.std(ddof=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"est": float(v.mean()), "lo": float(lo), "hi": float(hi), "d_z": float(v.mean() / sd) if sd > 0 else float("nan"),
            "n": int(len(v)), "ci_excludes_0": bool(lo > 0 or hi < 0)}


def group_masks(rows, flags):
    ds = np.array([r["dataset"] for r in rows])
    dup = np.array([flags.get(r["unit"], {}).get("near_dup", False) for r in rows])
    return {"LSUI": ds == "LSUI", "UIEB": ds == "UIEB", "ALL": np.ones(len(rows), bool),
            "LSUI_dedup": (ds == "LSUI") & ~dup, "ALL_dedup": ~dup}, ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--confirm-single-opening", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--vram-frac", type=float, default=0.50)
    a = ap.parse_args()
    if a.split == "test":
        if not a.confirm_single_opening or a.limit:
            raise SystemExit("[abort] the test split needs --confirm-single-opening and no --limit")
        if LOCK.exists():
            raise SystemExit(f"[abort] test split already opened: {LOCK.read_text(encoding='utf-8')}")
    commit = git_state()
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(a.vram_frac, 0)
    tm = Timer()
    out = ROOT / "runs/phase_f" / f"PHASE_F_{a.split.upper()}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)
    pp = Preproc(dev)
    pre = preproc_check(pp)
    lp = pin_lpips(dev)
    run = build_runners()
    if a.split == "test":
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        LOCK.write_text(json.dumps({"opened": datetime.now().isoformat(timespec="seconds"), "commit": commit,
                                    "out": str(out)}, indent=1), encoding="utf-8")
    rows, msha = read_rows(a.split, a.split == "test")
    if a.limit:
        rows = rows[:a.limit]
    print(f"[phase_f] split={a.split} n={len(rows)} manifest sha {msha[:12]} commit {commit} out {out}", flush=True)
    flags = dedup_flags(rows)
    write_json(out / "near_duplicates.json", flags)
    print(f"[dedup] LSUI near-duplicates of training: {sum(f['near_dup'] for f in flags.values())} / {len(flags)}", flush=True)
    rng = np.random.default_rng(20261117)
    lsui_i = [i for i, r in enumerate(rows) if r["dataset"] == "LSUI"]
    uieb_i = [i for i, r in enumerate(rows) if r["dataset"] == "UIEB"]
    grid = sorted([lsui_i[i] for i in rng.choice(len(lsui_i), min(4, len(lsui_i)), replace=False)] +
                  [uieb_i[i] for i in rng.choice(len(uieb_i), min(2, len(uieb_i)), replace=False)])
    write_json(out / "grid_units.json", [rows[i]["unit"] for i in grid])

    # parity (shared +/-8 checkerboard control) on 4 images per model
    par = {}
    order = np.random.default_rng(BOOT_SEED).permutation(len(rows))[:4]
    for k, r_ in run.items():
        ds_, ctl = [], False
        for i in order:
            x8, y8 = load_u8(rows[i]["input"]), load_u8(rows[i]["reference"])
            z, _ = pp.depth(x8); g = pp.scene(x8)
            Jc, dc, _ = forward(k, r_, x8, z, g)
            if dc != "cuda":
                continue
            Jp, _, _ = forward(k, r_, x8, z, g, force_device="cpu")
            pc, pq = to_u8_tensor(Jc), to_u8_tensor(Jp)
            ds_.append(parity_delta(pc, pq, y8))
            if not ctl:
                if not control_trips(pc, pq, y8):
                    raise SystemExit(f"[abort] {k}: parity failure control did not fail")
                ctl = True
        if len(ds_) < 2 or parity_fails(ds_):
            raise SystemExit(f"[HALT] {k}: parity failed or too few CUDA images {ds_}")
        par[k] = {"n": len(ds_), "max_abs_d_psnr8": max(ds_), "control_tripped": ctl}
    print("[parity]", {k: round(v["max_abs_d_psnr8"], 5) for k, v in par.items()}, flush=True)

    t = lambda u: (torch.from_numpy(u).permute(2, 0, 1)[None].float().div(127.5) - 1.0).to(dev)  # noqa: E731
    recs = {k: [] for k in run}
    with torch.no_grad():
        for n, r in enumerate(rows):
            x8, y8 = load_u8(r["input"]), load_u8(r["reference"])
            z, degen = pp.depth(x8); g = pp.scene(x8)
            gdir = None
            if n in grid:
                gdir = out / "grid" / r["unit"].replace(":", "_").replace("/", "_"); gdir.mkdir(parents=True)
                Image.fromarray(x8).save(gdir / "input.png"); Image.fromarray(y8).save(gdir / "reference.png")
            for k, r_ in run.items():
                J, dv, o = forward(k, r_, x8, z, g)
                p8 = to_u8_tensor(J)
                rec = {"unit": r["unit"], "dataset": r["dataset"], "device": dv, "psnr8": psnr_u8(p8, y8),
                       "ssim8": ssim_u8(p8, y8), "lpips": float(lp(t(p8), t(y8))), "uiqm": uiqm(p8)["uiqm"],
                       "uciqe": uciqe(p8)["uciqe"]}
                if o is not None and "nu" in o:
                    rec["sigma_total"] = float(predictive_std(o["nu"], o["alpha"], o["beta"], "total").mean())
                recs[k].append(rec)
                if gdir is not None and k in GRID_MODELS:
                    Image.fromarray(p8).save(gdir / f"{k}.png")
                    if k == "P1b_s0":
                        np.save(gdir / "P1b_s0_sigma.npy", predictive_std(o["nu"], o["alpha"], o["beta"], "total")[0].mean(0).numpy())
            if (n + 1) % 25 == 0:
                print(f"[{a.split}] {n + 1}/{len(rows)}  P1b_s0 mean PSNR8 {np.mean([q['psnr8'] for q in recs['P1b_s0']]):.3f}", flush=True)

    for k, rs in recs.items():
        with open(out / f"per_image_{k}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0])); w.writeheader(); w.writerows(rs)
    masks, ds = group_masks(rows, flags)
    absolute, paired = {}, {}
    for k, rs in recs.items():
        absolute[k] = {}
        for gname, m in masks.items():
            if m.sum() == 0:
                continue
            absolute[k][gname] = {mt: float(np.mean([rs[i][mt] for i in np.where(m)[0]]))
                                  for mt in ("psnr8", "ssim8", "lpips", "uiqm", "uciqe")}
        if k != "P1b_s0":
            paired[k] = {}
            for gname, m in masks.items():
                idx = np.where(m)[0]
                if len(idx) < 2:
                    continue
                paired[k][gname] = {mt: boot_groups([rs[i][mt] - recs["P1b_s0"][i][mt] for i in idx], list(ds[idx]))
                                    for mt in ("psnr8", "ssim8", "lpips")}
    seedpairs = {}
    for s in (0, 1, 2):
        a_, b_ = recs[f"P1b_s{s}"], recs[f"v2_s{s}"]
        seedpairs[f"s{s}"] = {gname: boot_groups([a_[i]["psnr8"] - b_[i]["psnr8"] for i in np.where(m)[0]], list(ds[np.where(m)[0]]))
                              for gname, m in masks.items() if m.sum() > 1}
    summary = {"split": a.split, "n": len(rows), "manifest_sha256": msha, "commit": commit, "preregistration": "report section Q",
               "models": {k: {"kind": v[0], "ckpt": v[1], "sha256": v[2]} for k, v in MODELS.items()},
               "preproc_check": pre, "parity": par, "near_duplicates_LSUI": int(sum(f["near_dup"] for f in flags.values())),
               "absolute": absolute, "paired_model_minus_P1b_s0": paired, "P1b_minus_v2_seed_matched_psnr8": seedpairs,
               "device_counts": {k: dict(getattr(r_, "device_counts", getattr(r_, "counts", {}))) for k, r_ in run.items()},
               "grid_units": [rows[i]["unit"] for i in grid], "env": environment(), "runtime": tm.done()}
    write_json(out / "summary.json", summary)
    print(f"[phase_f] done -> {out}", flush=True)


if __name__ == "__main__":
    main()

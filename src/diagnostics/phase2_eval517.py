"""
Phase 2 (multi-seed replication): 517-image evaluation of one checkpoint, and within-seed pairing.

  python phase2_eval517.py eval --arm P1b_s1 --ckpt <last.pt> --sha <sha256>
      Same protocol as s1_eval517 / p1b_eval517 (fp32, full resolution, CUDA with CPU-on-OOM fallback, 10-image
      CUDA/CPU parity gate with planted-fault control, n == 517). A DGT checkpoint gets its depth map as an
      aligned aux input; a non-DGT checkpoint (v2) is run exactly as in s1_eval517. Writes per-image records.
  python phase2_eval517.py pair --a <per_image.csv>:<sha> --b <per_image.csv>:<sha> --name "P1b_s1 - v2_s1"
      Paired bootstrap (B = 2000, seed 20261107) of a - b on PSNR8 / SSIM8 per dataset + stratified ALL, d_z, and
      the pre-registered section-L test (>= 0.2 dB and CI excluding 0) per dataset.
EXPLORATORY per seed; the pre-registered decision needs all seeds (report section L).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import common  # noqa: E402
from common import (Runner, git_state, new_outdir, write_json, environment, Timer, val_rows, boot_groups,  # noqa: E402
                    by_dataset, mean_stat, pooled_psnr_stat, fmt_ci, sha256_file, B_BOOT, BOOT_SEED)
from s1_eval517 import (parity_gate as parity_gate_plain, parity_fails, dz, save_csv, PARITY_TOL_DB,  # noqa: E402
                        N_EXPECTED, evaluate as evaluate_plain)
from p1_eval517 import DepthRunner, parity_gate as parity_gate_depth, DEPTH  # noqa: E402
import p1b_eval517  # noqa: E402
from train import DepthStore, SceneStore  # noqa: E402


class SceneDepthRunner(DepthRunner):
    """DepthRunner that also passes the image's global scene descriptor g (scene_prior models). The unit id is set
    by _SceneProxyStore.load(), which every evaluation path calls for an image before running it."""

    def __init__(self, key, scene):
        super().__init__(key)
        self.scene, self.uid = scene, None

    @torch.no_grad()
    def __call__(self, x, z, force_device=None):
        g = self.scene.load(self.uid)[None]
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


class _SceneProxyStore:
    def __init__(self, depth, runner):
        self.depth, self.runner = depth, runner

    def load(self, uid, hw):
        self.runner.uid = uid
        return self.depth.load(uid, hw)

MARGIN_DB = 0.2
SCENE = Path(__file__).resolve().parents[2] / "data/scene/da2_small_5426e4f/lsui_uieb_all_d6_seed0"


def do_eval(a):
    commit = git_state()
    tm = Timer()
    torch.set_num_threads(12)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    depth = bool(ck["cfg"]["model_cfg"].get("depth_transmission", False))
    scene_prior = bool(ck["cfg"]["model_cfg"].get("scene_prior", False))
    scene_pca = ck["cfg"]["model_cfg"].get("scene_pca_file", "")
    common.CKPTS[a.arm] = (Path(a.ckpt), a.sha, 100000)        # SHA-256 + stored step verified by Runner
    del ck
    out = new_outdir(f"P2_EVAL517_{a.arm}")
    rows = val_rows()
    counts = {d: sum(r["dataset"] == d for r in rows) for d in N_EXPECTED}
    if counts != N_EXPECTED:
        raise SystemExit(f"[abort] counts {counts}")
    print(f"[phase2_eval517] {a.arm} depth={depth} commit {commit} out {out}", flush=True)
    if depth:
        rows4 = [(r["dataset"], r["unit"], r["input"], r["reference"]) for r in rows]
        store = DepthStore(DEPTH, rows4, check_sha=True)
        if scene_prior:
            pf = str(Path(__file__).resolve().parents[2] / scene_pca) if scene_pca else None
            run = SceneDepthRunner(a.arm, SceneStore(SCENE, rows4, check_sha=True, pca_file=pf))
            store = _SceneProxyStore(store, run)
        else:
            run = DepthRunner(a.arm)
        par, ctl = parity_gate_depth(run, rows, store)
    else:
        run = Runner(a.arm)
        par, ctl = parity_gate_plain(run, rows)
    mx = max(r["abs_d_psnr8"] for r in par)
    write_json(out / "parity.json", {"tolerance_db": PARITY_TOL_DB, "max_abs_d_psnr8": mx,
                                     "failure_control_tripped": ctl, "images": par})
    print(f"[parity] max |dPSNR8| = {mx:.6f} dB over {len(par)} images", flush=True)
    if parity_fails([r["abs_d_psnr8"] for r in par]):
        raise SystemExit("[HALT] parity gate failed")
    if depth:
        p1b_eval517.KEY = a.arm
        recs = p1b_eval517.evaluate(run, rows, store)
    else:
        recs = evaluate_plain(run, rows)
    if len(recs) != 517 or [r["unit"] for r in recs] != [r["unit"] for r in rows]:
        raise SystemExit(f"[abort] n={len(recs)} or order mismatch")
    meta_run = run.meta()
    path = out / f"per_image_{a.arm}.csv"
    save_csv(path, recs)
    absolute = {"psnr8_mean": boot_groups(by_dataset(recs, "psnr8"), mean_stat),
                "psnr8_pooled": boot_groups(by_dataset(recs, "mse8"), pooled_psnr_stat),
                "ssim8_mean": boot_groups(by_dataset(recs, "ssim8"), mean_stat),
                "device_counts": meta_run["device_counts"]}
    write_json(out / "summary.json", {"arm": a.arm, "absolute": absolute, "per_image_csv": str(path),
                                      "per_image_sha256": sha256_file(path),
                                      "labels": "validation (517), not held-out; exploratory"})
    write_json(out / "meta.json", {"commit": commit, "model": meta_run, "env": environment(), **tm.done(),
                                   "depth": depth, "images": [r["unit"] for r in rows]})
    for g in ("LSUI", "UIEB", "ALL"):
        print(f"{g:4s} PSNR {fmt_ci(absolute['psnr8_mean'][g])}  SSIM {fmt_ci(absolute['ssim8_mean'][g], 4)}")
    print(f"devices {absolute['device_counts']}\nper-image {path}\nsha256 {sha256_file(path)}\n"
          f"[phase2_eval517] done {tm.done()}", flush=True)


def _load(spec, rows):
    head, _, tail = spec.rpartition(":")
    is_sha = len(tail) == 64 and all(ch in "0123456789abcdef" for ch in tail.lower())
    p, sha = (head, tail.lower()) if (head and is_sha) else (spec, None)
    p = Path(p)
    got = sha256_file(p)
    if sha and got != sha:
        raise SystemExit(f"[abort] {p.name} SHA-256 {got} != {sha}")
    with open(p, newline="", encoding="utf-8") as f:
        recs = [{k: (float(v) if k in ("psnr8", "ssim8", "mse8") else v) for k, v in r.items()} for r in csv.DictReader(f)]
    if len(recs) != 517 or [r["unit"] for r in recs] != [r["unit"] for r in rows]:
        raise SystemExit(f"[abort] {p.name}: image order/count differs")
    return recs, str(p), got


def do_pair(a):
    rows = val_rows()
    A, pa, sa = _load(a.a, rows)
    B, pb, sb = _load(a.b, rows)
    tmp = [{"dataset": x["dataset"], "d_psnr8": x["psnr8"] - y["psnr8"], "d_ssim8": x["ssim8"] - y["ssim8"]}
           for x, y in zip(A, B)]
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
    crit = {grp: {"meets_margin_and_ci": bool(abs(c["est"]) >= MARGIN_DB and c["ci_excludes_0"]),
                  "direction": "a>b" if c["est"] > 0 else "a<b"} for grp, c in res["d_psnr8"].items()}
    out = new_outdir("P2_PAIR")
    write_json(out / "paired.json", {"name": a.name, "a": {"csv": pa, "sha256": sa}, "b": {"csv": pb, "sha256": sb},
                                     "paired": res, "section_L_per_seed_test": crit, "margin_db": MARGIN_DB,
                                     "bootstrap": {"B": B_BOOT, "seed": BOOT_SEED}})
    for m in ("d_psnr8", "d_ssim8"):
        nd = 3 if m == "d_psnr8" else 4
        print(f"{a.name:22s} {m}  " + "  ".join(
            f"{g} {c['est']:+.{nd}f} [{c['lo']:+.{nd}f},{c['hi']:+.{nd}f}] dz={c['d_z']:+.2f}" for g, c in res[m].items()))
    print("section-L per-seed test:", {g: v["meets_margin_and_ci"] for g, v in crit.items()}, f"\n-> {out}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("eval"); e.add_argument("--arm", required=True); e.add_argument("--ckpt", required=True)
    e.add_argument("--sha", required=True)
    e.add_argument("--vram-frac", type=float, default=0.0,
                   help="optional CUDA memory cap (0 = none, the original behaviour); OOM images fall back to CPU")
    p = sp.add_parser("pair"); p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("--name", required=True)
    a = ap.parse_args()
    if a.cmd == "eval" and a.vram_frac > 0 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(a.vram_frac, 0)
        print(f"[vram] CUDA allocator capped at {a.vram_frac:.2f}", flush=True)
    do_eval(a) if a.cmd == "eval" else do_pair(a)


if __name__ == "__main__":
    main()

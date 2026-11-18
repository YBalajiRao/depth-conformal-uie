"""
F3 data collection (pre-registered in report section S): per-image pixel samples of residual |y - mu| and NIG total
std sigma for P1b_s0 and the 3-seed ensemble E3, plus per-image PSNR8/SSIM8 of both.

  python src/diagnostics/f3_conformal.py --split val
  python src/diagnostics/f3_conformal.py --split dark
  python src/diagnostics/f3_conformal.py --split test --confirm-f3      # once; lock runs/phase_f/F3_TEST_OPENED.json
Output: runs/f3/<split>_<ts>/{samples.npz, per_image.csv, meta.json}
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE)]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import common  # noqa: E402
from common import git_state, load_u8, u8_to_tensor, to_u8_tensor, psnr_u8, ssim_u8, predictive_std, Timer  # noqa: E402
from p1_eval517 import DepthRunner  # noqa: E402
import phase_f_test as PF  # noqa: E402
import phase_e_eval as PE  # noqa: E402

LOCK = ROOT / "runs/phase_f/F3_TEST_OPENED.json"
SEEDS = ["P1b_s0", "P1b_s1", "P1b_s2"]
K = 4096


def rows_for(split):
    if split in ("val", "test"):
        rows, sha = PF.read_rows(split, split == "test")
        return [(r["unit"], r["input"], r["reference"]) for r in rows], sha
    return [(p.name, str(p), str(q)) for p, q in PE.dark_pairs()], "euvp_underwater_dark_trainA_trainB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test", "dark"], required=True)
    ap.add_argument("--confirm-f3", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.split == "test":
        if not a.confirm_f3 or a.limit:
            raise SystemExit("[abort] F3 test read needs --confirm-f3 and no --limit")
        if LOCK.exists():
            raise SystemExit(f"[abort] F3 already read the test split: {LOCK.read_text(encoding='utf-8')}")
    commit = git_state(); dev = torch.device("cuda"); tm = Timer()
    torch.cuda.set_per_process_memory_fraction(0.50, 0)
    out = ROOT / "runs/f3" / f"{a.split}_{datetime.now():%Y%m%d_%H%M%S}"; out.mkdir(parents=True)
    pp = PF.Preproc(dev)
    run = {}
    for k in SEEDS:
        _, ck, sh = PF.MODELS[k]; common.CKPTS[k] = (ROOT / ck, sh, 100000); run[k] = DepthRunner(k)
    if a.split == "test":
        LOCK.write_text(json.dumps({"opened": datetime.now().isoformat(timespec="seconds"), "commit": commit, "out": str(out)},
                                   indent=1), encoding="utf-8")
    rows, sha = rows_for(a.split)
    rows = rows[: a.limit or None]
    S = {"r1": [], "s1": [], "rE": [], "sE": []}
    per = []
    with torch.no_grad():
        for n, (unit, ip, rp) in enumerate(rows):
            x8, y8 = load_u8(ip), load_u8(rp)
            z, _ = pp.depth(x8)
            mus, var = [], []
            for k in SEEDS:
                o, _ = run[k](u8_to_tensor(x8), z[None])
                mus.append(o["J"][0].clamp(0, 1).numpy())
                var.append(predictive_std(o["nu"], o["alpha"], o["beta"], "total")[0].numpy() ** 2)
            mu1, s1 = mus[0], np.sqrt(var[0])
            muE = np.mean(mus, 0); sE = np.sqrt(np.mean(var, 0) + np.var(mus, 0))
            y = y8.transpose(2, 0, 1).astype(np.float32) / 255.0
            H, W = y.shape[1:]
            kk = 1024 if a.split == "dark" else K                 # dark: 5550 images -> 1024 px each (memory)
            idx = np.random.default_rng(zlib.crc32(unit.encode())).choice(H * W, size=min(kk, H * W), replace=False)
            for key, m, s in (("1", mu1, s1), ("E", muE, sE)):
                S["r" + key].append(np.abs(y - m).reshape(3, -1)[:, idx].astype(np.float32))
                S["s" + key].append(s.reshape(3, -1)[:, idx].astype(np.float32))
            p1 = to_u8_tensor(torch.from_numpy(mu1)[None]); pE = to_u8_tensor(torch.from_numpy(muE)[None])
            per.append({"unit": unit, "psnr8_s0": psnr_u8(p1, y8), "ssim8_s0": ssim_u8(p1, y8),
                        "psnr8_E3": psnr_u8(pE, y8), "ssim8_E3": ssim_u8(pE, y8)})
            if (n + 1) % 250 == 0:
                print(f"[{a.split}] {n + 1}/{len(rows)}", flush=True)
    np.savez_compressed(out / "samples.npz", **{k: np.stack(v) for k, v in S.items()},
                        units=np.array([r[0] for r in rows]))
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0])); w.writeheader(); w.writerows(per)
    (out / "meta.json").write_text(json.dumps({"split": a.split, "n": len(rows), "source_sha": sha, "commit": commit,
                                               "K": K, "runtime": tm.done(),
                                               "device_counts": {k: dict(r.device_counts) for k, r in run.items()}}, indent=1),
                                   encoding="utf-8")
    print("[f3] done", out, flush=True)


if __name__ == "__main__":
    main()

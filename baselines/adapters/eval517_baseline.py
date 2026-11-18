"""
517-image native-resolution evaluation of a trained baseline (roadmap phase D). Validation only; test refused.

Same protocol as the SS-PGN evaluations: fp32, no autocast, deterministic cuDNN, full native resolution through
registry.predict (pad to the architecture's multiple; U-shape: bilinear 256 round trip), CUDA with full-image CPU fp32
fallback on OOM (device recorded per image), 10-image CUDA/CPU parity gate (|dPSNR8| < 0.01 dB) with a planted-fault
control, n == 517 asserted, PSNR8/SSIM8/MSE8 from the SS-PGN metric code. Writes per_image_<name>.csv + summary.json.
Pairing against SS-PGN models uses src/diagnostics/phase2_eval517.py pair (validated, K13 reproduction).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src/diagnostics"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import registry as R  # noqa: E402
import data as D  # noqa: E402
from common import (new_outdir, write_json, environment, Timer, val_rows, load_u8, u8_to_tensor, to_u8_tensor,  # noqa: E402
                    mse_u8, psnr_u8, ssim_u8, boot_groups, by_dataset, mean_stat, pooled_psnr_stat, fmt_ci, sha256_file,
                    git_state)
from s1_eval517 import save_csv, PARITY_TOL_DB, control_trips  # noqa: E402


class BaselineRunner:
    def __init__(self, name, ckpt, sha):
        got = sha256_file(Path(ckpt))
        if got != sha:
            raise SystemExit(f"[abort] {ckpt} SHA-256 {got} != {sha}")
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        if ck["step"] != 100000:
            raise SystemExit(f"[abort] stored step {ck['step']} != 100000")
        self.name, self.state = name, ck["model"]
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        self.gpu = R.build(name).cuda().eval(); self.gpu.load_state_dict(self.state, strict=True)
        self.cpu, self.counts = None, {}

    @torch.no_grad()
    def __call__(self, x, force_device=None):
        if (force_device or "cuda") == "cuda":
            try:
                J = R.predict(self.name, self.gpu, x.cuda()).float().cpu()
                self.counts["cuda"] = self.counts.get("cuda", 0) + 1
                return J, "cuda"
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
        if self.cpu is None:
            self.cpu = R.build(self.name).eval(); self.cpu.load_state_dict(self.state, strict=True)
        J = R.predict(self.name, self.cpu, x.cpu()).float()
        self.counts["cpu"] = self.counts.get("cpu", 0) + 1
        return J, "cpu"


def parity(run, rows):
    rng = np.random.default_rng(20261107)
    recs, ctl = [], False
    for ds in ("LSUI", "UIEB"):
        cand = [r for r in rows if r["dataset"] == ds]
        got = 0
        for j in rng.permutation(len(cand)):
            if got == 5:
                break
            r = cand[j]
            x8, y8 = load_u8(r["input"]), load_u8(r["reference"])
            x = u8_to_tensor(x8)
            Jc, dc = run(x)
            if dc != "cuda":
                continue
            Jp, _ = run(x, force_device="cpu")
            pc, pp = to_u8_tensor(Jc), to_u8_tensor(Jp)
            d = abs(psnr_u8(pc, y8) - psnr_u8(pp, y8))
            recs.append({"unit": r["unit"], "dataset": ds, "abs_d_psnr8": d})
            if not ctl:
                if not control_trips(pc, pp, y8):     # +/-8 checkerboard (s1_eval517, revised 2026-11-17)
                    raise SystemExit("[abort] parity failure control did not fail")
                ctl = True
            got += 1
        if got < 5:
            raise SystemExit(f"[abort] only {got} {ds} images fit on CUDA for parity")
    return recs, ctl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=sorted(R.BASELINES))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sha", required=True)
    a = ap.parse_args()
    commit = git_state()
    tm = Timer()
    torch.set_num_threads(12)
    out = new_outdir(f"BASELINE_EVAL517_{a.name}")
    rows = val_rows()
    if len(rows) != 517:
        raise SystemExit("[abort] n != 517")
    run = BaselineRunner(a.name, a.ckpt, a.sha)
    par, ctl = parity(run, rows)
    mx = max(r["abs_d_psnr8"] for r in par)
    write_json(out / "parity.json", {"tolerance_db": PARITY_TOL_DB, "max_abs_d_psnr8": mx, "control_tripped": ctl,
                                     "images": par})
    print(f"[parity] max |dPSNR8| = {mx:.6f} dB", flush=True)
    if mx >= PARITY_TOL_DB:
        raise SystemExit("[HALT] parity gate failed")
    recs = []
    for i, r in enumerate(rows):
        x8, y8 = load_u8(r["input"]), load_u8(r["reference"])
        J, dev = run(u8_to_tensor(x8))
        p8 = to_u8_tensor(J)
        if p8.shape != y8.shape:
            raise SystemExit(f"[abort] {r['unit']}: output {p8.shape} != reference {y8.shape}")
        recs.append({"unit": r["unit"], "dataset": r["dataset"], "H": y8.shape[0], "W": y8.shape[1], "device": dev,
                     "precision": "fp32", "mse8": mse_u8(p8, y8), "psnr8": psnr_u8(p8, y8), "ssim8": ssim_u8(p8, y8)})
        if (i + 1) % 100 == 0 or i + 1 == len(rows):
            print(f"  [{a.name}] {i + 1}/517 devices={run.counts}", flush=True)
    if len(recs) != 517:
        raise SystemExit("[abort] n != 517")
    path = out / f"per_image_{a.name}.csv"
    save_csv(path, recs)
    absolute = {"psnr8_mean": boot_groups(by_dataset(recs, "psnr8"), mean_stat),
                "psnr8_pooled": boot_groups(by_dataset(recs, "mse8"), pooled_psnr_stat),
                "ssim8_mean": boot_groups(by_dataset(recs, "ssim8"), mean_stat), "device_counts": run.counts}
    write_json(out / "summary.json", {"name": a.name, "ckpt": a.ckpt, "ckpt_sha256": a.sha, "absolute": absolute,
                                      "per_image_csv": str(path), "per_image_sha256": sha256_file(path),
                                      "spec": {**R.BASELINES[a.name].__dict__, "kwargs": dict(R.BASELINES[a.name].kwargs)},
                                      "labels": "validation (517), not held-out; unified recipe; exploratory"})
    write_json(out / "meta.json", {"commit": commit, "env": environment(), **tm.done()})
    for g in ("LSUI", "UIEB", "ALL"):
        print(f"{g:4s} PSNR {fmt_ci(absolute['psnr8_mean'][g])}  SSIM {fmt_ci(absolute['ssim8_mean'][g], 4)}")
    print(f"per-image {path}\nsha256 {sha256_file(path)}\n[eval517_baseline] done {tm.done()}", flush=True)


if __name__ == "__main__":
    main()

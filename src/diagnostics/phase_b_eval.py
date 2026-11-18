"""
Roadmap phase B (evaluation only; no training; validation split only; test refused).

B1  x8 self-ensemble for P1b seed 0 (last.pt): for each of the 8 dihedral transforms (rot90 k = 0..3 x horizontal
    flip), transform the input AND its depth map, run full native resolution (PadCropWrapper, fp32), invert the
    transform on the output, average the 8 outputs in float, then quantise once (same rounding as all evaluations).
    The k = 0 / no-flip member is the single-pass prediction; it must reproduce the pre-registered 517-image record
    exactly (abort otherwise). Paired delta (x8 - single) per dataset, B = 2000, seed 20261107.
B3  No-reference UIQM / UCIQE (src/validation/uw_metrics.py; UIQM variant 'funie') on: P1b seed 0 single-pass and
    x8 predictions, v2 seed 0 single-pass predictions, the degraded inputs and the references (context).
B2  256x256 resized protocol (literature-style CONTEXT numbers, never mixed with the native table): input and
    reference resized to 256x256 with cv2.INTER_LINEAR (as the U-shape Transformer test code), depth map resized
    bilinearly; PSNR8 / SSIM8 on 256x256 uint8. Models: v1_retrain, v2 seeds 0-2, P1b seeds 0-2 (pinned last.pt).
EXPLORATORY numbers; validation reused for decisions; not held-out.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import common  # noqa: E402
from common import (Runner, git_state, new_outdir, write_json, environment, Timer, val_rows, load_u8,  # noqa: E402
                    u8_to_tensor, to_u8_tensor, psnr_u8, ssim_u8, boot_groups, by_dataset, mean_stat, fmt_ci,
                    sha256_file, ROOT)
from s1_eval517 import save_csv  # noqa: E402
from p1_eval517 import DepthRunner, DEPTH  # noqa: E402
from train import DepthStore  # noqa: E402
from uw_metrics import uiqm, uciqe  # noqa: E402

CK = {  # key: (path, sha256, depth?)
    "P1b_s0": ("runs/phase1/P1b_v2_dgt_ordered/20261110_093229/last.pt",
               "5b22ea7c217b09b96ad2b6309b21b5cbca906444e2f442dd9732965ec9db52c0", True),
    "P1b_s1": ("runs/phase2/P1b_s1/20261111_070454/last.pt",
               "140f0e4f6cf60bbf7e7f186acae6f10fa3923f617654ebefd8bd97db4d888f88", True),
    "P1b_s2": ("runs/phase2/P1b_s2/20261112_082323/last.pt",
               "684b5a324ab71e1a0d004168af563c03e436f96c56c3401bfd9ed9b489956430", True),
    "v2_s0": ("runs/ablation/stage1/S1_v2_full/20261107_193709/last.pt",
              "8a50ea4dc3dd43c2dd23af343df531df4ca5185ff30ad0955dddb5dad60b2eaa", False),
    "v2_s1": ("runs/phase2/v2_s1/20261111_161156/last.pt",
              "9030037adcb124706f1577bb8c194f5ad3358ff2373e6c2c99858a0604875dc0", False),
    "v2_s2": ("runs/phase2/v2_s2/20261112_174530/last.pt",
              "0549d7445b07629e412cd0dbbcc95ac5ca1eadba1deb98e7351e3eba72d23816", False),
    "v1_retrain": ("runs/ablation/stage1/S1_v1_retrain/20261107_102639/last.pt",
                   "42e1b674cb864362275f3f04df061f6c07b815252df738db63993da7b986a624", False),
}
REF_P1B_S0 = (ROOT / "runs/diagnostics/P1B_EVAL517_20261110_184113/per_image_P1b_v2_dgt_ordered.csv",
              "c16fd969710197074a2faa0f7fda403c88c213a3929e1be6437bc1f66637dccc")
for _k, (_p, _s, _d) in CK.items():
    common.CKPTS[_k] = (ROOT / _p, _s, 100000)


def runner(key):
    return DepthRunner(key) if CK[key][2] else Runner(key)


def fwd(run, key, x, z):
    return (run(x, z) if CK[key][2] else run(x))[0]["J"]


def dihedral(t, k, flip):
    t = torch.rot90(t, k, (-2, -1))
    return torch.flip(t, (-1,)) if flip else t


def inv_dihedral(t, k, flip):
    t = torch.flip(t, (-1,)) if flip else t
    return torch.rot90(t, -k, (-2, -1))


def nr(img8):
    return {"uiqm": uiqm(img8)["uiqm"], "uciqe": uciqe(img8)["uciqe"]}


def b1_b3(rows, store, out):
    ref = {r["unit"]: float(r["psnr8"]) for r in csv.DictReader(open(REF_P1B_S0[0], encoding="utf-8"))}
    if sha256_file(REF_P1B_S0[0]) != REF_P1B_S0[1]:
        raise SystemExit("[abort] P1b_s0 reference CSV SHA mismatch")
    run = runner("P1b_s0")
    recs = []
    for i, r in enumerate(rows):
        x8, y8 = load_u8(r["input"]), load_u8(r["reference"])
        x = u8_to_tensor(x8)
        z = store.load(r["unit"], x8.shape[:2])[None]
        acc, single = None, None
        for k in range(4):
            for flip in (False, True):
                J = fwd(run, "P1b_s0", dihedral(x, k, flip), dihedral(z, k, flip)).float()
                J = inv_dihedral(J, k, flip)
                if k == 0 and not flip:
                    single = J
                acc = J if acc is None else acc + J
        p1, p8 = to_u8_tensor(single), to_u8_tensor(acc / 8.0)
        rec = {"unit": r["unit"], "dataset": r["dataset"],
               "psnr_single": psnr_u8(p1, y8), "psnr_x8": psnr_u8(p8, y8),
               "ssim_single": ssim_u8(p1, y8), "ssim_x8": ssim_u8(p8, y8)}
        if abs(rec["psnr_single"] - ref[r["unit"]]) > 1e-6:
            raise SystemExit(f"[abort] single-pass PSNR for {r['unit']} does not reproduce the pre-registered record")
        rec["d_psnr"], rec["d_ssim"] = rec["psnr_x8"] - rec["psnr_single"], rec["ssim_x8"] - rec["ssim_single"]
        for tag, img in (("single", p1), ("x8", p8), ("input", x8), ("ref", y8)):
            for mk, mv in nr(img).items():
                rec[f"{mk}_{tag}"] = mv
        recs.append(rec)
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            print(f"  [B1 x8 P1b_s0] {i + 1}/{len(rows)}", flush=True)
    save_csv(out / "B1_x8_P1b_s0_per_image.csv", recs)
    del run; torch.cuda.empty_cache()
    # v2 seed 0 single-pass predictions for UIQM / UCIQE
    run = runner("v2_s0")
    v2 = []
    for i, r in enumerate(rows):
        p = to_u8_tensor(fwd(run, "v2_s0", u8_to_tensor(load_u8(r["input"])), None))
        v2.append({"unit": r["unit"], "dataset": r["dataset"], **{f"{k}_v2_s0": v for k, v in nr(p).items()}})
        if (i + 1) % 100 == 0 or i + 1 == len(rows):
            print(f"  [B3 v2_s0] {i + 1}/{len(rows)}", flush=True)
    save_csv(out / "B3_v2_s0_nr_per_image.csv", v2)
    del run; torch.cuda.empty_cache()
    summ = {"x8_minus_single": {m: boot_groups(by_dataset(recs, m), mean_stat) for m in ("d_psnr", "d_ssim")},
            "psnr_single": boot_groups(by_dataset(recs, "psnr_single"), mean_stat),
            "psnr_x8": boot_groups(by_dataset(recs, "psnr_x8"), mean_stat),
            "ssim_x8": boot_groups(by_dataset(recs, "ssim_x8"), mean_stat)}
    nrs = {}
    for tag in ("single", "x8", "input", "ref"):
        for mk in ("uiqm", "uciqe"):
            nrs[f"P1b_s0_{tag}_{mk}" if tag in ("single", "x8") else f"{tag}_{mk}"] = \
                boot_groups(by_dataset(recs, f"{mk}_{tag}"), mean_stat)
    for mk in ("uiqm", "uciqe"):
        nrs[f"v2_s0_single_{mk}"] = boot_groups(by_dataset(v2, f"{mk}_v2_s0"), mean_stat)
    summ["no_reference"] = nrs
    return summ


def b2(rows, store, out):
    res = {}
    for key in CK:
        run = runner(key)
        recs = []
        for r in rows:
            x8 = cv2.resize(load_u8(r["input"]), (256, 256), interpolation=cv2.INTER_LINEAR)
            y8 = cv2.resize(load_u8(r["reference"]), (256, 256), interpolation=cv2.INTER_LINEAR)
            z = None
            if CK[key][2]:
                zf = store.load(r["unit"], load_u8(r["input"]).shape[:2])[None]
                z = F.interpolate(zf, size=(256, 256), mode="bilinear", align_corners=False)
            p = to_u8_tensor(fwd(run, key, u8_to_tensor(x8), z))
            recs.append({"unit": r["unit"], "dataset": r["dataset"], "psnr256": psnr_u8(p, y8), "ssim256": ssim_u8(p, y8)})
        save_csv(out / f"B2_256_{key}.csv", recs)
        res[key] = {"psnr256": boot_groups(by_dataset(recs, "psnr256"), mean_stat),
                    "ssim256": boot_groups(by_dataset(recs, "ssim256"), mean_stat)}
        print(f"  [B2 256] {key}: ALL {fmt_ci(res[key]['psnr256']['ALL'])} LSUI {res[key]['psnr256']['LSUI']['est']:.3f} "
              f"UIEB {res[key]['psnr256']['UIEB']['est']:.3f}", flush=True)
        del run; torch.cuda.empty_cache()
    return res


def main():
    commit = git_state()
    tm = Timer()
    torch.set_num_threads(12)
    out = new_outdir("PHASE_B")
    rows = val_rows()
    if len(rows) != 517:
        raise SystemExit("[abort] n != 517")
    store = DepthStore(DEPTH, [(r["dataset"], r["unit"], r["input"], r["reference"]) for r in rows], check_sha=True)
    print(f"[phase_b] commit {commit} out {out}", flush=True)
    s2 = b2(rows, store, out)
    s13 = b1_b3(rows, store, out)
    write_json(out / "summary.json", {"B1_B3": s13, "B2_256": s2,
                                      "labels": "validation (517), not held-out; exploratory; B2 is NOT the native protocol"})
    write_json(out / "meta.json", {"commit": commit, "env": environment(), **tm.done(),
                                   "ckpts": {k: {"path": v[0], "sha256": v[1]} for k, v in CK.items()}})
    print(f"[phase_b] done -> {out} {tm.done()}", flush=True)


if __name__ == "__main__":
    main()

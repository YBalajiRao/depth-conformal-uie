"""
T3: dataset-specific generalisation gap and crop vs native inference (protocol §T3). last.pt.

Native: G_d = mean PSNR8(train,d) - mean PSNR8(val,d); D = G_UIEB - G_LSUI; train and val resampled
independently within dataset (B=2000). Train = fixed 517-image stratified sample (seed 0); val = all 517.
Crop vs native: per image K=4 crops (256x256 clipped to the image), origins uniform with seed 0 + image index
(index within its split list). Each crop inferred ALONE (PadCropWrapper reflect); the native prediction is cut
to the same window. PSNR8 pooled over the K crop windows (overlapping pixels counted once per window).
d_crop = PSNR8(crop-inference) - PSNR8(native-inference) on identical pixels; paired bootstrap, per split.
Criteria: UIEB-specific gap PASS if D >= 1.0 dB and CI(D) excludes 0.
          Scale sensitivity PASS if UIEB mean d_crop >= 1.0 dB with CI excluding 0 on BOTH train and val.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from common import (Runner, git_state, new_outdir, write_json, environment, Timer, val_rows,  # noqa: E402
                    train_sample_rows, load_u8, u8_to_tensor, to_u8_tensor, psnr_u8, psnr_from_mse,
                    boot_groups, by_dataset, mean_stat, B_BOOT, BOOT_SEED, _rng)

K, CROP = 4, 256


def evaluate(run, rows, split):
    recs = []
    for i, r in enumerate(rows):
        x8, y8 = load_u8(r["input"]), load_u8(r["reference"])
        x = u8_to_tensor(x8)
        o, dev = run(x)
        p8 = to_u8_tensor(o["J"])
        H, W = y8.shape[:2]
        ch, cw = min(CROP, H), min(CROP, W)
        rng = np.random.default_rng(0 + i)
        sse_c = sse_n = 0.0
        npx = 0
        devs = {dev}
        for _ in range(K):
            cy, cx = int(rng.integers(0, H - ch + 1)), int(rng.integers(0, W - cw + 1))
            oc, dc = run(x[..., cy:cy + ch, cx:cx + cw])
            devs.add(dc)
            pc8 = to_u8_tensor(oc["J"])
            yw = y8[cy:cy + ch, cx:cx + cw].astype(np.float64)
            sse_c += float(((pc8.astype(np.float64) - yw) ** 2).sum())
            sse_n += float(((p8[cy:cy + ch, cx:cx + cw].astype(np.float64) - yw) ** 2).sum())
            npx += yw.size
        pcrop, pnat = psnr_from_mse(sse_c / npx), psnr_from_mse(sse_n / npx)
        recs.append({"split": split, "unit": r["unit"], "dataset": r["dataset"], "H": H, "W": W,
                     "devices": "+".join(sorted(devs)), "psnr8_native_full": psnr_u8(p8, y8),
                     "psnr8_crops_cropinf": pcrop, "psnr8_crops_nativeinf": pnat, "d_crop": pcrop - pnat})
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            print(f"  [T3 {split}] {i + 1}/{len(rows)}", flush=True)
    return recs


def gap_D(tr, va, B=B_BOOT):
    """D = (mean trU - mean vaU) - (mean trL - mean vaL), all four resampled independently."""
    g = lambda t, v: np.mean(t) - np.mean(v)
    est = g(tr["UIEB"], va["UIEB"]) - g(tr["LSUI"], va["LSUI"])
    rng = _rng()
    bs = []
    for _ in range(B):
        s = {k: a[rng.integers(0, a.size, a.size)] for k, a in
             (("tU", tr["UIEB"]), ("vU", va["UIEB"]), ("tL", tr["LSUI"]), ("vL", va["LSUI"]))}
        bs.append(g(s["tU"], s["vU"]) - g(s["tL"], s["vL"]))
    return {"est": float(est), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}


def gap_d(tr, va, B=B_BOOT):
    est = np.mean(tr) - np.mean(va)
    rng = _rng()
    bs = [np.mean(tr[rng.integers(0, tr.size, tr.size)]) - np.mean(va[rng.integers(0, va.size, va.size)])
          for _ in range(B)]
    return {"est": float(est), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "n_train": int(tr.size), "n_val": int(va.size)}


def main():
    commit = git_state()
    tm = Timer()
    run = Runner("last")
    out = new_outdir("T3")
    train, val = train_sample_rows(), val_rows()
    rt = evaluate(run, train, "train")
    rv = evaluate(run, val, "val")
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rt[0])); w.writeheader(); w.writerows(rt + rv)
    arr = lambda recs, d, k: np.array([r[k] for r in recs if r["dataset"] == d], np.float64)
    tr = {d: arr(rt, d, "psnr8_native_full") for d in ("LSUI", "UIEB")}
    va = {d: arr(rv, d, "psnr8_native_full") for d in ("LSUI", "UIEB")}
    G = {d: gap_d(tr[d], va[d]) for d in ("LSUI", "UIEB")}
    D = gap_D(tr, va)
    crop = {"train": boot_groups(by_dataset(rt, "d_crop"), mean_stat),
            "val": boot_groups(by_dataset(rv, "d_crop"), mean_stat)}
    excl = lambda c: c["lo"] > 0 or c["hi"] < 0
    gap_status = "PASS" if (D["est"] >= 1.0 and excl(D)) else "FAIL"
    scale_status = "PASS" if all(crop[s]["UIEB"]["est"] >= 1.0 and excl(crop[s]["UIEB"]) for s in ("train", "val")) \
        else "FAIL"
    write_json(out / "summary.json", {
        "diagnostic": "T3", "labels": "validation, not held-out; observational",
        "psnr8_native_mean": {"train": boot_groups(by_dataset(rt, "psnr8_native_full"), mean_stat),
                              "val": boot_groups(by_dataset(rv, "psnr8_native_full"), mean_stat)},
        "G_d_train_minus_val": G, "D_UIEB_minus_LSUI": D,
        "uieb_specific_gap": {"criterion": "D >= 1.0 dB AND CI(D) excludes 0", "status": gap_status},
        "d_crop_cropinf_minus_nativeinf": crop,
        "scale_sensitivity": {"criterion": "UIEB mean d_crop >= 1.0 dB AND CI excludes 0, on train AND val",
                              "status": scale_status},
        "notes": ["crop PSNR pooled over K=4 windows; overlapping pixels counted once per window",
                  "training pads crops smaller than 256 by reflect with a loss mask; here PadCropWrapper "
                  "reflect-pads to a multiple of 16 only"],
        "bootstrap": {"B": B_BOOT, "seed": BOOT_SEED,
                      "type": "percentile 95%; independent train/val resampling for G_d and D; paired for d_crop"}})
    write_json(out / "meta.json", {"commit": commit, **run.meta(), "env": environment(), **tm.done(),
                                   "train_images": [r["unit"] for r in train], "val_images": [r["unit"] for r in val]})
    print(f"[T3] gap {gap_status}, scale {scale_status} -> {out}", flush=True)


if __name__ == "__main__":
    main()

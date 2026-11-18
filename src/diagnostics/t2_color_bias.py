"""
T2: dataset colour bias (protocol §T2). last.pt.

1. Signed Lab error per image: mean over pixels of Lab(pred8) - Lab(ref8); skimage.color.rgb2lab on sRGB/255
   (sRGB decoding inside, D65), float64. Mean over images per dataset, bootstrap CI.
2. One full affine transform per dataset, Y ~ W P + b (W 3x3, b 3), OLS in gamma-encoded sRGB [0,1], float64,
   no regularisation, fitted on TRAINING outputs only (fixed 517-image stratified train sample, seed 0),
   20,000 pixels per image drawn without replacement (seed 0 + image index; all pixels if fewer).
3. Applied to validation pred8/255 (float64), clip [0,1], re-quantised uint8(x*255+0.5), PSNR8.
   Recovery = PSNR8(transformed) - PSNR8(original), paired bootstrap CI.
Criterion (PASS = systematic dataset bias supported): UIEB mean recovery >= 1.0 dB, UIEB CI excludes 0,
LSUI mean recovery <= 0.2 dB. Control: identity transform through the same path -> recovery exactly 0 on
every image, else INVALID.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from skimage.color import rgb2lab  # noqa: E402
from common import (Runner, git_state, new_outdir, write_json, environment, Timer, val_rows,  # noqa: E402
                    train_sample_rows, load_u8, u8_to_tensor, to_u8_tensor, psnr_u8, boot_groups, by_dataset,
                    mean_stat, B_BOOT, BOOT_SEED)

N_PIX = 20_000


def apply_affine(p8, W, b):
    P = p8.reshape(-1, 3).astype(np.float64) / 255.0
    Q = np.clip(P @ W.T + b, 0.0, 1.0)
    return (Q * 255.0 + 0.5).astype(np.uint8).reshape(p8.shape)


def main():
    commit = git_state()
    tm = Timer()
    run = Runner("last")
    out = new_outdir("T2")
    train = train_sample_rows()

    # --- fit on training outputs
    Xs, Ys = {}, {}
    for i, r in enumerate(train):
        x8, y8 = load_u8(r["input"]), load_u8(r["reference"])
        o, _ = run(u8_to_tensor(x8))
        p = to_u8_tensor(o["J"]).reshape(-1, 3).astype(np.float64) / 255.0
        y = y8.reshape(-1, 3).astype(np.float64) / 255.0
        rng = np.random.default_rng(0 + i)
        idx = rng.choice(p.shape[0], size=min(N_PIX, p.shape[0]), replace=False)
        Xs.setdefault(r["dataset"], []).append(p[idx]); Ys.setdefault(r["dataset"], []).append(y[idx])
        if (i + 1) % 50 == 0 or i + 1 == len(train):
            print(f"  [T2 train] {i + 1}/{len(train)}", flush=True)
    fits = {}
    for d in sorted(Xs):
        X = np.concatenate(Xs[d]); Y = np.concatenate(Ys[d])
        A = np.concatenate([X, np.ones((X.shape[0], 1))], 1)
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)         # (4,3)
        fits[d] = {"W": coef[:3].T, "b": coef[3], "n_images": len(Xs[d]), "n_pixels": int(X.shape[0])}
        print(f"[T2] {d}: W=\n{fits[d]['W']}\n b={fits[d]['b']}", flush=True)

    # --- validation
    recs = []
    rows = val_rows()
    for i, r in enumerate(rows):
        x8, y8 = load_u8(r["input"]), load_u8(r["reference"])
        o, dev = run(u8_to_tensor(x8))
        p8 = to_u8_tensor(o["J"])
        lab_p, lab_y = rgb2lab(p8 / 255.0), rgb2lab(y8 / 255.0)
        dL, da, db = (lab_p - lab_y).reshape(-1, 3).mean(0)
        f = fits[r["dataset"]]
        t8 = apply_affine(p8, f["W"], f["b"])
        i8 = apply_affine(p8, np.eye(3), np.zeros(3))
        p0 = psnr_u8(p8, y8)
        recs.append({"unit": r["unit"], "dataset": r["dataset"], "device": dev,
                     "dL": float(dL), "da": float(da), "db": float(db),
                     "psnr8": p0, "psnr8_transformed": psnr_u8(t8, y8), "recovery": psnr_u8(t8, y8) - p0,
                     "identity_recovery": psnr_u8(i8, y8) - p0,
                     "identity_maxabs_u8": int(np.abs(i8.astype(int) - p8.astype(int)).max())})
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            print(f"  [T2 val] {i + 1}/{len(rows)}", flush=True)
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
    ctrl_ok = all(r["identity_recovery"] == 0 and r["identity_maxabs_u8"] == 0 for r in recs)
    rec_ci = boot_groups(by_dataset(recs, "recovery"), mean_stat)
    u, l = rec_ci["UIEB"], rec_ci["LSUI"]
    crit = u["est"] >= 1.0 and (u["lo"] > 0 or u["hi"] < 0) and l["est"] <= 0.2
    status = "INVALID" if not ctrl_ok else ("PASS" if crit else "FAIL")
    write_json(out / "summary.json", {
        "diagnostic": "T2", "status": status, "control_identity_exact": ctrl_ok,
        "criterion": "UIEB mean recovery >= 1.0 dB AND UIEB CI excludes 0 AND LSUI mean recovery <= 0.2 dB",
        "recovery_dB": rec_ci,
        "signed_lab_error": {k: boot_groups(by_dataset(recs, k), mean_stat) for k in ("dL", "da", "db")},
        "fits": {d: {"W": f["W"].tolist(), "b": f["b"].tolist(), "n_images": f["n_images"], "n_pixels": f["n_pixels"]}
                 for d, f in fits.items()},
        "colour_space": "gamma-encoded sRGB in [0,1] (not linear); Lab via skimage rgb2lab (D65)",
        "labels": "validation, not held-out; transform fitted on training outputs only",
        "bootstrap": {"B": B_BOOT, "seed": BOOT_SEED, "type": "paired percentile 95%"}})
    write_json(out / "meta.json", {"commit": commit, **run.meta(), "env": environment(), **tm.done(),
                                   "train_images": [r["unit"] for r in train], "val_images": [r["unit"] for r in rows]})
    print(f"[T2] {status} -> {out}", flush=True)


if __name__ == "__main__":
    main()

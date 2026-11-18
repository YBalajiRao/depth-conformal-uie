"""
T5 + G + R517 (protocol §T5, §G, §R517), sharing one fp32 full-image forward pass per image.

last.pt  : T5 (PSNR8/SSIM8/MSE8), G (gauge fraction + spatial gain), R517 (sigma/diff/grad Spearman, AUSE)
step4000 : T5 only (same PSNR convention)
All 517 validation images. Validation, not held-out.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from common import (Runner, git_state, new_outdir, write_json, environment, Timer, val_rows, load_u8,  # noqa: E402
                    u8_to_tensor, to_u8_tensor, mse_u8, psnr_u8, ssim_u8, predictive_std, sparsification,
                    ause, spearman, sobel_mag, boot_groups, by_dataset, mean_stat, median_stat,
                    pooled_psnr_stat, B_BOOT, BOOT_SEED)

MAPS = ("sigma", "diff", "grad")


def affine_fit_apply(P, Y):
    """Per-channel affine least squares P->Y (float64, HxWx3 in [0,1]); returns clip(A*P,0,1)."""
    out = np.empty_like(P)
    for c in range(3):
        p = P[..., c].ravel()
        A = np.stack([p, np.ones_like(p)], 1)
        coef, *_ = np.linalg.lstsq(A, Y[..., c].ravel(), rcond=None)
        out[..., c] = np.clip(P[..., c] * coef[0] + coef[1], 0.0, 1.0)
    return out


def psnr_float(a, b):
    m = float(np.mean((a - b) ** 2))
    return float("inf") if m == 0 else 10 * np.log10(1.0 / m)


def evaluate(run, rows, full: bool):
    recs = []
    for i, r in enumerate(rows):
        x8, y8 = load_u8(r["input"]), load_u8(r["reference"])
        if x8.shape != y8.shape:
            raise ValueError(f"{r['unit']}: input/reference size mismatch")
        x = u8_to_tensor(x8)
        o, dev = run(x)
        mu = o["J"]
        p8 = to_u8_tensor(mu)
        m8 = mse_u8(p8, y8)
        rec = {"unit": r["unit"], "dataset": r["dataset"], "H": y8.shape[0], "W": y8.shape[1], "device": dev,
               "mse8": m8, "psnr8": psnr_u8(p8, y8), "ssim8": ssim_u8(p8, y8)}
        if full:
            P, Y, I = p8 / 255.0, y8 / 255.0, x8 / 255.0
            mse_p = float(np.mean((P - Y) ** 2))
            AP = affine_fit_apply(P, Y)
            AI = affine_fit_apply(I, Y)
            rec["g"] = 1.0 - float(np.mean((AP - Y) ** 2)) / mse_p if mse_p > 0 else float("nan")
            rec["psnrA_pred"], rec["psnrA_input"] = psnr_float(AP, Y), psnr_float(AI, Y)
            rec["spatial_gain"] = rec["psnrA_pred"] - rec["psnrA_input"]
            y = u8_to_tensor(y8)
            nig = o["nig"]
            err = (mu - y).abs().flatten()
            maps = {"sigma": predictive_std(nig[:, 3:6], nig[:, 6:9], nig[:, 9:12], "total").flatten(),
                    "diff": (mu - x).abs().flatten(),
                    "grad": sobel_mag(mu).flatten()}
            for m, u in maps.items():
                assert u.shape == err.shape
                sp = sparsification(err, u)
                rec[f"{m}_spearman"] = spearman(u, err)
                rec[f"{m}_ause_mae"] = ause(sp, "mae")
                rec[f"{m}_ause_rmse"] = ause(sp, "rmse")
        recs.append(rec)
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print(f"  [{run.key}] {i + 1}/{len(rows)}", flush=True)
    return recs


def t5_summary(recs):
    return {"psnr8_mean": boot_groups(by_dataset(recs, "psnr8"), mean_stat),
            "psnr8_pooled": boot_groups(by_dataset(recs, "mse8"), pooled_psnr_stat),
            "ssim8_mean": boot_groups(by_dataset(recs, "ssim8"), mean_stat)}


def save_csv(path, recs):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)


def main():
    commit = git_state()
    tm = Timer()
    out = new_outdir("T5_G_R517")
    rows = val_rows()
    run_last = Runner("last")
    recs_last = evaluate(run_last, rows, full=True)
    meta_last = run_last.meta()
    del run_last
    torch.cuda.empty_cache()
    run_4k = Runner("step4000")
    recs_4k = evaluate(run_4k, rows, full=False)
    meta_4k = run_4k.meta()
    save_csv(out / "per_image_last.csv", recs_last)
    save_csv(out / "per_image_step4000.csv", recs_4k)

    # T5
    assert [r["unit"] for r in recs_last] == [r["unit"] for r in recs_4k]
    for a, b in zip(recs_last, recs_4k):
        a["d_psnr8_vs_4k"] = a["psnr8"] - b["psnr8"]
    t5 = {"last": t5_summary(recs_last), "step4000": t5_summary(recs_4k),
          "paired_diff_psnr8_mean_last_minus_4k": boot_groups(by_dataset(recs_last, "d_psnr8_vs_4k"), mean_stat),
          "status": "EXPLORATORY (descriptive)"}

    # G
    g = {"median_g": boot_groups(by_dataset(recs_last, "g"), median_stat),
         "quartiles_g": {d: [float(np.percentile(v[0], q)) for q in (25, 50, 75)]
                         for d, v in by_dataset(recs_last, "g").items()},
         "mean_spatial_gain_dB": boot_groups(by_dataset(recs_last, "spatial_gain"), mean_stat),
         "mean_psnrA_pred": boot_groups(by_dataset(recs_last, "psnrA_pred"), mean_stat),
         "mean_psnrA_input": boot_groups(by_dataset(recs_last, "psnrA_input"), mean_stat)}
    mg = g["median_g"]["UIEB"]["est"]
    g["criterion"] = "median(g_UIEB) >= 0.5"
    g["status"] = "PASS" if mg >= 0.5 else "FAIL"

    # R517
    r517 = {"spearman_mean": {m: boot_groups(by_dataset(recs_last, f"{m}_spearman"), mean_stat) for m in MAPS},
            "ause_mae_mean": {m: boot_groups(by_dataset(recs_last, f"{m}_ause_mae"), mean_stat) for m in MAPS},
            "ause_rmse_mean": {m: boot_groups(by_dataset(recs_last, f"{m}_ause_rmse"), mean_stat) for m in MAPS},
            "beats": {}}
    for b in ("diff", "grad"):
        for met in ("ause_mae", "ause_rmse"):
            for r in recs_last:
                r[f"d_{b}_{met}"] = r[f"{b}_{met}"] - r[f"sigma_{met}"]
            ci = boot_groups(by_dataset(recs_last, f"d_{b}_{met}"), mean_stat)
            r517["beats"][f"{b}_{met}"] = {grp: {**c, "status": "PASS" if c["lo"] > 0 else "FAIL"}
                                           for grp, c in ci.items()}
    r517["criterion"] = "paired-bootstrap 95% CI of mean(AUSE_baseline - AUSE_sigma) entirely > 0"
    r517["note"] = "517-image results are not directly comparable to the 40-image step-20k gate"

    write_json(out / "summary.json", {"T5": t5, "G": g, "R517": r517, "labels": "validation, not held-out",
                                      "bootstrap": {"B": B_BOOT, "seed": BOOT_SEED, "type": "percentile 95%"}})
    write_json(out / "meta.json", {"commit": commit, "last": meta_last, "step4000": meta_4k, "env": environment(),
                                   **tm.done(), "images": [r["unit"] for r in rows]})
    print(f"[T5/G/R517] done -> {out}", flush=True)


if __name__ == "__main__":
    main()

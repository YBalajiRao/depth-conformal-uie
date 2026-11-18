"""
T1: tiled vs full-image inference (protocol §T1). last.pt, all 517 validation images.

Tiled: tile T=256 (clipped to image dims), stride 128, origins 0,128,... plus a final tile flush with the
bottom/right edge; each tile inferred alone through PadCropWrapper. Centre-region stitching: each pixel
takes the tile where its distance to the nearest INTERIOR tile edge is largest (image-border edges do not
count); ties -> earliest tile in raster order.
Metric: dPSNR8 = PSNR8(tiled) - PSNR8(full), all pixels. Criterion (PASS = consistent with H1):
UIEB mean >= 1.0 dB, UIEB CI excludes 0, LSUI |mean| <= 0.2 dB.
Control: the same tiling/stitching code with T >= max(H,W) on the 20 T0 images must equal full mu exactly,
else INVALID.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from common import (Runner, git_state, new_outdir, write_json, environment, Timer, val_rows, load_u8,  # noqa: E402
                    u8_to_tensor, to_u8_tensor, psnr_u8, fixed_subset, boot_groups, by_dataset, mean_stat,
                    B_BOOT, BOOT_SEED)

T_TILE, STRIDE = 256, 128


def origins(n, t, s):
    o = list(range(0, n - t + 1, s))
    if o[-1] + t < n:
        o.append(n - t)
    return o


def tiled_mu(run, x, T, stride, device):
    _, _, H, W = x.shape
    th, tw = min(T, H), min(T, W)
    ys, xs = origins(H, th, stride), origins(W, tw, stride)
    out = torch.zeros(1, 3, H, W)
    best = torch.full((H, W), -1.0, dtype=torch.float64)
    rr = torch.arange(th, dtype=torch.float64)[:, None]
    cc = torch.arange(tw, dtype=torch.float64)[None, :]
    INF = 1e18
    devs = set()
    for y0 in ys:
        for x0 in xs:
            o, dev = run(x[..., y0:y0 + th, x0:x0 + tw], force_device=device)
            devs.add(dev)
            d_top = rr if y0 > 0 else torch.full_like(rr, INF)
            d_bot = (th - 1 - rr) if y0 + th < H else torch.full_like(rr, INF)
            d_lef = cc if x0 > 0 else torch.full_like(cc, INF)
            d_rig = (tw - 1 - cc) if x0 + tw < W else torch.full_like(cc, INF)
            score = torch.minimum(torch.minimum(d_top, d_bot).expand(th, tw), torch.minimum(d_lef, d_rig).expand(th, tw))
            region = best[y0:y0 + th, x0:x0 + tw]
            take = score > region
            region[take] = score[take]
            tgt = out[..., y0:y0 + th, x0:x0 + tw]
            tgt[..., take] = o["J"][..., take]
    assert bool((best >= 0).all()), "uncovered pixels"
    return out, len(ys) * len(xs), devs


def main():
    commit = git_state()
    tm = Timer()
    run = Runner("last")
    out = new_outdir("T1")
    rows = val_rows()

    # control: tile >= image -> single tile -> must equal full mu exactly (same device)
    ctrl = []
    for r in fixed_subset(rows, 40, 0)[:20]:
        x = u8_to_tensor(load_u8(r["input"]))
        full, dev = run(x)
        t, ntiles, devs = tiled_mu(run, x, max(x.shape[-2:]), STRIDE, dev)
        ctrl.append({"unit": r["unit"], "device": dev, "tiles": ntiles,
                     "maxabs": float((t - full["J"]).abs().max())})
    ctrl_ok = all(c["maxabs"] == 0 and c["tiles"] == 1 for c in ctrl)
    print(f"[T1] control exact on all 20: {ctrl_ok}", flush=True)

    recs = []
    for i, r in enumerate(rows):
        x8, y8 = load_u8(r["input"]), load_u8(r["reference"])
        x = u8_to_tensor(x8)
        full, dev_full = run(x)
        t, ntiles, devs = tiled_mu(run, x, T_TILE, STRIDE, None)
        pf, pt = psnr_u8(to_u8_tensor(full["J"]), y8), psnr_u8(to_u8_tensor(t), y8)
        recs.append({"unit": r["unit"], "dataset": r["dataset"], "H": y8.shape[0], "W": y8.shape[1],
                     "device_full": dev_full, "device_tiles": "+".join(sorted(devs)), "tiles": ntiles,
                     "psnr8_full": pf, "psnr8_tiled": pt, "d_psnr8": pt - pf})
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print(f"  [T1] {i + 1}/{len(rows)}", flush=True)
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
    with open(out / "control.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ctrl[0])); w.writeheader(); w.writerows(ctrl)
    ci = boot_groups(by_dataset(recs, "d_psnr8"), mean_stat)
    u, l = ci["UIEB"], ci["LSUI"]
    crit = u["est"] >= 1.0 and (u["lo"] > 0 or u["hi"] < 0) and abs(l["est"]) <= 0.2
    status = "INVALID" if not ctrl_ok else ("PASS" if crit else "FAIL")
    write_json(out / "summary.json", {
        "diagnostic": "T1", "status": status, "control_ok": ctrl_ok,
        "criterion": "UIEB mean dPSNR8 >= 1.0 dB AND UIEB CI excludes 0 AND LSUI |mean dPSNR8| <= 0.2 dB",
        "d_psnr8_tiled_minus_full": ci,
        "psnr8_full_mean": boot_groups(by_dataset(recs, "psnr8_full"), mean_stat),
        "psnr8_tiled_mean": boot_groups(by_dataset(recs, "psnr8_tiled"), mean_stat),
        "tile": T_TILE, "stride": STRIDE, "labels": "validation, not held-out",
        "bootstrap": {"B": B_BOOT, "seed": BOOT_SEED, "type": "paired percentile 95%"}})
    write_json(out / "meta.json", {"commit": commit, **run.meta(), "env": environment(), **tm.done(),
                                   "images": [r["unit"] for r in rows]})
    print(f"[T1] {status} -> {out}", flush=True)


if __name__ == "__main__":
    main()

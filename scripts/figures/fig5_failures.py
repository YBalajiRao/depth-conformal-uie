"""
Fig. 5: failure cases with the predicted uncertainty (no new read of the locked test split).

Rows 1-2: the two P1b failure cases among the 6 pre-selected Phase F test units (LSUI:364, UIEB:797); images, P1b
          output and the NIG TOTAL predictive std were saved by phase_f_test.py at the single test opening.
Rows 3-4: the two lowest-PSNR P1b images on the zero-shot EUVP underwater_dark set (Phase E per-image file); P1b is
          run here (Phase E runner: same checkpoint and on-the-fly depth), cached in runs/figures/fig5_cache.npz.
Columns: Input | P1b | |J - J_hat| (mean over RGB) | predictive sigma (NIG total, mean over RGB) | Reference.
Per row: PSNR8 and Spearman rho between sigma and |error| over all pixels. Error and sigma maps use per-row 1-99th
percentile scaling (shapes, not absolute values, are compared). Tiles: centre square crops resized to 256 px.
Outputs: figs/fig5_failures.pdf (vector) and .png (300 dpi).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "runs/phase_f/PHASE_F_TEST_20261117_212802/grid"
PHE = ROOT / "runs/diagnostics/PHASE_E_20261117_182411"
CACHE = ROOT / "runs/figures/fig5_cache.npz"
TEST_UNITS = [("LSUI_364.jpg", "test LSUI 364"), ("UIEB_797_img_.png", "test UIEB 797")]


def psnr(p, y):
    m = float(np.mean((p.astype(np.float64) - y.astype(np.float64)) ** 2))
    return 99.0 if m == 0 else 10 * np.log10(255 ** 2 / m)


def dark_rows():
    if CACHE.exists():
        with np.load(CACHE, allow_pickle=False) as d:
            return [{k: d[f"{i}|{k}"] for k in ("input", "out", "sigma", "ref")} | {"name": str(d[f"{i}|name"])} for i in range(2)]
    sys.path[:0] = [str(ROOT / "src/diagnostics")]
    import torch
    import phase_e_eval as PE
    from common import load_u8, to_u8_tensor, predictive_std
    torch.cuda.set_per_process_memory_fraction(0.50, 0)
    worst = sorted(csv.DictReader(open(PHE / "dark_P1b_s0.csv", encoding="utf-8")), key=lambda r: float(r["psnr8"]))[:2]
    pairs = {p.name: (p, q) for p, q in PE.dark_pairs()}
    run = PE.P1bRunner(torch.device("cuda")); rows = []
    with torch.no_grad():
        for w in worst:
            p, q = pairs[w["image"]]; x8, y8 = load_u8(p), load_u8(q); o, _ = run(x8)
            rows.append({"input": x8, "out": to_u8_tensor(o["J"]), "ref": y8, "name": f"EUVP-dark {p.stem[:6]}",
                         "sigma": predictive_std(o["nu"], o["alpha"], o["beta"], "total")[0].mean(0).numpy()})
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **{f"{i}|{k}": np.asarray(v) for i, r in enumerate(rows) for k, v in r.items()})
    return rows


def sq(a, n=256):
    h, w = a.shape[:2]; s = min(h, w); t, l = (h - s) // 2, (w - s) // 2
    a = a[t:t + s, l:l + s]
    if a.ndim == 2:
        return np.asarray(Image.fromarray(a.astype(np.float32)).resize((n, n), Image.BILINEAR))
    return np.asarray(Image.fromarray(a).resize((n, n), Image.LANCZOS))


def scaled(m):
    lo, hi = np.percentile(m, [1, 99]); return np.clip((m - lo) / max(hi - lo, 1e-8), 0, 1)


def main():
    rows = []
    for unit, name in TEST_UNITS:
        g = GRID / unit
        rows.append({"input": np.asarray(Image.open(g / "input.png")), "out": np.asarray(Image.open(g / "P1b_s0.png")),
                     "ref": np.asarray(Image.open(g / "reference.png")), "sigma": np.load(g / "P1b_s0_sigma.npy"), "name": name})
    rows += dark_rows()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix",
                         "pdf.fonttype": 42})
    cols = ["Input", "P1b (ours)", r"$|J-\hat J|$ (error)", r"predictive $\sigma$ (NIG total)", "Reference"]
    fig, axs = plt.subplots(len(rows), 5, figsize=(7.2, 7.2 * len(rows) / 5 + 0.3), gridspec_kw={"wspace": 0.03, "hspace": 0.16})
    for i, r in enumerate(rows):
        err = np.abs(r["out"].astype(np.float32) - r["ref"].astype(np.float32)).mean(2) / 255.0
        sig = r["sigma"].astype(np.float32)
        sub = np.random.default_rng(0).choice(err.size, size=min(200_000, err.size), replace=False)
        rho = float(spearmanr(sig.ravel()[sub], err.ravel()[sub]).statistic)
        r["psnr"], r["rho"] = psnr(r["out"], r["ref"]), rho
        tiles = [sq(r["input"]), sq(r["out"]), plt.get_cmap("magma")(scaled(sq(err)))[..., :3],
                 plt.get_cmap("viridis")(scaled(sq(sig)))[..., :3], sq(r["ref"])]
        for j, t in enumerate(tiles):
            a = axs[i, j]; a.imshow(t); a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_linewidth(0.4)
            if i == 0:
                a.set_title(cols[j], fontsize=7, pad=2)
        axs[i, 1].text(0.03, 0.03, f"{r['psnr']:.2f} dB", color="white", fontsize=6, transform=axs[i, 1].transAxes,
                       bbox=dict(fc="black", alpha=0.6, pad=0.6, lw=0))
        axs[i, 3].text(0.03, 0.03, f"Spearman $\\rho$ = {rho:+.2f}", color="white", fontsize=6, transform=axs[i, 3].transAxes,
                       bbox=dict(fc="black", alpha=0.6, pad=0.6, lw=0))
        axs[i, 0].set_ylabel(r["name"], fontsize=5.5, labelpad=2)
    out = ROOT / "figs"; out.mkdir(exist_ok=True)
    fig.savefig(out / "fig5_failures.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / "fig5_failures.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    print("[fig5] rows:", [(r["name"], round(r["psnr"], 2), round(r["rho"], 3)) for r in rows])


if __name__ == "__main__":
    main()

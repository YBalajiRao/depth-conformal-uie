"""
Qualitative comparison grids (run after Phase F; GPU only for the zero-shot grid).

  test     : the 6 test units fixed by Phase F (rng 20261117, drawn before any output existed), outputs saved by
             src/diagnostics/phase_f_test.py under runs/phase_f/PHASE_F_TEST_*/grid/. Per-tile PSNR from its CSVs.
  zeroshot : 3 EUVP underwater_dark pairs + 3 Challenge-60 images drawn with rng 20261117 from the sorted file lists;
             P1b_s0 and the three retrained baselines run here (Phase E runners: same checkpoints and depth handling).
Tiles are centre square crops resized to 256 px (stated in the caption). Outputs: figs/fig_qualitative_{test,zeroshot}.{pdf,png}
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figs"
COLS = [("input", "Input"), ("fiveaplus", "Five A$^+$"), ("ushape", "U-shape"), ("wweuie", "WWE-UIE"),
        ("v2_s0", "v2 (no DGT)"), ("P1e_s0", "P1e"), ("P1b_s0", "P1b (ours)"), ("sigma", r"P1b $\sigma$"),
        ("reference", "Reference")]


def sq(a, n=256):
    h, w = a.shape[:2]; s = min(h, w); t, l = (h - s) // 2, (w - s) // 2
    a = a[t:t + s, l:l + s]
    if a.ndim == 2:
        lo, hi = np.percentile(a, [1, 99]); a = np.clip((a - lo) / max(hi - lo, 1e-8), 0, 1)
        a = (plt.get_cmap("viridis")(a)[..., :3] * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(a).resize((n, n), Image.LANCZOS))


def psnr(p, y):
    m = float(np.mean((p.astype(np.float64) - y.astype(np.float64)) ** 2))
    return 99.0 if m == 0 else 10 * np.log10(255 ** 2 / m)


def short_label(s, max_id=8):
    """Compact row label that fits within one tile height: 'UIEB:675_img_.png' -> 'UIEB 675',
    'LSUI:1496.jpg' -> 'LSUI 1496', 'EUVP-dark: 270365_n02319095_7146.JPEG' -> 'EUVP-dark 270365_…', 'C60: 52.png' -> 'C60 52'."""
    src, _, name = s.partition(":")
    stem = Path(name.strip()).stem.replace("_img_", "").strip("_")
    if len(stem) > max_id:
        stem = stem[:max_id - 1].rstrip("_") + "_…"
    return f"{src.strip()} {stem}"


def grid(rows, cols, out_stem, title_rows):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix",
                         "pdf.fonttype": 42})
    nr, nc = len(rows), len(cols)
    fig, axs = plt.subplots(nr, nc, figsize=(7.2, 7.2 * nr / nc + 0.35), gridspec_kw={"wspace": 0.03, "hspace": 0.12})
    axs = np.atleast_2d(axs)
    for i, r in enumerate(rows):
        for j, (k, lab) in enumerate(cols):
            a = axs[i, j]; a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_linewidth(0.4)
            if r.get(k) is None:
                a.set_facecolor("#F2F2F2"); a.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=6, transform=a.transAxes)
            else:
                a.imshow(sq(r[k]))
            if i == 0:
                a.set_title(lab, fontsize=7, pad=2)
            if f"{k}_psnr" in r:
                a.text(0.03, 0.03, f"{r[k + '_psnr']:.2f}", color="white", fontsize=5.5, transform=a.transAxes,
                       bbox=dict(fc="black", alpha=0.55, pad=0.6, lw=0))
        axs[i, 0].set_ylabel(short_label(title_rows[i]), fontsize=4.8, rotation=90, labelpad=2)
    OUT.mkdir(exist_ok=True)
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def test_grid():
    d = sorted((ROOT / "runs/phase_f").glob("PHASE_F_TEST_*"))[-1]
    ps = {k: {r["unit"]: float(r["psnr8"]) for r in csv.DictReader(open(d / f"per_image_{k}.csv", encoding="utf-8"))}
          for k, _ in COLS if (d / f"per_image_{k}.csv").exists()}
    rows, names = [], []
    for gd in sorted((d / "grid").iterdir()):
        unit = next(u for u in ps["P1b_s0"] if u.replace(":", "_").replace("/", "_") == gd.name)
        r = {"input": np.asarray(Image.open(gd / "input.png")), "reference": np.asarray(Image.open(gd / "reference.png")),
             "sigma": np.load(gd / "P1b_s0_sigma.npy")}
        for k, _ in COLS:
            if (gd / f"{k}.png").exists():
                r[k] = np.asarray(Image.open(gd / f"{k}.png"))
                if k in ps:                     # model outputs only (input/reference have no PSNR)
                    r[f"{k}_psnr"] = ps[k][unit]
        rows.append(r); names.append(unit)
    grid(rows, COLS, str(OUT / "fig_qualitative_test"), names)
    print("test grid from", d)


CACHE = ROOT / "runs/figures/zeroshot_grid_cache.npz"     # model outputs of the zero-shot grid (re-render on CPU)


def zeroshot_from_cache():
    with np.load(CACHE, allow_pickle=False) as d:
        names = [str(n) for n in d["names"]]; keys = [str(k) for k in d["keys"]]
        rows = [{} for _ in names]
        for k in keys:
            i, f = k.split("|", 1)
            rows[int(i)][f] = d[k] if d[k].ndim else float(d[k])
    cols = [c for c in COLS if c[0] not in ("v2_s0", "P1e_s0")]
    for r in rows:
        r.setdefault("reference", None)
    grid(rows, cols, str(OUT / "fig_qualitative_zeroshot"), names)
    print("zero-shot grid (from cache):", names)


def zeroshot_grid():
    if CACHE.exists() and "--refresh" not in sys.argv:
        return zeroshot_from_cache()
    sys.path[:0] = [str(ROOT / "src/diagnostics")]
    import torch
    import phase_e_eval as PE
    from common import load_u8, u8_to_tensor, to_u8_tensor, predictive_std
    torch.cuda.set_per_process_memory_fraction(0.50, 0)
    rng = np.random.default_rng(20261117)
    pairs = PE.dark_pairs(); c60 = sorted(p for p in PE.C60.iterdir() if p.suffix.lower() in PE.IMG)
    sel = [("dark", *pairs[i]) for i in sorted(rng.choice(len(pairs), 3, replace=False))] + \
          [("c60", c60[i], None) for i in sorted(rng.choice(len(c60), 3, replace=False))]
    p1b = PE.P1bRunner(torch.device("cuda"))
    base = {n: PE.BaselineRunner(n, str(ROOT / ck), sh) for n, (ck, sh) in PE.BASELINES.items()}
    cols = [c for c in COLS if c[0] not in ("v2_s0", "P1e_s0")]
    rows, names = [], []
    with torch.no_grad():
        for s, p, q in sel:
            x8 = load_u8(p); y8 = load_u8(q) if q else None
            o, _ = p1b(x8)
            r = {"input": x8, "reference": y8, "P1b_s0": to_u8_tensor(o["J"]),
                 "sigma": predictive_std(o["nu"], o["alpha"], o["beta"], "total")[0].mean(0).numpy()}
            for n, br in base.items():
                J, _ = br(u8_to_tensor(x8)); r[n] = to_u8_tensor(J)
            if y8 is not None:
                for k in ("P1b_s0", *base):
                    r[f"{k}_psnr"] = psnr(r[k], y8)
            rows.append(r); names.append(f"{'EUVP-dark' if s == 'dark' else 'C60'}: {p.name}")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, names=np.array(names), keys=np.array([f"{i}|{k}" for i, r in enumerate(rows) for k, v in r.items()
                                                                       if v is not None]),
                        **{f"{i}|{k}": np.asarray(v) for i, r in enumerate(rows) for k, v in r.items() if v is not None})
    grid(rows, cols, str(OUT / "fig_qualitative_zeroshot"), names)
    print("zero-shot grid:", names)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("which", choices=["test", "zeroshot", "both"]); ap.add_argument("--refresh", action="store_true"); a = ap.parse_args()
    if a.which in ("test", "both"):
        test_grid()
    if a.which in ("zeroshot", "both"):
        zeroshot_grid()

"""
Fig. 1: architecture of the locked flagship model P1b (SS-PGN v2 + ordered depth-grounded transmission).

Every block, width and equation below is taken from src/models/ss_pgn.py, src/models/dgt_head.py and
src/training/losses_sspgn.py (config of runs/phase1/P1b_v2_dgt_ordered/20261110_093229/last.pt):
  widths 64/128/256/512, 1 NAFBlock per encoder/decoder stage, one SS-PGB bottleneck, phys_conditioning v2,
  compose 'direct' (mu = x + R), depth_transmission with depth_mode 'one_minus' (z = 1 - z~), dgt_ordered.
  No decoder FiLM (P1c rejected), no scene prior (P1d/P1e not adopted), no CSA.
  J_phys (inverse model) is NOT on the inference path; physics enters training only through L_phys (re-rendering).
Thumbnails: one CPU forward pass of P1b on a centre crop of one VALIDATION image (never test).

  python scripts/figures/fig1_architecture.py                      # default beta annotation ('ordered')
  python scripts/figures/fig1_architecture.py --beta-label collapsed
  python scripts/figures/fig1_architecture.py --beta-label "custom text with $\\beta$"
  python scripts/figures/fig1_architecture.py --no-thumbs          # boxes only (no model run)
Outputs: figs/fig1_architecture.pdf (vector) and figs/fig1_architecture.png (300 dpi).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "runs/phase1/P1b_v2_dgt_ordered/20261110_093229/last.pt"
OUT = ROOT / "figs"
BETA_PRESETS = {
    "ordered": r"$\beta_R\geq\beta_G\geq\beta_B$ (sequential softplus)",
    "collapsed": r"$\beta_R\gtrsim\beta_G\gtrsim\beta_B$ (learned gaps $\approx 0$)",
    # Phase E (2026-11-17, runs/diagnostics/PHASE_E_BETA_20261117_185226): beta^D channel spread median 0.2-0.6 %
    # (dark / val517 / C60); beta^B: R = G (gap ~1e-6) and B lower by 4-15 % (median relative spread)
    "empirical": "imposed " + r"$\beta_R\geq\beta_G\geq\beta_B$" + "; learned:\n"
                 + r"$\beta^{D}_R\approx\beta^{D}_G\approx\beta^{D}_B$,   $\beta^{B}_R\approx\beta^{B}_G>\beta^{B}_B$",
}
PARAMS = {"total": "4,284,413", "enc": "1.31 M", "ss_pgb": "1.46 M", "dgt": "132 K", "gphys": "69 K", "dec": "1.30 M",
          "tails": "6.9 K"}   # enc = head+enc+down, dec = up+dec; numel() on the P1b checkpoint (2026-11-17)
C = {"depth": "#E3F0FB", "backbone": "#EFEFEF", "pgb": "#FFF4D6", "out": "#E8F5E9", "edge": "#4A4A4A",
     "block": "#FFFFFF", "phys": "#FBE3D6", "loss": "#C0392B", "naf": "#F6C9A0", "dwt": "#B9D7EA", "head": "#D7C6E8",
     "z": "#1F5FA8"}


def thumbs(n_crop=256):
    """CPU forward of P1b on the centre crop of one validation image; returns arrays in [0, 1]."""
    sys.path[:0] = [str(ROOT / s) for s in ("src/models", "src/training", "src/validation", "src/diagnostics")]
    import torch
    from ss_pgn import SSPGN, SSPGNConfig
    from common import val_rows, load_u8
    from train import DepthStore
    from p1_eval517 import DEPTH
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    mc = ck["cfg"]["model_cfg"]
    cfg = SSPGNConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in mc.items()
                         if k in SSPGNConfig.__dataclass_fields__})
    m = SSPGN(cfg).eval(); m.load_state_dict(ck["model"], strict=True)
    rows = val_rows()
    r = next(r for r in rows if r["dataset"] == "LSUI")
    x8 = load_u8(r["input"])
    rows4 = [(q["dataset"], q["unit"], q["input"], q["reference"]) for q in rows]
    z = DepthStore(DEPTH, rows4, check_sha=True).load(r["unit"], x8.shape[:2]).numpy()[0]
    H, W = x8.shape[:2]; t, l = (H - n_crop) // 2, (W - n_crop) // 2
    xc, zc = x8[t:t + n_crop, l:l + n_crop], z[t:t + n_crop, l:l + n_crop]
    with torch.no_grad():
        o = m(torch.from_numpy(xc).permute(2, 0, 1)[None].float() / 255, torch.from_numpy(zc)[None, None].float())
    tot = (o["beta"] * (1 + o["nu"]) / (o["nu"] * (o["alpha"] - 1))).sqrt().mean(1)[0].numpy()
    return {"unit": r["unit"], "x": xc / 255.0, "z": zc, "tD": o["tD"][0].permute(1, 2, 0).clamp(0, 1).numpy(),
            "mu": o["J"][0].permute(1, 2, 0).clamp(0, 1).numpy(), "sigma": tot,
            "beta_D": o["beta_D"][0].tolist(), "beta_B": o["beta_B"][0].tolist()}


def box(ax, x, y, w, h, text, fc, fs=7.0, bold=False, ec=None, lw=0.8, ls="-", r=0.6, color="black", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec or C["edge"],
                                lw=lw, ls=ls, zorder=z))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, zorder=z + 1,
                fontweight="bold" if bold else "normal", linespacing=1.15)


def panel(ax, x, y, w, h, title, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec="#9A9A9A",
                                lw=0.8, ls=(0, (4, 2)), zorder=0))
    ax.text(x + 1.0, y + h - 1.0, title, ha="left", va="top", fontsize=7.6, fontweight="bold", zorder=1)


def arrow(ax, p, q, ls="-", color=None, lw=0.85):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>,head_length=3.0,head_width=1.7", lw=lw, ls=ls,
                                 color=color or C["edge"], shrinkA=0, shrinkB=0, zorder=4))


def poly(ax, pts, ls="-", color=None, lw=0.85):
    """Orthogonal polyline through pts with an arrow head on the last segment."""
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], ls=ls, color=color or C["edge"], lw=lw, zorder=4, solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1], ls=ls, color=color, lw=lw)


def dot(ax, x, y, r=0.32):
    """Explicit junction marker where one line branches off another."""
    ax.add_patch(plt.Circle((x, y), r, fc=C["edge"], ec="none", zorder=6))


def op(ax, x, y, s, r=0.9):
    ax.add_patch(plt.Circle((x, y), r, fc="white", ec=C["edge"], lw=0.8, zorder=5))
    ax.text(x, y - 0.05, s, ha="center", va="center", fontsize=7.5, zorder=6)


def img(fig, ax, data, x, y, w, h, cmap=None, label=None, fs=6.2):
    """Image thumbnail at data coords (x, y, w, h) of ax."""
    inv = fig.transFigure.inverted()
    (fx0, fy0), (fx1, fy1) = inv.transform(ax.transData.transform((x, y))), inv.transform(ax.transData.transform((x + w, y + h)))
    a = fig.add_axes([fx0, fy0, fx1 - fx0, fy1 - fy0]); a.set_xticks([]); a.set_yticks([])
    if data is None:
        a.set_facecolor("#DDDDDD")
    else:
        a.imshow(data, cmap=cmap, interpolation="lanczos", aspect="auto")
    for s in a.spines.values():
        s.set_linewidth(0.6); s.set_color(C["edge"])
    if label:
        ax.text(x + w / 2, y - 0.6, label, ha="center", va="top", fontsize=fs, zorder=6)


def draw(T, beta_label, out_stem):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.2, 4.98))                       # 183 mm wide (full text width)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 120); ax.set_ylim(-4.4, 80); ax.axis("off")
    g = lambda k: None if T is None else T[k]   # noqa: E731
    ST = r"$^{\star}$"
    DASH = (0, (3, 2))

    # ---------------- (a) depth prior ----------------
    panel(ax, 1, 44, 23, 35.5, "(a) Depth prior", C["depth"])
    img(fig, ax, g("x"), 3.0, 61.5, 8.0, 8.0, label=r"input $x$")
    box(ax, 3.0, 47.5, 9.0, 9.5, "Depth-\nAnything-\nV2-Small\n[frozen]", C["block"], fs=5.9)
    arrow(ax, (7.0, 59.4), (7.0, 57.0))
    img(fig, ax, g("z"), 14.0, 48.0, 8.5, 8.5, cmap="inferno", label=r"$\tilde z$")
    arrow(ax, (12.0, 52.25), (14.0, 52.25))
    ax.text(2.4, 45.2, r"precomputed once; $z=1-\tilde z$ (relative)", fontsize=5.5, va="center")
    # depth-prior path (a) -> (c) in blue: right edge of (a), along the inter-panel gap, down into "z from (a)"
    poly(ax, [(22.5, 52.25), (23.3, 52.25), (23.3, 43.25), (60.0, 43.25), (60.0, 38.5)], color=C["z"], lw=0.8)

    # ---------------- (b) backbone ----------------
    panel(ax, 25.5, 44, 93.5, 35.5, "(b) Restoration backbone (NAFNet-style U-Net, 4 levels)", C["backbone"])
    yc, bh = 62.0, 9.0
    yb = yc - bh / 2
    arrow(ax, (11.0, 65.5), (27.0, 65.5)); ax.text(19.0, 66.1, r"$x$", fontsize=7, ha="center")
    box(ax, 27.0, yb, 5.0, bh, "conv\n3×3", C["head"], fs=6.2)
    xs, x = {}, 33.6
    for k, (lab, res) in enumerate([("NAF\n@64", "H"), ("NAF\n@128", "H/2"), ("NAF\n@256", "H/4")]):
        arrow(ax, (x - 1.6, yc), (x, yc))
        box(ax, x, yb, 5.0, bh, lab, C["naf"], fs=6.2)
        ax.text(x + 2.5, yb - 0.6, res, ha="center", va="top", fontsize=5.8)
        xs[k] = x + 2.5
        arrow(ax, (x + 5.0, yc), (x + 6.0, yc))
        box(ax, x + 6.0, yb + 1.8, 2.0, bh - 3.6, "", C["dwt"])
        ax.text(x + 7.0, yc, "↓", ha="center", va="center", fontsize=7)
        x += 9.6
    arrow(ax, (x - 1.6, yc), (x, yc))
    box(ax, x, yb - 1.0, 8.0, bh + 2.0, "SS-PGB\n@512\n(H/8)", C["pgb"], fs=6.6, bold=True, lw=1.2)
    xpgb = x
    # callout bracket (b) -> (c): one straight vertical connector from the box's bottom edge to panel (c)'s top border,
    # with short end ticks; it runs through clear whitespace (no arrow, no operator, no text on its path)
    xc, CO = x + 4.0, "#8C7A3C"
    ax.plot([xc, xc], [yb - 1.0, 42.5], lw=0.7, color=CO, zorder=3)
    for yy in (yb - 1.0, 42.5):
        ax.plot([xc - 1.6, xc + 1.6], [yy, yy], lw=0.9, color=CO, zorder=3)
    ax.text(xc - 0.8, 54.3, "detail in (c)", ha="right", va="center", fontsize=5.8, style="italic", color=CO)
    x += 8.0
    for k, (lab, res) in enumerate([("NAF\n@256", "H/4"), ("NAF\n@128", "H/2"), ("NAF\n@64", "H")]):
        arrow(ax, (x, yc), (x + 1.6, yc)); x += 1.6
        box(ax, x, yb + 1.8, 2.0, bh - 3.6, "", C["dwt"])
        ax.text(x + 1.0, yc, "↑", ha="center", va="center", fontsize=7)
        arrow(ax, (x + 2.0, yc), (x + 2.7, yc)); op(ax, x + 3.6, yc, "+")
        top = 69.0 + 2.0 * k
        poly(ax, [(xs[2 - k], yb + bh), (xs[2 - k], top), (x + 3.6, top), (x + 3.6, yc + 0.9)], ls=DASH, lw=0.7)
        arrow(ax, (x + 4.5, yc), (x + 5.2, yc)); x += 5.2
        box(ax, x, yb, 5.0, bh, lab, C["naf"], fs=6.2)
        ax.text(x + 2.5, yb - 0.6, res, ha="center", va="top", fontsize=5.8)
        x += 5.0
    # output heads; (+) sits on the same row as R, to its right
    xt, xo = 108.0, 117.0
    box(ax, xt, 63.5, 7.0, 5.0, r"$R$" + " (3 ch)", C["out"], fs=6.0)
    box(ax, xt, 55.5, 7.0, 5.0, r"$\nu,\alpha,\beta$" + " (9)", C["out"], fs=6.0)
    poly(ax, [(x, yc), (106.6, yc), (106.6, 66.0), (xt, 66.0)])
    poly(ax, [(106.6, yc), (106.6, 58.0), (xt, 58.0)])
    dot(ax, 106.6, yc)
    op(ax, xo, 66.0, "+")
    arrow(ax, (xt + 7.0, 66.0), (xo - 0.9, 66.0))
    # input skip x: dashed wire from the input branch, above every skip connection, into the top of (+)
    dot(ax, 24.75, 65.5)
    poly(ax, [(24.75, 65.5), (24.75, 75.4), (xo, 75.4), (xo, 66.9)], ls=DASH, lw=0.75)
    ax.text(108.0, 74.9, r"input skip $x$ (identity)", fontsize=5.6, ha="center", va="top", style="italic")
    # notes: left block ends before the funnel, right block between the funnel and the NIG wire
    ax.text(27.0, 51.0, "↓ stride-2 conv 2×2   ↑ conv 1×1 + PixelShuffle", fontsize=5.6, va="center")
    ax.text(27.0, 48.5, f"parameters: encoder {PARAMS['enc']}, SS-PGB {PARAMS['ss_pgb']},", fontsize=5.6, va="center")
    ax.text(27.0, 46.0, f"DGT {PARAMS['dgt']}, global head {PARAMS['gphys']}, decoder {PARAMS['dec']}", fontsize=5.6,
            va="center")
    ax.text(xpgb + 9.5, 51.0, "- - - skip (+)    heads: conv 3×3", fontsize=5.6, va="center")
    ax.text(xpgb + 9.5, 48.5, f"heads {PARAMS['tails']};  total {PARAMS['total']}", fontsize=5.6, va="center")

    # ---------------- (c) SS-PGB detail: parameter lane (top) and feature lane (bottom); no crossings ----------------
    bg = C["pgb"]
    panel(ax, 1, 1, 78.5, 41.5, "(c) SS-PGB with depth-grounded transmission", bg)
    yl, yh = 14.25, 5.75                          # LL lane and H lane heights
    box(ax, 3.0, 3.0, 6.5, 16.5, "Layer-\nNorm\n+\nHaar\nDWT", C["dwt"], fs=6.0)
    box(ax, 21.0, 11.0, 30.0, 6.5, r"$LL'=G\odot((1+\gamma)\odot LL+\delta)$" + "\n" + r"$+\,(1-G)\odot\mathrm{SA}(LL)$",
        C["block"], fs=5.5)
    box(ax, 21.0, 2.5, 30.0, 6.5, r"$H'=H+\mathrm{DW}(H\odot g_h)$" + "\n"
        + r"$g_h=1+\lambda\,\mathrm{softplus}(\mathrm{conv}(1-t^{D}))$", C["block"], fs=5.9)
    arrow(ax, (9.5, yl), (21.0, yl)); ax.text(15.3, yl + 0.5, r"$LL$ (H/16)", fontsize=5.6, ha="center")
    arrow(ax, (9.5, yh), (21.0, yh)); ax.text(15.25, yh + 0.5, r"$H=[LH,HL,HH]$", fontsize=5.3, ha="center")
    box(ax, 66.0, 2.5, 11.5, 15.0, "IDWT\n" + r"$x_{in}+s\cdot y$" + "\n(residual,\n" + r"$s$ zero-init)",
        C["dwt"], fs=6.0)
    arrow(ax, (51.0, yl), (66.0, yl))
    arrow(ax, (51.0, 4.5), (66.0, 4.5))
    # parameter lane
    poly(ax, [(12.0, yl), (12.0, 35.0), (14.0, 35.0)])
    poly(ax, [(12.0, 25.5), (14.0, 25.5)])
    dot(ax, 12.0, yl); dot(ax, 12.0, 25.5)
    box(ax, 14.0, 31.5, 13.0, 7.0, "DGT head\nmean‖max pool → MLP\n1024→128→9", C["phys"], fs=5.5)
    box(ax, 14.0, 22.5, 13.0, 6.0, "grouped field\nbranch (per colour)\n" + r"$\rightarrow B^{low}_c$", C["phys"], fs=5.8)
    box(ax, 30.0, 33.0, 26.0, 6.5, r"$\beta^{D}_c,\ \beta^{B}_c$" + ST + "  two ordered chains\n" + beta_label,
        C["block"], fs=5.7 if "\n" not in beta_label else 5.3)
    box(ax, 30.0, 26.8, 11.5, 4.4, r"$B^{\infty}_c=\sigma(\cdot)$" + ST, C["block"], fs=6.0)
    box(ax, 44.0, 26.8, 12.0, 4.4, r"$t^{D,B}_c=e^{-\beta^{D,B}_c\,z}$", C["block"], fs=6.0)
    box(ax, 58.0, 33.5, 9.5, 5.0, r"$z$ from (a)," + "\npooled to H/16", "white", fs=5.6, ec=C["z"], color=C["z"])
    arrow(ax, (27.0, 36.25), (30.0, 36.25))
    poly(ax, [(27.0, 33.0), (28.5, 33.0), (28.5, 29.0), (30.0, 29.0)])
    arrow(ax, (50.0, 33.0), (50.0, 31.2))
    poly(ax, [(62.75, 33.5), (62.75, 30.2), (56.0, 30.2)], color=C["z"])
    box(ax, 30.0, 21.0, 26.0, 4.0, r"$\hat B=\mathrm{clip}(B^{\infty}\odot B^{low}+\mathrm{beam},\,0,\,1)$", C["block"],
        fs=5.8)
    arrow(ax, (35.75, 26.8), (35.75, 25.0))
    arrow(ax, (27.0, 23.0), (30.0, 23.0))
    ax.text(47.5, 25.9, "beam: global head on LL", fontsize=5.2, ha="center", va="center", style="italic")
    box(ax, 69.0, 22.0, 9.0, 15.5, r"$\Pi=$" + "\n" + r"$[t^{D},t^{B},\hat B]$" + "\n9 maps\nH/16", C["phys"], fs=6.0)
    arrow(ax, (56.0, 28.0), (69.0, 28.0))
    arrow(ax, (56.0, 23.0), (69.0, 23.0))
    # Pi -> conditioning box (directly beneath Pi) -> straight down into LL'
    box(ax, 49.5, 17.9, 20.5, 2.4, r"$G=\sigma(\mathrm{conv}[LL,\Pi])$,  $(\gamma,\delta)=\mathrm{conv}(\Pi)$", "white",
        fs=5.3)
    poly(ax, [(73.5, 22.0), (73.5, 19.1), (70.0, 19.1)])
    poly(ax, [(49.5, 19.1), (35.25, 19.1), (35.25, 17.5)])
    # Pi -> high-band gain: around the right edge and along the bottom into H' (crosses nothing)
    poly(ax, [(78.0, 30.0), (78.8, 30.0), (78.8, 1.6), (35.25, 1.6), (35.25, 2.5)], lw=0.8)
    ax.text(72.0, 0.95, r"$t^{D}$ to the high-band gain", fontsize=5.0, ha="center", va="top", style="italic") if False else None

    # ---------------- (d) outputs ----------------
    panel(ax, 81, 1, 38, 41.5, "(d) Outputs and training objective", C["out"])
    img(fig, ax, g("tD"), 83.0, 19.5, 9.5, 9.5, label=r"$t^{D}$ (R,G,B)")
    img(fig, ax, g("sigma"), 94.5, 19.5, 9.5, 9.5, cmap="viridis", label=r"predictive std $\sigma$")
    img(fig, ax, g("mu"), 106.0, 19.5, 9.5, 9.5, label=r"restored $\hat J=\mu=x+R$", fs=5.9)
    # mu = x + R from (+): straight down the right column, then into the restored thumbnail
    poly(ax, [(xo, 65.1), (xo, 33.0), (112.5, 33.0), (112.5, 29.0)])
    ax.text(xo - 0.6, 50.0, r"$\mu=\hat J$ (3 ch)", fontsize=5.8, ha="right", va="center")   # output after (+)
    # NIG head: from its bottom edge down to the sigma thumbnail (left of the mu wire, never crossing it)
    poly(ax, [(109.0, 55.5), (109.0, 31.0), (99.25, 31.0), (99.25, 29.0)])
    ax.text(84.0, 36.5, r"$\sigma^2=\beta(1+\nu)/(\nu(\alpha-1))$  (NIG)", fontsize=5.7, va="center")
    ax.text(84.0, 34.2, r"$\hat J=\mu=x+R$  (direct composition)", fontsize=5.7, va="center")
    ax.text(83.0, 15.2, "training only (not on the inference path):", fontsize=5.9, style="italic", color=C["loss"])
    box(ax, 82.5, 2.5, 35.5, 11.0, r"$\mathcal{L}_{phys}=\|I-M\,(\hat J\odot t^{D}+B\,(1-t^{B}))\|_1$" + "\n"
        + r"$+\,\mathrm{TV}(\log t)+$ coupling" + "\n" + r"$+\,\mathcal{L}_{rec}+\mathcal{L}_{NIG}+$ barrier on $\beta^{D,B}$",
        "white", fs=5.4, ec=C["loss"], ls=DASH, color=C["loss"])

    # legend: two centred rows below all panels (clear margin from every panel border)
    ax.text(60.0, -0.5, r"[frozen]: weights fixed, depth precomputed      $\star$: learned per-image physics parameters"
            r"      $\odot$: element-wise product", fontsize=5.7, ha="center", va="center")
    ax.text(60.0, -2.0, "dashed: skip / identity / training-only      gold bracket: zoom (b) → (c)",
            fontsize=5.7, ha="center", va="center")
    ax.text(60.0, -3.5, "blue: depth prior path", fontsize=5.7, ha="center", va="center", color=C["z"])
    OUT.mkdir(exist_ok=True)
    fig.savefig(f"{out_stem}.pdf"); fig.savefig(f"{out_stem}.png", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta-label", default="ordered", help=f"preset {list(BETA_PRESETS)} or literal text "
                    "(default 'ordered' = architecture only; empirical findings belong in the text, not the diagram)")
    ap.add_argument("--no-thumbs", action="store_true")
    ap.add_argument("--out", default=str(OUT / "fig1_architecture"))
    a = ap.parse_args()
    T = None if a.no_thumbs else thumbs()
    if T:
        print(f"[fig1] thumbnails from validation {T['unit']} (centre crop) | beta_D {np.round(T['beta_D'], 4).tolist()}"
              f" | beta_B {np.round(T['beta_B'], 4).tolist()}")
    draw(T, BETA_PRESETS.get(a.beta_label, a.beta_label), a.out)
    print(f"[fig1] wrote {a.out}.pdf and {a.out}.png")


if __name__ == "__main__":
    main()

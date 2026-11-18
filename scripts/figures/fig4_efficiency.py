"""
Fig. 4: quality-efficiency trade-off. All values read from tracked files (no hard-coded numbers):
  reports/experiments/phase_f/phase_f_test_summary.json   locked single test evaluation: PSNR8, ALL group
  reports/experiments/efficiency/efficiency_summary.json  RTX 3070, fp32, batch 1: FPS at 256x256, parameter counts
Bubble AREA is proportional to the parameter count. P1b is also shown with the frozen Depth-Anything-V2-Small cost
included ('P1b + depth', same PSNR, end-to-end FPS), joined to P1b by a dashed line.
Outputs: figs/fig4_efficiency_pareto.pdf (vector) and .png (300 dpi).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
J = lambda p: json.loads((ROOT / p).read_text(encoding="utf-8"))  # noqa: E731
LABEL = {"P1b_s0": "P1b (ours)", "v2_s0": "v2 (no DGT)", "P1e_s0": "P1e", "fiveaplus": "Five A$^+$",
         "wweuie": "WWE-UIE", "ushape": "U-shape"}
COLOR = {"P1b_s0": "#C0392B", "v2_s0": "#E59866", "P1e_s0": "#F0B27A", "fiveaplus": "#5DADE2", "wweuie": "#2E86C1",
         "ushape": "#1B4F72"}


def main():
    pf = J("reports/experiments/phase_f/phase_f_test_summary.json")["absolute"]
    ef = J("reports/experiments/efficiency/efficiency_summary.json")
    R = ef["results"]
    pts = {k: (R[k]["fps_256"], pf[k]["ALL"]["psnr8"], R[k]["params"]) for k in LABEL}
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix",
                         "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    smax = max(p[2] for p in pts.values())
    area = lambda n: 30 + 900 * n / smax  # noqa: E731   (area proportional to parameters, small floor for visibility)
    for k, (fps, psnr, n) in pts.items():
        top = k == "P1b_s0"                                   # ours drawn last-on-top and fully opaque
        ax.scatter(fps, psnr, s=area(n), color=COLOR[k], alpha=0.95 if top else 0.6, edgecolor="black",
                   linewidth=0.8 if top else 0.5, zorder=6 if top else 3)
        # leader lines leave the label and stop at the bubble edge (shrinkB = bubble radius in points), so no line
        # crosses a bubble; P1b / P1e / v2 labels fan out left-down / up / right, away from WWE-UIE
        off = {"P1b_s0": (0, -24, "center", "top"), "P1e_s0": (6, 18, "left", "bottom"),
               "v2_s0": (24, 2, "left", "center"), "fiveaplus": (0, 9, "center", "bottom"),
               "wweuie": (9, 0, "left", "center"), "ushape": (0, -24, "center", "top")}[k]
        r_pt = float(np.sqrt(area(n) / np.pi))
        ax.annotate(f"{LABEL[k]} ({n / 1e6:.2f} M)" if n >= 1e5 else f"{LABEL[k]} ({n / 1e3:.1f} K)",
                    (fps, psnr), xytext=off[:2], textcoords="offset points", ha=off[2], va=off[3], fontsize=6,
                    fontweight="bold" if top else "normal", zorder=7,
                    arrowprops=dict(arrowstyle="-", lw=0.45, color="0.3", shrinkA=1, shrinkB=r_pt)
                    if k in ("P1b_s0", "P1e_s0", "v2_s0") else None)
    fd = R["P1b_s0 + depth"]["fps_256"]; p1b = pts["P1b_s0"]
    ax.plot([fd, p1b[0]], [p1b[1], p1b[1]], ls=(0, (3, 2)), color=COLOR["P1b_s0"], lw=0.8, zorder=2)
    ax.scatter(fd, p1b[1], s=area(p1b[2]), facecolor="white", edgecolor=COLOR["P1b_s0"], linewidth=0.9, zorder=3)
    ax.annotate("P1b + depth prior\n(end-to-end)", (fd, p1b[1]), xytext=(0, -16), textcoords="offset points",
                ha="center", va="top", fontsize=6)
    ax.text(0.40, 0.47, "filled = restoration network (depth map precomputed)\n"
            "hollow = end-to-end incl. Depth-Anything-V2-Small", transform=ax.transAxes, fontsize=5.5, va="center",
            ha="left", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.6", lw=0.5))
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    ax.xaxis.set_major_locator(FixedLocator([20, 30, 50, 100, 200, 300])); ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(f"Throughput at 256$\\times$256 (FPS, log scale; {ef['gpu'].replace('NVIDIA GeForce ', '')}, fp32, batch 1)",
                  fontsize=6.5)
    ax.set_ylabel("Test PSNR (dB), all 517 images", fontsize=6.5)
    ys = [p[1] for p in pts.values()]
    ax.set_ylim(min(ys) - 1.2, max(ys) + 1.35)
    xs = [p[0] for p in pts.values()] + [fd]
    ax.set_xlim(min(xs) / 1.6, max(xs) * 1.6)
    ax.tick_params(labelsize=6); ax.grid(True, which="both", lw=0.3, alpha=0.5)
    ax.text(0.99, 0.98, "bubble area $\\propto$ parameters", transform=ax.transAxes, ha="right", va="top", fontsize=5.5,
            style="italic")
    fig.tight_layout(pad=0.3)
    out = ROOT / "figs"; out.mkdir(exist_ok=True)
    fig.savefig(out / "fig4_efficiency_pareto.pdf"); fig.savefig(out / "fig4_efficiency_pareto.png", dpi=300); plt.close(fig)
    print("[fig4] points (fps_256, test PSNR ALL, params):", {k: (round(a, 2), round(b, 3), c) for k, (a, b, c) in pts.items()},
          "| P1b+depth fps", fd)


if __name__ == "__main__":
    main()

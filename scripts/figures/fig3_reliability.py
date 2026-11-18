"""
Fig. 3: reliability of the uncertainty intervals (report section S). All values read from
reports/experiments/f3/f3_results.json (calibrated on validation only).
(a) empirical vs nominal pixel coverage on the locked test (517) and zero-shot EUVP-dark (5,550): conformal (P1b_s0 and
    3-seed ensemble) vs uncalibrated NIG Gaussian; (b) interval half-width (grey levels) on test at matched coverage:
    sigma-adaptive conformal vs constant-width conformal.
Outputs: figs/fig3_reliability.pdf (vector) and .png (300 dpi).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R = json.loads((ROOT / "reports/experiments/f3/f3_results.json").read_text(encoding="utf-8"))
LV = ["0.8", "0.9", "0.95"]
X = [float(v) for v in LV]


def cov(pred, s, kind):
    return [R["predictors"][pred]["eval"][s][lv][kind]["coverage"] for lv in LV]


def hw(pred, s, kind):
    return [R["predictors"][pred]["eval"][s][lv][kind]["half_width_x255"] for lv in LV]


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "font.size": 7})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.3})
    a.plot([0.75, 1.0], [0.75, 1.0], color="0.5", lw=0.8, ls=":", label="ideal")
    sty = {"test": "-", "dark": "--"}
    for s, lab in (("test", "locked test"), ("dark", "EUVP-dark (zero-shot)")):
        a.plot(X, cov("P1b_s0", s, "adaptive_conformal"), sty[s], marker="o", ms=3.5, color="#C0392B", lw=1.1,
               label=f"P1b conformal, {lab}")
        a.plot(X, cov("E3_ensemble", s, "adaptive_conformal"), sty[s], marker="s", ms=3.2, color="#1F5FA8", lw=1.1,
               label=f"ensemble conformal, {lab}")
        a.plot(X, cov("P1b_s0", s, "nig_gaussian_uncalibrated"), sty[s], marker="^", ms=3.2, color="0.45", lw=0.9,
               label=f"P1b NIG uncalibrated, {lab}")
    TR = json.loads((ROOT / "reports/experiments/f3/target_recal.json").read_text(encoding="utf-8"))["predictors"]
    for pred, col, mk in (("P1b_s0", "#1E8449", "D"), ("E3_ensemble", "#58D68D", "d")):
        a.plot(X, [TR[pred][lv]["target_recalibrated_adaptive"]["coverage"] for lv in LV], "-.", marker=mk, ms=3.2,
               color=col, lw=1.0)
    a.set_xlabel("nominal coverage"); a.set_ylabel("empirical pixel coverage")
    a.set_xticks(X); a.set_xticklabels([f"{x:.0%}" for x in X]); a.set_xlim(0.77, 0.97); a.set_ylim(0.4, 1.17); a.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    a.grid(True, lw=0.3, alpha=0.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="#C0392B", marker="o", ms=3.5, lw=1.1, label="P1b conformal"),
         Line2D([], [], color="#1F5FA8", marker="s", ms=3.2, lw=1.1, label="ensemble conformal"),
         Line2D([], [], color="0.45", marker="^", ms=3.2, lw=0.9, label="P1b NIG, uncalibrated"),
         Line2D([], [], color="black", ls="-", lw=0.9, label="locked test"),
         Line2D([], [], color="black", ls="--", lw=0.9, label="EUVP-dark (zero-shot)"),
         Line2D([], [], color="0.5", ls=":", lw=0.8, label="ideal"),
         Line2D([], [], color="#1E8449", marker="D", ms=3.2, ls="-.", lw=1.0, label="EUVP-dark, target-recal. (P1b)"),
         Line2D([], [], color="#58D68D", marker="d", ms=3.2, ls="-.", lw=1.0, label="EUVP-dark, target-recal. (ens.)")]
    a.legend(handles=h, fontsize=5.4, loc="upper left", frameon=True, framealpha=0.95, ncol=2, columnspacing=0.8,
             handlelength=1.8)
    a.set_title("(a) coverage: validation-fitted; green = target-recalibrated (exploratory)", fontsize=7.0)
    w = 0.26
    for i, (pred, kind, col, lab) in enumerate((("P1b_s0", "adaptive_conformal", "#C0392B", "P1b, $\\sigma$-adaptive"),
                                                ("E3_ensemble", "adaptive_conformal", "#1F5FA8", "ensemble, $\\sigma$-adaptive"),
                                                ("E3_ensemble", "constant_conformal", "0.6", "constant width"))):
        xs = [k + (i - 1) * w for k in range(3)]
        bars = b.bar(xs, hw(pred, "test", kind), width=w, color=col, edgecolor="black", lw=0.4, label=lab)
        for x_, v in zip(xs, hw(pred, "test", kind)):
            b.text(x_, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=5.4)
    b.set_xticks(range(3)); b.set_xticklabels([f"{x:.0%}" for x in X]); b.set_xlabel("nominal coverage (locked test)")
    b.set_ylabel("mean half-width (grey levels)"); b.grid(True, axis="y", lw=0.3, alpha=0.5)
    b.legend(fontsize=5.4, loc="upper left"); b.set_title("(b) sharpness at matched coverage", fontsize=7.5)
    b.set_ylim(0, max(hw("P1b_s0", "test", "adaptive_conformal")) * 1.18)
    out = ROOT / "figs"; out.mkdir(exist_ok=True)
    fig.savefig(out / "fig3_reliability.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / "fig3_reliability.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    print("[fig3] wrote figs/fig3_reliability.pdf/.png")


if __name__ == "__main__":
    main()

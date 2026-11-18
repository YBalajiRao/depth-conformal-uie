"""
Fig. 2: leakage-controlled evaluation protocol. Every number is read from tracked files (no hard-coded values):
  data/splits/lsui_uieb_all_d6_seed0/split_summary.json      units, components, pHash threshold, split counts
  reports/integrity_audit/overlap_audit_20261113.json         dHash audit: val near-frames, zero-shot sets vs training
  reports/experiments/phase_f/test_near_duplicates.json       test near-duplicates (dHash + pixel PSNR)
  reports/experiments/phase_f/TEST_OPENED.json                single test opening (lock)
  reports/experiments/phase_f/phase_f_test_summary.json       n test, number of frozen models
  reports/experiments/phase_f/P1e_G2_decision.json            Gate G2 outcome
  reports/experiments/phase1/p1b_eval517_summary.json, phase2/pair_P1b_s{1,2}_minus_v2_s{1,2}.json   Gate G1 (val, UIEB)
  reports/experiments/phase_e/phase_e_summary.json            zero-shot set sizes
Outputs: figs/fig2_protocol.pdf (vector) and .png (300 dpi).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
J = lambda p: json.loads((ROOT / p).read_text(encoding="utf-8"))  # noqa: E731
EDGE = "#4A4A4A"
C = {"data": "#E3F0FB", "dev": "#EFEFEF", "audit": "#FBE3D6", "test": "#FDE2E2", "zs": "#E8F5E9", "f3": "#FFF4D6"}


def facts():
    sp = J("data/splits/lsui_uieb_all_d6_seed0/split_summary.json")
    au = J("reports/integrity_audit/overlap_audit_20261113.json")["sets"]
    nd = J("reports/experiments/phase_f/test_near_duplicates.json")
    pf = J("reports/experiments/phase_f/phase_f_test_summary.json")
    lock = J("reports/experiments/phase_f/TEST_OPENED.json")
    g2 = J("reports/experiments/phase_f/P1e_G2_decision.json")["result"]
    pe = J("reports/experiments/phase_e/phase_e_summary.json")
    g1 = [J("reports/experiments/phase1/p1b_eval517_summary.json")["paired"]["P1b - v2_full"]["d_psnr8"]["UIEB"]["est"]] + \
         [J(f"reports/experiments/phase2/pair_P1b_s{s}_minus_v2_s{s}.json")["paired"]["d_psnr8"]["UIEB"]["est"] for s in (1, 2)]
    c = sp["counts"]
    return {"units": sp["n_units"], "comp": sp["n_components"], "phash": sp["args"]["max_phash"],
            "lsui": sum(c[s]["LSUI"] for s in c), "uieb": sum(c[s]["UIEB"] for s in c),
            "train": sum(c["train"].values()), "val": sum(c["val"].values()), "test": sum(c["test"].values()),
            "val_near": au["our_val"]["near_by_dataset_pix_ge25dB"]["LSUI"], "val_lsui": c["val"]["LSUI"],
            "test_near": sum(v["near_dup"] for v in nd.values()), "test_lsui": len(nd),
            "dark_dup": au["EUVP_underwater_dark"]["verified_near_pix_ge25dB"], "c60_dup": au["UIEB_challenge_60"]["verified_near_pix_ge25dB"],
            "n_dark": pe["n"]["dark"], "n_c60": pe["n"]["c60"], "n_models": len(pf["models"]), "n_test": pf["n"],
            "opened": lock["opened"].replace("T", " ")[:16], "g1": g1, "g2_winner": g2["gate_G2"]["winner"],
            "g2_all": g2["pairs"]["P1b_s0"]["ALL"]["est"]}


HEAD = {"data": "#B9D7EA", "dev": "#D5D5D5", "audit": "#F4C7AE", "test": "#F6B9B9", "zs": "#BFE3C2", "f3": "#F7DE9C"}


def bnum(s):
    """Bold key number in mathtext (commas protected)."""
    return r"$\mathbf{" + str(s).replace(",", "{,}") + "}$"


def card(ax, x, y, w, h, title, body, kind, lw=0.8, hh=3.4, fs=7.0):
    """Stage card: tinted header strip with the numbered title, body text filling the remaining height."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9", fc=C[kind], ec=EDGE, lw=lw, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y + h - hh), w, hh, boxstyle="round,pad=0,rounding_size=0.9", fc=HEAD[kind], ec="none",
                                zorder=2.1))
    ax.add_patch(plt.Rectangle((x + 0.05, y + h - hh), w - 0.1, hh / 2, fc=HEAD[kind], ec="none", zorder=2.15))  # square bottom
    ax.plot([x, x + w], [y + h - hh, y + h - hh], color=EDGE, lw=0.5, zorder=2.2)
    ax.text(x + 1.2, y + h - hh / 2, title, ha="left", va="center", fontsize=7.4, fontweight="bold", zorder=3)
    ax.text(x + w / 2, y + (h - hh) / 2, body, ha="center", va="center", fontsize=fs, zorder=3, linespacing=1.3,
            color="#111111")


def arrow(ax, p, q, ls="-", color=EDGE):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>,head_length=3.0,head_width=1.7", lw=0.9, ls=ls, color=color,
                                 shrinkA=0, shrinkB=0, zorder=4))


def badge(ax, x, y, s):
    ax.text(x, y, s, fontsize=5.8, va="center", ha="left", style="italic", zorder=7,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))


def main():
    F = facts()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix",
                         "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.2, 2.835)); ax = fig.add_axes([0, 0, 1, 1])     # 183 mm x 72 mm
    ax.set_xlim(0, 120); ax.set_ylim(0, 47.25); ax.axis("off")
    W, H = 37.0, 12.6
    xs = (1.0, 41.5, 82.0)
    r1, r2, r3 = 33.9, 17.6, 1.0
    g1 = " / ".join(bnum(f"{v:+.2f}") for v in F["g1"])
    # row 1: data
    card(ax, xs[0], r1, W, H, "1  Dataset pool",
         f"LSUI {bnum(format(F['lsui'], ','))} + UIEB {bnum(format(F['uieb'], ','))} paired images\n"
         f"= {bnum(format(F['units'], ','))} units", "data")
    card(ax, xs[1], r1, W, H, "2  Duplicate grouping",
         f"pHash distance $\\leq$ {F['phash']} duplicate graph\n"
         f"{bnum(format(F['units'], ','))} units $\\rightarrow$ {bnum(format(F['comp'], ','))} components", "data")
    card(ax, xs[2], r1, W, H, "3  Component-disjoint split (80/10/10)",
         f"train / val / test = {bnum(format(F['train'], ','))} / {bnum(F['val'])} / {bnum(F['test'])}\n"
         "near-copies never straddle splits; test locked", "data")
    # row 2: development on train/val only
    card(ax, xs[0], r2, W, H, "4  Post-split leakage audit",
         "dHash $\\leq$ 8 + pixel-PSNR $\\geq$ 25 dB vs. training\n"
         f"val {bnum(F['val_near'])}/{F['val_lsui']}  and  test {bnum(F['test_near'])}/{F['test_lsui']} LSUI near-frames\n"
         "$\\rightarrow$ near-duplicate-filtered subset", "audit", fs=6.6)
    card(ax, xs[1], r2, W, H, "5  Validation-only decisions",
         f"Gate G1 (P1b vs v2, UIEB, 3 seeds):\n{g1} dB $\\rightarrow$ passed\n"
         f"Gate G2 (P1e vs P1b): {bnum(format(F['g2_all'], '+.2f'))} dB $\\rightarrow$ {F['g2_winner']} kept", "dev", fs=6.6)
    card(ax, xs[2], r2, W, H, "6  Pre-registered F3",
         "split-conformal NIG intervals fitted on\nvalidation only (80 / 90 / 95 %); coverage on 7 and 8", "f3", fs=6.8)
    # row 3: held-out evaluation
    card(ax, 13.0, r3, 45.0, H, "7  Single locked test evaluation",
         f"{bnum(F['n_test'])} images, {bnum(F['n_models'])} SHA-pinned frozen models, opened once\n"
         f"({F['opened']}, lock file); full + near-duplicate-filtered", "test", lw=1.1, fs=6.8)
    card(ax, 63.0, r3, 45.0, H, "8  Zero-shot benchmarks (never trained on)",
         f"EUVP underwater_dark {bnum(format(F['n_dark'], ','))} pairs + Challenge-60 {bnum(F['n_c60'])} images\n"
         f"verified near-duplicates vs. training: {F['dark_dup']} / {F['c60_dup']}", "zs", fs=6.8)
    # flow
    line = lambda pts: ax.plot([p[0] for p in pts], [p[1] for p in pts], color=EDGE, lw=0.9, zorder=4)  # noqa: E731
    for a, b in ((0, 1), (1, 2)):
        arrow(ax, (xs[a] + W, r1 + H / 2 - 1.7), (xs[b], r1 + H / 2 - 1.7))
    ybus = (r2 + H + r1) / 2
    line([(xs[2] + W / 2, r1), (xs[2] + W / 2, ybus), (xs[0] + W / 2, ybus)])
    ax.plot(xs[2] + W / 2, ybus, "o", ms=2.2, color=EDGE, zorder=5)
    for xx in (xs[0] + W / 2, xs[1] + W / 2, xs[2] + W / 2):
        arrow(ax, (xx, ybus), (xx, r2 + H))
    badge(ax, xs[1] + W / 2 + 1.2, ybus, "train / val only")
    arrow(ax, (xs[1] + W, r2 + H / 2 - 1.7), (xs[2], r2 + H / 2 - 1.7))
    arrow(ax, (xs[0] + W / 2, r2), (xs[0] + W / 2, r3 + H))
    yf = (r3 + H + r2) / 2
    line([(xs[1] + W / 2, r2), (xs[1] + W / 2, yf)])
    line([(48.0, yf), (72.0, yf)])
    ax.plot(xs[1] + W / 2, yf, "o", ms=2.2, color=EDGE, zorder=5)
    arrow(ax, (48.0, yf), (48.0, r3 + H)); arrow(ax, (72.0, yf), (72.0, r3 + H))
    badge(ax, xs[1] + W / 2 + 1.2, yf, "frozen models")
    arrow(ax, (xs[2] + W / 2, r2), (xs[2] + W / 2, r3 + H)) if xs[2] + W / 2 < 108 else None
    line([(xs[2] + W / 2, r2), (xs[2] + W / 2, yf), (100.0, yf)]) if xs[2] + W / 2 >= 108 else None
    if xs[2] + W / 2 >= 108:
        arrow(ax, (100.0, yf), (100.0, r3 + H))
    out = ROOT / "figs"; out.mkdir(exist_ok=True)
    fig.savefig(out / "fig2_protocol.pdf"); fig.savefig(out / "fig2_protocol.png", dpi=300); plt.close(fig)
    print("[fig2] wrote figs/fig2_protocol.pdf/.png |", {k: F[k] for k in ("units", "comp", "train", "val", "test", "test_near", "n_models", "g1", "g2_winner")})


if __name__ == "__main__":
    main()

"""
Build docs/master_tables.md from TRACKED artifacts only (reports/experiments/**, reports/integrity_audit/**).
No number is typed by hand; every cell is read from a file named in the table caption. Missing metrics are printed as
'n/c' (not computed) rather than estimated.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "reports/experiments"
J = lambda p: json.loads((E / p).read_text(encoding="utf-8"))  # noqa: E731
f2, f3, f4 = (lambda v: f"{v:.2f}"), (lambda v: f"{v:.3f}"), (lambda v: f"{v:.4f}")
ci = lambda c, f=f2: f"{c['est']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}]" if f is f2 else f"{c['est']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"  # noqa: E731


def t1(L):
    pf = J("phase_f/phase_f_test_summary.json"); ab, pr = pf["absolute"], pf["paired_model_minus_P1b_s0"]
    ef = J("efficiency/efficiency_summary.json")["results"]; dv = J("f3/derived.json")
    params = {k: ef[k]["params"] for k in ef if "params" in ef[k]}; params["v1_retrain"] = dv["v1_retrain_params"]
    fps = {k: ef[k].get("fps_256") for k in ef}
    L += ["## Table 1. Locked in-domain test (single opening, n = 517; dedup n = 493)",
          "Source: `phase_f/phase_f_test_summary.json` (report §R), `efficiency/efficiency_summary.json` (RTX 3070, fp32, batch 1).",
          "PSNR8 / SSIM8 / LPIPS (alex) means; Δ = model − P1b_s0 on ALL, paired bootstrap 95% CI (B = 2000).", "",
          "| Model | Params | FPS@256 | PSNR ALL | PSNR dedup (493) | SSIM ALL | LPIPS ALL | UIEB PSNR | Δ PSNR vs P1b (ALL) |",
          "|---|---|---|---|---|---|---|---|---|"]
    rows = ["P1b_s0", "P1e_s0", "v2_s0", "v1_retrain", "wweuie", "ushape", "fiveaplus"]
    for k in rows:
        a = ab[k]; d = "—" if k == "P1b_s0" else ci(pr[k]["ALL"]["psnr8"])
        fp = f"{fps[k]:.1f}" if fps.get(k) else "n/c"
        if k == "P1b_s0":
            fp += f" ({ef['P1b_s0 + depth']['fps_256']:.1f} end-to-end)"
        L.append(f"| {k} | {params[k] / 1e6:.3f} M | {fp} | {f3(a['ALL']['psnr8'])} | {f3(a['ALL_dedup']['psnr8'])} | "
                 f"{f4(a['ALL']['ssim8'])} | {f4(a['ALL']['lpips'])} | {f3(a['UIEB']['psnr8'])} | {d} |")
    L += ["", "**3-seed statistics (test, ALL PSNR8):**", ""]
    for fam in ("P1b", "v2"):
        v = [ab[f"{fam}_s{s}"]["ALL"]["psnr8"] for s in range(3)]
        L.append(f"- {fam}: {', '.join(f3(x) for x in v)} → mean {np.mean(v):.3f} ± {np.std(v, ddof=1):.3f} (sd)")
    sp = pf["P1b_minus_v2_seed_matched_psnr8"]
    L.append("- P1b − v2 seed-matched, UIEB: " + "; ".join(f"s{s} {ci(sp[f's{s}']['UIEB'])}" for s in range(3)) + " (Gate G1 not replicated on test)")
    L += ["- CIEDE2000 (ΔE00) on the locked test: **omitted by design** — computing it would require a further read of the",
          "  test split, which the single-read lock (report §Q/§S) forbids. ΔE00 is reported on the zero-shot set (Table 2).", ""]


def t2(L):
    pe = J("phase_e/phase_e_summary.json"); p2 = J("phase_e_pretrained2/summary.json")
    f3r = J("f3/f3_results.json")["ensemble_accuracy"]["dark"]
    sm = J("saved_metrics/summary.json") if (E / "saved_metrics/summary.json").exists() else {"c60": {}, "dark200_ciede2000": {}}
    nr = lambda key, m: (f"{sm['c60'][key][m]:.3f}" if key in sm["c60"] else "n/c")  # noqa: E731
    de = lambda key: (f"{sm['dark200_ciede2000'][key]['ciede2000']:.2f}" if key in sm["dark200_ciede2000"] else "n/c")  # noqa: E731
    L += ["## Table 2. Zero-shot benchmarks (never trained on)",
          "Sources: `phase_e/phase_e_summary.json`, `phase_e_pretrained2/summary.json`, `f3/f3_results.json` (ensemble),",
          "`saved_metrics/summary.json` (MUSIQ, URanker on C60; ΔE00 on the saved 200-image EUVP-dark subset, rng 20261117).",
          "EUVP underwater_dark: 5,550 pairs (full-reference); Challenge-60: 60 images (no-reference). Pretrained = official",
          "checkpoints with their official test protocols; Δ = P1b_s0 − model, paired 95% CI. ↑ higher better, ↓ lower better.", "",
          "| Model | Training data | dark PSNR↑ | dark SSIM↑ | dark LPIPS↓ | dark ΔE00↓ (200) | C60 UIQM↑ | C60 UCIQE↑ | C60 MUSIQ↑ | C60 URanker↑ | Δ PSNR (P1b − model) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    ad, ac = pe["absolute_dark"], pe["absolute_c60"]
    L.append(f"| Unenhanced input | — | — | — | — | {de('input')} | {f3(ac['input']['uiqm'])} | {f4(ac['input']['uciqe'])} | {nr('input', 'musiq')} | {nr('input', 'uranker')} | — |")
    L.append(f"| P1b_s0 (ours) | LSUI+UIEB train split | {f3(ad['P1b_s0']['psnr8'])} | {f4(ad['P1b_s0']['ssim8'])} | {f4(ad['P1b_s0']['lpips'])} | {de('P1b_s0')} | {f3(ac['P1b_s0']['uiqm'])} | {f4(ac['P1b_s0']['uciqe'])} | {nr('P1b_s0', 'musiq')} | {nr('P1b_s0', 'uranker')} | — |")
    L.append(f"| P1b 3-seed ensemble | same | {f3(f3r['E3_psnr8'])} | n/c | n/c | n/c | n/c | n/c | n/c | n/c | {f3r['E3_minus_s0_psnr8']['est']:+.3f} (E3 − s0) |")
    for k, lab, sk in (("wweuie", "WWE-UIE (retrained)", "wweuie_retrained"), ("ushape", "U-shape (retrained)", "ushape_retrained"),
                       ("fiveaplus", "Five A+ (retrained)", "fiveaplus_retrained")):
        d = pe["paired_P1b_minus"][k]["dark"]["psnr8"]
        L.append(f"| {lab} | same split | {f3(ad[k]['psnr8'])} | {f4(ad[k]['ssim8'])} | {f4(ad[k]['lpips'])} | {de(sk)} | {f3(ac[k]['uiqm'])} | {f4(ac[k]['uciqe'])} | {nr(sk, 'musiq')} | {nr(sk, 'uranker')} | {ci(d)} |")
    train = {"MARINE-Net-LSUI": "EUVP-scenes → LSUI ft", "FiveA+_official_a0.1": "UIEB 800",
             "PhaseFormer-UIEB": "UIEB", "PhaseFormer-UFO120": "UFO-120 (C60 only)", "UIR-PolyKernel": "undisclosed (C60 only)"}
    for k, tr in train.items():
        dk = p2["absolute_dark"].get(k); c = p2["absolute_c60"][k]; pd = p2["paired_P1b_minus"].get(k, {}).get("dark")
        L.append(f"| {k} (pretrained) | {tr} | {f3(dk['psnr8']) if dk else '—'} | {f4(dk['ssim8']) if dk else '—'} | "
                 f"{f4(dk['lpips']) if dk else '—'} | {de(k) if dk else '—'} | {f3(c['uiqm'])} | {f4(c['uciqe'])} | "
                 f"{nr(k, 'musiq')} | {nr(k, 'uranker')} | {ci(pd['psnr8']) if pd else '—'} |")
    a4 = p2["absolute_dark"]["FiveA+_official_a0.4"]
    L += ["", "- **Dropped from the main table → Supplementary:** Five A+ official α = 0.4 (EUVP-dark PSNR "
          f"{a4['psnr8']:.3f}, SSIM {a4['ssim8']:.4f}). Protocol verified on 2026-11-18: the adapter already runs at native",
          "  resolution without padding (multiple = 1), and the repo's own forward path reproduces our saved outputs within",
          "  1 grey level (official-path PSNR 11.7–14.3 dB on 5 images), so the low score is the checkpoint's behaviour, not an",
          "  inference mismatch.",
          "- Ensemble no-reference metrics and ΔE00: **n/c** (ensemble outputs were not saved; regenerating them needs the GPU",
          "  depth pass). ΔE00 on the full 5,550 and on validation: **n/c** (no saved outputs).", ""]


def t3(L):
    pf = J("phase_f/phase_f_test_summary.json"); ab, pr = pf["absolute"], pf["paired_model_minus_P1b_s0"]
    b = J("phase_e_beta/summary.json"); pe = J("phase_e/phase_e_summary.json")["p1b_uncertainty"]; dv = J("f3/derived.json")
    g1 = [J("phase1/p1b_eval517_summary.json")["paired"]["P1b - v2_full"]["d_psnr8"]["UIEB"]] + \
         [J(f"phase2/pair_P1b_s{s}_minus_v2_s{s}.json")["paired"]["d_psnr8"]["UIEB"] for s in (1, 2)]
    g2 = J("phase_f/P1e_G2_decision.json")["result"]
    L += ["## Table 3. Ablations, gates, learned attenuation, OOD uncertainty",
          "Sources: `phase_f/phase_f_test_summary.json`, `phase1/`, `phase2/`, `phase_f/P1e_G2_decision.json`, `phase_e_beta/summary.json`,",
          "`phase_e/phase_e_summary.json`, `f3/derived.json`.", "",
          "**(a) Component ablations on the locked test (Δ PSNR8 vs P1b_s0, ALL and UIEB):**", "",
          "| Model | PSNR ALL | Δ ALL | Δ UIEB |", "|---|---|---|---|"]
    for k in ("v2_s0", "v1_retrain", "A2_no_physics", "A3_v2_no_Lphys", "A4_v2_no_NIG", "P1e_s0"):
        L.append(f"| {k} | {f3(ab[k]['ALL']['psnr8'])} | {ci(pr[k]['ALL']['psnr8'])} | {ci(pr[k]['UIEB']['psnr8'])} |")
    L += ["", "**(b) Gates (validation, 517):**", "",
          "- G1 (P1b − v2, UIEB, seeds 0/1/2): " + "; ".join(ci(c) for c in g1) + " → passed",
          f"- G2 (P1e − P1b_s0): ALL {ci(g2['pairs']['P1b_s0']['ALL'])}, UIEB {ci(g2['pairs']['P1b_s0']['UIEB'])} → {g2['gate_G2']['winner']} kept", "",
          "**(c) Learned per-image attenuation of P1b_s0 (mean ± sd over images; median relative spread (R−B)/B):**", "",
          "| Set | β^D (R, G, B) | spread β^D | β^B (R, G, B) | spread β^B |", "|---|---|---|---|---|"]
    for s in ("val517", "dark", "c60"):
        d, bb = b[s]["beta_D"], b[s]["beta_B"]
        fm = lambda x: ", ".join(f"{m:.3f}±{v:.3f}" for m, v in zip(x["mean"], x["sd"]))  # noqa: E731
        L.append(f"| {s} (n={b[s]['n']}) | {fm(d)} | {100 * d['rel_spread_RB_median']:.2f}% | {fm(bb)} | {100 * bb['rel_spread_RB_median']:.2f}% |")
    L += ["", "**(d) Uncertainty scaling relative to validation:**", "",
          "| Set | epistemic std ratio (Phase E) | total σ ratio P1b_s0 (F3) | total σ ratio ensemble (F3) |", "|---|---|---|---|",
          f"| validation | 1.00 | 1.00 | 1.00 |",
          f"| locked test | n/c | {dv['sigma_ratio_P1b_s0_test_over_val']:.3f} | {dv['sigma_ratio_E3_test_over_val']:.3f} |",
          f"| EUVP-dark | {pe['dark']['epistemic_vs_val517']:.3f} | {dv['sigma_ratio_P1b_s0_dark_over_val']:.3f} | {dv['sigma_ratio_E3_dark_over_val']:.3f} |",
          f"| Challenge-60 | {pe['c60']['epistemic_vs_val517']:.3f} | n/c | n/c |", ""]


def t4(L):
    r = J("f3/f3_results.json")
    L += ["## Table 4. Split-conformal calibration (fitted on validation only; report §S)",
          "Source: `f3/f3_results.json`. Pixel coverage; half-width in grey levels (×255); worst-5% = 5th percentile of per-image coverage.", ""]
    for name, P in r["predictors"].items():
        L += [f"**{name}**", "", "| Set | Level | Conformal cov. | Half-width | Constant-width cov. | Const. half-width | NIG uncalibrated cov. | Worst-5% image cov. |",
              "|---|---|---|---|---|---|---|---|"]
        tr = J("f3/target_recal.json")["predictors"][name]
        for s in ("test", "dark"):
            for lv in ("0.8", "0.9", "0.95"):
                e = P["eval"][s][lv]; a, c, n = e["adaptive_conformal"], e["constant_conformal"], e["nig_gaussian_uncalibrated"]
                L.append(f"| {s} | {float(lv):.0%} | {a['coverage']:.4f} | {a['half_width_x255']:.2f} | {c['coverage']:.4f} | "
                         f"{c['half_width_x255']:.2f} | {n['coverage']:.4f} | {a['per_image_p05']:.3f} |")
        for lv in ("0.8", "0.9", "0.95"):                    # target-domain recalibration (exploratory; 500 cal / 5,050 eval)
            t = tr[lv]; a, c = t["target_recalibrated_adaptive"], t["target_recalibrated_constant"]
            L.append(f"| dark, target-recalibrated† | {float(lv):.0%} | {a['coverage']:.4f} | {a['half_width_x255']:.2f} | "
                     f"{c['coverage']:.4f} | {c['half_width_x255']:.2f} | — | {a['per_image_p05']:.3f} |")
        L += ["", "Spearman ρ(σ, |residual|): " + ", ".join(f"{s} {P['eval'][s]['spearman_sigma_vs_abs_residual']:.3f}"
                                                     for s in ("val", "test", "dark")), ""]
    tg = J("f3/target_recal.json")
    L += [f"† EXPLORATORY (not pre-registered): conformal quantiles re-fitted on {tg['n_cal']} EUVP-dark images (~9%, rng "
          f"{tg['split_seed']}) and evaluated on the disjoint {tg['n_eval']} (`f3/target_recal.json`). Coverage is restored at the",
          "cost of much wider intervals; in this domain a constant-width band is sharper than the σ-scaled one.", ""]
    ea = r["ensemble_accuracy"]
    L += ["**Ensemble accuracy (E3 − P1b_s0, PSNR8):** " + "; ".join(f"{k}: {ci(v['E3_minus_s0_psnr8'], f3)}"
                                                               for k, v in ea.items()) + " (test row post-hoc).", ""]


def t5(L):
    d = J("downstream/summary.json")
    L += ["## Table 5. Downstream feature-level check (OpenCV, CPU)",
          "Source: `downstream/summary.json`. SIFT/ORB: keypoints and RANSAC inliers against a fixed synthetic homography of the",
          "same image; Canny (100, 200): edge density (%) and mean connected edge length (px). Descriptive proxies, no task accuracy.", ""]
    for s, lab in (("c60", "Challenge-60 (60)"), ("dark", "EUVP-dark subset (200)")):
        L += [f"**{lab}**", "", "| Method | SIFT kp | SIFT inliers | ORB kp | ORB inliers | Edge density % | Mean edge length |",
              "|---|---|---|---|---|---|---|"]
        for m, v in sorted(d[s].items(), key=lambda kv: (kv[0] != "input", kv[0])):
            L.append(f"| {m} | {v['sift_kp']:.1f} | {v['sift_inliers']:.1f} | {v['orb_kp']:.1f} | {v['orb_inliers']:.1f} | "
                     f"{v['edge_density_pct']:.2f} | {v['edge_mean_len_px']:.2f} |")
        L.append("")


def main():
    L = ["# Master tables (auto-generated by `scripts/tables/build_master_tables.py`; do not edit by hand)", "",
         "`n/c` = not computed (never estimated).", ""]
    for f in (t1, t2, t3, t4, t5):
        f(L)
    out = ROOT / "docs/master_tables.md"; out.write_text("\n".join(L) + "\n", encoding="utf-8"); print("wrote", out)


if __name__ == "__main__":
    main()

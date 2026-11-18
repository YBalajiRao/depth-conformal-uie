"""
Metrics on ALREADY-SAVED model outputs only (CPU; no model is run, no test-split read). Added 2026-11-18, EXPLORATORY.

  * Challenge-60 (60 images): MUSIQ (pyiqa 'musiq', KonIQ) and URanker (pyiqa 'uranker'), higher = better, on the
    outputs saved by src/diagnostics/phase_e_pretrained2.py (incl. P1b_s0, retrained baselines, pretrained models, input).
  * EUVP underwater_dark: CIEDE2000 (skimage deltaE_ciede2000 on sRGB->Lab, mean over pixels) on the saved 200-image subset
    (rng 20261117), against trainB references. The full 5,550 set and validation have no saved outputs (n/c).
Output: runs/diagnostics/SAVED_OUTPUT_METRICS_<ts>/summary.json (+ per-image CSV)
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab

ROOT = Path(__file__).resolve().parents[2]
SRC = sorted((ROOT / "runs/diagnostics").glob("PHASE_E_PRETRAINED2_*"))[-1] / "outputs"
REF = ROOT / "data/raw/EUVP/Paired/underwater_dark"


def main():
    import pyiqa
    out = ROOT / "runs/diagnostics" / f"SAVED_OUTPUT_METRICS_{datetime.now():%Y%m%d_%H%M%S}"; out.mkdir(parents=True)
    torch.set_num_threads(8)
    musiq, urk = pyiqa.create_metric("musiq", device="cpu"), pyiqa.create_metric("uranker", device="cpu")
    summ, rows = {"source": str(SRC), "c60": {}, "dark200_ciede2000": {}}, []
    for md in sorted(p for p in (SRC / "c60").iterdir() if p.is_dir()):
        mu, ur = [], []
        for f in sorted(md.glob("*.png")):
            with torch.no_grad():
                mu.append(float(musiq(str(f)))); ur.append(float(urk(str(f))))
            rows.append({"set": "c60", "method": md.name, "image": f.name, "musiq": mu[-1], "uranker": ur[-1]})
        summ["c60"][md.name] = {"musiq": float(np.mean(mu)), "uranker": float(np.mean(ur)), "n": len(mu)}
        print("c60", md.name, summ["c60"][md.name], flush=True)
    refs = {p.stem: p for p in (REF / "trainB").iterdir()}
    for md in sorted(p for p in (SRC / "dark").iterdir() if p.is_dir()):
        de = []
        for f in sorted(md.glob("*.png")):
            o = np.asarray(Image.open(f).convert("RGB"), dtype=np.float64) / 255
            y = np.asarray(Image.open(refs[f.stem]).convert("RGB"), dtype=np.float64) / 255
            de.append(float(deltaE_ciede2000(rgb2lab(y), rgb2lab(o)).mean()))
            rows.append({"set": "dark200", "method": md.name, "image": f.name, "ciede2000": de[-1]})
        summ["dark200_ciede2000"][md.name] = {"ciede2000": float(np.mean(de)), "n": len(de)}
        print("dark200", md.name, summ["dark200_ciede2000"][md.name], flush=True)
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as fh:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8"); print("wrote", out)


if __name__ == "__main__":
    main()

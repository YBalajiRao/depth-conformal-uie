"""
CPU-only downstream vision check (report section S) on outputs saved by src/diagnostics/phase_e_pretrained2.py.

Per image and method (C60: 60 images; EUVP underwater_dark: 200-image subset, rng 20261117):
  * SIFT and ORB keypoint repeatability: match the image against a fixed synthetic homography of itself
    (rotate 8 deg, scale 0.9, mild perspective), RANSAC (3 px) -> inlier count (higher = more stable, matchable texture)
  * Canny (100, 200 on grey): edge density (% of pixels) and mean connected edge-component length in pixels
    (continuity; 8-connectivity)
Images are processed at native size. Output: runs/diagnostics/DOWNSTREAM_CV_<ts>/summary.json (+ per-image CSV).
These are descriptive feature-level proxies, not task accuracy (no detector/segmenter is run).
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def homography(w, h):
    c = (w / 2, h / 2)
    A = cv2.getRotationMatrix2D(c, 8, 0.9)
    Hm = np.vstack([A, [0, 0, 1]]).astype(np.float64)
    Hm[2, 0], Hm[2, 1] = 1e-4, -5e-5
    return Hm


def inliers(det, norm, img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY); h, w = g.shape
    Hm = homography(w, h); g2 = cv2.warpPerspective(g, Hm, (w, h))
    k1, d1 = det.detectAndCompute(g, None); k2, d2 = det.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        return 0, len(k1 or [])
    m = cv2.BFMatcher(norm, crossCheck=True).match(d1, d2)
    if len(m) < 4:
        return 0, len(k1)
    src = np.float32([k1[x.queryIdx].pt for x in m]); dst = np.float32([k2[x.trainIdx].pt for x in m])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return int(mask.sum()) if mask is not None else 0, len(k1)


def edges(img):
    e = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 100, 200)
    n, lab, st, _ = cv2.connectedComponentsWithStats((e > 0).astype(np.uint8), connectivity=8)
    lens = st[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([0])
    return float((e > 0).mean() * 100), float(lens.mean())


def main():
    src = sorted((ROOT / "runs/diagnostics").glob("PHASE_E_PRETRAINED2_*"))[-1] / "outputs"
    out = ROOT / "runs/diagnostics" / f"DOWNSTREAM_CV_{datetime.now():%Y%m%d_%H%M%S}"; out.mkdir(parents=True)
    sift, orb = cv2.SIFT_create(), cv2.ORB_create(nfeatures=2000)
    summ, rows = {"source": str(src)}, []
    for s in ("c60", "dark"):
        summ[s] = {}
        for md in sorted(p for p in (src / s).iterdir() if p.is_dir()):
            vals = []
            for f in sorted(md.glob("*.png")):
                img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
                si, sk = inliers(sift, cv2.NORM_L2, img); oi, ok = inliers(orb, cv2.NORM_HAMMING, img); ed, el = edges(img)
                r = {"set": s, "method": md.name, "image": f.name, "sift_kp": sk, "sift_inliers": si, "orb_kp": ok,
                     "orb_inliers": oi, "edge_density_pct": ed, "edge_mean_len_px": el}
                vals.append(r); rows.append(r)
            summ[s][md.name] = {k: float(np.mean([v[k] for v in vals])) for k in
                                ("sift_kp", "sift_inliers", "orb_kp", "orb_inliers", "edge_density_pct", "edge_mean_len_px")}
            summ[s][md.name]["n"] = len(vals)
            print(s, md.name, {k: round(v, 2) for k, v in summ[s][md.name].items()}, flush=True)
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8"); print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())

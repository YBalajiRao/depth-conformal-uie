"""
Fit the P1e scene-descriptor PCA on the TRAINING descriptors only (PI decision 2026-11-13, report section P).

Input : <scene-dir>/scene_train.npz (768-d descriptors of the 4,135 training images; precompute_scene_features.py).
Output: <scene-dir>/pca<k>_train.npz with mean (768), components (k x 768, orthonormal rows), scale (k; sqrt of the
        component variances on train, used to whiten), explained_variance_ratio (k), train_npz_sha256, n_train.
Validation/test descriptors are only ever TRANSFORMED with this file (z = ((g - mean) @ components.T) / scale);
nothing is refitted on them. The script refuses to read any file other than scene_train.npz.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "data/scene/da2_small_5426e4f/lsui_uieb_all_d6_seed0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--scene-dir", type=Path, default=DEFAULT)
    a = ap.parse_args()
    src = a.scene_dir / "scene_train.npz"
    with np.load(src, allow_pickle=False) as d:
        G = d["g"].astype(np.float64)
        units = [str(u) for u in d["unit"]]
    if len(units) != 4135 or any(not (u.startswith("LSUI:") or u.startswith("UIEB:")) for u in units):
        raise SystemExit(f"[abort] unexpected training descriptor set ({len(units)} units)")
    mean = G.mean(0)
    U, S, Vt = np.linalg.svd(G - mean, full_matrices=False)
    var = S ** 2 / (len(G) - 1)
    comps, scale = Vt[:a.k], np.sqrt(var[:a.k])
    evr = var[:a.k] / var.sum()
    Z = ((G - mean) @ comps.T) / scale
    out = a.scene_dir / f"pca{a.k}_train.npz"
    np.savez(out, mean=mean.astype(np.float32), components=comps.astype(np.float32), scale=scale.astype(np.float32),
             explained_variance_ratio=evr.astype(np.float32),
             train_npz_sha256=np.array(hashlib.sha256(src.read_bytes()).hexdigest()), n_train=np.array(len(G)))
    info = {"k": a.k, "explained_variance_total": float(evr.sum()), "first5": [round(float(v), 4) for v in evr[:5]],
            "train_whitened_std_min_max": [float(Z.std(0, ddof=1).min()), float(Z.std(0, ddof=1).max())],
            "orthonormal_max_err": float(np.abs(comps @ comps.T - np.eye(a.k)).max()), "out": str(out)}
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()

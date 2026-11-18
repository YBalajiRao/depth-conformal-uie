"""
Offline global scene descriptors from the frozen Depth-Anything-V2-Small encoder (roadmap phase C; PI decision
2026-11-13). Same pinned model, revision and weights SHA-256 as precompute_depth.py; same image loading; train/val
only (test refused); fp16 inference_mode.

For each image (the degraded input, full image, the processor's standard 518-px resize keeping aspect, multiple of
14), the DINOv2 backbone's last feature map (layer 12, layer-normed; tokens = 1 CLS + patch tokens, 384-d) is reduced
to a 768-d descriptor g = [CLS token, mean of patch tokens]. g is a whole-image context vector: training crops
receive the descriptor of their full image, so the restorer sees global scene context it cannot see in a 256 crop.

Output: <out>/<split-dir name>/scene_<split>.npz with arrays unit (str), g (N x 768 float32), input_sha256 (str),
plus run_meta_<stamp>.json (model, revision, weights SHA, commit, counts).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/validation"))
sys.path.insert(0, str(ROOT / "src/data"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from calibrate_uncertainty import read_manifest  # noqa: E402
from evaluate_metrics import load_u8  # noqa: E402
import precompute_depth as PD  # noqa: E402

DEFAULT_OUT = ROOT / "data/scene/da2_small_5426e4f"


@torch.inference_mode()
def descriptor(proc, model16, x8, device):
    pv = proc(images=x8, return_tensors="pt")["pixel_values"].to(device).half()
    tok = model16.backbone(pixel_values=pv).feature_maps[-1].float()       # (1, 1 + N, 384)
    if tok.dim() != 3 or tok.shape[1] < 2:
        raise RuntimeError(f"unexpected backbone output {tuple(tok.shape)}")
    g = torch.cat([tok[:, 0], tok[:, 1:].mean(1)], 1)[0]                   # (768,)
    if not torch.isfinite(g).all():
        raise RuntimeError("non-finite descriptor")
    return g.cpu().numpy().astype(np.float32), int(pv.shape[-2]), int(pv.shape[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    ap.add_argument("--split-dir", type=Path, default=PD.SPLIT_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    commit = PD.git_commit_clean()
    device = torch.device("cuda")
    proc, _, model16, winfo = PD.load_depth_model(device, fp16=True)
    out_root = a.out / a.split_dir.name
    out_root.mkdir(parents=True, exist_ok=True)
    t0, summary = time.time(), {}
    for split in a.splits:
        mpath = a.split_dir / f"{split}_manifest.txt"
        if "test" in mpath.name.lower():
            raise SystemExit("[abort] test split refused")
        rows = read_manifest(mpath, allow_test=False)
        if a.limit:
            rows = rows[:a.limit]
        units, G, shas, sizes = [], [], [], []
        for i, r in enumerate(rows):
            g, ph, pw = descriptor(proc, model16, load_u8(r["input"]), device)
            units.append(r["unit"]); G.append(g); shas.append(PD.sha256_file(Path(r["input"]))); sizes.append((ph, pw))
            if (i + 1) % 500 == 0 or i + 1 == len(rows):
                print(f"  [{split}] {i + 1}/{len(rows)} elapsed={time.time() - t0:.0f}s", flush=True)
        if len(set(units)) != len(units):
            raise SystemExit("[abort] duplicate units")
        np.savez_compressed(out_root / f"scene_{split}.npz", unit=np.array(units), g=np.stack(G),
                            input_sha256=np.array(shas), proc_hw=np.array(sizes, dtype=np.int32))
        summary[split] = {"rows": len(rows), "manifest_sha256": PD.sha256_file(mpath), "limit": a.limit or None,
                          "g_mean_abs": float(np.abs(np.stack(G)).mean())}
    meta = {"model_id": PD.MODEL_ID, "revision": PD.MODEL_REV, **winfo, "commit": commit,
            "descriptor": "[CLS, mean(patch tokens)] of backbone feature_maps[-1] (layer 12, layer-normed), 768-d",
            "precision": "fp16", "runtime_s": round(time.time() - t0, 1), "splits": summary}
    (out_root / f"run_meta_{datetime.now():%Y%m%d_%H%M%S}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

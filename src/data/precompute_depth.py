"""
Offline depth precomputation (redesign Phase 1): frozen Depth-Anything-V2-Small on the sanitized
LSUI+UIEB train and val splits, so that training incurs no depth-inference cost.

What is stored
  For every manifest row, the network is run on the *degraded input* image (the only image available at
  inference). Its output is relative inverse depth (disparity, affine-invariant: unknown scale and shift).
  It is bilinearly resized to the native H x W of the input and robustly normalised per image with the
  1st/99th percentiles (approved 2026-11-09; replaces min-max):

      z~(x) = clip((d(x) - P1(d)) / (P99(d) - P1(d)), 0, 1)   in [0, 1]^(H x W),   1 = nearest, 0 = farthest

  This makes the scaling insensitive to the <=1% most extreme pixels at each end. It does NOT undo the
  stretching of a scene whose true depth range is narrow (any per-image normalisation maps the observed
  range onto [0, 1]); the raw P1/P99/min/max are stored so that this can be audited.
  Saved as float16 in a compressed .npz (key "z"), together with P1, P99, raw min/max and the input file's
  SHA-256 so that every map is bound to the exact image it came from. z~ is NOT metric depth and NOT
  depth: it is normalised inverse depth. Any use as a depth proxy (e.g. 1 - z~) is a modelling choice
  made downstream (see src/models/dgt_head.py).

Guarantees
  * Test split refused: only train/val manifests are accepted (read_manifest(allow_test=False) plus an
    explicit name check).
  * Model pinned: Hugging Face repo + commit revision; the safetensors SHA-256 is verified before use.
  * Frozen: eval(), requires_grad False, torch.inference_mode(). fp16 by default; if an fp16 output is
    non-finite the image is recomputed in fp32 and the precision is recorded.
  * Pixel alignment: images are loaded exactly as training loads them (PIL .convert("RGB"), no EXIF
    rotation); output shape == input shape is asserted.
  * Provenance: src/ must be clean (commit recorded); index.jsonl has one line per written map.
  * Resumable: existing maps are skipped unless --overwrite.

Usage
  python src/data/precompute_depth.py --splits val --limit 4 --out <dir>    # smoke test
  python src/data/precompute_depth.py --splits train val                     # full run (~4,652 images)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/validation"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from calibrate_uncertainty import read_manifest  # noqa: E402
from evaluate_metrics import load_u8  # noqa: E402  (same loader convention as train.py load_rgb_u8)

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
MODEL_REV = "5426e4f0f36572d16453bbda7a8389317b1bef99"          # Hub commit, checked 2026-11-09
WEIGHTS_SHA256 = "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"  # model.safetensors
SPLIT_DIR = ROOT / "data/splits/lsui_uieb_all_d6_seed0"
DEFAULT_OUT = ROOT / "data/depth/da2_small_5426e4f"
EPS_RANGE = 1e-6
PCT_LO, PCT_HI = 1.0, 99.0


def sha256_file(p: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_commit_clean() -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"[abort] src/ has uncommitted changes; commit first:\n{dirty}")
    return head


def load_depth_model(device: torch.device, fp16: bool):
    from huggingface_hub import hf_hub_download
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    wpath = Path(hf_hub_download(MODEL_ID, "model.safetensors", revision=MODEL_REV))
    got = sha256_file(wpath)
    if got != WEIGHTS_SHA256:
        raise SystemExit(f"[abort] weights SHA-256 {got} != pinned {WEIGHTS_SHA256}")
    proc = AutoImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REV)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID, revision=MODEL_REV, torch_dtype=torch.float32)
    model.eval().requires_grad_(False).to(device)
    model_fp16 = None
    if fp16:
        if device.type != "cuda":
            raise SystemExit("[abort] --fp16 requires CUDA; use --no-fp16 on CPU")
        import copy
        model_fp16 = copy.deepcopy(model).half()
    return proc, model, model_fp16, {"weights_path": str(wpath), "weights_sha256": got}


@torch.inference_mode()
def disparity(proc, model, model_fp16, x8: np.ndarray, device):
    """x8: HxWx3 uint8 -> (HxW float32 relative inverse depth at native resolution, precision used)."""
    H, W = x8.shape[:2]
    pv = proc(images=x8, return_tensors="pt")["pixel_values"].to(device)
    prec = "fp32"
    if model_fp16 is not None:
        d = model_fp16(pixel_values=pv.half()).predicted_depth.float()
        prec = "fp16"
        if not torch.isfinite(d).all():
            d, prec = model(pixel_values=pv).predicted_depth.float(), "fp32 (fp16 non-finite)"
    else:
        d = model(pixel_values=pv).predicted_depth.float()
    if d.dim() == 3:
        d = d[:, None]
    d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    return d.cpu().numpy().astype(np.float32), prec


def normalise(d: np.ndarray):
    """-> (z~ float32 in [0,1], stats dict, degenerate flag). Percentile normalisation with clipping."""
    if not np.isfinite(d).all():
        raise ValueError("non-finite disparity")
    p_lo, p_hi = (float(v) for v in np.percentile(d, (PCT_LO, PCT_HI)))
    st = {"p_lo": p_lo, "p_hi": p_hi, "raw_min": float(d.min()), "raw_max": float(d.max())}
    if p_hi - p_lo < EPS_RANGE:
        return np.zeros_like(d, dtype=np.float32), st, True
    return np.clip((d - p_lo) / (p_hi - p_lo), 0.0, 1.0).astype(np.float32), st, False


def out_path(root: Path, row) -> Path:
    ds = row["dataset"]
    unit = row["unit"].split(":", 1)[1] if ":" in row["unit"] else row["unit"]
    rel = Path(unit.replace("\\", "/")).with_suffix(".npz")
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe unit id {row['unit']}")
    return root / ds / rel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    ap.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--limit", type=int, default=0, help="first N rows per split (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    commit = git_commit_clean()
    device = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if a.device == "cuda" and device.type != "cuda":
        raise SystemExit("[abort] CUDA requested but unavailable")
    proc, model, model_fp16, winfo = load_depth_model(device, a.fp16)
    n_params = sum(p.numel() for p in model.parameters())
    out_root = a.out / a.split_dir.name
    out_root.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    summary = {}
    for split in a.splits:
        mpath = a.split_dir / f"{split}_manifest.txt"
        if "test" in mpath.name.lower():
            raise SystemExit("[abort] test split refused")
        rows = read_manifest(mpath, allow_test=False)
        targets = [out_path(out_root, r) for r in rows]
        if len(set(targets)) != len(targets):
            raise SystemExit(f"[abort] {split}: output path collision")
        if a.limit:
            rows, targets = rows[:a.limit], targets[:a.limit]
        idx_path = out_root / f"index_{split}.jsonl"
        n_written = n_skipped = n_degenerate = 0
        precs = {}
        with open(idx_path, "a", encoding="utf-8") as idx:
            for i, (r, tp) in enumerate(zip(rows, targets)):
                if tp.exists() and not a.overwrite:
                    n_skipped += 1
                    continue
                x8 = load_u8(r["input"])
                ts = time.time()
                d, prec = disparity(proc, model, model_fp16, x8, device)
                z, st, degen = normalise(d)
                if z.shape != x8.shape[:2] or not np.isfinite(z).all() or z.min() < 0 or z.max() > 1:
                    raise SystemExit(f"[abort] invalid map for {r['unit']}: shape {z.shape} vs {x8.shape[:2]}")
                tp.parent.mkdir(parents=True, exist_ok=True)
                tmp = tp.with_name(tp.stem + ".tmp.npz")   # np.savez_compressed keeps a .npz suffix
                np.savez_compressed(tmp, z=z.astype(np.float16), **{k: np.float32(v) for k, v in st.items()})
                tmp.replace(tp)
                n_written += 1
                n_degenerate += int(degen)
                precs[prec] = precs.get(prec, 0) + 1
                idx.write(json.dumps({"unit": r["unit"], "dataset": r["dataset"], "split": split,
                                      "input": r["input"], "input_sha256": sha256_file(Path(r["input"])),
                                      "depth": str(tp.relative_to(out_root)), "H": int(z.shape[0]),
                                      "W": int(z.shape[1]), **st,
                                      "degenerate": degen, "precision": prec,
                                      "sec": round(time.time() - ts, 3)}) + "\n")
                if (i + 1) % 100 == 0 or i + 1 == len(rows):
                    print(f"  [{split}] {i + 1}/{len(rows)} written={n_written} skipped={n_skipped} "
                          f"elapsed={time.time() - t0:.0f}s", flush=True)
        summary[split] = {"manifest": str(mpath), "manifest_sha256": sha256_file(mpath), "rows": len(rows),
                          "written": n_written, "skipped_existing": n_skipped, "degenerate": n_degenerate,
                          "precision_counts": precs, "limit": a.limit or None}
    meta = {"model_id": MODEL_ID, "revision": MODEL_REV, **winfo, "n_params": n_params,
            "licence": "apache-2.0 (Hub card metadata for this repo, checked 2026-11-09)",
            "stored": f"z~ = clip((d - P{PCT_LO:g}) / (P{PCT_HI:g} - P{PCT_LO:g}), 0, 1) of relative inverse depth d, float16, "
                      "native input resolution, bilinear upsampling; 1 = near, 0 = far; not metric",
            "fp16": a.fp16, "device": str(device), "commit": commit, "started": started,
            "runtime_s": round(time.time() - t0, 1), "splits": summary,
            "env": {"python": platform.python_version(), "torch": torch.__version__,
                    "transformers": __import__("transformers").__version__,
                    "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}}
    (out_root / f"run_meta_{datetime.now():%Y%m%d_%H%M%S}.json").write_text(json.dumps(meta, indent=2),
                                                                             encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

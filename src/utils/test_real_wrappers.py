"""
V3 on real restorers: PadCropWrapper vs direct forward with the actual checkpoints.

Models (strict checkpoint load required, else the model is reported FAIL_LOAD):
  PSPL-HFBranch  PSPL_HRNet_net_G.pth   returns (feature, image)   input range [-1, 1]
  PSPL-UWCNN     PSPL_UWCNN_net_G.pth   returns image              input range [-1, 1]
  UVEB-small     samll_net_g.pth        Deblur_samll(num_feat=12, num_block=15),
                 5-D video input (B,T,C,H,W) with T=1, input range [0, 1];
                 the adapter feeds T=1 and returns the full-res frame only.

Inputs: real images from the split manifests (default data/splits/lsui_uieb_all_d6_seed0),
loaded as float32 RGB WITHOUT resizing.

  V3a  H, W already multiples of 16 -> the wrapper must not pad, and
       max|wrapped - direct| must be < --tol (default 1e-7). Bit-exactness is also reported.
  V3b  H or W not a multiple -> wrapped must equal direct(reflect_pad(x))[..., :H, :W]
       (same tolerance). This is the operation the wrapper claims to perform.
  SHAPE  every output equals the input's (H, W).

Determinism: cuDNN deterministic, benchmark off, TF32 off,
torch.use_deterministic_algorithms(True). This makes "direct" and "wrapped" run the
same kernels, so drift should be 0. A non-zero drift is reported, not hidden.

Out-of-memory at a size is recorded as SKIP_OOM (not PASS, not FAIL).

Usage:
  python src/utils/test_real_wrappers.py                    # 6 V3a + 3 V3b images per dataset
  python src/utils/test_real_wrappers.py --n-aligned 20 --n-unaligned 5 --models uveb_small
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # must precede CUDA init

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "utils"))
from pad_crop_wrapper import PadCropWrapper, pad_amounts  # noqa: E402

PSPL_DIR = ROOT / "src/restoration/repositories/PSPL/PSPL-main/official_source/DL_2025PSPL_github"
UVEB_DIR = ROOT / "src/restoration/repositories/UVEB/UVEB-main/UVE-Net"
MULTIPLE = 16


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def _strict_load(model: nn.Module, ckpt: Path) -> nn.Module:
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "params" in state and isinstance(state["params"], dict):
        state = state["params"]
    model.load_state_dict(state, strict=True)       # raises on any missing/unexpected key
    return model.eval()


class UVEBSingleFrame(nn.Module):
    """(N,3,H,W) -> Deblur_samll((N,1,3,H,W)) -> full-res frame (N,3,H,W)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out, _quarter = self.net(x.unsqueeze(1))
        return out[:, 0]


def build_models(which):
    models = {}
    if any(m.startswith("pspl") for m in which):
        sys.path.insert(0, str(PSPL_DIR))
        from models.semi.HFBranch import HFBranch
        from models.semi.UWCNN import UWCNN
        ck = PSPL_DIR / "pre-trained/unsuped"
        if "pspl_hfbranch" in which:
            models["pspl_hfbranch"] = (lambda: _strict_load(HFBranch(), ck / "PSPL_HRNet_net_G.pth"), "pm1")
        if "pspl_uwcnn" in which:
            models["pspl_uwcnn"] = (lambda: _strict_load(UWCNN(), ck / "PSPL_UWCNN_net_G.pth"), "pm1")
    if "uveb_small" in which:
        sys.path.insert(0, str(UVEB_DIR))
        from basicsr.archs.deblur_arch_small import Deblur_samll
        models["uveb_small"] = (lambda: UVEBSingleFrame(_strict_load(
            Deblur_samll(num_feat=12, num_block=15), UVEB_DIR / "pretrained/samll_net_g.pth")), "01")
    return models


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def read_manifest(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path} (run src/data/build_splits.py first)")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        ds, uid, _cid, inp, _ref = line.split("\t")
        inp = str(Path(inp) if Path(inp).is_absolute() else Path(__file__).resolve().parents[2] / inp)
        rows.append((ds, uid, inp))
    return rows


def image_hw(p):
    with Image.open(p) as im:
        return im.height, im.width


def load_rgb(p, rng: str, device):
    with Image.open(p) as im:
        a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    x = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).contiguous()
    if rng == "pm1":
        x = x * 2.0 - 1.0
    return x.to(device)


def pick_images(rows, n_aligned, n_unaligned):
    """Per dataset: up to n_aligned images with H,W % 16 == 0 (distinct sizes first), n_unaligned others."""
    picked = []
    for ds in sorted({r[0] for r in rows}):
        aligned, unaligned, seen = [], [], set()
        for _, uid, p in sorted(r for r in rows if r[0] == ds):
            h, w = image_hw(p)
            key = (h, w)
            is_al = pad_amounts(h, w, MULTIPLE) == (0, 0)
            bucket = aligned if is_al else unaligned
            limit = n_aligned if is_al else n_unaligned
            if len(bucket) < limit and key not in seen:
                bucket.append((ds, uid, p, h, w)); seen.add(key)
        picked += aligned + unaligned
        print(f"[data] {ds}: {len(aligned)} aligned (x{MULTIPLE}) + {len(unaligned)} unaligned images (distinct sizes)")
    return picked


# --------------------------------------------------------------------------
# Test
# --------------------------------------------------------------------------
def _first(y):
    return y[1] if isinstance(y, (tuple, list)) and len(y) == 2 and y[1].shape[1] == 3 else (
        y[0] if isinstance(y, (tuple, list)) else y)


def _all_tensors(y):
    return list(y) if isinstance(y, (tuple, list)) else [y]


def run(args):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device(args.device)
    rows = read_manifest(Path(args.manifest))
    images = pick_images(rows, args.n_aligned, args.n_unaligned)
    builders = build_models(args.models)
    results, overall = [], True

    for name, (build, rng) in builders.items():
        try:
            model = build().to(device)
        except Exception as e:
            print(f"[{name}] FAIL_LOAD: {e}")
            results.append({"model": name, "status": "FAIL_LOAD", "error": repr(e)})
            overall = False
            continue
        wrapped = PadCropWrapper(model, MULTIPLE, mode="reflect")
        print(f"[{name}] strict load OK, params={sum(p.numel() for p in model.parameters()):,}")
        for ds, uid, p, h, w in images:
            rec = {"model": name, "dataset": ds, "unit": uid, "H": h, "W": w,
                   "test": "V3a" if pad_amounts(h, w, MULTIPLE) == (0, 0) else "V3b"}
            try:
                x = load_rgb(p, rng, device)
                assert tuple(x.shape) == (1, 3, h, w), f"loaded shape {tuple(x.shape)} != (1,3,{h},{w})"
                with torch.no_grad():
                    yw = wrapped(x)
                    if rec["test"] == "V3a":
                        yd = model(x)
                    else:
                        ph, pw = pad_amounts(h, w, MULTIPLE)
                        yd_full = model(F.pad(x, (0, pw, 0, ph), mode="reflect"))
                        yd = [t[..., :h, :w] for t in _all_tensors(yd_full)]
                        yd = yd if isinstance(yd_full, (tuple, list)) else yd[0]
                tw, td = _all_tensors(yw), _all_tensors(yd)
                shapes_ok = all(tuple(t.shape[-2:]) == (h, w) for t in tw) and len(tw) == len(td)
                diffs = [(a.double() - b.double()).abs().max().item() for a, b in zip(tw, td)]
                exact = all(torch.equal(a, b) for a, b in zip(tw, td))
                finite = all(torch.isfinite(t).all().item() for t in tw)
                drift = max(diffs)
                ok = shapes_ok and finite and drift < args.tol
                rec.update(status="PASS" if ok else "FAIL", max_abs_drift=drift, bit_exact=exact,
                           shapes_ok=shapes_ok, finite=finite,
                           out_shapes=[list(t.shape) for t in tw],
                           out_range=[float(_first(yw).min()), float(_first(yw).max())])
                overall &= ok
            except torch.cuda.OutOfMemoryError:
                rec.update(status="SKIP_OOM")
                torch.cuda.empty_cache()
            except Exception as e:
                rec.update(status="FAIL", error=repr(e), trace=traceback.format_exc(limit=3))
                overall = False
            results.append(rec)
            print(f"  {rec['status']:8s} {rec['test']} {ds:4s} {h:4d}x{w:<4d} "
                  f"drift={rec.get('max_abs_drift', float('nan')):.2e} bit_exact={rec.get('bit_exact')}"
                  + (f"  {rec.get('error', '')}" if rec["status"] == "FAIL" and "error" in rec else ""))
        del model, wrapped
        if device.type == "cuda":
            torch.cuda.empty_cache()

    n = {s: sum(r.get("status") == s for r in results) for s in ("PASS", "FAIL", "SKIP_OOM", "FAIL_LOAD")}
    print(f"\nSUMMARY {n}  ->  OVERALL: {'PASS' if overall else 'FAIL'}")
    out = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({
        "torch": torch.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "tol": args.tol, "multiple": MULTIPLE, "manifest": args.manifest,
        "summary": n, "overall": "PASS" if overall else "FAIL", "results": results}, indent=2), encoding="utf-8")
    print(f"[out] {out / 'results.json'}")
    return 0 if overall else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/splits/lsui_uieb_all_d6_seed0/test_manifest.txt"))
    ap.add_argument("--models", nargs="+", default=["pspl_hfbranch", "pspl_uwcnn", "uveb_small"],
                    choices=["pspl_hfbranch", "pspl_uwcnn", "uveb_small"])
    ap.add_argument("--n-aligned", type=int, default=6)
    ap.add_argument("--n-unaligned", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(ROOT / "reports/integrity_audit/wrapper_v3_real"))
    sys.exit(run(ap.parse_args()))


if __name__ == "__main__":
    main()

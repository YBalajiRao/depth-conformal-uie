"""
Standardized data + metric adapter for external baselines. Everything is imported from the SS-PGN code, so the
protocol is identical by construction (no re-implementation):

  split      : data/splits/lsui_uieb_all_d6_seed0 (train 4,135 = 3,423 LSUI + 712 UIEB; val 517 = 428 + 89);
               test manifest refused (train.read_manifest)
  training   : train.PairedCropDataset - 256x256 random crop, reflect-pad + mask for small images,
               rot90 (k in 0..3) + horizontal flip, RGB in [0, 1] (PIL .convert("RGB"), no EXIF rotation)
  evaluation : full native resolution, per-image PSNR8 / SSIM8 on uint8-rounded outputs
               (diagnostics/common: to_u8_tensor, psnr_u8, ssim_u8, mse_u8), per-dataset means, paired bootstrap
               B = 2000 seed 20261107 (common.boot_groups) - same functions as S1_EVAL517 / P1B_EVAL517
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("src/validation", "src/utils", "src/training", "src/models", "src/diagnostics"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

import train as _T  # noqa: E402
from evaluate_metrics import psnr_u8, ssim_u8, load_u8  # noqa: E402

SPLIT = ROOT / "data/splits/lsui_uieb_all_d6_seed0"
EXPECTED = {"train": {"LSUI": 3423, "UIEB": 712}, "val": {"LSUI": 428, "UIEB": 89}}


def rows(split: str):
    if split not in EXPECTED:
        raise ValueError("only 'train' and 'val' are available to baselines (test refused)")
    r = _T.read_manifest(SPLIT / f"{split}_manifest.txt")
    got = {d: sum(x[0] == d for x in r) for d in EXPECTED[split]}
    if got != EXPECTED[split]:
        raise RuntimeError(f"{split} counts {got} != {EXPECTED[split]}")
    return r


def train_dataset(crop: int = 256):
    return _T.PairedCropDataset(rows("train"), crop, augment=True)


def val_dataset():
    return _T.FullImageDataset(rows("val"))


def to_u8(J):
    """(1,3,H,W) float -> HxWx3 uint8 with the same rounding as the SS-PGN evaluation."""
    return (J.clamp(0, 1)[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0 + 0.5).astype("uint8")


__all__ = ["rows", "train_dataset", "val_dataset", "to_u8", "psnr_u8", "ssim_u8", "load_u8", "SPLIT"]

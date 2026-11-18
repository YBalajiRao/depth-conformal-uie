"""
Pixel-aligned pad-and-crop wrapper + V1-V3 self-test.

Policy (agreed 2026-11-05): NO resizing for quantitative error targets.
Inputs are padded on the BOTTOM and RIGHT edges only, up to the next multiple
of `multiple`, run through the model, and the output is cropped back to
[:H, :W]. Pixel (i, j) of the output therefore corresponds to pixel (i, j) of
the input.

Tests (run:  python src/utils/pad_crop_wrapper.py):
  V1  Synthetic coordinate grid
      V1a  identity / pixel-unshuffle->shuffle models: every output pixel must
           carry exactly its own (row, col) coordinate (bit-exact).
      V1b  translation-equivariant conv model: interior pixels (outside the
           receptive field of the padded border) must equal the direct
           forward on the unpadded image -> detects any 1-px offset.
  V2  Exact shape assertions on every real UIEB / LSUI / EUVP_test resolution
      (read from reports/dataset_audit/paired_dimension_audit.csv).
  V3  Wrapper vs direct forward:
      V3a  size already a multiple -> wrapper output is bit-identical to model(x)
           and no padding is applied.
      V3b  otherwise -> wrapper output is bit-identical to manually
           reflect-padding, calling the model, and cropping.
  NEG  A deliberately wrong wrapper (pads top/left) MUST fail V1 - proves the
       test can detect the error it targets.

Note: for real CNNs the border pixels within the receptive field of the padded
region DO depend on the padding mode (reflect vs zero). That is expected and is
not a misalignment; V1b therefore checks only the interior.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Wrapper
# --------------------------------------------------------------------------
def pad_amounts(h: int, w: int, multiple: int) -> tuple[int, int]:
    """Bottom and right padding needed to reach the next multiple."""
    return (-h) % multiple, (-w) % multiple


def pad_bottom_right(x: torch.Tensor, multiple: int, mode: str = "reflect"):
    """Pad a (N,C,H,W) tensor on bottom/right. Returns padded tensor and (H, W)."""
    if x.dim() != 4:
        raise ValueError(f"expected 4-D (N,C,H,W) tensor, got shape {tuple(x.shape)}")
    h, w = x.shape[-2:]
    ph, pw = pad_amounts(h, w, multiple)
    if mode == "reflect" and (ph >= h or pw >= w):
        raise ValueError(f"reflect pad ({ph},{pw}) must be < image size ({h},{w})")
    if ph == 0 and pw == 0:
        return x, (h, w)
    # F.pad order for last two dims: (left, right, top, bottom)
    return F.pad(x, (0, pw, 0, ph), mode=mode), (h, w)


def crop_to(y: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    h, w = hw
    return y[..., :h, :w]


class PadCropWrapper(nn.Module):
    """Wrap any image-to-image model that needs H, W divisible by `multiple`.

    The model must return a tensor with the same spatial size as its input
    (asserted). Extra outputs (tuple / dict) are cropped element-wise.
    """

    def __init__(self, model: nn.Module, multiple: int = 16, mode: str = "reflect"):
        super().__init__()
        self.model, self.multiple, self.mode = model, multiple, mode

    def _crop_out(self, y, padded_hw, hw):
        if torch.is_tensor(y):
            if y.dim() < 4:          # per-image outputs, e.g. (N,3) kappa or (N,3,3) colour matrix
                return y
            assert tuple(y.shape[-2:]) == padded_hw, (
                f"model changed spatial size: in {padded_hw}, out {tuple(y.shape[-2:])}")
            return crop_to(y, hw)
        if isinstance(y, (tuple, list)):
            return type(y)(self._crop_out(v, padded_hw, hw) for v in y)
        if isinstance(y, dict):
            return {k: self._crop_out(v, padded_hw, hw) for k, v in y.items()}
        return y

    def forward(self, x: torch.Tensor, *aux: torch.Tensor):
        """aux: optional extra per-pixel inputs aligned with x (e.g. a (N,1,H,W) depth map); each is padded
        exactly like x (same amounts, same mode) and passed positionally after x."""
        xp, hw = pad_bottom_right(x, self.multiple, self.mode)
        auxp = []
        for a in aux:
            if a.dim() < 4:                     # per-image vectors (e.g. (N, 768) scene descriptor): no padding
                auxp.append(a)
                continue
            assert tuple(a.shape[-2:]) == hw, f"aux input {tuple(a.shape)} not aligned with x {hw}"
            auxp.append(pad_bottom_right(a, self.multiple, self.mode)[0])
        y = self.model(xp, *auxp)
        out = self._crop_out(y, tuple(xp.shape[-2:]), hw)
        if torch.is_tensor(out):
            assert tuple(out.shape[-2:]) == hw
        return out


class WrongTopLeftWrapper(PadCropWrapper):
    """NEGATIVE CONTROL ONLY: pads top/left but crops [:H,:W] -> misaligned."""

    def forward(self, x):
        h, w = x.shape[-2:]
        ph, pw = pad_amounts(h, w, self.multiple)
        xp = F.pad(x, (pw, 0, ph, 0), mode=self.mode)
        return crop_to(self.model(xp), (h, w))


# --------------------------------------------------------------------------
# Synthetic test models
# --------------------------------------------------------------------------
class DivisibleIdentity(nn.Module):
    """Identity that raises unless H, W are divisible (mimics strided U-Nets)."""

    def __init__(self, multiple):
        super().__init__()
        self.m = multiple

    def forward(self, x):
        assert x.shape[-2] % self.m == 0 and x.shape[-1] % self.m == 0, tuple(x.shape)
        return x.clone()


class ShuffleRoundTrip(nn.Module):
    """PixelUnshuffle(m) -> PixelShuffle(m): exact identity, requires divisibility."""

    def __init__(self, multiple):
        super().__init__()
        self.down, self.up = nn.PixelUnshuffle(multiple), nn.PixelShuffle(multiple)

    def forward(self, x):
        return self.up(self.down(x))


class EquivariantConv(nn.Module):
    """Bias-free stack of 3x3 convs (translation-equivariant). RF radius = n_layers."""

    def __init__(self, ch=3, n_layers=3, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.convs = nn.ModuleList(nn.Conv2d(ch, ch, 3, padding=1, bias=False) for _ in range(n_layers))
        for c in self.convs:
            c.weight.data = torch.randn(c.weight.shape, generator=g) * 0.2
        self.radius = n_layers

    def forward(self, x):
        for c in self.convs:
            x = c(x)
        return x


def coord_grid(h, w, device):
    """(1,3,H,W) float32: channel0=row, channel1=col, channel2=row*W+col (all exact in fp32)."""
    r = torch.arange(h, device=device, dtype=torch.float32).view(h, 1).expand(h, w)
    c = torch.arange(w, device=device, dtype=torch.float32).view(1, w).expand(h, w)
    return torch.stack([r, c, r * w + c]).unsqueeze(0)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def load_resolutions(csv_path: Path):
    by_ds = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            by_ds.setdefault(r["Dataset"], set()).add((int(r["InputHeight"]), int(r["InputWidth"])))
    return {k: sorted(v) for k, v in by_ds.items()}


def v1_coordinate(sizes, multiple, device, wrapper_cls=PadCropWrapper):
    """Returns list of failure strings."""
    fails = []
    models = {"identity": DivisibleIdentity(multiple), "shuffle": ShuffleRoundTrip(multiple)}
    for name, m in models.items():
        wrap = wrapper_cls(m.to(device), multiple)
        for h, w in sizes:
            g = coord_grid(h, w, device)
            out = wrap(g)
            if tuple(out.shape) != tuple(g.shape) or not torch.equal(out, g):
                fails.append(f"V1a[{name}] {h}x{w}")
    conv = EquivariantConv().to(device)
    wrap = wrapper_cls(conv, multiple)
    rad = conv.radius
    for h, w in sizes:
        g = coord_grid(h, w, device) / max(h * w, 1)  # keep magnitudes O(1)
        ref = conv(g)                                  # direct, zero-padded by conv itself
        out = wrap(g)
        # interior: away from all borders by > rad (top/left borders are identical
        # in both paths; bottom/right differ only within rad of the original edge)
        a, b = ref[..., rad:h - rad, rad:w - rad], out[..., rad:h - rad, rad:w - rad]
        if out.shape != ref.shape or not torch.allclose(a, b, atol=1e-5, rtol=1e-4):
            err = (a - b).abs().max().item() if out.shape == ref.shape else float("nan")
            fails.append(f"V1b {h}x{w} max|diff|={err:.2e}")
    return fails


def v2_shapes(sizes, multiple):
    fails = []
    for h, w in sizes:
        ph, pw = pad_amounts(h, w, multiple)
        x = torch.zeros(1, 3, h, w)
        xp, hw = pad_bottom_right(x, multiple)
        ok = (0 <= ph < multiple and 0 <= pw < multiple
              and xp.shape[-2] == h + ph and xp.shape[-1] == w + pw
              and xp.shape[-2] % multiple == 0 and xp.shape[-1] % multiple == 0
              and hw == (h, w) and crop_to(xp, hw).shape == x.shape)
        # reflect content check: padded rows/cols mirror the edge (excluding the edge itself)
        if ok and ph:
            ok = torch.equal(xp[..., h:, :w], torch.flip(x[..., h - 1 - ph:h - 1, :], dims=[-2]))
        if not ok:
            fails.append(f"V2 {h}x{w}")
    return fails


def v3_wrapper_vs_direct(sizes, multiple, device):
    fails = []
    model = EquivariantConv().to(device)
    wrap = PadCropWrapper(model, multiple)
    g = torch.Generator(device="cpu").manual_seed(1)
    for h, w in sizes:
        x = torch.rand(1, 3, h, w, generator=g).to(device)
        out = wrap(x)
        if h % multiple == 0 and w % multiple == 0:
            if not torch.equal(out, model(x)):
                fails.append(f"V3a {h}x{w}")
        else:
            ph, pw = pad_amounts(h, w, multiple)
            manual = model(F.pad(x, (0, pw, 0, ph), mode="reflect"))[..., :h, :w]
            if not torch.equal(out, manual):
                fails.append(f"V3b {h}x{w}")
    return fails


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(root / "reports/dataset_audit/paired_dimension_audit.csv"))
    ap.add_argument("--multiples", default="4,16,32,64")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    res = load_resolutions(Path(args.csv))
    sizes = sorted({s for v in res.values() for s in v})
    print(f"device={args.device}  torch={torch.__version__}")
    print("resolutions: " + ", ".join(f"{k}={len(v)}" for k, v in res.items()) + f"  (union={len(sizes)})")

    all_ok = True
    with torch.no_grad():
        for m in [int(v) for v in args.multiples.split(",")]:
            r = {"V1": v1_coordinate(sizes, m, args.device),
                 "V2": v2_shapes(sizes, m),
                 "V3": v3_wrapper_vs_direct(sizes, m, args.device)}
            neg = v1_coordinate(sizes, m, args.device, wrapper_cls=WrongTopLeftWrapper)
            n_unaligned = sum(1 for h, w in sizes if pad_amounts(h, w, m) != (0, 0))
            neg_ok = n_unaligned == 0 or len(neg) > 0
            for k, f in r.items():
                print(f"[multiple={m:>2}] {k}: {'PASS' if not f else 'FAIL'} "
                      f"({len(sizes) - len(set(x.split()[1] for x in f))}/{len(sizes)} sizes ok)")
                for line in f[:5]:
                    print("     ", line)
            print(f"[multiple={m:>2}] NEG control (top/left pad must fail): "
                  f"{'DETECTED' if neg_ok else 'NOT DETECTED -> test is too weak'} "
                  f"({len(neg)} failures over {n_unaligned} sizes needing padding)")
            all_ok &= all(not f for f in r.values()) and neg_ok

    print("\nOVERALL:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

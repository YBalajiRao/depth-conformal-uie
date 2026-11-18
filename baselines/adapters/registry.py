"""
External baseline registry: builds each official architecture from its pinned clone in baselines/repos/ and
exposes it through one interface used by our standardized training/evaluation protocol.

    spec = BASELINES[name]
    model = build(name)                       # nn.Module, CPU, randomly initialised (no pretrained weights)
    J = predict(name, model, x)               # x: (N,3,H,W) in [0,1] -> J: (N,3,H,W)

Pretrained checkpoints shipped in the repos are NOT used: they were trained on author splits of LSUI/UIEB that
can overlap our validation/test images, so every baseline must be retrained on our fixed train split.

Input-size handling at evaluation (native resolution, as for SS-PGN):
    'pad'      : reflect-pad bottom/right to `multiple`, crop back (PadCropWrapper semantics), native metrics.
    'resize256': U-shape Transformer only (learned positional encoding fixed at img_dim = 256): bilinear resize
                 (antialiased) to 256x256, enhance, bilinear resize back to native H x W (PI decision 2026-11-13),
                 metrics at native resolution against
                 the native reference. This is a protocol deviation imposed by the architecture (see README).
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

REPOS = Path(__file__).resolve().parents[1] / "repos"


@dataclass(frozen=True)
class Spec:
    repo: str
    commit: str
    url: str
    licence: str
    module: str          # module path inside the repo
    cls: str
    kwargs: tuple        # constructor kwargs as tuple of pairs
    eval_mode: str       # 'pad' | 'resize256'
    multiple: int        # spatial multiple required by the architecture (pad mode)
    output: str          # how to get J from forward(): 'last' (list), 'first' (tuple), 'tensor'
    notes: str


BASELINES = {
    "ushape": Spec(
        repo="U-shape_Transformer_for_Underwater_Image_Enhancement",
        commit="41f9897669d2e6f6718e00772ece2523d0f149a2",
        url="https://github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement",
        licence="MIT", module="net.Ushape_Trans", cls="Generator", kwargs=(),
        eval_mode="resize256", multiple=256, output="last",
        notes="Peng et al., IEEE TIP 2023. Fixed 256x256 input (learned positional encoding over 16x16 patches); "
              "official train/test resize to 256. Official training: GAN (Discriminator) + RGB/LAB/LCH + "
              "perceptual losses, L2 then L1, batch 1."),
    "fiveaplus": Spec(
        repo="FiveAPlus-Network",
        commit="c15e2234235ef4823ec6f5b28f0f1dfdf7173e87",
        url="https://github.com/Owen718/FiveAPlus-Network",
        licence="none stated (no LICENSE file; reuse rights unclear)", module="archs.FIVE_APLUS",
        cls="FIVE_APLUSNet", kwargs=(), eval_mode="pad", multiple=1, output="first",   # any H, W (size probe)
        notes="Jiang et al., BMVC 2023. forward() returns (pred, stage2_head). Official training: Adam lr 4e-4, "
              "CyclicLR, 256 crops, 400 epochs (pytorch_lightning, not installed; not needed for our loop)."),
    "spectroformer": Spec(
        repo="Spectroformer",
        commit="0a0c17b37e478bcc786cf923698f43c999587de9",
        url="https://github.com/Mdraqibkhan/Spectroformer",
        licence="MIT (stated in README)", module="Final_model_AGSSF", cls="mymodel", kwargs=(),
        eval_mode="pad", multiple=8, output="tensor",
        notes="Khan et al., WACV 2024. mymodel is defined in 5 files; Train.py imports a non-existent "
              "'final_model' module; the released best.pth pickles model_without_CA.mymodel (an ablation). "
              "Default here: Final_model_AGSSF.mymodel (full model, assumed) - PI decision pending."),
    "wweuie": Spec(
        repo="WWE-UIE",
        commit="fd06975",
        url="https://github.com/chingheng0808/WWE-UIE",
        licence="Apache-2.0", module="model", cls="myModel",
        kwargs=(("in_channels", 3), ("feature_channels", 32), ("use_white_balance", True)),
        eval_mode="pad", multiple=4, output="tensor",   # size probe: 252x340 ok, 250x330 fails
        notes="Cheng et al., WACV 2026 (arXiv 2511.16321). Constructor as in official train.py. Official training adds "
              "an HVI-space loss using a pretrained CIDNet (LOLv2 weights); not used under the unified recipe."),
    "phaseformer": Spec(
        repo="PhaseFormer",
        commit="6e759dc",
        url="https://github.com/Mdraqibkhan/Phaseformer",
        licence="MIT", module="model_with_eca", cls="Restormer", kwargs=(),
        eval_mode="pad", multiple=8, output="first",    # forward -> (J at input res, 2x head); official test uses [0]
        notes="Khan et al. (Phaseformer). Official train.py builds model_with_eca.Restormer(); same group and "
              "Restormer-style backbone as Spectroformer. Its second (2x upsampled) output head is trained by the "
              "official recipe against a 512 target; the unified recipe supervises output [0] only."),
}

_SHARED = ("net", "archs", "utils", "model", "loss", "data", "dataset", "network1")


def _import(spec: Spec):
    root = REPOS / spec.repo
    if not root.is_dir():
        raise FileNotFoundError(f"{root} (clone {spec.url} at {spec.commit})")
    for m in [m for m in list(sys.modules) if m.split(".")[0] in _SHARED or m == spec.module.split(".")[0]]:
        del sys.modules[m]                         # repos share top-level module names (net, utils, model, ...)
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module(spec.module)
    finally:
        sys.path.remove(str(root))


# Numerical-stability wrappers (disclosed; recorded in each run's config.json). Same mathematics, higher precision.
NUMERICS_PATCH = {
    "ushape": "ChannelTransformer (model.mtc, net/CMSFFT.py) evaluated in fp32 with autocast disabled (inputs cast to "
              "float); rest of the network fp16 autocast. Unscaled attention scores Q.K^T (CMSFFT.py:166) reached "
              "6.22e4 = 0.95x fp16 max at step 12k; the 2026-11-13 23:49 run then hit non-finite losses from step 12141.",
}


def _fp32_forward(fwd):
    def forward(*args, **kw):
        with torch.amp.autocast("cuda", enabled=False):
            return fwd(*[a.float() if torch.is_tensor(a) and a.is_floating_point() else a for a in args], **kw)
    return forward


def build(name: str, module_override: str | None = None) -> torch.nn.Module:
    spec = BASELINES[name]
    if module_override:
        spec = Spec(**{**spec.__dict__, "module": module_override})
    mod = _import(spec)
    model = getattr(mod, spec.cls)(**dict(spec.kwargs))
    if name == "ushape":
        if type(getattr(model, "mtc", None)).__name__ != "ChannelTransformer":
            raise RuntimeError("ushape: model.mtc ChannelTransformer not found to patch")
        model.mtc.forward = _fp32_forward(model.mtc.forward)
        model._numerics_patched = 1
    return model


def raw_forward(name: str, model, x):
    out = model(x)
    how = BASELINES[name].output
    if how == "last":
        return out[-1]
    if how == "first":
        return out[0]
    return out


def predict(name: str, model, x: torch.Tensor) -> torch.Tensor:
    """Native-resolution prediction under the standardized protocol (see module docstring)."""
    spec = BASELINES[name]
    h, w = x.shape[-2:]
    if spec.eval_mode == "resize256":
        xs = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False, antialias=True).clamp(0, 1)
        J = raw_forward(name, model, xs)
        return F.interpolate(J, size=(h, w), mode="bilinear", align_corners=False)
    m = spec.multiple
    if m and m > 1:
        ph, pw = (-h) % m, (-w) % m
        xp = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x
        return raw_forward(name, model, xp)[..., :h, :w]
    return raw_forward(name, model, x)

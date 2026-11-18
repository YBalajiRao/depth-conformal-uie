"""
Phase E add-on 2 (report section S): external pretrained checkpoints, zero-shot sets only, each with its OFFICIAL test
protocol (read from the repos on 2026-11-17). EXPLORATORY.

  MARINE-Net-LSUI   src/restoration/repositories/MARINE-Net/.../marine_lsui_best.zip -> best_finetuned.pth (state dict,
                    weights_only load). Official: torchvision Resize((256,256)) + ToTensor, outputs['enhanced'].clamp(0,1).
                    Sets: dark (FR) + C60 (NR). Training data: EUVP underwater_scenes (pre-train, assumed) + LSUI; both
                    verified 0 duplicates vs dark/C60 (2026-11-17 audit).
  FiveA+ official   baselines/repos/FiveAPlus-Network/model/FAPlusNet-alpha-{0.1,0.4}.pth (state dict). Official: native
                    resolution, ToTensor, first output (our adapter pads to the model multiple; disclosed). UIEB 800.
                    Sets: dark + C60.
  UIR-PolyKernel    baselines/repos/UIR-PolyKernel/checkpoints/weight.pth (state dict, GitHub release 'Weight').
                    Official test: cv2 INTER_LINEAR resize to 256x256 (A.Resize), ToTensor, res.clamp(0,1).
                    Training data undisclosed -> C60 only.
  PhaseFormer-UFO120 checkpoints/UFO-120/best.pth; protocol as phase_e_pretrained.py. UFO-120 not on disk -> C60 only.
Outputs of 256-px protocols are resized back to the native size (bicubic) before metrics (disclosed).
Also saves outputs (C60 all 60; dark subset of 200 drawn with rng 20261117) of every method incl. P1b_s0 and the
retrained baselines, for the CPU downstream check (scripts/diagnostics/downstream_cv.py).
"""
from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "src/validation"), str(ROOT / "baselines/adapters")]
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms as TT  # noqa: E402
from PIL import Image  # noqa: E402
from common import git_state, load_u8, psnr_u8, ssim_u8, u8_to_tensor, to_u8_tensor, write_json  # noqa: E402
import phase_e_eval as PE  # noqa: E402
import phase_e_pretrained as PP  # noqa: E402
import registry as RG  # noqa: E402
from uw_metrics import uiqm, uciqe  # noqa: E402

MAR = ROOT / "src/restoration/repositories/MARINE-Net/MARINE-Net-main"
UIR = ROOT / "baselines/repos/UIR-PolyKernel"
FIVE = ROOT / "baselines/repos/FiveAPlus-Network/model"
PFU = ROOT / "baselines/repos/PhaseFormer/checkpoints/UFO-120/best.pth"
PHASE_E = ROOT / "runs/diagnostics/PHASE_E_20261117_182411"


def back(o8, hw):
    return o8 if o8.shape[:2] == hw else np.asarray(Image.fromarray(o8).resize((hw[1], hw[0]), Image.BICUBIC))


def marine(dev):
    sys.path.insert(0, str(MAR))
    from marine_net_model import create_model
    raw = zipfile.ZipFile(MAR / "marine_lsui_best.zip").read("kaggle/working/checkpoints/marine_net_lsui_finetuned/best_finetuned.pth")
    ck = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    m = create_model(); m.load_state_dict(ck["model_state_dict"]); m.to(dev).eval()
    tf = TT.Compose([TT.Resize((256, 256)), TT.ToTensor()])

    @torch.no_grad()
    def f(x8):
        o = m(tf(Image.fromarray(x8))[None].to(dev))["enhanced"].clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        return back((o * 255 + 0.5).astype(np.uint8), x8.shape[:2])
    return f


def fiveaplus(dev, alpha):
    m = RG.build("fiveaplus")
    m.load_state_dict(torch.load(FIVE / f"FAPlusNet-alpha-{alpha}.pth", map_location="cpu", weights_only=True), strict=True)
    m.to(dev).eval()

    @torch.no_grad()
    def f(x8):
        return to_u8_tensor(RG.predict("fiveaplus", m, u8_to_tensor(x8).to(dev)).float().cpu())
    return f


def _mmcv_shim():
    """UIR-PolyKernel imports mmcv only for build_norm_layer(dict(type='SyncBN'), C) -> ('bn', nn.SyncBatchNorm(C)).
    mmcv is not installed; this shim returns the same module type with identical parameter/buffer names (verified by a
    strict state-dict load). In eval mode SyncBatchNorm uses running statistics, i.e. it is exact BatchNorm inference."""
    import types
    import torch.nn as nn
    if "mmcv.cnn" in sys.modules:
        return
    cnn = types.ModuleType("mmcv.cnn")

    def build_norm_layer(cfg, num_features):
        if cfg.get("type") not in ("SyncBN", "BN", "BN2d"):
            raise NotImplementedError(cfg)
        layer = nn.SyncBatchNorm(num_features)
        for p in layer.parameters():
            p.requires_grad = cfg.get("requires_grad", True)
        return "bn", layer
    cnn.build_norm_layer = build_norm_layer
    mm = types.ModuleType("mmcv"); mm.cnn = cnn
    sys.modules["mmcv"], sys.modules["mmcv.cnn"] = mm, cnn


def uir(dev):
    _mmcv_shim()
    sys.path.insert(0, str(UIR))
    from models import UIR_PolyKernel
    sd = torch.load(UIR / "checkpoints/weight.pth", map_location="cpu", weights_only=True)["state_dict"]
    m = UIR_PolyKernel(); m.load_state_dict({k[7:] if k.startswith("module") else k: v for k, v in sd.items()}, strict=True)
    m.to(dev).eval()

    @torch.no_grad()
    def f(x8):
        a = cv2.resize(x8, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        o = m(torch.from_numpy(a).permute(2, 0, 1)[None].to(dev)).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        return back((o * 255 + 0.5).astype(np.uint8), x8.shape[:2])
    return f


def phaseformer(dev, ck):
    old = PP.CKPT; PP.CKPT = ck
    net, _ = PP.load_phaseformer(dev); PP.CKPT = old
    return lambda x8: PP.enhance(net, x8, dev)


def main():
    commit = git_state(); dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(0.50, 0)
    out = ROOT / "runs/diagnostics" / f"PHASE_E_PRETRAINED2_{datetime.now():%Y%m%d_%H%M%S}"; out.mkdir(parents=True)
    lp = PE.pin_lpips(dev)
    t = lambda u: (torch.from_numpy(u).permute(2, 0, 1)[None].float().div(127.5) - 1.0).to(dev)  # noqa: E731
    pairs = PE.dark_pairs(); c60 = sorted(p for p in PE.C60.iterdir() if p.suffix.lower() in PE.IMG)
    sub = set(int(i) for i in np.random.default_rng(20261117).choice(len(pairs), 200, replace=False))
    ext = {"MARINE-Net-LSUI": (marine(dev), True), "FiveA+_official_a0.1": (fiveaplus(dev, "0.1"), True),
           "FiveA+_official_a0.4": (fiveaplus(dev, "0.4"), True), "UIR-PolyKernel": (uir(dev), False),
           "PhaseFormer-UFO120": (phaseformer(dev, PFU), False),
           "PhaseFormer-UIEB": (phaseformer(dev, PP.ROOT / "baselines/repos/PhaseFormer/checkpoints/UIEB/best.pth"), True)}
    p1b = PE.P1bRunner(dev)
    ours = {"P1b_s0": lambda x8: to_u8_tensor(p1b(x8)[0]["J"])}
    for n, (ck, sh) in PE.BASELINES.items():
        br = PE.BaselineRunner(n, str(ROOT / ck), sh)
        ours[n + "_retrained"] = (lambda br: lambda x8: to_u8_tensor(br(u8_to_tensor(x8))[0]))(br)
    summ = {"commit": commit, "absolute_dark": {}, "absolute_c60": {}, "paired_P1b_minus": {}}
    p1b_rows = {s: {r["image"]: r for r in csv.DictReader(open(PHASE_E / f"{s}_P1b_s0.csv", encoding="utf-8"))} for s in ("dark", "c60")}
    for name, (f, on_dark) in ext.items():
        rows_d, rows_c = [], []
        with torch.no_grad():
            if on_dark:
                for i, (p, q) in enumerate(pairs):
                    x8, y8 = load_u8(p), load_u8(q); o = f(x8)
                    rows_d.append({"image": p.name, "psnr8": psnr_u8(o, y8), "ssim8": ssim_u8(o, y8), "lpips": float(lp(t(o), t(y8)))})
                    if i in sub:
                        d = out / "outputs" / "dark" / name; d.mkdir(parents=True, exist_ok=True); Image.fromarray(o).save(d / (p.stem + ".png"))
            for p in c60:
                o = f(load_u8(p)); rows_c.append({"image": p.name, "uiqm": uiqm(o)["uiqm"], "uciqe": uciqe(o)["uciqe"]})
                d = out / "outputs" / "c60" / name; d.mkdir(parents=True, exist_ok=True); Image.fromarray(o).save(d / (p.stem + ".png"))
        if rows_d:
            summ["absolute_dark"][name] = {m: float(np.mean([r[m] for r in rows_d])) for m in ("psnr8", "ssim8", "lpips")}
            summ["paired_P1b_minus"].setdefault(name, {})["dark"] = {m: PE.boot([float(p1b_rows["dark"][r["image"]][m]) - r[m] for r in rows_d]) for m in ("psnr8", "ssim8", "lpips")}
        summ["absolute_c60"][name] = {m: float(np.mean([r[m] for r in rows_c])) for m in ("uiqm", "uciqe")}
        summ["paired_P1b_minus"].setdefault(name, {})["c60"] = {m: PE.boot([float(p1b_rows["c60"][r["image"]][m]) - r[m] for r in rows_c]) for m in ("uiqm", "uciqe")}
        print(name, summ["absolute_dark"].get(name), summ["absolute_c60"][name], flush=True)
    with torch.no_grad():                                   # outputs of our model and retrained baselines for downstream
        for name, f in {**ours, "input": lambda x8: x8}.items():
            for i in sorted(sub):
                p, _ = pairs[i]; d = out / "outputs" / "dark" / name; d.mkdir(parents=True, exist_ok=True)
                Image.fromarray(f(load_u8(p))).save(d / (p.stem + ".png"))
            for p in c60:
                d = out / "outputs" / "c60" / name; d.mkdir(parents=True, exist_ok=True); Image.fromarray(f(load_u8(p))).save(d / (p.stem + ".png"))
    write_json(out / "summary.json", summ); print("[pretrained2] done", out, flush=True)


if __name__ == "__main__":
    main()

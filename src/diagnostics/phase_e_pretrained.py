"""
Phase E add-on: EXTERNAL PRETRAINED checkpoint, zero-shot sets only (never LSUI/UIEB val/test). EXPLORATORY.

Model: PhaseFormer (WACV 2025) official checkpoint baselines/repos/PhaseFormer/checkpoints/UIEB/best.pth
  (repo @6e759dc; full-module pickle of model_with_eca.Restormer; pickletools inspection on 2026-11-17 showed only
  torch / collections / builtins-set / model_with_eca globals, i.e. no code-execution primitives).
Official test protocol, reproduced from the repo's test.py / dataset.py / utils.py: bicubic resize to 512x512,
  Normalize(0.5, 0.5) to [-1, 1], output = model(x)[1], image = clip((y + 1) / 2 * 255). For full-reference scoring the
  512x512 output is resized back to the native size (bicubic) -- our adaptation, disclosed.
Training data: UIEB (per the checkpoint folder name and the paper); EUVP underwater_dark is a different dataset and
  Challenge-60 is disjoint from UIEB's 890 paired images by construction. Not hash-verified against the authors' exact
  training list (not published in the repo).
Sets/metrics identical to src/diagnostics/phase_e_eval.py; paired against P1b_s0's Phase E per-image records by image name.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "src/validation")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from common import git_state, load_u8, psnr_u8, ssim_u8, write_json  # noqa: E402
import phase_e_eval as PE  # noqa: E402
from uw_metrics import uiqm, uciqe  # noqa: E402

REPO = ROOT / "baselines/repos/PhaseFormer"
CKPT = REPO / "checkpoints/UIEB/best.pth"
PHASE_E = ROOT / "runs/diagnostics/PHASE_E_20261117_182411"


def load_phaseformer(dev):
    sha = hashlib.sha256(CKPT.read_bytes()).hexdigest()
    sys.path.insert(0, str(REPO))
    try:
        net = torch.load(CKPT, map_location="cpu", weights_only=False)   # inspected full-module pickle (see docstring)
    finally:
        sys.path.remove(str(REPO))
    return net.to(dev).eval(), sha


@torch.no_grad()
def enhance(net, x8, dev):
    H, W = x8.shape[:2]
    a = np.asarray(Image.fromarray(x8).resize((512, 512), Image.BICUBIC), dtype=np.float32) / 255.0
    t = torch.from_numpy((a - 0.5) / 0.5).permute(2, 0, 1)[None].to(dev)
    y = net(t)[1][0].float().cpu().numpy()
    o = np.clip((np.transpose(y, (1, 2, 0)) + 1) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(o).resize((W, H), Image.BICUBIC))


def main():
    commit = git_state(); dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(0.50, 0)
    out = ROOT / "runs/diagnostics" / f"PHASE_E_PRETRAINED_{datetime.now():%Y%m%d_%H%M%S}"; out.mkdir(parents=True)
    net, sha = load_phaseformer(dev)
    lp = PE.pin_lpips(dev)
    t = lambda u: (torch.from_numpy(u).permute(2, 0, 1)[None].float().div(127.5) - 1.0).to(dev)  # noqa: E731
    dark = []
    for i, (p, q) in enumerate(PE.dark_pairs()):
        x8, y8 = load_u8(p), load_u8(q); o = enhance(net, x8, dev)
        with torch.no_grad():
            dark.append({"image": p.name, "psnr8": psnr_u8(o, y8), "ssim8": ssim_u8(o, y8), "lpips": float(lp(t(o), t(y8)))})
        if (i + 1) % 1000 == 0:
            print(f"[dark] {i + 1}", flush=True)
    c60 = []
    for p in sorted(x for x in PE.C60.iterdir() if x.suffix.lower() in PE.IMG):
        o = enhance(net, load_u8(p), dev); c60.append({"image": p.name, "uiqm": uiqm(o)["uiqm"], "uciqe": uciqe(o)["uciqe"]})
    for name, rows in (("dark", dark), ("c60", c60)):
        with open(out / f"{name}_phaseformer_pretrained_UIEB.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    p1b = {s: {r["image"]: r for r in csv.DictReader(open(PHASE_E / f"{s}_P1b_s0.csv", encoding="utf-8"))} for s in ("dark", "c60")}
    paired = {"dark": {m: PE.boot([float(p1b["dark"][r["image"]][m]) - r[m] for r in dark]) for m in ("psnr8", "ssim8", "lpips")},
              "c60": {m: PE.boot([float(p1b["c60"][r["image"]][m]) - r[m] for r in c60]) for m in ("uiqm", "uciqe")}}
    summ = {"commit": commit, "model": "PhaseFormer official UIEB checkpoint (external pretrained)", "ckpt_sha256": sha,
            "protocol": "official: bicubic 512, [-1,1], output[1]; resized back to native (bicubic) for FR metrics",
            "absolute_dark": {m: float(np.mean([r[m] for r in dark])) for m in ("psnr8", "ssim8", "lpips")},
            "absolute_c60": {m: float(np.mean([r[m] for r in c60])) for m in ("uiqm", "uciqe")},
            "paired_P1b_minus_pretrained": paired, "n": {"dark": len(dark), "c60": len(c60)}}
    write_json(out / "summary.json", summ); print(json.dumps({k: summ[k] for k in ("absolute_dark", "absolute_c60")}, indent=1))


if __name__ == "__main__":
    main()

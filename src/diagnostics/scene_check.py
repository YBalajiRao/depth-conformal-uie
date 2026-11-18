"""
Scene-prior checks (CPU; no training; roadmap phase C, PI decision 2026-11-13).

1. Identity: a scene_prior model holding the P1b seed-0 last.pt weights plus zero-initialised scene layers produces
   outputs bit-identical to P1b for ANY descriptor g (random inputs, and 4 real validation images at native size
   through PadCropWrapper with their real depth maps). Only scene.* parameters are missing from the P1b state dict.
   Control: a 1e-3 change in one scene_film weight must break the identity.
2. Gradient routing: at zero init the restoration loss gives non-zero gradient to scene_film and scene_dgt (weights
   and biases) and exactly zero to the trunk Linear(768 -> 64) (its only downstream layers are zero). After the
   output layers become non-zero, the trunk receives gradient and the DGT head's beta responds to g.
3. Contract: g missing / wrong shape rejected; g without scene_prior rejected; scene_prior without DGT rejected;
   parameter count = P1b + scene layers and <= 5M; PadCropWrapper passes (N, 768) through unpadded.
Exit 1 on any failure.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("src/validation", "src/utils", "src/training", "src/models"):
    sys.path.insert(0, str(ROOT / sub))
import torch  # noqa: E402
import train as T  # noqa: E402
from ss_pgn import SSPGN, SSPGNConfig  # noqa: E402
from pad_crop_wrapper import PadCropWrapper  # noqa: E402

P1B = ROOT / "runs/phase1/P1b_v2_dgt_ordered/20261110_093229/last.pt"
P1B_SHA = "5b22ea7c217b09b96ad2b6309b21b5cbca906444e2f442dd9732965ec9db52c0"
FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAIL.append(name)


def control(name, rejected):
    print(f"[{'CTRL-OK' if rejected else 'CTRL-FAIL'}] {name}", flush=True)
    if not rejected:
        FAIL.append("control: " + name)


def same(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in b)


def rejected(fn):
    try:
        fn()
        return False
    except (AssertionError, ValueError):
        return True


def main():
    torch.set_num_threads(8)
    if hashlib.sha256(P1B.read_bytes()).hexdigest() != P1B_SHA:
        raise SystemExit("[abort] P1b last.pt SHA-256 mismatch")
    ck = torch.load(P1B, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"]["model_cfg"])
    base = SSPGN(SSPGNConfig(**cfg)).eval(); base.load_state_dict(ck["model"], strict=True)
    sc = SSPGN(SSPGNConfig(**{**cfg, "scene_prior": True})).eval()
    res = sc.load_state_dict(ck["model"], strict=False)
    n_sc = sum(p.numel() for m in (sc.scene, sc.scene_film, sc.scene_dgt) for p in m.parameters())
    check("only scene.* parameters missing from the P1b state dict",
          all(k.startswith(("scene.", "scene_film.", "scene_dgt.")) for k in res.missing_keys)
          and len(res.missing_keys) == 6 and not res.unexpected_keys, f"({len(res.missing_keys)} missing)")
    check("parameter count = P1b + scene layers, <= 5M", sc.n_params == base.n_params + n_sc and sc.n_params <= 5_000_000,
          f"({base.n_params:,} + {n_sc:,} = {sc.n_params:,})")

    torch.manual_seed(0)
    x, z = torch.rand(2, 3, 64, 96), torch.rand(2, 1, 64, 96)
    ok = True
    for s in range(3):
        g = torch.randn(2, 768) * (1 + 3 * s)
        with torch.no_grad():
            ok &= same(sc(x, z, g), base(x, z))
    check("zero init: bit-identical to P1b for random descriptors (3 scales)", ok)

    split = ROOT / "data/splits/lsui_uieb_all_d6_seed0"
    rows = T.read_manifest(split / "val_manifest.txt")
    rows = [r for r in rows if r[0] == "UIEB"][:2] + [r for r in rows if r[0] == "LSUI"][:2]
    store = T.DepthStore(ROOT / "data/depth/da2_small_5426e4f" / split.name, rows, check_sha=True)
    ok = True
    for r in rows:
        x8 = T.load_rgb_u8(r[2]); xi = T.to_tensor(x8)[None]; zi = store.load(r[1], x8.shape[:2])[None]
        gi = torch.randn(1, 768)
        with torch.no_grad():
            ok &= same(PadCropWrapper(sc, 16)(xi, zi, gi), PadCropWrapper(base, 16)(xi, zi))
    check("zero init: bit-identical on 4 real val images at native size (PadCropWrapper, (N,768) passed unpadded)", ok)
    with torch.no_grad():
        sc.scene_film.weight[0, 0] += 1e-3
        o2 = sc(x, z, torch.randn(2, 768))
        sc.scene_film.weight[0, 0] -= 1e-3
    control("identity check rejects a 1e-3 scene_film weight change", not same(o2, base(x, z)))

    fresh = SSPGN(SSPGNConfig(**{**cfg, "scene_prior": True})); fresh.load_state_dict(ck["model"], strict=False)
    fresh.train()
    g = torch.randn(2, 768); y = torch.rand_like(x)
    (fresh(x, z, g)["J"] - y).abs().mean().backward()
    gf = [p.grad.abs().sum().item() for p in fresh.scene_film.parameters()]
    gd = [p.grad.abs().sum().item() if p.grad is not None else 0.0 for p in fresh.scene_dgt.parameters()]
    gt = [p.grad.abs().sum().item() if p.grad is not None else 0.0 for p in fresh.scene.parameters()]
    check("zero init: scene_film weight+bias receive non-zero gradient", all(v > 0 for v in gf), f"({gf})")
    check("zero init: scene_dgt weight+bias receive non-zero gradient", all(v > 0 for v in gd), f"({gd})")
    check("zero init: trunk Linear(768->64) receives exactly zero gradient", all(v == 0 for v in gt), f"({gt})")
    fresh.zero_grad()
    with torch.no_grad():
        fresh.scene_film.weight.normal_(0, 1e-2); fresh.scene_dgt.weight.normal_(0, 1e-1)
    o_a, o_b = fresh(x, z, g), fresh(x, z, -g)
    (o_a["J"] - y).abs().mean().backward()
    gt2 = sum(p.grad.abs().sum().item() for p in fresh.scene.parameters() if p.grad is not None)
    check("after output layers are non-zero: trunk receives gradient", gt2 > 0, f"({gt2:.2e})")
    check("after output layers are non-zero: DGT beta depends on g", not torch.equal(o_a["beta_D"], o_b["beta_D"]),
          f"(max|d beta_D| {(o_a['beta_D'] - o_b['beta_D']).abs().max():.2e})")
    check("ordering preserved with the scene term (beta_R >= beta_G >= beta_B)",
          bool((o_a["beta_D"][:, 0] >= o_a["beta_D"][:, 1]).all() and (o_a["beta_D"][:, 1] >= o_a["beta_D"][:, 2]).all()))

    check("g missing rejected", rejected(lambda: sc(x, z)))
    check("g of wrong size rejected", rejected(lambda: sc(x, z, torch.randn(2, 512))))
    check("g without scene_prior rejected", rejected(lambda: base(x, z, torch.randn(2, 768))))
    check("scene_prior without DGT rejected",
          rejected(lambda: SSPGN(SSPGNConfig(phys_conditioning="v2", scene_prior=True))))
    print(f"\n[scene_check] {'ALL PASS' if not FAIL else 'FAILED: ' + '; '.join(FAIL)}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

"""
Decoder-FiLM checks (CPU; no training). PI decision 2026-11-10.

1. Identity: a dec_film model holding the P1b (v2 + ordered DGT) last.pt weights plus zero-initialised FiLM convs
   produces outputs bit-identical to the P1b model, on random inputs and on real validation images with their
   depth maps (via PadCropWrapper). Only the FiLM parameters are missing from the P1b state dict.
   Control: a 1e-3 change in one FiLM weight must break the identity.
2. Gradient flow at zero init: a restoration loss gives non-zero gradients on every FiLM conv weight and bias.
   After one non-zero FiLM weight is set, restoration gradient also reaches the DGT head through the FiLM path
   (it is exactly zero through that path at zero init, by construction).
3. Contract: dec_film without DGT is rejected; parameter count = P1b + FiLM.
Exit 1 on any failure.
"""
from __future__ import annotations

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


def same_out(a, b):
    return set(a) - {"film_mag"} == set(b) and all(torch.equal(a[k], b[k]) for k in b)


def main():
    import hashlib
    torch.set_num_threads(8)
    if hashlib.sha256(P1B.read_bytes()).hexdigest() != P1B_SHA:
        raise SystemExit("[abort] P1b last.pt SHA-256 mismatch")
    ck = torch.load(P1B, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"]["model_cfg"])
    base = SSPGN(SSPGNConfig(**cfg)).eval()
    base.load_state_dict(ck["model"], strict=True)
    film = SSPGN(SSPGNConfig(**{**cfg, "dec_film": True})).eval()
    res = film.load_state_dict(ck["model"], strict=False)
    n_film = sum(p.numel() for p in film.film.parameters())
    check("only FiLM parameters are missing from the P1b state dict; no unexpected keys",
          all(k.startswith("film.") for k in res.missing_keys) and len(res.missing_keys) == 6
          and not res.unexpected_keys, f"(missing {len(res.missing_keys)})")
    check("parameter count = P1b + FiLM", film.n_params == base.n_params + n_film,
          f"({base.n_params:,} + {n_film:,} = {film.n_params:,})")

    torch.manual_seed(0)
    x, z = torch.rand(2, 3, 64, 96), torch.rand(2, 1, 64, 96)
    with torch.no_grad():
        ob, of = base(x, z), film(x, z)
    check("zero-init FiLM: all outputs bit-identical to P1b (random input)", same_out(of, ob))
    check("film_mag == 0 at zero init", float(of["film_mag"]) == 0.0)

    split = ROOT / "data/splits/lsui_uieb_all_d6_seed0"
    rows = T.read_manifest(split / "val_manifest.txt")
    rows = [r for r in rows if r[0] == "UIEB"][:2] + [r for r in rows if r[0] == "LSUI"][:2]
    store = T.DepthStore(ROOT / "data/depth/da2_small_5426e4f" / split.name, rows, check_sha=True)
    ok = True
    for r in rows:
        x8 = T.load_rgb_u8(r[2])
        xi = T.to_tensor(x8)[None]
        zi = store.load(r[1], x8.shape[:2])[None]
        with torch.no_grad():
            ok &= same_out(PadCropWrapper(film, 16)(xi, zi), PadCropWrapper(base, 16)(xi, zi))
    check("zero-init FiLM: bit-identical to P1b on 4 real validation images (native size, padded)", ok)

    with torch.no_grad():
        film.film[0].weight[0, 0] += 1e-3
        of2 = film(x, z)
        film.film[0].weight[0, 0] -= 1e-3
    control("identity check rejects a 1e-3 FiLM weight change", not same_out(of2, ob))

    # gradient flow
    fresh = SSPGN(SSPGNConfig(**{**cfg, "dec_film": True}))
    fresh.load_state_dict(ck["model"], strict=False)
    fresh.train()
    y = torch.rand_like(x)
    (fresh(x, z)["J"] - y).abs().mean().backward()
    gw = [(c.weight.grad.abs().sum().item(), c.bias.grad.abs().sum().item()) for c in fresh.film]
    check("zero init: every FiLM conv weight and bias receives non-zero gradient", all(a > 0 and b > 0 for a, b in gw),
          f"({[(f'{a:.1e}', f'{b:.1e}') for a, b in gw]})")
    g_dgt0 = sum(p.grad.abs().sum().item() for p in fresh.dgt.parameters() if p.grad is not None)
    fresh.zero_grad()
    with torch.no_grad():
        for c in fresh.film:
            c.weight.normal_(0, 1e-2)
    (fresh(x, z)["J"] - y).abs().mean().backward()
    g_dgt1 = sum(p.grad.abs().sum().item() for p in fresh.dgt.parameters() if p.grad is not None)
    check("after FiLM weights become non-zero, restoration gradient into the DGT head increases (FiLM path)",
          g_dgt1 > g_dgt0, f"({g_dgt0:.2e} -> {g_dgt1:.2e})")

    try:
        SSPGN(SSPGNConfig(phys_conditioning="v2", dec_film=True)); check("dec_film without DGT rejected", False)
    except (AssertionError, ValueError):
        check("dec_film without DGT rejected", True)

    print(f"\n[film_check] {'ALL PASS' if not FAIL else 'FAILED: ' + '; '.join(FAIL)}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

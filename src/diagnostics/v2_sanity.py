"""
v2 pre-training sanity checks (CPU only, deterministic, no training). Spec: execution report section G.

S0 v1 checkpoint SHA-256 before/after          S6 gradient routing from the RESTORATION loss only
S1 v1 default == HEAD model, bit-identical      S7 B_hat (H/16) vs full-resolution physics-loss B
S2 v2 tensor shapes at 256x256 + params         S8 matched initialisation (same seed)
S3 call order (acyclic: gphys before gate)      S9 finite forward/loss/gradients (full objective)
S4/S5 forward sensitivity of mu to B_inf / beams (v2 > 0, v1 = 0) and to M / kappa (expected 0 in both)
  python src/diagnostics/v2_sanity.py --head-model <copy of HEAD ss_pgn.py>
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
assert not torch.cuda.is_available(), "CPU only"
from common import ROOT, CKPTS, sha256_file, val_rows, load_u8, u8_to_tensor, fixed_subset  # noqa: E402
sys.path[:0] = [str(ROOT / "src/models"), str(ROOT / "src/training"), str(ROOT / "src/utils")]
from ss_pgn import SSPGN, SSPGNConfig, render_beams  # noqa: E402
from losses_sspgn import SSPGNLoss, mmean, ssim, haar_dwt  # noqa: E402
from pad_crop_wrapper import PadCropWrapper  # noqa: E402

RES = []


def check(name, ok, detail=""):
    RES.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def rec_loss(out, y):
    J = out["J"]
    l = mmean((J - y).abs()) + 0.2 * (1 - ssim(J.clamp(0, 1), y))
    _, a, b, c = haar_dwt(J); _, ya, yb, yc = haar_dwt(y)
    return l + 0.1 * ((a - ya).abs().mean() + (b - yb).abs().mean() + (c - yc).abs().mean())


def gnorm(params):
    gs = [p.grad for p in params if p.grad is not None]
    return float(sum(g.norm() ** 2 for g in gs)) ** 0.5 if gs else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--head-model", required=True); args = ap.parse_args()
    torch.manual_seed(0)
    path, sha_exp, _ = CKPTS["last"]
    sha0 = sha256_file(path)
    check("S0a v1 checkpoint SHA before", sha0 == sha_exp, sha0)
    ck = torch.load(path, map_location="cpu", weights_only=False); mc = ck["cfg"]["model_cfg"]

    # S1 v1 default bit-identical to HEAD implementation
    old = load_module(args.head_model, "ss_pgn_head")
    mo = old.SSPGN(old.SSPGNConfig(**mc)).eval(); mo.load_state_dict(ck["model"], strict=True)
    m1 = SSPGN(SSPGNConfig(**mc)).eval(); m1.load_state_dict(ck["model"], strict=True)
    rows = fixed_subset(val_rows(), 40, 0)[:3] + [r for r in val_rows() if r["dataset"] == "UIEB"][:2]
    same = True
    for r in rows:
        x = u8_to_tensor(load_u8(r["input"]))
        with torch.no_grad():
            a, b = PadCropWrapper(mo, 16, "reflect")(x), PadCropWrapper(m1, 16, "reflect")(x)
        same &= all(torch.equal(a[k], b[k]) for k in a)
    check("S1 v1 default == HEAD model on 5 val images (all outputs bit-identical)", same)

    # S2 shapes/params (untrained v2, seed 0) + S3 call order
    torch.manual_seed(0); v2 = SSPGN(SSPGNConfig(phys_conditioning="v2")).eval()
    torch.manual_seed(0); v1i = SSPGN(SSPGNConfig()).eval()
    check("S2a parameter count v2 == v1 == 4,152,052", v2.n_params == v1i.n_params == 4_152_052, f"({v2.n_params:,})")
    S, order = {}, []
    blk = v2.bottleneck[0]
    blk.to_groups.register_forward_pre_hook(lambda m, i: S.__setitem__("LL", tuple(i[0].shape)) or None)
    v2.gphys.register_forward_pre_hook(lambda m, i: (S.__setitem__("gphys_in", tuple(i[0].shape)), order.append("gphys"))[-1] and None)
    v2.gphys.register_forward_hook(lambda m, i, o: S.__setitem__("g", {k: tuple(v.shape) for k, v in o.items()}) or None)
    blk.gate.register_forward_pre_hook(lambda m, i: (S.__setitem__("gate_in", tuple(i[0].shape)), order.append("gate"))[-1] and None)
    blk.film.register_forward_pre_hook(lambda m, i: (S.__setitem__("film_in", tuple(i[0].shape)), S.__setitem__("pi", i[0].detach().clone()))[-1] and None)
    x256 = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        o = v2(x256)
    check("S2b LL (1,512,16,16)", S["LL"] == (1, 512, 16, 16), str(S["LL"]))
    check("S2c gphys input = LL (1,512,16,16)", S["gphys_in"] == (1, 512, 16, 16), str(S["gphys_in"]))
    check("S2d gphys outputs", S["g"] == {"B_inf": (1, 3), "M": (1, 3, 3), "kappa": (1, 3), "beam_mu": (1, 1, 2),
                                          "beam_sigma": (1, 1, 2), "beam_amp": (1, 1, 3)}, str(S["g"]))
    check("S2e Pi (1,9,16,16); gate in (1,521,16,16); FiLM in (1,9,16,16)",
          S["film_in"] == (1, 9, 16, 16) and S["gate_in"] == (1, 521, 16, 16), f"{S['film_in']} {S['gate_in']}")
    check("S2f full-res outputs", all(tuple(o[k].shape) == (1, 3, 256, 256) for k in ("J", "B", "tD", "tB", "J_phys"))
          and tuple(o["nig"].shape) == (1, 12, 256, 256))
    check("S3 acyclic call order: gphys before gate (v2)", order[:2] == ["gphys", "gate"], str(order))

    # S7 B_hat (Pi[:,6:9]) vs full-res physics-loss B: same params, different grids
    B_hat = S["pi"][:, 6:9]
    upB = F.interpolate(B_hat, size=(256, 256), mode="bilinear", align_corners=False)
    d = (upB - o["B"]).abs()
    print(f"  [info] S7 max|up(B_hat) - B_full| = {float(d.max()):.3e}, mean = {float(d.mean()):.3e} (bilinear + beam-grid + clamp differences; not expected 0)")

    # S4/S5 forward sensitivity on a real crop (trained v1 vs untrained v2 is irrelevant for the mechanism test)
    r = [r for r in val_rows() if r["dataset"] == "UIEB"][0]
    xc = u8_to_tensor(load_u8(r["input"])[:256, :256]); yc = u8_to_tensor(load_u8(r["reference"])[:256, :256])

    def pert(model, key, delta):
        def hk(m, i, out):
            out = dict(out); out[key] = out[key] + delta; return out
        hh = model.gphys.register_forward_hook(hk)
        with torch.no_grad():
            o2 = model(xc)
        hh.remove(); return o2

    # SS-PGB out_scale is zero-initialised (f1 = f0 + 0*y), so at initialisation NO conditioning (v1 fields or v2
    # B_hat) can reach mu and only out_scale gets gradient. The mechanism is therefore tested twice:
    #   v2_init : untrained v2, out_scale == 0  -> sensitivity and restoration gradient into gphys must be exactly 0
    #   v2_probe: v2 config with the trained v1 weights loaded (identical shapes; out_scale ~0.28) -> must be > 0.
    #             Mechanism probe only; these weights are never used to initialise v2 training.
    v2p = SSPGN(SSPGNConfig(**{**mc, "phys_conditioning": "v2"})).eval()
    v2p.load_state_dict(ck["model"], strict=True)
    check("S4pre v2 init has out_scale == 0", float(v2.bottleneck[0].out_scale.abs().max()) == 0.0)
    for name, model in (("v2_init", v2), ("v2_probe", v2p), ("v1", m1)):
        with torch.no_grad():
            base = model(xc)
        dm = {k: float((pert(model, k, dv)["J"] - base["J"]).abs().max()) for k, dv in
              (("B_inf", 0.2), ("beam_amp", 0.3), ("M", 0.1), ("kappa", 0.5))}
        if name == "v2_probe":
            check("S4a v2 (out_scale != 0): perturbing B_inf / beam_amp changes mu", dm["B_inf"] > 0 and dm["beam_amp"] > 0, str(dm))
            check("S5a v2: M / kappa do not reach mu (expected residual disconnection)", dm["M"] == 0 and dm["kappa"] == 0)
        elif name == "v2_init":
            check("S4b v2 at init (out_scale == 0): mu insensitive to all gphys outputs (expected)", all(v == 0 for v in dm.values()), str(dm))
        else:
            check("S5b v1: no gphys output reaches mu", all(v == 0 for v in dm.values()), str(dm))

    # S6 gradient routing from the restoration loss only; S9 full objective finite
    v2.zero_grad(set_to_none=True); rec_loss(v2(xc), yc).backward()
    check("S6pre v2 at init: restoration loss -> gphys exactly 0, out_scale grad > 0 (pathway opens after step 1)",
          gnorm(v2.gphys.parameters()) == 0 and gnorm([v2.bottleneck[0].out_scale]) > 0,
          f"(out_scale grad {gnorm([v2.bottleneck[0].out_scale]):.2e})")
    for name, model in (("v2", v2p), ("v1", m1)):
        model.zero_grad(set_to_none=True)
        rec_loss(model(xc), yc).backward()
        gp = model.gphys
        gb, gbeam, gm, gk, gmlp = (gnorm(gp.binf.parameters()), gnorm(gp.beam.parameters()), gnorm(gp.dM.parameters()),
                                   gnorm(gp.kappa.parameters()), gnorm(gp.mlp.parameters()))
        fin = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if name == "v2":
            check("S6a v2: restoration loss -> B_inf head, beam head, gphys MLP: non-zero & finite",
                  gb > 0 and gbeam > 0 and gmlp > 0 and fin, f"(binf {gb:.2e}, beam {gbeam:.2e}, mlp {gmlp:.2e})")
            check("S6b v2: restoration loss -> dM / kappa heads zero (not in B_hat)", gm == 0 and gk == 0)
        else:
            check("S6c v1: restoration loss -> gphys all zero", gb == gbeam == gm == gk == gmlp == 0)
        model.zero_grad(set_to_none=True)
        tot, _ = SSPGNLoss()(model(xc), xc, yc)
        tot.backward()
        fin = torch.isfinite(tot) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        check(f"S9 {name}: full objective finite loss and gradients", fin, f"(loss {float(tot):.4f})")

    # S8 matched initialisation
    def init(**kw):
        torch.manual_seed(0); return SSPGN(SSPGNConfig(**kw)).state_dict()
    a, b, c = init(), init(phys_conditioning="v2"), init(phys_conditioning="v2", field_conditioning=False)
    check("S8 seed-0 init identical: v1 == v2 == v2+no-conditioning", all(torch.equal(a[k], b[k]) and torch.equal(a[k], c[k]) for k in a))
    try:
        SSPGN(SSPGNConfig(phys_conditioning="v2", bottleneck_blocks=2)); check("S8b v2 rejects bottleneck_blocks=2", False)
    except ValueError:
        check("S8b v2 rejects bottleneck_blocks=2", True)

    sha1 = sha256_file(path)
    check("S0b v1 checkpoint SHA after (unchanged)", sha1 == sha0, sha1)
    n_ok = sum(ok for _, ok in RES)
    print(f"\n{n_ok}/{len(RES)} passed")
    sys.exit(0 if n_ok == len(RES) else 1)


if __name__ == "__main__":
    main()

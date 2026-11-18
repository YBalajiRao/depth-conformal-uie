"""
CPU checks for the Phase-1 scaffolds (src/models/dgt_head.py, src/models/csa.py). No GPU, no data, no training.

Each property check is paired with a failure control: a deliberately broken variant that the same check
must reject. The script exits non-zero if any property fails or any control is not rejected.
"""
from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/models"))
import torch  # noqa: E402
from dgt_head import DGTHead  # noqa: E402
from csa import ChromaStyleAdapter, pool_stats  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def control(name, rejected):
    print(f"[{'CTRL-OK' if rejected else 'CTRL-FAIL'}] {name}")
    if not rejected:
        FAIL.append("control: " + name)


# ------------------------------------------------------------------ DGT
def bounds_ok(beta, lo, hi):
    return bool((beta >= lo).all() and (beta <= hi).all())


def dgt_checks():
    torch.manual_seed(0)
    C, N = 32, 4
    feat = torch.randn(N, C, 8, 8)
    z = torch.linspace(0, 1, 64).view(1, 1, 8, 8).expand(N, 1, 8, 8).contiguous()

    h = DGTHead(C, depth_mode="one_minus")
    o = h(feat, z)
    check("DGT n_params == 9", h.n_params == 9 and o["beta_D"].shape == (N, 3) and o["B_inf"].shape == (N, 3))
    check("DGT init beta = 0.50 (softplus init), B_inf = 0.5",
          torch.allclose(o["beta_D"], torch.full_like(o["beta_D"], 0.5), atol=1e-6) and
          torch.allclose(o["beta_B"], torch.full_like(o["beta_B"], 0.5), atol=1e-6) and
          torch.allclose(o["B_inf"], torch.full_like(o["B_inf"], 0.5)))

    # softplus parameterisation: beta > b_min always; d beta/d u > 0 everywhere; no hard ceiling at b_max
    u = torch.linspace(-30, 30, 2001, requires_grad=True)
    beta_u = h.b_min + torch.nn.functional.softplus(u + h.u_init)
    g = torch.autograd.grad(beta_u.sum(), u)[0]
    # In exact arithmetic beta > b_min; in float32 b_min + softplus(.) rounds to b_min once softplus(.) is
    # below half an ulp of 0.05 (first run of this check: strict '>' failed at u = -30). Stated as >=, and
    # the u at which equality starts is reported.
    eq = (beta_u.detach() == h.b_min)
    u_eq = float(u[eq].max()) if eq.any() else float("nan")
    check("DGT beta >= b_min for all u (float32)", bool((beta_u >= h.b_min).all()),
          f"(beta == b_min exactly for u <= {u_eq:.2f}; d beta/d u there <= {float(g[eq].max()) if eq.any() else 0:.1e})")
    check("DGT d beta / d u > 0 for all u, ~1 above init (non-saturating at the top)",
          bool((g > 0).all()) and float(g[-1]) > 0.999, f"(min {g.min():.2e}, at u=+30: {g[-1]:.4f})")
    sig = h.b_min + (h.b_max - h.b_min) * torch.sigmoid(u)
    gs = torch.autograd.grad(sig.sum(), u)[0]
    control("top-saturation check rejects the old sigmoid map", not (float(gs[-1]) > 0.999))
    check("DGT beta can exceed b_max (soft upper bound)", float(beta_u.max()) > h.b_max)

    with torch.no_grad():                                 # extreme weights: lower bound must still hold
        h.mlp[-1].weight.normal_(0, 1e3); h.mlp[-1].bias.normal_(0, 1e3)
    o = h(feat * 100, z)
    allb = torch.cat([o["beta_D"], o["beta_B"]], 1)
    check("DGT beta >= b_min under extreme weights", bounds_ok(allb, 0.05, float("inf")),
          f"min {allb.min():.4f} max {allb.max():.4f}")
    check("DGT t in (0, 1]", bool((o["t_D"] >= 0).all() and (o["t_D"] <= 1).all()))
    unbounded = torch.cat([h.mlp(torch.cat([(feat * 100).mean((2, 3)), (feat * 100).amax((2, 3))], 1))[:, :6]], 1)
    control("lower-bound check rejects raw (un-reparameterised) logits", not bounds_ok(unbounded, 0.05, float("inf")))

    # soft barrier: zero inside, positive above b_max, gradient pushes beta down
    bi = torch.tensor([[0.08, 1.0, 2.5]], requires_grad=True)
    bo = torch.tensor([[0.5, 2.6, 4.0]], requires_grad=True)
    check("barrier == 0 for beta in [b_floor, b_max]", float(h.barrier(bi)) == 0.0)
    bl = torch.tensor([[0.06, 0.5, 0.0501]], requires_grad=True)
    pl = h.barrier(bl)
    gl = torch.autograd.grad(pl, bl)[0]
    check("floor barrier > 0 below b_floor; gradient pushes beta up there and is 0 inside",
          float(pl) > 0 and bool((gl[0, [0, 2]] < 0).all()) and float(gl[0, 1]) == 0,
          f"(value {float(pl):.4f}; dL/dbeta at 0.06 {float(gl[0, 0]):.2f}, at 0.0501 {float(gl[0, 2]):.2f})")
    up, lo = h.barrier_parts(bo)
    check("barrier_parts: upper-only for beta above b_max", float(up) > 0 and float(lo) == 0)
    pb = h.barrier(bo)
    gb = torch.autograd.grad(pb, bo)[0]
    check("barrier > 0 above b_max, gradient only on the exceeding entries and positive",
          float(pb) > 0 and float(gb[0, 0]) == 0 and bool((gb[0, 1:] > 0).all()), f"(value {float(pb):.4f})")
    control("barrier check rejects a penalty that is active inside the range",
            not (float(((bi - 1.0) ** 2).mean()) == 0.0))

    # ordered (Jerlov-type) chain
    ho = DGTHead(C, depth_mode="one_minus", ordered=True)
    oo = ho(feat, z)
    init_ok = all(torch.allclose(oo[k], torch.tensor([[0.70, 0.45, 0.30]]).expand(N, 3), atol=1e-6)
                  for k in ("beta_D", "beta_B"))
    check("ordered DGT init beta (R, G, B) = (0.70, 0.45, 0.30)", init_ok)
    with torch.no_grad():
        ho.mlp[-1].weight.normal_(0, 30.0); ho.mlp[-1].bias.normal_(0, 30.0)
    ok_ord, strict_ok = True, True
    for s in range(20):
        torch.manual_seed(100 + s)
        ob = ho(torch.randn(N, C, 8, 8) * 3, z)
        for k in ("beta_D", "beta_B"):
            b = ob[k]
            ok_ord &= bool((b[:, 0] >= b[:, 1]).all() and (b[:, 1] >= b[:, 2]).all() and (b[:, 2] >= 0.05).all())
            strict_ok &= bool((b[:, 0] > b[:, 1]).all() and (b[:, 1] > b[:, 2]).all())
    check("ordered DGT: beta_R >= beta_G >= beta_B >= b_min under random large weights (20 draws)", ok_ord,
          f"(strict in all draws: {strict_ok})")
    hu = DGTHead(C, depth_mode="one_minus")
    with torch.no_grad():
        hu.mlp[-1].weight.normal_(0, 30.0); hu.mlp[-1].bias.normal_(0, 30.0)
    bu = hu(torch.randn(N, C, 8, 8) * 3, z)["beta_D"]
    control("ordering check rejects the unordered head", not bool((bu[:, 0] >= bu[:, 1]).all() and (bu[:, 1] >= bu[:, 2]).all()))
    ho.zero_grad()
    with torch.no_grad():
        ho.mlp[-1].weight.normal_(0, 0.1); ho.mlp[-1].bias.zero_()
    (ho(feat, z)["t_D"].mean()).backward()
    check("ordered DGT gradients reach the head", all(p.grad is not None and p.grad.abs().sum() > 0 for p in ho.parameters()))

    h2 = DGTHead(C, depth_mode="one_minus")
    t = h2(feat, z)["t_D"][0, 0].flatten()                # z~ increases along the ramp = nearer
    check("one_minus: t_D increases toward near (z~ -> 1)", bool((t[1:] >= t[:-1]).all()) and t[-1] > t[0])
    h3 = DGTHead(C, depth_mode="disparity")
    t3 = h3(feat, z)["t_D"][0, 0].flatten()
    control("near-higher-t check rejects literal disparity mode", not (bool((t3[1:] >= t3[:-1]).all()) and t3[-1] > t3[0]))

    o = h2(feat, z, out_size=(2, 2))
    check("DGT out_size pooling", o["t_D"].shape == (N, 3, 2, 2))
    try:
        DGTHead(C, depth_mode=None)  # type: ignore[arg-type]
        check("DGT requires explicit depth_mode", False)
    except ValueError:
        check("DGT requires explicit depth_mode", True)

    h4 = DGTHead(C, depth_mode="one_minus")
    with torch.no_grad():
        h4.mlp[-1].weight.normal_(0, 0.1)
    o = h4(feat, z)
    (o["t_D"].mean() + o["t_B"].mean() + o["B_inf"].mean()).backward()
    g = [p.grad for p in h4.parameters()]
    check("DGT gradients reach all parameters", all(x is not None and torch.isfinite(x).all() and x.abs().sum() > 0 for x in g))


# ------------------------------------------------------------------ CSA
def grad_blocked(J_struct, out):
    J_struct.grad = None
    out.sum().backward(retain_graph=True)
    return J_struct.grad is None


def stats_only(fn, I):
    """fn(I) -> tensor. Pixel permutation keeps mean/std; output must be unchanged."""
    perm = torch.randperm(I.shape[2] * I.shape[3])
    Ip = I.flatten(2)[:, :, perm].view_as(I)
    return torch.allclose(fn(I), fn(Ip), atol=1e-6)


def csa_checks():
    torch.manual_seed(0)
    N = 3
    I = torch.rand(N, 3, 16, 16)
    J = torch.rand(N, 3, 16, 16, requires_grad=True)
    a = ChromaStyleAdapter()
    o = a(J, I)
    check("CSA identity at init", torch.equal(o["J_style"], J.detach()), f"max|d| {(o['J_style'] - J).abs().max():.2e}")
    check("CSA 12 params per image", o["M"].shape == (N, 3, 3) and o["b"].shape == (N, 3))

    with torch.no_grad():
        a.mlp[-1].weight.normal_(0, 0.5); a.mlp[-1].bias.normal_(0, 0.5)
    o = a(J, I)
    check("CSA blocks gradient into J_struct (detached)", grad_blocked(J, o["J_style"]))
    leaky = torch.einsum("nij,njhw->nihw", o["M"].detach(), J)
    control("detach check rejects a non-detached map", not grad_blocked(J, leaky))
    a.zero_grad(); a(J, I)["J_style"].sum().backward()
    check("CSA parameters receive gradient", all(p.grad is not None and p.grad.abs().sum() > 0 for p in a.parameters()))

    check("CSA conditioned only on Pool_mu,sigma(I) (pixel-permutation invariant)",
          stats_only(lambda x: a(J, x)["M"], I))
    control("permutation check rejects a spatially-conditioned variant",
            not stats_only(lambda x: a(J, x)["M"] + x[:, :, :1, :1].mean(1, keepdim=True).flatten(1)[:, :1, None], I))
    sig = list(inspect.signature(ChromaStyleAdapter.forward).parameters)
    check("CSA interface has no dataset-label argument", sig == ["self", "J_struct", "I", "clamp"], str(sig))
    check("pool_stats shape (N,6)", pool_stats(I).shape == (N, 6))


if __name__ == "__main__":
    dgt_checks()
    csa_checks()
    print(f"\n[phase1_scaffold_check] {'ALL PASS' if not FAIL else 'FAILED: ' + ', '.join(FAIL)}")
    sys.exit(1 if FAIL else 0)

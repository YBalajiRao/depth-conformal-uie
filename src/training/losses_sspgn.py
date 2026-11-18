"""
SS-PGN total objective (draft, matches §3.4 of
reports/research_formulation/SS-PGN_gaps_objectives_methodology_2026-11-06.md).

    L = L_rec + lw * L_wav + lp * L_phys + le * (L_NIG + lr * L_R)

L_rec  = |J - Y|_1 + ls * (1 - SSIM(J, Y))
L_wav  = sum_{b in LH,HL,HH} |W_b(J) - W_b(Y)|_1           (single-level Haar)
L_phys = |I - M * Phi(J; tD, tB, B)|_1                      (re-rendering loop)
         + e1 * sum_c [TV(log tD_c) + TV(log tB_c)]
         + e2 * sum_c |log tD_c - kappa_c * log tD_G|_1      (shared range z)
         + e3 * |dM|_F^2
L_NIG  = Student-t NLL of the Normal-Inverse-Gamma evidential head (Amini et al. 2020)
L_R    = |Y - mu| * (2 nu + alpha)

Shapes (per batch): images (N,3,H,W) in [0,1]; tD, tB, B (N,3,H,W);
M (N,3,3) or (N,3,3,H,W); kappa (N,3); NIG params (N,3,H,W).

Run `python src/training/losses_sspgn.py` for the self-tests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
def _gauss_window(size=11, sigma=1.5, ch=3, device=None, dtype=None):
    x = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).expand(ch, 1, size, size).contiguous()
    return w


def ssim(x, y, data_range=1.0, size=11, sigma=1.5, mask=None):
    """Mean SSIM (Wang et al. 2004), Gaussian window, reflect padding (keeps H, W).
    If mask (N,1,H,W) is given, the SSIM map is averaged over mask==1 only."""
    c = x.shape[1]
    w = _gauss_window(size, sigma, c, x.device, x.dtype)
    pad = size // 2
    f = lambda t: F.conv2d(F.pad(t, (pad,) * 4, mode="reflect"), w, groups=c)
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return s.mean() if mask is None else mmean(s, mask)


def mmean(x, mask=None):
    """Mean of x over pixels where mask==1 (mask broadcast over channels)."""
    if mask is None:
        return x.mean()
    m = mask.expand_as(x)
    return (x * m).sum() / m.sum().clamp_min(1.0)


def haar_dwt(x):
    """Single-level orthonormal Haar DWT. x: (N,C,H,W) with even H, W.
    Returns LL, LH, HL, HH each (N,C,H/2,W/2)."""
    if x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError("haar_dwt needs even H, W (use PadCropWrapper upstream)")
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    ll = (a + b + c + d) / 2
    lh = (a - b + c - d) / 2   # horizontal detail
    hl = (a + b - c - d) / 2   # vertical detail
    hh = (a - b - c + d) / 2
    return ll, lh, hl, hh


def tv(x):
    """Anisotropic total variation (mean of absolute finite differences)."""
    return (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean()


def apply_color_matrix(M, x):
    """M: (N,3,3) global or (N,3,3,H,W) per-pixel; x: (N,3,H,W)."""
    if M.dim() == 3:
        return torch.einsum("nij,njhw->nihw", M, x)
    return torch.einsum("nijhw,njhw->nihw", M, x)


def render(J, tD, tB, B, M=None):
    """Forward image-formation model Phi (revised model, no forward-scatter term):
        I_c = J_c * tD_c + B_c * (1 - tB_c),  then optional cross-talk M."""
    I = J * tD + B * (1.0 - tB)
    return I if M is None else apply_color_matrix(M, I)


def nig_nll(y, mu, nu, alpha, beta):
    """Per-element NLL of y under the NIG marginal (Student-t), Amini et al. 2020, Eq. 8."""
    omega = 2.0 * beta * (1.0 + nu)
    return (0.5 * torch.log(math.pi / nu)
            - alpha * torch.log(omega)
            + (alpha + 0.5) * torch.log(nu * (y - mu) ** 2 + omega)
            + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5))


def nig_reg(y, mu, nu, alpha):
    return (y - mu).abs() * (2.0 * nu + alpha)


def nig_params_from_raw(raw, eps=1e-6):
    """Map a (N,4*3,H,W) head output to valid NIG parameters (mu unconstrained here;
    in SS-PGN mu is the restored image J so only 9 channels are needed)."""
    nu_r, a_r, b_r = raw.chunk(3, dim=1)
    nu = F.softplus(nu_r) + eps
    alpha = F.softplus(a_r) + 1.0 + eps
    beta = F.softplus(b_r) + eps
    return nu, alpha, beta


def nig_variance(nu, alpha, beta):
    """Aleatoric E[sigma^2] = beta/(alpha-1); epistemic Var[mu] = beta/(nu(alpha-1))."""
    return beta / (alpha - 1.0), beta / (nu * (alpha - 1.0))


# --------------------------------------------------------------------------
# Total objective
# --------------------------------------------------------------------------
@dataclass
class SSPGNLossWeights:
    ls: float = 0.2      # SSIM inside L_rec
    lw: float = 0.1      # wavelet high-band
    lp: float = 0.1      # physics
    e1: float = 0.01     # TV on log-transmission
    e2: float = 0.01     # shared-range coupling
    e3: float = 0.01     # cross-talk prior
    le: float = 0.01     # evidential NLL (start small; NLL scale differs from L1)
    lr: float = 0.01     # evidential regulariser
    t_min: float = 1e-3  # clamp for log(t)
    lb: float = 0.1      # DGT soft barrier on log(beta): above b_max and below b_floor (only with beta_barrier)


class SSPGNLoss(nn.Module):
    def __init__(self, w: SSPGNLossWeights | None = None):
        super().__init__()
        self.w = w or SSPGNLossWeights()

    def physics(self, I, J, tD, tB, B, M=None, kappa=None, mask=None):
        w = self.w
        rerender = mmean((I - render(J, tD, tB, B, M)).abs(), mask)
        ltD, ltB = torch.log(tD.clamp_min(w.t_min)), torch.log(tB.clamp_min(w.t_min))
        smooth = tv(ltD) + tv(ltB)
        if kappa is not None:   # kappa: (N,3) per-image beta_c / beta_G, predicted
            couple = mmean((ltD - kappa[:, :, None, None] * ltD[:, 1:2]).abs(), mask)
        else:
            couple = ltD.new_zeros(())
        if M is not None:
            eye = torch.eye(3, device=M.device, dtype=M.dtype)
            eye = eye.view(1, 3, 3) if M.dim() == 3 else eye.view(1, 3, 3, 1, 1)
            cross = ((M - eye) ** 2).sum(dim=(1, 2)).mean()
        else:
            cross = rerender.new_zeros(())
        total = rerender + w.e1 * smooth + w.e2 * couple + w.e3 * cross
        return total, {"phys_rerender": rerender, "phys_tv": smooth,
                       "phys_couple": couple, "phys_cross": cross}

    def recon(self, J: torch.Tensor, Y: torch.Tensor, mask: torch.Tensor | None = None):
        """Reconstruction terms only: L1 + ls*(1 - SSIM) + lw*wavelet. Returns (loss, logs)."""
        w = self.w
        l1 = mmean((J - Y).abs(), mask)
        l_ssim = 1.0 - ssim(J.clamp(0, 1), Y, mask=mask)
        _, lh, hl, hh = haar_dwt(J)
        _, ylh, yhl, yhh = haar_dwt(Y)
        m2 = None if mask is None else -F.max_pool2d(-mask, 2)
        wav = mmean((lh - ylh).abs(), m2) + mmean((hl - yhl).abs(), m2) + mmean((hh - yhh).abs(), m2)
        tot = l1 + w.ls * l_ssim + w.lw * wav
        return tot, {"l1": l1, "ssim_loss": l_ssim, "wav": wav, "total": tot}

    def forward(self, out: dict, I: torch.Tensor, Y: torch.Tensor, mask: torch.Tensor | None = None,
                J_rec: torch.Tensor | None = None, J_nig: torch.Tensor | None = None):
        """out keys: 'J', 'tD', 'tB', 'B', optional 'M', 'kappa', 'nu', 'alpha', 'beta'.
        mask: optional (N,1,H,W) with 1 = real pixel, 0 = padding (excluded from all
        per-pixel terms; TV smoothness is left unmasked).
        J_rec: optional tensor used instead of out['J'] in the reconstruction, wavelet and NIG terms (e.g. a
        colour-aligned J); the physics term always uses out['J']. None (default) = original objective.
        J_nig: optional NIG mean (detached) instead of the reconstruction J, e.g. the deployed J_final."""
        w = self.w
        J_phys_in = out["J"]
        J = J_phys_in if J_rec is None else J_rec
        l1 = mmean((J - Y).abs(), mask)
        l_ssim = 1.0 - ssim(J.clamp(0, 1), Y, mask=mask)
        rec = l1 + w.ls * l_ssim

        _, lh, hl, hh = haar_dwt(J)
        _, ylh, yhl, yhh = haar_dwt(Y)
        m2 = None if mask is None else -F.max_pool2d(-mask, 2)   # a 2x2 cell is valid only if all 4 pixels are
        wav = mmean((lh - ylh).abs(), m2) + mmean((hl - yhl).abs(), m2) + mmean((hh - yhh).abs(), m2)

        phys, logs = self.physics(I, J_phys_in, out["tD"], out["tB"], out["B"], out.get("M"), out.get("kappa"), mask)

        if "nu" in out:
            # J.detach() removes only the DIRECT gradient of the evidential terms w.r.t. mu=J.
            # nu/alpha/beta come from tail_nig on the shared decoder features, so these terms
            # still back-propagate into the shared encoder/decoder (CPU gradient trace, audit
            # 2026-11-07). The head is therefore not isolated from restoration training.
            mu = (J if J_nig is None else J_nig).detach()
            nll = mmean(nig_nll(Y, mu, out["nu"], out["alpha"], out["beta"]), mask)
            reg = mmean(nig_reg(Y, mu, out["nu"], out["alpha"]), mask)
        else:
            nll = reg = l1.new_zeros(())

        total = rec + w.lw * wav + w.lp * phys + w.le * (nll + w.lr * reg)
        if "beta_barrier" in out:                         # DGT only; absent in v1/v2 (objective unchanged)
            total = total + w.lb * out["beta_barrier"]
            logs["beta_barrier"] = out["beta_barrier"]
            if "film_mag" in out:
                logs["film_mag"] = out["film_mag"]
            for k in ("beta_barrier_up", "beta_barrier_lo"):
                if k in out:
                    logs[k] = out[k]
            for nm in ("beta_D", "beta_B"):
                for ci, cn in enumerate("RGB"):
                    logs[f"{nm}_{cn}"] = out[nm][:, ci].mean()
                b_ = out[nm].detach()                     # per-sample ordering margins (min over batch)
                logs[f"{nm}_gapRG_min"] = (b_[:, 0] - b_[:, 1]).min()
                logs[f"{nm}_gapGB_min"] = (b_[:, 1] - b_[:, 2]).min()
        logs.update({"total": total, "l1": l1, "ssim_loss": l_ssim, "wav": wav,
                     "phys": phys, "nig_nll": nll, "nig_reg": reg,
                     "mean_tD": out["tD"].mean(), "mean_tB": out["tB"].mean()})  # collapse monitor
        return total, {k: v.detach() for k, v in logs.items()}


# --------------------------------------------------------------------------
# Redesign Phase 1: structure / style split (CSA)
# --------------------------------------------------------------------------
@torch.no_grad()
def affine_align_coeffs(J, Y, mask=None, eps=1e-6, a_min=0.1, a_max=10.0):
    """Per-image, per-channel least-squares (a, b) with a*J + b ~ Y over valid pixels. No gradient.
    a is clamped to [a_min, a_max] so the aligned target never removes the signal (a -> 0) or inverts it."""
    m = torch.ones_like(J[:, :1]) if mask is None else mask
    m = m.expand_as(J)
    cnt = m.sum((2, 3)).clamp_min(1.0)
    mj, my = (J * m).sum((2, 3)) / cnt, (Y * m).sum((2, 3)) / cnt
    dj, dy = J - mj[..., None, None], Y - my[..., None, None]
    a = ((dj * dy * m).sum((2, 3)) / cnt) / ((dj * dj * m).sum((2, 3)) / cnt + eps)
    a = a.clamp(a_min, a_max)
    return a, my - a * mj


def affine_align(J, Y, mask=None, a_min=0.1, a_max=10.0):
    a, b = affine_align_coeffs(J.detach().float(), Y.float(), mask, a_min=a_min, a_max=a_max)
    return a[..., None, None] * J + b[..., None, None]


class StructStyleLoss(nn.Module):
    """L = L_struct(J_struct) + lf * L_total(J_final), for SS-PGN with the ChromaStyle-Adapter.

    L_struct : the SS-PGN objective (reconstruction + wavelet + physics) on J_struct = x + R; its NIG terms
               use the deployed J_final as the mean (PI decision 2026-11-09), so sigma models J_final's error.
               struct_mode='full'    : reconstruction terms against the reference as-is.
               struct_mode='aligned' : reconstruction terms on the per-image affine colour-aligned
                                       J_struct (coefficients detached); the physics term still uses the
                                       raw J_struct. Removes the reference's global colour from J_struct's
                                       supervision.
    L_total  : reconstruction terms (L1 + ls(1-SSIM) + lw*wav) on J_final = CSA(stopgrad(J_struct), x).
    J_struct is detached inside the CSA, so L_total trains only the CSA, and L_struct never reaches the CSA.

    Gauge anchor (aligned mode only; PI decision 2026-11-09 after the 2,000-step pilot showed J_struct drifting
    to channel means 0.74/1.32/1.65 with the bounded CSA saturated):
        L_struct += la * L1(J_struct, Y) + lrange * (mean relu(-J_struct) + mean relu(J_struct - 1))
    and the alignment gain is clamped to [a_min, a_max] = [0.70, 1.40].
    """

    def __init__(self, w: SSPGNLossWeights | None = None, struct_mode: str = "aligned", lf: float = 1.0,
                 la: float = 0.1, lrange: float = 0.1, a_min: float = 0.70, a_max: float = 1.40):
        super().__init__()
        if struct_mode not in ("full", "aligned"):
            raise ValueError(f"struct_mode must be 'full' or 'aligned', got {struct_mode!r}")
        self.base = SSPGNLoss(w)
        self.w, self.struct_mode, self.lf = self.base.w, struct_mode, float(lf)
        self.la, self.lrange, self.a_min, self.a_max = float(la), float(lrange), float(a_min), float(a_max)

    def forward(self, out, I, Y, mask=None):
        Js = out["J_struct"]
        o = dict(out)
        o["J"] = Js
        Jr = affine_align(Js, Y, mask, self.a_min, self.a_max) if self.struct_mode == "aligned" else None
        L_s, logs = self.base(o, I, Y, mask, J_rec=Jr, J_nig=out["J"])
        L_f, fl = self.base.recon(out["J"], Y, mask)
        if self.struct_mode == "aligned":
            anchor = mmean((Js - Y).abs(), mask)
            rng = mmean(F.relu(-Js), mask) + mmean(F.relu(Js - 1.0), mask)
            L_s = L_s + self.la * anchor + self.lrange * rng
            logs.update(struct_anchor_l1=anchor.detach(), struct_range=rng.detach())
        total = L_s + self.lf * L_f
        with torch.no_grad():
            m_ = torch.ones_like(Js[:, :1]) if mask is None else mask
            cnt = m_.expand_as(Js).sum((0, 2, 3)).clamp_min(1.0)
            jm = (Js * m_).sum((0, 2, 3)) / cnt
            dM = (out["csa_M"] - torch.eye(3, device=Js.device, dtype=Js.dtype)).abs()
            logs.update({f"Jstruct_mean_{c}": jm[i] for i, c in enumerate("RGB")})
            logs.update(Jstruct_out_of_range=((Js < 0) | (Js > 1)).float().mean(),
                        csa_M_maxdev=dM.max(), csa_b_max=out["csa_b"].abs().max(),
                        csa_sat_frac=torch.cat([(dM > 0.95 * 0.5).flatten(1),
                                                (out["csa_b"].abs() > 0.95 * 0.2)], 1).float().mean())
        eye = torch.eye(3, device=out["csa_M"].device, dtype=out["csa_M"].dtype)
        logs.update({"struct_total": L_s.detach(), "total": total.detach(),
                     **{f"final_{k}": v.detach() for k, v in fl.items()},
                     "csa_dev": (((out["csa_M"] - eye) ** 2).sum((1, 2)).mean()
                                 + (out["csa_b"] ** 2).sum(1).mean()).detach(),
                     "csa_MI": (out["csa_M"] - eye).abs().mean().detach()})
        return total, logs


# --------------------------------------------------------------------------
# Self-tests (each can fail for the reason it names)
# --------------------------------------------------------------------------
def _selftest(device="cpu"):
    torch.manual_seed(0)
    N, H, W = 2, 64, 96
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    # T1 Haar is orthonormal and invertible (energy preserved)
    x = torch.rand(N, 3, H, W, device=device)
    bands = haar_dwt(x)
    e_in, e_out = (x ** 2).sum(), sum((b ** 2).sum() for b in bands)
    check("T1 Haar energy preserved", torch.allclose(e_in, e_out, rtol=1e-5),
          f"({e_in.item():.4f} vs {e_out.item():.4f})")

    # T2 SSIM(x,x)=1 and SSIM(x,noise)<1
    check("T2 SSIM(x,x)=1", abs(ssim(x, x).item() - 1) < 1e-5)
    check("T2b SSIM drops with noise", ssim(x, (x + 0.2 * torch.randn_like(x)).clamp(0, 1)).item() < 0.9)

    # T3 Re-rendering residual is zero when I is generated by the same physics
    J = torch.rand(N, 3, H, W, device=device)
    tD = torch.rand(N, 3, H, W, device=device) * 0.8 + 0.1
    tB = torch.rand(N, 3, H, W, device=device) * 0.8 + 0.1
    Bm = torch.rand(N, 3, 1, 1, device=device).expand(N, 3, H, W)
    M = torch.eye(3, device=device).expand(N, 3, 3) + 0.05 * torch.randn(N, 3, 3, device=device)
    I = render(J, tD, tB, Bm, M)
    loss = SSPGNLoss()
    _, lg = loss.physics(I, J, tD, tB, Bm, M)
    check("T3 re-render residual = 0 for consistent physics", lg["phys_rerender"].item() < 1e-6,
          f"({lg['phys_rerender'].item():.2e})")
    _, lg2 = loss.physics(I, J.roll(1, dims=-1), tD, tB, Bm, M)
    check("T3b re-render residual > 0 when J is shifted 1 px", lg2["phys_rerender"].item() > 1e-3,
          f"({lg2['phys_rerender'].item():.2e})")

    # T4 Coupling term is zero when log tD_c = kappa_c * log tD_G exactly
    z = torch.rand(N, 1, H, W, device=device) * 3 + 0.5
    beta_c = torch.tensor([0.6, 0.2, 0.15], device=device)
    tD_phys = torch.exp(-beta_c.view(1, 3, 1, 1) * z)
    kappa = (beta_c / beta_c[1]).expand(N, 3)
    _, lg3 = loss.physics(render(J, tD_phys, tD_phys, Bm), J, tD_phys, tD_phys, Bm, None, kappa)
    check("T4 shared-range coupling = 0 for single-z physics", lg3["phys_couple"].item() < 1e-5,
          f"({lg3['phys_couple'].item():.2e})")

    # T5 NIG NLL equals the Student-t log-density it is derived from:
    #    y ~ St(df=2a, loc=mu, scale^2 = b(1+nu)/(nu*a))
    y = torch.randn(1000, device=device, dtype=torch.float64)
    mu = torch.randn(1000, device=device, dtype=torch.float64) * 0.3
    nu = torch.rand(1000, device=device, dtype=torch.float64) * 3 + 0.1
    al = torch.rand(1000, device=device, dtype=torch.float64) * 3 + 1.1
    be = torch.rand(1000, device=device, dtype=torch.float64) * 2 + 0.1
    st = torch.distributions.StudentT(2 * al, mu, torch.sqrt(be * (1 + nu) / (nu * al)))
    diff = (nig_nll(y, mu, nu, al, be) + st.log_prob(y)).abs().max().item()
    check("T5 NIG NLL == -log StudentT", diff < 1e-9, f"(max|diff|={diff:.1e})")

    # T6 Full loss: finite, gradients reach every predicted quantity, no grad into J from NIG
    J_p = torch.rand(N, 3, H, W, device=device, requires_grad=True)
    raw = {k: torch.rand(N, 3, H, W, device=device, requires_grad=True) for k in ["tD", "tB", "B"]}
    nraw = torch.randn(N, 9, H, W, device=device, requires_grad=True)
    nu_, a_, b_ = nig_params_from_raw(nraw)
    Mp = (torch.eye(3, device=device).expand(N, 3, 3).clone()).requires_grad_(True)
    kp = torch.ones(N, 3, device=device, requires_grad=True)
    out = {"J": J_p, **raw, "M": Mp, "kappa": kp, "nu": nu_, "alpha": a_, "beta": b_}
    Y = torch.rand(N, 3, H, W, device=device)
    total, logs = loss(out, I, Y)
    total.backward()
    grads_ok = all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0
                   for t in [J_p, *raw.values(), nraw, Mp, kp])
    check("T6 total finite and all inputs receive finite non-zero grads",
          torch.isfinite(total) and grads_ok, f"(total={total.item():.4f})")

    # T6b grad wrt J must be identical with the evidential weight on (le=1) and off (le=0)
    gJ = []
    for le in (0.0, 1.0):
        J2 = J_p.detach().clone().requires_grad_(True)
        o2 = {"J": J2, **{k: v.detach() for k, v in raw.items()}, "M": Mp.detach(),
              "kappa": kp.detach(), "nu": nu_.detach(), "alpha": a_.detach(), "beta": b_.detach()}
        SSPGNLoss(SSPGNLossWeights(le=le))(o2, I, Y)[0].backward()
        gJ.append(J2.grad)
    # Bit-exact on CPU; on CUDA, conv backward is non-deterministic at ~1e-11 even for
    # identical runs, so a 1e-9 tolerance is used (a real leak is >= ~1e-5 here).
    d = (gJ[0] - gJ[1]).abs().max().item()
    check("T6b evidential term does not change grad wrt J", d < 1e-9, f"(max|diff|={d:.1e})")

    # T7 mask: all-ones mask == no mask; content in masked-out region must not change the loss
    det = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in out.items()}
    l_none = loss(det, I, Y)[0]
    l_ones = loss(det, I, Y, torch.ones(N, 1, H, W, device=device))[0]
    check("T7 all-ones mask == unmasked", torch.allclose(l_none, l_ones, rtol=1e-6, atol=1e-7),
          f"({l_none.item():.6f} vs {l_ones.item():.6f})")
    mask = torch.ones(N, 1, H, W, device=device); mask[..., H // 2:, :] = 0   # bottom half = padding
    Y2, I2 = Y.clone(), I.clone(); Y2[..., H // 2:, :] = torch.rand_like(Y2[..., H // 2:, :]); I2[..., H // 2:, :] = 0.5
    det2 = dict(det); det2["J"] = det["J"].clone(); det2["J"][..., H // 2:, :] = 0.0
    det2.update({k: det[k] for k in ["nu", "alpha", "beta"]})
    la, lb = loss(det, I, Y, mask)[1], loss(det2, I2, Y2, mask)[1]
    keys = ["l1", "wav", "phys_rerender", "nig_nll", "nig_reg"]
    dmax = max(abs(la[k].item() - lb[k].item()) for k in keys)
    check("T7b masked-out pixels do not affect L1/wav/rerender/NIG", dmax < 1e-6, f"(max|diff|={dmax:.1e})")
    return ok


if __name__ == "__main__":
    import sys
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}")
    sys.exit(0 if _selftest(dev) else 1)

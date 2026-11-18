"""
Depth-Grounded Transmission (DGT) head. Redesign Phase 1 SCAFFOLD: not wired into SS-PGN, not trained.

Nine per-image parameters predicted from pooled features:
    beta^D_c, beta^B_c  > b_min        (c = R, G, B)          direct-signal and backscatter attenuation
    B^inf_c             in (0, 1)                             veiling light
Transmission from the precomputed depth map (src/data/precompute_depth.py):
    t^D_c(x) = exp(-beta^D_c * z(x)),   t^B_c(x) = exp(-beta^B_c * z(x))
so that the image model I = J * t^D + B^inf * (1 - t^B) is available downstream.

z(x) and the stored map z~
    The stored map z~ in [0, 1] is per-image normalised *inverse* depth (1 = nearest). Using z = z~
    directly (depth_mode="disparity", the literal Phase-1 formula) makes near pixels the most attenuated,
    the reverse of the physical ordering. depth_mode="one_minus" uses z = 1 - z~, a monotone depth proxy
    (0 = nearest, 1 = farthest). Because z~ is affine-invariant (unknown scale and shift), neither choice
    is metric depth: beta absorbs the unknown per-image scale, so beta is a *per-normalised-depth*
    coefficient. [b_min, b_max] = [0.05, 2.50] is therefore a normalised scene-attenuation range in those
    units; it coincides numerically with a Jerlov-type range in m^-1 only if z were in metres, which it is not.
    The mode is a required argument (no default) so the choice is explicit in every config.
    Wording: physics-parameterized; no claim that beta identifies physical attenuation.

Parameterisation (PI decision 2026-11-09, replaces the saturating sigmoid):
    beta = b_min + softplus(u + u_init),  u_init = log(expm1(beta_init - b_min)),  beta_init = 0.50
  * lower bound b_min holds by construction; d beta / d u = sigmoid(u + u_init) > 0 everywhere and -> 1
    above the init (no saturation at the top). Near b_min it is ~ (beta - b_min), as for any smooth map
    onto (b_min, inf): a beta driven to b_min still receives a vanishing gradient.
  * b_max is soft: barrier() = sum over {beta_D, beta_B} of mean(relu(log beta - log b_max)^2), zero for
    beta <= b_max, added to the loss by the caller (weight SSPGNLossWeights.lb).
  * Floor margin (PI decision 2026-11-09, after the 200-step pilot): barrier() also adds
    mean(relu(log b_floor - log beta)^2) with b_floor = 0.08, zero for beta >= b_floor. Its gradient w.r.t.
    beta is non-zero below b_floor and pushes beta up; w.r.t. u it is still scaled by d beta/d u ~ beta - b_min
    (~0.03 at beta = 0.08), so it keeps beta away from the softplus tail rather than removing that tail.
  * t in (0, 1] because z in [0, 1] and beta > 0.

Ordered mode (ordered=True; PI decision 2026-11-10 after the Phase-1 100k run): a sequential softplus chain
    beta_B = b_min + softplus(u_B + c_B),  beta_G = beta_B + softplus(u_G + c_G),  beta_R = beta_G + softplus(u_R + c_R)
applied to both beta^D and beta^B, so beta_R >= beta_G >= beta_B holds by construction (strict in exact
arithmetic; in float32 an increment below half an ulp of the running sum rounds to 0, i.e. ties). Init:
beta = (R, G, B) = (0.70, 0.45, 0.30). This is the clear-ocean (Jerlov oceanic-type) ordering imposed as a
prior; in coastal/turbid water blue can attenuate more than green, and with relative depth beta stays a
per-normalised-depth coefficient (no identification claim).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DEPTH_MODES = ("disparity", "one_minus")


def depth_proxy(z_tilde: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "disparity":
        return z_tilde
    if mode == "one_minus":
        return 1.0 - z_tilde
    raise ValueError(f"depth_mode must be one of {DEPTH_MODES}, got {mode!r}")


class DGTHead(nn.Module):
    n_params = 9

    def __init__(self, in_ch: int, depth_mode: str, hidden: int = 128, b_min: float = 0.05, b_max: float = 2.50,
                 beta_init: float = 0.50, b_floor: float = 0.08, ordered: bool = False,
                 ordered_init: tuple = (0.70, 0.45, 0.30)):
        super().__init__()
        if depth_mode not in DEPTH_MODES:
            raise ValueError(f"depth_mode must be one of {DEPTH_MODES}, got {depth_mode!r}")
        if not 0 < b_min < b_max:
            raise ValueError("need 0 < b_min < b_max")
        if not b_min < b_floor < beta_init < b_max:
            raise ValueError("need b_min < b_floor < beta_init < b_max")
        self.depth_mode, self.b_min, self.b_max, self.b_floor = depth_mode, float(b_min), float(b_max), float(b_floor)
        self.u_init = math.log(math.expm1(beta_init - b_min))
        self.ordered = bool(ordered)
        if self.ordered:
            bR, bG, bB = ordered_init
            if not b_min < bB < bG < bR < b_max:
                raise ValueError("ordered_init must satisfy b_min < beta_B < beta_G < beta_R < b_max")
            inv = lambda v: math.log(math.expm1(v))
            self.c_ord = (inv(bR - bG), inv(bG - bB), inv(bB - b_min))      # increments for R, G and base B
        self.mlp = nn.Sequential(nn.Linear(2 * in_ch, hidden), nn.GELU(), nn.Linear(hidden, self.n_params))
        nn.init.zeros_(self.mlp[-1].weight)   # start at beta = beta_init (0.50) and B^inf = 0.5
        nn.init.zeros_(self.mlp[-1].bias)

    def params(self, feat: torch.Tensor, extra: torch.Tensor | None = None):
        """feat: (N, C, h, w) -> beta_D (N,3), beta_B (N,3), B_inf (N,3). extra: optional (N, 9) added to the
        pre-activations (e.g. a zero-initialised projection of a global scene descriptor); None = unchanged."""
        s = torch.cat([feat.mean((2, 3)), feat.amax((2, 3))], 1)
        p = self.mlp(s)
        if extra is not None:
            p = p + extra.to(p.dtype)
        if self.ordered:
            return self._chain(p[:, 0:3]), self._chain(p[:, 3:6]), torch.sigmoid(p[:, 6:9])
        beta = self.b_min + F.softplus(p[:, :6] + self.u_init)
        return beta[:, :3], beta[:, 3:6], torch.sigmoid(p[:, 6:9])

    def _chain(self, u: torch.Tensor) -> torch.Tensor:
        """u: (N,3) columns (R, G, B) -> beta (N,3) with beta_R >= beta_G >= beta_B > b_min (sequential softplus)."""
        cR, cG, cB = self.c_ord
        bB = self.b_min + F.softplus(u[:, 2] + cB)
        bG = bB + F.softplus(u[:, 1] + cG)
        bR = bG + F.softplus(u[:, 0] + cR)
        return torch.stack([bR, bG, bB], 1)

    def barrier_parts(self, *betas: torch.Tensor):
        """(upper, lower): sum_b mean(relu(log b - log b_max)^2), sum_b mean(relu(log b_floor - log b)^2).
        Each is exactly 0 when every beta lies in [b_floor, b_max]."""
        lm, lf = math.log(self.b_max), math.log(self.b_floor)
        up = sum(F.relu(torch.log(b) - lm).pow(2).mean() for b in betas)
        lo = sum(F.relu(lf - torch.log(b)).pow(2).mean() for b in betas)
        return up, lo

    def barrier(self, *betas: torch.Tensor) -> torch.Tensor:
        """Soft two-sided barrier = upper + lower (see barrier_parts)."""
        up, lo = self.barrier_parts(*betas)
        return up + lo

    def fields(self, beta_D: torch.Tensor, beta_B: torch.Tensor, z_tilde: torch.Tensor, out_size=None):
        """t_D, t_B (N,3,*,*) and z (N,1,*,*) from given coefficients; z~ area-pooled to out_size if given."""
        if z_tilde.dim() != 4 or z_tilde.shape[1] != 1:
            raise ValueError(f"z_tilde must be (N,1,H,W), got {tuple(z_tilde.shape)}")
        z = depth_proxy(z_tilde.to(beta_D.dtype), self.depth_mode)
        if out_size is not None:
            z = F.adaptive_avg_pool2d(z, out_size)
        return torch.exp(-beta_D[:, :, None, None] * z), torch.exp(-beta_B[:, :, None, None] * z), z

    def forward(self, feat: torch.Tensor, z_tilde: torch.Tensor, out_size=None):
        """
        feat    : (N, C, h, w) encoder/bottleneck features
        z_tilde : (N, 1, H, W) precomputed normalised inverse depth in [0, 1], aligned with the input crop
        out_size: optional (h', w') to evaluate t at a lower resolution (area-pooled z), e.g. H/16 fields
        returns dict with t_D, t_B (N, 3, *, *), beta_D, beta_B, B_inf (N, 3), z (N, 1, *, *)
        """
        beta_D, beta_B, B_inf = self.params(feat)
        t_D, t_B, z = self.fields(beta_D, beta_B, z_tilde, out_size)
        return {"t_D": t_D, "t_B": t_B, "beta_D": beta_D, "beta_B": beta_B, "B_inf": B_inf, "z": z}

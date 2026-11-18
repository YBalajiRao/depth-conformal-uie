"""
Post-hoc per-pixel uncertainty head on a FROZEN SS-PGN restorer (diagnostic experiment after the
pre-registered step-20k reliability gate failed; see §11 of the methodology doc).

What the head sees (all available at inference; nothing reference-derived):
  F1   : output of SS-PGN decoder stage 1 (`model.dec[0]`, 64 ch, full resolution). This is exactly
         the tensor fed to SS-PGN's own output convolutions (tail_R / tail_nig).
  F2   : output of decoder stage 2 (`model.dec[1]`, 128 ch, H/2), bilinearly upsampled to H x W
         (wider context).
  I    : the degraded input (3 ch).
  J    : the frozen restorer's output mu (3 ch).
  |J-I|: magnitude of the restorer's correction (3 ch).
  Features are captured with forward hooks that only READ module outputs (no return value, so the
  forward pass is unchanged; this is checked bit-exactly in tests/test_post_hoc.py). They are detached,
  so no gradient can reach the restorer.

Head (~0.13 M params): 1x1 reduce (F2 128->32) | concat -> 3x3 conv 105->48 -> 2 dilated residual
blocks (d=2, 4; receptive field +13 px) -> 1x1 -> 3 raw channels.
  sigma = softplus(raw) + SIGMA_FLOOR,  SIGMA_FLOOR = 1e-3 (images in [0,1]).

NIG adapter (so the UNCHANGED evaluation/calibration code can score this head)
  Project equation (src/validation/calibrate_uncertainty.py: predictive_std, kind="total"), used by
  calibration, evaluate_metrics.py and the step-20k monitor:
      Var_total = beta (1 + nu) / (nu (alpha - 1))
  This is the variance of the NIG marginal Student-t: df = 2 alpha, scale^2 = beta (1+nu) / (nu alpha),
  Var = scale^2 * df / (df - 2) = beta (1+nu) / (nu (alpha - 1)), valid for alpha > 1.
  Inverse for a desired sigma^2 at any chosen nu > 0, alpha > 1:
      beta = sigma^2 * nu * (alpha - 1) / (1 + nu)
  We use nu = NU_ADAPT = 1e4 and alpha = 2 (any valid pair gives the same total variance).
  Fields WITHOUT evidential meaning for this model: nu, alpha, beta individually; the 'aleatoric' and
  'epistemic' decompositions; Student-t NLLs computed from them. Only the 'total' variance (= sigma^2)
  is meaningful. Note: train.py's in-training validation logs corr(std,|err|) with the EPISTEMIC
  variance beta/(nu(alpha-1)) = sigma^2/(1+nu) for this adapter: a constant rescaling of sigma, so
  rank metrics agree but the values are not sigma.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from ss_pgn import SSPGN, SSPGNConfig

SIGMA_FLOOR = 1e-3
NU_ADAPT = 1e4


class _DilatedRes(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=d, dilation=d)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.c2(self.act(self.c1(self.act(x))))


class UncertaintyHead(nn.Module):
    def __init__(self, c1=64, c2=128, width=48, c2_reduce=32):
        super().__init__()
        self.red2 = nn.Conv2d(c2, c2_reduce, 1)
        self.inp = nn.Conv2d(c1 + c2_reduce + 9, width, 3, padding=1)
        self.body = nn.Sequential(_DilatedRes(width, 2), _DilatedRes(width, 4))
        self.out = nn.Conv2d(width, 3, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, -3.0)            # softplus(-3)+1e-3 ~ 0.05 initial sigma

    def forward(self, f1, f2, x, j):
        h, w = f1.shape[-2:]
        f2u = F.interpolate(self.red2(f2), size=(h, w), mode="bilinear", align_corners=False)
        z = torch.cat([f1, f2u, x, j, (j - x).abs()], 1)
        raw = self.out(self.body(F.gelu(self.inp(z))))
        return F.softplus(raw) + SIGMA_FLOOR


@dataclass
class PostHocConfig:
    restorer_cfg: dict = field(default_factory=dict)       # SSPGNConfig fields of the frozen restorer
    width: int = 48
    c2_reduce: int = 32

    def restorer(self) -> SSPGNConfig:
        return SSPGNConfig(**self.restorer_cfg)

    @property
    def multiple(self):
        return self.restorer().multiple

    @property
    def compose(self):
        return self.restorer().compose


class SSPGNWithPostHoc(nn.Module):
    """Frozen SS-PGN + trainable post-hoc head. forward(x) needs only the degraded image."""

    def __init__(self, cfg: PostHocConfig):
        super().__init__()
        self.cfg = cfg
        self.restorer = SSPGN(cfg.restorer())
        w = cfg.restorer().widths
        self.head = UncertaintyHead(w[0], w[1], cfg.width, cfg.c2_reduce)
        self._feats = {}
        self._hooks = [self.restorer.dec[0].register_forward_hook(self._grab("f1")),
                       self.restorer.dec[1].register_forward_hook(self._grab("f2"))]
        self.freeze_restorer()
        self.n_params = sum(p.numel() for p in self.parameters())
        self.n_head_params = sum(p.numel() for p in self.head.parameters())

    def _grab(self, key):
        def hook(_module, _inp, out):                      # returns None -> output unchanged
            self._feats[key] = out
        return hook

    def freeze_restorer(self):
        for p in self.restorer.parameters():
            p.requires_grad_(False)
        self.restorer.eval()

    def train(self, mode: bool = True):                    # restorer always stays in eval mode
        super().train(mode)
        self.restorer.eval()
        return self

    def forward(self, x):
        self._feats.clear()
        with torch.no_grad():
            r = self.restorer(x)
        f1, f2 = self._feats["f1"].detach().float(), self._feats["f2"].detach().float()
        J = r["J"].detach().float()
        sigma = self.head(f1, f2, x.float(), J)
        with torch.autocast(device_type=x.device.type, enabled=False):
            sigma = sigma.float()
            nu, alpha, beta = nig_from_sigma(sigma)
            nig = torch.cat([J, nu, alpha, beta], 1)
        return {"nig": nig, "J": J, "sigma": sigma, "nu": nu, "alpha": alpha, "beta": beta,
                "tD": r["tD"], "tB": r["tB"], "B": r["B"], "M": r["M"], "kappa": r["kappa"]}


def nig_from_sigma(sigma, nu=NU_ADAPT, alpha=2.0):
    """Return (nu, alpha, beta) tensors such that beta (1+nu) / (nu (alpha-1)) == sigma^2."""
    nu = torch.as_tensor(nu, dtype=sigma.dtype, device=sigma.device).expand_as(sigma)
    alpha = torch.as_tensor(alpha, dtype=sigma.dtype, device=sigma.device).expand_as(sigma)
    if not bool((nu > 0).all() and (alpha > 1).all()):
        raise ValueError("NIG adapter needs nu > 0 and alpha > 1")
    beta = sigma ** 2 * nu * (alpha - 1) / (1 + nu)
    return nu.clone(), alpha.clone(), beta


def gaussian_nll(e, sigma, mask=None):
    """Heteroscedastic Gaussian NLL of the signed residual e = Y - J_hat:
        0.5*log(2*pi) + log(sigma) + e^2 / (2 sigma^2), averaged over valid pixels and channels."""
    nll = 0.9189385332046727 + torch.log(sigma) + 0.5 * (e / sigma) ** 2
    if mask is None:
        return nll.mean()
    m = mask.expand_as(nll)
    return (nll * m).sum() / m.sum().clamp_min(1.0)

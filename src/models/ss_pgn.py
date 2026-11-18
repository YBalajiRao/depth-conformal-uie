"""
SS-PGN: Spatial-Spectral Physical Gating Network (draft implementation of §3 in
reports/research_formulation/SS-PGN_gaps_objectives_methodology_2026-11-06.md).

Topology (default widths 64/128/256/512, i.e. 4 resolution levels, 3 strided downsamplings):

  x (N,3,H,W)  H, W % 16 == 0   (8 for the 3 downsamplings x 2 for the DWT inside SS-PGB)
  HEAD        conv3x3 3->64                                   (N, 64, H,    W)
  ENC1        NAFBlock x e1 @64        -- skip1 --            (N, 64, H,    W)
  DOWN1       conv2x2/s2 64->128                              (N,128, H/2,  W/2)
  ENC2        NAFBlock x e2 @128       -- skip2 --
  DOWN2       conv2x2/s2 128->256                             (N,256, H/4,  W/4)
  ENC3        NAFBlock x e3 @256       -- skip3 --
  DOWN3       conv2x2/s2 256->512                             (N,512, H/8,  W/8)
  BOTTLENECK  SS-PGB x b @512  (Haar DWT -> LL at H/16)       (N,512, H/8,  W/8)
  UP3         conv1x1 512->1024 + PixelShuffle(2) -> 256, + skip3
  DEC3        NAFBlock x d3 @256
  UP2 / DEC2  -> 128, + skip2
  UP1 / DEC1  -> 64,  + skip1
  TAIL        conv3x3 64->3 = residual R  and  conv3x3 64->9 = raw (nu, alpha, beta)

Physics composition (fp32): the physical field estimated in the last SS-PGB at LL
resolution (H/16) is bilinearly upsampled; background light adds K parametric beams
rendered at full resolution:
  J_phys = clamp((I - B (1 - tB)) / max(tD, t_floor), 0, 1)
  mu     = I + R        (J_hat; compose="direct": the DEFAULT and the trained configuration)
  mu     = J_phys + R   (compose="phys_residual": ablation option only)
In direct mode J_phys is computed and returned but feeds neither mu nor any loss term, and the
global-head outputs (B_inf, M, kappa, beams) reach only B / J_phys and the physics loss, not mu
(code audit and T0, 2026-11-07).
Evidential head (Amini et al. 2020): nu = softplus+eps, alpha = softplus + 1.0001, beta = softplus+eps.

forward() returns a dict:
  'nig'   (N,12,H,W)  [mu(3) | nu(3) | alpha(3) | beta(3)]   <- the single NIG tensor
  'J','nu','alpha','beta'   views of 'nig' (keys expected by SSPGNLoss)
  'tD','tB','B'   (N,3,H,W)   physical field at full resolution
  'M'     (N,3,3)  colour cross-talk matrix, I + tanh-bounded delta
  'kappa' (N,3)    beta_c / beta_G ratios (positive)
  'J_phys' (N,3,H,W)

Run `python src/models/ss_pgn.py` for the self-test (shapes, param budget, DWT
invertibility, AMP forward/backward, peak VRAM at 256x256).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Haar DWT / IDWT (orthonormal, exact inverse)
# --------------------------------------------------------------------------
def dwt(x):
    _assert(x.shape[-2] % 2 == 0 and x.shape[-1] % 2 == 0, f"DWT needs even H,W, got {tuple(x.shape)}")
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return (a + b + c + d) / 2, (a - b + c - d) / 2, (a + b - c - d) / 2, (a - b - c + d) / 2


def idwt(ll, lh, hl, hh):
    n, ch, h, w = ll.shape
    out = ll.new_empty(n, ch, 2 * h, 2 * w)
    out[..., 0::2, 0::2] = (ll + lh + hl + hh) / 2
    out[..., 0::2, 1::2] = (ll - lh + hl - hh) / 2
    out[..., 1::2, 0::2] = (ll + lh - hl - hh) / 2
    out[..., 1::2, 1::2] = (ll - lh - hl + hh) / 2
    return out


def _assert(cond, msg):
    if not cond:
        raise ValueError(msg)


def _check(t, n, c, h, w, where):
    _assert(tuple(t.shape) == (n, c, h, w), f"[{where}] expected {(n, c, h, w)}, got {tuple(t.shape)}")


# --------------------------------------------------------------------------
# NAFNet components (Chen et al., ECCV 2022)
# --------------------------------------------------------------------------
class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight, self.bias, self.eps = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c)), eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    """Activation-free block: LN -> 1x1 (2c) -> DW3x3 -> SimpleGate(GLU) -> SCA -> 1x1, then
    LN -> 1x1 (2c) -> SimpleGate -> 1x1 FFN; both with learnable residual scales."""

    def __init__(self, c):
        super().__init__()
        self.norm1, self.norm2 = LayerNorm2d(c), LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, 2 * c, 1)
        self.dw = nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.conv3 = nn.Conv2d(c, c, 1)
        self.conv4 = nn.Conv2d(c, 2 * c, 1)
        self.conv5 = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.sg(self.dw(self.conv1(self.norm1(x))))
        y = self.conv3(y * self.sca(y))
        x = x + self.beta * y
        y = self.conv5(self.sg(self.conv4(self.norm2(x))))
        return x + self.gamma * y


class ChannelSA(nn.Module):
    """Transposed (channel-wise) multi-head self-attention, linear in H*W (Restormer MDTA style)."""

    def __init__(self, c, heads=8):
        super().__init__()
        _assert(c % heads == 0, f"channels {c} not divisible by heads {heads}")
        self.heads = heads
        self.temp = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(c, 3 * c, 1, bias=False)
        self.dw = nn.Conv2d(3 * c, 3 * c, 3, padding=1, groups=3 * c, bias=False)
        self.proj = nn.Conv2d(c, c, 1, bias=False)

    def forward(self, x):
        n, c, h, w = x.shape
        q, k, v = self.dw(self.qkv(x)).chunk(3, dim=1)
        q, k, v = (t.reshape(n, self.heads, c // self.heads, h * w) for t in (q, k, v))
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1) * self.temp).softmax(dim=-1)
        return self.proj((attn @ v).reshape(n, c, h, w))


# --------------------------------------------------------------------------
# SS-PGB
# --------------------------------------------------------------------------
class SpatialSpectralPhysicalGatingBlock(nn.Module):
    """One SS-PGB (§3.2). Input/output (N,C,h,w) with even h, w.

    1. Haar DWT -> LL (N,C,h/2,w/2), high bands H = [LH;HL;HH].
    2. Channel-separated physical estimation from LL: a 1x1 projection to 3 colour groups
       of g channels, then grouped convs (groups=3) so each colour's tD/tB/B_low comes only
       from its own group.
    3. Low band: LL' = G*(gamma*LL + beta) + (1-G)*ChannelSA(LL), with G, gamma, beta driven by Pi.
    4. High band: H' = H + DW3x3(H * (1 + lam*softplus(psi_h(1 - tD)))).
    5. IDWT + residual.
    Returns (features, phys) where phys holds the LL-resolution field.
    """

    def __init__(self, c, g=32, heads=8, lam_init=0.1, condition=True):
        super().__init__()
        self.c, self.g = c, g
        # condition=False (ablation only): the field branch is still computed (for L_phys), but gate,
        # FiLM and high-band gain receive constant zeros instead of the fields. Parameter set unchanged.
        self.condition = condition
        self.norm = LayerNorm2d(c)
        self.to_groups = nn.Conv2d(c, 3 * g, 1)
        self.phys_body = nn.Sequential(nn.Conv2d(3 * g, 3 * g, 3, padding=1, groups=3), nn.GELU(),
                                       nn.Conv2d(3 * g, 3 * g, 3, padding=1, groups=3), nn.GELU())
        self.phys_out = nn.Conv2d(3 * g, 9, 1, groups=3)       # per colour: (tD, tB, B_low)
        self.gate = nn.Conv2d(c + 9, c, 1)
        self.film = nn.Conv2d(9, 2 * c, 1)
        self.sa = ChannelSA(c, heads)
        self.h_gain = nn.Conv2d(3, c, 1)
        self.h_dw = nn.Conv2d(3 * c, 3 * c, 3, padding=1, groups=3 * c)
        self.lam = nn.Parameter(torch.tensor(float(lam_init)))
        self.out_scale = nn.Parameter(torch.zeros(1, c, 1, 1))
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)   # starts as gamma=1, beta=0 below

    def forward(self, x, gphys=None, dgt=None, z_tilde=None, dgt_extra=None):
        """gphys=None: v1 (Pi = [tD, tB, B_low]). gphys given: v2 (physics-parameterized conditioning):
        g = gphys(LL) and Pi = [tD, tB, B_hat] with B_hat = clamp(B_inf * B_low + beams_{h2 x w2}, 0, 1),
        the H/16 counterpart of the full-resolution physics-loss B (same parameters, different grid).
        dgt given (v2 only; depth_transmission): tD, tB in Pi are replaced by the depth-grounded
        exp(-beta * z) at H/16 (z area-pooled from the precomputed map), beta and B_inf from dgt.params(LL);
        the field branch then supplies only B_low."""
        n, c, h, w = x.shape
        _assert(c == self.c, f"SS-PGB expects {self.c} channels, got {c}")
        ll, lh, hl, hh = dwt(self.norm(x))
        h2, w2 = h // 2, w // 2
        _check(ll, n, c, h2, w2, "SS-PGB/LL")

        raw = self.phys_out(self.phys_body(self.to_groups(ll)))              # (N,9,h2,w2) grouped R|G|B
        raw = raw.view(n, 3, 3, h2, w2)                                       # (N, colour, {tD,tB,B}, h, w)
        tD = torch.sigmoid(raw[:, :, 0] + 2.0)                                # bias toward clear water at init
        tB = torch.sigmoid(raw[:, :, 1] + 2.0)
        B_low = torch.sigmoid(raw[:, :, 2])
        g, dgt_p = None, None
        if dgt is not None:
            _assert(gphys is not None and z_tilde is not None, "DGT requires v2 (gphys) and z_tilde")
            beta_D, beta_B, B_inf_d = dgt.params(ll, dgt_extra)
            tD, tB, _ = dgt.fields(beta_D, beta_B, z_tilde, out_size=(h2, w2))
            dgt_p = {"beta_D": beta_D, "beta_B": beta_B, "B_inf": B_inf_d}
        if gphys is None:
            pi = torch.cat([tD, tB, B_low], dim=1)                            # (N,9,h2,w2)  v1
        else:
            g = gphys(ll)                                                     # v2: global head on LL (acyclic)
            B_inf = g["B_inf"] if dgt_p is None else dgt_p["B_inf"]
            B_hat = B_inf[:, :, None, None] * B_low                           # (N,3,h2,w2)
            beams = render_beams(g, h2, w2)
            if beams is not None:
                B_hat = (B_hat + beams).clamp(0, 1)
            pi = torch.cat([tD, tB, B_hat], dim=1)                            # (N,9,h2,w2)  v2
        _check(pi, n, 9, h2, w2, "SS-PGB/Pi")

        pc = pi if self.condition else torch.zeros_like(pi)
        G = torch.sigmoid(self.gate(torch.cat([ll, pc], 1)))
        gam, bet = self.film(pc).chunk(2, 1)
        ll2 = G * ((1 + gam) * ll + bet) + (1 - G) * self.sa(ll)

        hb = torch.cat([lh, hl, hh], 1)                                       # (N,3C,h2,w2)
        hin = (1 - tD) if self.condition else torch.zeros_like(tD)
        gain = 1 + self.lam * F.softplus(self.h_gain(hin))                    # (N,C,h2,w2)
        hb2 = hb + self.h_dw(hb * gain.repeat(1, 3, 1, 1))
        lh2, hl2, hh2 = hb2.chunk(3, 1)

        y = idwt(ll2, lh2, hl2, hh2)
        _check(y, n, c, h, w, "SS-PGB/out")
        phys = {"tD": tD, "tB": tB, "B_low": B_low}
        if g is not None:
            phys["g"] = g
        if dgt_p is not None:
            phys["dgt"] = dgt_p
        return x + self.out_scale * y, phys


# --------------------------------------------------------------------------
# Global physics head (per image): B_inf, beams, M, kappa
# --------------------------------------------------------------------------
class GlobalPhysicsHead(nn.Module):
    def __init__(self, c, n_beams=1, rho=0.2):
        super().__init__()
        self.k, self.rho = n_beams, rho
        hidden = 128
        self.mlp = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, hidden), nn.GELU())
        self.binf = nn.Linear(hidden, 3)
        self.dM = nn.Linear(hidden, 9)
        self.kappa = nn.Linear(hidden, 3)
        self.beam = nn.Linear(hidden, 7 * n_beams) if n_beams > 0 else None   # mu(2), log-sigma(2), amp(3)
        for lin in [self.dM, self.kappa] + ([self.beam] if self.beam is not None else []):
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)

    def forward(self, f):
        z = self.mlp(f)
        n = z.shape[0]
        out = {"B_inf": torch.sigmoid(self.binf(z)),                                     # (N,3)
               "M": torch.eye(3, device=z.device, dtype=z.dtype).expand(n, 3, 3)
                    + self.rho * torch.tanh(self.dM(z)).view(n, 3, 3),                   # (N,3,3)
               "kappa": F.softplus(self.kappa(z)) / math.log(2.0)}                       # (N,3), =1 at init
        if self.beam is not None:
            p = self.beam(z).view(n, self.k, 7)
            out["beam_mu"] = torch.sigmoid(p[..., 0:2])                                  # (N,K,2) in [0,1]
            out["beam_sigma"] = 0.05 + torch.sigmoid(p[..., 2:4])                        # (N,K,2) fraction of size
            out["beam_amp"] = torch.sigmoid(p[..., 4:7] - 4.0)                           # (N,K,3) ~0 at init
        return out


def render_beams(g, h, w):
    """Sum of K axis-aligned Gaussian beams at full resolution -> (N,3,h,w)."""
    if "beam_mu" not in g:
        return None
    dev, dt = g["beam_mu"].device, g["beam_mu"].dtype
    yy = torch.linspace(0, 1, h, device=dev, dtype=dt).view(1, 1, h, 1)
    xx = torch.linspace(0, 1, w, device=dev, dtype=dt).view(1, 1, 1, w)
    mu, sg, amp = g["beam_mu"], g["beam_sigma"], g["beam_amp"]                          # (N,K,2),(N,K,2),(N,K,3)
    gy = torch.exp(-0.5 * ((yy - mu[..., 0, None, None]) / sg[..., 0, None, None]) ** 2)  # (N,K,h,1)
    gx = torch.exp(-0.5 * ((xx - mu[..., 1, None, None]) / sg[..., 1, None, None]) ** 2)  # (N,K,1,w)
    prof = gy * gx                                                                          # (N,K,h,w)
    return torch.einsum("nkc,nkhw->nchw", amp, prof)


# --------------------------------------------------------------------------
# SS-PGN
# --------------------------------------------------------------------------
@dataclass
class SSPGNConfig:
    widths: tuple = (64, 128, 256, 512)
    enc_blocks: tuple = (1, 1, 1)
    dec_blocks: tuple = (1, 1, 1)
    bottleneck_blocks: int = 1
    phys_groups: int = 32
    heads: int = 8
    n_beams: int = 1
    t_floor: float = 0.1
    max_params: int = 5_000_000
    # 'direct'        : mu = x + R. The physical field must independently re-render x from mu,
    #                   so L_phys is a real constraint on (tD, tB, B). DEFAULT.
    # 'phys_residual' : mu = J_phys + R with J_phys = inverse model. Re-rendering then holds for
    #                   ANY field (L_phys only penalises R): kept for the ablation only.
    compose: str = "direct"
    # True (DEFAULT, the trained configuration): SS-PGB gate/FiLM/high-band gain are driven by the fields.
    # False: ablation A2 (fields do not condition the restoration path).
    field_conditioning: bool = True
    # "v1" (DEFAULT, the trained configuration): Pi = [tD, tB, B_low]; gphys reads the bottleneck OUTPUT and
    #      reaches only B / J_phys / L_phys.
    # "v2": physics-parameterized conditioning. gphys reads the SS-PGB's normalized Haar LL band (H/16) and
    #      Pi = [tD, tB, B_hat], B_hat = clamp(B_inf * B_low + beams at H/16). Requires bottleneck_blocks == 1.
    phys_conditioning: str = "v1"
    # Redesign Phase 1 (default off = configurations above unchanged, identical parameter set and outputs):
    # depth_transmission: DGT head (src/models/dgt_head.py); tD/tB = exp(-beta * z) from the precomputed
    #   depth map z~ (forward(x, z)); beta in [dgt_b_min, dgt_b_max], B_inf from the DGT head. Requires v2.
    # csa: blind ChromaStyle-Adapter (src/models/csa.py) on the detached J_struct = x + R; out["J"] is then
    #   J_final = CSA(J_struct, x) and out["J_struct"] is returned for L_struct. NIG mean = J_final.
    depth_transmission: bool = False
    dgt_depth_mode: str = "one_minus"
    dgt_b_min: float = 0.05
    dgt_b_max: float = 2.50
    dgt_b_floor: float = 0.08
    dgt_ordered: bool = False      # Jerlov-type ordering beta_R >= beta_G >= beta_B (sequential softplus chain)
    # dec_film (requires depth_transmission; PI decision 2026-11-10): after each decoder stage i,
    #   f <- f * (1 + gamma_i) + beta_i,  (gamma_i, beta_i) = Conv1x1_i([z~, t_D, t_B] area-pooled to stage i),
    # t = exp(-beta_DGT (1 - z~)) from the DGT coefficients; Conv1x1_i zero-initialised (weights and bias), so a
    # fresh dec_film model computes exactly what the same weights compute without it.
    dec_film: bool = False
    # scene_prior (requires depth_transmission; PI decision 2026-11-13, roadmap phase C): a precomputed global scene
    #   descriptor g (768-d, frozen Depth-Anything-V2-Small encoder: [CLS, mean patch tokens] of the full image) is
    #   mapped by h = GELU(Linear(768 -> scene_rank)) to (i) a channel FiLM on the bottleneck input,
    #   f <- f * (1 + gamma) + beta with (gamma, beta) = Linear(h), and (ii) an additive term on the DGT head's 9
    #   pre-activations. Both output layers are zero-initialised, so a fresh model computes exactly what the same
    #   weights compute without it. No decoder-side injection (the P1c bypass is avoided). forward(x, z, g).
    scene_prior: bool = False
    scene_dim: int = 768
    scene_rank: int = 64
    # P1e (PI decision 2026-11-13, report section P): scene_mode "dgt_only" removes the bottleneck FiLM path - the
    # descriptor reaches ONLY the 9 ordered-DGT pre-activations. scene_dropout (per-sample: the whole descriptor is
    # zeroed, i.e. set to the training mean in the centred PCA space) and scene_noise (Gaussian, in whitened PCA
    # units) act in training mode only. scene_pca_file records the training-only PCA used to build g.
    scene_mode: str = "film_dgt"
    scene_dropout: float = 0.0
    scene_noise: float = 0.0
    scene_pca_file: str = ""
    csa: bool = False

    @property
    def multiple(self):
        return 2 ** (len(self.widths) - 1) * 2      # downsamplings x DWT


class SSPGN(nn.Module):
    def __init__(self, cfg: SSPGNConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or SSPGNConfig()
        W = cfg.widths
        _assert(len(W) == 4 and len(cfg.enc_blocks) == 3 and len(cfg.dec_blocks) == 3,
                "expects 4 widths, 3 encoder and 3 decoder stage counts")
        _assert(cfg.phys_conditioning in ("v1", "v2"), f"unknown phys_conditioning {cfg.phys_conditioning!r}")
        _assert(cfg.phys_conditioning == "v1" or cfg.bottleneck_blocks == 1,
                "phys_conditioning='v2' is defined for a single SS-PGB (bottleneck_blocks == 1)")
        self.head = nn.Conv2d(3, W[0], 3, padding=1)
        self.enc = nn.ModuleList(nn.Sequential(*[NAFBlock(W[i]) for _ in range(cfg.enc_blocks[i])]) for i in range(3))
        self.down = nn.ModuleList(nn.Conv2d(W[i], W[i + 1], 2, stride=2) for i in range(3))
        self.bottleneck = nn.ModuleList(SpatialSpectralPhysicalGatingBlock(W[3], cfg.phys_groups, cfg.heads,
                                                                           condition=cfg.field_conditioning)
                                        for _ in range(cfg.bottleneck_blocks))
        self.gphys = GlobalPhysicsHead(W[3], cfg.n_beams)
        self.up = nn.ModuleList(nn.Sequential(nn.Conv2d(W[i + 1], W[i] * 4, 1, bias=False), nn.PixelShuffle(2))
                                for i in range(3))
        self.dec = nn.ModuleList(nn.Sequential(*[NAFBlock(W[i]) for _ in range(cfg.dec_blocks[i])]) for i in range(3))
        self.tail_R = nn.Conv2d(W[0], 3, 3, padding=1)
        self.tail_nig = nn.Conv2d(W[0], 9, 3, padding=1)
        nn.init.zeros_(self.tail_R.weight); nn.init.zeros_(self.tail_R.bias)
        self.dgt = self.csa = None
        if cfg.depth_transmission:
            _assert(cfg.phys_conditioning == "v2", "depth_transmission requires phys_conditioning='v2'")
            from dgt_head import DGTHead
            self.dgt = DGTHead(W[3], depth_mode=cfg.dgt_depth_mode, b_min=cfg.dgt_b_min, b_max=cfg.dgt_b_max,
                               b_floor=cfg.dgt_b_floor, ordered=cfg.dgt_ordered)
        if cfg.csa:
            from csa import ChromaStyleAdapter
            self.csa = ChromaStyleAdapter()
        self.scene = self.scene_film = self.scene_dgt = None
        if cfg.scene_prior:
            _assert(cfg.depth_transmission, "scene_prior requires depth_transmission (DGT)")
            _assert(cfg.scene_mode in ("film_dgt", "dgt_only"), f"unknown scene_mode {cfg.scene_mode!r}")
            self.scene = nn.Sequential(nn.Linear(cfg.scene_dim, cfg.scene_rank), nn.GELU())
            self.scene_film = nn.Linear(cfg.scene_rank, 2 * W[3]) if cfg.scene_mode == "film_dgt" else None
            self.scene_dgt = nn.Linear(cfg.scene_rank, 9)
            for lin in (self.scene_film, self.scene_dgt):
                if lin is not None:
                    nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
        self.film = None
        if cfg.dec_film:
            _assert(cfg.depth_transmission, "dec_film requires depth_transmission (DGT)")
            self.film = nn.ModuleList(nn.Conv2d(7, 2 * W[i], 1) for i in range(3))
            for conv in self.film:
                nn.init.zeros_(conv.weight); nn.init.zeros_(conv.bias)
        n_par = sum(p.numel() for p in self.parameters())
        _assert(n_par <= cfg.max_params, f"SS-PGN has {n_par:,} params > budget {cfg.max_params:,}")
        self.n_params = n_par

    def forward(self, x, z=None, g=None):
        """x: (N,3,H,W) in [0,1]. z: (N,1,H,W) precomputed normalised inverse depth, required iff
        cfg.depth_transmission (aligned pixel-for-pixel with x; pad/crop/augment identically)."""
        n, c, h, w = x.shape
        m = self.cfg.multiple
        _assert(c == 3, f"expected 3 input channels, got {c}")
        if self.dgt is not None:
            _assert(z is not None and tuple(z.shape) == (n, 1, h, w),
                    f"depth_transmission needs z of shape {(n, 1, h, w)}, got {None if z is None else tuple(z.shape)}")
        else:
            _assert(z is None, "z was given but depth_transmission is off")
        if self.scene is not None:
            _assert(g is not None and tuple(g.shape) == (n, self.cfg.scene_dim),
                    f"scene_prior needs g of shape {(n, self.cfg.scene_dim)}, got {None if g is None else tuple(g.shape)}")
        else:
            _assert(g is None, "g was given but scene_prior is off")
        _assert(h % m == 0 and w % m == 0, f"H, W must be multiples of {m} (use PadCropWrapper); got {h}x{w}")
        W = self.cfg.widths

        f = self.head(x); _check(f, n, W[0], h, w, "head")
        skips = []
        for i in range(3):
            f = self.enc[i](f); _check(f, n, W[i], h >> i, w >> i, f"enc{i + 1}")
            skips.append(f)
            f = self.down[i](f); _check(f, n, W[i + 1], h >> (i + 1), w >> (i + 1), f"down{i + 1}")
        phys = None
        if self.cfg.phys_conditioning == "v2":                                # g from the block's own LL
            dgt_extra = None
            if self.scene is not None:                                        # global scene prior (zero-init)
                gi = g
                if self.training and (self.cfg.scene_dropout > 0 or self.cfg.scene_noise > 0):
                    if self.cfg.scene_noise > 0:
                        gi = gi + self.cfg.scene_noise * torch.randn_like(gi)
                    if self.cfg.scene_dropout > 0:
                        keep = (torch.rand(gi.shape[0], 1, device=gi.device) >= self.cfg.scene_dropout).to(gi.dtype)
                        gi = gi * keep
                hs = self.scene(gi.to(f.dtype))
                if self.scene_film is not None:
                    sg, sb = self.scene_film(hs).chunk(2, 1)
                    f = f * (1 + sg[:, :, None, None]) + sb[:, :, None, None]
                dgt_extra = self.scene_dgt(hs)
            f, phys = self.bottleneck[0](f, gphys=self.gphys, dgt=self.dgt, z_tilde=z, dgt_extra=dgt_extra)
            g = phys["g"]
        else:                                                                 # v1: g from the block output
            for blk in self.bottleneck:
                f, phys = blk(f)
        _check(f, n, W[3], h >> 3, w >> 3, "bottleneck")
        if self.cfg.phys_conditioning != "v2":
            g = self.gphys(f)
        film_mag = None
        for i in reversed(range(3)):
            f = self.up[i](f); _check(f, n, W[i], h >> i, w >> i, f"up{i + 1}")
            f = self.dec[i](f + skips[i]); _check(f, n, W[i], h >> i, w >> i, f"dec{i + 1}")
            if self.film is not None:
                dpf = phys["dgt"]
                hw = (h >> i, w >> i)
                tDi, tBi, _ = self.dgt.fields(dpf["beta_D"], dpf["beta_B"], z, out_size=hw)
                cond = torch.cat([F.adaptive_avg_pool2d(z, hw).to(tDi.dtype), tDi, tBi], 1)
                gam, bet = self.film[i](cond).chunk(2, 1)
                f = f * (1 + gam) + bet
                mag = (gam.abs().mean() + bet.abs().mean()).detach().float()
                film_mag = mag if film_mag is None else film_mag + mag

        R = self.tail_R(f)
        nig_raw = self.tail_nig(f)
        with torch.autocast(device_type=x.device.type, enabled=False):          # physics + NIG in fp32
            xf, R, nig_raw = x.float(), R.float(), nig_raw.float()
            up = lambda t: F.interpolate(t.float(), size=(h, w), mode="bilinear", align_corners=False)
            dp = phys.get("dgt")
            if dp is None:
                tD, tB = up(phys["tD"]), up(phys["tB"])
                B = g["B_inf"].float()[:, :, None, None] * up(phys["B_low"])
                kappa = g["kappa"].float()
            else:                                                             # exact full-res exp(-beta z)
                dp = {k: v.float() for k, v in dp.items()}
                tD, tB, _ = self.dgt.fields(dp["beta_D"], dp["beta_B"], z.float())
                B = dp["B_inf"][:, :, None, None] * up(phys["B_low"])
                kappa = dp["beta_D"] / dp["beta_D"][:, 1:2]                   # exact for a single z: L_couple = 0
            beams = render_beams({k: v.float() for k, v in g.items()}, h, w)
            if beams is not None:
                B = (B + beams).clamp(0, 1)
            J_phys = ((xf - B * (1 - tB)) / tD.clamp_min(self.cfg.t_floor)).clamp(0, 1)
            if self.cfg.compose == "direct":
                mu = xf + R
            elif self.cfg.compose == "phys_residual":
                mu = J_phys + R
            else:
                raise ValueError(f"unknown compose mode {self.cfg.compose!r}")
            nu_r, a_r, b_r = nig_raw.chunk(3, 1)
            nu = F.softplus(nu_r) + 1e-4
            alpha = F.softplus(a_r) + 1.0001
            beta = F.softplus(b_r) + 1e-4
            nig = torch.cat([mu, nu, alpha, beta], 1)
            extra = {}
            if self.csa is not None:                                          # J_struct detached inside CSA
                cs = self.csa(mu, xf)
                extra = {"J_struct": mu, "csa_M": cs["M"], "csa_b": cs["b"]}
                J_out = cs["J_style"]
                nig = torch.cat([J_out, nig[:, 3:]], 1)                       # NIG mean = deployed J_final
            else:
                J_out = nig[:, 0:3]
            if dp is not None:
                extra.update(beta_D=dp["beta_D"], beta_B=dp["beta_B"], B_inf=dp["B_inf"],
                             **dict(zip(("beta_barrier_up", "beta_barrier_lo"),
                                        self.dgt.barrier_parts(dp["beta_D"], dp["beta_B"]))))
                extra["beta_barrier"] = extra["beta_barrier_up"] + extra["beta_barrier_lo"]
        _check(nig, n, 12, h, w, "nig")
        return {"nig": nig, "J": J_out, "nu": nig[:, 3:6], "alpha": nig[:, 6:9], "beta": nig[:, 9:12],
                "tD": tD, "tB": tB, "B": B, "M": g["M"].float(), "kappa": kappa, "J_phys": J_phys, **extra,
                **({"film_mag": film_mag} if film_mag is not None else {})}


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
def _selftest():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    x = torch.randn(2, 8, 16, 24)
    check("DWT/IDWT exact inverse", torch.allclose(idwt(*dwt(x)), x, atol=1e-6))

    net = SSPGN().to(dev)
    print(f"  params: {net.n_params:,}  (budget {net.cfg.max_params:,}); input multiple = {net.cfg.multiple}")
    for name, mod in [("head", net.head), ("encoder", net.enc), ("down", net.down), ("SS-PGB", net.bottleneck),
                      ("global phys", net.gphys), ("up", net.up), ("decoder", net.dec),
                      ("tails", nn.ModuleList([net.tail_R, net.tail_nig]))]:
        print(f"    {name:12s} {sum(p.numel() for p in mod.parameters()):>10,}")
    check("param budget <= 5M", net.n_params <= 5_000_000)

    try:
        net(torch.rand(1, 3, 250, 256, device=dev)); check("rejects H not multiple of 16", False)
    except ValueError:
        check("rejects H not multiple of 16", True)

    out = net(torch.rand(1, 3, 240, 320, device=dev))
    check("non-square 240x320 shapes", tuple(out["nig"].shape) == (1, 12, 240, 320) and
          tuple(out["M"].shape) == (1, 3, 3) and tuple(out["kappa"].shape) == (1, 3))
    check("NIG constraints nu>0, alpha>1, beta>0",
          bool((out["nu"] > 0).all() and (out["alpha"] > 1).all() and (out["beta"] > 0).all()))
    check("all outputs finite", all(torch.isfinite(v).all() for v in out.values() if torch.is_tensor(v)))

    # The physics loop must be non-trivial in 'direct' mode and (by construction) trivial in 'phys_residual'
    import sys as _s, pathlib as _pl
    _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "training"))
    from losses_sspgn import render
    xr = torch.rand(1, 3, 64, 64, device=dev)
    for mode in ("direct", "phys_residual"):
        nm = SSPGN(SSPGNConfig(compose=mode)).to(dev)
        with torch.no_grad():
            nn.init.normal_(nm.tail_R.weight, std=1e-3)        # small non-zero residual
            o = nm(xr)
            rr = (xr - render(o["J"], o["tD"], o["tB"], o["B"])).abs().mean().item()
        print(f"    compose={mode:13s} re-render residual at init = {rr:.2e}")
        if mode == "direct":
            check("direct mode: re-render residual is non-trivial at init", rr > 1e-3)

    if dev == "cuda":
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "training"))
        from losses_sspgn import SSPGNLoss
        crit = SSPGNLoss()
        opt = torch.optim.AdamW(net.parameters(), 1e-4)
        scaler = torch.amp.GradScaler("cuda")
        for bs in (4, 8):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                xb, yb = torch.rand(bs, 3, 256, 256, device=dev), torch.rand(bs, 3, 256, 256, device=dev)
                with torch.autocast("cuda", dtype=torch.float16):
                    o = net(xb)
                loss, _ = crit(o, xb, yb)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
                gn = [p.grad for p in net.parameters() if p.grad is not None]
                peak = torch.cuda.max_memory_allocated() / 2 ** 30
                check(f"AMP train step bs={bs} @256", torch.isfinite(loss) and len(gn) > 0,
                      f"(loss={loss.item():.4f}, peak VRAM={peak:.2f} GiB)")
            except torch.cuda.OutOfMemoryError:
                check(f"AMP train step bs={bs} @256", False, "(OOM)")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)

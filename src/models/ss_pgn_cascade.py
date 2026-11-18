"""
SS-PGN-Cascade: multi-scale variant with an SS-PGB at every resolution level and a
coarse-to-fine cascade of the physical field.

Differences from src/models/ss_pgn.py (the single-bottleneck baseline):
  * Encoder level i (64/128/256): NAFBlock x e_i  ->  SS-PGB (independent field estimate).
  * Bottleneck (384, see budget note): SS-PGB that produces the coarsest field (H/16).
  * Decoder level i (256/128/64): PixelShuffle up + skip -> NAFBlock x d_i -> CascadeSSPGB, which
    REFINES the field passed down from the coarser level (residual update in logit space):
        logits_i = phys_head_i(LL_i) + upsample(logits_{i+1})
    The finest decoder block (64 ch, LL at H/2) gives the final field. This is 8x finer than
    the baseline's H/16 field.
  * Budget: an SS-PGB at 512 costs ~1.4M params, so with 7 SS-PGBs the deepest width is
    reduced to 384 (widths 64/128/256/384). Total is asserted <= 5M at construction.

Risk to test in the ablation: a finer field is more flexible, so L_phys is easier to
satisfy (with per-pixel tD/tB/B, the formation model can fit any I). Smoothness comes
from the TV term on log t and from the cascade's coarse prior. Compare the field
resolution (H/16 vs H/2) on L_phys and on the PSNR / reliability metrics.

Output dict and NIG conventions are identical to SSPGN, so SSPGNLoss, PadCropWrapper and
train.py work unchanged (pass --arch cascade once train.py exposes it).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ss_pgn import (NAFBlock, SpatialSpectralPhysicalGatingBlock, GlobalPhysicsHead, render_beams,
                    dwt, idwt, _assert, _check)


class CascadeSSPGB(SpatialSpectralPhysicalGatingBlock):
    """SS-PGB that accepts the coarser level's field logits as a prior and returns its own logits."""

    def forward(self, x, prior_logits=None):
        n, c, h, w = x.shape
        _assert(c == self.c, f"SS-PGB expects {self.c} channels, got {c}")
        ll, lh, hl, hh = dwt(self.norm(x))
        h2, w2 = h // 2, w // 2
        _check(ll, n, c, h2, w2, "CascadeSSPGB/LL")

        logits = self.phys_out(self.phys_body(self.to_groups(ll)))             # (N,9,h2,w2), grouped R|G|B
        if prior_logits is not None:
            _assert(prior_logits.shape[:2] == (n, 9), f"prior logits shape {tuple(prior_logits.shape)}")
            logits = logits + F.interpolate(prior_logits, size=(h2, w2), mode="bilinear", align_corners=False)
        raw = logits.view(n, 3, 3, h2, w2)
        tD = torch.sigmoid(raw[:, :, 0] + 2.0)
        tB = torch.sigmoid(raw[:, :, 1] + 2.0)
        B_low = torch.sigmoid(raw[:, :, 2])
        pi = torch.cat([tD, tB, B_low], 1)

        G = torch.sigmoid(self.gate(torch.cat([ll, pi], 1)))
        gam, bet = self.film(pi).chunk(2, 1)
        ll2 = G * ((1 + gam) * ll + bet) + (1 - G) * self.sa(ll)
        hb = torch.cat([lh, hl, hh], 1)
        gain = 1 + self.lam * F.softplus(self.h_gain(1 - tD))
        hb2 = hb + self.h_dw(hb * gain.repeat(1, 3, 1, 1))
        y = idwt(ll2, *hb2.chunk(3, 1))
        _check(y, n, c, h, w, "CascadeSSPGB/out")
        return x + self.out_scale * y, {"tD": tD, "tB": tB, "B_low": B_low, "logits": logits}


@dataclass
class SSPGNCascadeConfig:
    widths: tuple = (64, 128, 256, 384)
    enc_blocks: tuple = (1, 1, 1)
    dec_blocks: tuple = (1, 1, 1)
    phys_groups: tuple = (16, 16, 32, 32)      # per level (enc/dec share), last = bottleneck
    heads: tuple = (4, 4, 8, 8)
    n_beams: int = 1
    t_floor: float = 0.1
    max_params: int = 5_000_000
    compose: str = "direct"                    # see ss_pgn.SSPGNConfig

    @property
    def multiple(self):
        return 2 ** (len(self.widths) - 1) * 2


class SSPGNCascade(nn.Module):
    def __init__(self, cfg: SSPGNCascadeConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or SSPGNCascadeConfig()
        W, g, hd = cfg.widths, cfg.phys_groups, cfg.heads
        _assert(len(W) == 4 and len(g) == 4 and len(hd) == 4, "expects 4 widths / groups / heads")
        self.head = nn.Conv2d(3, W[0], 3, padding=1)
        self.enc = nn.ModuleList(nn.Sequential(*[NAFBlock(W[i]) for _ in range(cfg.enc_blocks[i])]) for i in range(3))
        self.enc_pgb = nn.ModuleList(CascadeSSPGB(W[i], g[i], hd[i]) for i in range(3))
        self.down = nn.ModuleList(nn.Conv2d(W[i], W[i + 1], 2, stride=2) for i in range(3))
        self.bottleneck = CascadeSSPGB(W[3], g[3], hd[3])
        self.gphys = GlobalPhysicsHead(W[3], cfg.n_beams)
        self.up = nn.ModuleList(nn.Sequential(nn.Conv2d(W[i + 1], W[i] * 4, 1, bias=False), nn.PixelShuffle(2))
                                for i in range(3))
        self.dec = nn.ModuleList(nn.Sequential(*[NAFBlock(W[i]) for _ in range(cfg.dec_blocks[i])]) for i in range(3))
        self.dec_pgb = nn.ModuleList(CascadeSSPGB(W[i], g[i], hd[i]) for i in range(3))
        self.tail_R = nn.Conv2d(W[0], 3, 3, padding=1)
        self.tail_nig = nn.Conv2d(W[0], 9, 3, padding=1)
        nn.init.zeros_(self.tail_R.weight); nn.init.zeros_(self.tail_R.bias)
        self.n_params = sum(p.numel() for p in self.parameters())
        _assert(self.n_params <= cfg.max_params, f"cascade has {self.n_params:,} params > {cfg.max_params:,}")

    def forward(self, x):
        n, c, h, w = x.shape
        m, W = self.cfg.multiple, self.cfg.widths
        _assert(c == 3, f"expected 3 input channels, got {c}")
        _assert(h % m == 0 and w % m == 0, f"H, W must be multiples of {m} (use PadCropWrapper); got {h}x{w}")

        f = self.head(x); _check(f, n, W[0], h, w, "head")
        skips, enc_fields = [], []
        for i in range(3):
            f = self.enc[i](f)
            f, ph = self.enc_pgb[i](f); enc_fields.append(ph)
            _check(f, n, W[i], h >> i, w >> i, f"enc{i + 1}")
            skips.append(f)
            f = self.down[i](f); _check(f, n, W[i + 1], h >> (i + 1), w >> (i + 1), f"down{i + 1}")
        f, ph = self.bottleneck(f)
        _check(f, n, W[3], h >> 3, w >> 3, "bottleneck")
        g = self.gphys(f)
        logits = ph["logits"]
        for i in reversed(range(3)):
            f = self.up[i](f); _check(f, n, W[i], h >> i, w >> i, f"up{i + 1}")
            f = self.dec[i](f + skips[i])
            f, ph = self.dec_pgb[i](f, prior_logits=logits)
            logits = ph["logits"]
            _check(f, n, W[i], h >> i, w >> i, f"dec{i + 1}")
        phys = ph                                                    # finest field: LL at H/2

        R = self.tail_R(f)
        nig_raw = self.tail_nig(f)
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf, R, nig_raw = x.float(), R.float(), nig_raw.float()
            up = lambda t: F.interpolate(t.float(), size=(h, w), mode="bilinear", align_corners=False)
            tD, tB = up(phys["tD"]), up(phys["tB"])
            B = g["B_inf"].float()[:, :, None, None] * up(phys["B_low"])
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
            nig = torch.cat([mu, F.softplus(nu_r) + 1e-4, F.softplus(a_r) + 1.0001, F.softplus(b_r) + 1e-4], 1)
        _check(nig, n, 12, h, w, "nig")
        return {"nig": nig, "J": nig[:, 0:3], "nu": nig[:, 3:6], "alpha": nig[:, 6:9], "beta": nig[:, 9:12],
                "tD": tD, "tB": tB, "B": B, "M": g["M"].float(), "kappa": g["kappa"].float(), "J_phys": J_phys,
                "field_res": (phys["tD"].shape[-2], phys["tD"].shape[-1])}


def _selftest(device=None, vram=False):
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "training"))
    from losses_sspgn import SSPGNLoss, render
    dev = device or "cpu"
    torch.manual_seed(0)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    net = SSPGNCascade().to(dev)
    print(f"  params: {net.n_params:,} (budget 5,000,000), multiple={net.cfg.multiple}")
    parts = {"head+tails": [net.head, net.tail_R, net.tail_nig], "enc NAF": [net.enc], "enc SS-PGB": [net.enc_pgb],
             "down": [net.down], "bottleneck SS-PGB": [net.bottleneck], "global phys": [net.gphys],
             "up": [net.up], "dec NAF": [net.dec], "dec SS-PGB": [net.dec_pgb]}
    for k, mods in parts.items():
        print(f"    {k:18s} {sum(p.numel() for mo in mods for p in mo.parameters()):>10,}")
    check("param budget <= 5M", net.n_params <= 5_000_000)

    x = torch.rand(2, 3, 64, 96, device=dev)
    out = net(x)
    check("output shapes", tuple(out["nig"].shape) == (2, 12, 64, 96) and tuple(out["tD"].shape) == (2, 3, 64, 96))
    check("final field at H/2", out["field_res"] == (32, 48), f"(got {out['field_res']})")
    check("NIG constraints", bool((out["nu"] > 0).all() and (out["alpha"] > 1).all() and (out["beta"] > 0).all()))
    loss, logs = SSPGNLoss()({k: v for k, v in out.items() if k != "field_res"}, x, torch.rand_like(x))
    loss.backward()
    nograd = [n for n, p in net.named_parameters() if p.grad is None]
    check("loss finite and every parameter receives a gradient", torch.isfinite(loss) and not nograd,
          f"(no grad: {nograd[:4]})")

    if vram and dev == "cuda":
        crit = SSPGNLoss()
        opt = torch.optim.AdamW(net.parameters(), 1e-4)
        scaler = torch.amp.GradScaler("cuda")
        for bs in (2, 4):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                xb, yb = torch.rand(bs, 3, 256, 256, device=dev), torch.rand(bs, 3, 256, 256, device=dev)
                with torch.autocast("cuda", dtype=torch.float16):
                    o = net(xb)
                o.pop("field_res")
                l, _ = crit(o, xb, yb)
                opt.zero_grad(set_to_none=True); scaler.scale(l).backward(); scaler.step(opt); scaler.update()
                check(f"AMP step bs={bs} @256", torch.isfinite(l),
                      f"(peak VRAM={torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB)")
            except torch.cuda.OutOfMemoryError:
                check(f"AMP step bs={bs} @256", False, "(OOM)")
    return ok


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--vram", action="store_true", help="measure peak VRAM (needs a free GPU)")
    a = ap.parse_args()
    sys.exit(0 if _selftest(a.device, a.vram) else 1)

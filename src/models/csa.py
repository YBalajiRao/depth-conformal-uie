"""
Blind ChromaStyle-Adapter (CSA). Redesign Phase 1 SCAFFOLD: not wired into SS-PGN, not trained.

    J_style = M(s) . stopgrad(J_struct) + b(s),     s = Pool_{mu,sigma}(I) in R^6

* Input is the *detached* structural output J_struct: no CSA gradient reaches the restorer.
* The only conditioning is s = per-channel mean and standard deviation of the degraded input I. There is
  no dataset-label input anywhere in the interface (forward(J_struct, I) only), so the adapter cannot
  branch on LSUI vs UIEB except through image statistics.
* Per-image affine colour map: M = I_3 + m_scale * tanh(.) (3x3), b = b_scale * tanh(.) (3). The last
  layer is zero-initialised, so at initialisation J_style == J_struct exactly (identity).
* 12 output parameters per image; the global map cannot change spatial structure.

Open design items (not decided here): the loss that trains J_struct versus J_style (e.g. a colour-
aligned structural loss on J_struct and the reference loss on J_style), and whether J_style is clamped
in the loss or only for output (clamping zeroes gradients outside [0, 1]).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def pool_stats(I: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """I: (N, 3, H, W) -> (N, 6) = [mean_R, mean_G, mean_B, std_R, std_G, std_B]."""
    mu = I.mean((2, 3))
    sd = (I.var((2, 3), unbiased=False) + eps).sqrt()
    return torch.cat([mu, sd], 1)


class ChromaStyleAdapter(nn.Module):
    n_out = 12

    def __init__(self, hidden: int = 64, m_scale: float = 0.5, b_scale: float = 0.2):
        super().__init__()
        self.m_scale, self.b_scale = float(m_scale), float(b_scale)
        self.mlp = nn.Sequential(nn.Linear(6, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, self.n_out))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, J_struct: torch.Tensor, I: torch.Tensor, clamp: bool = False):
        """
        J_struct: (N, 3, H, W) structural restoration output (detached here)
        I       : (N, 3, H', W') degraded input (statistics only; any resolution)
        returns dict J_style (N,3,H,W), M (N,3,3), b (N,3), stats (N,6)
        """
        if J_struct.shape[1] != 3 or I.shape[1] != 3:
            raise ValueError("J_struct and I must have 3 channels")
        s = pool_stats(I.detach())
        p = self.mlp(s)
        eye = torch.eye(3, device=p.device, dtype=p.dtype).expand(p.shape[0], 3, 3)
        M = eye + self.m_scale * torch.tanh(p[:, :9]).view(-1, 3, 3)
        b = self.b_scale * torch.tanh(p[:, 9:12])
        J = torch.einsum("nij,njhw->nihw", M, J_struct.detach()) + b[:, :, None, None]
        if clamp:
            J = J.clamp(0.0, 1.0)
        return {"J_style": J, "M": M, "b": b, "stats": s}

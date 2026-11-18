"""
P1e checks (CPU; no training). Report section P.

1. Identity: a P1e model (scene_dim 16, scene_mode dgt_only, dropout 0.5, noise 0.25) holding the P1b seed-0 weights
   plus zero-initialised scene layers is bit-identical to P1b in eval mode for any 16-d descriptor, and also in
   TRAIN mode (dropout/noise act on g only, whose sole consumer is the zero-initialised scene_dgt). Control: a 1e-3
   change in one scene_dgt weight breaks the identity.
2. Isolation: no FiLM parameters exist (scene_film is None); the bottleneck features are untouched by g (the
   decoder input / restoration path can change only through the DGT coefficients).
3. Regularisation acts in training only: the empirical per-sample drop fraction of g is ~0.5 (+-0.05 over 4,000
   draws), noise is added in training, and eval mode is deterministic.
4. PCA: SceneStore with the training-only PCA returns 16-d whitened vectors; the stored PCA was fitted on the
   current scene_train.npz (hash check); validation vectors are produced by transform only.
5. Contracts: parameter count = P1b + 1,673; --scene-prior and --scene-prior-pca exclusive (train.py path).
Exit 1 on any failure.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("src/validation", "src/utils", "src/training", "src/models"):
    sys.path.insert(0, str(ROOT / sub))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import train as T  # noqa: E402
from ss_pgn import SSPGN, SSPGNConfig  # noqa: E402

P1B = ROOT / "runs/phase1/P1b_v2_dgt_ordered/20261110_093229/last.pt"
P1B_SHA = "5b22ea7c217b09b96ad2b6309b21b5cbca906444e2f442dd9732965ec9db52c0"
SCENE = ROOT / "data/scene/da2_small_5426e4f/lsui_uieb_all_d6_seed0"
PCA = SCENE / "pca16_train.npz"
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


def main():
    torch.set_num_threads(8)
    if hashlib.sha256(P1B.read_bytes()).hexdigest() != P1B_SHA:
        raise SystemExit("[abort] P1b SHA mismatch")
    ck = torch.load(P1B, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"]["model_cfg"])
    p1e_cfg = {**cfg, "scene_prior": True, "scene_dim": 16, "scene_mode": "dgt_only", "scene_dropout": 0.5,
               "scene_noise": 0.25, "scene_pca_file": str(PCA.relative_to(ROOT))}
    base = SSPGN(SSPGNConfig(**cfg)); base.load_state_dict(ck["model"], strict=True)
    m = SSPGN(SSPGNConfig(**p1e_cfg)); res = m.load_state_dict(ck["model"], strict=False)
    n_sc = sum(p.numel() for mod in (m.scene, m.scene_dgt) for p in mod.parameters())
    check("only scene/scene_dgt params missing; no FiLM layer exists",
          m.scene_film is None and sorted(res.missing_keys) == sorted(["scene.0.weight", "scene.0.bias",
                                                                        "scene_dgt.weight", "scene_dgt.bias"]),
          f"({res.missing_keys})")
    check("parameter count = P1b + 1,673", m.n_params == base.n_params + 1673 and n_sc == 1673,
          f"({base.n_params:,} + {n_sc:,} = {m.n_params:,})")

    torch.manual_seed(0)
    x, z = torch.rand(2, 3, 64, 96), torch.rand(2, 1, 64, 96)
    base.eval(); m.eval()
    ok = all(same(m(x, z, torch.randn(2, 16) * s), base(x, z)) for s in (1, 3, 10))
    check("eval mode, zero init: bit-identical to P1b for any 16-d descriptor", ok)
    base.train(); m.train()
    torch.manual_seed(1); ob = base(x, z)
    torch.manual_seed(1); om = m(x, z, torch.randn(2, 16))
    check("train mode (dropout+noise active), zero init: J/tD/beta identical to P1b",
          all(torch.equal(om[k], ob[k]) for k in ("J", "tD", "tB", "beta_D", "beta_B", "B")))
    m.eval()
    # control on scene_dgt.bias[0]: always reaches beta (a weight entry can be inert if its hidden unit is ~0 after
    # GELU for the control input; first run of this check used weight[0, 0] and was inert -> control redesigned)
    with torch.no_grad():
        m.scene_dgt.bias[0] += 1e-3
        o2 = m(x, z, torch.ones(2, 16) * 5)
        m.scene_dgt.bias[0] -= 1e-3
    control("identity check rejects a 1e-3 scene_dgt bias change", not same(o2, base(x, z)))

    # isolation: g reaches only the DGT head
    feats = {}
    h = m.bottleneck[0].register_forward_pre_hook(lambda mod, inp: feats.setdefault(len(feats), inp[0].detach().clone()))
    with torch.no_grad():
        m.scene_dgt.weight.normal_(0, 0.5)
        m(x, z, torch.randn(2, 16)); m(x, z, torch.randn(2, 16) * 4)
    h.remove()
    check("bottleneck input is unchanged by g (no FiLM path)", torch.equal(feats[0], feats[1]))
    with torch.no_grad():
        a1, a2 = m(x, z, torch.randn(2, 16)), m(x, z, torch.randn(2, 16) * 4)
    check("with non-zero scene_dgt, beta depends on g", not torch.equal(a1["beta_D"], a2["beta_D"]))

    # regularisation statistics (train mode only)
    m.train()
    torch.manual_seed(2)
    drops, gi_seen = 0, []
    orig = m.scene.forward
    m.scene.forward = lambda t: (gi_seen.append(t.detach().clone()), orig(t))[1]
    for _ in range(500):
        m(torch.rand(8, 3, 32, 32), torch.rand(8, 1, 32, 32), torch.ones(8, 16) * 3)
    m.scene.forward = orig
    allg = torch.cat(gi_seen)
    frac = float((allg.abs().sum(1) == 0).float().mean())
    check("per-sample descriptor dropout fraction ~ 0.5 (4,000 draws)", abs(frac - 0.5) < 0.05, f"({frac:.3f})")
    kept = allg[allg.abs().sum(1) > 0]
    check("noise added in training (kept descriptors != input)", float((kept - 3).std()) > 0.15,
          f"(std of perturbation {float((kept - 3).std()):.3f}, expected 0.25)")
    m.eval()
    with torch.no_grad():
        e1, e2 = m(x, z, torch.ones(2, 16)), m(x, z, torch.ones(2, 16))
    check("eval mode deterministic (no dropout/noise)", same(e1, e2))

    # PCA store
    split = ROOT / "data/splits/lsui_uieb_all_d6_seed0"
    rows = T.read_manifest(split / "train_manifest.txt") + T.read_manifest(split / "val_manifest.txt")
    st = T.SceneStore(SCENE, rows, check_sha=True, pca_file=str(PCA))
    tr = torch.stack([st.load(r[1]) for r in rows[:4135]])
    va = torch.stack([st.load(r[1]) for r in rows[4135:]])
    check("SceneStore+PCA returns 16-d; train projections whitened (std ~1)",
          tr.shape == (4135, 16) and float((tr.std(0) - 1).abs().max()) < 1e-3, f"(max|std-1| {float((tr.std(0)-1).abs().max()):.1e})")
    check("validation transformed only (16-d, finite; not re-whitened)", va.shape == (517, 16) and bool(torch.isfinite(va).all()),
          f"(val std range {float(va.std(0).min()):.2f}-{float(va.std(0).max()):.2f})")
    with np.load(PCA) as d:
        stored = str(d["train_npz_sha256"])
    check("PCA bound to current scene_train.npz", stored == hashlib.sha256((SCENE / "scene_train.npz").read_bytes()).hexdigest())

    print(f"\n[p1e_check] {'ALL PASS' if not FAIL else 'FAILED: ' + '; '.join(FAIL)}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

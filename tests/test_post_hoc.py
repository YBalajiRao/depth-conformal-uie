"""
CPU tests for the post-hoc uncertainty head (no GPU is touched: CUDA_VISIBLE_DEVICES is cleared).

Stand-in restorer: runs/sspgn/20261106_173805/snapshots/step4000.pt. These tests were written before
the final 100k checkpoint existed; it now exists (runs/sspgn/20261106_173805/last.pt), but the tests
still deliberately use the step-4000 snapshot. Every test can fail for the reason it names.

  python tests/test_post_hoc.py
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import hashlib
import inspect
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src/models", "src/training", "src/utils", "src/validation"):
    sys.path.insert(0, str(ROOT / sub))
from post_hoc_head import SSPGNWithPostHoc, PostHocConfig, gaussian_nll, SIGMA_FLOOR  # noqa: E402
from ss_pgn import SSPGN, SSPGNConfig  # noqa: E402
from pad_crop_wrapper import PadCropWrapper, pad_amounts  # noqa: E402
import train_post_hoc as TP  # noqa: E402

SNAP = ROOT / "runs/sspgn/20261106_173805/snapshots/step4000.pt"
SNAP_SHA = (ROOT / "runs/sspgn/20261106_173805/snapshots/step4000.sha256").read_text().split()[0]
torch.set_num_threads(4)
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


def main():
    sha_before = TP.sha256(SNAP)
    check("T0 stand-in checkpoint SHA matches recorded value", sha_before == SNAP_SHA, f"({sha_before[:12]})")

    # T7 checkpoint loading + SHA verification
    model, info = TP.load_frozen(SNAP, SNAP_SHA, require_final=False)
    check("T7a load with correct SHA", info["sha256"] == SNAP_SHA and info["step"] == 4000)
    for bad, label in [(("0" * 64, False), "wrong SHA rejected"), ((None, True), "non-final checkpoint rejected")]:
        try:
            TP.load_frozen(SNAP, bad[0], require_final=bad[1]); check(f"T7 {label}", False)
        except RuntimeError:
            check(f"T7 {label}", True)

    # T1 hooks do not change the restorer's forward pass
    ck = torch.load(SNAP, map_location="cpu", weights_only=False)
    ref_net = SSPGN(SSPGNConfig(**ck["cfg"]["model_cfg"])); ref_net.load_state_dict(ck["model"]); ref_net.eval()
    x = torch.rand(1, 3, 128, 160)
    with torch.no_grad():
        a, b = ref_net(x), model(x)
    check("T1 restored output bit-identical to standalone SS-PGN", torch.equal(a["J"], b["J"]))

    # T2 shapes / sizes through the wrapper (no reference passed anywhere)
    wrapped = PadCropWrapper(model, model.cfg.multiple, mode="reflect")
    ok = True
    for h, w in [(256, 256), (240, 320), (159, 225), (413, 550)]:
        with torch.no_grad():
            o = wrapped(torch.rand(1, 3, h, w))
        ok &= (tuple(o["nig"].shape) == (1, 12, h, w) and tuple(o["sigma"].shape) == (1, 3, h, w)
               and bool((o["sigma"] >= SIGMA_FLOOR).all()) and all(torch.isfinite(o[k]).all() for k in ("nig", "sigma")))
    check("T2 shapes, floor and finiteness at 256x256, 240x320, 159x225, 413x550", ok)

    # T3 V3a / V3b on the combined model
    with torch.no_grad():
        xa = torch.rand(1, 3, 240, 320)
        v3a = torch.equal(wrapped(xa)["sigma"], model(xa)["sigma"])
        xb = torch.rand(1, 3, 159, 225); ph, pw = pad_amounts(159, 225, 16)
        man = model(F.pad(xb, (0, pw, 0, ph), mode="reflect"))["sigma"][..., :159, :225]
        v3b = torch.equal(wrapped(xb)["sigma"], man)
    check("T3 V3a (aligned) wrapper == direct, bit-exact", v3a)
    check("T3 V3b (159x225) wrapper == reflect-pad -> model -> crop, bit-exact", v3b)

    # T5 inference needs only the degraded image
    params = list(inspect.signature(model.forward).parameters)
    check("T5 forward(x) takes only the degraded image", params == ["x"], f"{params}")

    # T4/T6 freezing and gradient flow (one optimiser step)
    before = {k: v.clone() for k, v in model.restorer.state_dict().items()}
    check("T4a all restorer params have requires_grad=False", all(not p.requires_grad for p in model.restorer.parameters()))
    opt = torch.optim.AdamW(model.head.parameters(), lr=1e-3)
    model.train()
    check("T4b restorer stays in eval mode under model.train()", not model.restorer.training)
    x, y = torch.rand(2, 3, 128, 128), torch.rand(2, 3, 128, 128)
    o = model(x)
    loss = gaussian_nll(y - o["J"], o["sigma"])
    check("T6 NLL and sigma finite", torch.isfinite(loss) and torch.isfinite(o["sigma"]).all(), f"(nll={loss.item():.4f})")
    opt.zero_grad(); loss.backward()
    check("T4c no gradient reaches the restorer", all(p.grad is None for p in model.restorer.parameters()))
    hg = [p.grad for p in model.head.parameters()]
    check("T4d every head parameter receives a finite gradient",
          all(g is not None and torch.isfinite(g).all() for g in hg) and sum(g.abs().sum() for g in hg) > 0)
    opt.step()
    after = model.restorer.state_dict()
    check("T4e restorer weights unchanged after an optimiser step", all(torch.equal(before[k], after[k]) for k in before))

    # T11 NIG adapter reproduces sigma under the existing 'total' predictive variance
    from calibrate_uncertainty import predictive_std
    with torch.no_grad():
        o = model(torch.rand(1, 3, 64, 64))
    s2 = predictive_std(o["nu"], o["alpha"], o["beta"], "total")
    check("T11 adapter: predictive_std(total) == sigma", torch.allclose(s2, o["sigma"], rtol=1e-5, atol=1e-7),
          f"(max rel err {((s2 - o['sigma']).abs() / o['sigma']).max().item():.1e})")

    # T11b adapter inverse over a grid of (sigma, nu, alpha), checked against the PROJECT equation
    from post_hoc_head import nig_from_sigma
    sig = torch.tensor([1e-3, 1e-2, 0.05, 0.2, 1.0], dtype=torch.float64)
    worst = 0.0
    for nu_v in [1e-3, 0.1, 1.0, 10.0, 1e4]:
        for a_v in [1.0001, 1.5, 2.0, 5.0, 50.0]:
            n_, a_, b_ = nig_from_sigma(sig, nu_v, a_v)
            got = predictive_std(n_, a_, b_, "total")
            worst = max(worst, ((got - sig).abs() / sig).max().item())
    check("T11b adapter exact for 5 sigma x 5 nu x 5 alpha (float64, project predictive_std)", worst < 1e-12,
          f"(max rel err {worst:.1e})")
    try:
        nig_from_sigma(sig, 1.0, 1.0); check("T11c adapter rejects alpha <= 1", False)
    except ValueError:
        check("T11c adapter rejects alpha <= 1", True)
    # T11d the model's float32 path at its own (nu=1e4, alpha=2) on a sigma spread
    s32 = torch.tensor([1e-3, 0.03, 0.3, 1.0])
    n_, a_, b_ = nig_from_sigma(s32)
    check("T11d float32 path at nu=1e4, alpha=2", torch.allclose(predictive_std(n_, a_, b_, "total"), s32, rtol=1e-5))

    # T13 Gaussian NLL equals -log N(e; 0, sigma^2)
    e, s = torch.randn(1000, dtype=torch.float64), torch.rand(1000, dtype=torch.float64) + 0.1
    ref = -torch.distributions.Normal(0, s).log_prob(e).mean()
    check("T13 gaussian_nll == -log Normal", abs(gaussian_nll(e, s).item() - ref.item()) < 1e-12)

    # T8 deterministic split, disjoint components, never val/test
    tr1, ho1 = TP.split_rows_by_component(ROOT / "data/splits/lsui_uieb_all_d6_seed0/train_manifest.txt", 0.05, 0)
    tr2, ho2 = TP.split_rows_by_component(ROOT / "data/splits/lsui_uieb_all_d6_seed0/train_manifest.txt", 0.05, 0)
    val_units = {l.split("\t")[1] for l in (ROOT / "data/splits/lsui_uieb_all_d6_seed0/val_manifest.txt").read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")}
    test_units = {l.split("\t")[1] for l in (ROOT / "data/splits/lsui_uieb_all_d6_seed0/test_manifest.txt").read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")}
    used = {r[1] for r in tr1 + ho1}
    check("T8a split deterministic", tr1 == tr2 and ho1 == ho2, f"(train {len(tr1)}, holdout {len(ho1)})")
    check("T8b no val/test units used by the head", not (used & val_units) and not (used & test_units))

    # T9 reproducible initialisation
    torch.manual_seed(0); h1 = SSPGNWithPostHoc(PostHocConfig(restorer_cfg=dict(ck["cfg"]["model_cfg"]))).head
    torch.manual_seed(0); h2 = SSPGNWithPostHoc(PostHocConfig(restorer_cfg=dict(ck["cfg"]["model_cfg"]))).head
    check("T9 same seed -> identical head init", all(torch.equal(a, b) for a, b in zip(h1.parameters(), h2.parameters())))

    # T14 end-to-end CPU mini-run of the training script + reload through the UNCHANGED evaluation loader
    with tempfile.TemporaryDirectory() as td:
        run = TP.main(["--ckpt", str(SNAP), "--expect-sha", SNAP_SHA, "--allow-nonfinal", "--steps", "3",
                       "--batch", "2", "--workers", "0", "--device", "cpu", "--holdout-frac", "0.01",
                       "--log-every", "1", "--out", td])
        saved = torch.load(run / "model_last.pt", map_location="cpu", weights_only=False)
        rest = {k[len("restorer."):]: v for k, v in saved["model"].items() if k.startswith("restorer.")}
        check("T14a saved model_last.pt carries the restorer weights unchanged",
              all(torch.equal(rest[k], ck["model"][k]) for k in ck["model"]))
        from calibrate_uncertainty import load_model
        m2, info2 = load_model(run / "model_last.pt", torch.device("cpu"))
        xx = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            check("T14b checkpoint reloads via calibrate_uncertainty.load_model and reproduces sigma",
                  info2["arch"] == "posthoc" and torch.equal(m2(xx)["sigma"], SSPGNWithPostHoc.forward(saved_model(saved, ck), xx)["sigma"]))
        cfgj = (run / "config.json").read_text()
        check("T14c config records seed, split hash, ckpt hash, optimizer, lr, steps, argv",
              all(k in cfgj for k in ['"seed"', "train_manifest_sha256", SNAP_SHA, '"AdamW"', '"lr"', '"steps"', '"argv"']))

    # T10 main checkpoint untouched
    check("T10 stand-in checkpoint byte-identical after all tests", TP.sha256(SNAP) == sha_before)

    print(f"\n{sum(results)}/{len(results)} passed")
    return all(results)


def saved_model(saved, ck):
    m = SSPGNWithPostHoc(PostHocConfig(restorer_cfg=dict(ck["cfg"]["model_cfg"])))
    m.load_state_dict(saved["model"], strict=True)
    return m.eval()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

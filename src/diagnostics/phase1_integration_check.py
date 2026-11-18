"""
Phase-1 integration checks (DGT + CSA + depth data path). CPU only; no training, no checkpoint writes.

A  Default preservation: with the new switches off, the model, loss, pad/crop wrapper and crop dataset are
   numerically identical to the versions at git HEAD (loaded from `git show HEAD:<file>`), and the Stage-1
   checkpoints still load strictly.
B  DGT: parameter count, full-resolution t = exp(-beta (1 - z~)) exactly, beta bounds, coupling term = 0,
   depth dependence and gradient routing, z argument contract, pad/crop with depth.
C  CSA: identity at init, gradient separation (L_total -> CSA only; L_struct -> never CSA), aligned
   L_struct exactness, NIG mean = J_struct.
D  Data: every train/val unit has a depth map bound to its input SHA-256; pad/crop/rot90/flip keep depth
   aligned with the image (depth replaced by a known function of the image), including padded small images.
Every property has a failure control that must be rejected. Exit code 1 on any failure.
"""
from __future__ import annotations

import importlib.util
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("src/validation", "src/utils", "src/training", "src/models", "src/data"):
    sys.path.insert(0, str(ROOT / sub))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import train as T  # noqa: E402
from ss_pgn import SSPGN, SSPGNConfig  # noqa: E402
from losses_sspgn import SSPGNLoss, StructStyleLoss, affine_align  # noqa: E402
from pad_crop_wrapper import PadCropWrapper  # noqa: E402

FAIL = []
SCR = ROOT / "runs/diagnostics/_phase1_head_copies"


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAIL.append(name)


def control(name, rejected):
    print(f"[{'CTRL-OK' if rejected else 'CTRL-FAIL'}] {name}", flush=True)
    if not rejected:
        FAIL.append("control: " + name)


def head_module(rel, name, patch=None):
    src = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8", check=True).stdout
    if patch:
        for a, b in patch:
            assert src.count(a) == 1, a
            src = src.replace(a, b)
    SCR.mkdir(parents=True, exist_ok=True)
    f = SCR / f"{name}.py"
    f.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, f)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                       # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


def same(a, b):
    return torch.equal(a, b)


# ============================================================== A. default preservation
def part_a():
    old_m = head_module("src/models/ss_pgn.py", "head_ss_pgn")
    old_l = head_module("src/training/losses_sspgn.py", "head_losses")
    old_w = head_module("src/utils/pad_crop_wrapper.py", "head_wrapper")
    torch.manual_seed(0)
    x = torch.rand(2, 3, 64, 96)
    for mode in ("v1", "v2"):
        torch.manual_seed(1); new = SSPGN(SSPGNConfig(phys_conditioning=mode)).eval()
        torch.manual_seed(1); old = old_m.SSPGN(old_m.SSPGNConfig(phys_conditioning=mode)).eval()
        check(f"A {mode}: parameter count 4,152,052 and identical state_dict keys",
              new.n_params == 4152052 and list(new.state_dict()) == list(old.state_dict()))
        sd = {k: v + 0.01 * torch.randn_like(v) for k, v in old.state_dict().items()}   # non-trivial weights
        old.load_state_dict(sd); new.load_state_dict(sd)
        with torch.no_grad():
            on, oo = new(x), old(x)
        check(f"A {mode}: forward outputs bit-identical to HEAD", set(on) == set(oo) and all(same(on[k], oo[k]) for k in oo))
    oo_bad = dict(oo); oo_bad["J"] = oo["J"] + 1e-7
    control("bit-identity check rejects a 1e-7 output change", not all(same(on[k], oo_bad[k]) for k in oo_bad))

    for name, path in (("v1_retrain", "S1_v1_retrain/20261107_102639"), ("v2_full", "S1_v2_full/20261107_193709")):
        ck = torch.load(ROOT / "runs/ablation/stage1" / path / "last.pt", map_location="cpu", weights_only=False)
        m = SSPGN(SSPGNConfig(**ck["cfg"]["model_cfg"]))
        m.load_state_dict(ck["model"], strict=True)
        check(f"A Stage-1 {name} last.pt loads strictly into the new SSPGN", True)

    # loss: default path identical, value and gradients
    torch.manual_seed(2)
    N, H, W = 2, 64, 64
    I, Y = torch.rand(N, 3, H, W), torch.rand(N, 3, H, W)
    mk = torch.ones(N, 1, H, W); mk[..., 50:, :] = 0
    res = []
    for L in (SSPGNLoss(), old_l.SSPGNLoss()):
        torch.manual_seed(3)
        out = {"J": torch.rand(N, 3, H, W, requires_grad=True),
               "tD": (torch.rand(N, 3, H, W) * 0.8 + 0.1).requires_grad_(True),
               "tB": (torch.rand(N, 3, H, W) * 0.8 + 0.1).requires_grad_(True),
               "B": torch.rand(N, 3, H, W, requires_grad=True),
               "M": (torch.eye(3).expand(N, 3, 3) + 0.05 * torch.randn(N, 3, 3)).requires_grad_(True),
               "kappa": torch.rand(N, 3, requires_grad=True),
               "nu": torch.rand(N, 3, H, W) + 0.1, "alpha": torch.rand(N, 3, H, W) + 1.1,
               "beta": torch.rand(N, 3, H, W) + 0.1}
        tot, logs = L(out, I, Y, mk)
        tot.backward()
        res.append((tot, logs, [out[k].grad for k in ("J", "tD", "tB", "B", "M", "kappa")]))
    (tn, ln, gn), (to, lo, go) = res
    check("A SSPGNLoss default: total, logs and gradients bit-identical to HEAD",
          same(tn, to) and all(same(ln[k], lo[k]) for k in lo) and all(same(a, b) for a, b in zip(gn, go)))

    # wrapper without aux
    torch.manual_seed(4)
    mdl = SSPGN(SSPGNConfig()).eval()
    xs = torch.rand(1, 3, 70, 90)
    with torch.no_grad():
        a, b = PadCropWrapper(mdl, 16)(xs), old_w.PadCropWrapper(mdl, 16)(xs)
    check("A PadCropWrapper without aux identical to HEAD", all(same(a[k], b[k]) for k in a))

    # crop dataset without depth: identical samples for identical RNG state
    old_t = head_module("src/training/train.py", "head_train",
                        patch=[("ROOT = Path(__file__).resolve().parents[2]", f"ROOT = Path(r'{ROOT}')")])
    rows = T.read_manifest(ROOT / "data/splits/lsui_uieb_all_d6_seed0/val_manifest.txt")
    small = sorted(rows, key=lambda r: np.prod(T.load_rgb_u8(r[2]).shape[:2]))[:2] + rows[:2]
    ok = True
    for r in small:
        outs = []
        for ds_cls in (T.PairedCropDataset, old_t.PairedCropDataset):
            random.seed(7)
            outs.append(ds_cls([r], 256)[0])
        ok &= all(same(p, q) for p, q in zip(*outs))
    check("A PairedCropDataset without depth identical to HEAD (incl. padded small images)", ok)


# ============================================================== B. DGT
def dgt_model(csa=False):
    torch.manual_seed(5)
    m = SSPGN(SSPGNConfig(phys_conditioning="v2", depth_transmission=True, csa=csa))
    with torch.no_grad():                         # leave the zero-init so that R and the block output are active
        m.tail_R.weight.normal_(0, 1e-2)
        m.bottleneck[0].out_scale.fill_(0.1)
        m.dgt.mlp[-1].weight.normal_(0, 0.05)
    return m


def part_b():
    m = dgt_model()
    n_dgt = sum(p.numel() for p in m.dgt.parameters())
    check("B DGT params = 9 outputs; total params", m.dgt.n_params == 9 and m.n_params == 4152052 + n_dgt,
          f"(DGT {n_dgt:,}; model {m.n_params:,})")
    torch.manual_seed(6)
    x = torch.rand(2, 3, 64, 64)
    z = torch.rand(2, 1, 64, 64)
    out = m(x, z)
    tD_ref = torch.exp(-out["beta_D"][:, :, None, None] * (1 - z))
    check("B full-res tD == exp(-beta_D (1 - z~)) exactly", torch.allclose(out["tD"], tD_ref, atol=0, rtol=1e-6))
    control("formula check rejects the literal-disparity form exp(-beta z~)",
            not torch.allclose(out["tD"], torch.exp(-out["beta_D"][:, :, None, None] * z), rtol=1e-6))
    b = torch.cat([out["beta_D"], out["beta_B"]], 1)
    check("B beta > b_min (0.05)", bool((b > 0.05).all()))
    crit_b = SSPGNLoss()
    tot_b, lg_b = crit_b(out, x, torch.rand_like(x))
    check("B barrier reaches the loss when beta > b_max",
          "beta_barrier" in lg_b and float(m.dgt.barrier(torch.full((1, 3), 3.0))) > 0)
    zf, tf = z[0, 0].flatten(), out["tD"][0].flatten(1)
    near, far = zf.argmax(), zf.argmin()
    check("B nearer pixel has higher tD (all channels)", bool((tf[:, near] > tf[:, far]).all()))
    crit = SSPGNLoss()
    _, lg = crit.physics(x, out["J"], out["tD"], out["tB"], out["B"], out["M"], out["kappa"])
    check("B coupling term is 0 by construction (kappa = beta_D / beta_D,G)", lg["phys_couple"].item() < 1e-6,
          f"({lg['phys_couple'].item():.1e})")

    # depth reaches the restoration output; restoration loss reaches the DGT
    z2 = z.clone(); z2[..., :32, :] = 1 - z2[..., :32, :]
    with torch.no_grad():
        dJ = (m(x, z)["J"] - m(x, z2)["J"]).abs().max().item()
    check("B restoration output depends on the depth map", dJ > 0, f"(max|dJ| {dJ:.2e})")
    m.zero_grad()
    (m(x, z)["J"] - x).abs().mean().backward()
    g = sum(p.grad.abs().sum().item() for p in m.dgt.parameters() if p.grad is not None)
    check("B restoration-only loss reaches the DGT head", g > 0, f"(sum|grad| {g:.2e})")
    w = m.bottleneck[0].phys_out.weight.grad                      # (9, g, 1, 1) grouped: per colour (tD,tB,B)
    m.zero_grad()
    tot, _ = SSPGNLoss()(m(x, z), x, torch.rand_like(x))
    tot.backward()
    w = m.bottleneck[0].phys_out.weight.grad.view(3, 3, -1)
    unused = w[:, 0:2].abs().sum().item()
    used = w[:, 2].abs().sum().item()
    check("B field-branch tD/tB channels unused under DGT; B_low channel used", unused == 0 and used > 0,
          f"(|g tD,tB| {unused:.1e}, |g B_low| {used:.1e})")

    for bad, label in ((None, "missing z"), (torch.rand(2, 1, 32, 32), "misaligned z")):
        try:
            m(x, bad); check(f"B {label} rejected", False)
        except (AssertionError, ValueError):
            check(f"B {label} rejected", True)
    try:
        SSPGN(SSPGNConfig())(x, z); check("B z without DGT rejected", False)
    except (AssertionError, ValueError):
        check("B z without DGT rejected", True)
    try:
        SSPGN(SSPGNConfig(depth_transmission=True)); check("B DGT without v2 rejected", False)
    except (AssertionError, ValueError):
        check("B DGT without v2 rejected", True)

    m.eval()
    xs, zs = torch.rand(1, 3, 70, 90), torch.rand(1, 1, 70, 90)
    with torch.no_grad():
        o = PadCropWrapper(m, 16)(xs, zs)
        xp = torch.nn.functional.pad(xs, (0, 6, 0, 10), mode="reflect")
        zp = torch.nn.functional.pad(zs, (0, 6, 0, 10), mode="reflect")
        ref = m(xp, zp)
    check("B PadCropWrapper pads depth exactly like the image",
          o["J"].shape[-2:] == (70, 90) and same(o["J"], ref["J"][..., :70, :90]) and same(o["tD"], ref["tD"][..., :70, :90]))
    with torch.no_grad():
        zp_bad = torch.nn.functional.pad(zs, (0, 6, 0, 10), mode="constant", value=0.0)
        bad = m(xp, zp_bad)
    # tD at full resolution is pointwise in z, so padding can only show up through the H/16 pooled depth
    # that conditions the restoration path: the control therefore compares J.
    control("padding check rejects zero-padded depth (via J)", not same(o["J"], bad["J"][..., :70, :90]))


# ============================================================== C. CSA
def part_c():
    torch.manual_seed(8)
    x, z, Y = torch.rand(2, 3, 64, 64), torch.rand(2, 1, 64, 64), torch.rand(2, 3, 64, 64)
    m = dgt_model(csa=True)
    n_csa = sum(p.numel() for p in m.csa.parameters())
    check("C model params (v2 + DGT + CSA) within 5M budget", m.n_params <= 5_000_000, f"({m.n_params:,}; CSA {n_csa:,})")
    out = m(x, z)
    check("C CSA identity at init: J_final == J_struct", same(out["J"], out["J_struct"]))
    check("C NIG mean in out['nig'] == deployed J_final", same(out["nig"][:, 0:3], out["J"]))
    from losses_sspgn import nig_nll, mmean
    with torch.no_grad():
        m.csa.mlp[-1].bias.fill_(0.2)                       # make J_final != J_struct
        o2 = m(x, z)
        _, lg2 = StructStyleLoss(struct_mode="aligned")(o2, x, Y)
        nll_final = mmean(nig_nll(Y, o2["J"], o2["nu"], o2["alpha"], o2["beta"]))
        nll_struct = mmean(nig_nll(Y, o2["J_struct"], o2["nu"], o2["alpha"], o2["beta"]))
    check("C L_NIG uses mean = J_final (aligned mode)", torch.allclose(lg2["nig_nll"], nll_final, rtol=1e-6),
          f"({lg2['nig_nll'].item():.5f} vs {nll_final.item():.5f})")
    control("NIG-mean check rejects mean = J_struct", not torch.allclose(lg2["nig_nll"], nll_struct, rtol=1e-6))
    with torch.no_grad():
        m.csa.mlp[-1].bias.zero_()
    with torch.no_grad():
        m.csa.mlp[-1].weight.normal_(0, 0.3); m.csa.mlp[-1].bias.normal_(0, 0.3)
    csa_ids = {id(p) for p in m.csa.parameters()}
    for mode in ("full", "aligned"):
        L = StructStyleLoss(struct_mode=mode)
        out = m(x, z)
        tot, logs = L(out, x, Y)
        o_s = dict(out); o_s["J"] = out["J_struct"]
        Jr = affine_align(out["J_struct"], Y) if mode == "aligned" else None
        m.zero_grad(); L.base(o_s, x, Y, J_rec=Jr)[0].backward(retain_graph=True)
        s_csa = sum(p.grad.abs().sum().item() for p in m.csa.parameters() if p.grad is not None)
        s_rest = sum(p.grad.abs().sum().item() for p in m.parameters() if id(p) not in csa_ids and p.grad is not None)
        m.zero_grad(); L.base.recon(out["J"], Y)[0].backward()
        f_csa = sum(p.grad.abs().sum().item() for p in m.csa.parameters() if p.grad is not None)
        f_rest = sum(p.grad.abs().sum().item() for p in m.parameters() if id(p) not in csa_ids and p.grad is not None)
        check(f"C [{mode}] L_struct reaches restorer only; L_total reaches CSA only",
              s_csa == 0 and s_rest > 0 and f_csa > 0 and f_rest == 0,
              f"(struct: csa {s_csa:.1e} rest {s_rest:.1e}; total: csa {f_csa:.1e} rest {f_rest:.1e})")
        check(f"C [{mode}] combined loss finite and = L_struct + L_total",
              torch.isfinite(tot) and abs(tot.item() - logs["struct_total"].item() - logs["final_total"].item()) < 1e-5)
    leak = torch.einsum("nij,njhw->nihw", out["csa_M"].detach(), out["J_struct"])
    m.zero_grad(); L.base.recon(leak, Y)[0].backward()
    control("gradient-separation check rejects a non-detached colour map",
            sum(p.grad.abs().sum().item() for p in m.parameters() if id(p) not in csa_ids and p.grad is not None) > 0)

    Js = torch.rand(2, 3, 32, 32)
    a, b = torch.tensor([[0.75, 1.3, 1.1], [0.9, 1.0, 1.35]]), torch.tensor([[0.1, -0.2, 0.05], [0.0, 0.3, -0.1]])
    Yl = a[..., None, None] * Js + b[..., None, None]
    d_al = (affine_align(Js, Yl, a_min=0.7, a_max=1.4) - Yl).abs().max().item()
    check("C aligned L_struct: exact for an affine cast with gain in [0.70, 1.40]", d_al < 1e-3, f"(max|d| {d_al:.1e})")
    Y4 = 4.0 * Js
    from losses_sspgn import affine_align_coeffs
    a4, _ = affine_align_coeffs(Js, Y4, a_min=0.7, a_max=1.4)
    check("C alignment gain clamped at 1.40 (true gain 4)", bool(torch.allclose(a4, torch.full_like(a4, 1.4))))
    control("clamp check rejects the old [0.1, 10] range", not torch.allclose(affine_align_coeffs(Js, Y4)[0],
                                                                            torch.full_like(a4, 1.4)))

    # gauge anchor + range penalty (aligned mode)
    Ls = StructStyleLoss(struct_mode="aligned")
    o3 = dict(out)
    o3["J_struct"] = (out["J_struct"].detach() * 3.0).requires_grad_(True)     # inflated, out of range
    tot3, lg3 = Ls(o3, x, Y)
    exp_anchor = (o3["J_struct"] - Y).abs().mean()
    exp_rng = torch.relu(o3["J_struct"] - 1).mean() + torch.relu(-o3["J_struct"]).mean()
    check("C anchor = L1(J_struct, Y) and range penalty > 0 for an inflated J_struct",
          torch.allclose(lg3["struct_anchor_l1"], exp_anchor) and torch.allclose(lg3["struct_range"], exp_rng)
          and float(lg3["struct_range"]) > 0)
    g3 = torch.autograd.grad(tot3, o3["J_struct"])[0]
    check("C anchor/range gradient on inflated J_struct points downward (mean dL/dJ > 0)", float(g3.mean()) > 0)
    o4 = dict(out); o4["J_struct"] = out["J_struct"].detach().clamp(0, 1)
    _, lg4 = Ls(o4, x, Y)
    check("C range penalty == 0 for J_struct in [0, 1]", float(lg4["struct_range"]) == 0.0)
    _, lg5 = StructStyleLoss(struct_mode="full")(o3, x, Y)
    control("full mode carries no anchor/range terms (ablation unchanged)",
            "struct_anchor_l1" not in lg5 and "struct_range" not in lg5)
    control("full-mode residual is non-zero for the same cast", (Js - Yl).abs().max().item() > 0.1)


# ============================================================== D. data
def part_d():
    split = ROOT / "data/splits/lsui_uieb_all_d6_seed0"
    tr, va = T.read_manifest(split / "train_manifest.txt"), T.read_manifest(split / "val_manifest.txt")
    t0 = time.time()
    store = T.DepthStore(ROOT / "data/depth/da2_small_5426e4f" / split.name, tr + va, check_sha=True)
    check("D every train+val unit has a depth map bound to its input SHA-256", True,
          f"({len(tr) + len(va)} units, {time.time() - t0:.1f}s)")
    vals = [store.load(r[1], T.load_rgb_u8(r[2]).shape[:2]) for r in va[:20]]
    check("D real maps are finite and in [0, 1]", all(torch.isfinite(v).all() and v.min() >= 0 and v.max() <= 1 for v in vals))

    class GrayStore:                              # depth := image luminance -> alignment is checkable
        def load(self, uid, hw):
            r = next(r for r in rows if r[1] == uid)
            return T.to_tensor(T.load_rgb_u8(r[2])).mean(0, keepdim=True)

    rows = sorted(va, key=lambda r: np.prod(T.load_rgb_u8(r[2]).shape[:2]))[:3] + tr[:5]
    ds = T.PairedCropDataset(rows, 256, augment=True, depth=GrayStore())
    random.seed(11)
    ok, ctl = True, True
    for i in range(len(rows)):
        for _ in range(4):
            x, y, mk, z = ds[i]
            ok &= torch.allclose(z, x.mean(0, keepdim=True), atol=1e-6)
            ctl &= not torch.allclose(z, torch.flip(x, (2,)).mean(0, keepdim=True), atol=1e-6)
    check("D pad/crop/rot90/flip keep depth aligned with the image (32 samples, incl. padded)", ok)
    control("alignment check rejects a depth map flipped relative to the image", ctl)

    dl = torch.utils.data.DataLoader(T.PairedCropDataset(tr[:16], 256, depth=store), batch_size=4, num_workers=0)
    b = next(iter(dl))
    check("D real DataLoader batch: x, y, m, z shapes", len(b) == 4 and tuple(b[3].shape) == (4, 1, 256, 256))


if __name__ == "__main__":
    torch.set_num_threads(8)
    for part in (part_a, part_b, part_c, part_d):
        print(f"\n== {part.__name__} ==", flush=True)
        part()
    print(f"\n[phase1_integration_check] {'ALL PASS' if not FAIL else 'FAILED: ' + '; '.join(FAIL)}")
    sys.exit(1 if FAIL else 0)

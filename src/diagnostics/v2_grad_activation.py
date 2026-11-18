"""
Gradient-activation timeline over the first optimizer steps (GPU, existing FP16 AMP protocol). Diagnostic only:
the model is trained for --steps steps in memory and DISCARDED (no checkpoint is written).

Re-uses train.py's own pieces (build_model, PairedCropDataset, read_manifest, cosine_lr, SSPGNLoss, seeds, AdamW
betas, GradScaler, clip, accumulation). Per optimizer step it logs, from the TRAINING loss (after unscale_):
grad norms of tail_R, out_scale, the field branch, gphys sub-heads (B_inf, beam, dM, kappa, MLP), tail_nig;
GradScaler scale; whether the step was skipped (inf/NaN grads); loss finiteness.
At probe steps it additionally measures the RESTORATION-loss-only gradient (L1 + SSIM + wav) into gphys on a fixed
batch (separate backward; grads cleared before the real step, so the optimizer is unaffected).

  python src/diagnostics/v2_grad_activation.py --phys-conditioning v2 --steps 300
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, git_state, new_outdir, write_json, environment, Timer, CKPTS, sha256_file  # noqa: E402
sys.path[:0] = [str(ROOT / "src/training"), str(ROOT / "src/models"), str(ROOT / "src/utils")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
import train as T  # noqa: E402
from losses_sspgn import SSPGNLoss, SSPGNLossWeights, mmean, ssim, haar_dwt  # noqa: E402

PROBES = {0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 299}


def gn(params):
    gs = [p.grad.detach().float() for p in params if p.grad is not None]
    if not gs:
        return 0.0
    v = float(torch.sqrt(sum((g * g).sum() for g in gs)))
    return v


def groups(m):
    gp = m.gphys
    return {"tail_R": list(m.tail_R.parameters()), "out_scale": [m.bottleneck[0].out_scale],
            "field_branch": list(m.bottleneck[0].phys_out.parameters()) + list(m.bottleneck[0].phys_body.parameters()),
            "gphys_binf": list(gp.binf.parameters()), "gphys_beam": list(gp.beam.parameters()),
            "gphys_dM": list(gp.dM.parameters()), "gphys_kappa": list(gp.kappa.parameters()),
            "gphys_mlp": list(gp.mlp.parameters()), "tail_nig": list(m.tail_nig.parameters())}


def rec_loss(out, y, m):
    J = out["J"]
    l = mmean((J - y).abs(), m) + 0.2 * (1 - ssim(J.clamp(0, 1), y, mask=m))
    _, a, b, c = haar_dwt(J); _, ya, yb, yc = haar_dwt(y)
    return l + 0.1 * ((a - ya).abs().mean() + (b - yb).abs().mean() + (c - yc).abs().mean())


def probe_grads(model, x, y, m, mode, scale=1.0):
    """Restoration-loss-only (L1 + SSIM + wav) gradient norms per group, as UNSCALED values.
    mode 'fp32'          : autocast disabled (primary; no fp16 underflow).
    mode 'fp16_scaled'   : fp16 autocast forward, loss * scale backward, grads / scale (GradScaler-style).
    mode 'fp16_unscaled' : the original (pre-fix) probe; kept only to document fp16 gradient underflow.
    Gradients are cleared before and after; the optimizer and model parameters are not changed."""
    model.zero_grad(set_to_none=True)
    if mode == "fp32":
        with torch.autocast(x.device.type, enabled=False):
            loss = rec_loss(model(x.float()), y, m)
        loss.backward()
        div = 1.0
    else:
        with torch.autocast("cuda", dtype=torch.float16):
            o = model(x)
        loss = rec_loss(o, y, m)
        (loss * (scale if mode == "fp16_scaled" else 1.0)).backward()
        div = scale if mode == "fp16_scaled" else 1.0
    G = groups(model)
    res = {k: gn(v) / div for k, v in G.items()}
    res["_finite"] = all(np.isfinite(v) for v in res.values())
    model.zero_grad(set_to_none=True)
    return res


def positive_control():
    """Same probe function on the S6a setup of v2_sanity.py (trained v1 weights loaded into the v2 config as a
    mechanism probe; first UIEB validation image, top-left 256x256 crop; mask None). Expected B_inf-head grad ~2.98e-2."""
    from common import val_rows, load_u8, u8_to_tensor
    ck = torch.load(CKPTS["last"][0], map_location="cpu", weights_only=False)
    m = T.build_model("baseline", {**ck["cfg"]["model_cfg"], "phys_conditioning": "v2"})
    m.load_state_dict(ck["model"], strict=True); m.eval()
    r = [r for r in val_rows() if r["dataset"] == "UIEB"][0]
    x = u8_to_tensor(load_u8(r["input"])[:256, :256]); y = u8_to_tensor(load_u8(r["reference"])[:256, :256])
    out = {"image": r["unit"], "cpu_fp32": probe_grads(m, x, y, None, "fp32")}
    m = m.cuda(); xg, yg = x.cuda(), y.cuda()
    out["gpu_fp32"] = probe_grads(m, xg, yg, None, "fp32")
    out["gpu_fp16_scaled_65536"] = probe_grads(m, xg, yg, None, "fp16_scaled", 65536.0)
    out["gpu_fp16_unscaled"] = probe_grads(m, xg, yg, None, "fp16_unscaled")
    ref = out["cpu_fp32"]["gphys_binf"]
    out["ok_cpu_matches_S6a_2.98e-2"] = abs(ref - 2.98e-2) / 2.98e-2 < 0.01
    out["ok_gpu_fp32_vs_cpu_rel"] = abs(out["gpu_fp32"]["gphys_binf"] - ref) / ref
    out["ok_gpu_fp16_scaled_vs_cpu_rel"] = abs(out["gpu_fp16_scaled_65536"]["gphys_binf"] - ref) / ref
    del m
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phys-conditioning", choices=["v1", "v2"], default="v2")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    commit = git_state()                                   # aborts on uncommitted src/ changes (no bypass)
    sha0 = sha256_file(CKPTS["last"][0])
    tm = Timer()
    assert torch.cuda.is_available(), "GPU sanity check requires CUDA"
    pos = positive_control()                                # probe validation on the S6a setup (before seeding)
    print("[positive control]", json.dumps({k: (v if not isinstance(v, dict) else
                                                 {kk: vv for kk, vv in v.items() if kk.startswith("gphys") or kk == "_finite"})
                                             for k, v in pos.items()}), flush=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True                  # as train.py
    torch.cuda.set_per_process_memory_fraction(0.85, 0)    # as train.py --vram-frac default
    dev = torch.device("cuda")
    split = ROOT / "data/splits/lsui_uieb_all_d6_seed0"
    rows = T.read_manifest(split / "train_manifest.txt")
    loader = DataLoader(T.PairedCropDataset(rows, 256), batch_size=4, shuffle=True, num_workers=2,
                        pin_memory=True, drop_last=True, persistent_workers=True)
    mcfg = {"phys_conditioning": "v2"} if args.phys_conditioning == "v2" else None
    model = T.build_model("baseline", mcfg).to(dev).train()
    crit = SSPGNLoss(SSPGNLossWeights())
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.9), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    G = groups(model)
    it = iter(loader)
    probe_x, probe_y, probe_m = (t.to(dev) for t in next(it))
    torch.cuda.reset_peak_memory_stats()
    out_dir = new_outdir(f"V2GRAD_{args.phys_conditioning}")
    log = open(out_dir / "steps.jsonl", "w", encoding="utf-8")
    first = {}
    for step in range(args.steps):
        rec = {"step": step}
        if step in PROBES:                                 # restoration-loss-only gradient probe (no optimizer)
            rec["probe_rec_grad"] = probe_grads(model, probe_x, probe_y, probe_m, "fp32")          # primary
            rec["probe_rec_grad_fp16_scaled"] = probe_grads(model, probe_x, probe_y, probe_m, "fp16_scaled",
                                                            scaler.get_scale())
            rec["probe_rec_grad_fp16_unscaled"] = probe_grads(model, probe_x, probe_y, probe_m, "fp16_unscaled")
            opt.zero_grad(set_to_none=True)
        lr = T.cosine_lr(step, 100_000, 5e-4, 1000, 1e-6)
        for gpar in opt.param_groups:
            gpar["lr"] = lr
        opt.zero_grad(set_to_none=True)
        finite = True; tot = 0.0
        for _ in range(2):                                 # accum = 2, as train.py
            x, y, m = (t.to(dev, non_blocking=True) for t in next(it))
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x)
            loss, _ = crit(out, x, y, mask=m)
            finite &= bool(torch.isfinite(loss)); tot += float(loss) / 2
            scaler.scale(loss / 2).backward()
        scale_before = scaler.get_scale()
        scaler.unscale_(opt)
        g = {k: gn(v) for k, v in G.items()}
        total_gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        scaler.step(opt); scaler.update()
        rec.update(lr=lr, loss=tot, loss_finite=finite, grad=g, total_grad_norm=total_gn,
                   grads_finite=bool(np.isfinite(total_gn)), scale_before=scale_before, scale_after=scaler.get_scale(),
                   step_skipped=not np.isfinite(total_gn), out_scale_absmax=float(model.bottleneck[0].out_scale.abs().max()))
        for k, v in g.items():
            if v > 0 and np.isfinite(v) and k not in first:
                first[k] = step
        if "probe_rec_grad" in rec:
            for k, v in rec["probe_rec_grad"].items():
                if not k.startswith("_") and v > 0 and np.isfinite(v) and f"rec_{k}" not in first:
                    first[f"rec_{k}"] = step
        log.write(json.dumps(rec) + "\n"); log.flush()
    log.close()
    L = [json.loads(l) for l in open(out_dir / "steps.jsonl")]
    summary = {"phys_conditioning": args.phys_conditioning, "steps": args.steps, "commit": commit,
               "first_nonzero_step_training_loss": {k: v for k, v in first.items() if not k.startswith("rec_")},
               "first_nonzero_step_restoration_probe": {k[4:]: v for k, v in first.items() if k.startswith("rec_")},
               "all_losses_finite": all(r["loss_finite"] for r in L),
               "skipped_steps_inf_grad": [r["step"] for r in L if r["step_skipped"]],
               "scale_first_last": [L[0]["scale_before"], L[-1]["scale_after"]],
               "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
               "out_scale_absmax_last": L[-1]["out_scale_absmax"],
               "probe_rec_grad_last": [r for r in L if "probe_rec_grad" in r][-1]["probe_rec_grad"],
               "probe_method": "primary fp32 (autocast off); also fp16 GradScaler-scaled then unscaled; "
                               "fp16 unscaled kept only to document underflow",
               "all_probes_finite": all(r[k]["_finite"] for r in L for k in
                                        ("probe_rec_grad", "probe_rec_grad_fp16_scaled", "probe_rec_grad_fp16_unscaled")
                                        if k in r),
               "positive_control": pos,
               "ckpt_sha_unchanged": sha256_file(CKPTS["last"][0]) == sha0, "no_checkpoint_written": True}
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "meta.json", {"commit": commit, "env": environment(), **tm.done(), "seed": args.seed,
                                       "protocol": "train.py: AdamW(0.9,0.9) lr 5e-4 cosine warmup 1000, batch 4 x accum 2, "
                                                   "crop 256, fp16 autocast + GradScaler, clip 1.0, vram_frac 0.85"})
    print(json.dumps(summary, indent=1))
    print("->", out_dir)


if __name__ == "__main__":
    main()

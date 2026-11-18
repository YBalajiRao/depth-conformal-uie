"""
Unified training recipe for external baselines (roadmap phase D; PI decision 2026-11-13: one common recipe).

Every baseline architecture (registry.BASELINES, random init, no pretrained weights) is trained with exactly the
SS-PGN schedule and data pipeline; only the objective differs from SS-PGN (no physics / NIG terms, since baselines
have no such outputs):
    data       : train.PairedCropDataset on the fixed split (4,135 train), 256 crops, rot90 + flip, mask
    objective  : SSPGNLoss.recon = L1 + 0.2 * (1 - SSIM) + 0.1 * wavelet-high-band L1  (the SS-PGN reconstruction terms)
    optimiser  : AdamW(betas 0.9, 0.9, wd 0), lr 5e-4, 1k linear warmup, cosine to 1e-6, 100k steps,
                 batch 4 x accum 2, grad clip 1.0, fp16 autocast + GradScaler (loss in fp32), seed 0
    selection  : last.pt at the final step only
    monitoring : fixed 100-image validation subset (same selection rule and val_id as SS-PGN seed 0), native
                 resolution via registry.predict
Official losses (GAN, perceptual, colour-space) and schedules are NOT used: this is an architecture comparison under
an identical recipe. Pre-registered fallback: if a baseline's 500-step pilot diverges at lr 5e-4, lr 2e-4 is used for
that baseline and reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import registry as R  # noqa: E402
import data as D  # noqa: E402  (puts src/* on sys.path)
import train as T  # noqa: E402
from losses_sspgn import SSPGNLoss  # noqa: E402


def val_subset(rows, n, seed):
    by = {}
    for r in rows:
        by.setdefault(r[0], []).append(r)
    rng = random.Random(seed)
    sub = []
    for d in sorted(by):
        k = max(1, round(n * len(by[d]) / len(rows)))
        sub += rng.sample(sorted(by[d], key=lambda r: r[1]), min(k, len(by[d])))
    return sorted(sub, key=lambda r: r[1])[:n]


@torch.no_grad()
def validate(name, model, rows, device):
    model.eval()
    ps, per, skipped = [], {}, 0
    for ds, uid, pin, pref in rows:
        x8, y8 = T.load_rgb_u8(pin), T.load_rgb_u8(pref)
        x = T.to_tensor(x8)[None].to(device)
        try:
            J = R.predict(name, model, x).float()
        except torch.cuda.OutOfMemoryError:
            skipped += 1; torch.cuda.empty_cache(); continue
        p = D.psnr_u8(D.to_u8(J), y8)
        ps.append(p); per.setdefault(ds, []).append(p)
    model.train(); torch.cuda.empty_cache()
    out = {"psnr": float(np.mean(ps)) if ps else float("nan"), "n": len(ps), "skipped_oom": skipped}
    out.update({f"psnr_{k}": float(np.mean(v)) for k, v in per.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=sorted(R.BASELINES))
    ap.add_argument("--iters", type=int, default=100_000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr-min", type=float, default=1e-6)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-subset", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--vram-frac", type=float, default=0.85)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    dev = torch.device("cuda")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    if a.vram_frac > 0:
        torch.cuda.set_per_process_memory_fraction(a.vram_frac, 0)
    amp = not a.no_amp
    spec = R.BASELINES[a.name]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "baselines/adapters"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    if dirty:
        raise SystemExit("[abort] src/ or baselines/adapters has uncommitted changes")
    repo_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=R.REPOS / spec.repo, capture_output=True,
                               text=True).stdout.strip()

    train_rows, val_rows = D.rows("train"), D.rows("val")
    vrows = val_subset(val_rows, a.val_subset, a.seed)
    val_id = hashlib.sha1(chr(10).join(sorted(r[1] for r in vrows)).encode()).hexdigest()[:16]
    model = R.build(a.name).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    crit = SSPGNLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.9), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    run_dir = Path(a.out or ROOT / "runs/baselines" / a.name) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg = {"args": vars(a), "baseline": {**spec.__dict__, "kwargs": dict(spec.kwargs)}, "repo_head": repo_head,
           "git": {"head": head, "dirty": dirty}, "n_params": n_params, "val_id": val_id, "n_train": len(train_rows),
           "objective": "L1 + 0.2*(1-SSIM) + 0.1*wavelet (SSPGNLoss.recon)", "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0), "started": datetime.now().isoformat(timespec="seconds"),
           "numerics_patch": R.NUMERICS_PATCH.get(a.name), "numerics_patched_modules": getattr(model, "_numerics_patched", 0),
           "nonfinite_policy": "forward non-finite loss: skip+count, GradScaler untouched; scale floor 1e-4"}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    print(f"[baseline] {a.name} params={n_params:,} val_id={val_id} repo@{repo_head[:8]} commit {head[:8]} -> {run_dir}", flush=True)

    loader = torch.utils.data.DataLoader(T.PairedCropDataset(train_rows, a.crop), batch_size=a.batch, shuffle=True,
                                         num_workers=a.workers, pin_memory=True, drop_last=True,
                                         persistent_workers=a.workers > 0)
    tlog = open(run_dir / "train_log.jsonl", "a", encoding="utf-8")
    vlog = open(run_dir / "val_log.jsonl", "a", encoding="utf-8")
    step, bad, t0, it = 0, 0, time.time(), iter(loader)
    nonfinite_fwd, scale_floor_hits, SCALE_FLOOR = 0, 0, 1e-4
    model.train()
    while step < a.iters:
        lr = T.cosine_lr(step, a.iters, a.lr, a.warmup, a.lr_min)
        for gp in opt.param_groups:
            gp["lr"] = lr
        opt.zero_grad(set_to_none=True)
        acc, finite = {}, True
        for _ in range(a.accum):
            try:
                x, y, m = next(it)
            except StopIteration:
                it = iter(loader); x, y, m = next(it)
            x, y, m = (t.to(dev, non_blocking=True) for t in (x, y, m))
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                J = R.raw_forward(a.name, model, x)
            loss, logs = crit.recon(J.float(), y, m)
            if not torch.isfinite(loss):
                finite = False
                del J, loss, logs          # free the skipped step's autograd graph (OOM fix, 2026-11-13 21:17 crash)
                break
            scaler.scale(loss / a.accum).backward()
            for k, v in logs.items():
                acc[k] = acc.get(k, 0.0) + float(v) / a.accum
        if not finite:
            bad += 1
            opt.zero_grad(set_to_none=True)
            nonfinite_fwd += 1                      # forward-loss overflow: skip + count; do NOT touch the GradScaler
            torch.cuda.empty_cache()
            print(f"[warn] non-finite loss at step {step} (streak {bad}); step skipped", flush=True)
            if bad >= 20:
                raise RuntimeError("20 consecutive non-finite losses; aborting")
            continue
        bad = 0
        scaler.unscale_(opt)
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip))
        scaler.step(opt); scaler.update()           # GradScaler halves only on backward-gradient overflow
        if amp and scaler.get_scale() < SCALE_FLOOR:
            scaler.update(SCALE_FLOOR); scale_floor_hits += 1
        step += 1
        rec = {"step": step, "lr": lr, "grad_norm": gn, "scale": scaler.get_scale() if amp else 1.0,
               "nonfinite_fwd_total": nonfinite_fwd, "scale_floor_hits": scale_floor_hits,
               "sec": round(time.time() - t0, 2), "vram_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), **acc}
        tlog.write(json.dumps(rec) + "\n")
        if step % 50 == 0 or step == 1:
            tlog.flush()
            print(f"[{step:>6}/{a.iters}] loss={acc['total']:.4f} l1={acc['l1']:.4f} gn={gn:.2f} lr={lr:.2e} "
                  f"vram={rec['vram_gib']}G", flush=True)
        if step % a.val_every == 0 or step == a.iters:
            v = validate(a.name, model, vrows, dev)
            v["step"] = step
            vlog.write(json.dumps(v) + "\n"); vlog.flush()
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                        "step": step, "cfg": cfg, "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                                                          "torch": torch.get_rng_state(),
                                                          "cuda": torch.cuda.get_rng_state()}}, run_dir / "last.pt")
            print(f"[val {step}] PSNR={v['psnr']:.3f} n={v['n']} oom={v['skipped_oom']} "
                  + " ".join(f"{k}={v[k]:.3f}" for k in v if k.startswith("psnr_")), flush=True)
    tlog.close(); vlog.close()
    print(f"[done] {a.name} wall={time.time() - t0:.0f}s run dir: {run_dir}", flush=True)


if __name__ == "__main__":
    main()

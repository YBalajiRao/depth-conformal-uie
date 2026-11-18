"""
Train the post-hoc uncertainty head on a FROZEN, FINAL SS-PGN checkpoint.

Data roles (split: data/splits/lsui_uieb_all_d6_seed0)
  train_manifest  -> head training (95% of duplicate components) + head holdout (5%, by component,
                     deterministic hash). The holdout is used only for logging; there is no
                     selection or early stopping, and the final head = last step.
  val_manifest    -> NOT used here. Reserved for conformal calibration
                     (calibrate_uncertainty.py, all 517) and the fixed 40-image monitor.
                     The monitor's 40 images are a subset of val, so they are never seen.
  test_manifest   -> NOT used here (evaluate_metrics.py --final only).
Reference images are used only as the loss target e = Y - J_hat. The head's inputs
(decoder features, I, J_hat, |J_hat - I|) never include Y.

KNOWN BIAS (logged, not hidden): the restorer was trained on these same training images, so
training residuals are likely smaller than on unseen images. Conformal calibration on val
corrects a global scale error (q_hat), not a ranking error. The script logs mean |e| on the
train holdout, and evaluation reports val residuals, so the size of the gap can be measured.

Loss: L = mean over valid pixels/channels of [ 0.5 log(2 pi) + log sigma + e^2 / (2 sigma^2) ],
e = Y - J_hat (signed). Padding (images < 256 px) is masked, as in main training.

Outputs: runs/posthoc/<run-name>/{config.json, train_log.jsonl, head_last.pt, model_last.pt}
  run-name defaults to "pilot_run_check" for the 500-step feasibility pilot, else a timestamp.
  An existing non-empty run directory is never overwritten (the script aborts).
Abort conditions (checked BEFORE the model is built, any GPU memory is touched, or anything is written):
  missing --expect-sha (unless --allow-nonfinal for tests), SHA-256 mismatch, non-final checkpoint,
  existing run directory. Each prints "ABORT: <reason>" to stderr and exits with code 1 (SystemExit).
  model_last.pt is in the main checkpoint format (arch 'posthoc', step == iters), so the unchanged
  calibrate_uncertainty.py / evaluate_metrics.py / monitor_reliability.py can load it.

  python src/training/train_post_hoc.py --ckpt runs/sspgn/20261106_173805/last.pt --steps 500   # feasibility pilot
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
for sub in ("src/models", "src/training", "src/utils"):
    sys.path.insert(0, str(ROOT / sub))
from post_hoc_head import PostHocConfig, SSPGNWithPostHoc, gaussian_nll  # noqa: E402
from train import PairedCropDataset, read_manifest  # noqa: E402
from pad_crop_wrapper import PadCropWrapper  # noqa: E402


def sha256(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_frozen(ckpt: Path, expect_sha: str | None, require_final: bool):
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    got = sha256(ckpt)
    if expect_sha and got.lower() != expect_sha.lower():
        raise RuntimeError(f"checkpoint SHA-256 mismatch: expected {expect_sha[:16]}..., got {got[:16]}...")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    arch = ck.get("cfg", {}).get("args", {}).get("arch", "baseline")
    if arch != "baseline":
        raise RuntimeError(f"post-hoc head expects a baseline SS-PGN checkpoint, got arch {arch!r}")
    step, total = ck.get("step"), ck.get("cfg", {}).get("args", {}).get("iters")
    if require_final and (total is None or step != total):
        raise RuntimeError(f"{ckpt.name} is at step {step} of {total}; the head must be fitted on the FINAL "
                           "restorer (pass --allow-nonfinal only for tests)")
    cfg = PostHocConfig(restorer_cfg=dict(ck["cfg"]["model_cfg"]))
    model = SSPGNWithPostHoc(cfg)
    model.restorer.load_state_dict(ck["model"], strict=True)
    model.freeze_restorer()
    return model, {"ckpt": str(ckpt), "sha256": got, "step": step, "total": total}


def split_rows_by_component(manifest: Path, holdout_frac: float, seed: int):
    """Return (train_rows, holdout_rows) as train.py tuples (ds, uid, inp, ref); split by component."""
    comp = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        ds, uid, cid, _, _ = line.split("\t")
        comp[uid] = cid
    rows = read_manifest(manifest)
    h = lambda c: int(hashlib.sha1(f"posthoc:{seed}:{c}".encode()).hexdigest(), 16) / 16 ** 40
    tr = [r for r in rows if h(comp[r[1]]) >= holdout_frac]
    ho = [r for r in rows if h(comp[r[1]]) < holdout_frac]
    assert not ({comp[r[1]] for r in tr} & {comp[r[1]] for r in ho})
    return tr, ho


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--expect-sha", default=None, help="required SHA-256 of --ckpt (recommended)")
    ap.add_argument("--allow-nonfinal", action="store_true", help="tests only")
    ap.add_argument("--split-dir", default=str(ROOT / "data/splits/lsui_uieb_all_d6_seed0"))
    ap.add_argument("--holdout-frac", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(ROOT / "runs/posthoc"))
    ap.add_argument("--run-name", default=None,
                    help='default: "pilot_run_check" when --steps == 500, otherwise a timestamp')
    args = ap.parse_args(argv)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    split = Path(args.split_dir)
    for forbidden in ("val", "test"):
        assert forbidden not in "train_manifest.txt"
    if args.steps <= 0:
        raise SystemExit("ABORT: --steps must be positive")
    if not args.expect_sha and not args.allow_nonfinal:
        raise SystemExit("ABORT: --expect-sha <SHA-256 of the final last.pt> is required")
    name = args.run_name or ("pilot_run_check" if args.steps == 500 else datetime.now().strftime("%Y%m%d_%H%M%S"))
    run = Path(args.out) / name
    if run.exists() and any(run.iterdir()):
        raise SystemExit(f"ABORT: run directory {run} already exists and is not empty (will not overwrite); "
                         "pass a different --run-name")
    tr_rows, ho_rows = split_rows_by_component(split / "train_manifest.txt", args.holdout_frac, args.seed)

    try:
        model, ck_info = load_frozen(Path(args.ckpt), args.expect_sha, not args.allow_nonfinal)
    except (RuntimeError, FileNotFoundError) as e:
        raise SystemExit(f"ABORT: {e}")
    model.to(device)
    model.freeze_restorer()
    opt = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=args.wd)
    run.mkdir(parents=True, exist_ok=True)
    cfg = {"argv": sys.argv, "args": vars(args), "restorer": ck_info,
           "split_dir": str(split), "train_manifest_sha256": sha256(split / "train_manifest.txt"),
           "n_head_train": len(tr_rows), "n_head_holdout": len(ho_rows),
           "head_params": model.n_head_params, "torch": torch.__version__,
           "optimizer": {"type": "AdamW", "lr": args.lr, "weight_decay": args.wd, "schedule": "cosine to 0"},
           "loss": "gaussian NLL of signed residual e = Y - J_hat, masked", "started": datetime.now().isoformat()}
    (run / "config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    print(f"[posthoc] restorer step {ck_info['step']} sha {ck_info['sha256'][:12]} | head params {model.n_head_params:,} | "
          f"train {len(tr_rows)} / holdout {len(ho_rows)} images | run {run}", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(PairedCropDataset(tr_rows, args.crop), batch_size=args.batch, shuffle=True, generator=g,
                        num_workers=args.workers, drop_last=True, pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0)
    ho_loader = DataLoader(PairedCropDataset(ho_rows, args.crop, augment=False), batch_size=args.batch,
                           shuffle=False, num_workers=0, drop_last=False)
    wrapped = PadCropWrapper(model, model.cfg.multiple, mode="reflect")
    amp = device.type == "cuda"
    log = open(run / "train_log.jsonl", "a", encoding="utf-8")
    it, step, t0 = iter(loader), 0, time.time()
    model.train()
    while step < args.steps:
        try:
            x, y, m = next(it)
        except StopIteration:
            it = iter(loader); x, y, m = next(it)
        x, y, m = x.to(device), y.to(device), m.to(device)
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * step / args.steps))
        for gp in opt.param_groups:
            gp["lr"] = lr
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = wrapped(x)
        e = y.float() - out["J"]                         # signed residual; Y used only as target
        loss = gaussian_nll(e, out["sigma"], m)
        assert step < args.steps
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.head.parameters(), args.clip).item()
        opt.step()
        step += 1
        rec = {"step": step, "lr": lr, "nll": loss.item(), "grad_norm": gn, "sec": round(time.time() - t0, 2),
               "sigma_mean": out["sigma"].mean().item(), "abs_e_mean": e.abs().mean().item()}
        log.write(json.dumps(rec) + "\n")
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            log.flush()
            print(f"[{step:>5}/{args.steps}] nll={rec['nll']:.4f} sigma={rec['sigma_mean']:.4f} |e|={rec['abs_e_mean']:.4f} "
                  f"gn={gn:.2f} lr={lr:.2e}", flush=True)

    # holdout NLL (logging only; no selection)
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, y, m in ho_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = wrapped(x)
            tot += gaussian_nll(y.float() - out["J"], out["sigma"], m).item() * x.shape[0]; n += x.shape[0]
    cfg["holdout_nll"] = tot / max(n, 1)
    cfg["finished"] = datetime.now().isoformat()
    (run / "config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    torch.save({"head": model.head.state_dict(), "cfg": cfg}, run / "head_last.pt")
    torch.save({"model": model.state_dict(), "step": args.steps,
                "cfg": {"args": {"arch": "posthoc", "iters": args.steps}, "model_cfg": model.cfg.__dict__,
                        "posthoc": cfg}}, run / "model_last.pt")
    print(f"[posthoc] done. holdout NLL={cfg['holdout_nll']:.4f}  -> {run / 'model_last.pt'}", flush=True)
    return run


if __name__ == "__main__":
    main()

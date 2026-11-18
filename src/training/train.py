"""
SS-PGN training (RTX 3070 8 GB profile).

Data
  Manifests from src/data/build_splits.py (default data/splits/lsui_uieb_all_d6_seed0/).
  The test manifest is never opened here (guarded).
  Train: random 256x256 crop, NO resizing. 35% of training images have a side < 256
         (LSUI 240x320, UIEB down to 159 px): these are reflect-padded to 256, and a
         validity mask (1 = real pixel) is passed to the loss, so padding never contributes.
         Same flips/rot90 applied to input, GT and mask.
  Val:   full images, batch 1, through PadCropWrapper (bottom/right reflect pad to the
         model multiple, crop back). Metrics are computed on original pixels only.

Optimisation
  AdamW(betas=0.9,0.9), cosine LR, mixed precision (torch.amp autocast fp16 +
  GradScaler; physics/NIG/loss in fp32), grad clipping after unscale, non-finite guard.

Checkpoints / logs  (runs/sspgn/<timestamp>/)
  config.json, train_log.jsonl (every iteration), val_log.jsonl,
  last.pt (every validation), best.pt (highest val PSNR), plus an EMA of the train loss.

Usage
  python src/training/train.py                       # full run (defaults below)
  python src/training/train.py --iters 40 --val-every 20 --val-max 8 --workers 0   # smoke test
  python src/training/train.py --resume runs/sspgn/<run>/last.pt
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

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
for sub in ("src/models", "src/training", "src/utils"):
    sys.path.insert(0, str(ROOT / sub))
from ss_pgn import SSPGN, SSPGNConfig            # noqa: E402
from ss_pgn_cascade import SSPGNCascade, SSPGNCascadeConfig  # noqa: E402

ARCHS = {"baseline": (SSPGN, SSPGNConfig), "cascade": (SSPGNCascade, SSPGNCascadeConfig)}


def build_model(arch: str, model_cfg: dict | None = None, compose: str | None = None):
    """Construct a model by name; model_cfg (e.g. from a checkpoint) overrides defaults."""
    if arch == "posthoc":                       # load-only (frozen SS-PGN + post-hoc head); not trainable via --arch
        from post_hoc_head import PostHocConfig, SSPGNWithPostHoc
        return SSPGNWithPostHoc(PostHocConfig(**dict(model_cfg or {})))
    if arch not in ARCHS:
        raise ValueError(f"unknown --arch {arch!r}; choices: {sorted(ARCHS)}")
    cls, cfg_cls = ARCHS[arch]
    kw = dict(model_cfg or {})
    if compose is not None:
        kw["compose"] = compose
    return cls(cfg_cls(**kw))
from losses_sspgn import SSPGNLoss, SSPGNLossWeights, StructStyleLoss, ssim  # noqa: E402
from pad_crop_wrapper import PadCropWrapper      # noqa: E402


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def read_manifest(path: Path):
    if "test" in path.name.lower():
        raise RuntimeError(f"refusing to read a test manifest during training: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{path} (run src/data/build_splits.py)")
    rows = []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 5:
            raise ValueError(f"{path.name}:{ln}: expected 5 tab-separated fields, got {len(parts)}")
        ds, uid, _cid, inp, ref = parts
        inp, ref = (str(Path(p) if Path(p).is_absolute() else Path(__file__).resolve().parents[2] / p) for p in (inp, ref))
        for p in (inp, ref):
            if not Path(p).is_file():
                raise FileNotFoundError(f"{path.name}:{ln}: missing file {p}")
        rows.append((ds, uid, inp, ref))
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def load_rgb_u8(p) -> np.ndarray:
    with Image.open(p) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)  # writable copy


def to_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float().div_(255.0)


class DepthStore:
    """Precomputed depth maps (src/data/precompute_depth.py) looked up by unit id from index_*.jsonl.
    Construction fails if any row has no map, or (check_sha) if an input image's SHA-256 differs from the one
    the map was computed from. load() returns a (1,H,W) float32 tensor and checks H, W against the image."""

    def __init__(self, root: Path, rows, check_sha: bool = True):
        self.root, self.idx = Path(root), {}
        for f in sorted(self.root.glob("index_*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.idx[r["unit"]] = r
        missing = [r[1] for r in rows if r[1] not in self.idx]
        if missing:
            raise FileNotFoundError(f"{len(missing)} units have no depth map under {self.root} (e.g. {missing[:3]})")
        if check_sha:
            bad = [r[1] for r in rows if hashlib.sha256(Path(r[2]).read_bytes()).hexdigest()
                   != self.idx[r[1]]["input_sha256"]]
            if bad:
                raise RuntimeError(f"{len(bad)} inputs changed since depth precomputation (e.g. {bad[:3]})")

    def load(self, uid, hw) -> torch.Tensor:
        r = self.idx[uid]
        with np.load(self.root / r["depth"]) as f:
            a = f["z"].astype(np.float32)
        if a.shape != tuple(hw):
            raise ValueError(f"{uid}: depth {a.shape} != image {tuple(hw)}")
        return torch.from_numpy(a)[None]


class SceneStore:
    """Precomputed global scene descriptors (src/data/precompute_scene_features.py), scene_<split>.npz.
    Construction fails if a row has no descriptor or (check_sha) its input SHA-256 differs from the one used."""

    def __init__(self, root: Path, rows, check_sha: bool = True, pca_file: str | None = None):
        """pca_file: optional training-only PCA (src/data/fit_scene_pca.py); descriptors are then returned as
        whitened k-d projections ((g - mean) @ components.T) / scale. The PCA is never refitted here."""
        self.root, self.g = Path(root), {}
        sha = {}
        for f in sorted(self.root.glob("scene_*.npz")):
            with np.load(f, allow_pickle=False) as d:
                for u, v, s in zip(d["unit"], d["g"], d["input_sha256"]):
                    self.g[str(u)] = torch.from_numpy(np.asarray(v, dtype=np.float32))
                    sha[str(u)] = str(s)
        missing = [r[1] for r in rows if r[1] not in self.g]
        if missing:
            raise FileNotFoundError(f"{len(missing)} units have no scene descriptor under {self.root} (e.g. {missing[:3]})")
        if check_sha:
            bad = [r[1] for r in rows if hashlib.sha256(Path(r[2]).read_bytes()).hexdigest() != sha[r[1]]]
            if bad:
                raise RuntimeError(f"{len(bad)} inputs changed since scene precomputation (e.g. {bad[:3]})")
        self.pca_file = pca_file
        if pca_file:
            with np.load(pca_file, allow_pickle=False) as d:
                mean = torch.from_numpy(d["mean"].astype(np.float32))
                comps = torch.from_numpy(d["components"].astype(np.float32))
                scale = torch.from_numpy(d["scale"].astype(np.float32))
                train_sha = str(d["train_npz_sha256"])
            if hashlib.sha256((self.root / "scene_train.npz").read_bytes()).hexdigest() != train_sha:
                raise RuntimeError("PCA was fitted on a different scene_train.npz")
            self.g = {u: ((v - mean) @ comps.T) / scale for u, v in self.g.items()}

    def load(self, uid) -> torch.Tensor:
        return self.g[uid].clone()


class PairedCropDataset(Dataset):
    """Random crop (no resize). Images smaller than the crop are reflect-padded + masked.
    With depth (DepthStore) the depth map gets exactly the same pad, crop, rot90 and flip; returns (x, y, m, z)."""

    def __init__(self, rows, crop=256, augment=True, depth=None, scene=None):
        self.rows, self.crop, self.augment, self.depth, self.scene = rows, crop, augment, depth, scene
        if scene is not None and depth is None:
            raise ValueError("scene descriptors are only supported together with depth")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ds, uid, pin, pref = self.rows[i]
        x, y = load_rgb_u8(pin), load_rgb_u8(pref)
        if x.shape != y.shape:
            raise ValueError(f"{uid}: input {x.shape} != reference {y.shape}")
        h, w = x.shape[:2]
        c = self.crop
        x, y = to_tensor(x), to_tensor(y)
        m = torch.ones(1, h, w)
        ts = [x, y] + ([self.depth.load(uid, (h, w))] if self.depth is not None else [])   # reflect-padded
        ph, pw = max(0, c - h), max(0, c - w)
        if ph or pw:
            if ph >= h or pw >= w:
                raise ValueError(f"{uid}: {h}x{w} too small to reflect-pad to {c}")
            ts = [F.pad(t[None], (0, pw, 0, ph), mode="reflect")[0] for t in ts]
            m = F.pad(m[None], (0, pw, 0, ph), value=0.0)[0]
        ts.append(m)
        H, W = ts[0].shape[-2:]
        top, left = random.randint(0, H - c), random.randint(0, W - c)
        sl = (slice(None), slice(top, top + c), slice(left, left + c))
        ts = [t[sl] for t in ts]
        if self.augment:
            k = random.randint(0, 3)
            flip = random.random() < 0.5
            ts = [torch.rot90(t, k, (1, 2)) for t in ts]
            if flip:
                ts = [torch.flip(t, (2,)) for t in ts]
        if self.depth is not None:
            x, y, z, m = ts
        else:
            x, y, m = ts
        assert x.shape == (3, c, c) and y.shape == (3, c, c) and m.shape == (1, c, c), (uid, x.shape, m.shape)
        if self.depth is not None:
            assert z.shape == (1, c, c), (uid, z.shape)
            if self.scene is not None:                    # whole-image descriptor: crop/augmentation independent
                return x, y, m, z, self.scene.load(uid)
            return x, y, m, z
        return x, y, m


class FullImageDataset(Dataset):
    def __init__(self, rows, depth=None, scene=None):
        self.rows, self.depth, self.scene = rows, depth, scene

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ds, uid, pin, pref = self.rows[i]
        x, y = load_rgb_u8(pin), load_rgb_u8(pref)
        if x.shape != y.shape:
            raise ValueError(f"{uid}: input {x.shape} != reference {y.shape}")
        if self.depth is not None:
            extra = (self.scene.load(uid),) if self.scene is not None else ()
            return (to_tensor(x), to_tensor(y), ds, uid, self.depth.load(uid, x.shape[:2])) + extra
        return to_tensor(x), to_tensor(y), ds, uid


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
@torch.no_grad()
def validate(model, rows, device, amp, max_n=0, depth=None, scene=None):
    # Full-resolution validation (up to 1450x2180) inflates the CUDA caching allocator and, with
    # cudnn.benchmark, adds per-shape workspaces. Under Windows WDDM the excess silently spills into
    # shared system memory and training slows ~28x (observed after step 4000 on 2026-11-06).
    # Mitigation: benchmark off during validation, release the cache afterwards.
    bench = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    model.eval()
    wrapped = PadCropWrapper(model, model.cfg.multiple, mode="reflect")
    crit = SSPGNLoss()
    ds = FullImageDataset(rows if max_n <= 0 else rows[:max_n], depth, scene)
    stats = {"psnr": [], "ssim": [], "nll": [], "corr_std_err": []}
    per_ds, skipped = {}, 0
    for i in range(len(ds)):
        x, y, dname, uid, *zz = ds[i]
        x, y = x[None].to(device), y[None].to(device)
        aux = [t[None].to(device) for t in zz]
        try:
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                out = wrapped(x, *aux)
        except torch.cuda.OutOfMemoryError:
            skipped += 1
            torch.cuda.empty_cache()
            continue
        assert out["J"].shape == y.shape, (uid, out["J"].shape, y.shape)
        J = out["J"].float().clamp(0, 1)
        mse = F.mse_loss(J, y).item()
        psnr = 10 * math.log10(1.0 / max(mse, 1e-12))
        s = ssim(J, y).item()
        from losses_sspgn import nig_nll
        mu_nig = out["nig"][:, 0:3].float()                   # NIG mean (== out["J"] unless the CSA is on)
        nll = nig_nll(y, mu_nig, out["nu"], out["alpha"], out["beta"]).mean().item()
        std = torch.sqrt(out["beta"] / (out["nu"] * (out["alpha"] - 1))).flatten()
        err = (mu_nig - y).abs().flatten()
        if "J_struct" in out:
            mse_s = F.mse_loss(out["J_struct"].float().clamp(0, 1), y).item()
            stats.setdefault("psnr_struct", []).append(10 * math.log10(1.0 / max(mse_s, 1e-12)))
        sc, ec = std - std.mean(), err - err.mean()
        corr = (sc * ec).sum() / (sc.norm() * ec.norm()).clamp_min(1e-12)
        for k, v in zip(("psnr", "ssim", "nll", "corr_std_err"), (psnr, s, nll, corr.item())):
            stats[k].append(v)
        per_ds.setdefault(dname, []).append(psnr)
    model.train()
    del x, y
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = bench
    res = {k: float(np.mean(v)) if v else float("nan") for k, v in stats.items()}
    res.update({f"psnr_{k}": float(np.mean(v)) for k, v in per_ds.items()})
    res.update(n=len(stats["psnr"]), skipped_oom=skipped)
    return res


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def other_gpu_python_pids():
    """PIDs of other python processes holding a CUDA context (via nvidia-smi).
    Free-memory checks are unreliable under Windows WDDM (shared-memory paging), so we look
    at processes instead. Returns [] if nvidia-smi is unavailable."""
    import os
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    pids = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and "python" in parts[1].lower() and int(parts[0]) != os.getpid():
            pids.append(int(parts[0]))
    if not pids:   # WDDM often omits process_name; fall back to every listed PID that is a python.exe
        try:
            tl = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
                                capture_output=True, text=True, timeout=15).stdout
            py = {int(l.split(",")[1].strip('"')) for l in tl.splitlines() if l.count(",") >= 2}
            gpu = {int(l.split(",")[0]) for l in out.splitlines() if l.split(",")[0].strip().isdigit()}
            pids = sorted((py & gpu) - {os.getpid()})
        except (OSError, subprocess.TimeoutExpired, ValueError):
            pass
    return pids


def git_provenance(run_dir: Path, seg: str = "") -> dict:
    """HEAD commit plus SHA-256 of any uncommitted diff (the diff itself is saved next to the config)."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        diff = subprocess.run(["git", "diff", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              encoding="utf-8").stdout
    except OSError as e:
        return {"error": f"{type(e).__name__}: {e}"}
    out = {"head": head, "dirty": bool(diff)}
    if diff:
        (run_dir / f"uncommitted{seg}.diff").write_text(diff, encoding="utf-8")
        out["diff_sha256"] = hashlib.sha256(diff.encode("utf-8")).hexdigest()
    return out


def cosine_lr(it, total, base, warmup, floor):
    if it < warmup:
        return base * (it + 1) / warmup
    p = (it - warmup) / max(1, total - warmup)
    return floor + 0.5 * (base - floor) * (1 + math.cos(math.pi * min(p, 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=sorted(ARCHS), default="baseline")
    ap.add_argument("--compose", choices=["direct", "phys_residual"], default="direct",
                    help="'phys_residual' only for the L_phys ablation (see ss_pgn.SSPGNConfig)")
    ap.add_argument("--split-dir", default=str(ROOT / "data/splits/lsui_uieb_all_d6_seed0"))
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4, help="bs=4 peaks ~4.6 GiB; bs=8 ~7.9 GiB (too close to 8 GB)")
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation steps (effective batch = batch*accum)")
    ap.add_argument("--iters", type=int, default=100_000, help="optimizer steps")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr-min", type=float, default=1e-6)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-max", type=int, default=0, help="0 = full val split")
    ap.add_argument("--val-subset", type=int, default=0,
                    help="validate during training on a FIXED dataset-stratified subset of this many val images "
                         "(seed --seed). 0 = all. Full-val numbers come from evaluate_metrics.py at the end.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    # Ablation switches (defaults reproduce the original SS-PGN training exactly)
    ap.add_argument("--lp", type=float, default=None, help="override physics-loss weight (ablation A3: 0)")
    ap.add_argument("--le", type=float, default=None, help="override evidential-loss weight (ablation A4: 0)")
    ap.add_argument("--no-field-conditioning", action="store_true",
                    help="ablation A2: SS-PGB gate/FiLM/high-band gain do not receive the fields (baseline arch only)")
    ap.add_argument("--phys-conditioning", choices=["v1", "v2"], default="v1",
                    help="v1 = original SS-PGN (default); v2 = physics-parameterized conditioning "
                         "(gphys on the SS-PGB LL band, Pi = [tD, tB, B_hat at H/16]); baseline arch only")
    # Redesign Phase 1 switches (default off)
    ap.add_argument("--scene-prior-pca", type=int, default=0,
                    help="P1e: with --dgt, scene prior from a k-d training-only PCA of the descriptors, fed ONLY to the "
                         "ordered-DGT pre-activations, with 50%% descriptor dropout and Gaussian noise (sigma 0.25, "
                         "whitened units) in training. 0 = off")
    ap.add_argument("--scene-prior", action="store_true",
                    help="with --dgt: zero-init global scene prior from precomputed DA-V2 encoder descriptors")
    ap.add_argument("--scene-dir", default=None,
                    help="precomputed scene descriptors (default data/scene/da2_small_5426e4f/<split-dir name>)")
    ap.add_argument("--dec-film", action="store_true",
                    help="with --dgt: zero-init decoder FiLM from [z~, t_D, t_B] at each decoder stage")
    ap.add_argument("--dgt-ordered", action="store_true",
                    help="with --dgt: enforce beta_R >= beta_G >= beta_B (sequential softplus chain) for beta_D and beta_B")
    ap.add_argument("--dgt", action="store_true",
                    help="Depth-Grounded Transmission: tD/tB = exp(-beta*z) from precomputed depth (needs --phys-conditioning v2)")
    ap.add_argument("--depth-dir", default=None,
                    help="precomputed depth root (default data/depth/da2_small_5426e4f/<split-dir name>)")
    ap.add_argument("--no-depth-sha-check", action="store_true", help="skip input SHA-256 check against the depth index")
    ap.add_argument("--csa", action="store_true", help="blind ChromaStyle-Adapter on detached J_struct")
    ap.add_argument("--struct-loss", choices=["full", "aligned"], default=None,
                    help="with --csa: L_struct on the colour-aligned J_struct (default 'aligned', PI decision "
                         "2026-11-09) or against the reference as-is ('full', ablation only)")
    ap.add_argument("--lf", type=float, default=1.0, help="with --csa: weight of L_total on J_final")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--out", default=str(ROOT / "runs/sspgn"))
    ap.add_argument("--pilot", action="store_true",
                    help="500-step trial on <split-dir>/pilot/*_pilot_manifest.txt (src/data/create_pilot_manifest.py); "
                         "ends with a PASS/FAIL check that L1 and the evidential NLL decrease")
    ap.add_argument("--force", action="store_true", help="pilot: run even if the GPU looks busy")
    ap.add_argument("--pilot-iters", type=int, default=500, help="pilot length in steps (default 500)")
    ap.add_argument("--vram-frac", type=float, default=0.85,
                    help="cap the CUDA allocator at this fraction of device memory, so it raises OOM instead of "
                         "silently spilling into shared system memory under Windows WDDM (0 = no cap)")
    ap.add_argument("--slow-factor", type=float, default=2.0,
                    help="warn when the mean step time over 200 steps exceeds this multiple of the first 200-step mean")
    args = ap.parse_args()
    if args.pilot:
        args.iters, args.accum, args.warmup, args.val_every, args.val_max = args.pilot_iters, 1, 50, 250, 0
        args.workers = min(args.workers, 2)
        if args.out == str(ROOT / "runs/sspgn"):
            args.out = str(ROOT / "runs/pilot")
        busy = other_gpu_python_pids()
        if busy and not args.force:
            raise RuntimeError(f"another Python process is using the GPU (PIDs {busy}); running the pilot "
                               "alongside it risks OOM for both. Stop it or pass --force.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    device = torch.device("cuda")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    if args.vram_frac > 0:
        torch.cuda.set_per_process_memory_fraction(args.vram_frac, 0)
        print(f"[vram] allocator capped at {args.vram_frac:.2f} x {torch.cuda.get_device_properties(0).total_memory / 2 ** 30:.2f} GiB")
    amp = not args.no_amp

    split = Path(args.split_dir)
    if args.pilot:
        train_rows = read_manifest(split / "pilot" / "train_pilot_manifest.txt")
        val_rows = read_manifest(split / "pilot" / "val_pilot_manifest.txt")
    else:
        train_rows, val_rows = read_manifest(split / "train_manifest.txt"), read_manifest(split / "val_manifest.txt")
    tr_ids, va_ids = {r[1] for r in train_rows}, {r[1] for r in val_rows}
    if tr_ids & va_ids:
        raise RuntimeError(f"{len(tr_ids & va_ids)} units in both train and val manifests")
    val_full_id = hashlib.sha1(chr(10).join(sorted(r[1] for r in val_rows)).encode()).hexdigest()[:16]
    if args.val_subset > 0 and args.val_subset < len(val_rows):
        by = {}
        for r in val_rows:
            by.setdefault(r[0], []).append(r)
        rng = random.Random(args.seed)
        sub = []
        for d in sorted(by):
            k = max(1, round(args.val_subset * len(by[d]) / len(val_rows)))
            sub += rng.sample(sorted(by[d], key=lambda r: r[1]), min(k, len(by[d])))
        val_rows = sorted(sub, key=lambda r: r[1])[:args.val_subset]
    val_id = hashlib.sha1(chr(10).join(sorted(r[1] for r in val_rows)).encode()).hexdigest()[:16]
    print(f"[data] train={len(train_rows)} val={len(val_rows)} (val_id {val_id}"
          f"{', subset of full val ' + val_full_id if val_id != val_full_id else ''}) from {split}")

    ck = None
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        ck_arch = ck.get("cfg", {}).get("args", {}).get("arch", "baseline")   # runs before --arch existed = baseline
        if ck_arch != args.arch:
            raise RuntimeError(f"--arch {args.arch} does not match checkpoint arch {ck_arch}")
        model = build_model(args.arch, ck["cfg"]["model_cfg"]).to(device)
    else:
        if (args.no_field_conditioning or args.phys_conditioning != "v1") and args.arch != "baseline":
            raise RuntimeError("--no-field-conditioning / --phys-conditioning v2 are implemented for --arch baseline only")
        mcfg = {}
        if args.no_field_conditioning:
            mcfg["field_conditioning"] = False
        if args.phys_conditioning != "v1":
            mcfg["phys_conditioning"] = args.phys_conditioning
        if (args.dgt or args.csa) and args.arch != "baseline":
            raise RuntimeError("--dgt / --csa are implemented for --arch baseline only")
        if args.dgt:
            mcfg["depth_transmission"] = True
            if args.dgt_ordered:
                mcfg["dgt_ordered"] = True
            if args.dec_film:
                mcfg["dec_film"] = True
            if args.scene_prior:
                mcfg["scene_prior"] = True
            if args.scene_prior_pca:
                if args.scene_prior:
                    raise RuntimeError("use either --scene-prior or --scene-prior-pca")
                sdir = Path(args.scene_dir) if args.scene_dir else ROOT / "data/scene/da2_small_5426e4f" / split.name
                pfile = sdir / f"pca{args.scene_prior_pca}_train.npz"
                if not pfile.is_file():
                    raise FileNotFoundError(f"{pfile} (run src/data/fit_scene_pca.py --k {args.scene_prior_pca})")
                mcfg.update(scene_prior=True, scene_dim=args.scene_prior_pca, scene_mode="dgt_only",
                            scene_dropout=0.5, scene_noise=0.25, scene_pca_file=str(pfile.relative_to(ROOT)))
        elif args.scene_prior or args.scene_prior_pca:
            raise RuntimeError("--scene-prior / --scene-prior-pca require --dgt")
        elif args.dec_film:
            raise RuntimeError("--dec-film requires --dgt")
        elif args.dgt_ordered:
            raise RuntimeError("--dgt-ordered requires --dgt")
        if args.csa:
            mcfg["csa"] = True
        model = build_model(args.arch, mcfg or None, compose=args.compose).to(device)
    lw_over = {k: v for k, v in (("lp", args.lp), ("le", args.le)) if v is not None}
    if ck is not None and (lw_over or args.no_field_conditioning or args.phys_conditioning != "v1"
                           or args.dgt or args.csa):
        raise RuntimeError("ablation switches cannot be combined with --resume")
    use_csa = bool(getattr(model.cfg, "csa", False))
    use_depth = bool(getattr(model.cfg, "depth_transmission", False))
    if use_csa:
        smode = (args.struct_loss or "aligned") if ck is None else ck["cfg"].get("struct_loss")
        crit = StructStyleLoss(SSPGNLossWeights(**lw_over), smode, args.lf if ck is None else ck["cfg"]["lf"])
    else:
        if args.struct_loss is not None:
            raise RuntimeError("--struct-loss only applies with --csa")
        crit = SSPGNLoss(SSPGNLossWeights(**lw_over))
    depth = None
    if use_depth:
        droot = Path(args.depth_dir) if args.depth_dir else ROOT / "data/depth/da2_small_5426e4f" / split.name
        depth = DepthStore(droot, train_rows + val_rows, check_sha=not args.no_depth_sha_check)
        print(f"[depth] {len(train_rows) + len(val_rows)} maps indexed under {droot}"
              f"{' (input SHA-256 verified)' if not args.no_depth_sha_check else ' (SHA check skipped)'}")
    scene = None
    if bool(getattr(model.cfg, "scene_prior", False)):
        sroot = Path(args.scene_dir) if args.scene_dir else ROOT / "data/scene/da2_small_5426e4f" / split.name
        pf = getattr(model.cfg, "scene_pca_file", "")
        scene = SceneStore(sroot, train_rows + val_rows, check_sha=not args.no_depth_sha_check,
                           pca_file=str(ROOT / pf) if pf else None)
        print(f"[scene] {len(train_rows) + len(val_rows)} descriptors indexed under {sroot}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.9), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    print(f"[model] arch={args.arch} compose={model.cfg.compose} params={model.n_params:,} multiple={model.cfg.multiple}")

    step, best_psnr, ema_loss = 0, -1.0, None
    seg = ""
    if ck is not None:
        model.load_state_dict(ck["model"], strict=True); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        step, best_psnr, ema_loss = ck["step"], ck["best_psnr"], ck.get("ema_loss")
        run_dir = Path(args.resume).parent
        if "rng" in ck:
            random.setstate(ck["rng"]["python"]); np.random.set_state(ck["rng"]["numpy"])
            torch.set_rng_state(ck["rng"]["torch"]); torch.cuda.set_rng_state(ck["rng"]["cuda"])
        else:
            print("[resume] checkpoint has no RNG state (older format): data order/augmentation will differ "
                  "from an uninterrupted run; model/optimizer/scaler/step are restored exactly")
        ck_val_id = ck.get("val_id", val_full_id)          # older checkpoints validated on the full split
        if ck_val_id != val_id:
            if (run_dir / "best.pt").exists():
                keep = run_dir / f"best_val{ck_val_id}_upto_step{step}.pt"
                (run_dir / "best.pt").rename(keep)
                print(f"[resume] validation set changed ({ck_val_id} -> {val_id}): previous best.pt kept as {keep.name}; "
                      "best tracking restarts (PSNRs on different val sets are not comparable)")
            best_psnr = -1.0
        n_seg = len(list(run_dir.glob("train_log*.jsonl")))
        seg = f".resume{n_seg}"
        print(f"[resume] step={step} best_psnr={best_psnr:.3f} from {args.resume}; logs -> *{seg}.jsonl")
    else:
        run_dir = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=False)
    cfg = {"args": vars(args), "model_cfg": model.cfg.__dict__, "loss_w": crit.w.__dict__,
           "struct_loss": getattr(crit, "struct_mode", None), "lf": getattr(crit, "lf", None),
           "depth_dir": str(depth.root) if depth is not None else None,
           "n_params": model.n_params, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "n_train": len(train_rows), "n_val": len(val_rows), "val_id": val_id, "val_full_id": val_full_id,
           "resumed_from": args.resume, "resume_step": step if args.resume else None,
           "started": datetime.now().isoformat(timespec="seconds"), "git": git_provenance(run_dir, seg)}
    (run_dir / f"config{seg}.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")

    loader = DataLoader(PairedCropDataset(train_rows, args.crop, depth=depth, scene=scene), batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True,
                        persistent_workers=args.workers > 0)
    tlog = open(run_dir / f"train_log{seg}.jsonl", "a", encoding="utf-8")
    vlog = open(run_dir / f"val_log{seg}.jsonl", "a", encoding="utf-8")

    def save(name):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "best_psnr": best_psnr, "ema_loss": ema_loss, "cfg": cfg, "val_id": val_id,
                    "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}}, run_dir / name)

    model.train()
    hist = []                                             # (l1, nig_nll) per step, for the pilot verdict
    step_times, ref_step_time = [], None                  # slowdown watchdog (excludes validation pauses only coarsely)
    it_loader, bad_streak, t0 = iter(loader), 0, time.time()
    while step < args.iters:
        lr = cosine_lr(step, args.iters, args.lr, args.warmup, args.lr_min)
        for gp in opt.param_groups:
            gp["lr"] = lr
        opt.zero_grad(set_to_none=True)
        logs_acc, finite = {}, True
        for _ in range(args.accum):
            try:
                batch = next(it_loader)
            except StopIteration:
                it_loader = iter(loader)
                batch = next(it_loader)
            x, y, m, *zz = (t.to(device, non_blocking=True) for t in batch)
            assert x.shape == (args.batch, 3, args.crop, args.crop) and m.shape == (args.batch, 1, args.crop, args.crop)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                out = model(x, *zz)
            out.pop("field_res", None)                    # cascade-only metadata
            loss, logs = crit(out, x, y, mask=m)          # fp32 (model returns fp32 physics/NIG)
            if not torch.isfinite(loss):
                finite = False
                break
            scaler.scale(loss / args.accum).backward()
            for k, v in logs.items():
                logs_acc[k] = logs_acc.get(k, 0.0) + v.item() / args.accum
        if not finite:
            bad_streak += 1
            opt.zero_grad(set_to_none=True)
            scaler.update(scaler.get_scale() / 2) if amp else None
            print(f"[warn] non-finite loss at step {step} (streak {bad_streak}); step skipped")
            if bad_streak >= 20:
                raise RuntimeError("20 consecutive non-finite losses; aborting")
            continue
        bad_streak = 0
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()
        scaler.step(opt)
        scaler.update()
        step += 1
        ema_loss = logs_acc["total"] if ema_loss is None else 0.99 * ema_loss + 0.01 * logs_acc["total"]
        rec = {"step": step, "lr": lr, "grad_norm": gnorm, "scale": scaler.get_scale() if amp else 1.0,
               "ema_total": ema_loss, "sec": round(time.time() - t0, 2),
               "vram_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), **logs_acc}
        tlog.write(json.dumps(rec) + "\n")
        hist.append((logs_acc["l1"], logs_acc["nig_nll"]))
        step_times.append(time.time())
        if len(step_times) >= 201:
            cur = (step_times[-1] - step_times[-201]) / 200
            if ref_step_time is None:
                ref_step_time = cur
            elif cur > args.slow_factor * ref_step_time and step % 200 == 0:
                print(f"[warn] step time {cur:.3f}s > {args.slow_factor:.1f} x reference {ref_step_time:.3f}s "
                      f"(GPU memory spill? reserved={torch.cuda.memory_reserved() / 2 ** 30:.2f} GiB)", flush=True)
        if step % 50 == 0 or step == 1:
            tlog.flush()
            print(f"[{step:>6}/{args.iters}] loss={logs_acc['total']:.4f} ema={ema_loss:.4f} l1={logs_acc['l1']:.4f} "
                  f"phys={logs_acc['phys']:.4f} nll={logs_acc['nig_nll']:.3f} tD={logs_acc['mean_tD']:.3f} "
                  f"gn={gnorm:.2f} lr={lr:.2e} vram={rec['vram_gib']}G")
        if step % args.val_every == 0 or step == args.iters:
            v = validate(model, val_rows, device, amp, args.val_max, depth, scene)
            step_times.clear()                            # do not count the validation pause as slowdown
            v["step"] = step
            vlog.write(json.dumps(v) + "\n"); vlog.flush()
            improved = v["psnr"] > best_psnr
            if improved:
                best_psnr = v["psnr"]
                save("best.pt")
            save("last.pt")
            print(f"[val {step}] PSNR={v['psnr']:.3f} SSIM={v['ssim']:.4f} NLL={v['nll']:.3f} "
                  f"corr(std,|err|)={v['corr_std_err']:.3f} n={v['n']} oom={v['skipped_oom']}"
                  + ("  * best" if improved else ""))
    tlog.close(); vlog.close()
    print(f"[done] best val PSNR={best_psnr:.3f}  run dir: {run_dir}  wall={time.time() - t0:.0f}s")
    if args.pilot:
        h = np.array(hist)
        w = min(50, len(h) // 4)
        first, last = h[:w].mean(0), h[-w:].mean(0)
        verdict = {}
        for j, name in enumerate(("l1", "nig_nll")):
            # require a drop of at least 10% of |first| (NLL can be negative, so use the absolute scale)
            ok = bool(last[j] < first[j] - 0.10 * abs(first[j]))
            verdict[name] = {"first_mean": float(first[j]), "last_mean": float(last[j]), "pass": ok}
            print(f"[pilot] {name:8s} first-{w} mean={first[j]:.4f}  last-{w} mean={last[j]:.4f}  "
                  f"-> {'PASS' if ok else 'FAIL'}")
        verdict["wall_seconds"] = round(time.time() - t0, 1)
        verdict["overall"] = all(v["pass"] for v in verdict.values() if isinstance(v, dict))
        (run_dir / "pilot_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
        print(f"[pilot] OVERALL: {'PASS' if verdict['overall'] else 'FAIL'}  ({verdict['wall_seconds']}s)")
        if not verdict["overall"]:
            sys.exit(1)


if __name__ == "__main__":
    main()

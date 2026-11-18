"""
Shared infrastructure for the post-training diagnostics (protocol:
reports/diagnostics/DIAGNOSTIC_PROTOCOL_2026-11-07.md). Inference only: no gradients, no training,
no checkpoint writes, no test split.

Re-uses the project's own definitions (imported, not re-implemented):
  calibrate_uncertainty: read_manifest, load_model (strict), predictive_std, sparsification, ause, spearman
  evaluate_metrics:      psnr_u8, ssim_u8, load_u8
  monitor_reliability:   fixed_subset, sobel_mag
  residual_gap:          stratified_sample
monitor_reliability / residual_gap set CUDA_VISIBLE_DEVICES at import time; the environment is restored
right after import (before CUDA initialises), so GPU inference remains possible.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("src/validation", "src/utils", "src/training", "src/models"):
    sys.path.insert(0, str(ROOT / sub))

_env = os.environ.get("CUDA_VISIBLE_DEVICES")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from calibrate_uncertainty import (read_manifest, load_model, predictive_std, sparsification,  # noqa: E402
                                   ause, spearman)
from evaluate_metrics import psnr_u8, ssim_u8, load_u8  # noqa: E402
from monitor_reliability import fixed_subset, sobel_mag  # noqa: E402
from residual_gap import stratified_sample  # noqa: E402
from pad_crop_wrapper import PadCropWrapper  # noqa: E402
if _env is None:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _env

RUN = ROOT / "runs/sspgn/20261106_173805"
CKPTS = {
    "last": (RUN / "last.pt", "be3c23cf157295ffcae738cb3468c79b09611ad04b6c0ee7bc1003895f81499a", 100000),
    "step4000": (RUN / "snapshots/step4000.pt",
                 "0831bf2566197ce16b33bbae68a555ab6df6646d709aca1e277ae019af4906ea", 4000),
}
SPLIT = ROOT / "data/splits/lsui_uieb_all_d6_seed0"
B_BOOT, BOOT_SEED = 2000, 20261107
__all__ = ["read_manifest", "predictive_std", "sparsification", "ause", "spearman", "psnr_u8", "ssim_u8",
           "load_u8", "fixed_subset", "sobel_mag", "stratified_sample"]


# ----------------------------------------------------------------------------- provenance
def git_state():
    """Commit hash; abort if any tracked/untracked change exists under src/ (exact source version)."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"[abort] src/ has uncommitted changes; commit before running diagnostics:\n{dirty}")
    return head


def sha256_file(path: Path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def verify_ckpt(key: str):
    path, expected, step_expected = CKPTS[key]
    got = sha256_file(path)
    if got != expected:
        raise SystemExit(f"[abort] SHA-256 mismatch for {path}: {got} != expected {expected}")
    print(f"[sha] {path.name}: {got} == expected", flush=True)
    return path, got, step_expected


def environment():
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": platform.platform(), "torch_threads": torch.get_num_threads()}


def new_outdir(diag_id: str) -> Path:
    out = ROOT / "runs/diagnostics" / f"{diag_id}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                    encoding="utf-8")


# ----------------------------------------------------------------------------- model / inference
class Runner:
    """Loads a verified checkpoint on CUDA (deterministic cuDNN) with a lazily created CPU copy used
    only if CUDA raises OOM for an image. fp32, eval, no_grad, PadCropWrapper(reflect, multiple 16)."""

    def __init__(self, key: str, device="cuda"):
        self.key = key
        self.path, self.sha, step_expected = verify_ckpt(key)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.model, self.info = load_model(self.path, self.device)
        if self.info["step"] != step_expected:
            raise SystemExit(f"[abort] {self.path} stored step {self.info['step']} != expected {step_expected}")
        self.wrapped = PadCropWrapper(self.model, self.model.cfg.multiple, mode="reflect")
        self._cpu = None
        self.device_counts = {}

    def _cpu_wrapped(self):
        if self._cpu is None:
            m, _ = load_model(self.path, torch.device("cpu"))
            self._cpu = (m, PadCropWrapper(m, m.cfg.multiple, mode="reflect"))
        return self._cpu[1]

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, force_device: str | None = None):
        """x: (1,3,H,W) float in [0,1] on CPU. Returns (outputs dict on CPU float32, device string)."""
        dev = force_device or self.device.type
        if dev == "cuda":
            try:
                o = self.wrapped(x.to(self.device))
                o = {k: v.float().cpu() if torch.is_tensor(v) else v for k, v in o.items()}
                self.device_counts["cuda"] = self.device_counts.get("cuda", 0) + 1
                return o, "cuda"
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
        o = self._cpu_wrapped()(x.cpu())
        o = {k: v.float() if torch.is_tensor(v) else v for k, v in o.items()}
        self.device_counts["cpu"] = self.device_counts.get("cpu", 0) + 1
        return o, "cpu"

    def meta(self):
        return {"ckpt_key": self.key, "ckpt": str(self.path), "ckpt_sha256_computed": self.sha,
                "ckpt_sha256_expected": CKPTS[self.key][1], "step": self.info["step"], "model_info": self.info,
                "model_cfg": self.model.cfg.__dict__, "device": self.device.type,
                "device_counts": dict(self.device_counts), "precision": "fp32, no autocast",
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark}


def to_u8_tensor(mu: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) float -> (H,W,3) uint8; identical rounding to evaluate_metrics.py."""
    return (mu.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)


def u8_to_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a).permute(2, 0, 1)[None].float().div(255.0)


def mse_u8(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr_from_mse(m):
    return float("inf") if m == 0 else 10 * np.log10(255.0 ** 2 / m)


def val_rows():
    return read_manifest(SPLIT / "val_manifest.txt", allow_test=False)


def train_sample_rows():
    return stratified_sample(read_manifest(SPLIT / "train_manifest.txt", allow_test=False), 517, 0)


# ----------------------------------------------------------------------------- bootstrap
def _rng():
    return np.random.default_rng(BOOT_SEED)


def boot_groups(groups: dict, stat, B=B_BOOT):
    """groups: {name: tuple of aligned per-image arrays}. stat(*arrays)->float.
    Per-group CI resamples images within the group (paired: arrays share indices). 'ALL' resamples each
    group independently and concatenates (stratified). Returns {name: {est, lo, hi, n}}."""
    out = {}
    names = list(groups)
    for g in names:
        arrs = [np.asarray(a, np.float64) for a in groups[g]]
        n = arrs[0].size
        rng = _rng()
        idx = rng.integers(0, n, size=(B, n))
        bs = np.array([stat(*[a[i] for a in arrs]) for i in idx])
        out[g] = {"est": float(stat(*arrs)), "lo": float(np.percentile(bs, 2.5)),
                  "hi": float(np.percentile(bs, 97.5)), "n": int(n)}
    rng = _rng()
    full = [np.concatenate([np.asarray(groups[g][k], np.float64) for g in names]) for k in range(len(groups[names[0]]))]
    idxs = {g: rng.integers(0, len(groups[g][0]), size=(B, len(groups[g][0]))) for g in names}
    offs, o = {}, 0
    for g in names:
        offs[g] = o; o += len(groups[g][0])
    bs = []
    for b in range(B):
        sel = np.concatenate([idxs[g][b] + offs[g] for g in names])
        bs.append(stat(*[a[sel] for a in full]))
    out["ALL"] = {"est": float(stat(*full)), "lo": float(np.percentile(bs, 2.5)),
                  "hi": float(np.percentile(bs, 97.5)), "n": int(full[0].size)}
    return out


def by_dataset(recs, *keys):
    ds = sorted({r["dataset"] for r in recs})
    return {d: tuple(np.array([r[k] for r in recs if r["dataset"] == d], np.float64) for k in keys) for d in ds}


mean_stat = lambda a: float(np.mean(a))
median_stat = lambda a: float(np.median(a))
pooled_psnr_stat = lambda m: psnr_from_mse(float(np.mean(m)))


def fmt_ci(c, nd=3):
    return f"{c['est']:.{nd}f} [{c['lo']:.{nd}f}, {c['hi']:.{nd}f}] (n={c['n']})"


class Timer:
    def __init__(self):
        self.t0 = time.time()
        self.start = datetime.now().isoformat(timespec="seconds")

    def done(self):
        return {"start": self.start, "end": datetime.now().isoformat(timespec="seconds"),
                "runtime_s": round(time.time() - self.t0, 1)}

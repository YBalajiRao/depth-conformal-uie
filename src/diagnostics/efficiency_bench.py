"""
Efficiency benchmark (run alone on the GPU, after Phase F). fp32, batch 1, eval mode, cuDNN deterministic.

Per model: parameters (numel), GMACs at 256x256 and 512x512 (thop; multiply-accumulates, reported as such),
peak allocated VRAM (MB) and FPS (20 warm-up + 100 timed forwards, CUDA-synchronised) at both sizes.
P1b-family models take the depth map as an input; the frozen Depth-Anything-V2-Small cost is reported separately and
added in a 'P1b_s0 + depth' row. U-shape is architecturally fixed to 256x256 (its protocol resizes larger inputs), so
its 512 row is the resize-protocol cost (resize + 256 forward + resize back).
Output: runs/diagnostics/EFFICIENCY_<ts>/summary.json
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "src/data"), str(ROOT / "src/validation"), str(ROOT / "baselines/adapters")]
import torch  # noqa: E402
import thop  # noqa: E402
import common  # noqa: E402
from common import load_model, git_state  # noqa: E402
import registry as RG  # noqa: E402
import precompute_depth as PD  # noqa: E402
from phase_f_test import MODELS  # noqa: E402

DEV = torch.device("cuda")
SIZES = (256, 512)


def timeit(fn, n_warm=20, n=100):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


def peak(fn):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); fn(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2 ** 20


def bench(name, model, make_inputs, predict=None):
    model = model.to(DEV).eval()
    r = {"params": int(sum(p.numel() for p in model.parameters()))}
    for s in SIZES:
        inp = make_inputs(s)
        fwd = (lambda: predict(inp)) if predict else (lambda: model(*inp))
        with torch.no_grad():
            if name == "ushape" and s != 256:
                r[f"gmacs_{s}"] = "fixed 256x256 architecture (resize protocol)"
            else:
                try:
                    macs, _ = thop.profile(model, inputs=make_inputs(256) if name == "ushape" else inp, verbose=False)
                    r[f"gmacs_{s}"] = round(macs / 1e9, 3)
                except Exception as e:  # noqa: BLE001
                    r[f"gmacs_{s}"] = f"n/a ({type(e).__name__})"
            r[f"peak_mb_{s}"] = round(peak(fwd), 1)
            r[f"fps_{s}"] = round(timeit(fwd), 2)
    print(name, r, flush=True)
    return r


def main():
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    commit = git_state()
    out = ROOT / "runs/diagnostics" / f"EFFICIENCY_{datetime.now():%Y%m%d_%H%M%S}"; out.mkdir(parents=True)
    res = {}
    for k in ("P1b_s0", "v2_s0", "P1e_s0"):
        kind, ck, sh = MODELS[k]
        m, _ = load_model(ROOT / ck, torch.device("cpu"))
        if kind == "dgt":
            mk = lambda s: (torch.rand(1, 3, s, s, device=DEV), torch.rand(1, 1, s, s, device=DEV))  # noqa: E731
        elif kind == "scene":
            mk = lambda s: (torch.rand(1, 3, s, s, device=DEV), torch.rand(1, 1, s, s, device=DEV), torch.randn(1, 16, device=DEV))  # noqa: E731
        else:
            mk = lambda s: (torch.rand(1, 3, s, s, device=DEV),)  # noqa: E731
        res[k] = bench(k, m, mk)
    for k in ("fiveaplus", "wweuie", "ushape"):
        m = RG.build(k)
        mk = lambda s: (torch.rand(1, 3, s, s, device=DEV),)  # noqa: E731
        if k == "ushape":
            mg = m.to(DEV).eval()
            res[k] = bench(k, m, mk, predict=lambda inp: RG.predict("ushape", mg, inp[0]))
        else:
            res[k] = bench(k, m, mk, predict=lambda inp, k=k, m=m: RG.predict(k, m.to(DEV).eval(), inp[0]))
    # frozen depth prior (Depth-Anything-V2-Small, fp16 as used), processor resize included in the timing
    proc, dm, dm16, meta = PD.load_depth_model(DEV, fp16=True)
    import numpy as np
    d = {"params": int(sum(p.numel() for p in dm16.parameters())), "precision": "fp16 (as precomputed)"}
    for s in SIZES:
        x8 = (np.random.default_rng(0).random((s, s, 3)) * 255).astype(np.uint8)
        f = lambda: PD.disparity(proc, dm, dm16, x8, DEV)  # noqa: E731
        d[f"peak_mb_{s}"] = round(peak(f), 1); d[f"fps_{s}"] = round(timeit(f, 5, 30), 2)
    res["depth_anything_v2_small"] = d; print("depth", d, flush=True)
    res["P1b_s0 + depth"] = {f"fps_{s}": round(1 / (1 / res["P1b_s0"][f"fps_{s}"] + 1 / d[f"fps_{s}"]), 2) for s in SIZES}
    summ = {"commit": commit, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "precision": "fp32 (models)",
            "batch": 1, "notes": "gmacs = thop multiply-accumulates; ushape 512 = resize protocol", "results": res}
    (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()

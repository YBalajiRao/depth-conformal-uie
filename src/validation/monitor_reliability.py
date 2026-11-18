"""
CPU-only reliability monitor for a running SS-PGN training job (never touches the GPU; runs at
below-normal priority).

Each time <run>/last.pt reaches a new step (and once for every --extra-ckpt), it evaluates a FIXED,
dataset-stratified set of 40 validation images (seed 0) and appends to <run>/reliability_trend_v2.jsonl.
Three per-pixel "uncertainty" maps are scored against |Y - J_hat| on identical pixels:
    sigma : SS-PGN predictive std (NIG, total variance)
    diff  : |J_hat - I|               (trivial baseline 1: how much the restorer changed the pixel)
    grad  : |Sobel(J_hat)| per channel (trivial baseline 2: edges are hard)
For each: mean/median Spearman, AUSE-MAE, AUSE-RMSE, remaining-RMSE ratio after dropping the
top 25% / 50%. Also PSNR.

STEP-20k RULE (fixed 2026-11-06 18:20, before any step >= 20000 data existed; see §11 of the methodology doc)
  Baseline = step 4000 on the same 40 images (snapshot runs/.../snapshots/step4000.pt; the step-2000
  checkpoint was overwritten). At the first monitor point with step >= 20000, the head is "improving"
  only if ALL hold:
    (a) mean Spearman(sigma)  >= Spearman_4000 + 0.10
    (b) AUSE-RMSE(sigma)      <= 0.90 * AUSE-RMSE_4000
    (c) AUSE-RMSE(sigma)      <  AUSE-RMSE(diff)  and  < AUSE-RMSE(grad)   (at that step)
  The main run continues to 100k either way. On FAIL: prepare a PILOT (not the main run) that changes
  exactly one thing: the evidential weight (le, currently 0.01), or a post-hoc Gaussian-NLL sigma head
  on the frozen restorer.

  python src/validation/monitor_reliability.py --run runs/sspgn/20261106_173805 \
         --extra-ckpt runs/sspgn/20261106_173805/snapshots/step4000.pt
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""                   # hard guarantee: CPU only
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/validation"))
sys.path.insert(0, str(ROOT / "src/utils"))
from calibrate_uncertainty import read_manifest, load_model, predictive_std, sparsification, ause, spearman, to_tensor  # noqa: E402
from pad_crop_wrapper import PadCropWrapper  # noqa: E402

RULE = {"baseline_step": 4000, "decision_step": 20000, "d_spearman": 0.10, "ause_factor": 0.90,
        "fixed_at": "2026-11-06T18:20"}
MAPS = ("sigma", "diff", "grad")


def lower_priority():
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
        return "below-normal"
    except Exception as e:                                # psutil missing or no permission
        return f"unchanged ({type(e).__name__})"


def fixed_subset(rows, n, seed):
    by = {}
    for r in rows:
        by.setdefault(r["dataset"], []).append(r)
    rng = random.Random(seed)
    out = []
    for d in sorted(by):
        k = max(1, round(n * len(by[d]) / len(rows)))
        out += rng.sample(by[d], min(k, len(by[d])))
    return sorted(out, key=lambda r: r["unit"])[:n]


def sobel_mag(x):
    """Per-channel Sobel gradient magnitude, (1,3,H,W) -> (1,3,H,W), replicate padding."""
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=x.dtype).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(xp, kx, groups=3)
    gy = F.conv2d(xp, kx.transpose(-1, -2), groups=3)
    return torch.sqrt(gx * gx + gy * gy)


@torch.no_grad()
def evaluate(ckpt, rows):
    model, info = load_model(ckpt, torch.device("cpu"))
    wrapped = PadCropWrapper(model, model.cfg.multiple, mode="reflect")
    per = {m: {"spearman": [], "ause_mae": [], "ause_rmse": [], "drop25": [], "drop50": []} for m in MAPS}
    psnrs, by_ds = [], {}
    for r in rows:
        x, y = to_tensor(r["input"]), to_tensor(r["reference"])
        nig = wrapped(x)["nig"].float()
        mu = nig[:, 0:3]
        err = (mu - y).abs().flatten()
        maps = {"sigma": predictive_std(nig[:, 3:6], nig[:, 6:9], nig[:, 9:12], "total").flatten(),
                "diff": (mu - x).abs().flatten(),
                "grad": sobel_mag(mu).flatten()}
        for m, u in maps.items():
            assert u.shape == err.shape, (m, u.shape, err.shape)
            sp = sparsification(err, u)
            per[m]["spearman"].append(spearman(u, err))
            per[m]["ause_mae"].append(ause(sp, "mae"))
            per[m]["ause_rmse"].append(ause(sp, "rmse"))
            per[m]["drop25"].append(sp["rmse_model"][25] / max(sp["rmse_model"][0], 1e-12))
            per[m]["drop50"].append(sp["rmse_model"][50] / max(sp["rmse_model"][0], 1e-12))
        by_ds.setdefault(r["dataset"], []).append(per["sigma"]["spearman"][-1])
        psnrs.append(10 * np.log10(1 / max(float(((mu.clamp(0, 1) - y) ** 2).mean()), 1e-12)))
    res = {"step": info["step"], "n": len(rows), "psnr": float(np.mean(psnrs)),
           "sigma_spearman_by_dataset": {d: float(np.mean(v)) for d, v in by_ds.items()}}
    for m in MAPS:
        res[m] = {"spearman_mean": float(np.mean(per[m]["spearman"])),
                  "spearman_median": float(np.median(per[m]["spearman"])),
                  "ause_mae": float(np.mean(per[m]["ause_mae"])), "ause_rmse": float(np.mean(per[m]["ause_rmse"])),
                  "rmse_ratio_drop25": float(np.mean(per[m]["drop25"])),
                  "rmse_ratio_drop50": float(np.mean(per[m]["drop50"]))}
    return res


def check_rule(entries):
    base = next((e for e in entries if e["step"] == RULE["baseline_step"]), None)
    cur = next((e for e in sorted(entries, key=lambda e: e["step"]) if e["step"] >= RULE["decision_step"]), None)
    if base is None or cur is None:
        return None
    s0, a0 = base["sigma"]["spearman_mean"], base["sigma"]["ause_rmse"]
    s1, a1 = cur["sigma"]["spearman_mean"], cur["sigma"]["ause_rmse"]
    a = s1 >= s0 + RULE["d_spearman"]
    b = a1 <= RULE["ause_factor"] * a0
    c = a1 < cur["diff"]["ause_rmse"] and a1 < cur["grad"]["ause_rmse"]
    return {"rule": RULE, "baseline_step": base["step"], "decision_step": cur["step"],
            "a": {"pass": a, "spearman_now": s1, "threshold": s0 + RULE["d_spearman"], "spearman_4000": s0},
            "b": {"pass": b, "ause_rmse_now": a1, "threshold": RULE["ause_factor"] * a0, "ause_rmse_4000": a0},
            "c": {"pass": c, "ause_rmse_sigma": a1, "ause_rmse_diff": cur["diff"]["ause_rmse"],
                  "ause_rmse_grad": cur["grad"]["ause_rmse"]},
            "overall": "PASS" if (a and b and c) else "FAIL"}


def fmt(res):
    s = f"[monitor] step {res['step']}: PSNR={res['psnr']:.2f}"
    for m in MAPS:
        e = res[m]
        s += (f" | {m}: Spearman={e['spearman_mean']:.3f} AUSE-RMSE={e['ause_rmse']:.3f} "
              f"drop25/50={e['rmse_ratio_drop25']:.3f}/{e['rmse_ratio_drop50']:.3f}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--extra-ckpt", nargs="*", default=[])
    ap.add_argument("--split-dir", default=str(ROOT / "data/splits/lsui_uieb_all_d6_seed0"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    prio = lower_priority()

    run = Path(args.run)
    rows = fixed_subset(read_manifest(Path(args.split_dir) / "val_manifest.txt", allow_test=False), args.n, args.seed)
    trend = run / "reliability_trend_v2.jsonl"
    entries = [json.loads(l) for l in open(trend, encoding="utf-8")] if trend.exists() else []
    done = {e["step"] for e in entries}
    rule_path = run / "step20k_rule_result.json"
    print(f"[monitor] {len(rows)} fixed val images (units: {[r['unit'] for r in rows][:3]}...), priority={prio}, "
          f"done steps={sorted(done)}", flush=True)

    def record(path):
        nonlocal entries
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "snap.pt"
            try:
                shutil.copy2(path, tmp)
                step = torch.load(tmp, map_location="cpu", weights_only=False)["step"]
            except Exception as e:
                print(f"[monitor] {path.name} not readable yet ({type(e).__name__}); retrying", flush=True)
                return
            if step in done:
                return
            t0 = time.time()
            res = evaluate(tmp, rows)
            res["eval_seconds"], res["ckpt"] = round(time.time() - t0, 1), str(path)
        with open(trend, "a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")
        entries.append(res); done.add(res["step"])
        print(fmt(res) + f" ({res['eval_seconds']}s)", flush=True)
        if not rule_path.exists():
            verdict = check_rule(entries)
            if verdict is not None:
                rule_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
                print(f"[RULE step>={RULE['decision_step']}] (a) {verdict['a']} (b) {verdict['b']} (c) {verdict['c']} "
                      f"-> {verdict['overall']}", flush=True)

    for p in args.extra_ckpt:
        record(Path(p))
    while True:
        if (run / "last.pt").exists():
            record(run / "last.pt")
        if args.once:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()

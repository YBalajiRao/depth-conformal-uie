"""
P1e + Gate G2 chain (pre-registered in report section P; PI launch approval 2026-11-17). Runs unattended:

  1. train P1e = --phys-conditioning v2 --dgt --dgt-ordered --scene-prior-pca 16, seed 0, Stage-1 schedule, 100k steps
     (identical command to P1b/P1d except the scene flag), out runs/phase1/P1e_scene_pca16;
  2. check completion (last train-log step == 100000), SHA-256 of last.pt (the only selection rule);
  3. 517-image evaluation (phase2_eval517.py eval, fp32, parity gate) as arm P1e_s0;
  4. paired bootstrap vs P1b_s0 and v2_s0 (phase2_eval517.py pair);
  5. Gate G2: adopt P1e only if dPSNR8 (P1e_s0 - P1b_s0) >= +0.2 dB with the 95% CI excluding 0 on ALL or UIEB;
     otherwise P1b remains the model. Single seed; provisional.
  6. charge GPU wall time to runs/baselines/phase_d_status.json (P1e reserve) and write runs/phase1/P1e_G2_decision.json.
Stops (no retry) on any failure and records it.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
OUT = ROOT / "runs/phase1"
STATUS = ROOT / "runs/baselines/phase_d_status.json"
DECISION = OUT / "P1e_G2_decision.json"
P1B = ("runs/diagnostics/P1B_EVAL517_20261110_184113/per_image_P1b_v2_dgt_ordered.csv",
       "c16fd969710197074a2faa0f7fda403c88c213a3929e1be6437bc1f66637dccc")
V2 = ("runs/diagnostics/S1_EVAL517_20261109_165139/per_image_v2_full.csv",
      "6388aa038b1193a00bc8a39e28c965635de6be36184b6e65ff833909e5766f39")
PCA = ROOT / "data/scene/da2_small_5426e4f/lsui_uieb_all_d6_seed0/pca16_train.npz"
CMD = [PY, "-u", "src/training/train.py", "--crop", "256", "--batch", "4", "--accum", "2", "--iters", "100000",
       "--lr", "5e-4", "--lr-min", "1e-6", "--warmup", "1000", "--clip", "1.0", "--val-every", "2000",
       "--val-subset", "100", "--workers", "4", "--vram-frac", "0.85", "--seed", "0", "--phys-conditioning", "v2",
       "--dgt", "--dgt-ordered", "--scene-prior-pca", "16", "--out", "runs/phase1/P1e_scene_pca16"]
state = {"started": datetime.now().isoformat(timespec="seconds"), "events": []}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def log(msg, **kw):
    state["events"].append({"t": datetime.now().isoformat(timespec="seconds"), "msg": msg, **kw})
    DECISION.write_text(json.dumps(state, indent=1), encoding="utf-8")
    print(f"[p1e {datetime.now():%H:%M:%S}] {msg} {kw if kw else ''}", flush=True)


def charge(seconds, what):
    s = json.loads(STATUS.read_text(encoding="utf-8"))
    s["spent_days"] += seconds / 86400
    ev = {"t": datetime.now().isoformat(timespec="seconds"), "msg": f"charged {what} (P1e reserve)",
          "days": round(seconds / 86400, 4), "spent": round(s["spent_days"], 4),
          "remaining": round(s["budget_start"] - s["spent_days"], 4)}
    s["events"].append(ev)
    STATUS.write_text(json.dumps(s, indent=1), encoding="utf-8")
    log(ev["msg"], days=ev["days"], spent=ev["spent"], remaining=ev["remaining"])


def run(cmd, logfile):
    t0 = time.time()
    with open(logfile, "w", encoding="utf-8") as f:
        code = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    return code, time.time() - t0


def latest(pattern):
    c = sorted(ROOT.glob(pattern))
    return c[-1] if c else None


def main():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    launch = {"created": state["started"], "git_head": head, "git_clean_src": not dirty, "pca16_sha256": sha(PCA),
              "gate": "G2 (report sections M, P): adopt only if paired 517 dPSNR8 (P1e_s0 - P1b_s0) >= +0.2 dB with 95% "
                      "CI excluding 0 on ALL or UIEB; else P1b remains the model",
              "comparator": "P1b seed 0 runs/phase1/P1b_v2_dgt_ordered/20261110_093229/last.pt (5b22ea7c...)",
              "test_split": "not accessed", "checkpoint_selection": "last.pt at step 100000",
              "command": " ".join(["python"] + CMD[1:])}
    eval_only = "--eval-only" in sys.argv
    if eval_only:      # 2026-11-17: re-run evaluation/pairing/G2 on the existing last.pt after the parity-control fix
        prev = json.loads(DECISION.read_text(encoding="utf-8")) if DECISION.exists() else {}
        state["events"] = prev.get("events", []); state["launch"] = prev.get("launch", launch)
        log("eval-only rerun (parity failure control revised: +/-8 checkerboard)", git_head=head, git_clean_src=not dirty)
        code = 0
    else:
        (OUT / "phase1e_launch.json").write_text(json.dumps(launch, indent=1), encoding="utf-8")
        state["launch"] = launch
        log("launch P1e")
        code, w = run(CMD, OUT / "console_P1e_scene_pca16.log")
        charge(w, "train P1e")
    rd = latest("runs/phase1/P1e_scene_pca16/2*")
    steps = None
    if rd and (rd / "train_log.jsonl").exists():
        last = [l for l in open(rd / "train_log.jsonl", encoding="utf-8") if l.strip()][-1]
        steps = json.loads(last).get("step")
    if code != 0 or steps != 100000 or not (rd / "last.pt").exists():
        log("STOP: P1e training did not complete", code=code, steps=steps); return
    ck = rd / "last.pt"; ck_sha = sha(ck)
    log("P1e trained", run_dir=str(rd), last_pt_sha256=ck_sha)
    code, w = run([PY, "-u", "src/diagnostics/phase2_eval517.py", "eval", "--arm", "P1e_s0", "--ckpt", str(ck),
                   "--sha", ck_sha, "--vram-frac", "0.50"], ROOT / "runs/diagnostics/console_P2_EVAL517_P1e_s0.log")
    charge(w, "eval517 P1e")
    ev = latest("runs/diagnostics/P2_EVAL517_P1e_s0_*")
    if code != 0 or ev is None:
        log("STOP: P1e evaluation failed", code=code); return
    summ = json.loads((ev / "summary.json").read_text(encoding="utf-8"))
    a = f"{summ['per_image_csv']}:{summ['per_image_sha256']}"
    res = {"eval_dir": str(ev), "absolute_psnr8": {g: summ["absolute"]["psnr8_mean"][g]["est"] for g in ("ALL", "LSUI", "UIEB")},
           "absolute_ssim8": {g: summ["absolute"]["ssim8_mean"][g]["est"] for g in ("ALL", "LSUI", "UIEB")}, "pairs": {}}
    for ref, (rp, rs) in (("P1b_s0", P1B), ("v2_s0", V2)):
        code, _ = run([PY, "src/diagnostics/phase2_eval517.py", "pair", "--a", a, "--b", f"{ROOT / rp}:{rs}",
                       "--name", f"P1e_s0 - {ref}"], ROOT / f"runs/diagnostics/console_pair_P1e_s0_{ref}.log")
        pr = latest("runs/diagnostics/P2_PAIR_*")
        if code != 0 or pr is None:
            log(f"STOP: pairing vs {ref} failed", code=code); return
        res["pairs"][ref] = {"pair_dir": str(pr), **{g: {k: c[k] for k in ("est", "lo", "hi", "d_z", "ci_excludes_0")}
                             for g, c in json.loads((pr / "paired.json").read_text(encoding="utf-8"))["paired"]["d_psnr8"].items()}}
    g2 = {g: bool(res["pairs"]["P1b_s0"][g]["est"] >= 0.2 and res["pairs"]["P1b_s0"][g]["lo"] > 0) for g in ("ALL", "UIEB")}
    res["gate_G2"] = {"per_group_pass": g2, "passed": any(g2.values()),
                      "winner": "P1e" if any(g2.values()) else "P1b", "provisional": "single seed"}
    state["result"] = res
    log(f"GATE G2 {'PASSED -> P1e adopted' if any(g2.values()) else 'FAILED -> P1b remains the model'}", per_group=g2)
    s = json.loads(STATUS.read_text(encoding="utf-8"))
    s["results"]["P1e_s0"] = {**res, "ckpt_sha256": ck_sha}
    s["events"].append({"t": datetime.now().isoformat(timespec="seconds"), "msg": f"Gate G2: winner {res['gate_G2']['winner']}"})
    STATUS.write_text(json.dumps(s, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

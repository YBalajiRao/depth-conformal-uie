"""
Phase D driver (PI decisions 2026-11-13): run the remaining baselines sequentially and unattended.

  1. wait for the running Five A+ job (runs/baselines/console_fiveaplus.log 'exit=' line), then 517-image evaluation
     + pairing against P1b seed 0 and v2 seed 0;
  2. for U-shape, WWE-UIE, Spectroformer (this order; PI 2026-11-14): 500-step pilot (fp16; if a half-precision 'not implemented'
     error occurs -> fp32, the disclosed O1 exception; if the pilot diverges -> lr 2e-4, the pre-registered fallback),
     projected 100k cost from the measured step time; STOP (status file) if launching would leave less than the
     0.38 GPU-day reserve for P1e; otherwise train 100k, then evaluate + pair;
  3. write runs/baselines/phase_d_status.json after every step (budget ledger, decisions, results).
GPU time is charged as wall-clock (single GPU). Starting budget: 1.88 GPU-days remaining at Five A+ launch.
"""
from __future__ import annotations

import json
import math
import re
import statistics as st
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = sys.executable
OUT = ROOT / "runs/baselines"
STATUS = OUT / "phase_d_status.json"
BUDGET0, RESERVE = 1.88, 0.38
ORDER = ["ushape", "wweuie", "spectroformer"]          # PI priority 2026-11-14 (was ushape, spectroformer, wweuie)
P1B = ("runs/diagnostics/P1B_EVAL517_20261110_184113/per_image_P1b_v2_dgt_ordered.csv",
       "c16fd969710197074a2faa0f7fda403c88c213a3929e1be6437bc1f66637dccc")
V2 = ("runs/diagnostics/S1_EVAL517_20261109_165139/per_image_v2_full.csv",
      "6388aa038b1193a00bc8a39e28c965635de6be36184b6e65ff833909e5766f39")
state = {"started": datetime.now().isoformat(timespec="seconds"), "budget_start": BUDGET0, "reserve_P1e": RESERVE,
         "spent_days": 0.0, "events": [], "results": {}}


def log(msg, **kw):
    state["events"].append({"t": datetime.now().isoformat(timespec="seconds"), "msg": msg, **kw})
    STATUS.write_text(json.dumps(state, indent=1), encoding="utf-8")
    print(f"[driver {datetime.now():%H:%M:%S}] {msg} {kw if kw else ''}", flush=True)


def charge(seconds, what):
    state["spent_days"] += seconds / 86400
    log(f"charged {what}", days=round(seconds / 86400, 4), spent=round(state["spent_days"], 4),
        remaining=round(BUDGET0 - state["spent_days"], 4))


def run(cmd, logfile):
    t0 = time.time()
    with open(logfile, "w", encoding="utf-8") as f:
        code = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    return code, time.time() - t0


def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def latest(dirpat):
    c = sorted((ROOT / dirpat.rsplit("/", 1)[0]).glob(dirpat.rsplit("/", 1)[1]))
    return c[-1] if c else None


def eval_and_pair(name, run_dir):
    ck = run_dir / "last.pt"
    s = sha(ck)
    code, w = run([PY, "-u", str(HERE / "eval517_baseline.py"), "--name", name, "--ckpt", str(ck), "--sha", s],
                  OUT / f"console_eval517_{name}.log")
    charge(w, f"eval517 {name}")
    if code != 0:
        log(f"EVAL FAILED {name}", code=code); return None
    ev = latest(f"runs/diagnostics/BASELINE_EVAL517_{name}_*")
    summ = json.loads((ev / "summary.json").read_text(encoding="utf-8"))
    csv_, csha = summ["per_image_csv"], summ["per_image_sha256"]
    res = {"ckpt_sha256": s, "eval_dir": str(ev),
           "absolute": {g: summ["absolute"]["psnr8_mean"][g]["est"] for g in ("LSUI", "UIEB", "ALL")},
           "ssim_all": summ["absolute"]["ssim8_mean"]["ALL"]["est"], "pairs": {}}
    for ref_name, (rp, rs) in (("P1b_s0", P1B), ("v2_s0", V2)):
        code, _ = run([PY, str(ROOT / "src/diagnostics/phase2_eval517.py"), "pair", "--a", f"{csv_}:{csha}",
                       "--b", f"{ROOT / rp}:{rs}", "--name", f"{name} - {ref_name}"], OUT / f"console_pair_{name}_{ref_name}.log")
        pr = latest("runs/diagnostics/P2_PAIR_*")
        res["pairs"][ref_name] = {g: {k: v for k, v in c.items() if k in ("est", "lo", "hi")}
                                  for g, c in json.loads((pr / "paired.json").read_text(encoding="utf-8"))["paired"]["d_psnr8"].items()}
    state["results"][name] = res
    log(f"evaluated {name}", absolute=res["absolute"])
    return res


def train_log_stats(run_dir):
    L = [json.loads(l) for l in open(run_dir / "train_log.jsonl", encoding="utf-8") if l.strip()]
    t = [L[i]["sec"] - L[i - 1]["sec"] for i in range(min(101, len(L) - 1), len(L))]
    fin = [x for x in L if math.isfinite(x["total"])]
    return {"steps": len(L), "s_per_step": st.mean(t) if t else float("nan"),
            "l1_first50": st.mean(x["l1"] for x in fin[:50]), "l1_last50": st.mean(x["l1"] for x in fin[-50:])}


def pilot(name):
    # --force-fp32 NAME: PI-approved O1 exception applied from the start (2026-11-17: Spectroformer's fft.ifftn in
    # ComplexHalf gives non-finite losses at step 0, which the 'not implemented for Half' detector cannot catch)
    forced = "--force-fp32" in sys.argv and sys.argv[sys.argv.index("--force-fp32") + 1] == name
    flags, lr = (["--no-amp"] if forced else []), "5e-4"
    if forced:
        log(f"{name}: fp32 forced by PI (O1 exception; ComplexHalf FFT non-finite at step 0)")
    # --small-batch NAME: PI-approved memory accommodation (2026-11-17): micro-batch 2 x accum 4 = effective batch 8,
    # identical to the unified 4 x 2; needed because Spectroformer in fp32 OOMs at micro-batch 4 under the 6.8 GiB cap
    if "--small-batch" in sys.argv and sys.argv[sys.argv.index("--small-batch") + 1] == name:
        flags += ["--batch", "2", "--accum", "4"]
        log(f"{name}: micro-batch 2 x accum 4 (effective batch 8 unchanged; memory accommodation, PI-approved)")
    for attempt in range(3):
        pdir = OUT / "pilots" / name
        cmd = [PY, "-u", str(HERE / "train_baseline.py"), "--name", name, "--iters", "500", "--val-every", "500",
               "--lr", lr, "--out", str(pdir)] + flags
        lf = OUT / f"console_pilot_{name}_{attempt}.log"
        code, w = run(cmd, lf)
        charge(w, f"pilot {name} attempt {attempt}")
        text = lf.read_text(encoding="utf-8", errors="replace")
        if code != 0 and re.search(r"not implemented for 'Half'", text) and "--no-amp" not in flags:
            flags.append("--no-amp"); log(f"{name}: fp16 unsupported -> fp32 (O1 exception)"); continue
        rd = sorted(pdir.glob("*"))[-1] if pdir.exists() else None
        if code != 0 or rd is None:
            if lr == "5e-4" and "non-finite" in text:
                lr = "2e-4"; log(f"{name}: pilot diverged -> lr 2e-4 (pre-registered fallback)"); continue
            log(f"PILOT FAILED {name}", code=code, tail=text[-800:]); return None
        s = train_log_stats(rd)
        if not (s["l1_last50"] < s["l1_first50"]) and lr == "5e-4":
            lr = "2e-4"; log(f"{name}: pilot L1 did not decrease -> lr 2e-4 (pre-registered fallback)", **s); continue
        log(f"pilot ok {name}", flags=flags, lr=lr, **s)
        return {"flags": flags, "lr": lr, **s}
    log(f"PILOT FAILED {name} after retries"); return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start_at = sys.argv[sys.argv.index("--start-at") + 1] if "--start-at" in sys.argv else None
    reused = {}
    if start_at:
        # restart after a STOP: keep the ledger and results of the previous driver; reuse its passed pilot for start_at
        prev = json.loads(STATUS.read_text(encoding="utf-8"))
        state.update(spent_days=prev["spent_days"], results=prev["results"], events=prev["events"])
        ok = [e for e in prev["events"] if e["msg"] == f"pilot ok {start_at}"]
        if ok and "--fresh-pilot" not in sys.argv:
            reused = {k: ok[-1][k] for k in ("flags", "lr", "steps", "s_per_step", "l1_first50", "l1_last50")}
        log(f"driver restart at {start_at} (previous STOP); ledger carried over", spent=round(state["spent_days"], 4),
            reused_pilot=bool(reused))
        order = ORDER[ORDER.index(start_at):]
    else:
        log("driver start; waiting for Five A+")
        con = OUT / "console_fiveaplus.log"
        while "exit=" not in con.read_text(encoding="utf-8", errors="replace"):
            time.sleep(60)
        rd = latest("runs/baselines/fiveaplus/2*")
        s = train_log_stats(rd)
        last = [l for l in open(rd / "train_log.jsonl", encoding="utf-8") if l.strip()][-1]
        charge(json.loads(last)["sec"], "train fiveaplus (wall)")
        if not re.search(r"exit=0", con.read_text(encoding="utf-8", errors="replace")) or s["steps"] != 100000:
            log("FIVEAPLUS DID NOT COMPLETE", steps=s["steps"]); return
        eval_and_pair("fiveaplus", rd)
        order = ORDER
    for name in order:
        p = reused if (name == start_at and reused) else pilot(name)
        if p is None:
            log(f"STOP: pilot failed for {name}; waiting for PI"); return
        proj = p["s_per_step"] * 100000 * 1.05 / 86400 + 0.01          # +5% validation, + eval
        left = BUDGET0 - state["spent_days"]
        log(f"budget check {name}", projected_days=round(proj, 3), remaining=round(left, 3),
            after=round(left - proj, 3), reserve=RESERVE)
        if left - proj < RESERVE:
            log(f"STOP: launching {name} would cut into the P1e reserve; waiting for PI"); return
        cmd = [PY, "-u", str(HERE / "train_baseline.py"), "--name", name, "--lr", p["lr"]] + p["flags"]
        code, w = run(cmd, OUT / f"console_{name}.log")
        charge(w, f"train {name}")
        rd = latest(f"runs/baselines/{name}/2*")
        if code != 0 or rd is None or train_log_stats(rd)["steps"] != 100000:
            log(f"STOP: training {name} did not complete", code=code); return
        eval_and_pair(name, rd)
    log("DONE: all baselines trained and evaluated; GPU idle; P1e reserve intact",
        remaining=round(BUDGET0 - state["spent_days"], 3))


if __name__ == "__main__":
    main()

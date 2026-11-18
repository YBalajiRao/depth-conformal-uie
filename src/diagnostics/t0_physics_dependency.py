"""
T0: physics-output dependency test (protocol §T0). Inference only.

  intervention : forward hook on model.gphys replaces every output with a constant
                 (B_inf=0.5, M=I, kappa=1, beam_mu=0.5, beam_sigma=0.3, beam_amp=0)
  criterion    : PASS if max|mu_hooked - mu| == 0 exactly on all 20 images; FAIL otherwise
  positive ctl : forward hook on model.tail_R adds 0.01 -> max|mu_ctrl - mu| must be > 0 on every image,
                 else INVALID
  determinism  : mu computed twice without hooks must agree exactly (precondition; else INVALID)
Images: first 20 (sorted) of monitor_reliability.fixed_subset(val, 40, seed 0). Same device, same process.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from common import (Runner, git_state, new_outdir, write_json, environment, Timer, val_rows,  # noqa: E402
                    fixed_subset, load_u8, u8_to_tensor)


def const_hook(_m, _inp, out):
    rep = {}
    for k, v in out.items():
        if k == "B_inf":
            rep[k] = torch.full_like(v, 0.5)
        elif k == "M":
            rep[k] = torch.eye(3, device=v.device, dtype=v.dtype).expand_as(v).clone()
        elif k == "kappa":
            rep[k] = torch.ones_like(v)
        elif k == "beam_mu":
            rep[k] = torch.full_like(v, 0.5)
        elif k == "beam_sigma":
            rep[k] = torch.full_like(v, 0.3)
        elif k == "beam_amp":
            rep[k] = torch.zeros_like(v)
        else:
            raise RuntimeError(f"unexpected gphys output {k}")
    return rep


def main():
    commit = git_state()
    tm = Timer()
    run = Runner("last")
    out = new_outdir("T0")
    rows = fixed_subset(val_rows(), 40, 0)[:20]
    recs = []
    for r in rows:
        x = u8_to_tensor(load_u8(r["input"]))
        base, dev = run(x)
        base2, dev2 = run(x, force_device=dev)
        h = run.model.gphys.register_forward_hook(const_hook)
        if run._cpu is not None:
            hc = run._cpu[0].gphys.register_forward_hook(const_hook)
        try:
            hooked, dev3 = run(x, force_device=dev)
        finally:
            h.remove()
            if run._cpu is not None:
                hc.remove()
        h = run.model.tail_R.register_forward_hook(lambda _m, _i, o: o + 0.01)
        if run._cpu is not None:
            hc = run._cpu[0].tail_R.register_forward_hook(lambda _m, _i, o: o + 0.01)
        try:
            ctrl, dev4 = run(x, force_device=dev)
        finally:
            h.remove()
            if run._cpu is not None:
                hc.remove()
        assert dev == dev2 == dev3 == dev4, (dev, dev2, dev3, dev4)
        mu = base["J"]
        # the hook must actually have changed the physics outputs (else the test is vacuous)
        phys_changed = float((hooked["B"] - base["B"]).abs().max())
        rec = {"unit": r["unit"], "dataset": r["dataset"], "H": mu.shape[-2], "W": mu.shape[-1], "device": dev,
               "repeat_maxabs": float((base2["J"] - mu).abs().max()),
               "hooked_maxabs": float((hooked["J"] - mu).abs().max()),
               "hooked_B_maxabs_change": phys_changed,
               "hooked_Jphys_maxabs_change": float((hooked["J_phys"] - base["J_phys"]).abs().max()),
               "ctrl_maxabs": float((ctrl["J"] - mu).abs().max())}
        recs.append(rec)
        print(rec, flush=True)
    det_ok = all(r["repeat_maxabs"] == 0 for r in recs)
    ctrl_ok = all(r["ctrl_maxabs"] > 0 for r in recs)
    hook_effective = all(r["hooked_B_maxabs_change"] > 0 for r in recs)
    indep = all(r["hooked_maxabs"] == 0 for r in recs)
    status = "INVALID" if not (det_ok and ctrl_ok and hook_effective) else ("PASS" if indep else "FAIL")
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
    summary = {"diagnostic": "T0", "status": status, "criterion": "max|mu_hooked - mu| == 0 on all 20 images",
               "max_hooked_maxabs": max(r["hooked_maxabs"] for r in recs),
               "determinism_ok": det_ok, "positive_control_ok": ctrl_ok,
               "min_ctrl_maxabs": min(r["ctrl_maxabs"] for r in recs),
               "hook_changed_physics_outputs_on_all": hook_effective,
               "n": len(recs), "labels": "validation, not held-out"}
    write_json(out / "summary.json", summary)
    write_json(out / "meta.json", {"commit": commit, **run.meta(), "env": environment(), **tm.done(),
                                   "images": [r["unit"] for r in rows]})
    print(f"[T0] {status}  -> {out}", flush=True)


if __name__ == "__main__":
    main()

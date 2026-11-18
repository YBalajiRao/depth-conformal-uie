"""
CPU-only dry run for the external baselines. No CUDA is touched (CUDA_VISIBLE_DEVICES=-1 is set before torch is
imported and torch.cuda.is_available() is asserted False). No training beyond one optimizer step on one crop.

Per baseline:
  1. build the official architecture (random init), count parameters;
  2. forward a dummy (1,3,256,256) tensor;
  3. probe input sizes to find the spatial multiple the architecture needs (output must equal input size);
  4. one training iteration on one real 256 crop from our train split via train.PairedCropDataset
     (batch_size = 1): forward, L1 loss, backward, Adam step; check finite loss and gradients on all params;
  5. one native-resolution prediction on a small real validation image through registry.predict(), with
     PSNR8 / SSIM8 from the SS-PGN metric code (untrained: values are only a pipeline check).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import registry as R  # noqa: E402
import data as D  # noqa: E402

PROBE = [(256, 256), (240, 320), (248, 328), (252, 340), (250, 330), (257, 333)]


def main():
    assert not torch.cuda.is_available(), "CUDA must not be visible"
    torch.set_num_threads(4)                      # leave CPU for the running training job's data workers
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    tr = D.train_dataset(256)
    va_rows = D.rows("val")
    small_val = min(va_rows, key=lambda r: np.prod(D.load_u8(r[2]).shape[:2]))
    report = {}
    for name, spec in R.BASELINES.items():
        t0 = time.time()
        rec = {"repo": spec.repo, "commit": spec.commit, "url": spec.url, "licence": spec.licence}
        try:
            model = R.build(name)
            rec["params"] = int(sum(p.numel() for p in model.parameters()))
            model.eval()
            with torch.no_grad():
                y = R.raw_forward(name, model, torch.rand(1, 3, 256, 256))
            rec["dummy_256_out"] = list(y.shape)
            ok_sizes = []
            for hw in PROBE:
                try:
                    with torch.no_grad():
                        o = R.raw_forward(name, model, torch.rand(1, 3, *hw))
                    ok_sizes.append([*hw, list(o.shape[-2:]) == list(hw)])
                except Exception as e:                # size not supported
                    ok_sizes.append([*hw, f"error: {type(e).__name__}"])
            rec["size_probe"] = ok_sizes
            # one training iteration on a real crop
            model.train()
            random.seed(1)
            x, yref, m = tr[0]
            x, yref, m = x[None], yref[None], m[None]
            opt = torch.optim.Adam(model.parameters(), lr=1e-4)
            J = R.raw_forward(name, model, x)
            loss = ((J - yref).abs() * m).sum() / (m.sum() * 3)
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.requires_grad]
            rec["train_iter"] = {"loss": float(loss), "finite": bool(torch.isfinite(loss)),
                                 "params_with_grad": sum(g is not None and bool(torch.isfinite(g).all()) for g in grads),
                                 "params_total": len(grads)}
            opt.step()
            # native-resolution prediction on a real validation image
            model.eval()
            x8, y8 = D.load_u8(small_val[2]), D.load_u8(small_val[3])
            xt = torch.from_numpy(x8).permute(2, 0, 1)[None].float() / 255.0
            with torch.no_grad():
                Jv = R.predict(name, model, xt)
            p8 = D.to_u8(Jv)
            rec["native_eval"] = {"unit": small_val[1], "HxW": list(x8.shape[:2]), "out_HxW": list(p8.shape[:2]),
                                  "psnr8_untrained": D.psnr_u8(p8, y8), "ssim8_untrained": D.ssim_u8(p8, y8),
                                  "eval_mode": spec.eval_mode}
            rec["status"] = "OK"
        except Exception as e:
            rec["status"] = f"FAILED: {type(e).__name__}: {e}"
        rec["sec"] = round(time.time() - t0, 1)
        report[name] = rec
        print(json.dumps({name: rec}, indent=1, default=str), flush=True)
    out = Path(__file__).resolve().parents[1] / "cpu_dry_run_report.json"
    out.write_text(json.dumps({"torch": torch.__version__, "cuda_visible": torch.cuda.is_available(),
                               "threads": torch.get_num_threads(), "baselines": report}, indent=1, default=str),
                   encoding="utf-8")
    print(f"[dry-run] wrote {out}; cuda_visible={torch.cuda.is_available()}")


if __name__ == "__main__":
    main()

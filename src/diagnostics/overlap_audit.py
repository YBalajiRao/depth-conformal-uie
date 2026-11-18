"""
Data-overlap audit (CPU only; 2026-11-13). Detects images in candidate evaluation sets that also occur in OUR
training split (data/splits/lsui_uieb_all_d6_seed0/train_manifest.txt). The test manifest is never read.

Method: 64-bit difference hash (9x8 grayscale, bilinear) of every training input; for each candidate image, the
minimum Hamming distance to any training input. Hash hits are then verified at the pixel level (both images resized
to 128x128 bilinear, PSNR). Positive control: training images vs themselves must all give distance 0.

Candidates:
  EUVP test_samples/Inp (official EUVP test), EUVP Paired/{underwater_dark, underwater_imagenet, underwater_scenes}
  inputs (trainA + validation), UIEB challenge_60, and our own validation inputs (near-duplicate frames).
External training lists (pretrained-checkpoint contamination): UDnet splits/V2/{LSUI_train, UIEB_train},
splits/UIEB/train800 vs our validation unit names.
Writes reports/integrity_audit/overlap_audit_<date>.json.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/validation"), str(ROOT / "src/training")]
import train as T  # noqa: E402

IMG = (".jpg", ".jpeg", ".png", ".bmp")


def dhash(p, n=8):
    with Image.open(p) as im:
        g = np.asarray(im.convert("L").resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).ravel())


def hashes(paths):
    return np.stack([dhash(p) for p in paths])


def nearest(H, H_tr, chunk=2000):
    d, j = [], []
    for i in range(0, len(H), chunk):
        D = np.unpackbits(np.bitwise_xor(H[i:i + chunk, None, :], H_tr[None]), axis=2).sum(2)
        d.append(D.min(1)); j.append(D.argmin(1))
    return np.concatenate(d), np.concatenate(j)


def pix_psnr(a, b, size=(128, 128)):
    A = np.asarray(Image.open(a).convert("RGB").resize(size, Image.BILINEAR), dtype=np.float32)
    B = np.asarray(Image.open(b).convert("RGB").resize(size, Image.BILINEAR), dtype=np.float32)
    mse = float(((A - B) ** 2).mean())
    return 99.0 if mse == 0 else float(10 * np.log10(255 ** 2 / mse))


def main():
    split = ROOT / "data/splits/lsui_uieb_all_d6_seed0"
    tr = T.read_manifest(split / "train_manifest.txt")
    va = T.read_manifest(split / "val_manifest.txt")
    tr_paths = [r[2] for r in tr]
    H_tr = hashes(tr_paths)
    ctl = bool((nearest(H_tr[:50], H_tr)[0] == 0).all())
    out = {"date": datetime.now().isoformat(timespec="seconds"), "n_train": len(tr), "positive_control_ok": ctl,
           "sets": {}}
    E = ROOT / "data/raw/EUVP"
    cands = {
        "EUVP_test_samples": sorted(p for p in (E / "test_samples/Inp").glob("*") if p.suffix.lower() in IMG),
        **{f"EUVP_{s}": sorted(p for p in (E / "Paired" / s).rglob("*")
                               if p.suffix.lower() in IMG and p.parent.name in ("trainA", "validation"))
           for s in ("underwater_dark", "underwater_imagenet", "underwater_scenes")},
        "UIEB_challenge_60": sorted(p for p in (ROOT / "data/raw/UIEB/challenge_60").glob("*") if p.suffix.lower() in IMG),
        "our_val": [Path(r[2]) for r in va],
    }
    for name, paths in cands.items():
        d, j = nearest(hashes(paths), H_tr)
        hits = np.where(d <= 8)[0]
        ver = [pix_psnr(paths[i], tr_paths[j[i]]) for i in hits]
        dup = int(sum(v >= 35 for v in ver))
        near = int(sum(v >= 25 for v in ver))
        out["sets"][name] = {"n": len(paths), "hash_exact": int((d == 0).sum()), "hash_le4": int((d <= 4).sum()),
                             "hash_le8": len(hits), "verified_duplicates_pix_ge35dB": dup,
                             "verified_near_pix_ge25dB": near, "clean_fraction_hash_gt8": float((d > 8).mean())}
        if name == "our_val":
            out["sets"][name]["near_by_dataset_pix_ge25dB"] = {
                ds: int(sum(1 for i, v in zip(hits, ver) if v >= 25 and va[i][0] == ds)) for ds in ("LSUI", "UIEB")}
        print(name, out["sets"][name], flush=True)
    stem = lambda s: Path(s.strip().replace("\\", "/")).stem.lower()  # noqa: E731
    ours = {ds: {stem(r[1].split(":", 1)[1]) for r in va if r[0] == ds} for ds in ("LSUI", "UIEB")}
    S = ROOT / "baselines/repos/UDnet/splits"
    ext = {}
    for f, ds in (("V2/LSUI_train.txt", "LSUI"), ("V2/UIEB_train.txt", "UIEB"), ("UIEB/train800.txt", "UIEB")):
        names = {stem(n) for n in (S / f).read_text(encoding="utf-8", errors="replace").splitlines() if n.strip()}
        ext[f] = {"list_size": len(names), "overlap_with_our_val": len(ours[ds] & names), "our_val_size": len(ours[ds])}
    out["external_training_lists_vs_our_val"] = ext
    print(ext)
    dst = ROOT / "reports/integrity_audit" / f"overlap_audit_{datetime.now():%Y%m%d}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", dst)


if __name__ == "__main__":
    main()

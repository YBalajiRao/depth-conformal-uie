"""
Create small pilot manifests from the leakage-safe split manifests.

  python src/data/create_pilot_manifest.py                      # 100 train / 20 val, stratified
  python src/data/create_pilot_manifest.py --strategy head      # literally the first N lines

Strategies
  stratified (default): N rows drawn deterministically (seeded) in proportion to the datasets in
              the manifest, so the pilot contains both LSUI and UIEB. This exercises the
              small-image reflect-pad + mask path (UIEB images down to 159 px).
  head:       the first N data lines. The manifests are sorted by unit id, so this
              gives LSUI images only.

Output: <split-dir>/pilot/train_pilot_manifest.txt and val_pilot_manifest.txt, in the same 5-column
format as the parent manifests. Pilot rows are a subset of their own parent split, so they
add no leakage. The test manifest is never read.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"{path} (run src/data/build_splits.py first)")
    lines = path.read_text(encoding="utf-8").splitlines()
    header = [l for l in lines if l.startswith("#")]
    rows = [l for l in lines if l.strip() and not l.startswith("#")]
    for i, r in enumerate(rows, 1):
        if len(r.split("\t")) != 5:
            raise ValueError(f"{path.name}: data line {i} does not have 5 tab-separated fields")
    return (header[0] if header else "# dataset\tunit_id\tcomponent_id\tinput_path\treference_path"), rows


def pick(rows, n, strategy, seed):
    if n > len(rows):
        raise ValueError(f"asked for {n} rows but manifest has {len(rows)}")
    if strategy == "head":
        return rows[:n]
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r.split("\t")[0]].append(r)
    rng = random.Random(seed)
    quota = {d: max(1, round(n * len(v) / len(rows))) for d, v in by_ds.items()}
    while sum(quota.values()) > n:                       # fix rounding overshoot
        quota[max(quota, key=quota.get)] -= 1
    while sum(quota.values()) < n:
        quota[max(by_ds, key=lambda d: len(by_ds[d]) - quota[d])] += 1
    out = []
    for d in sorted(by_ds):
        out += rng.sample(by_ds[d], quota[d])
    return sorted(out, key=lambda r: r.split("\t")[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", default=str(ROOT / "data/splits/lsui_uieb_all_d6_seed0"))
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--strategy", choices=["stratified", "head"], default="stratified")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    split = Path(args.split_dir)
    out = split / "pilot"
    out.mkdir(exist_ok=True)
    picked = {}
    for name, n in (("train", args.n_train), ("val", args.n_val)):
        header, rows = read(split / f"{name}_manifest.txt")
        sel = pick(rows, n, args.strategy, args.seed)
        missing = [r.split("\t")[3] for r in sel if not Path(r.split("\t")[3]).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} input files missing, e.g. {missing[0]}")
        (out / f"{name}_pilot_manifest.txt").write_text(
            header + f"\n# pilot: {args.strategy}, n={n}, seed={args.seed}, parent={name}_manifest.txt\n"
            + "\n".join(sel) + "\n", encoding="utf-8", newline="\n")
        picked[name] = sel
        print(f"[pilot] {name}: {len(sel)} rows {dict(Counter(r.split(chr(9))[0] for r in sel))} -> "
              f"{out / f'{name}_pilot_manifest.txt'}")
    overlap = {r.split("\t")[1] for r in picked["train"]} & {r.split("\t")[1] for r in picked["val"]}
    if overlap:
        print(f"ERROR: {len(overlap)} units in both pilot train and val", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

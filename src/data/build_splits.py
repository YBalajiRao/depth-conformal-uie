"""
Leakage-safe train/val/test splitter for LSUI + UIEB (paired units).

A *unit* is one (input, reference) pair, identified as "<DATASET>:<filename>"
(e.g. "LSUI:102.jpg", "UIEB:100_img_.png"), matching the overlap-audit CSVs.

Algorithm
  1. Enumerate all LSUI and UIEB units from disk (input and reference must both exist).
  2. Read unit-pair edges from the overlap audits (v1 unit_level_overlap.csv and
     v2 unit_pairs_best_evidence.csv). EUVP units are kept as graph nodes, so two
     LSUI units linked only *through* an EUVP image still end up in one component.
  3. Union-Find (path compression + union by size) -> connected components.
  4. Assign whole components to train/val/test (default 80/10/10) with a
     deterministic greedy balancer, stratified by dataset.
  5. Verify, and abort on any violation:
       - every unit assigned exactly once;
       - every edge between two assigned units stays within one split;
       - no file path in two splits;
       - no SHA-256 file content in two splits (independent check, via the hash inventory).
  6. Write manifests + component table + summary into a NEW versioned directory
     (refuses to overwrite unless --overwrite).

Edge policy
  --edges all   (default) exact SHA-256 duplicates + pHash candidates (d <= --max-phash).
                Conservative for SPLITTING: an unverified candidate can only merge
                components (keep data together); it never removes data.
  --edges exact only byte-identical duplicates.
  Precision/recall of the pHash candidates is still unknown (no manual sample yet).
  Grouping by them is safe for leakage; it is NOT a basis for exclusion.

Manifest line format (tab-separated, header line starts with '#'):
  dataset  unit_id  component_id  input_path  reference_path

Usage
  python src/data/build_splits.py                       # writes data/splits/<auto-name>/
  python src/data/build_splits.py --edges exact --seed 1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")

DATASETS = {  # dataset -> (input dir, reference dir)
    "LSUI": (ROOT / "data/raw/LSUI/input", ROOT / "data/raw/LSUI/GT"),
    "UIEB": (ROOT / "data/raw/UIEB/raw_images", ROOT / "data/raw/UIEB/reference_images"),
}
DEFAULT_PAIR_CSVS = [
    ROOT / "reports/dataset_audit/unit_level_overlap/unit_level_overlap.csv",
    ROOT / "reports/dataset_audit/overlap_audit_v2_20261105/unit_pairs_best_evidence.csv",
]
DEFAULT_HASH_CSV = ROOT / "reports/dataset_audit/overlap_audit_v2_20261105/combined_hash_inventory.csv"


class SplitError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Union-Find
# --------------------------------------------------------------------------
class DSU:
    def __init__(self):
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def add(self, x: str):
        if x not in self.parent:
            self.parent[x], self.size[x] = x, 1

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:            # path compression (iterative)
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def enumerate_units() -> dict[str, dict]:
    units = {}
    for ds, (din, dref) in DATASETS.items():
        if not din.is_dir() or not dref.is_dir():
            raise SplitError(f"{ds}: missing folder {din if not din.is_dir() else dref}")
        ins = {p.name for p in din.iterdir() if p.suffix.lower() in IMG_EXT}
        refs = {p.name for p in dref.iterdir() if p.suffix.lower() in IMG_EXT}
        if ins != refs:
            raise SplitError(f"{ds}: input/reference name mismatch "
                             f"(only input: {sorted(ins - refs)[:5]}, only ref: {sorted(refs - ins)[:5]})")
        for name in sorted(ins):
            units[f"{ds}:{name}"] = {"dataset": ds, "input": str(din / name), "reference": str(dref / name)}
        print(f"[units] {ds}: {len(ins)} paired units")
    return units


def read_edges(csv_paths, policy: str, max_phash: int):
    edges, stats = set(), Counter()
    for p in csv_paths:
        p = Path(p)
        if not p.is_file():
            raise SplitError(f"pair CSV not found: {p}")
        with open(p, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            need = {"UnitA", "UnitB", "BestEvidence", "pHashDistance"}
            if not need <= set(reader.fieldnames or []):
                raise SplitError(f"{p.name}: missing columns {need - set(reader.fieldnames or [])}")
            n = 0
            for r in reader:
                n += 1
                a, b, ev = r["UnitA"].strip(), r["UnitB"].strip(), r["BestEvidence"].strip()
                if not a or not b or a == b:
                    stats["skipped_self_or_empty"] += 1
                    continue
                exact = ev.upper().startswith("EXACT")
                if not exact:
                    try:
                        d = int(float(r["pHashDistance"]))
                    except ValueError:
                        raise SplitError(f"{p.name}: bad pHashDistance {r['pHashDistance']!r} for {a}|{b}")
                    if policy == "exact" or d > max_phash:
                        stats["filtered_out"] += 1
                        continue
                edges.add((min(a, b), max(a, b), "exact" if exact else "phash"))
            stats[f"rows:{p.name}"] = n
    # collapse duplicate (a,b) seen as both exact and phash -> keep exact label
    best = {}
    for a, b, kind in edges:
        if best.get((a, b)) != "exact":
            best[(a, b)] = kind
    return [(a, b, k) for (a, b), k in sorted(best.items())], stats


def load_sha(hash_csv: Path) -> dict[str, str]:
    if not hash_csv.is_file():
        raise SplitError(f"hash inventory not found: {hash_csv}")
    out = {}
    with open(hash_csv, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            out[str(Path(r["FullPath"])).lower()] = r["SHA256"].upper()
    return out


# --------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------
def assign(components: dict[str, list[str]], units, ratios, seed):
    totals = Counter(units[u]["dataset"] for c in components.values() for u in c)
    target = {d: {s: ratios[i] * totals[d] for i, s in enumerate(SPLITS)} for d in totals}
    cur = {d: Counter() for d in totals}
    order = sorted(components)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda c: -len(components[c]))       # stable: large first, random within size
    comp_split = {}
    for cid in order:
        comp = Counter(units[u]["dataset"] for u in components[cid])

        def score(s):
            return sum(n * (target[d][s] - cur[d][s]) / max(target[d][s], 1e-9) for d, n in comp.items())
        best = max(SPLITS, key=lambda s: (score(s), -SPLITS.index(s)))
        comp_split[cid] = best
        for d, n in comp.items():
            cur[d][best] += n
    return comp_split, cur, totals


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="+", default=[str(p) for p in DEFAULT_PAIR_CSVS])
    ap.add_argument("--hash-csv", default=str(DEFAULT_HASH_CSV))
    ap.add_argument("--edges", choices=["all", "exact"], default="all")
    ap.add_argument("--max-phash", type=int, default=6)
    ap.add_argument("--ratios", default="0.8,0.1,0.1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output dir (default data/splits/<auto>)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    ratios = [float(x) for x in args.ratios.split(",")]
    if len(ratios) != 3 or abs(sum(ratios) - 1) > 1e-9 or min(ratios) <= 0:
        raise SplitError(f"--ratios must be 3 positive numbers summing to 1, got {ratios}")
    tag = f"lsui_uieb_{args.edges}{'' if args.edges == 'exact' else f'_d{args.max_phash}'}_seed{args.seed}"
    out = Path(args.out) if args.out else ROOT / "data/splits" / tag
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SplitError(f"{out} exists and is not empty (use --overwrite or --out)")
    out.mkdir(parents=True, exist_ok=True)

    units = enumerate_units()
    edges, estats = read_edges(args.pairs, args.edges, args.max_phash)
    print(f"[edges] {len(edges)} unique unit edges kept "
          f"({sum(k == 'exact' for *_, k in edges)} exact, {sum(k == 'phash' for *_, k in edges)} pHash) | {dict(estats)}")

    unknown_target = sorted({x for a, b, _ in edges for x in (a, b)
                             if x.split(":", 1)[0] in DATASETS and x not in units})
    if unknown_target:
        raise SplitError(f"{len(unknown_target)} LSUI/UIEB units in the CSVs are not on disk, e.g. {unknown_target[:5]}")

    dsu = DSU()
    for u in units:
        dsu.add(u)
    for a, b, _ in edges:
        dsu.union(a, b)

    comp_all = defaultdict(list)
    for node in dsu.parent:
        comp_all[dsu.find(node)].append(node)
    components, comp_meta = {}, {}
    for root, members in comp_all.items():
        tgt = sorted(m for m in members if m in units)
        if not tgt:
            continue
        cid = "C" + hashlib.sha1(tgt[0].encode()).hexdigest()[:10]   # stable id from smallest member
        components[cid] = tgt
        comp_meta[cid] = {"n_units": len(tgt),
                          "external_nodes": sorted(m for m in members if m not in units)}
    sizes = Counter(len(v) for v in components.values())
    largest = max(components.values(), key=len)
    print(f"[dsu] {len(components)} components over {len(units)} units; "
          f"singletons={sizes[1]}, largest={len(largest)} ({Counter(u.split(':')[0] for u in largest)})")

    comp_split, cur, totals = assign(components, units, ratios, args.seed)
    unit_split = {u: comp_split[c] for c, us in components.items() for u in us}

    # ---------------- verification (abort on any failure) ----------------
    errs = []
    if len(unit_split) != len(units):
        errs.append(f"assigned {len(unit_split)} != units {len(units)}")
    for a, b, k in edges:
        if a in unit_split and b in unit_split and unit_split[a] != unit_split[b]:
            errs.append(f"edge leak ({k}) {a}[{unit_split[a]}] -- {b}[{unit_split[b]}]")
    path_split = {}
    for u, s in unit_split.items():
        for p in (units[u]["input"], units[u]["reference"]):
            key = str(Path(p)).lower()
            if path_split.setdefault(key, s) != s:
                errs.append(f"path in two splits: {p}")
    sha = load_sha(Path(args.hash_csv))
    sha_split, missing_sha = {}, 0
    for u, s in unit_split.items():
        for p in (units[u]["input"], units[u]["reference"]):
            h = sha.get(str(Path(p)).lower())
            if h is None:
                missing_sha += 1
                continue
            prev = sha_split.setdefault(h, (s, p))
            if prev[0] != s:
                errs.append(f"identical content (SHA {h[:12]}) in {prev[0]} ({prev[1]}) and {s} ({p})")
    if missing_sha:
        errs.append(f"{missing_sha} files have no SHA-256 in the inventory (re-run the hash audit)")
    if errs:
        (out / "VERIFICATION_FAILED.txt").write_text("\n".join(errs), encoding="utf-8")
        raise SplitError(f"{len(errs)} verification errors, first: {errs[:3]} (see {out}/VERIFICATION_FAILED.txt)")
    print(f"[verify] PASS: {len(edges)} edges, {len(path_split)} paths, {len(sha_split)} SHA-256 contents, 0 leaks")

    # ---------------- outputs ----------------
    for s in SPLITS:
        rows = sorted(u for u, sp in unit_split.items() if sp == s)
        with open(out / f"{s}_manifest.txt", "w", encoding="utf-8", newline="\n") as f:
            f.write("# dataset\tunit_id\tcomponent_id\tinput_path\treference_path\n")
            uc = {u: c for c, us in components.items() for u in us}
            for u in rows:
                m = units[u]
                f.write(f"{m['dataset']}\t{u}\t{uc[u]}\t{m['input']}\t{m['reference']}\n")
    with open(out / "components.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component_id", "split", "n_units", "units", "external_nodes"])
        for cid in sorted(components, key=lambda c: (-len(components[c]), c)):
            w.writerow([cid, comp_split[cid], len(components[cid]), " ".join(components[cid]),
                        " ".join(comp_meta[cid]["external_nodes"])])

    ext_by_split = {s: Counter() for s in SPLITS}
    for cid, meta in comp_meta.items():
        for fam in {n.split(":", 1)[0] for n in meta["external_nodes"]}:
            ext_by_split[comp_split[cid]][fam] += 1
    summary = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "edge_stats": dict(estats),
        "n_edges": len(edges),
        "n_units": len(units),
        "n_components": len(components),
        "component_size_hist": dict(sorted(sizes.items())),
        "largest_component": len(largest),
        "counts": {s: {d: cur[d][s] for d in totals} for s in SPLITS},
        "achieved_ratio": {d: {s: round(cur[d][s] / totals[d], 4) for s in SPLITS} for d in totals},
        "components_linked_to_external_family_by_split": {s: dict(c) for s, c in ext_by_split.items()},
        "verification": "PASS",
        "caveats": [
            "pHash edges are unverified candidates (precision/recall unknown); used only to group, never to exclude.",
            "Checks cover duplicates WITHIN the local LSUI/UIEB/EUVP files only. Pretrained restorer checkpoints may "
            "have been trained on any of these images (training data unknown).",
            "EUVP-linked components are reported so EUVP-test can be cleaned before cross-dataset evaluation.",
        ],
    }
    (out / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for s in SPLITS:
        print(f"[split] {s:5s} " + "  ".join(f"{d}={cur[d][s]} ({cur[d][s] / totals[d]:.1%})" for d in sorted(totals)))
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SplitError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

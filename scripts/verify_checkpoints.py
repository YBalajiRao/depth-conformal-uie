"""
Verify model checkpoints against checkpoints_sha256.json.

Place each checkpoint at the path listed in checkpoints_sha256.json (relative to the repository root), then run
  python scripts/verify_checkpoints.py
Prints OK / MISSING / MISMATCH per checkpoint and exits with status 1 if any present file does not match.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((ROOT / "checkpoints_sha256.json").read_text(encoding="utf-8"))
    mismatch = 0
    for name, entry in manifest.items():
        p = ROOT / entry["path"]
        if not p.is_file():
            print(f"MISSING   {name:15s} {entry['path']}")
            continue
        ok = sha256(p) == entry["sha256"]
        mismatch += not ok
        print(f"{'OK' if ok else 'MISMATCH':9s} {name:15s} {entry['path']}")
    sys.exit(1 if mismatch else 0)


if __name__ == "__main__":
    main()

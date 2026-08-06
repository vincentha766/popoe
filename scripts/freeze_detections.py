#!/usr/bin/env python3
"""Materialize and verify the frozen detection inputs (B2-F2).

The frozen run tables pin detection files by SHA256, but the working tree
held 19 of them as ABSOLUTE-PATH symlinks — a clone or pod rsync gets
FileNotFoundError (or silently empty inputs) the moment the link target is
not there. Freezing therefore means two invariants, both enforced here:

  1. No symlinks under data/detections/ — every file is real bytes
     (default mode replaces each symlink with a copy of its target,
     `cp -L` semantics; a broken link is a hard failure).
  2. Every *.json's SHA256 appears in its directory's PROVENANCE.md —
     the registry of officially-downloaded (or explicitly local) artefacts.
     A truncated download, a wrong-batch file or a stray edit changes the
     hash, the hash is then absent from the registry, and the run refuses
     to start. Set-membership is deliberate: PROVENANCE stays the single
     authority for which bytes are legitimate, and this script never needs
     a filename↔hash table of its own.

Usage:
    python scripts/freeze_detections.py            # materialize + verify
    python scripts/freeze_detections.py --check    # verify only; symlinks
                                                   # are themselves failures
                                                   # (pod-side preflight)

Exit 0 = every file real and registered. Non-zero = list of violations on
stderr. JSON-parse checking stays the downloader's job (see PROVENANCE:
size alone lies) — this gate is about identity, not schema.
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

DET_ROOT = Path(__file__).resolve().parents[1] / "data" / "detections"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def materialize_symlink(link: Path) -> str:
    """Replace `link` with a real copy of its target (cp -L). Returns the
    resolved target path for reporting. Broken target raises FileNotFoundError.
    Write-to-tmp + os.replace so a crash never leaves a half-written file
    where the link used to be."""
    target = link.resolve(strict=True)
    tmp = link.with_name(link.name + ".materialize.tmp")
    tmp.write_bytes(target.read_bytes())
    os.replace(tmp, link)
    return str(target)


def run(root: Path, check_only: bool) -> int:
    failures = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        jsons = sorted(d.glob("*.json"))
        if not jsons:
            continue
        prov = d / "PROVENANCE.md"
        if not prov.exists():
            failures.append(f"{d}: no PROVENANCE.md — unregistered directory")
            continue
        registered = prov.read_text()
        for j in jsons:
            if j.is_symlink():
                if check_only:
                    failures.append(f"{j}: still a symlink (freeze invariant "
                                    f"is real bytes; run without --check)")
                    continue
                try:
                    src = materialize_symlink(j)
                except FileNotFoundError:
                    failures.append(f"{j}: BROKEN symlink "
                                    f"-> {os.readlink(j)}")
                    continue
                print(f"materialized {j.relative_to(root)}  <- {src}")
            digest = sha256_file(j)
            if digest in registered:
                print(f"ok  {j.relative_to(root)}  {digest}")
            else:
                failures.append(f"{j}: sha256 {digest} not registered in "
                                f"{prov.name}")
    for f in failures:
        print(f"FAIL  {f}", file=sys.stderr)
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="verify only; any remaining symlink is a failure")
    ap.add_argument("--root", default=str(DET_ROOT),
                    help="detections root (default: repo data/detections)")
    args = ap.parse_args()
    return run(Path(args.root), args.check)


if __name__ == "__main__":
    raise SystemExit(main())

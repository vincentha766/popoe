#!/usr/bin/env python3
"""Materialize and verify the frozen detection inputs (B2-F2).

The frozen run tables pin detection files by SHA256, but the working tree
held 19 of them as ABSOLUTE-PATH symlinks — a clone or pod rsync gets
FileNotFoundError (or silently empty inputs) the moment the link target is
not there. Freezing therefore means these invariants, all enforced here:

  1. No symlinks under data/detections/ — neither files nor DIRECTORIES
     (default mode replaces file symlinks with real bytes, `cp -L`
     semantics; a broken link or a symlinked source directory is a hard
     failure — a dangling directory link makes a whole source vanish
     silently).
  2. Every present *.json matches `MANIFEST.sha256` (sha256sum format,
     paths relative to the detections root, tracked in git) — PATH AND
     BYTES together. Set-membership against PROVENANCE alone would pass
     two registered files whose contents were swapped, or a registered
     546-batch file copied over a 441 path; the manifest binds each hash
     to its one legitimate path. Unregistered files fail too.
  3. Every *.json's SHA256 also appears in its directory's PROVENANCE.md
     (the origin registry: which official artefact these bytes are).
  4. A tree with zero jsons fails — "rsync lost everything" must not
     read as "nothing to check". Per-run completeness is `--need`:
     runbooks pass the relative paths that run consumes, and each must
     exist (a partial tree for other runs stays legal).
  5. Leftover `*.materialize.tmp*` files fail (interrupted materialize).

Usage:
    python scripts/freeze_detections.py                # materialize + verify
    python scripts/freeze_detections.py --check \\
        --need cnos/cnos-fastsam_lmo-test.json ...     # pod-side preflight

Exit 0 = clean. Non-zero = violations on stderr. JSON-parse checking stays
the downloader's job (size alone lies) — this gate is about identity.
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

DET_ROOT = Path(__file__).resolve().parents[1] / "data" / "detections"
MANIFEST = "MANIFEST.sha256"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def materialize_symlink(link: Path) -> str:
    """Replace `link` with a real copy of its target (cp -L). Returns the
    resolved target path for reporting. Broken target raises FileNotFoundError.
    Write-to-tmp + os.replace so a crash never leaves a half-written file at
    the link's path; pid in the tmp name so two concurrent materializers
    cannot truncate each other's tmp."""
    target = link.resolve(strict=True)
    tmp = link.with_name(f"{link.name}.materialize.tmp.{os.getpid()}")
    tmp.write_bytes(target.read_bytes())
    os.replace(tmp, link)
    return str(target)


def load_manifest(root: Path):
    """{relative-posix-path: sha256} from MANIFEST.sha256 (sha256sum format)."""
    mf = root / MANIFEST
    if not mf.exists():
        return None
    out = {}
    for line in mf.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, rel = line.partition("  ")
        out[rel.strip().lstrip("*")] = digest.strip()
    return out


def run(root: Path, check_only: bool, need: tuple = ()) -> int:
    failures = []
    manifest = load_manifest(root)
    if manifest is None:
        failures.append(f"{root / MANIFEST}: missing — the path<->hash "
                        f"binding is the freeze table's ground; regenerate "
                        f"with: (cd {root} && sha256sum */*.json > {MANIFEST})")
        manifest = {}

    for p in sorted(root.rglob("*")):
        if p.is_dir() and p.is_symlink():
            failures.append(f"{p}: DIRECTORY is a symlink — a dangling link "
                            f"here silently vanishes a whole source; make it "
                            f"a real directory")
        if ".materialize.tmp" in p.name:
            failures.append(f"{p}: leftover materialize tmp (interrupted "
                            f"run) — inspect and delete")

    jsons = sorted(p for p in root.rglob("*.json") if not p.parent.is_symlink())
    if not jsons:
        failures.append(f"{root}: zero *.json files — an empty tree is a "
                        f"lost-inputs state, not a clean one")

    for j in jsons:
        rel = j.relative_to(root).as_posix()
        if j.is_symlink():
            if check_only:
                failures.append(f"{j}: still a symlink (freeze invariant is "
                                f"real bytes; run without --check)")
                continue
            try:
                src = materialize_symlink(j)
            except FileNotFoundError:
                failures.append(f"{j}: BROKEN symlink -> {os.readlink(j)}")
                continue
            print(f"materialized {rel}  <- {src}")
        digest = sha256_file(j)
        want = manifest.get(rel)
        if want is None:
            failures.append(f"{j}: not in {MANIFEST} — unregistered file")
        elif digest != want:
            failures.append(f"{j}: sha256 {digest} != manifest {want} — "
                            f"wrong bytes at this path (swap/truncation/"
                            f"wrong batch)")
        prov = j.parent / "PROVENANCE.md"
        if not prov.exists():
            failures.append(f"{j.parent}: no PROVENANCE.md — unregistered "
                            f"directory")
        elif digest not in prov.read_text():
            failures.append(f"{j}: sha256 {digest} not registered in "
                            f"{prov.relative_to(root).as_posix()}")
        if want is not None and digest == want:
            print(f"ok  {rel}  {digest}")

    present = {j.relative_to(root).as_posix() for j in jsons}
    for rel in need:
        if rel not in present:
            failures.append(f"{root / rel}: REQUIRED by this run (--need) "
                            f"but absent")

    for f in failures:
        print(f"FAIL  {f}", file=sys.stderr)
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="verify only; any remaining symlink is a failure")
    ap.add_argument("--need", nargs="*", default=[], metavar="RELPATH",
                    help="relative paths this run consumes; each must exist")
    ap.add_argument("--root", default=str(DET_ROOT),
                    help="detections root (default: repo data/detections)")
    args = ap.parse_args()
    return run(Path(args.root), args.check, tuple(args.need))


if __name__ == "__main__":
    raise SystemExit(main())

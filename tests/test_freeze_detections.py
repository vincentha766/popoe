"""scripts/freeze_detections.py — the B2-F2 freeze gate.

Two invariants: no symlinks under the detections root (pods get real bytes),
and every *.json's sha256 registered in its directory's PROVENANCE.md.
"""
import hashlib
import importlib.util
import os
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "freeze_detections.py"

spec = importlib.util.spec_from_file_location("freeze_detections", _SCRIPT)
fd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fd)


def _tree(tmp_path, with_unregistered=False):
    root = tmp_path / "detections"
    d = root / "src1"
    d.mkdir(parents=True)
    a = d / "a.json"
    a.write_text('[{"scene_id": 1}]')
    target = tmp_path / "elsewhere" / "b_target.json"
    target.parent.mkdir()
    target.write_text('[{"scene_id": 2}]')
    (d / "b.json").symlink_to(target)
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (a, target)]
    (d / "PROVENANCE.md").write_text(
        "# reg\n" + "\n".join(f"`{h}`" for h in hashes) + "\n")
    if with_unregistered:
        (d / "c.json").write_text('[{"scene_id": 3}]')
    return root, d


def test_materialize_replaces_symlink_and_verifies(tmp_path):
    root, d = _tree(tmp_path)
    assert fd.run(root, check_only=False) == 0
    b = d / "b.json"
    assert not b.is_symlink() and b.is_file()          # real bytes now
    assert fd.run(root, check_only=True) == 0          # preflight passes


def test_check_mode_refuses_symlinks(tmp_path):
    root, _ = _tree(tmp_path)
    assert fd.run(root, check_only=True) == 1          # symlink still present


def test_unregistered_hash_fails(tmp_path):
    root, _ = _tree(tmp_path, with_unregistered=True)
    assert fd.run(root, check_only=False) == 1


def test_broken_symlink_fails(tmp_path):
    root, d = _tree(tmp_path)
    (d / "b.json").unlink()
    (d / "b.json").symlink_to(tmp_path / "gone.json")
    assert fd.run(root, check_only=False) == 1


def test_missing_provenance_is_a_failure(tmp_path):
    root, d = _tree(tmp_path)
    (d / "PROVENANCE.md").unlink()
    assert fd.run(root, check_only=False) == 1

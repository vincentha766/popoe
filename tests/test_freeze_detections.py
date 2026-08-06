"""scripts/freeze_detections.py — the B2-F2 freeze gate.

Invariants: no symlinks (files OR directories), every present json bound to
its path by MANIFEST.sha256 AND registered in its directory's PROVENANCE.md,
zero-json trees fail, --need paths must exist, materialize tmp leftovers fail.
"""
import hashlib
import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "freeze_detections.py"

spec = importlib.util.spec_from_file_location("freeze_detections", _SCRIPT)
fd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fd)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _tree(tmp_path):
    """src1/a.json (real), src1/b.json (symlink to elsewhere), both registered
    in PROVENANCE and MANIFEST."""
    root = tmp_path / "detections"
    d = root / "src1"
    d.mkdir(parents=True)
    a = d / "a.json"
    a.write_text('[{"scene_id": 1}]')
    target = tmp_path / "elsewhere" / "b_target.json"
    target.parent.mkdir()
    target.write_text('[{"scene_id": 2}]')
    (d / "b.json").symlink_to(target)
    (d / "PROVENANCE.md").write_text(
        "# reg\n" + "\n".join(f"`{_sha(p)}`" for p in (a, target)) + "\n")
    (root / "MANIFEST.sha256").write_text(
        f"{_sha(a)}  src1/a.json\n{_sha(target)}  src1/b.json\n")
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


def test_unregistered_file_fails(tmp_path):
    root, d = _tree(tmp_path)
    (d / "c.json").write_text('[{"scene_id": 3}]')
    assert fd.run(root, check_only=False) == 1


def test_swapped_contents_fail_path_binding(tmp_path):
    """Both hashes are registered, but at each other's paths — set-membership
    would pass; the manifest's path<->hash binding must not."""
    root, d = _tree(tmp_path)
    assert fd.run(root, check_only=False) == 0         # materialize b first
    a, b = d / "a.json", d / "b.json"
    ta, tb = a.read_text(), b.read_text()
    a.write_text(tb)
    b.write_text(ta)
    assert fd.run(root, check_only=True) == 1


def test_missing_manifest_fails(tmp_path):
    root, _ = _tree(tmp_path)
    (root / "MANIFEST.sha256").unlink()
    assert fd.run(root, check_only=False) == 1


def test_directory_symlink_fails(tmp_path):
    root, _ = _tree(tmp_path)
    real = tmp_path / "real_src2"
    real.mkdir()
    (root / "src2").symlink_to(real)
    assert fd.run(root, check_only=False) == 1


def test_zero_json_tree_fails(tmp_path):
    """'rsync lost everything' must not read as 'nothing to check'."""
    root = tmp_path / "detections"
    (root / "src1").mkdir(parents=True)
    (root / "MANIFEST.sha256").write_text("")
    assert fd.run(root, check_only=True) == 1


def test_need_missing_file_fails(tmp_path):
    root, _ = _tree(tmp_path)
    assert fd.run(root, check_only=False) == 0
    assert fd.run(root, check_only=True,
                  need=("src1/a.json",)) == 0
    assert fd.run(root, check_only=True,
                  need=("src1/absent.json",)) == 1


def test_broken_symlink_fails(tmp_path):
    root, d = _tree(tmp_path)
    (d / "b.json").unlink()
    (d / "b.json").symlink_to(tmp_path / "gone.json")
    assert fd.run(root, check_only=False) == 1


def test_materialize_tmp_leftover_fails(tmp_path):
    root, d = _tree(tmp_path)
    assert fd.run(root, check_only=False) == 0
    (d / "x.json.materialize.tmp.123").write_text("junk")
    assert fd.run(root, check_only=True) == 1


def test_missing_provenance_is_a_failure(tmp_path):
    root, d = _tree(tmp_path)
    (d / "PROVENANCE.md").unlink()
    assert fd.run(root, check_only=False) == 1

"""The rerank symmetry gate must fire on *inflation* and stay quiet on deflation.

An acceptance gate nobody tested is a liability. Both directions are pinned:
the 2026-07-30 skew (flips too high) fails; a healthy post-fix dump with
flips lower than unflipped (smoke 2026-07-31, ratio ~0.17) passes with WARN.
"""
import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_rerank_symmetry.py"

_spec = importlib.util.spec_from_file_location("check_rerank_symmetry", SCRIPT)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

R_EYE = " ".join(["1", "0", "0", "0", "1", "0", "0", "0", "1"])
# 180 deg about z — well past the 5 deg "moved" cut.
R_FLIP = " ".join(["-1", "0", "0", "0", "-1", "0", "0", "0", "1"])


def _dump(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "im_id", "obj_id", "s_icp", "R", "R_prererank"])
        for s_icp, moved in rows:
            w.writerow([1, 1, 1, f"{s_icp:.6f}", R_EYE,
                        R_FLIP if moved else R_EYE])
    return str(path)


def _run(path, *extra):
    return subprocess.run([sys.executable, str(SCRIPT), path, *extra],
                          capture_output=True, text=True)


def test_gate_passes_when_both_groups_agree(tmp_path):
    rng = np.random.default_rng(0)
    rows = [(float(v), True) for v in rng.normal(0.30, 0.02, 200)]
    rows += [(float(v), False) for v in rng.normal(0.30, 0.02, 200)]
    r = _run(_dump(tmp_path / "ok.csv", rows))
    assert r.returncode == 0, r.stdout
    assert "PASS" in r.stdout


def test_gate_fires_on_the_2026_07_30_skew(tmp_path):
    """Flipped 0.86 vs unflipped 0.21 — the ratio the real dumps showed."""
    rng = np.random.default_rng(1)
    rows = [(float(v), True) for v in rng.normal(0.86, 0.02, 200)]
    rows += [(float(v), False) for v in rng.normal(0.21, 0.02, 200)]
    r = _run(_dump(tmp_path / "bad.csv", rows))
    assert r.returncode == 1, r.stdout
    assert "inflated" in r.stdout


def test_gate_passes_with_warn_on_post_fix_deflation(tmp_path):
    """Smoke 2026-07-31 shape: flipped median ~0.045, unflipped ~0.27 → 0.17x."""
    rng = np.random.default_rng(2)
    rows = [(float(v), True) for v in rng.normal(0.045, 0.005, 200)]
    rows += [(float(v), False) for v in rng.normal(0.270, 0.02, 200)]
    r = _run(_dump(tmp_path / "deflate.csv", rows))
    assert r.returncode == 0, r.stdout
    assert "PASS" in r.stdout
    assert "WARN" in r.stdout


def test_gate_refuses_to_judge_without_flips(tmp_path):
    """No flips means the gate proved nothing — that must not read as a pass."""
    rows = [(0.3, False)] * 50
    r = _run(_dump(tmp_path / "noflip.csv", rows))
    assert r.returncode == 2, r.stdout
    assert "CANNOT JUDGE" in r.stdout


def test_gate_refuses_to_judge_without_the_prererank_column(tmp_path):
    p = tmp_path / "nocol.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "im_id", "obj_id", "s_icp", "R"])
        w.writerow([1, 1, 1, "0.3", R_EYE])
    r = _run(str(p))
    assert r.returncode == 2, r.stdout
    assert "lacks" in r.stdout


@pytest.mark.parametrize("ratio,expected,need_warn", [
    (1.0, 0, False),   # parity
    (1.3, 0, False),   # mild, under hi
    (2.0, 1, False),   # inflation — FAIL
    (0.4, 0, True),    # deflation — PASS + WARN (was FAIL under bilateral)
    (0.17, 0, True),   # smoke 2026-07-31 shape
])
def test_gate_boundary_onesided(tmp_path, ratio, expected, need_warn):
    base = 0.30
    rows = [(base * ratio, True)] * 100 + [(base, False)] * 100
    r = _run(_dump(tmp_path / f"r{ratio}.csv", rows))
    assert r.returncode == expected, r.stdout
    if need_warn:
        assert "WARN" in r.stdout

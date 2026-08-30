"""Regression tests for the cache-hit canonicalisation frame (B1-F1).

A query cache HIT must hand the target side the scale the cached GeDi
features were BUILT at. Rebuilding it from the stored points is only valid
in the one configuration where that reproduces the historical convention
bit-for-bit (extent basis, no visibility gate): under
POPOE_CANON_BASIS=diameter the build scale is 1/hull-diameter, and under
POPOE_QUERY_MIN_VIEWS>0 the stored cloud is the filtered one while the
scale came from the full cloud. Serving a re-derived frame there silently
encodes every target at a mismatched scale (the faithful arms run
diameter + MIN_VIEWS=18).

Loads the example module by path (examples/ is not a package), same as
test_bop_eval_cli.py. Needs cv2 + pycocotools (the example's own imports).
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("pycocotools")

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "bop_eval.py"


@pytest.fixture(scope="module")
def bop_eval():
    spec = importlib.util.spec_from_file_location("bop_eval", _EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Extent of this cloud is 2.0 on every axis, so from_points() would say 0.5 —
# any test whose expected scale is not 0.5 cannot pass by accident.
PTS = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [1.0, 0.5, 1.5]],
               dtype=np.float32)


def test_recorded_scale_wins_over_stored_points(bop_eval):
    arrays = {"pts": PTS, "feats": np.zeros((3, 1)),
              "canon_scale": np.float64(0.123)}
    frame = bop_eval.restore_canon_frame(
        arrays, {"canon_basis": "diameter"}, PTS, obj_id=1)
    assert frame.scale == pytest.approx(0.123, abs=0)


def test_legacy_extent_ungated_falls_back_to_from_points(bop_eval):
    from popoe.interfaces import CanonFrame
    frame = bop_eval.restore_canon_frame({"pts": PTS}, {}, PTS, obj_id=1)
    assert frame.scale == CanonFrame.from_points(PTS).scale == 0.5


def test_legacy_diameter_entry_is_refused(bop_eval):
    with pytest.raises(SystemExit, match="canon_scale"):
        bop_eval.restore_canon_frame(
            {"pts": PTS}, {"canon_basis": "diameter"}, PTS, obj_id=7)


def test_legacy_visibility_gated_entry_is_refused(bop_eval):
    # Even in the extent basis: the stored cloud is post-filter, the scale
    # was pre-filter — from_points() is not the build-time value.
    with pytest.raises(SystemExit, match="canon_scale"):
        bop_eval.restore_canon_frame(
            {"pts": PTS}, {"query_min_views": "18"}, PTS, obj_id=7)


def test_roundtrip_through_cache_is_exact(bop_eval, tmp_path):
    from popoe.cache import StageCache
    cache = StageCache(str(tmp_path))
    scale = 1.0 / 0.07734567890123456   # not representable at float32
    cache.put_arrays("query", "k", pts=PTS, feats=np.zeros((3, 1)),
                     canon_scale=np.float64(scale))
    arrays = cache.get_arrays("query", "k")
    frame = bop_eval.restore_canon_frame(
        arrays, {"canon_basis": "diameter"}, arrays["pts"], obj_id=3)
    assert frame.scale == scale

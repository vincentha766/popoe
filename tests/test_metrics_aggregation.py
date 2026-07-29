"""metrics.aggregate — BOP official flat (per-instance) AR aggregation.

The defining property of the flat calibre: every annotated instance carries
equal weight, so it must equal the instance-count-weighted mean of per-object
recalls — NOT the equal-weight-per-object mean popoe used to report.
"""
from pathlib import Path

import numpy as np
import pytest

from popoe.metrics import aggregate
from popoe.metrics.aggregate import (
    MSSD_THRS,
    MSPD_THRS,
    VSD_THS,
    flat_ar_mssd,
    flat_ar_mspd,
    flat_ar_vsd,
)


def test_flat_mssd_weights_instances_not_objects():
    # obj A: 1 instance, fails every threshold; obj B: 9 instances, passes all.
    errs = np.array([1e9] + [0.0] * 9)
    diams = np.full(10, 100.0)
    # Flat: 9/10 instances correct at every threshold.
    assert flat_ar_mssd(errs, diams) == pytest.approx(0.9)
    # The old per-object calibre would have said 0.5 — assert we moved off it.
    assert flat_ar_mssd(errs, diams) != pytest.approx(0.5)


def test_flat_mspd_weights_instances_not_objects():
    errs = np.array([0.0, 0.0, 0.0, 1e9])  # 3 perfect rows, 1 hopeless row
    assert flat_ar_mspd(errs) == pytest.approx(0.75)


def test_flat_mssd_equals_instance_weighted_per_object_mean():
    rng = np.random.default_rng(0)
    diam_by_obj = {1: 80.0, 2: 150.0, 3: 260.0}
    n_by_obj = {1: 3, 2: 40, 3: 117}  # deliberately unbalanced
    errs, diams, per_obj = [], [], {}
    for o, n in n_by_obj.items():
        e = rng.uniform(0, 0.6 * diam_by_obj[o], size=n)
        errs.append(e)
        diams.append(np.full(n, diam_by_obj[o]))
        per_obj[o] = np.mean([(e < th * diam_by_obj[o]).mean()
                              for th in MSSD_THRS])
    flat = flat_ar_mssd(np.concatenate(errs), np.concatenate(diams))
    n_total = sum(n_by_obj.values())
    weighted = sum(per_obj[o] * n_by_obj[o] for o in per_obj) / n_total
    assert flat == pytest.approx(weighted, abs=1e-12)


def test_flat_mspd_equals_instance_weighted_per_object_mean():
    rng = np.random.default_rng(1)
    groups = {1: rng.uniform(0, 60, size=5), 2: rng.uniform(0, 60, size=200)}
    per_obj = {o: np.mean([(e < th).mean() for th in MSPD_THRS])
               for o, e in groups.items()}
    flat = flat_ar_mspd(np.concatenate(list(groups.values())))
    n_total = sum(len(e) for e in groups.values())
    weighted = sum(per_obj[o] * len(groups[o]) for o in groups) / n_total
    assert flat == pytest.approx(weighted, abs=1e-12)


def test_flat_vsd_hand_case():
    # 2 rows, 2 taus: one row perfect (err 0), one hopeless (err 1).
    # Every (tau, th) cell has recall 0.5, so the grid mean is 0.5.
    arr = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert flat_ar_vsd(arr) == pytest.approx(0.5)


def test_flat_vsd_weights_instances_not_objects():
    # obj A: 1 row all-fail; obj B: 9 rows all-pass -> flat 0.9 (per-object
    # calibre would have said 0.5).
    n_taus = len(VSD_THS)
    arr = np.vstack([np.ones((1, n_taus)), np.zeros((9, n_taus))])
    assert flat_ar_vsd(arr) == pytest.approx(0.9)


def test_flat_vsd_equals_instance_weighted_per_object_mean():
    rng = np.random.default_rng(2)
    groups = {5: rng.uniform(0, 1, size=(4, 10)),
              7: rng.uniform(0, 1, size=(96, 10))}
    per_obj = {}
    for o, arr in groups.items():
        per_obj[o] = np.mean([[(arr[:, i] < th).mean() for th in VSD_THS]
                              for i in range(arr.shape[1])])
    flat = flat_ar_vsd(np.concatenate(list(groups.values()), axis=0))
    n_total = sum(len(a) for a in groups.values())
    weighted = sum(per_obj[o] * len(groups[o]) for o in groups) / n_total
    assert flat == pytest.approx(weighted, abs=1e-12)


def test_shape_validation():
    with pytest.raises(ValueError):
        flat_ar_mssd(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError):
        flat_ar_vsd(np.zeros(5))  # 1-D: needs (N, n_taus)


# --- N3 / M1 / M3 guards (2026-07-29) -----------------------------------------

def test_mspd_normalize_factor_identity_at_ref_width():
    assert aggregate.mspd_normalize_factor(640) == pytest.approx(1.0)
    assert aggregate.mspd_normalize_factor(640.0) == pytest.approx(1.0)


def test_mspd_normalize_factor_scales_with_width():
    # T-LESS 720: factor 640/720; ITODD 1280: 0.5
    assert aggregate.mspd_normalize_factor(720) == pytest.approx(640 / 720)
    assert aggregate.mspd_normalize_factor(1280) == pytest.approx(0.5)
    # Normalising a raw error then comparing to thr 5..50 equals
    # comparing the raw error to thr * (W/640).
    raw_err = 10.0
    w = 1280.0
    thr = 5.0
    assert (raw_err * aggregate.mspd_normalize_factor(w) < thr) == (
        raw_err < thr * (w / 640.0)
    )


def test_mspd_normalize_factor_rejects_nonpositive():
    with pytest.raises(ValueError):
        aggregate.mspd_normalize_factor(0)
    with pytest.raises(ValueError):
        aggregate.mspd_normalize_factor(-1)


def test_vsd_delta_mm_itodd_is_5():
    assert aggregate.vsd_delta_mm("itodd") == 5.0
    assert aggregate.vsd_delta_mm("ITODD") == 5.0
    assert aggregate.vsd_delta_mm("itoddmv") == 5.0


def test_vsd_delta_mm_default_15():
    for ds in ("lmo", "ycbv", "tless", "hb", "icbin", "tudl"):
        assert aggregate.vsd_delta_mm(ds) == 15.0
    assert aggregate.vsd_delta_mm("unknown_dataset") == 15.0


def test_assert_csv_covers_targets_ok(tmp_path):
    targets = [
        {"scene_id": 1, "im_id": 2, "obj_id": 3},
        {"scene_id": 1, "im_id": 2, "obj_id": 4},
    ]
    p = tmp_path / "test_targets_bop19.json"
    p.write_text(__import__("json").dumps(targets))
    # exact cover
    aggregate.assert_csv_covers_targets(
        [(1, 2, 3), (1, 2, 4)], p)
    # extra CSV keys are fine
    aggregate.assert_csv_covers_targets(
        [(1, 2, 3), (1, 2, 4), (9, 9, 9)], p)


def test_assert_csv_covers_targets_missing_raises(tmp_path):
    targets = [
        {"scene_id": 1, "im_id": 2, "obj_id": 3},
        {"scene_id": 1, "im_id": 2, "obj_id": 4},
    ]
    p = tmp_path / "test_targets_bop19.json"
    p.write_text(__import__("json").dumps(targets))
    with pytest.raises(SystemExit, match="instance-rows missing"):
        aggregate.assert_csv_covers_targets([(1, 2, 3)], p)


def test_assert_csv_covers_targets_inst_count(tmp_path):
    # One key, inst_count=3 → need 3 CSV rows for that key.
    targets = [
        {"scene_id": 2, "im_id": 3, "obj_id": 9, "inst_count": 3},
    ]
    p = tmp_path / "test_targets_bop19.json"
    p.write_text(__import__("json").dumps(targets))
    with pytest.raises(SystemExit, match="instance-rows missing"):
        aggregate.assert_csv_covers_targets([(2, 3, 9)], p)  # only 1
    aggregate.assert_csv_covers_targets(
        [(2, 3, 9), (2, 3, 9), (2, 3, 9)], p)


def test_assert_csv_covers_targets_repeated_rows(tmp_path):
    # Two identical target rows without inst_count → need 2 CSV rows.
    targets = [
        {"scene_id": 2, "im_id": 3, "obj_id": 14},
        {"scene_id": 2, "im_id": 3, "obj_id": 14},
    ]
    p = tmp_path / "test_targets_bop19.json"
    p.write_text(__import__("json").dumps(targets))
    with pytest.raises(SystemExit, match="instance-rows missing"):
        aggregate.assert_csv_covers_targets([(2, 3, 14)], p)
    aggregate.assert_csv_covers_targets([(2, 3, 14), (2, 3, 14)], p)


def test_assert_csv_covers_targets_skips_missing_file(tmp_path):
    # No targets file → no-op (synthetic / unit tests).
    aggregate.assert_csv_covers_targets(
        [(1, 2, 3)], tmp_path / "does_not_exist.json")


def test_m3_ar_py_headline_uses_flat_ar_mssd():
    """M3: ar.py's primary AR assignment must call aggregate.flat_ar_*."""
    src = Path(__file__).resolve().parents[1] / "src" / "popoe" / "metrics" / "ar.py"
    text = src.read_text()
    assert "AR_MSSD = aggregate.flat_ar_mssd(" in text
    assert "AR_MSPD = aggregate.flat_ar_mspd(" in text
    # Must not reintroduce a primary equal-weight mean assignment.
    assert "AR_MSSD = np.mean([v[0] for v in per_obj.values()])" not in text


def test_m3_vsd_py_headline_uses_flat_ar_vsd():
    src = Path(__file__).resolve().parents[1] / "src" / "popoe" / "metrics" / "vsd.py"
    text = src.read_text()
    assert "AR_VSD = aggregate.flat_ar_vsd(" in text
    assert "AR_VSD = float(np.mean([v[0] for v in per_obj_ar.values()]))" not in text

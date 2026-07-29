"""metrics.aggregate — BOP official flat (per-instance) AR aggregation.

The defining property of the flat calibre: every annotated instance carries
equal weight, so it must equal the instance-count-weighted mean of per-object
recalls — NOT the equal-weight-per-object mean popoe used to report.
"""
import numpy as np
import pytest

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

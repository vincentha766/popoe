import numpy as np
import pytest

from popoe.interfaces import ObjectModel, Scene
from popoe.segmentor import SegmentorUnavailable
from popoe.segmentor_cnos_lab import (
    CNOSLabSegmentor,
    DepthSizeGate,
    DiameterSizeModel,
    PatchForegroundScorer,
    select_by_nearest_diameter,
    select_by_soft_affinity,
)


def _scene():
    K = np.array([[100.0, 0.0, 20.0],
                  [0.0, 100.0, 20.0],
                  [0.0, 0.0, 1.0]])
    return Scene(rgb=np.zeros((40, 40, 3), np.uint8),
                 depth=np.ones((40, 40), np.float32),
                 K=K, scene_id=0, im_id=1)


def _mask(y0=10, y1=30, x0=10, x1=30):
    m = np.zeros((40, 40), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_depth_size_gate_keeps_plausible_extent_and_rejects_wrong_size():
    scene = _scene()
    gate = DepthSizeGate(min_pixels=1, min_points=10)
    mask = _mask()
    extent = gate.extent_3d(mask, scene.depth, scene.K)

    ok, same_extent = gate.accepts(scene, ObjectModel(1, "x", diameter=0.25), mask)
    too_small, _ = gate.accepts(scene, ObjectModel(1, "x", diameter=0.05), mask)

    assert extent is not None
    assert same_extent == pytest.approx(extent)
    assert ok is True
    assert too_small is False


def test_patch_foreground_scorer_uses_foreground_and_topk():
    scorer = PatchForegroundScorer(topk=2)
    fg = np.zeros((224, 224), bool)
    fg[:14, :14] = True
    keep = scorer.foreground_patch_mask(fg)
    assert keep.shape == (16 * 16,)
    assert keep.sum() >= 3  # tiny fg falls back to all patches

    q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    t = np.array([[1.0, 0.0]])
    assert scorer.score_tokens(q, t) == pytest.approx(1.0)


class _Proposer:
    def __init__(self, masks):
        self._masks = masks

    def propose(self, scene):
        return self._masks


class _Bank:
    scorer = None

    def patches_for(self, obj):
        return np.array([[1.0, 0.0]])


class _Gate:
    def accepts(self, scene, obj, mask):
        return True, float(mask.sum())


class _Scorer:
    def score_mask(self, rgb, mask, template_tokens):
        return float(mask.sum())


def test_cnos_lab_segmentor_sorts_and_stamps_source():
    small = _mask(1, 8, 1, 8)
    large = _mask(10, 30, 10, 30)
    seg = CNOSLabSegmentor(
        proposer=_Proposer([small, large]),
        template_bank=_Bank(),
        scorer=_Scorer(),
        size_gate=_Gate(),
        n_masks=1,
    )

    dets = seg.segment(_scene(), ObjectModel(9, "x", diameter=0.25))

    assert len(dets) == 1
    assert dets[0].source == "cnos-lab"
    assert dets[0].score == pytest.approx(float(large.sum()))
    assert dets[0].bbox == (10.0, 10.0, 30.0, 30.0)


def test_cnos_lab_reuses_template_bank_scorer_when_not_explicit():
    class BankWithScorer(_Bank):
        scorer = _Scorer()

    large = _mask(10, 30, 10, 30)
    seg = CNOSLabSegmentor(
        proposer=_Proposer([large]),
        template_bank=BankWithScorer(),
        size_gate=_Gate(),
    )

    dets = seg.segment(_scene(), ObjectModel(9, "x", diameter=0.25))

    assert len(dets) == 1
    assert dets[0].score == pytest.approx(float(large.sum()))


def test_cnos_lab_requires_explicit_heavy_components():
    with pytest.raises(SegmentorUnavailable, match="mask proposer"):
        CNOSLabSegmentor().segment(_scene(), ObjectModel(9, "x", diameter=0.25))


def test_diameter_size_model_prefers_matching_extent():
    model = DiameterSizeModel()
    d19, d20 = 0.175, 0.217
    # Small mask should prefer 19; large mask prefer 20.
    assert model.nearest_obj_id(0.16, {19: d19, 20: d20}) == 19
    assert model.nearest_obj_id(0.23, {19: d19, 20: d20}) == 20
    # Soft affinity ranks the correct diameter higher.
    assert model.affinity(0.16, d19) > model.affinity(0.16, d20)
    assert model.affinity(0.23, d20) > model.affinity(0.23, d19)
    # Perfect match → affinity 1; large mismatch → near 0.
    assert model.affinity(d19, d19) == pytest.approx(1.0)
    assert model.affinity(0.05, d19) < 0.05


def test_select_by_soft_affinity_breaks_appearance_swap():
    # Appearance prefers the large confuser (index 0); size prefers small (1).
    # Raw Gaussian affinity is not always enough against a large score gap —
    # competitive (softmax) share vs the partner diameter is.
    appearances = [0.90, 0.55, 0.20]
    extents = [0.23, 0.16, None]
    d19, d20 = 0.175, 0.217
    pick = select_by_soft_affinity(
        appearances, extents, d19, rival_diameters=[d20]
    )
    assert pick is not None
    assert pick.index == 1
    # Missing extent alone should not win when affinity is zeroed.
    pick2 = select_by_soft_affinity(
        [0.99, 0.50], [None, 0.16], d19, missing_extent_affinity=0.0,
        rival_diameters=[d20],
    )
    assert pick2 is not None
    assert pick2.index == 1


def test_select_by_nearest_diameter_assigns_and_fallback():
    diameters = {19: 0.175, 20: 0.217}
    appearances = [0.90, 0.55]
    extents = [0.23, 0.16]
    # Query 19 → keep only nearest-to-19 mask (index 1).
    pick = select_by_nearest_diameter(
        appearances, extents, 19, diameters, fallback_appearance=False
    )
    assert pick is not None
    assert pick.index == 1
    assert pick.assigned_obj_id == 19
    # Query 20 → large mask.
    pick20 = select_by_nearest_diameter(
        appearances, extents, 20, diameters, fallback_appearance=False
    )
    assert pick20 is not None
    assert pick20.index == 0
    # No matching extent → fallback to appearance top-1.
    pick_fb = select_by_nearest_diameter(
        [0.8, 0.2], [None, None], 19, diameters, fallback_appearance=True
    )
    assert pick_fb is not None
    assert pick_fb.index == 0
    assert pick_fb.assigned_obj_id is None
    # Strict empty.
    assert select_by_nearest_diameter(
        [0.8], [None], 19, diameters, fallback_appearance=False
    ) is None

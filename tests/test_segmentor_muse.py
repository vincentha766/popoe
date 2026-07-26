import json

import numpy as np
import pytest

from popoe.interfaces import ObjectModel, Scene
from popoe.segmentor import SegmentorUnavailable
from popoe.segmentor_cnos_v3 import DepthSizeGate
from popoe.segmentor_muse import (
    DESCRIPTOR_FIELDS,
    MUSE_SOURCE,
    MuseClass,
    MuseSegmentor,
    absolute_score,
    final_scores,
    gem_pool,
    joint_scores,
    muse_records,
    relative_scores,
    tanimoto,
    write_muse_detections,
)


# ── scoring core (pure numpy) ───────────────────────────────────────────

def test_gem_pool_clamps_negative_tokens_instead_of_producing_nan():
    tokens = np.array([[-3.0, 4.0], [1.0, 2.0]])

    out = gem_pool(tokens, p=1.5)

    assert out.shape == (2,)
    assert np.all(np.isfinite(out)), "a fractional power of a negative token is NaN"
    # p=1 is the plain mean, so the clamp is the only difference from averaging
    assert gem_pool(np.array([[1.0, 3.0], [3.0, 5.0]]), p=1.0) == pytest.approx([2.0, 4.0])


def test_gem_pool_rejects_empty_and_wrong_rank():
    with pytest.raises(ValueError):
        gem_pool(np.zeros((0, 4)))
    with pytest.raises(ValueError):
        gem_pool(np.zeros(4))


def test_tanimoto_is_one_for_identical_and_symmetric():
    a = np.array([0.4, 1.2, 0.0])
    b = np.array([0.1, 0.9, 0.3])

    assert tanimoto(a, a) == pytest.approx(1.0, abs=1e-6)
    assert tanimoto(a, b) == pytest.approx(tanimoto(b, a))
    assert tanimoto(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0, abs=1e-6)


def test_absolute_score_takes_the_best_template_view():
    cls_q, gem_q = np.array([1.0, 0.0]), np.array([1.0, 0.0])
    cls_bank = np.array([[0.0, 1.0], [1.0, 0.0]])      # 2nd view matches
    gem_bank = np.array([[0.0, 1.0], [1.0, 0.0]])

    s = absolute_score(cls_q, gem_q, cls_bank, gem_bank, alpha=0.5)

    assert s == pytest.approx(1.0, abs=1e-6)


def test_absolute_score_weights_class_and_patch_terms_by_alpha():
    cls_q, gem_q = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    cls_bank, gem_bank = np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]])

    # cos = 1, tanimoto ~ 0 -> the score IS alpha
    assert absolute_score(cls_q, gem_q, cls_bank, gem_bank, alpha=0.25) == pytest.approx(0.25, abs=1e-6)
    assert absolute_score(cls_q, gem_q, cls_bank, gem_bank, alpha=0.75) == pytest.approx(0.75, abs=1e-6)


def test_absolute_score_rejects_mismatched_bank():
    with pytest.raises(ValueError, match="template bank mismatch"):
        absolute_score(np.array([1.0, 0.0]), np.array([1.0, 0.0]),
                       np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[1.0, 0.0]]))


def test_relative_scores_normalise_per_proposal_and_sharpen_with_small_tau():
    s_abs = np.array([[0.7, 0.5], [0.2, 0.9]])

    warm = relative_scores(s_abs, tau=1.0)
    cold = relative_scores(s_abs, tau=0.02)

    assert warm.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert cold.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert cold[0, 0] > warm[0, 0]              # colder temperature = more decisive
    assert cold[0, 0] == pytest.approx(1.0, abs=1e-4)


def test_relative_scores_survive_a_row_where_every_class_failed():
    """A proposal whose crop failed is -inf in every column. The naive
    exp/sum softmax turns that row into NaN, which then poisons the ranking of
    EVERY class, not just that proposal."""
    s_abs = np.array([[-np.inf, -np.inf], [0.4, 0.2]])

    out = relative_scores(s_abs, tau=0.02)

    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx([0.5, 0.5])


def test_relative_scores_validates_shape_and_temperature():
    with pytest.raises(ValueError):
        relative_scores(np.zeros(3), tau=0.02)
    with pytest.raises(ValueError):
        relative_scores(np.zeros((2, 2)), tau=0.0)


def test_joint_and_final_score_formulas():
    s_abs = np.array([[0.6, 0.2]])
    s_rel = np.array([[1.0, 0.0]])

    joint = joint_scores(s_abs, s_rel, beta=0.8)
    final = final_scores(joint, np.array([0.5]), gamma=1.0)

    assert joint == pytest.approx(np.array([[0.8 * 0.6 + 0.2, 0.8 * 0.2]]))
    assert final == pytest.approx(0.5 * joint)


# ── the live segmentor, driven by fakes (no GPU, no network) ────────────

def _scene(fill: int = 0) -> Scene:
    K = np.array([[100.0, 0.0, 20.0],
                  [0.0, 100.0, 20.0],
                  [0.0, 0.0, 1.0]])
    return Scene(rgb=np.full((40, 40, 3), fill, np.uint8),
                 depth=np.ones((40, 40), np.float32),
                 K=K, scene_id=7, im_id=3)


def _mask(y0, y1, x0, x1) -> np.ndarray:
    m = np.zeros((40, 40), bool)
    m[y0:y1, x0:x1] = True
    return m


MASK_A = _mask(4, 24, 4, 24)      # matches class 9
MASK_B = _mask(16, 36, 16, 36)    # matches class 14


class _Proposer:
    """Records how often the (expensive) proposal stage actually ran."""

    def __init__(self, n=2, threshold=0.15):
        self.calls = 0
        self.n = n
        self.threshold = threshold

    def config(self):
        """Declared identity — keeps the call counter out of the cache key."""
        return {"n": self.n, "threshold": self.threshold}

    def propose(self, scene):
        self.calls += 1
        boxes = np.array([[4.0, 4.0, 24.0, 24.0], [16.0, 16.0, 36.0, 36.0]])[: self.n]
        scores = np.array([0.9, 0.5])[: self.n]
        return boxes, scores


class _Refiner:
    def __init__(self, masks=(MASK_A, MASK_B)):
        self.masks = list(masks)

    def masks_for_boxes(self, rgb, boxes):
        return [(m, 0.95) for m in self.masks[: len(boxes)]]


class _Embedder:
    """Deterministic per-proposal embeddings, in call order."""

    def __init__(self):
        self.calls = 0

    def config(self):
        return {}          # fixed behaviour; the counter must not enter the key

    def embed(self, rgb_crop, fg_mask):
        pairs = [(np.array([1.0, 0.0]), np.array([1.0, 0.01])),
                 (np.array([0.0, 1.0]), np.array([0.01, 1.0]))]
        out = pairs[self.calls % len(pairs)]
        self.calls += 1
        return out


class _Bank:
    """Class 9 looks like the first proposal, class 14 like the second."""

    BANKS = {
        9: (np.array([[1.0, 0.0]]), np.array([[1.0, 0.01]])),
        14: (np.array([[0.0, 1.0]]), np.array([[0.01, 1.0]])),
    }

    def embeddings_for(self, obj):
        return self.BANKS[obj.obj_id]


def _obj(obj_id, diameter=0.3):
    return ObjectModel(obj_id, f"obj_{obj_id:06d}.ply", diameter=diameter)


def _classes(diameters=(0.3, 0.3)):
    return [MuseClass(_obj(9, diameters[0]), "/tpl/9"),
            MuseClass(_obj(14, diameters[1]), "/tpl/14")]


def _segmentor(classes=None, proposer=None, refiner=None, size_gate=None, **kw):
    return MuseSegmentor(
        classes or _classes(),
        proposer=proposer or _Proposer(),
        refiner=refiner or _Refiner(),
        embedder=_Embedder(),
        template_bank=_Bank(),
        size_gate=size_gate or DepthSizeGate(min_pixels=100, min_points=10),
        **kw,
    )


def test_single_class_registration_is_refused_unless_asked_for():
    one = [MuseClass(_obj(9), "/tpl/9")]

    with pytest.raises(ValueError, match="relative score"):
        MuseSegmentor(one)

    seg = MuseSegmentor(one, allow_single_class=True)
    assert len(seg.classes) == 1


def test_duplicate_and_empty_class_lists_are_refused():
    with pytest.raises(ValueError, match="duplicate obj_id"):
        MuseSegmentor([MuseClass(_obj(9), "/a"), MuseClass(_obj(9), "/b")])
    with pytest.raises(ValueError, match="at least one"):
        MuseSegmentor([])


def test_each_class_gets_its_own_winner_from_the_shared_proposal_pool():
    """The point of scoring classes jointly: one proposal pool, per-class
    ranking. Class 9 must win MASK_A and class 14 MASK_B."""
    seg = _segmentor()
    scene = _scene()

    top9 = seg.segment(scene, _obj(9))[0]
    top14 = seg.segment(scene, _obj(14))[0]

    assert np.array_equal(top9.mask, MASK_A)
    assert np.array_equal(top14.mask, MASK_B)
    assert top9.score > seg.segment(scene, _obj(9))[1].score


def test_detection_carries_source_bbox_and_score_breakdown():
    seg = _segmentor()

    det = seg.segment(_scene(), _obj(9))[0]

    assert det.source == MUSE_SOURCE
    assert det.bbox == (4.0, 4.0, 24.0, 24.0)
    assert len(det.descriptor) == len(DESCRIPTOR_FIELDS)
    s_abs, s_rel, p_obj, extent = det.descriptor
    assert s_abs == pytest.approx(1.0, abs=1e-5)     # exact template match
    assert s_rel == pytest.approx(1.0, abs=1e-4)     # wins the cross-class softmax
    assert p_obj == pytest.approx(0.9, abs=1e-6)     # objectness carried through
    assert 0.0 < extent < 0.4


def test_proposal_stage_runs_once_per_frame_not_once_per_object():
    proposer = _Proposer()
    seg = _segmentor(proposer=proposer)
    scene = _scene()

    seg.segment(scene, _obj(9))
    seg.segment(scene, _obj(14))
    seg.segment(scene, _obj(9))

    assert proposer.calls == 1


def test_a_different_frame_is_recomputed_even_with_the_same_ids():
    """Frames are keyed by CONTENT: real captures often reuse scene_id/im_id
    (or leave them at -1), and reusing masks across frames is the stale-mask
    bug this guards against."""
    proposer = _Proposer()
    seg = _segmentor(proposer=proposer)

    seg.segment(_scene(fill=0), _obj(9))
    seg.segment(_scene(fill=17), _obj(9))

    assert proposer.calls == 2


def test_changing_a_component_setting_invalidates_the_cached_frame():
    """The stale case the frame key exists to prevent: raise/lower a proposal
    threshold on a live segmentor and the same frame must be recomputed, not
    served from the memo."""
    proposer = _Proposer()
    seg = _segmentor(proposer=proposer)
    scene = _scene()

    seg.segment(scene, _obj(9))
    proposer.threshold = 0.05
    seg.segment(scene, _obj(9))

    assert proposer.calls == 2


def test_config_identity_covers_the_inputs_that_change_output():
    a = _segmentor()
    diameters = _segmentor(classes=[MuseClass(_obj(9, 0.9), "/tpl/9"),
                                    MuseClass(_obj(14, 0.3), "/tpl/14")])
    gate = _segmentor(size_gate=DepthSizeGate(min_pixels=100, min_points=11))

    # class diameters drive the size gate, so they are part of the identity
    assert a.config()["classes"] != diameters.config()["classes"]
    # every gate field, not just the three the first version listed
    assert a.config()["size_gate"] != gate.config()["size_gate"]
    assert {"min_points", "depth_band_m", "min_depth_m", "percentiles"} <= set(
        a.config()["size_gate"])
    # and each component's settings
    assert a.config()["proposer"]["threshold"] == pytest.approx(0.15)


def test_unregistered_object_is_a_loud_error():
    seg = _segmentor()

    with pytest.raises(ValueError, match="not registered"):
        seg.segment(_scene(), _obj(21))


def test_missing_heavy_components_raise_segmentor_unavailable():
    seg = MuseSegmentor(_classes())

    with pytest.raises(SegmentorUnavailable, match="Grounding DINO"):
        seg.segment(_scene(), _obj(9))


def test_size_gate_accepts_a_mask_plausible_for_any_registered_class():
    """The gate runs BEFORE matching, so it must use the union of the classes'
    intervals — otherwise the second class's candidates are gone before it is
    ever scored."""
    scene = _scene()
    gate = DepthSizeGate(min_pixels=10, min_points=10)
    tiny = _mask(10, 15, 10, 15)
    extent = gate.extent_3d(tiny, scene.depth, scene.K)
    small_d = extent / 0.5          # interval [0.5, 2.2] x extent -> accepts
    big_d = extent * 10             # interval [2.5, 11] x extent -> rejects

    def _seg(diameters):
        return _segmentor(
            classes=[MuseClass(_obj(9, diameters[0]), "/tpl/9"),
                     MuseClass(_obj(14, diameters[1]), "/tpl/14")],
            proposer=_Proposer(n=1), refiner=_Refiner(masks=[tiny]), size_gate=gate)

    # implausible for class 9, plausible for class 14 -> the union keeps it
    assert _seg((big_d, small_d)).scene_result(scene).n_proposals == 1
    # implausible for every registered class -> dropped before matching
    assert _seg((big_d, big_d)).scene_result(scene).n_proposals == 0


def test_size_gate_union_is_not_the_hull_of_the_class_intervals():
    """With diameters spanning more than the 4.4x ratio band there is a GAP
    between the per-class intervals. `[min(lo), max(hi)]` would swallow it and
    admit proposals plausible for no registered class at all."""
    scene = _scene()
    gate = DepthSizeGate(min_pixels=10, min_points=10)
    tiny = _mask(10, 15, 10, 15)
    extent = gate.extent_3d(tiny, scene.depth, scene.K)
    below = extent / 2.0            # interval [0.125, 0.55] x extent -> too big for it
    above = extent * 8.0            # interval [2.0, 8.8] x extent -> too small for it

    seg = _segmentor(classes=[MuseClass(_obj(9, below), "/tpl/9"),
                              MuseClass(_obj(14, above), "/tpl/14")],
                     proposer=_Proposer(n=1), refiner=_Refiner(masks=[tiny]),
                     size_gate=gate)

    assert seg.scene_result(scene).n_proposals == 0


# ── the producer half: dumped masks reload as an ordinary source ─────────

def test_records_round_trip_through_the_production_detections_loader(tmp_path):
    pytest.importorskip("pycocotools")
    from popoe.segmentor_detections import decode_detection_mask, load_bop_detections

    seg = _segmentor()
    scene = _scene()
    live = {c.obj.obj_id: seg.segment(scene, c.obj)[0].mask for c in seg.classes}

    out = str(tmp_path / "muse.json")
    write_muse_detections(muse_records(scene, seg, n_masks=1), out)
    loaded = load_bop_detections(out)

    assert len(loaded) == 2
    for rec in loaded:
        assert rec["scene_id"] == 7 and rec["image_id"] == 3
        assert rec["source"] == MUSE_SOURCE
        assert np.array_equal(decode_detection_mask(rec["segmentation"]),
                              live[rec["category_id"]]), "dumped mask must be pixel-identical"

    raw = json.loads(open(out).read())
    assert raw[0]["bbox"] == [4.0, 4.0, 20.0, 20.0]          # COCO xywh, not xyxy
    assert set(raw[0]["muse"]) == set(DESCRIPTOR_FIELDS)


def test_the_reserved_official_source_name_cannot_be_written(tmp_path):
    """`muse` names artefacts from the official method, which publishes none.
    Documentation alone would not stop a `source=` keyword doing it by accident."""
    from popoe.segmentor_muse import MuseDetectionsSegmentor

    seg = _segmentor()

    with pytest.raises(ValueError, match="reserved"):
        muse_records(_scene(), seg, source="muse")
    with pytest.raises(ValueError, match="reserved"):
        muse_records(_scene(), seg, source="MUSE")

    out = str(tmp_path / "muse.json")
    write_muse_detections(muse_records(_scene(), seg, n_masks=1), out)
    with pytest.raises(ValueError, match="reserved"):
        MuseDetectionsSegmentor(out, source="muse")


def test_out_of_memory_is_a_runtime_failure_not_unavailability():
    """A fallback chain may route around a missing backend; routing around a
    CUDA OOM would bury it under a weaker method's results."""
    from popoe.segmentor_muse import _is_runtime_failure

    class OutOfMemoryError(RuntimeError):    # torch.cuda.OutOfMemoryError's shape
        pass

    assert _is_runtime_failure(OutOfMemoryError("CUDA out of memory"))
    assert _is_runtime_failure(MemoryError())
    assert not _is_runtime_failure(OSError("checkpoint not found"))
    assert not _is_runtime_failure(ImportError("no module named transformers"))


def test_cli_class_spec_reads_diameters_in_metres(tmp_path):
    from popoe.segmentor_muse import _classes_from_args, _parse_classes

    info = tmp_path / "models_info.json"
    info.write_text(json.dumps({"9": {"diameter": 130.0}, "14": {"diameter": 125.0}}))

    classes = _classes_from_args(_parse_classes("9=/tpl/9,14=/tpl/14"),
                                 str(info), models_dir="/models")

    assert [c.obj.obj_id for c in classes] == [9, 14]
    assert classes[0].obj.diameter == pytest.approx(0.130)   # BOP mm -> metres
    assert classes[0].obj.mesh_path == "/models/obj_000009.ply"
    assert classes[1].template_dir == "/tpl/14"

    with pytest.raises(ValueError, match="obj_id=template_dir"):
        _parse_classes("9:/tpl/9")


def test_dumped_detections_serve_the_file_backed_segmentor(tmp_path):
    pytest.importorskip("pycocotools")
    from popoe.segmentor_muse import MuseDetectionsSegmentor

    seg = _segmentor()
    scene = _scene()
    out = str(tmp_path / "muse.json")
    write_muse_detections(muse_records(scene, seg, n_masks=1), out)

    replay = MuseDetectionsSegmentor(out, topk=2, min_pixels=10)
    dets = replay.segment(scene, _obj(9))

    assert len(dets) == 1
    assert dets[0].source == MUSE_SOURCE
    assert np.array_equal(dets[0].mask, MASK_A)

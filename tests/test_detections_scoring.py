"""Framework-layer tests for BOPDetectionsSegmentor pooling and ChampionScorer
arbitration — numpy-only (no GPU, no heavy models)."""

import json

import numpy as np
import pytest

from popoe.interfaces import ObjectModel, PointFeatures, PoseHypothesis, Scene


def _rle(mask):
    from pycocotools import mask as cm
    r = cm.encode(np.asfortranarray(mask.astype(np.uint8)))
    r["counts"] = r["counts"].decode()
    return {"size": list(mask.shape), "counts": r["counts"]}


def _det(scene, im, cat, score, mask, source=None):
    d = {"scene_id": scene, "image_id": im, "category_id": cat,
         "score": score, "segmentation": _rle(mask)}
    if source:
        d["source"] = source
    return d


@pytest.fixture
def masks():
    a = np.zeros((48, 64), bool); a[5:25, 5:25] = True
    b = np.zeros((48, 64), bool); b[25:45, 30:60] = True
    c = np.zeros((48, 64), bool); c[6:26, 5:25] = True     # ~dup of a
    return a, b, c


def _scene():
    return Scene(rgb=np.zeros((48, 64, 3), np.uint8),
                 depth=np.zeros((48, 64), np.float32),
                 K=np.eye(3), scene_id=1, im_id=7)


def _obj(oid):
    return ObjectModel(obj_id=oid, mesh_path="x.ply", diameter=0.1)


def test_pooling_merges_partner_label(tmp_path, masks):
    a, b, _ = masks
    dets = [_det(1, 7, 19, 0.9, a), _det(1, 7, 20, 0.8, b)]
    p = tmp_path / "d.json"; p.write_text(json.dumps(dets))
    from popoe.segmentor_detections import BOPDetectionsSegmentor

    plain = BOPDetectionsSegmentor(str(p), topk=2)
    assert len(plain.segment(_scene(), _obj(19))) == 1        # own label only

    pooled = BOPDetectionsSegmentor(
        str(p), topk=2, merge_labels={19: [19, 20], 20: [19, 20]})
    assert len(pooled.segment(_scene(), _obj(19))) == 2       # partner pooled
    assert len(pooled.segment(_scene(), _obj(20))) == 2
    # non-merged object unaffected
    assert pooled.segment(_scene(), _obj(3)) == []


def test_topk_per_bucket_and_dedupe(tmp_path, masks):
    a, b, c = masks
    dets = [_det(1, 7, 19, 0.9, a), _det(1, 7, 19, 0.85, c),   # c ~dup of a
            _det(1, 7, 19, 0.5, b)]
    p = tmp_path / "d.json"; p.write_text(json.dumps(dets))
    from popoe.segmentor_detections import BOPDetectionsSegmentor
    seg = BOPDetectionsSegmentor(str(p), topk=2, iou_dedupe=0.8)
    out = seg.segment(_scene(), _obj(19))
    # topk=2 keeps a + c by score; c dropped as duplicate -> only a survives
    assert len(out) == 1 and out[0].score == pytest.approx(0.9)


def _pf(pts, feats):
    return PointFeatures(pts=pts, feats=feats)


def test_champion_scorer_size_aware_prefers_true_size():
    """Same shape registered perfectly in a scale-blind sense: the size-aware
    term must separate right-size from a 25%-scaled clone of the target."""
    from popoe.scoring import ChampionScorer
    rng = np.random.default_rng(0)
    pts_q = rng.uniform(-0.05, 0.05, (400, 3))               # 10 cm object
    feats = rng.standard_normal((400, 8)).astype(np.float64)
    pose = PoseHypothesis(R=np.eye(3), t=np.zeros(3), score=0.0,
                          breakdown={"s_icp": 1.0})

    right = ChampionScorer(size_aware=True).score(
        pose, _pf(pts_q, feats), _pf(pts_q.copy(), feats.copy()))
    wrong = ChampionScorer(size_aware=True).score(
        pose, _pf(pts_q, feats), _pf(pts_q * 1.25, feats.copy()))
    assert right.breakdown["metric_fit"] > 0.9
    assert wrong.breakdown["metric_fit"] < 0.6
    assert right.score > wrong.score


def test_champion_scorer_uses_w1_feats_from_meta():
    """s_feat_1 must be computed in the w=1 space even when .feats are
    weight-scaled (the cross-weight arbitration bug the rule fixes)."""
    from popoe.scoring import ChampionScorer
    rng = np.random.default_rng(1)
    pts = rng.uniform(-0.05, 0.05, (200, 3))
    f1 = rng.standard_normal((200, 8))
    scaled = f1.copy(); scaled[:, :4] *= 0.2                 # some other weight
    q = PointFeatures(pts=pts, feats=scaled, meta={"feats_w1": f1})
    t = PointFeatures(pts=pts.copy(), feats=scaled.copy(), meta={"feats_w1": f1.copy()})
    pose = PoseHypothesis(R=np.eye(3), t=np.zeros(3), score=0.0,
                          breakdown={"s_icp": 1.0})
    out = ChampionScorer(size_aware=False).score(pose, q, t)
    # identical clouds + identical w=1 feats -> cosine ~1 regardless of scaling
    assert out.breakdown["s_feat_1"] == pytest.approx(1.0, abs=1e-6)

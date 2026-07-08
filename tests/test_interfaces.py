"""Contract + Pipeline orchestration tests — GPU-free, mock stages only."""
import numpy as np

import popoe
from popoe import (
    Scene, ObjectModel, CanonFrame, Detection, PointFeatures, PoseHypothesis, Pipeline,
)


def test_canon_frame_from_points():
    pts = np.random.default_rng(0).standard_normal((100, 3)).astype(np.float32) * 0.05
    cf = CanonFrame.from_points(pts)
    assert np.isclose(cf.scale, 1.0 / max(float(np.ptp(pts, axis=0).max()), 1e-6))
    assert np.allclose(cf.center, 0.0)


class _Seg:
    def segment(self, scene, obj):
        return [Detection(np.ones((4, 4), bool), 0.9), Detection(np.ones((4, 4), bool), 0.4)]


class _QEnc:
    def __init__(self): self.calls = 0
    def encode_query(self, obj):
        self.calls += 1
        pts = np.zeros((6, 3), np.float32)
        return PointFeatures(pts, np.ones((6, 4), np.float32),
                             meta={"canon_frame": CanonFrame(np.zeros(3), 3.0)})


class _TEnc:
    def __init__(self): self.frames = []
    def encode_target(self, scene, det, obj, frame):
        self.frames.append(frame.scale)
        return PointFeatures(np.zeros((6, 3), np.float32), np.ones((6, 4), np.float32))


class _Solver:
    def solve(self, q, t, frame):
        return [PoseHypothesis(np.eye(3), np.zeros(3), 0.8, breakdown={"s_coarse": 0.8})]


class _RefinerGeom:
    def refine(self, pose, scene, obj, q, t):
        assert q is not None  # refiner gets query (ICP geometry)
        return PoseHypothesis(pose.R, pose.t, pose.score,
                              breakdown={**pose.breakdown, "s_icp": 0.5})


class _Scorer:
    def score(self, pose, q, t):
        b = pose.breakdown
        return PoseHypothesis(pose.R, pose.t, b["s_coarse"] * 0.6 * b["s_icp"],
                              breakdown={**b, "s_fine": 0.6})


class _Selector:
    def select(self, cands):
        cands = [c for c in cands if c is not None]
        return max(cands, key=lambda h: h.score) if cands else None


def _mocks_satisfy_protocols():
    assert isinstance(_Seg(), popoe.Segmentor)
    assert isinstance(_QEnc(), popoe.QueryEncoder)
    assert isinstance(_TEnc(), popoe.TargetEncoder)
    assert isinstance(_Solver(), popoe.PoseSolver)
    assert isinstance(_RefinerGeom(), popoe.PoseRefiner)
    assert isinstance(_Scorer(), popoe.PoseScorer)
    assert isinstance(_Selector(), popoe.Selector)


def test_mocks_satisfy_protocols():
    _mocks_satisfy_protocols()


def test_pipeline_run_orchestration():
    q = _QEnc()
    pipe = Pipeline(segmentor=_Seg(), query_encoder=q, target_encoder=_TEnc(),
                    solver=_Solver(), refiners=[_RefinerGeom()], selector=_Selector(),
                    scorer=_Scorer(), topk=2)
    scene = Scene(np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), np.float32), np.eye(3))
    obj = ObjectModel(5, "x.ply", 0.1)

    best = pipe.run(scene, obj)
    assert best is not None
    assert np.isclose(best.score, 0.8 * 0.6 * 0.5)          # scorer applied after refine
    assert set(("s_coarse", "s_icp", "s_fine")) <= set(best.breakdown)
    pipe.run(scene, obj)
    assert q.calls == 1                                     # query cached across runs
    assert np.isclose(pipe.target_encoder.frames[0], 3.0)  # query frame reused on target

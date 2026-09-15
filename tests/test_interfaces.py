"""Contract + PoseMethod orchestration tests — GPU-free, mock stages only."""
import numpy as np
import pytest

import popoe
from popoe import (
    Scene, ObjectModel, CanonFrame, Detection, PointFeatures, PoseHypothesis,
    Pipeline, DirectPoseMethod, correspond_pair,
)


def test_canon_frame_from_points():
    pts = np.random.default_rng(0).standard_normal((100, 3)).astype(np.float32) * 0.05
    cf = CanonFrame.from_points(pts)
    assert np.isclose(cf.scale, 1.0 / max(float(np.ptp(pts, axis=0).max()), 1e-6))


class _Seg:
    def segment(self, scene, obj):
        return [Detection(np.ones((4, 4), bool), 0.9), Detection(np.ones((4, 4), bool), 0.4)]


class _QEnc:
    def __init__(self): self.calls = 0
    def encode_query(self, obj):
        self.calls += 1
        pts = np.zeros((6, 3), np.float32)
        return PointFeatures(pts, np.ones((6, 4), np.float32),
                             meta={"canon_frame": CanonFrame(3.0)})


class _TEnc:
    def __init__(self): self.frames = []
    def encode_target(self, scene, det, obj, frame):
        self.frames.append(frame.scale)
        return PointFeatures(np.zeros((6, 3), np.float32), np.ones((6, 4), np.float32))


class _Solver:
    def solve(self, q, t, frame=None):
        return [PoseHypothesis(np.eye(3), np.zeros(3), 0.8, breakdown={"s_coarse": 0.8})]


class _Coarse:
    def estimate(self, scene, obj, det=None):
        return [PoseHypothesis(np.eye(3), np.zeros(3), 0.7, breakdown={"source": "mock"})]


class _RefinerGeom:
    def refine(self, pose, scene, obj, q, t):
        assert q is not None
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


def test_mocks_satisfy_protocols():
    assert isinstance(_Seg(), popoe.Segmentor)
    assert isinstance(_QEnc(), popoe.QueryEncoder)
    assert isinstance(_TEnc(), popoe.TargetEncoder)
    assert isinstance(_Solver(), popoe.PoseSolver)
    assert isinstance(_Coarse(), popoe.CoarseEstimator)
    assert isinstance(_RefinerGeom(), popoe.PoseRefiner)
    assert isinstance(_Scorer(), popoe.PoseScorer)
    assert isinstance(_Selector(), popoe.Selector)
    from popoe.adapters import ICPRefiner
    icp = ICPRefiner()
    assert isinstance(icp, popoe.PoseRefiner)
    assert isinstance(icp, popoe.GeometricRefiner)
    pipe = Pipeline(segmentor=_Seg(), query_encoder=_QEnc(),
                    target_encoder=_TEnc(), solver=_Solver(),
                    refiners=[_RefinerGeom()], selector=_Selector())
    assert isinstance(pipe, popoe.PoseMethod)
    assert popoe.CorrespondencePipeline is Pipeline
    direct = DirectPoseMethod(estimator=_Coarse(), selector=_Selector())
    assert isinstance(direct, popoe.PoseMethod)


def test_pipeline_run_orchestration():
    q = _QEnc()
    pipe = Pipeline(segmentor=_Seg(), query_encoder=q, target_encoder=_TEnc(),
                    solver=_Solver(), refiners=[_RefinerGeom()], selector=_Selector(),
                    scorer=_Scorer(), topk=2)
    scene = Scene(np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), np.float32), np.eye(3))
    obj = ObjectModel(5, "x.ply", 0.1)

    best = pipe.run(scene, obj)
    assert best is not None
    assert np.isclose(best.score, 0.8 * 0.6 * 0.5)
    assert set(("s_coarse", "s_icp", "s_fine")) <= set(best.breakdown)
    pipe.run(scene, obj)
    assert q.calls == 1
    assert np.isclose(pipe.target_encoder.frames[0], 3.0)


def test_correspond_pair_matches_pipeline_run():
    """Pipeline.run is encode + correspond_pair + select, not a second kernel."""
    qenc, tenc = _QEnc(), _TEnc()
    pipe = Pipeline(segmentor=_Seg(), query_encoder=qenc, target_encoder=tenc,
                    solver=_Solver(), refiners=[_RefinerGeom()],
                    selector=_Selector(), scorer=_Scorer(), topk=2)
    scene = Scene(np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), np.float32),
                  np.eye(3))
    obj = ObjectModel(5, "x.ply", 0.1)
    best = pipe.run(scene, obj)

    q = qenc.encode_query(obj)
    frame = q.meta["canon_frame"]
    cands = []
    for det in _Seg().segment(scene, obj)[:2]:
        t = tenc.encode_target(scene, det, obj, frame)
        cands.extend(correspond_pair(
            q, t, scene, obj, pipe.solver, pipe.refiners, pipe.scorer, frame))
    replay = _Selector().select(cands)
    assert replay is not None and best is not None
    assert np.isclose(replay.score, best.score)
    assert np.allclose(replay.R, best.R)


class _QEncPca:
    def __init__(self): self.calls = 0
    def encode_query(self, obj):
        self.calls += 1
        return PointFeatures(np.zeros((6, 3), np.float32), np.ones((6, 4), np.float32),
                             meta={"canon_frame": CanonFrame(3.0),
                                   "pca_vis": f"pca-{obj.obj_id}"})


class _TEncPca:
    def __init__(self): self.installed = None; self.used = []
    def install_pca(self, pca_vis):
        if pca_vis is None:
            raise ValueError("install_pca(None) must be refused")
        self.installed = pca_vis
    def encode_target(self, scene, det, obj, frame):
        self.used.append((obj.obj_id, self.installed))
        return PointFeatures(np.zeros((6, 3), np.float32), np.ones((6, 4), np.float32))


def _pca_pipe(tenc, qenc):
    return Pipeline(segmentor=_Seg(), query_encoder=qenc, target_encoder=tenc,
                    solver=_Solver(), refiners=[_RefinerGeom()], selector=_Selector(),
                    scorer=_Scorer(), topk=2)


def test_pipeline_reinstalls_query_pca_on_cached_runs():
    tenc, qenc = _TEncPca(), _QEncPca()
    scene = Scene(np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), np.float32), np.eye(3))
    a, b = ObjectModel(1, "a.ply", 0.1), ObjectModel(9, "b.ply", 0.1)
    pa, pb = _pca_pipe(tenc, qenc), _pca_pipe(tenc, qenc)

    for _ in range(2):
        pa.run(scene, a)
        pb.run(scene, b)

    assert qenc.calls == 2
    assert all(pca == f"pca-{oid}" for oid, pca in tenc.used), tenc.used


def test_pipeline_refuses_missing_pca_snapshot():
    tenc = _TEncPca()
    pipe = _pca_pipe(tenc, _QEnc())
    scene = Scene(np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), np.float32), np.eye(3))
    with pytest.raises(ValueError, match="pca_vis"):
        pipe.run(scene, ObjectModel(1, "a.ply", 0.1))


def test_direct_pose_method_selects_estimator_hypotheses():
    class _Est:
        def __init__(self):
            self.dets = []
        def estimate(self, scene, obj, det=None):
            self.dets.append(det)
            s = 0.4 if det is None else det.score
            return [PoseHypothesis(np.eye(3), np.zeros(3), s,
                                   breakdown={"source": "mock"})]

    est = _Est()
    scene = Scene(np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), np.float32),
                  np.eye(3))
    obj = ObjectModel(5, "x.ply", 0.1)

    best = DirectPoseMethod(estimator=est, selector=_Selector()).run(scene, obj)
    assert best is not None
    assert np.isclose(best.score, 0.4)
    assert est.dets == [None]

    best = DirectPoseMethod(estimator=est, selector=_Selector(),
                            segmentor=_Seg(), topk=2).run(scene, obj)
    assert best is not None
    assert np.isclose(best.score, 0.9)
    assert len(est.dets) == 3
    assert est.dets[1].score == 0.9


def test_direct_pose_method_runs_geometric_refiners():
    class _Est:
        def estimate(self, scene, obj, det=None):
            return [PoseHypothesis(np.eye(3), np.zeros(3), 0.4,
                                   breakdown={"source": "mock"})]

    class _Geom:
        def __init__(self):
            self.seen = []

        def refine_geometry(self, pose, scene, obj, pts_src, pts_tgt):
            self.seen.append((pts_src.shape, pts_tgt.shape))
            return PoseHypothesis(pose.R, pose.t + np.array([0.01, 0, 0]),
                                  pose.score, breakdown={**pose.breakdown,
                                                         "s_icp": 0.5})

    geom = _Geom()
    src = np.zeros((5, 3), np.float32)
    tgt = np.ones((8, 3), np.float32)

    def clouds(scene, obj, det):
        del scene, obj, det
        return src, tgt

    scene = Scene(np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), np.float32),
                  np.eye(3))
    obj = ObjectModel(5, "x.ply", 0.1)
    with pytest.raises(ValueError, match="clouds"):
        DirectPoseMethod(estimator=_Est(), selector=_Selector(),
                         geometric_refiners=[geom]).run(scene, obj)

    hyp = DirectPoseMethod(
        estimator=_Est(), selector=_Selector(),
        geometric_refiners=[geom], clouds=clouds,
    ).run(scene, obj)
    assert hyp is not None
    assert np.allclose(hyp.t, [0.01, 0, 0])
    assert hyp.breakdown["s_icp"] == 0.5
    assert geom.seen == [(src.shape, tgt.shape)]
    assert isinstance(geom, popoe.GeometricRefiner)


def test_freeze_package_exports():
    import popoe.freeze as fz
    for name in fz.__all__:
        assert getattr(fz, name) is not None, name

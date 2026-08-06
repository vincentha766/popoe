"""CLI-level tests for examples/bop_eval.py source routing: the mutually
exclusive --detections / --sources knobs resolve to the right segmentor, and
the per-(source, label) top-K semantics survive a multi-source union.

Loads the example module by path (examples/ is not a package). Needs cv2 +
pycocotools (the example's own imports); numpy otherwise. No GPU — only the
lightweight arg-routing helper is exercised, never main().
"""
import importlib.util
import json
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


def _rle(mask):
    from pycocotools import mask as cm
    r = cm.encode(np.asfortranarray(mask.astype(np.uint8)))
    r["counts"] = r["counts"].decode()
    return {"size": list(mask.shape), "counts": r["counts"]}


def _mask(r0, c0):
    m = np.zeros((48, 64), bool)
    m[r0:r0 + 15, c0:c0 + 15] = True
    return m


def _file(tmp_path, name, dets):
    tmp_path.mkdir(parents=True, exist_ok=True)
    recs = [{"scene_id": 1, "image_id": 7, "category_id": cat,
             "score": sc, "segmentation": _rle(mask)} for cat, sc, mask in dets]
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(recs))
    return str(p)


# ── routing: exactly one of --detections / --sources ─────────────────────

def test_neither_source_errors(bop_eval):
    with pytest.raises(SystemExit, match="exactly one"):
        bop_eval.resolve_segmentor(None, "", topk=2, merge_labels=None)


def test_both_sources_error(bop_eval, tmp_path):
    p = _file(tmp_path, "d", [(5, 0.9, _mask(2, 2))])
    with pytest.raises(SystemExit, match="exactly one"):
        bop_eval.resolve_segmentor(p, f"nids={p}", topk=2, merge_labels=None)


def test_empty_sources_string_errors(bop_eval):
    with pytest.raises(SystemExit, match="empty"):
        bop_eval.resolve_segmentor(None, "  ,  ", topk=2, merge_labels=None)


def test_single_detections_file(bop_eval, tmp_path):
    p = _file(tmp_path, "d", [(5, 0.9, _mask(2, 2))])
    seg = bop_eval.resolve_segmentor(p, "", topk=2, merge_labels=None)
    assert [s.name for s in seg.sources] == ["bop-detections"]


def test_single_detections_known_path_gets_source_tag(bop_eval, tmp_path):
    p = _file(tmp_path / "cnos", "cnos-fastsam_lmo-test",
              [(5, 0.9, _mask(2, 2))])
    seg = bop_eval.resolve_segmentor(p, "", topk=2, merge_labels=None)
    assert [s.name for s in seg.sources] == ["cnos"]


def test_sources_list_builds_named_union(bop_eval, tmp_path):
    a = _file(tmp_path, "a", [(5, 0.9, _mask(2, 2))])
    b = _file(tmp_path, "b", [(5, 0.8, _mask(2, 40))])
    seg = bop_eval.resolve_segmentor(None, f"cnos={a},nids={b}",
                                     topk=2, merge_labels=None)
    assert [s.name for s in seg.sources] == ["cnos", "nids"]


# ── the max-inst topk floor ──────────────────────────────────────────────

def test_cand_csv_header_s_coarse_and_solver_columns(bop_eval):
    """The header always ends with a `solver` column; --score-coarse inserts the
    coarse block (s_coarse and the pre-ICP pose it was measured at) just before
    it. The pose columns share the switch because they share the mechanism —
    ICPRefiner(keep_coarse=True) is what produces both."""
    off = bop_eval.cand_csv_header(False)
    on = bop_eval.cand_csv_header(True)
    assert off[-4:] == ["solver", "source", "R_prererank", "t_prererank"]
    assert "s_coarse" not in off
    assert "R_coarse" not in off and "t_coarse" not in off
    assert on[-4:] == ["solver", "source", "R_prererank", "t_prererank"]
    assert on == (off[:-4]
                  + ["s_coarse", "R_coarse", "t_coarse"]
                  + ["solver", "source", "R_prererank", "t_prererank"])


def test_cand_csv_legacy_headers_are_compatible(bop_eval):
    base = ["scene_id", "im_id", "obj_id", "cand", "w", "s_icp",
            "s_feat_1", "metric_fit", "score", "R", "t"]
    assert base in bop_eval.cand_csv_compatible_headers(False)
    assert base + ["solver"] in bop_eval.cand_csv_compatible_headers(False)
    assert base + ["s_coarse"] in bop_eval.cand_csv_compatible_headers(True)
    assert (base + ["s_coarse", "R_coarse", "t_coarse", "solver"]
            in bop_eval.cand_csv_compatible_headers(True))


def test_cand_csv_header_s_feat_w_is_appended_last(bop_eval):
    """--score-feat-w appends at the END, so every pre-existing column keeps its
    index and position-addressing readers still line up."""
    off = bop_eval.cand_csv_header(False)
    on = bop_eval.cand_csv_header(False, True)
    assert on == off + ["s_feat_w"]
    assert bop_eval.cand_csv_header(True, True) == \
        bop_eval.cand_csv_header(True) + ["s_feat_w"]


def test_cand_csv_s_feat_w_refuses_legacy_headers(bop_eval):
    """Appending an s_feat_w run to a dump without that column would silently
    drop the one number the run exists to produce — so only the exact header is
    accepted."""
    assert bop_eval.cand_csv_compatible_headers(False, True) == \
        [bop_eval.cand_csv_header(False, True)]
    legacy = ["scene_id", "im_id", "obj_id", "cand", "w", "s_icp",
              "s_feat_1", "metric_fit", "score", "R", "t"]
    assert legacy not in bop_eval.cand_csv_compatible_headers(False, True)


def test_cand_csv_row_s_feat_w_is_loud_when_the_scorer_did_not_record_it(bop_eval):
    """Flag on but no s_feat_w in the breakdown = misconfigured scorer. Fail,
    rather than writing 0.0000 into a column that will be read as evidence."""
    header = bop_eval.cand_csv_header(False, True)
    with pytest.raises(KeyError):
        bop_eval.cand_csv_row(1, 7, 5, 0, 1.0, _hyp(breakdown={"s_icp": 0.5}),
                              "", "o3d", False, header, True)
    row = bop_eval.cand_csv_row(
        1, 7, 5, 0, 1.0, _hyp(breakdown={"s_icp": 0.5, "s_feat_w": 0.4242}),
        "", "o3d", False, header, True)
    assert dict(zip(header, row))["s_feat_w"] == "0.4242"


def test_missing_target_encoder_is_explicit_not_attributeerror(bop_eval):
    """Regression for the removed cache-only probe path: a target cache miss
    with no live encoder used to call None.install_pca()."""
    with pytest.raises(RuntimeError, match="target cache miss"):
        bop_eval.require_target_encoder(None, "target cache miss")


def test_floored_topk(bop_eval):
    """The floor lifts topk to at least max_inst (so a k-instance target can get
    k champions), but never LOWERS a larger user topk."""
    assert bop_eval.floored_topk(2, 1) == 2      # single-instance: unchanged
    assert bop_eval.floored_topk(2, 4) == 4      # 4-instance target: lifted
    assert bop_eval.floored_topk(6, 4) == 6      # user asked more: kept


def _scene():
    from popoe.interfaces import Scene
    return Scene(rgb=np.zeros((48, 64, 3), np.uint8),
                 depth=np.zeros((48, 64), np.float32), K=np.eye(3),
                 scene_id=1, im_id=7)


def _obj():
    from popoe.interfaces import ObjectModel
    return ObjectModel(obj_id=5, mesh_path="x", diameter=0.1)


def _hyp(R=None, t=None, breakdown=None, score=0.123456):
    from popoe.interfaces import PoseHypothesis
    return PoseHypothesis(
        np.eye(3) if R is None else R,
        np.zeros(3) if t is None else t,
        score,
        {} if breakdown is None else breakdown,
    )


def _row_dict(bop_eval, hyp, source="", score_coarse=False):
    header = bop_eval.cand_csv_header(score_coarse)
    row = bop_eval.cand_csv_row(1, 7, 5, 0, 1.0, hyp, source, "o3d",
                                score_coarse, header)
    return dict(zip(header, row))


def test_cand_csv_row_writes_single_and_multi_source_values(bop_eval, tmp_path):
    single = _file(tmp_path / "cnos", "cnos-fastsam_lmo-test",
                   [(5, 0.9, _mask(2, 2))])
    seg = bop_eval.resolve_segmentor(single, "", topk=2, merge_labels=None)
    det = seg.segment(_scene(), _obj())[0]
    assert _row_dict(bop_eval, _hyp(), det.source)["source"] == "cnos"

    a = _file(tmp_path, "a", [(5, 0.9, _mask(2, 2))])
    b = _file(tmp_path, "b", [(5, 0.8, _mask(2, 40))])
    seg = bop_eval.resolve_segmentor(None, f"cnos={a},nids={b}",
                                     topk=2, merge_labels=None)
    by_source = {d.source: _row_dict(bop_eval, _hyp(), d.source)["source"]
                 for d in seg.segment(_scene(), _obj())}
    assert by_source == {"cnos": "cnos", "nids": "nids"}


def test_cand_csv_prererank_columns_blank_without_breakdown(bop_eval):
    row = _row_dict(bop_eval, _hyp(), "cnos")
    assert row["R_prererank"] == ""
    assert row["t_prererank"] == ""


def test_cand_csv_prererank_columns_written_from_breakdown(bop_eval):
    pre_R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    pre_t = np.array([0.001, 0.002, 0.003])
    post_t = np.array([0.004, 0.005, 0.006])
    h = _hyp(t=post_t, breakdown={"R_prererank": pre_R,
                                  "t_prererank": pre_t})
    row = _row_dict(bop_eval, h, "cnos")
    assert row["R"] == (
        "1.000000 0.000000 0.000000 0.000000 1.000000 0.000000 "
        "0.000000 0.000000 1.000000"
    )
    assert row["t"] == "4.0000 5.0000 6.0000"
    assert row["R_prererank"] == (
        "0.000000 -1.000000 0.000000 1.000000 0.000000 0.000000 "
        "0.000000 0.000000 1.000000"
    )
    assert row["t_prererank"] == "1.0000 2.0000 3.0000"


def test_topk_is_per_source_in_union(bop_eval, tmp_path):
    """With --sources, `topk` caps EACH source's bucket, not a shared global
    pool: two sources with 3 distinct masks each at topk=2 yield 2 per source."""
    from collections import Counter
    from popoe.interfaces import ObjectModel
    a = _file(tmp_path, "a", [(5, 0.9, _mask(0, 0)), (5, 0.8, _mask(0, 20)),
                              (5, 0.7, _mask(0, 40))])
    b = _file(tmp_path, "b", [(5, 0.6, _mask(20, 0)), (5, 0.5, _mask(20, 20)),
                              (5, 0.4, _mask(20, 40))])
    seg = bop_eval.resolve_segmentor(None, f"cnos={a},nids={b}",
                                     topk=2, merge_labels=None)
    out = seg.segment(_scene(), ObjectModel(obj_id=5, mesh_path="x", diameter=0.1))
    assert Counter(d.source for d in out) == {"cnos": 2, "nids": 2}


def test_topk_is_per_source_AND_label_under_merge(bop_eval, tmp_path):
    """Bucketing is per (source, LABEL), not per source: with label pooling
    (obj19 pools labels 19+20), one source contributes topk for EACH label
    (2+2=4), not topk shared across both (2). Distinguishes per-(source,label)
    from per-source bucketing — the semantics the CLI help promises."""
    from collections import Counter
    from popoe.interfaces import ObjectModel
    src = _file(tmp_path, "cnos", [
        (19, 0.9, _mask(0, 0)), (19, 0.8, _mask(0, 20)),      # label 19 x2 kept
        (20, 0.7, _mask(20, 0)), (20, 0.6, _mask(20, 20)),    # label 20 x2 kept
    ])
    seg = bop_eval.resolve_segmentor(None, f"cnos={src}", topk=2,
                                     merge_labels={19: [19, 20], 20: [19, 20]})
    out = seg.segment(_scene(), ObjectModel(obj_id=19, mesh_path="x", diameter=0.1))
    # 2 from label 19 + 2 from label 20, all one source -> 4 (not 2)
    assert Counter(d.source for d in out) == {"cnos": 4}


# ── Query cache entries are both files, or nothing ──────────────────────

def _stage_cache(tmp_path):
    from popoe.cache import StageCache
    return StageCache(str(tmp_path / "cache"))


def test_complete_query_entry_is_a_hit(bop_eval, tmp_path):
    from sklearn.decomposition import PCA
    cache = _stage_cache(tmp_path)
    pca = PCA(n_components=2).fit(np.random.default_rng(0).standard_normal((30, 8)))
    cache.put_pickle("query", "k1", pca)
    cache.put_arrays("query", "k1", pts=np.zeros((5, 3)), feats=np.zeros((5, 4)))

    arrays, pca_hit = bop_eval.load_cached_query(cache, "k1")
    assert arrays["pts"].shape == (5, 3)
    assert pca_hit is not None


def test_absent_query_entry_is_a_clean_miss(bop_eval, tmp_path):
    assert bop_eval.load_cached_query(_stage_cache(tmp_path), "nope") is None
    assert bop_eval.load_cached_query(None, "k1") is None       # cache disabled


def test_orphan_sidecar_is_a_clean_miss(bop_eval, tmp_path):
    """The only state the current write order can leave behind on a crash:
    sidecar written, arrays not. Harmless — re-encode and overwrite."""
    from sklearn.decomposition import PCA
    cache = _stage_cache(tmp_path)
    pca = PCA(n_components=2).fit(np.random.default_rng(1).standard_normal((30, 8)))
    cache.put_pickle("query", "k2", pca)
    assert bop_eval.load_cached_query(cache, "k2") is None


def test_arrays_without_sidecar_is_fatal(bop_eval, tmp_path):
    """Regression (codex review round 2): re-encoding the query here is NOT a
    repair. Target entries are keyed by the same qkey, which config+mesh
    content alone determine, so they stay reachable — and any of them may hold
    a target-fitted basis written by the pre-fix code, which a cached entry
    does not record. Must stop the run, not quietly continue."""
    cache = _stage_cache(tmp_path)
    cache.put_arrays("query", "k3", pts=np.zeros((5, 3)), feats=np.zeros((5, 4)))

    with pytest.raises(SystemExit, match="INCOMPLETE"):
        bop_eval.load_cached_query(cache, "k3")


# ── dataset layout / merge resolution (the seven-set portability fixes) ──

def test_resolve_layout_from_bop_basename(bop_eval, tmp_path):
    name, lay = bop_eval.resolve_layout(tmp_path / "ycbv")
    assert (name, lay["split"], lay["models_dir"]) == ("ycbv", "test", "models")


def test_resolve_layout_tless_gets_primesense_and_cad(bop_eval, tmp_path):
    name, lay = bop_eval.resolve_layout(tmp_path / "tless")
    assert lay["split"] == "test_primesense"
    assert lay["models_dir"] == "models_cad"


def test_resolve_layout_unknown_basename_is_fatal(bop_eval, tmp_path):
    """A wrong layout does not crash downstream — it completes as an all-zero
    CSV — so the guess has to be refused up front."""
    with pytest.raises(SystemExit, match="--dataset"):
        bop_eval.resolve_layout(tmp_path / "bop_data_v2")


def test_resolve_layout_dataset_flag_beats_basename(bop_eval, tmp_path):
    name, lay = bop_eval.resolve_layout(tmp_path / "itodd_copy", dataset="itodd")
    assert name == "itodd"
    assert (lay["img_dir"], lay["img_ext"]) == ("gray", ".tif")


def test_resolve_layout_overrides_apply(bop_eval, tmp_path):
    _, lay = bop_eval.resolve_layout(tmp_path / "hb", split="test_kinect",
                                     models_dir="models_reconst")
    assert (lay["split"], lay["models_dir"]) == ("test_kinect", "models_reconst")


def test_merge_auto_pools_only_on_ycbv(bop_eval):
    """Obj ids 19/20 exist on tless/itodd/hb too; the former literal 'ycbv'
    default silently pooled two unrelated objects there AND gave them the
    clamp-specific size-aware scorer."""
    from popoe.freeze.recipes import YCBV_MERGE_LABELS
    assert bop_eval.resolve_merge("auto", "ycbv") == YCBV_MERGE_LABELS
    for ds in ("tless", "itodd", "hb", "lmo", "tudl", "icbin"):
        assert bop_eval.resolve_merge("auto", ds) == {}


def test_merge_explicit_values_survive_any_dataset(bop_eval):
    from popoe.freeze.recipes import YCBV_MERGE_LABELS
    assert bop_eval.resolve_merge("ycbv", "tless") == YCBV_MERGE_LABELS
    assert bop_eval.resolve_merge("none", "ycbv") == {}
    assert bop_eval.resolve_merge("3:7", "tless") == {3: [3, 7], 7: [3, 7]}


# ── frame reading: fail loud, gray->3ch replication ──────────────────────

def _frame_dir(tmp_path, lay, im_id=3, gray=False):
    import cv2
    sdir = tmp_path / "000001"
    (sdir / "depth").mkdir(parents=True)
    (sdir / lay["img_dir"]).mkdir(exist_ok=True)
    cv2.imwrite(str(sdir / "depth" / f"{im_id:06d}{lay['depth_ext']}"),
                np.full((8, 8), 1000, np.uint16))
    img = np.full((8, 8), 128, np.uint8) if gray else \
        np.full((8, 8, 3), 128, np.uint8)
    cv2.imwrite(str(sdir / lay["img_dir"] / f"{im_id:06d}{lay['img_ext']}"), img)
    return sdir


def test_read_frame_images_rgb_png(bop_eval, tmp_path):
    from popoe.datasets.bop import bop_layout
    lay = bop_layout("ycbv")
    sdir = _frame_dir(tmp_path, lay)
    bgr, depth = bop_eval.read_frame_images(sdir, 3, lay)
    assert bgr.shape == (8, 8, 3)
    assert depth.dtype == np.uint16     # IMREAD_UNCHANGED keeps raw units


def test_read_frame_images_itodd_gray_tif_is_3ch(bop_eval, tmp_path):
    """Single-channel gray/*.tif must reach the visual branch replicated to
    3 channels (cv2's IMREAD_COLOR does this) — not crash, not stay 2-D."""
    from popoe.datasets.bop import bop_layout
    lay = bop_layout("itodd")
    sdir = _frame_dir(tmp_path, lay, gray=True)
    bgr, depth = bop_eval.read_frame_images(sdir, 3, lay)
    assert bgr.shape == (8, 8, 3)
    assert depth.dtype == np.uint16


def test_read_frame_images_names_the_missing_file(bop_eval, tmp_path):
    """The old code answered a missing image with inst_count zero rows and
    'done' — on itodd that meant a whole dataset of fabricated zeros. The
    reader now raises, naming the exact path, and the caller only writes
    zeros under an explicit --allow-missing-images."""
    from popoe.datasets.bop import bop_layout
    lay = bop_layout("ycbv")
    sdir = _frame_dir(tmp_path, lay)
    with pytest.raises(FileNotFoundError, match="000099.png"):
        bop_eval.read_frame_images(sdir, 99, lay)
    # depth missing (rgb present) names the depth file
    (sdir / "depth" / "000003.png").unlink()
    with pytest.raises(FileNotFoundError, match="depth"):
        bop_eval.read_frame_images(sdir, 3, lay)


def test_read_frame_images_16bit_gray_is_fatal(bop_eval, tmp_path):
    """IMREAD_COLOR would shift a 16-bit gray image to its top 8 bits and
    hand a near-black frame to the visual branch with no error anywhere.
    The gray path reads UNCHANGED and refuses non-uint8 instead."""
    import cv2
    from popoe.datasets.bop import bop_layout
    lay = bop_layout("itodd")
    sdir = tmp_path / "000001"
    (sdir / "depth").mkdir(parents=True)
    (sdir / "gray").mkdir()
    cv2.imwrite(str(sdir / "depth" / "000003.tif"),
                np.full((8, 8), 1000, np.uint16))
    cv2.imwrite(str(sdir / "gray" / "000003.tif"),
                np.full((8, 8), 30000, np.uint16))
    with pytest.raises(SystemExit, match="uint8"):
        bop_eval.read_frame_images(sdir, 3, lay)


def test_resolve_layout_blames_the_flag_that_was_passed(bop_eval, tmp_path):
    """A typo'd --dataset must not be answered with 'pass --dataset
    explicitly' — that hint belongs to the basename-inference path only."""
    with pytest.raises(SystemExit, match="--dataset 'tles' is not"):
        bop_eval.resolve_layout(tmp_path / "bop", dataset="tles")


# ── FreeZe fidelity fixes: dense ICP target, diameter-based tau ─────────────

def _icp_scene(bop_eval, depth, K=None):
    from popoe.interfaces import Scene
    K = np.array([[500., 0., 32.], [0., 500., 24.], [0., 0., 1.]]) if K is None else K
    return Scene(rgb=np.zeros(depth.shape + (3,), np.uint8), depth=depth, K=K,
                 scene_id=1, im_id=1)


def test_dense_mask_cloud_matches_the_extractor_construction(bop_eval):
    """--icp-dense must rebuild EXACTLY the cloud the feature extractor
    already computes for GeDi (`pcd_dense`): valid depth inside the mask,
    back-projected in metres. A different convention here (mm, or depth>0
    dropped) would silently register against a cloud in another frame."""
    depth = np.zeros((48, 64), np.float32)
    depth[10:14, 20:25] = 0.8
    depth[12, 22] = 0.0                     # a depth hole inside the mask
    mask = np.zeros((48, 64), bool)
    mask[10:14, 20:25] = True
    sc = _icp_scene(bop_eval, depth)
    got = bop_eval.dense_mask_cloud(sc, mask, max_pts=0)

    ys, xs = np.where((depth > 0) & mask)
    d = depth[ys, xs]
    want = np.stack([(xs - 32.) * d / 500., (ys - 24.) * d / 500., d], 1)
    assert got.shape == (19, 3)             # 20 masked pixels minus the hole
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)


def test_dense_mask_cloud_cap_is_deterministic_and_bounded(bop_eval):
    depth = np.full((48, 64), 0.9, np.float32)
    mask = np.ones((48, 64), bool)
    sc = _icp_scene(bop_eval, depth)
    a = bop_eval.dense_mask_cloud(sc, mask, max_pts=100)
    b = bop_eval.dense_mask_cloud(sc, mask, max_pts=100)
    assert len(a) == 100
    np.testing.assert_array_equal(a, b)     # same mask -> same cloud, always
    assert len(bop_eval.dense_mask_cloud(sc, mask, max_pts=0)) == 48 * 64


def test_dense_mask_cloud_declines_a_degenerate_mask(bop_eval):
    depth = np.zeros((48, 64), np.float32)
    depth[5, 5] = 0.7
    mask = np.zeros((48, 64), bool)
    mask[5, 5] = True
    assert bop_eval.dense_mask_cloud(_icp_scene(bop_eval, depth), mask, 0) is None


def test_tau_basis_switches_all_three_thresholds_together():
    """FreeZeV2 Sec. IV-A sets tau_inlier AND tau_ICP to 3% of the DIAMETER,
    and Eq. 5 scores over the Eq. 4 inlier set — one basis, three thresholds.
    Passing tau_basis_m must move all of them, and passing None must leave the
    historical extent-based values untouched."""
    from popoe.freeze.recipes import TAU_FRAC, stages_for_object
    extent, diam = 0.09, 0.12
    sol, ref, sco = stages_for_object(extent, tau_basis_m=diam)
    assert ref.tau_icp == pytest.approx(TAU_FRAC * diam)
    assert sco.tau_abs == pytest.approx(TAU_FRAC * diam)
    assert getattr(sol, "tau_inlier") == pytest.approx(TAU_FRAC * diam)

    sol0, ref0, sco0 = stages_for_object(extent)
    assert ref0.tau_icp == pytest.approx(TAU_FRAC * extent)
    assert sco0.tau_abs is None             # scorer recomputes it from extent
    assert getattr(sol0, "tau_inlier") == pytest.approx(TAU_FRAC * extent)


def test_scorer_tau_abs_overrides_the_extent_fraction():
    from popoe.scoring import ChampionScorer
    assert ChampionScorer().tau_abs is None
    assert ChampionScorer(tau_abs=0.004).tau_abs == pytest.approx(0.004)


def test_help_renders(tmp_path):
    """`--help` must actually print.

    argparse runs every help string through %-interpolation, so one literal
    "3%" in a help text is enough to make the whole CLI's --help raise
    `TypeError: %o format: an integer is required, not dict` — which is what
    happened on main until 2026-07-30. Nothing else catches it: the flag
    itself worked fine, only the help did not, so every run passed while the
    CLI was undiscoverable.

    Asserted as a subprocess because the parser is built inside main(); this
    tests the thing a user actually types.
    """
    import subprocess
    import sys
    r = subprocess.run([sys.executable, str(_EXAMPLE), "--help"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"--help failed:\n{r.stderr[-2000:]}"
    assert "--tau-diameter" in r.stdout          # the flag that carried the bug
    assert "3% of" in r.stdout                   # %% must render back as one %

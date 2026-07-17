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


def test_sources_list_builds_named_union(bop_eval, tmp_path):
    a = _file(tmp_path, "a", [(5, 0.9, _mask(2, 2))])
    b = _file(tmp_path, "b", [(5, 0.8, _mask(2, 40))])
    seg = bop_eval.resolve_segmentor(None, f"cnos={a},nids={b}",
                                     topk=2, merge_labels=None)
    assert [s.name for s in seg.sources] == ["cnos", "nids"]


# ── the max-inst topk floor ──────────────────────────────────────────────

def test_cand_csv_header_s_coarse_and_solver_columns(bop_eval):
    """The header always ends with a `solver` column; --score-coarse inserts an
    s_coarse column just before it."""
    off = bop_eval.cand_csv_header(False)
    on = bop_eval.cand_csv_header(True)
    assert off[-1] == "solver" and "s_coarse" not in off
    assert on[-1] == "solver" and on[-2] == "s_coarse"
    assert on == off[:-1] + ["s_coarse", "solver"]


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

"""Pluggable file-based detection backends: name a source, select it by name,
compose several into one segmentor, and keep per-source provenance on every
mask. Backward-compat with the single-file form is covered here and in
test_detections_load / test_detections_scoring.

pycocotools for RLE; numpy otherwise. No GPU.
"""

import json

import numpy as np
import pytest

pytest.importorskip("pycocotools")

from popoe.interfaces import ObjectModel, Scene
from popoe.segmentor_detections import (
    BOPDetectionsSegmentor, DetectionSource, _coerce_sources)


def _rle(mask):
    from pycocotools import mask as cm
    r = cm.encode(np.asfortranarray(mask.astype(np.uint8)))
    r["counts"] = r["counts"].decode()
    return {"size": list(mask.shape), "counts": r["counts"]}


def _det(cat, score, mask, scene=1, im=7):
    return {"scene_id": scene, "image_id": im, "category_id": cat,
            "score": score, "segmentation": _rle(mask)}


def _write(tmp_path, name, dets):
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(dets))
    return str(p)


def _scene():
    return Scene(rgb=np.zeros((48, 64, 3), np.uint8),
                 depth=np.zeros((48, 64), np.float32),
                 K=np.eye(3), scene_id=1, im_id=7)


def _obj(oid):
    return ObjectModel(obj_id=oid, mesh_path="x", diameter=0.1)


def _mask(r0, c0):
    m = np.zeros((48, 64), bool)
    m[r0:r0 + 15, c0:c0 + 15] = True         # 225 px, over min_pixels
    return m


# ── config parsing: select by name, several forms ────────────────────────

def test_coerce_sources_accepts_dict_tuples_strings():
    from_dict = _coerce_sources({"nids": "a.json", "cnos": "b.json"})
    assert from_dict == [DetectionSource("nids", "a.json"),
                         DetectionSource("cnos", "b.json")]
    from_tuples = _coerce_sources([("nids", "a.json"), DetectionSource("cnos", "b.json")])
    assert [s.name for s in from_tuples] == ["nids", "cnos"]
    from_strings = _coerce_sources(["nids=a.json", "cnos=b.json"])
    assert from_strings[0] == DetectionSource("nids", "a.json")
    # a path containing '=' still splits only on the first '='
    assert _coerce_sources(["nids=/p/a=b.json"])[0].path == "/p/a=b.json"


def test_coerce_sources_rejects_bad_and_duplicate():
    with pytest.raises(ValueError):
        _coerce_sources([])
    with pytest.raises(ValueError):
        _coerce_sources({})                  # empty config is a loud error, not a no-op
    with pytest.raises(ValueError, match="duplicate"):
        _coerce_sources([("nids", "a"), ("nids", "b")])
    with pytest.raises(TypeError):
        _coerce_sources([123])


def test_coerce_sources_rejects_empty_name():
    """An empty provenance name would silently blank Detection.source."""
    with pytest.raises(ValueError, match="non-empty"):
        _coerce_sources(["=file.json"])
    with pytest.raises(ValueError, match="non-empty"):
        _coerce_sources([DetectionSource("", "file.json")])


# ── single-file naming + arg exclusivity (back-compat) ───────────────────

def test_single_source_default_tag_is_unchanged(tmp_path):
    p = _write(tmp_path, "d", [_det(5, 0.9, _mask(2, 2))])
    seg = BOPDetectionsSegmentor(p, topk=2)
    dets = seg.segment(_scene(), _obj(5))
    assert [d.source for d in dets] == ["bop-detections"]


def test_single_source_named(tmp_path):
    p = _write(tmp_path, "d", [_det(5, 0.9, _mask(2, 2))])
    seg = BOPDetectionsSegmentor(p, topk=2, source="nids")
    assert [d.source for d in seg.segment(_scene(), _obj(5))] == ["nids"]


def test_single_source_overwrites_stray_record_source(tmp_path):
    """A single file whose records already carry a "source" field keeps the
    UNIFORM single-file tag (default 'bop-detections'), not the record's — the
    historical, documented invariant. Per-source provenance is the multi-source
    form's job."""
    d = _det(5, 0.9, _mask(2, 2)); d["source"] = "smuggled"
    p = _write(tmp_path, "d", [d])
    seg = BOPDetectionsSegmentor(p, topk=2)
    assert [x.source for x in seg.segment(_scene(), _obj(5))] == ["bop-detections"]


def test_empty_source_name_rejected(tmp_path):
    p = _write(tmp_path, "d", [_det(5, 0.9, _mask(2, 2))])
    with pytest.raises(ValueError, match="non-empty"):
        BOPDetectionsSegmentor(p, source="")


def test_arg_exclusivity(tmp_path):
    p = _write(tmp_path, "d", [_det(5, 0.9, _mask(2, 2))])
    with pytest.raises(ValueError, match="exactly one"):
        BOPDetectionsSegmentor(p, sources={"nids": p})
    with pytest.raises(ValueError, match="exactly one"):
        BOPDetectionsSegmentor()


# ── multi-source union: provenance + per-source top-K ────────────────────

def test_multi_source_union_keeps_provenance(tmp_path):
    """Two named backends, distinct masks for the same object: the union yields
    both, each stamped with the backend that produced it."""
    a = _write(tmp_path, "nids", [_det(5, 0.9, _mask(2, 2))])
    b = _write(tmp_path, "cnos", [_det(5, 0.8, _mask(2, 40))])
    seg = BOPDetectionsSegmentor(sources={"nids": a, "cnos": b}, topk=2)
    dets = seg.segment(_scene(), _obj(5))
    assert sorted(d.source for d in dets) == ["cnos", "nids"]


def test_multi_source_top_k_is_per_source(tmp_path):
    """topk applies PER (source, label) bucket, not globally: 3 candidates per
    source at topk=2 keeps 2 from EACH (4 total), so a strong source cannot
    crowd out a weaker one before scoring."""
    # distinct, non-overlapping masks so IoU-dedupe never fires
    a = _write(tmp_path, "nids", [_det(5, 0.9, _mask(0, 0)),
                                  _det(5, 0.8, _mask(0, 20)),
                                  _det(5, 0.7, _mask(0, 40))])
    b = _write(tmp_path, "cnos", [_det(5, 0.6, _mask(20, 0)),
                                  _det(5, 0.5, _mask(20, 20)),
                                  _det(5, 0.4, _mask(20, 40))])
    seg = BOPDetectionsSegmentor(sources=[("nids", a), ("cnos", b)], topk=2)
    dets = seg.segment(_scene(), _obj(5))
    from collections import Counter
    by_src = Counter(d.source for d in dets)
    assert by_src == {"nids": 2, "cnos": 2}


def test_best_segmentor_arg_exclusivity():
    """recipes.best_segmentor enforces the same one-of rule as the constructor,
    so `best_segmentor("old.json", sources=...)` cannot silently ignore the
    positional file and evaluate against the wrong detections."""
    from popoe.recipes import best_segmentor
    with pytest.raises(ValueError, match="exactly one"):
        best_segmentor("old.json", sources={"nids": "new.json"})
    with pytest.raises(ValueError, match="exactly one"):
        best_segmentor()


def test_multi_source_records_sources_attribute(tmp_path):
    a = _write(tmp_path, "nids", [_det(5, 0.9, _mask(2, 2))])
    b = _write(tmp_path, "cnos", [_det(5, 0.8, _mask(2, 40))])
    seg = BOPDetectionsSegmentor(sources={"nids": a, "cnos": b})
    assert [s.name for s in seg.sources] == ["nids", "cnos"]

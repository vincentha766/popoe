"""NIDS-Net (and BOP-generic) detections loading: field coercion + RLE
compatibility. Exercised against a fixture of REAL NIDS WA_Sappe records
(tests/fixtures/nids_lmo_sample.json — a handful pulled from the LM-O release,
genuine uncompressed RLE) plus synthetic string-typed / compressed variants.

Needs pycocotools (the RLE decoder); numpy otherwise. No GPU.
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pycocotools")

from popoe.interfaces import ObjectModel, Scene
from popoe.segmentor_detections import (
    BOPDetectionsSegmentor, decode_detection_mask, load_bop_detections)

FIXTURE = Path(__file__).parent / "fixtures" / "nids_lmo_sample.json"


def _mask_bbox(m):
    ys, xs = np.where(m)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _box_iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter) if (aa + ab - inter) else 0.0


# ── the fixture is real NIDS data ────────────────────────────────────────

def test_load_real_nids_fixture_is_typed_and_rle_normalised():
    recs = load_bop_detections(str(FIXTURE))
    assert recs, "fixture should be non-empty"
    for r in recs:
        assert isinstance(r["scene_id"], int)
        assert isinstance(r["image_id"], int)
        assert isinstance(r["category_id"], int)
        assert isinstance(r["score"], float)
        seg = r["segmentation"]
        # NIDS ships UNCOMPRESSED RLE: counts is a list of run lengths.
        assert isinstance(seg["counts"], list)
        assert all(isinstance(c, int) for c in seg["counts"])
        assert seg["size"] == [480, 640]


def test_decode_real_nids_rle_matches_its_bbox():
    """A decoded mask must be non-degenerate and agree with the record's own
    bbox — the loader/decoder's end-to-end correctness check on real data."""
    recs = load_bop_detections(str(FIXTURE))
    for r in recs:
        m = decode_detection_mask(r["segmentation"])
        assert m.shape == (480, 640)
        assert m.dtype == bool
        assert m.sum() > 0
        x, y, w, h = r["bbox"]
        assert _box_iou(_mask_bbox(m), (x, y, x + w, y + h)) > 0.7


def test_segmentor_loads_nids_fixture_end_to_end():
    recs = json.loads(FIXTURE.read_text())
    sid, iid = recs[0]["scene_id"], recs[0]["image_id"]
    cats = sorted({r["category_id"] for r in recs})
    seg = BOPDetectionsSegmentor(str(FIXTURE), topk=2)
    scene = Scene(rgb=np.zeros((480, 640, 3), np.uint8),
                  depth=np.zeros((480, 640), np.float32),
                  K=np.eye(3), scene_id=sid, im_id=iid)
    for c in cats:
        dets = seg.segment(scene, ObjectModel(obj_id=c, mesh_path="x", diameter=0.1))
        assert dets, f"category {c} should yield >=1 detection"
        assert all(d.source == "bop-detections" for d in dets)
        assert all(d.mask.shape == (480, 640) for d in dets)
    # a category absent from this image yields nothing (not an error)
    absent = max(cats) + 100
    assert seg.segment(scene, ObjectModel(obj_id=absent, mesh_path="x",
                                          diameter=0.1)) == []


# ── the "counts gotcha": uncompressed (list) vs compressed (str) RLE ──────

def _compress(mask):
    from pycocotools import mask as cm
    r = cm.encode(np.asfortranarray(mask.astype(np.uint8)))
    r["counts"] = r["counts"].decode()          # bytes -> str (COCO JSON form)
    return {"size": list(mask.shape), "counts": r["counts"]}


def _uncompress(mask):
    """Column-major run-length list — the NIDS/uncompressed COCO form."""
    flat = mask.astype(np.uint8).T.flatten()     # column-major
    counts, run, val = [], 0, 0
    for px in flat:
        if px == val:
            run += 1
        else:
            counts.append(run)
            run, val = 1, px
    counts.append(run)
    return {"size": list(mask.shape), "counts": counts}


def test_compressed_rle_counts_starting_with_bracket_is_not_json():
    """A compressed COCO RLE string can legitimately begin with '[' (LEB128
    bytes map into printable ASCII). load/normalise must NOT try to JSON-parse
    it — the old first-char check raised JSONDecodeError on it, breaking the
    pre-existing compressed-RLE path. Here: first foreground run at column-major
    offset 43 encodes to counts "[1d0Q6"."""
    from popoe.segmentor_detections import _normalize_segmentation
    mask = np.zeros((16, 16), bool)
    ff = mask.flatten(order="F"); ff[43:63] = True
    mask = ff.reshape((16, 16), order="F")
    compressed = _compress(mask)
    assert compressed["counts"].startswith("[")          # the trap
    norm = _normalize_segmentation(compressed)
    assert isinstance(norm["counts"], str)               # passed through, not parsed
    assert np.array_equal(decode_detection_mask(norm), mask)
    # and through the full loader (a record carrying this segmentation)
    import tempfile
    rec = [{"scene_id": "1", "image_id": "1", "category_id": "1",
            "score": "0.5", "segmentation": compressed}]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(rec, fh); path = fh.name
    out = load_bop_detections(path)[0]
    assert np.array_equal(decode_detection_mask(out["segmentation"]), mask)


def test_non_integral_id_raises_rather_than_truncating():
    """'1.9' as an id must RAISE, not silently truncate to 1 and route to the
    wrong image/object."""
    from popoe.segmentor_detections import _to_int
    assert _to_int("48") == 48 and _to_int("48.0") == 48 and _to_int(48) == 48
    for bad in ("1.9", "1e-1", "0.5"):
        with pytest.raises(ValueError):
            _to_int(bad)


def test_loader_preserves_unknown_fields():
    """BOP's per-detection `time` (and any custom field) survives normalisation,
    so the loader can normalise-and-rewrite without dropping metadata."""
    recs = load_bop_detections(str(FIXTURE))
    # the real NIDS fixture carries a `time` field
    assert all("time" in r for r in recs)


def test_uncompressed_and_compressed_rle_decode_identically():
    """A list `counts` (uncompressed) MUST route through frPyObjects; a str
    `counts` (compressed) MUST NOT. decode_detection_mask picks per-record, so
    both encodings of one mask return the same pixels. Feeding a list to
    `decode` directly (the wrong branch) would silently return garbage."""
    rng = np.random.default_rng(0)
    mask = np.zeros((32, 40), bool)
    mask[3:20, 5:28] = True
    mask[rng.integers(0, 32, 30), rng.integers(0, 40, 30)] = True  # speckle
    m_un = decode_detection_mask(_uncompress(mask))
    m_co = decode_detection_mask(_compress(mask))
    assert np.array_equal(m_un, mask)
    assert np.array_equal(m_co, mask)
    assert np.array_equal(m_un, m_co)


# ── the fully-stringified WA_Sappe-from-Box variant coerces, not silently
#    misses ────────────────────────────────────────────────────────────────

def test_stringified_variant_coerces_and_matches(tmp_path):
    """The documented Box release stringifies every field. Without coercion,
    `category_id "5" in [5]` is False and the object silently gets zero
    candidates. Coercion must recover it — and a stringified uncompressed
    `counts`/`size` must still decode."""
    mask = np.zeros((16, 20), bool)
    mask[2:14, 2:18] = True                      # 192 px, over the min_pixels=100 floor
    un = _uncompress(mask)
    stringy = [{
        "scene_id": "2", "image_id": "102", "category_id": "5",
        "score": "0.7300000119", "bbox": "[2, 2, 16, 12]",
        "segmentation": {"counts": json.dumps(un["counts"]),
                         "size": json.dumps(un["size"])},
    }]
    p = tmp_path / "stringy.json"
    p.write_text(json.dumps(stringy))

    recs = load_bop_detections(str(p))
    r = recs[0]
    assert r["scene_id"] == 2 and r["image_id"] == 102
    assert r["category_id"] == 5 and isinstance(r["category_id"], int)
    assert r["score"] == pytest.approx(0.7300000119)
    assert r["bbox"] == [2.0, 2.0, 16.0, 12.0]
    assert np.array_equal(decode_detection_mask(r["segmentation"]), mask)

    seg = BOPDetectionsSegmentor(str(p), topk=2)
    scene = Scene(rgb=np.zeros((16, 20, 3), np.uint8),
                  depth=np.zeros((16, 20), np.float32),
                  K=np.eye(3), scene_id=2, im_id=102)
    dets = seg.segment(scene, ObjectModel(obj_id=5, mesh_path="x", diameter=0.1))
    assert len(dets) == 1                        # NOT a silent zero-candidate miss
    assert dets[0].mask.sum() == mask.sum()


def test_source_tagging_from_argument_and_record():
    """load_bop_detections stamps `source` from the arg (union origin), else
    from the record, else leaves it absent (bucketed under '_')."""
    recs = load_bop_detections(str(FIXTURE), source="nids")
    assert all(r["source"] == "nids" for r in recs)
    # without the arg, the real NIDS records carry no source field
    assert all("source" not in r for r in load_bop_detections(str(FIXTURE)))

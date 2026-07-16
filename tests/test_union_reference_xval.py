"""Cross-validate popoe's two-source union INGESTION against the gedi-era
merge-script reference (CNOS + SAM-6D, LM-O).

The reference (`union_cnos_sam6d_lmo.reference.json`) is the merged detection
POOL: all CNOS + all SAM-6D records, source-tagged, with NO per-query top-M or
dedup baked in (count == exact sum of the two sources). That selection happens
at segment() time in both stacks. So the equivalence to check is: does loading
the two source files through popoe's loader and combining them — exactly what
BOPDetectionsSegmentor does into `_by_img` — reproduce the reference pool
record-for-record?

These files are large and downloaded (gitignored under data/detections/), so
the test SKIPS when they are not present locally; it is a local reproducibility
guard, not a CI gate. Needs pycocotools only transitively via the loader (not
here — no decode), numpy-free.
"""
import json
import os
from collections import Counter
from pathlib import Path

import pytest

from popoe.segmentor_detections import load_bop_detections

# Anchor to the repo root, not CWD — else running the test from elsewhere would
# silently skip even when the data is present.
_DET = Path(__file__).resolve().parents[1] / "data" / "detections"
CNOS = str(_DET / "cnos" / "cnos-fastsam_lmo-test.json")
SAM6D = str(_DET / "sam6d" / "sam6d_ism_lmo.json")
REF = str(_DET / "sam6d" / "union_cnos_sam6d_lmo.reference.json")


def _canon(r):
    """Canonical, hashable form of a FULL normalised record — every field
    (scene/image/category/source/score/bbox/time/segmentation) with EXACT
    scores (json repr round-trips floats), so the multiset proves record-for-
    record equality, not just detection-identity."""
    return json.dumps(r, sort_keys=True)


@pytest.mark.skipif(not all(os.path.exists(p) for p in (CNOS, SAM6D, REF)),
                    reason="local CNOS/SAM-6D/reference detection files absent")
def test_union_ingestion_matches_gedi_reference():
    mine = (load_bop_detections(CNOS, source="cnos")
            + load_bop_detections(SAM6D, source="sam6d"))
    ref = load_bop_detections(REF)                 # keeps its own source tags

    # same size and per-source composition
    assert len(mine) == len(ref)
    assert (Counter(r["source"] for r in mine)
            == Counter(r["source"] for r in ref))

    # identical as a multiset of FULL records — order-independent, exact scores
    assert Counter(map(_canon, mine)) == Counter(map(_canon, ref))

"""Paper Sec. III-D target protocol helpers (triage D3/D4).

D3: the sparse targets are the patch centres of the minimal axis-aligned
SQUARE bbox, features assigned per patch (no bilinear) — the geometry lives
in adapters.paper_grid_centers so it is testable without torch.

D4: P_T^dense is ONE cloud serving both the GeDi neighbourhood and ICP;
both sides subsample through adapters.fixed_seed_subsample over the same
index space, so sharing the function IS the equality guarantee.
"""
import numpy as np
import pytest

from popoe.adapters import fixed_seed_subsample, paper_grid_centers


def test_square_side_is_the_larger_bbox_side():
    # mask bbox 20 px tall (y 10..29), 10 px wide (x 5..14) -> side 20,
    # centred: the square extends the NARROW axis symmetrically.
    rows, cols, u, v, (bx0, by0, side) = paper_grid_centers(10, 29, 5, 14, 16)
    assert side == 20.0
    assert by0 == 10.0                       # tall axis untouched
    assert bx0 == 0.0                        # 10 px wide -> 5 px pad each side
    assert len(u) == len(v) == 16 * 16       # unfiltered full grid


def test_centres_are_patch_centres_of_the_square():
    rows, cols, u, v, (bx0, by0, side) = paper_grid_centers(0, 15, 0, 15, 16)
    # 16 px square, 16x16 grid -> one centre per pixel, at pixel centres.
    assert side == 16.0
    assert u.min() == 0 and u.max() == 15
    assert v.min() == 0 and v.max() == 15
    # tile (i, j) centre -> row i, col j of the DINO patch map
    k = np.flatnonzero((rows == 0) & (cols == 0))[0]
    assert (v[k], u[k]) == (0, 0)
    k = np.flatnonzero((rows == 15) & (cols == 0))[0]
    assert (v[k], u[k]) == (15, 0)           # rows move down, cols move right


def test_centres_can_fall_outside_the_image():
    # A mask hugging the left edge: the square pads LEFT of x=0 — centres
    # there come back negative and the CALLER drops them (no clipping here,
    # clipping would silently move centres onto wrong pixels).
    _, _, u, _, (bx0, _, side) = paper_grid_centers(0, 99, 0, 9, 16)
    assert bx0 < 0
    assert (u < 0).any()


def test_fixed_seed_subsample_contract():
    idx = fixed_seed_subsample(10, 3)
    assert list(idx) == sorted(idx)              # sorted
    assert len(set(idx)) == 3                    # no replacement
    assert np.array_equal(idx, fixed_seed_subsample(10, 3))   # deterministic
    assert fixed_seed_subsample(3, 0) is None    # 0 = no cap
    assert fixed_seed_subsample(3, 5) is None    # under cap = keep all


def test_box_is_integer_snapped_and_tiles_align_with_the_crop():
    """Review F1: the DINO crop and the tiling must share EXACTLY the same
    box. The corner is floored to integers, so int() on the crop edges is
    exact — and every centre's pixel maps back to its own tile."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        y0 = int(rng.integers(0, 50)); x0 = int(rng.integers(0, 50))
        y1 = y0 + int(rng.integers(16, 120)); x1 = x0 + int(rng.integers(16, 120))
        g = 16
        rows, cols, u, v, (bx0, by0, side) = paper_grid_centers(y0, y1, x0, x1, g)
        assert bx0 == int(bx0) and by0 == int(by0)     # integer box
        cx = bx0 + (cols + 0.5) * side / g
        cy = by0 + (rows + 0.5) * side / g
        # the stored pixel CONTAINS the continuous centre ...
        assert np.array_equal(u, np.floor(cx).astype(int))
        assert np.array_equal(v, np.floor(cy).astype(int))
        # ... and the centre sits in ITS OWN tile of the SAME box — which is
        # exactly the patch whose feature it receives (feat_map[rows, cols]).
        assert np.array_equal(np.floor((cx - bx0) * g / side).astype(int), cols)
        assert np.array_equal(np.floor((cy - by0) * g / side).astype(int), rows)


def test_query_camera_radius_modes():
    """Audit P2: legacy = the historical 1.5x (fill INERT); effective makes
    fill a real setting — the projected fraction of the 60-degree frame."""
    from popoe.adapters import query_camera_radius
    import math
    e = 0.1
    assert query_camera_radius(e, 0.5, "legacy") == pytest.approx(0.15)
    assert query_camera_radius(e, 0.45, "legacy") == pytest.approx(0.15)  # inert
    r50 = query_camera_radius(e, 0.5, "effective")
    r45 = query_camera_radius(e, 0.45, "effective")
    assert r50 != r45                                   # fill now matters
    # geometry: extent / (r * 2 tan(fov/2)) == fill exactly
    assert e / (r50 * 2 * math.tan(math.radians(30))) == pytest.approx(0.5)
    assert e / (r45 * 2 * math.tan(math.radians(30))) == pytest.approx(0.45)
    # the legacy constant the cancellation actually produced
    assert e / (0.15 * 2 * math.tan(math.radians(30))) == pytest.approx(0.577, abs=1e-3)
    with pytest.raises(ValueError):
        query_camera_radius(e, 0.5, "auto")


def test_conditional_enc_entries_single_authority(monkeypatch):
    """Review real-bug: the flip_rescore mirror of the conditional enc keys
    drifted THREE times. Both consumers now read popoe.cache — this pins the
    table's semantics (non-default only) and its current membership."""
    from popoe.cache import CONDITIONAL_ENC_KEYS, conditional_enc_entries
    for _, (env, _) in CONDITIONAL_ENC_KEYS.items():
        monkeypatch.delenv(env, raising=False)
    assert conditional_enc_entries() == {}          # all-default = empty
    monkeypatch.setenv("POPOE_QUERY_FILL_MODE", "effective")
    monkeypatch.setenv("POPOE_TARGET_PAPER_GRID", "1")
    assert conditional_enc_entries() == {"query_fill_mode": "effective",
                                         "target_paper_grid": "1"}
    assert set(CONDITIONAL_ENC_KEYS) == {
        "query_canon", "query_fill", "query_fill_mode", "query_min_views",
        "canon_basis", "query_views", "target_dense", "target_paper_grid",
        "query_sampler"}


def test_effective_fill_upper_bound_refused():
    from popoe.adapters import query_camera_radius
    with pytest.raises(ValueError, match="0.7"):
        query_camera_radius(0.1, 0.9, "effective")
    query_camera_radius(0.1, 0.7, "effective")      # boundary accepted

"""Paper Sec. III-D target protocol helpers (triage D3/D4).

D3: the sparse targets are the patch centres of the minimal axis-aligned
SQUARE bbox, features assigned per patch (no bilinear) — the geometry lives
in adapters.paper_grid_centers so it is testable without torch.

D4: P_T^dense is ONE cloud serving both the GeDi neighbourhood and ICP;
both sides subsample through adapters.fixed_seed_subsample over the same
index space, so sharing the function IS the equality guarantee.
"""
import numpy as np

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

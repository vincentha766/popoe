"""FPFH descriptor contract tests — CPU only, no GPU, no BOP data.

These check the properties that make FPFH a VALID control for GeDi rather than
just a thing that returns an array: rotation invariance (otherwise the
comparison measures pose handling, not descriptor quality), the NaN
invalid-row convention fusion relies on, and the canonical-scale/keypoint
lookup contract from popoe.interfaces.PointDescriptor.
"""

import numpy as np
import pytest

from popoe.descriptors import FPFHDescriptor, FPFH_DIM, load_fpfh
from popoe.interfaces import PointDescriptor

pytest.importorskip("open3d")


def _bumpy_sphere(n=800, seed=0):
    """A cloud with real curvature variation — a perfect sphere has an almost
    constant FPFH everywhere, which would make the invariance test vacuous."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = 0.5 + 0.05 * np.sin(4 * v[:, 0]) * np.cos(3 * v[:, 1])
    return (v * r[:, None]).astype(np.float32)


def _rot(axis, ang):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def test_satisfies_point_descriptor_protocol():
    assert isinstance(FPFHDescriptor(), PointDescriptor)


def test_shape_and_dtype_two_scale():
    pts = _bumpy_sphere()
    out = FPFHDescriptor(radii=(0.3, 0.4)).compute(pts, pts)
    assert out.shape == (len(pts), 2 * FPFH_DIM)
    assert out.dtype == np.float32


def test_single_scale_dim():
    pts = _bumpy_sphere(300)
    assert FPFHDescriptor(radii=(0.3,)).compute(pts, pts).shape[1] == FPFH_DIM


def test_rotation_invariant():
    """FPFH bins angles between normals, so rotating the whole cloud must not
    change the descriptors. If this fails the GeDi-vs-FPFH comparison is
    measuring frame handling instead of descriptor quality."""
    pts = _bumpy_sphere()
    desc = FPFHDescriptor(orient="outward")
    a = desc.compute(pts, pts)
    R = _rot([0.3, -0.7, 0.5], 1.1)
    b = desc.compute(pts @ R.T, pts @ R.T)

    valid = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    assert valid.sum() > 0.9 * len(pts)
    a_n = a[valid] / (np.linalg.norm(a[valid], axis=1, keepdims=True) + 1e-8)
    b_n = b[valid] / (np.linalg.norm(b[valid], axis=1, keepdims=True) + 1e-8)
    cos = (a_n * b_n).sum(1)
    # Not bitwise: normal estimation and the MST orientation pass are both
    # order-dependent under rotation. Near-identical is the real claim.
    assert np.median(cos) > 0.99, f"median cos {np.median(cos):.4f}"
    assert cos.mean() > 0.97


def test_keypoint_subset_lookup_is_exact_without_downsampling():
    """With voxel_frac=0 the described points ARE the support points, so a
    keypoint subset must get exactly those rows."""
    pts = _bumpy_sphere()
    desc = FPFHDescriptor(voxel_frac=0)
    full = desc.compute(pts, pts)
    sub = np.arange(0, len(pts), 7)
    got = desc.compute(pts[sub], pts)
    np.testing.assert_allclose(got, full[sub], rtol=0, atol=0)
    assert desc.last_lookup_displacement == 0.0


def test_downsampling_keeps_the_radius_binding_not_the_cap():
    """The reason density normalisation exists: Open3D returns the NEAREST
    max_nn points, so a binding cap silently shrinks the radius and the
    descriptor is no longer the one GeDi is being compared against."""
    dense = _bumpy_sphere(20000, seed=3)

    off = FPFHDescriptor(voxel_frac=0, max_nn_feature=100)
    off.compute(dense[::10], dense)
    max_nbrs, cap = off.last_cap_binding
    assert max_nbrs > cap, "fixture too sparse to exercise the cap"

    on = FPFHDescriptor()          # defaults: voxel_frac=0.1, cap 1000
    on.compute(dense[::10], dense)
    max_nbrs, cap = on.last_cap_binding
    assert max_nbrs < cap, f"cap still binds after downsampling: {max_nbrs} >= {cap}"
    assert on.last_support_size < len(dense)
    # Half a voxel = 0.5 * 0.1 * min(radii); the price of a radius-defined
    # support, and it must stay small next to the radius itself.
    assert on.last_lookup_displacement < 0.1 * min(on.radii)


@pytest.mark.parametrize("radii", [(0.3, 0.4), (0.25, 0.5), (0.2, 0.6)])
def test_cap_never_binds_across_radius_ratios(radii):
    """The voxel is sized off the LARGEST radius. Sizing it off the smallest
    left the coarse scale free to blow past the cap — measured 1559 and 3221
    against a cap of 1000 for (0.25,0.5) and (0.2,0.6) — which silently
    restores the nearest-max_nn truncation the downsampling exists to remove.
    Radii are a swept knob, so this has to hold for any ratio, not just the
    default."""
    dense = _bumpy_sphere(20000, seed=3)
    d = FPFHDescriptor(radii=radii)
    d.compute(dense[::10], dense, role="target")
    max_nbrs, cap = d.last_cap_binding
    assert max_nbrs < cap, f"radii={radii}: cap binds ({max_nbrs} >= {cap})"


def test_planar_support_does_not_crash_the_query_arm():
    """A planar cloud sends Open3D's consistent-tangent-plane pass into a Qhull
    failure. Only the query role runs that pass, so an unguarded call takes out
    the FPFH query arm on a flat object while the target arm survives."""
    g = np.linspace(-0.5, 0.5, 40)
    xx, yy = np.meshgrid(g, g)
    plane = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], 1).astype(np.float32)

    out = FPFHDescriptor().compute(plane, plane, role="query")
    assert out.shape == (len(plane), 2 * FPFH_DIM)
    assert np.isfinite(out).any()


def test_declared_role_overrides_a_pinned_orient():
    """POPOE_FPFH_ORIENT is one process-wide setting but the two sides need
    opposite conventions; honouring the pin on the live path would apply one
    convention to both and mirror them — the exact failure role was added to
    prevent. The pin still governs role-less calls."""
    pinned = FPFHDescriptor(orient="camera")
    pts = _bumpy_sphere(200)
    assert pinned.resolve_orient(pts, role="query") == "outward"
    assert pinned.resolve_orient(pts, role="target") == "camera"
    assert pinned.resolve_orient(pts) == "camera"      # role-less: pin applies


def test_support_is_density_independent():
    """Same surface sampled at 3 densities must give ~the same descriptors —
    GeDi gets this from its fixed per-patch point count, FPFH needs the voxel
    step. Without it a near (dense) object and a far (sparse) one are
    described differently, which would confound the ablation."""
    desc = FPFHDescriptor()
    ref = _bumpy_sphere(4000, seed=5)
    a = desc.compute(ref, ref)
    b = desc.compute(ref, _bumpy_sphere(16000, seed=6))

    valid = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    a_n = a[valid] / (np.linalg.norm(a[valid], axis=1, keepdims=True) + 1e-8)
    b_n = b[valid] / (np.linalg.norm(b[valid], axis=1, keepdims=True) + 1e-8)
    assert np.median((a_n * b_n).sum(1)) > 0.95


def test_invalid_rows_are_nan_not_zero():
    """fusion.DinoGeDiFusion treats NaN rows as invalid; an all-zero histogram
    would instead be matched against everything."""
    pts = _bumpy_sphere(200)
    # A lone point far from the cloud has an empty neighbourhood at r=0.05.
    far = np.vstack([pts, [[50.0, 50.0, 50.0]]]).astype(np.float32)
    out = FPFHDescriptor(radii=(0.05,)).compute(far, far)
    assert np.isnan(out[-1]).all()
    assert np.isfinite(out[:-1]).any()


def test_degenerate_input_returns_all_nan():
    tiny = np.zeros((3, 3), dtype=np.float32)
    out = FPFHDescriptor().compute(tiny, tiny)
    assert out.shape == (3, 2 * FPFH_DIM)
    assert np.isnan(out).all()


def test_accepts_torch_tensors():
    torch = pytest.importorskip("torch")
    pts = _bumpy_sphere(300)
    desc = FPFHDescriptor()
    np.testing.assert_allclose(
        desc.compute(torch.from_numpy(pts), torch.from_numpy(pts)),
        desc.compute(pts, pts))


def test_role_decides_the_convention_regardless_of_geometry():
    """Role is authoritative: the pipeline declares it rather than letting the
    descriptor guess, because the guess is provably ambiguous (see the
    resolve_orient docstring). Geometry that would fool every heuristic must
    still resolve correctly when the role is given."""
    desc = FPFHDescriptor(orient="auto")
    cad = _bumpy_sphere(400)
    far_cad = cad + [0.0, 0.0, 8.0]        # looks exactly like a depth cloud
    near_depth = cad[cad[:, 2] < -0.15] + [0.0, 0.0, 1.2]   # only ~1 extent out

    assert desc.resolve_orient(far_cad, role="query") == "outward"
    assert desc.resolve_orient(near_depth, role="target") == "camera"
    with pytest.raises(ValueError):
        desc.resolve_orient(cad, role="neither")


def test_auto_orient_fallback_without_a_role():
    """The no-role fallback still has to do something sensible for offline and
    interactive use (tests, notebooks), even though the pipeline passes a role."""
    desc = FPFHDescriptor(orient="auto")
    cad = _bumpy_sphere(400)                        # canonicalised CAD model
    assert desc.resolve_orient(cad) == "outward"
    # Canonicalised camera frame: object ~0.5-1.5 m out, extent scaled to 1.
    assert desc.resolve_orient(cad + [0.0, 0.0, 8.0]) == "camera"


def test_auto_orient_survives_an_off_centre_cad_origin():
    """A CAD model whose origin sits at a corner of the mesh is still a CAD
    model. An origin-inside-the-bounding-box test would misfile it as a depth
    cloud and mirror its normals against the query side."""
    desc = FPFHDescriptor(orient="auto")
    cad = _bumpy_sphere(400)
    # Shift so the origin lies just OUTSIDE the cloud's bounding box.
    off = cad + (cad.max(axis=0) - cad.min(axis=0)) * 0.75
    assert off.min(axis=0).max() > 0, "fixture must exclude the origin"
    assert desc.resolve_orient(off) == "outward"


def test_orient_modes_can_be_pinned_and_bad_modes_rejected():
    pts = _bumpy_sphere(200)
    for mode in ("outward", "camera", "none"):
        assert FPFHDescriptor(orient=mode).resolve_orient(pts) == mode
    with pytest.raises(ValueError):
        FPFHDescriptor(orient="sideways").resolve_orient(pts)


def test_query_and_target_of_one_surface_agree_on_the_convention():
    """The property all the orientation machinery exists for: the same patch
    seen as a CAD query and as a camera-frame depth cloud must produce
    matching descriptors. If the conventions mirror, cosine collapses and the
    FPFH arm would lose for a reason that has nothing to do with FPFH."""
    cad = _bumpy_sphere(3000, seed=11)
    # The half a depth sensor actually returns, at a realistic canonical
    # standoff: the object sits at +z, so the CAMERA-FACING surface is its -z
    # half. (Taking the +z half instead yields the occluded back face, whose
    # outward normals point away from the camera — a fixture that "fails" for
    # a reason unrelated to the code under test.)
    standoff = np.array([0.0, 0.0, 8.0])
    visible = cad[cad[:, 2] < -0.15] + standoff

    desc = FPFHDescriptor()
    q = desc.compute(cad, cad, role="query")
    t = desc.compute(visible, visible, role="target")

    # Match each visible point to its CAD counterpart (same index space is lost
    # after the mask, so re-derive by position).
    from scipy.spatial import cKDTree
    back = visible - np.array([0.0, 0.0, 8.0])
    _, idx = cKDTree(cad).query(back, k=1)
    valid = np.isfinite(q[idx]).all(1) & np.isfinite(t).all(1)
    qn = q[idx][valid] / (np.linalg.norm(q[idx][valid], axis=1, keepdims=True) + 1e-8)
    tn = t[valid] / (np.linalg.norm(t[valid], axis=1, keepdims=True) + 1e-8)
    cos = (qn * tn).sum(1)
    # Not ~1.0: the depth half genuinely lacks the back surface, so patches
    # near the silhouette differ. The claim is only that the two are on the
    # same convention — mirrored normals land far below this.
    assert np.median(cos) > 0.75, f"median cos {np.median(cos):.3f}"


def test_env_config_and_cache_fragment(monkeypatch):
    monkeypatch.setenv("POPOE_FPFH_RADII", "0.25,0.5")
    monkeypatch.setenv("POPOE_FPFH_ORIENT", "camera")
    monkeypatch.setenv("POPOE_FPFH_VOXEL_FRAC", "0")
    d = load_fpfh()
    assert d.radii == (0.25, 0.5) and d.orient == "camera" and d.voxel_frac == 0.0
    cfg = d.config()
    assert cfg["fpfh_radii"] == "0.25,0.5" and cfg["fpfh_orient"] == "camera"


def test_every_output_changing_knob_reaches_the_cache_key():
    """A knob missing from config() means a sweep over it silently reloads the
    previous setting's cached features — the failure is invisible in the AR."""
    base = FPFHDescriptor()
    variants = {
        "radii": FPFHDescriptor(radii=(0.4, 0.5)),
        "voxel_frac": FPFHDescriptor(voxel_frac=0.2),
        "normal_radius_frac": FPFHDescriptor(normal_radius_frac=0.5),
        "orient": FPFHDescriptor(orient="camera"),
        "far_extents": FPFHDescriptor(far_extents=5.0),
        "max_nn_normal": FPFHDescriptor(max_nn_normal=50),
        "max_nn_feature": FPFHDescriptor(max_nn_feature=500),
        "knn_orient": FPFHDescriptor(knn_orient=30),
    }
    # Guard against a knob being added to __init__ and forgotten here.
    tracked = set(variants) | {"radii"}
    assert set(vars(base)) - tracked <= {"last_support_size",
                                         "last_lookup_displacement",
                                         "last_cap_binding"}
    for name, v in variants.items():
        assert v.config() != base.config(), f"{name} missing from config()"


def test_orient_none_is_role_neutral():
    """"Orient nothing" asks for raw Open3D signs on both sides, so unlike an
    "outward"/"camera" pin it is not something a role should override — the
    knob exists to probe how much the orientation machinery is worth."""
    d = FPFHDescriptor(orient="none")
    pts = _bumpy_sphere(200)
    assert d.resolve_orient(pts, role="query") == "none"
    assert d.resolve_orient(pts, role="target") == "none"
    assert d.resolve_orient(pts) == "none"

"""Method-agnostic per-point 3D descriptors.

FreeZe's geometric branch is GeDi, but nothing in the pipeline requires it: the
encoders only ever call ``descriptor.compute(pts, pcd)`` (see
``popoe.interfaces.PointDescriptor``). Descriptors that are NOT part of the
FreeZe recipe live here rather than in ``popoe.freeze`` — starting with FPFH,
the classical control the reproduction study was missing.

Why FPFH is worth a module: the study's only geometric-backbone comparison was
GeDi vs dGeDi (-16.6 pt on LM-O), i.e. GeDi beating a *weaker learned*
descriptor. That does not establish that a learned descriptor is needed at all.
FPFH (Rusu et al., ICRA 2009) is the standard hand-crafted control — no
training, no GPU, Open3D built-in — and either outcome is publishable: a wide
GeDi win justifies the learned branch, while a near-tie reframes the geometric
branch as a swappable commodity.

**Scale convention** (get this wrong and the comparison is meaningless): both
call sites hand the descriptor points already scaled by the ``CanonFrame``, so
the object's largest extent is ~1.0. GeDi's ``r_lrf`` = 0.3/0.4 are radii in
that canonical space, so the FPFH radii default to the same 0.3/0.4 and the two
backbones integrate over the same physical neighbourhood.

Cost note: FPFH is pure CPU, so the whole arm can be developed, tested and (on
a small object subset) run locally — only the DINOv2 half of a *fused* run
needs the pod.
"""

from __future__ import annotations

import os

import numpy as np

FPFH_DIM = 33  # Open3D's FPFH histogram length (11 bins x 3 angular features)

# Below this many support points a neighbourhood cannot produce a meaningful
# histogram; the whole call returns the NaN "invalid" rows fusion expects.
MIN_SUPPORT = 10

# orient_normals_consistent_tangent_plane builds a Riemannian graph + MST, which
# is fine for a 3k-point query cloud and far too slow for a dense depth cloud.
# Above this size the "outward" mode degrades to the O(N) centroid rule.
MAX_PTS_FOR_MST = 20000

# Keep the original cloud when downsampling would retain more than this share of
# it: at that point the cloud is already near the target resolution, so the only
# effect left is displacing the keypoint lookup.
KEEP_ORIGINAL_ABOVE = 0.9


def _as_numpy(x) -> np.ndarray:
    """Accept torch tensors or numpy arrays; Open3D needs contiguous float64."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64))


class FPFHDescriptor:
    """Multi-scale FPFH, shaped to be a drop-in for ``_TwoScaleGeDi``.

    ``compute(pts, pcd)`` returns ``(len(pts), 33 * len(radii))`` float32, with
    NaN rows where the neighbourhood was too sparse to describe — the
    invalid-row convention ``popoe.freeze.fusion.DinoGeDiFusion`` expects.
    Fusion matches the visual PCA dim to the geometric dim at fuse() time, so
    the 66D two-scale output needs no other change (nothing downstream
    hard-codes 64).

    **Density normalisation is not optional here.** Open3D's hybrid search
    returns the NEAREST ``max_nn`` points inside the radius, so whenever the
    cap binds the effective radius silently shrinks — and it binds hard at
    these densities (a 3k-point query cloud has ~272/482 neighbours at r=0.3/
    0.4; a 20k-point masked depth cloud has ~1772/3166). Raising the cap
    instead is not an option: measured two-scale FPFH on a 20k cloud costs
    0.2 s at cap 100, 4.3 s at 1000 and 48.8 s at 4000, per detection. So the
    support cloud is voxel-downsampled to a radius-relative resolution first.
    That (a) keeps the cap from binding, so the RADIUS defines the support as
    it does for GeDi, (b) makes the neighbourhood independent of how near or
    dense the object happened to be — which GeDi gets for free by sampling a
    fixed point count per patch — and (c) keeps the cost bounded.

    Args:
        radii: feature radii in CANONICAL units (object extent ~= 1.0). The
            default mirrors two-scale GeDi's r_lrf, which is what makes the
            two backbones comparable.
        voxel_frac: downsample voxel as a fraction of the LARGEST radius
            (see above). 0 disables downsampling, which restores exact
            keypoint positions at the cost of a cap-truncated, density-
            dependent support — only for studying that effect.
        normal_radius_frac: normal-estimation radius as a fraction of the
            feature radius (Open3D's tutorial ratio is 1/3). Normals are
            re-estimated per scale so each scale is a genuinely different
            support, mirroring GeDi recomputing an LRF per r_lrf.
        orient: normal orientation policy. FPFH bins *angles between normals*,
            so a flipped normal is a different descriptor and the query and
            target sides must agree on the convention. "auto" decides per call
            from how far the cloud sits from the origin relative to its own
            extent: a camera-frame depth cloud is many extents away (the
            object is ~0.5-1.5 m out and canonical scaling makes its extent
            1.0), a CAD model is centred on or near its own origin. Depth ->
            orient toward the origin, CAD -> outward from the centroid; on a
            visible depth surface those agree (both = outward from the object
            surface), which is the consistency FPFH needs. Pass
            "outward"/"camera"/"none" to pin it — but note a declared `role`
            overrides the pin, because one process-wide pin applied to both
            sides is what mirrors them (see resolve_orient).
        far_extents: the "many extents away" threshold for that rule. The two
            live cases are ~0 and ~5-15 extents, so anything in between works;
            3.0 is the midpoint in log terms.
        max_nn_normal / max_nn_feature: Open3D hybrid-search caps. After
            downsampling these are guard rails that should NOT bind — if they
            do, the radius is no longer defining the support (see
            `last_cap_binding`).
        knn_orient: k for the consistent-tangent-plane pass in "outward" mode.

    Diagnostics after a compute(): `last_support_size` (points actually
    described), `last_lookup_displacement` (median keypoint -> support-point
    distance introduced by downsampling, in canonical units) and
    `last_cap_binding` (max observed neighbour count vs the cap).

    **Known limitation — thin structures.** The default voxel is 4% of the
    object's largest extent, so two surfaces closer than that merge into one
    averaged point with an averaged normal. On a 200 mm pair of scissors the
    voxel is 8 mm and the blades are ~2 mm thick: front and back faces are
    described as one surface. GeDi, which samples a fixed point count per
    patch without gridding, does not have this failure mode, so on thin
    objects part of any FPFH deficit is the discretisation rather than the
    descriptor. Sweep POPOE_FPFH_VOXEL_FRAC (it is in the cache key) or set it
    to 0 on those objects before drawing a per-object conclusion.
    """

    def __init__(self, radii=(0.3, 0.4), voxel_frac=0.1,
                 normal_radius_frac=1.0 / 3, orient="auto", far_extents=3.0,
                 max_nn_normal=100, max_nn_feature=1000, knn_orient=15):
        self.radii = tuple(float(r) for r in radii)
        if not self.radii:
            raise ValueError("FPFHDescriptor needs at least one radius")
        self.voxel_frac = float(voxel_frac)
        self.normal_radius_frac = float(normal_radius_frac)
        self.orient = orient
        self.far_extents = float(far_extents)
        self.max_nn_normal = int(max_nn_normal)
        self.max_nn_feature = int(max_nn_feature)
        self.knn_orient = int(knn_orient)
        self.last_support_size = None
        self.last_lookup_displacement = None
        self.last_cap_binding = None

    @property
    def dim(self) -> int:
        return FPFH_DIM * len(self.radii)

    def config(self) -> dict:
        """Everything that changes the output — merge into the stage cache key.
        Every constructor knob belongs here: one missing entry means a sweep
        over it silently reloads the previous setting's cached features."""
        return {
            "fpfh_radii": ",".join(f"{r:g}" for r in self.radii),
            "fpfh_voxel_frac": f"{self.voxel_frac:g}",
            "fpfh_normal_frac": f"{self.normal_radius_frac:g}",
            "fpfh_orient": self.orient,
            "fpfh_far_extents": f"{self.far_extents:g}",
            "fpfh_max_nn": f"{self.max_nn_normal},{self.max_nn_feature}",
            "fpfh_knn_orient": str(self.knn_orient),
        }

    # ── orientation ──────────────────────────────────────────────────────

    def resolve_orient(self, pts: np.ndarray, role: str | None = None) -> str:
        """Which orientation convention `pts` gets: outward | camera | none.

        Separated from the work so the decision is directly testable — the
        cost of getting it wrong (query and target on mirrored normal
        conventions, hence uncomparable descriptors) is silent.

        `role` is authoritative when given: "query" is a CAD model sampled all
        around (closed surface, orient outward), "target" is a single-view
        depth cloud (open surface, orient toward the camera at the origin).
        Both live call sites know their role and pass it to `compute()`.

        Without a role this falls back to a DISTANCE heuristic, and that
        fallback has a genuinely ambiguous band. Three candidate rules were
        measured and none separates the cases cleanly:

          rule                      centred CAD   corner-origin CAD   depth @2.4 extents
          origin inside AABB            CAD            depth (X)          depth
          centroid dist / extent       0.01            1.29              2.08
          |mean unit direction|        0.020           0.954             0.985

        A CAD model whose origin sits outside its own geometry is genuinely
        indistinguishable from a depth cloud by points alone. So the fallback
        is a best effort for offline/interactive use, not something the
        pipeline should rely on: `compute(..., role=)` passes the role instead.
        """
        if self.orient not in ("auto", "outward", "camera", "none"):
            raise ValueError(f"unknown orient mode: {self.orient!r}")
        if self.orient == "none":
            # "Orient nothing" is role-NEUTRAL: it asks for raw Open3D signs on
            # both sides, so a role does not override it. (The two sides are
            # then mutually inconsistent by construction — that is the point of
            # the knob, for probing how much orientation is worth.)
            return "none"
        # A declared role WINS over a pinned mode. The pin is one process-wide
        # setting (POPOE_FPFH_ORIENT) but the two sides need opposite answers,
        # so honouring it on the live path would apply one convention to both
        # and mirror them against each other — reintroducing the exact bug the
        # role exists to prevent. The pin therefore governs role-less calls
        # only (tests, notebooks, offline probing).
        if role is not None:
            if role not in ("query", "target"):
                raise ValueError(f"unknown role: {role!r}")
            return "outward" if role == "query" else "camera"
        if self.orient != "auto":
            return self.orient
        extent = float(np.ptp(pts, axis=0).max())
        if extent <= 1e-9:
            return "outward"
        centroid_dist = float(np.linalg.norm(pts.mean(axis=0)))
        return "camera" if centroid_dist > self.far_extents * extent else "outward"

    def _orient_normals(self, cloud, role=None) -> None:
        import open3d as o3d

        pts = np.asarray(cloud.points)
        mode = self.resolve_orient(pts, role)
        if mode == "none":
            return

        if mode == "camera":
            # Target clouds are in the camera frame, camera at the origin.
            cloud.orient_normals_towards_camera_location(np.zeros(3))
            return

        # Consistent-tangent-plane fixes LOCAL agreement but leaves the GLOBAL
        # sign arbitrary (it can settle all-inward). A majority vote against the
        # centroid pins it outward, so a CAD query and a depth target land on
        # the same convention rather than on mirrored histograms.
        if len(pts) <= MAX_PTS_FOR_MST:
            try:
                cloud.orient_normals_consistent_tangent_plane(self.knn_orient)
            except RuntimeError:
                # Degenerate support (a planar CAD face, or a thin object whose
                # cloud goes coplanar after the voxel merge) sends Open3D's
                # Riemannian-graph build into a Qhull failure. The centroid vote
                # below still gives a usable global sign, and crashing the query
                # arm on a flat object is a worse answer than a coarser one.
                pass
        n = np.asarray(cloud.normals)
        radial = pts - pts.mean(axis=0)
        if float(np.einsum("ij,ij->i", n, radial).sum()) < 0.0:
            cloud.normals = o3d.utility.Vector3dVector(-n)

    # ── the PointDescriptor contract ─────────────────────────────────────

    def compute(self, pts, pcd, role=None) -> np.ndarray:
        import open3d as o3d
        from scipy.spatial import cKDTree

        keypts = _as_numpy(pts)
        support = _as_numpy(pcd)
        if keypts.ndim != 2 or keypts.shape[1] != 3:
            raise ValueError(f"pts must be (N,3), got {keypts.shape}")
        if support.ndim != 2 or support.shape[1] != 3:
            raise ValueError(f"pcd must be (M,3), got {support.shape}")

        # Clear first: a stale reading from the previous call is worse than
        # none, because it reads as a healthy diagnostic for this one.
        self.last_support_size = None
        self.last_lookup_displacement = None
        self.last_cap_binding = None

        out = np.full((len(keypts), self.dim), np.nan, dtype=np.float32)
        if len(support) < MIN_SUPPORT or len(keypts) == 0:
            return out

        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(support))
        if self.voxel_frac > 0:
            # Normalise density BEFORE describing anything (see class docstring):
            # a cap-truncated neighbourhood is a shrunken radius, and a shrunken
            # radius is a different descriptor than the one GeDi is being
            # compared against.
            # Driven by the LARGEST radius, not the smallest: neighbour count
            # grows as (r/voxel)^2, so sizing the voxel off r_min leaves the
            # coarse scale free to blow past the cap — exactly the truncation
            # this exists to prevent. Measured on a 20k cloud with cap 1000:
            # radii (0.3,0.4) peaked at 725 (fine), but (0.25,0.5) hit 1559 and
            # (0.2,0.6) hit 3221. Sizing off r_max makes every scale safe for
            # any ratio, which matters because the radii are a swept knob.
            ds = cloud.voxel_down_sample(self.voxel_frac * max(self.radii))
            # Skip when it barely bites: a cloud already at (or below) the target
            # resolution — the query side, whose 3k surface samples are spaced
            # about one voxel apart — gains nothing and would pay the keypoint
            # displacement for free. Keeping the original there means query
            # features sit exactly on the points the solver uses.
            if MIN_SUPPORT <= len(ds.points) < KEEP_ORIGINAL_ABOVE * len(support):
                cloud = ds
        described = np.asarray(cloud.points)
        self.last_support_size = len(described)

        # Open3D describes every point of the cloud it is given, so features are
        # computed on the support cloud and read back at the keypoints. Without
        # downsampling this is exact: at both live call sites the keypoints are
        # drawn FROM the support cloud (query: pts is pcd; target: the grid
        # subsample of the same masked depth pixels), so the distance is 0.
        # Downsampling moves the described points by up to half a voxel, which
        # is the price of a radius-defined support; last_lookup_displacement
        # reports it so the approximation stays auditable rather than implicit.
        dist, idx = cKDTree(described).query(keypts, k=1)
        self.last_lookup_displacement = float(np.median(dist))

        blocks = []
        max_nbrs = 0
        tree = cKDTree(described)
        for r in self.radii:
            cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
                radius=r * self.normal_radius_frac, max_nn=self.max_nn_normal))
            self._orient_normals(cloud, role)
            fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                cloud, o3d.geometry.KDTreeSearchParamHybrid(
                    radius=r, max_nn=self.max_nn_feature))
            blocks.append(np.asarray(fpfh.data).T[idx])  # (N, 33)
            # Over EVERY described point, not a stride sample: the cap binds
            # exactly where the cloud is locally dense, which is what a strided
            # probe is most likely to step over.
            max_nbrs = max(max_nbrs, int(
                tree.query_ball_point(described, r, return_length=True).max()))
        self.last_cap_binding = (max_nbrs, self.max_nn_feature)

        feats = np.concatenate(blocks, axis=1).astype(np.float32)

        # A point with an empty histogram at ANY scale is undescribed, not
        # "described as zero": mark the whole row invalid so fusion drops it
        # instead of matching an all-zero vector against everything.
        per_scale_norm = np.stack(
            [np.linalg.norm(b, axis=1) for b in blocks], axis=1)
        valid = (per_scale_norm > 1e-8).all(axis=1) & np.isfinite(feats).all(axis=1)
        out[valid] = feats[valid]
        return out


def load_fpfh() -> FPFHDescriptor:
    """Env-configured FPFH, matching how the other backbones are selected.

    POPOE_FPFH_RADII        default "0.3,0.4" (two-scale, mirrors GeDi r_lrf).
                            Pass a single value for the cost-matched
                            single-scale control.
    POPOE_FPFH_VOXEL_FRAC   density-normalising voxel / largest radius,
                            default 0.1; 0 disables (see FPFHDescriptor).
    POPOE_FPFH_NORMAL_FRAC  normal radius / feature radius, default 1/3.
    POPOE_FPFH_ORIENT       auto | outward | camera | none, default auto.
                            Governs role-less calls only — the pipeline's
                            declared role wins (see resolve_orient).
    """
    radii = tuple(float(x) for x in
                  os.environ.get("POPOE_FPFH_RADII", "0.3,0.4").split(","))
    return FPFHDescriptor(
        radii=radii,
        voxel_frac=float(os.environ.get("POPOE_FPFH_VOXEL_FRAC", "0.1")),
        normal_radius_frac=float(os.environ.get("POPOE_FPFH_NORMAL_FRAC",
                                                str(1.0 / 3))),
        orient=os.environ.get("POPOE_FPFH_ORIENT", "auto"),
    )

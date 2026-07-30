"""
FreeZeV2 feature extraction module.
Visual features: DINOv2 (frozen)
Geometric features: GeDi from /workspace/gedi
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from popoe.descriptors import describe
from popoe.freeze.fusion import DinoGeDiFusion
from popoe.interfaces import CanonFrame

# cuDNN fails to initialize on this host; fall back to native CUDA kernels
torch.backends.cudnn.enabled = False

sys.path.insert(0, os.environ.get('POPOE_GEDI_PATH', '/workspace/gedi'))
try:
    from gedi import GeDi
except ImportError as _gedi_import_error:  # pragma: no cover - pod always has it
    # Deferred, not silenced. The GeDi checkout is a pod-side dependency, but
    # this module also holds pure-CPU logic (the renderer's shading resolver and
    # its cache-key parts) that must be importable and testable on a box without
    # it. _make_gedi_single re-raises on first use, so a real run still fails
    # loudly at load time rather than producing features from nothing.
    GeDi = None
    _GEDI_IMPORT_ERROR = _gedi_import_error
else:
    _GEDI_IMPORT_ERROR = None

DINO_DIM = 1536   # ViT-g/14 output dim
GEO_DIM = 32      # GeDi descriptor dim


def _dino_layer(dino):
    """Block index for patch features. FoundPose (arXiv 2311.18809) uses ViT-L layer
    18/24 (~0.78 depth): it balances positional + semantic info and gives more
    geometrically-consistent correspondences than later layers. We default to the same
    depth ratio on ViT-g (40 blocks -> 30); previously used n_blocks-2 (=38) which is
    exactly the "later layer" FoundPose warns against. Override: POPOE_DINO_LAYER."""
    n = dino.n_blocks
    env = os.environ.get("POPOE_DINO_LAYER")
    if env is not None:
        return max(0, min(n - 1, int(env)))
    return max(0, min(n - 1, round(0.78 * (n - 1))))


def icosphere_directions(n_subdiv: int = 2) -> "np.ndarray":
    """Unit view directions from a subdivided icosahedron: 12 -> 42 -> 162."""
    t = (1 + 5 ** 0.5) / 2
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
             (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
             (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10),
             (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
             (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2),
             (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
             (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [np.asarray(x, dtype=float) for x in verts]
    v = [x / np.linalg.norm(x) for x in v]
    for _ in range(n_subdiv):
        mid, nf = {}, []

        def mp(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                m = v[a] + v[b]
                v.append(m / np.linalg.norm(m))
                mid[k] = len(v) - 1
            return mid[k]

        for a, b, c in faces:
            ab, bc, ca = mp(a, b), mp(b, c), mp(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = nf
    return np.stack(v)


def load_dinov2(device='cuda'):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg', pretrained=True)
    model = model.to(device).eval()
    return model


def _make_gedi_single(r_lrf):
    if GeDi is None:
        raise ImportError(
            "the GeDi checkout is not importable, so the geometric branch "
            "cannot be built. Set POPOE_GEDI_PATH to it (default "
            "/workspace/gedi)."
        ) from _GEDI_IMPORT_ERROR
    cfg = {
        'dim': GEO_DIM,
        'samples_per_batch': 500,
        'samples_per_patch_lrf': 4000,
        'samples_per_patch_out': 512,
        'r_lrf': r_lrf,
        'fchkpt_gedi_net': os.environ.get('POPOE_GEDI_PATH', '/workspace/gedi') + '/data/chkpts/3dmatch/chkpt.tar',
    }
    return GeDi(cfg)


class _TwoScaleGeDi:
    """Wraps two GeDi instances at different r_lrf, concat their outputs.
    Mimics paper's 30%+40% diameter two-scale geometric features (64D total).
    """
    def __init__(self, r_a=0.3, r_b=0.4):
        self.a = _make_gedi_single(r_a)
        self.b = _make_gedi_single(r_b)

    def compute(self, pts, pcd):
        import numpy as np
        fa = self.a.compute(pts, pcd)
        fb = self.b.compute(pts, pcd)
        if hasattr(fa, "cpu"): fa = fa.cpu().numpy()
        if hasattr(fb, "cpu"): fb = fb.cpu().numpy()
        return np.concatenate([fa, fb], axis=1).astype(np.float32)


def load_gedi(device='cuda'):
    """Build the geometric branch (a popoe.interfaces.PointDescriptor).

    POPOE_GEOM_BACKBONE selects it: gedi (default) | fpfh | dgedi. The name is
    historical — prefer the `load_geometric_descriptor` alias in new code.
    """
    import os, sys
    backend = os.environ.get('POPOE_GEOM_BACKBONE', 'gedi').lower()
    if backend == 'fpfh':
        # Hand-crafted control: no training, no GPU, radii in the same canonical
        # units as r_lrf below, so it sees the same neighbourhoods GeDi does.
        from popoe.descriptors import load_fpfh
        return load_fpfh()
    if backend == 'dgedi':
        sys.path.insert(0, '/workspace/freezev2')
        from dgedi_adapter import load_dgedi
        mode = os.environ.get('POPOE_DGEDI_MODE', 'single_scale')
        ckpt = f'/workspace/dGeDi/checkpoints/dGeDi_{mode}.pth'
        return load_dgedi(ckpt, mode=mode, device=device, enable_flash=False)
    # Paper (v2.1 §IV.A) default: two-scale GeDi at 30%+40% of object diameter,
    # concatenated → 64D geometric. Set POPOE_TWO_SCALE_GEDI=0 to fall back to
    # single-scale 32D (r_lrf=0.5) for ablation.
    if os.environ.get('POPOE_TWO_SCALE_GEDI', '1') == '0':
        return _make_gedi_single(0.5)
    return _TwoScaleGeDi(r_a=0.3, r_b=0.4)


# Backend-neutral name. `load_gedi` predates dGeDi and FPFH and now returns
# whichever descriptor POPOE_GEOM_BACKBONE names; both spellings are kept so
# existing callers (recipes, monolith, eval scripts) stay valid.
load_geometric_descriptor = load_gedi


# --- how a query mesh gets its colour -------------------------------------
#
# These names go into the query cache key, so a value here changing means the
# rendered pixels changed. Do not rename one without bumping the key tag in
# mesh_shading_key_parts().
SHADING_UV = "uv"                 # UV atlas + image (YCB-V)
SHADING_VERTEX_COLOR = "vcolor"   # property uchar red/green/blue (LM-O, TUD-L, IC-BIN, HB)
SHADING_FACE_COLOR = "fcolor"     # per-face colours
SHADING_FLAT = "flat"             # no colour at all (T-LESS models_cad, ITODD)


def resolve_mesh_shading(mesh) -> str:
    """Which colour source a query mesh actually carries.

    Until 2026-07-28 the dispatch asked only `uv is not None and image is not
    None`, so a mesh whose colour lives in `property uchar red/green/blue` —
    which is how BOP ships LM-O, TUD-L, IC-BIN and HB — was classified as
    untextured and rendered as flat beige Lambertian. The DINOv2 half of every
    query feature for those datasets was therefore computed on a colourless
    image while the target half saw real RGB photographs.

    POPOE_MESH_SHADING=uv-only restores that behaviour. It exists so the A/B
    that prices this fix can run both arms at ONE commit, and because it
    reproduces the pre-fix cache keys byte for byte (see
    mesh_shading_key_parts), so the old arm still hits the campaign cache.
    """
    vis = getattr(mesh, "visual", None)
    if vis is None:
        return SHADING_FLAT
    uv = getattr(vis, "uv", None)
    mat = getattr(vis, "material", None)
    img = getattr(mat, "image", None) if mat is not None else None
    if uv is not None and img is not None:
        return SHADING_UV
    if os.environ.get("POPOE_MESH_SHADING", "auto").lower() == "uv-only":
        return SHADING_FLAT
    # trimesh's ColorVisuals.kind is None when the colours are its fabricated
    # default, and 'vertex'/'face' only when the file actually carried them —
    # which is the distinction we need, since a default-grey ColorVisuals must
    # keep taking the flat path (and keep its old cache key).
    kind = getattr(vis, "kind", None)
    if kind == "vertex":
        return SHADING_VERTEX_COLOR
    if kind == "face":
        return SHADING_FACE_COLOR
    return SHADING_FLAT


def mesh_shading_key_parts(mesh_path) -> tuple:
    """Extra query-cache-key parts for a mesh whose rendered pixels changed.

    EMPTY for UV-atlas and colourless meshes: nothing about their render moved,
    so appending a part would only throw away a valid cache (YCB-V's query and
    target features cost hours). Non-empty exactly for the meshes the old
    dispatch mis-classified — their cached DINOv2 features were computed on a
    grey render and must NOT be reused.

    This is deliberately not an `enc_cfg` entry: enc_cfg is per RUN, and which
    shading applies is per MESH. It follows the same rule as the FPFH knobs in
    examples/bop_eval.py — add key material only where the behaviour actually
    differs, so unaffected entries survive.

    The `-v1` is a version, not decoration: a future change to how vertex
    colours are rendered has to bump it, because the mode name alone would
    still be "vcolor" and the stale features would be served again.
    """
    import trimesh
    mode = resolve_mesh_shading(trimesh.load(str(mesh_path), force="mesh"))
    if mode in (SHADING_UV, SHADING_FLAT):
        return ()
    return (f"shading={mode}-v1",)


class QueryFeatureExtractor:
    """Extract fused DINOv2+GeDi features from 3D query model (offline, precomputed).

    The query side RENDERS the CAD model, so which renderer ran is part of this
    stage's configuration: nvdiffrast and the trimesh ray-caster produce
    different images and therefore different DINOv2 features. Read it back from
    `render_backend` and put it in the cache key — an entry computed on a
    CPU-only box is NOT interchangeable with one computed on a GPU box, and
    nothing else in the key distinguishes them (see cache.py).

    `render_backend='nvdiffrast'` demands the GPU rasteriser and raises
    RendererUnavailable without it, instead of silently running ~100x slower on
    a different method. 'auto' (default) keeps the historical preference.
    """

    def __init__(self, device='cuda', dino=None, gedi=None, render_backend='auto'):
        if render_backend not in ('auto', 'nvdiffrast', 'trimesh'):
            raise ValueError(f"unknown render_backend: {render_backend!r}")
        self.device = device
        self.dino = dino if dino is not None else load_dinov2(device)
        self.gedi = gedi if gedi is not None else load_gedi(device)
        # Fusion is a swappable component (popoe/freeze/fusion.py). It owns the
        # per-object visual PCA; `_pca_vis` below proxies it so external callers
        # (e.g. pose_estimator sharing query PCA -> target) keep working.
        self.fusion = DinoGeDiFusion()
        self._render_backend_pref = render_backend
        self._nvd_ctx = None
        self._nvd_init_tried = False

    @property
    def render_backend(self) -> str:
        """'nvdiffrast' or 'trimesh' — the renderer that WILL run (resolved on
        first access). Put this in the cache key."""
        return 'nvdiffrast' if self._init_nvdiffrast() else 'trimesh'

    @property
    def _pca_vis(self):
        return self.fusion.pca_vis

    @_pca_vis.setter
    def _pca_vis(self, value):
        self.fusion.pca_vis = value

    @property
    def canon_frame(self) -> CanonFrame:
        """Single named home for the canonicalisation convention (see #2 in
        ARCHITECTURE.md). Surfaces the existing `_canon_scale` (set during
        extract_query_features); center=0 because the code scales without
        centring. Additive/read-only — does not change existing behaviour."""
        return CanonFrame(center=np.zeros(3, np.float32),
                          scale=float(getattr(self, "_canon_scale", 1.0)))

    def _sample_query_pointcloud(self, mesh_path, n_raw=50000, n_views=18, min_views=1):
        """Poisson disk sample surface points, render from multiple views, keep visible."""
        import trimesh
        import open3d as o3d

        mesh = trimesh.load(mesh_path, force='mesh')
        # Normalise to fit ~50% of 480x480 image
        scale = 0.45 * 480 / max(mesh.extents)
        mesh.apply_scale(scale)
        pts, _ = trimesh.sample.sample_surface_even(mesh, n_raw)

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pts)

        # Simple visibility: keep points visible from at least min_views of 6 canonical views
        canonical_dirs = np.array([
            [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
        ], dtype=np.float32)
        normals = np.asarray(pcd_o3d.normals) if pcd_o3d.has_normals() else None

        pts_tensor = torch.from_numpy(pts.astype(np.float32))
        return pts_tensor

    def _init_nvdiffrast(self):
        """True if the GPU rasteriser will be used. Honours render_backend:
        'trimesh' never tries; 'nvdiffrast' raises rather than degrade."""
        if self._render_backend_pref == 'trimesh':
            return False
        if self._nvd_init_tried:
            return self._nvd_ctx is not None
        self._nvd_init_tried = True
        try:
            import nvdiffrast.torch as dr
            self._nvd_ctx = dr.RasterizeCudaContext(device=self.device)
            return True
        except Exception as e:
            from popoe.interfaces import is_runtime_failure
            if is_runtime_failure(e):
                # A full GPU is not a missing rasteriser. Degrading to the CPU
                # ray-caster here would silently change the rendered views, and
                # therefore every query feature, on an 'auto' run.
                #
                # Un-latch the probe before re-raising: `_nvd_init_tried` was
                # set BEFORE the attempt, so leaving it True would make every
                # later call answer "already tried, no context" and quietly use
                # trimesh — reintroducing the silent degrade one retry later,
                # after the caller freed VRAM and tried again.
                self._nvd_init_tried = False
                raise
            self._nvd_ctx = None
            if self._render_backend_pref == 'nvdiffrast':
                from popoe.renderer import RendererUnavailable
                raise RendererUnavailable(
                    f"render_backend='nvdiffrast' was required but it is unusable "
                    f"({type(e).__name__}: {e}).\n"
                    f"  pip install --no-build-isolation "
                    f"git+https://github.com/NVlabs/nvdiffrast.git\n"
                    f"Pass render_backend='auto' to accept the CPU ray-caster "
                    f"instead — it renders DIFFERENT images, so it yields "
                    f"DIFFERENT features.") from e
            bar = "!" * 80
            print(bar, flush=True)
            print("!! nvdiffrast NOT AVAILABLE — falling back to trimesh CPU raycast.", flush=True)
            print(f"!! Reason: {type(e).__name__}: {e}", flush=True)
            print("!! Renders differ from GPU -> query FEATURES differ. The cache key", flush=True)
            print("!! records render_backend, so this will not reuse GPU entries.", flush=True)
            print("!! LMO eval will be ~100x slower (10+ min per image with 18 views).", flush=True)
            print("!! Fix: pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git", flush=True)
            print("!! Or pass render_backend='nvdiffrast' to make this an error.", flush=True)
            print(bar, flush=True)
            return False

    def _nvdiffrast_render(self, mesh, cam_pos, H, W, fov_deg, shading=None):
        """GPU render via nvdiffrast, matching FE's camera convention.

        FE projection (used downstream): col = x_cam/z_cam*fx+cx, row = y_cam/z_cam*fy+cy.

        `shading` picks the ALBEDO only; the Lambertian term is the same in every
        case, so a coloured mesh renders exactly as it did before except that the
        fixed beige is replaced by the model's own colour. SHADING_UV never
        reaches here — it has its own function (texture sampling needs a v-flip
        that vertex attributes do not).
        """
        import nvdiffrast.torch as dr
        import math, PIL.Image

        if shading is None:
            shading = resolve_mesh_shading(mesh)
        V_np = np.asarray(mesh.vertices, dtype=np.float32)
        F_np = np.asarray(mesh.faces, dtype=np.int32)

        # Camera basis (must match FE's downstream projection code exactly)
        forward = (np.zeros(3) - cam_pos)
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        up_ref = np.array([0., 1., 0.]) if abs(forward[1]) < 0.9 else np.array([1., 0., 0.])
        right = np.cross(forward, up_ref); right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R_cw = np.stack([right, up, forward], axis=0).astype(np.float32)

        V_cam = (V_np - cam_pos.astype(np.float32)) @ R_cw.T  # (V, 3)

        fy = fx = (H / 2) / math.tan(math.radians(fov_deg) / 2)
        cx, cy = W / 2.0, H / 2.0
        # far must exceed max |z_cam|; LMO meshes scaled to ~100 units with cam ~150 out
        near, far = 0.001, 1.0e6

        # Clip-space so that rasterized pixel (row, col) matches FE projection.
        # nvdiffrast: y_ndc=+1 at BOTTOM row (verified empirically), y_ndc=-1 at TOP.
        # FE: v = y_cam/z*fy + cy grows DOWN with y_cam. So y_cam>0 → larger row → y_ndc>0.
        # => y_clip = +(2*fy/H) * y_cam
        x_clip = (2.0 * fx / W) * V_cam[:, 0]
        y_clip = (2.0 * fy / H) * V_cam[:, 1]
        w_clip = V_cam[:, 2]
        z_clip = ((far + near) / (far - near)) * V_cam[:, 2] - (2 * far * near) / (far - near)
        V_clip = np.stack([x_clip, y_clip, z_clip, w_clip], axis=1).astype(np.float32)

        pos = torch.from_numpy(V_clip).unsqueeze(0).to(self.device).contiguous()
        tri = torch.from_numpy(F_np).to(self.device).contiguous()

        rast, _ = dr.rasterize(self._nvd_ctx, pos, tri, resolution=[H, W])  # (1,H,W,4)
        hit = rast[0, :, :, 3] > 0

        # Per-vertex normals for Lambertian
        norms = np.zeros((len(V_np), 3), dtype=np.float64)
        fn = np.cross(V_np[F_np[:, 1]] - V_np[F_np[:, 0]],
                      V_np[F_np[:, 2]] - V_np[F_np[:, 0]])
        for i in range(3):
            np.add.at(norms, F_np[:, i], fn)
        nlen = np.linalg.norm(norms, axis=1, keepdims=True) + 1e-8
        vnorm = (norms / nlen).astype(np.float32)
        vnorm_t = torch.from_numpy(vnorm).unsqueeze(0).to(self.device)
        nrm_interp, _ = dr.interpolate(vnorm_t, rast, tri)  # (1,H,W,3)

        light_dir = torch.tensor(-forward.astype(np.float32), device=self.device)
        nrm_n = torch.nn.functional.normalize(nrm_interp[0], dim=-1)
        lambert = torch.clamp((nrm_n * light_dir).sum(dim=-1), 0.1, 1.0)  # (H,W)

        # Albedo. The default is the historical fixed beige; a mesh that carries
        # its own colour uses that instead. Note the colour arrays are indexed by
        # the SAME vertex/face arrays that were rasterised — trimesh keeps
        # visual.vertex_colors aligned with .vertices through load/scale/
        # translate — so an alignment slip would be a length mismatch, which the
        # asserts below turn into a crash rather than a scrambled render.
        if shading == SHADING_VERTEX_COLOR:
            vcol = np.asarray(mesh.visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
            assert len(vcol) == len(V_np), (
                f"vertex_colors {len(vcol)} != vertices {len(V_np)}")
            vcol_t = torch.from_numpy(np.ascontiguousarray(vcol)).unsqueeze(0).to(self.device)
            base, _ = dr.interpolate(vcol_t, rast, tri)  # (1,H,W,3)
            base = base[0]
        elif shading == SHADING_FACE_COLOR:
            fcol = np.asarray(mesh.visual.face_colors, dtype=np.float32)[:, :3] / 255.0
            assert len(fcol) == len(F_np), (
                f"face_colors {len(fcol)} != faces {len(F_np)}")
            fcol_t = torch.from_numpy(np.ascontiguousarray(fcol)).to(self.device)
            # rast[..., 3] is triangle_id + 1, 0 for background; clamp keeps the
            # background lookup in range (it is masked out by `hit` below).
            tri_id = (rast[0, :, :, 3].long() - 1).clamp(min=0)
            base = fcol_t[tri_id]  # (H,W,3)
        else:
            base = torch.tensor([180.0, 160.0, 140.0], device=self.device) / 255.0
        color = lambert.unsqueeze(-1) * base  # (H,W,3)
        bg = torch.tensor([200.0 / 255.0, 200.0 / 255.0, 200.0 / 255.0], device=self.device)
        color = torch.where(hit.unsqueeze(-1), color, bg)
        rgb = (color * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

        # Depth = z_cam at each pixel (used downstream only as >0 hit test)
        zcam_t = torch.from_numpy(V_cam[:, 2:3].astype(np.float32)).unsqueeze(0).to(self.device)
        zcam_interp, _ = dr.interpolate(zcam_t, rast, tri)
        depth = zcam_interp[0, :, :, 0]
        depth = torch.where(hit, depth, torch.zeros_like(depth)).cpu().numpy().astype(np.float32)

        return PIL.Image.fromarray(rgb), depth, fx, fy, cx, cy

    def _trimesh_render(self, mesh, cam_pos, H, W, fov_deg, shading=None):
        """CPU fallback — trimesh ray casting."""
        import math, PIL.Image
        if shading is None:
            shading = resolve_mesh_shading(mesh)
        center = np.zeros(3)
        forward = (center - cam_pos)
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        up_ref = np.array([0., 1., 0.]) if abs(forward[1]) < 0.9 else np.array([1.,0.,0.])
        right = np.cross(forward, up_ref); right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        fy = fx = (H / 2) / math.tan(math.radians(fov_deg) / 2)
        cx, cy = W / 2.0, H / 2.0

        py_grid, px_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        dx = (px_grid.flatten() - cx) / fx
        dy = (py_grid.flatten() - cy) / fy
        dirs_local = np.stack([dx, dy, np.ones_like(dx)], axis=1)
        dirs_world = dirs_local[:,0:1]*right + dirs_local[:,1:2]*up + dirs_local[:,2:3]*forward
        dirs_world = dirs_world / (np.linalg.norm(dirs_world, axis=1, keepdims=True) + 1e-8)

        origins = np.tile(cam_pos, (H*W, 1))
        locs, idx_ray, idx_tri = mesh.ray.intersects_location(origins, dirs_world, multiple_hits=False)

        rgb = np.ones((H*W, 3), dtype=np.uint8) * 200
        depth_map = np.zeros(H*W, dtype=np.float32)
        if len(locs) > 0:
            normals = mesh.face_normals[idx_tri]
            light_dir = -forward
            lambert = np.clip((normals * light_dir).sum(axis=1), 0.1, 1.0)
            # Same albedo rule as the nvdiffrast path, so the two backends agree
            # about what a colour mode MEANS. (They still differ numerically —
            # ray casting vs rasterising — which is why render_backend is in the
            # cache key.) Flat per-hit-triangle albedo: this path has no
            # barycentrics, so a vertex-coloured mesh is averaged per face.
            if shading == SHADING_VERTEX_COLOR:
                vcol = np.asarray(mesh.visual.vertex_colors,
                                  dtype=np.float32)[:, :3]
                base_color = vcol[mesh.faces[idx_tri]].mean(axis=1)  # (hits,3)
            elif shading == SHADING_FACE_COLOR:
                base_color = np.asarray(mesh.visual.face_colors,
                                        dtype=np.float32)[idx_tri, :3]
            else:
                base_color = np.array([[180, 160, 140]], dtype=np.float32)
            colors = (base_color * lambert[:, None]).astype(np.uint8)
            rgb[idx_ray] = colors
            depths = np.linalg.norm(locs - cam_pos, axis=1)
            depth_map[idx_ray] = depths

        rgb = rgb.reshape(H, W, 3)
        depth_map = depth_map.reshape(H, W)
        return PIL.Image.fromarray(rgb), depth_map, fx, fy, cx, cy

    def _nvdiffrast_render_textured(self, mesh, cam_pos, H, W, fov_deg):
        """Textured GPU render: same clip-space as _nvdiffrast_render, but samples
        vertex-UV texture instead of Lambertian lighting. For meshes with
        TextureVisuals (YCB-V has 4096x4096 maps per obj)."""
        import nvdiffrast.torch as dr
        import math, PIL.Image

        V_np = np.asarray(mesh.vertices, dtype=np.float32)
        F_np = np.asarray(mesh.faces, dtype=np.int32)
        uvs_np = np.asarray(mesh.visual.uv, dtype=np.float32)
        tex_img = np.asarray(mesh.visual.material.image, dtype=np.float32) / 255.0
        if tex_img.ndim == 2:
            tex_img = np.stack([tex_img] * 3, axis=-1)
        elif tex_img.shape[-1] == 4:
            tex_img = tex_img[..., :3]

        forward = (np.zeros(3) - cam_pos)
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        up_ref = np.array([0., 1., 0.]) if abs(forward[1]) < 0.9 else np.array([1., 0., 0.])
        right = np.cross(forward, up_ref); right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R_cw = np.stack([right, up, forward], axis=0).astype(np.float32)
        V_cam = (V_np - cam_pos.astype(np.float32)) @ R_cw.T

        fy = fx = (H / 2) / math.tan(math.radians(fov_deg) / 2)
        cx, cy = W / 2.0, H / 2.0
        near, far = 0.001, 1.0e6

        x_clip = (2.0 * fx / W) * V_cam[:, 0]
        y_clip = (2.0 * fy / H) * V_cam[:, 1]
        w_clip = V_cam[:, 2]
        z_clip = ((far + near) / (far - near)) * V_cam[:, 2] - (2 * far * near) / (far - near)
        V_clip = np.stack([x_clip, y_clip, z_clip, w_clip], axis=1).astype(np.float32)

        pos = torch.from_numpy(V_clip).unsqueeze(0).to(self.device).contiguous()
        tri = torch.from_numpy(F_np).to(self.device).contiguous()

        rast, _ = dr.rasterize(self._nvd_ctx, pos, tri, resolution=[H, W])
        hit = rast[0, :, :, 3] > 0

        # nvdiffrast dr.texture samples with v=0 at the FIRST texture row, but
        # PLY/PIL texture maps use v=0 at the BOTTOM (OpenGL/OBJ convention), so
        # the v axis must be flipped. Without this, YCB-V textured renders come
        # out vertically mirrored — text upside-down and UVs bleeding into the
        # unused (black) half of the 4096² atlas. LMO meshes are untextured so
        # they take the Lambertian path and never hit this (bug found 2026-07-06).
        uvs_np = uvs_np.copy()
        uvs_np[:, 1] = 1.0 - uvs_np[:, 1]
        uv_t = torch.from_numpy(uvs_np).unsqueeze(0).to(self.device)  # (1, V, 2)
        uv_interp, _ = dr.interpolate(uv_t, rast, tri)  # (1, H, W, 2)
        # nvdiffrast dr.texture expects uv in [0,1] and tex as (N, H_tex, W_tex, C)
        tex_t = torch.from_numpy(tex_img).unsqueeze(0).to(self.device).contiguous()  # (1, Ht, Wt, 3)
        rgb_t = dr.texture(tex_t, uv_interp, filter_mode='linear')  # (1, H, W, 3)
        color = rgb_t[0]
        bg = torch.tensor([200.0/255.0, 200.0/255.0, 200.0/255.0], device=self.device)
        color = torch.where(hit.unsqueeze(-1), color, bg)
        rgb = (color * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

        zcam_t = torch.from_numpy(V_cam[:, 2:3].astype(np.float32)).unsqueeze(0).to(self.device)
        zcam_interp, _ = dr.interpolate(zcam_t, rast, tri)
        depth = zcam_interp[0, :, :, 0]
        depth = torch.where(hit, depth, torch.zeros_like(depth)).cpu().numpy().astype(np.float32)

        return PIL.Image.fromarray(rgb), depth, fx, fy, cx, cy

    @staticmethod
    def _mesh_has_texture(mesh):
        """Deprecated: kept for callers outside this module. It answers "is
        there a UV atlas", which is NOT the same question as "does this mesh
        carry colour" — see resolve_mesh_shading."""
        return resolve_mesh_shading(mesh) == SHADING_UV

    def _raycast_render(self, mesh, cam_pos, H=224, W=224, fov_deg=60.0):
        """Render mesh from cam_pos. Returns (PIL.Image, depth, fx, fy, cx, cy).
        Dispatches on the mesh's colour source (resolve_mesh_shading): UV atlas
        -> textured sampler; vertex/face colour or none -> Lambertian with the
        matching albedo; no nvdiffrast -> trimesh CPU ray caster."""
        shading = resolve_mesh_shading(mesh)
        if self._init_nvdiffrast():
            if shading == SHADING_UV:
                return self._nvdiffrast_render_textured(mesh, cam_pos, H, W, fov_deg)
            return self._nvdiffrast_render(mesh, cam_pos, H, W, fov_deg, shading)
        return self._trimesh_render(mesh, cam_pos, H, W, fov_deg, shading)

    @torch.no_grad()
    def extract_query_features(self, mesh_path, pts_query, n_views=None):
        if n_views is None:
            n_views = int(os.environ.get("POPOE_N_VIEWS", "162"))  # paper: 162
        """
        Returns fused feature point cloud F_Q with actual DINOv2+GeDi features.
        pts_query: (N, 3) tensor
        """
        import trimesh, math
        from torchvision import transforms

        pcd_np = pts_query.numpy() if isinstance(pts_query, torch.Tensor) else pts_query
        pts_np = pcd_np

        # Load and normalise mesh (mesh is in mm, pts_np is in metres)
        mesh = trimesh.load(mesh_path, force='mesh')
        # Render canvas and fill fraction. Historical values (224, 0.45) are the
        # DINOv2 native canvas; FreeZeV2 Sec. IV-A renders at 480x480 with the
        # object at ~50% ("occupies approximately 50% of the width and height in
        # a 480x480 image"). 480 is not a multiple of the 14px patch, so a
        # faithful run passes 476 (= 34*14). Both knobs are cache keys in
        # bop_eval's enc_cfg — set them there or features go stale silently.
        canon = (int(os.environ.get("POPOE_QUERY_CANON", "224")) // 14) * 14
        fill = float(os.environ.get("POPOE_QUERY_FILL", "0.45"))
        scale = fill * canon / max(mesh.extents)
        mesh.apply_scale(scale)
        center = mesh.bounds.mean(0)
        mesh.apply_translation(-center)
        mesh_pts = (pts_np * 1000.0 * scale) - center  # convert m→mm before scaling

        # Canonical scale for GeDi: r_lrf=0.5m was trained on ~1m scenes; rescale
        # object to ~1m extent before GeDi (keeps features local, not whole-object).
        #
        # What "1m" is measured against is a paper-fidelity knob. Historical
        # ("extent"): the sampled cloud's largest bounding-box side. FreeZeV2
        # Sec. IV-A: GeDi neighbourhoods occupy "30% and 40% of the object's
        # DIAMETER" — and a bounding-box side runs 2-35% short of the diameter
        # on LM-O (same basis mismatch as tau, but here it shrinks every GeDi
        # neighbourhood). "diameter" uses the max pairwise distance over the
        # mesh's convex-hull vertices — the BOP models_info definition, computed
        # from the model itself so nothing external needs threading in.
        extent_m = float(np.ptp(pts_np, axis=0).max())
        if os.environ.get("POPOE_CANON_BASIS", "extent") == "diameter":
            hull = mesh.convex_hull.vertices / (1000.0 * scale)  # back to metres
            # Exact diameter: the farthest pair of a point set lies on its hull.
            d2 = 0.0
            for i in range(0, len(hull), 512):
                chunk = hull[i:i + 512]
                d2 = max(d2, float(((chunk[:, None, :] - hull[None, :, :]) ** 2)
                                   .sum(-1).max()))
            basis_m = math.sqrt(d2)
        else:
            basis_m = extent_m
        self._canon_scale = 1.0 / max(basis_m, 1e-6)
        geo_input = (pts_np * self._canon_scale).astype(np.float32)
        # role="query": a CAD model sampled all around. Role-blind backbones
        # (GeDi, dGeDi) ignore it; FPFH needs it to pick the normal convention,
        # which cannot be inferred from the points (see popoe.descriptors).
        geo_feats = describe(
            self.gedi,
            torch.from_numpy(geo_input),
            torch.from_numpy(geo_input),
            role="query",
        )  # (N, 32)

        # Visual features via DINOv2 on rendered views, aggregated per point
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

        H, W = canon, canon
        vis_feats_acc = np.zeros((len(pts_np), DINO_DIM), dtype=np.float64)
        vis_counts = np.zeros(len(pts_np), dtype=np.int32)

        golden = (1 + math.sqrt(5)) / 2
        radius_cam = max(mesh.extents) * 1.5

        layout = os.environ.get("POPOE_QUERY_VIEWS", "spiral")
        if layout == "ico162":
            ico = icosphere_directions(2)
            if n_views != len(ico):
                raise RuntimeError(
                    f"POPOE_QUERY_VIEWS=ico162 provides exactly {len(ico)} "
                    f"directions but n_views={n_views}; set POPOE_N_VIEWS=162 "
                    f"or drop the layout override.")
        elif layout != "spiral":
            raise RuntimeError(f"unknown POPOE_QUERY_VIEWS {layout!r}")

        for i in range(n_views):
            if layout == "ico162":
                cam_pos = radius_cam * ico[i]
            else:
                theta = math.acos(1 - 2*(i+0.5)/n_views)
                phi = 2*math.pi*i/golden
                cam_pos = radius_cam * np.array([
                    math.sin(theta)*math.cos(phi),
                    math.sin(theta)*math.sin(phi),
                    math.cos(theta)
                ])

            img_pil, depth_render, fx, fy, cx, cy = self._raycast_render(mesh, cam_pos, H, W)

            # DINOv2 forward pass
            img_t = transform(img_pil).unsqueeze(0).to(self.device)
            feat_out = self.dino.get_intermediate_layers(img_t, n=[_dino_layer(self.dino)], return_class_token=False)[0]
            n_ph, n_pw = H//14, W//14
            feat_map = feat_out[0].reshape(n_ph, n_pw, -1).cpu().numpy()

            # Project each 3D query point to this view and sample feature
            fwd = -cam_pos / np.linalg.norm(cam_pos)
            up_ref = np.array([0.,1.,0.]) if abs(fwd[1]) < 0.9 else np.array([1.,0.,0.])
            right = np.cross(fwd, up_ref); right /= np.linalg.norm(right)
            up = np.cross(right, fwd)
            R_cw = np.stack([right, up, fwd], axis=0)  # world->cam

            pts_cam = (R_cw @ (mesh_pts - cam_pos).T).T
            visible = pts_cam[:, 2] > 0

            u = (pts_cam[visible, 0] / pts_cam[visible, 2]) * fx + cx
            v = (pts_cam[visible, 1] / pts_cam[visible, 2]) * fy + cy

            # Check not occluded via depth test
            v_int = np.clip(v.astype(int), 0, H-1)
            u_int = np.clip(u.astype(int), 0, W-1)
            rendered_d = depth_render[v_int, u_int]
            actual_z = pts_cam[visible, 2]  # z_cam, same units as depth_render
            tol = 0.05 * float(np.ptp(mesh_pts, axis=0).max())
            not_occluded = (rendered_d > 0) & (np.abs(rendered_d - actual_z) < tol)

            vis_idx = np.where(visible)[0][not_occluded]
            u_ok = u[not_occluded]
            v_ok = v[not_occluded]

            # Bilinear DINO patch sampling (paper sec 3.1)
            up = u_ok * n_pw / W - 0.5  # patch coord, 0.5 offset for patch center
            vp = v_ok * n_ph / H - 0.5
            i0 = np.clip(np.floor(vp).astype(int), 0, n_ph - 1)
            j0 = np.clip(np.floor(up).astype(int), 0, n_pw - 1)
            i1 = np.clip(i0 + 1, 0, n_ph - 1)
            j1 = np.clip(j0 + 1, 0, n_pw - 1)
            di = np.clip(vp - i0, 0, 1)[:, None]
            dj = np.clip(up - j0, 0, 1)[:, None]
            f00 = feat_map[i0, j0]; f01 = feat_map[i0, j1]
            f10 = feat_map[i1, j0]; f11 = feat_map[i1, j1]
            sampled = ((1 - di) * (1 - dj)) * f00 + ((1 - di) * dj) * f01 + \
                      (di * (1 - dj)) * f10 + (di * dj) * f11
            vis_feats_acc[vis_idx] += sampled
            vis_counts[vis_idx] += 1

        # Average over views
        seen = vis_counts > 0
        vis_feats = np.zeros((len(pts_np), DINO_DIM), dtype=np.float32)
        vis_feats[seen] = (vis_feats_acc[seen] / vis_counts[seen, None]).astype(np.float32)

        # Visibility gate. FreeZeV2 Sec. IV-A: "we filter P_Q^raw by keeping
        # only the points that are visible from at least V = 18 views". The
        # historical behaviour (0) keeps every point — one seen from no view
        # keeps an all-zero visual vector, one seen once keeps a single grazing
        # view's features. Measured on LM-O the paper's gate touches only ~2.5%
        # of a 3000-point cloud, so this is a fidelity knob, not an expected
        # lever. Applied to points, features and the returned cloud together so
        # the solver sees the same set the features describe; the PCA below is
        # then fitted on the survivors only.
        min_views = int(os.environ.get("POPOE_QUERY_MIN_VIEWS", "0"))
        if min_views > 0:
            keep = vis_counts >= min_views
            if not keep.any():
                raise RuntimeError(
                    f"POPOE_QUERY_MIN_VIEWS={min_views} removed every query "
                    f"point (max count {int(vis_counts.max())} of {n_views} "
                    f"views) — the gate is misconfigured, not the mesh.")
            vis_feats, geo_feats = vis_feats[keep], geo_feats[keep]
            pts_query = pts_query[torch.from_numpy(keep)] \
                if isinstance(pts_query, torch.Tensor) else pts_query[keep]
            print(f"    visibility gate >= {min_views}/{n_views} views: "
                  f"kept {int(keep.sum())}/{len(keep)} points", flush=True)

        # Fit PCA on actual features
        fused = self._fuse_features(vis_feats, geo_feats)
        return fused, pts_query

    def _fuse_features(self, vis_feats, geo_feats):
        """Eq. (1)/(2): f = [w*norm(PCA(f_vis)), norm(f_geo)]. Delegates to the
        swappable fusion component. Query side honours POPOE_SKIP_VIS."""
        return self.fusion.fuse(vis_feats, geo_feats, apply_skip_vis=True)


class TargetFeatureExtractor:
    """Extract sparse fused features from RGBD target image (online, per candidate mask)."""

    def __init__(self, device='cuda', dino=None, gedi=None):
        self.device = device
        self.dino = dino if dino is not None else load_dinov2(device)
        self.gedi = gedi if gedi is not None else load_gedi(device)
        # See QueryFeatureExtractor: fusion owns the PCA, `_pca_vis` proxies it.
        self.fusion = DinoGeDiFusion()

    @property
    def _pca_vis(self):
        return self.fusion.pca_vis

    @_pca_vis.setter
    def _pca_vis(self, value):
        self.fusion.pca_vis = value

    @property
    def canon_frame(self) -> CanonFrame:
        """See QueryFeatureExtractor.canon_frame. On the target side `_canon_scale`
        is assigned from the query's value by the caller before extraction."""
        return CanonFrame(center=np.zeros(3, np.float32),
                          scale=float(getattr(self, "_canon_scale", 1.0)))

    @torch.no_grad()
    def extract_target_features(self, rgb, depth, mask, intrinsics, pca_vis=None):
        """
        rgb: (H,W,3) uint8
        depth: (H,W) float32, metres
        mask: (H,W) bool
        intrinsics: dict with fx,fy,cx,cy
        Returns: pts_sparse (N_T,3), feats_fused (N_T, D)
        """
        import cv2

        H, W = depth.shape
        fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']

        # Grid of patch centres within the bounding box of the mask
        grid_size = int(os.environ.get("POPOE_TARGET_GRID", "16"))
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return None, None
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        grid_y = np.linspace(y0, y1, grid_size).astype(int).clip(0, H-1)
        grid_x = np.linspace(x0, x1, grid_size).astype(int).clip(0, W-1)
        gx, gy = np.meshgrid(grid_x, grid_y)
        patch_u = gx.flatten()
        patch_v = gy.flatten()

        # Filter to points inside mask
        in_mask = mask[patch_v, patch_u]
        patch_u = patch_u[in_mask]
        patch_v = patch_v[in_mask]

        # 2D -> 3D backprojection using depth
        d = depth[patch_v, patch_u]
        valid_depth = d > 0
        patch_u = patch_u[valid_depth]
        patch_v = patch_v[valid_depth]
        d = d[valid_depth]

        X = (patch_u - cx) * d / fx
        Y = (patch_v - cy) * d / fy
        pts_sparse = np.stack([X, Y, d], axis=1).astype(np.float32)  # (N_T, 3)

        if len(pts_sparse) < 4:
            return None, None

        # Visual features: extract DINOv2 patch features at these 2D locations
        vis_feats = self._extract_dino_at_points(rgb, patch_u, patch_v)

        # Dense depth point cloud for GeDi neighbourhood
        ys_all, xs_all = np.where((depth > 0) & mask)
        d_all = depth[ys_all, xs_all]
        X_all = (xs_all - cx) * d_all / fx
        Y_all = (ys_all - cy) * d_all / fy
        pcd_dense = np.stack([X_all, Y_all, d_all], axis=1).astype(np.float32)

        canon = getattr(self, '_canon_scale', 1.0)
        # role="target": a single-view depth cloud, camera at the origin.
        geo_feats = describe(
            self.gedi,
            torch.from_numpy((pts_sparse * canon).astype(np.float32)),
            torch.from_numpy((pcd_dense * canon).astype(np.float32)),
            role="target",
        )

        if pca_vis is not None:
            self._pca_vis = pca_vis
        fused = self._fuse_features(vis_feats, geo_feats)
        return pts_sparse, fused

    @torch.no_grad()
    def _extract_dino_at_points(self, rgb, us, vs):
        """Extract DINOv2 patch features at the given pixel locations.
        By default crops a square object region (FoundPose-style crop descriptor) and
        resizes it to a canonical size, so the target object appears at a scale and
        framing comparable to the query renders. Previously DINOv2 ran on the full
        scene image, leaving query (object-filling render) and target (small object
        in clutter) in mismatched feature scales. Set POPOE_TARGET_CROP=0 for the
        legacy full-image behaviour."""
        from torchvision import transforms
        import PIL.Image

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
        h, w = rgb.shape[:2]
        us_arr = np.asarray(us, dtype=np.float64)
        vs_arr = np.asarray(vs, dtype=np.float64)

        if os.environ.get("POPOE_TARGET_CROP", "1") == "1":
            # Square crop so the object fills ~FILL of the frame (matching query render
            # framing), then resize to a canonical patch grid.
            fill = float(os.environ.get("POPOE_TARGET_FILL", "0.5"))
            canon = (int(os.environ.get("POPOE_TARGET_CANON", "224")) // 14) * 14
            x0, x1 = us_arr.min(), us_arr.max()
            y0, y1 = vs_arr.min(), vs_arr.max()
            side = max(max(x1 - x0, y1 - y0) / max(fill, 1e-3), 14.0)
            cxc, cyc = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            bx0, by0 = cxc - side / 2.0, cyc - side / 2.0
            box = (int(round(bx0)), int(round(by0)),
                   int(round(bx0 + side)), int(round(by0 + side)))
            # PIL.crop zero-pads out-of-bounds, keeping the object centred.
            img_pil = PIL.Image.fromarray(rgb).crop(box).resize((canon, canon))
            img_t = transform(img_pil).unsqueeze(0).to(self.device)
            out = self.dino.get_intermediate_layers(img_t, n=[_dino_layer(self.dino)], return_class_token=False)[0]
            n_h = n_w = canon // 14
            feat_map = out[0].reshape(n_h, n_w, -1).cpu().numpy()
            cw, ch = box[2] - box[0], box[3] - box[1]
            up = (us_arr - box[0]) * n_w / cw - 0.5
            vp = (vs_arr - box[1]) * n_h / ch - 0.5
        else:
            new_h = (h // 14) * 14
            new_w = (w // 14) * 14
            img_pil = PIL.Image.fromarray(rgb).resize((new_w, new_h))
            img_t = transform(img_pil).unsqueeze(0).to(self.device)
            out = self.dino.get_intermediate_layers(img_t, n=[_dino_layer(self.dino)], return_class_token=False)[0]
            n_h = new_h // 14
            n_w = new_w // 14
            feat_map = out[0].reshape(n_h, n_w, -1).cpu().numpy()
            up = us_arr * n_w / w - 0.5
            vp = vs_arr * n_h / h - 0.5
        i0 = np.clip(np.floor(vp).astype(int), 0, n_h - 1)
        j0 = np.clip(np.floor(up).astype(int), 0, n_w - 1)
        i1 = np.clip(i0 + 1, 0, n_h - 1)
        j1 = np.clip(j0 + 1, 0, n_w - 1)
        di = np.clip(vp - i0, 0, 1)[:, None]
        dj = np.clip(up - j0, 0, 1)[:, None]
        f00 = feat_map[i0, j0]; f01 = feat_map[i0, j1]
        f10 = feat_map[i1, j0]; f11 = feat_map[i1, j1]
        feats = ((1 - di) * (1 - dj)) * f00 + ((1 - di) * dj) * f01 + \
                (di * (1 - dj)) * f10 + (di * dj) * f11
        return feats.astype(np.float32)

    def _fuse_features(self, vis_feats, geo_feats):
        """Delegates to the swappable fusion component. NOTE: unlike the query
        side, the target side does NOT honour POPOE_SKIP_VIS — preserved
        as-is from the original code (pre-existing asymmetry, see ARCHITECTURE.md)."""
        return self.fusion.fuse(vis_feats, geo_feats, apply_skip_vis=False)

"""
Renderers for FreeZeV2 / CNOS templates.
  nvdiffrast — CUDA rasteriser, headless
  trimesh    — CPU ray caster, headless

They are two METHODS, not a primary and a hidden safety net: the images differ,
so the DINOv2 features derived from them differ. `get_renderer(backend=...)`
makes the choice explicit and `renderer.source` reports what you got — put it in
the cache key (see cache.py).
"""

import numpy as np
import torch
import math
from typing import Tuple, Optional

from popoe.interfaces import BackendUnavailable

_NVDIFFRAST_AVAILABLE = None


class RendererUnavailable(BackendUnavailable):
    """nvdiffrast is not installed / has no CUDA context on this host."""


def _check_nvdiffrast():
    global _NVDIFFRAST_AVAILABLE
    if _NVDIFFRAST_AVAILABLE is None:
        try:
            import nvdiffrast.torch as dr
            _NVDIFFRAST_AVAILABLE = True
        except ImportError:
            _NVDIFFRAST_AVAILABLE = False
    return _NVDIFFRAST_AVAILABLE


def fibonacci_viewpoints(n: int, radius: float) -> np.ndarray:
    """Return (n, 3) camera positions on a sphere of given radius."""
    golden = (1 + math.sqrt(5)) / 2
    positions = []
    for i in range(n):
        theta = math.acos(1 - 2 * (i + 0.5) / n)
        phi = 2 * math.pi * i / golden
        positions.append([
            radius * math.sin(theta) * math.cos(phi),
            radius * math.sin(theta) * math.sin(phi),
            radius * math.cos(theta),
        ])
    return np.array(positions, dtype=np.float32)


def look_at_matrix(cam_pos: np.ndarray, target: np.ndarray = None) -> np.ndarray:
    """Return 4x4 world-to-camera matrix."""
    if target is None:
        target = np.zeros(3)
    forward = target - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    up_ref = np.array([0., 1., 0.]) if abs(forward[1]) < 0.9 else np.array([1., 0., 0.])
    right = np.cross(forward, up_ref)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    R = np.stack([right, up, -forward], axis=0)  # rows = camera axes in world frame
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = -R @ cam_pos
    return T


def perspective_matrix(fov_deg: float, aspect: float = 1.0,
                        near: float = 0.001, far: float = 100.0) -> np.ndarray:
    """OpenGL-style perspective projection matrix."""
    f = 1.0 / math.tan(math.radians(fov_deg) / 2)
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = (far + near) / (near - far)
    P[2, 3] = 2 * far * near / (near - far)
    P[3, 2] = -1.0
    return P


class NvdiffrastRenderer:
    """GPU renderer using nvdiffrast. Raises RendererUnavailable without it."""

    source = 'nvdiffrast'

    def __init__(self, H: int = 480, W: int = 480, device: str = 'cuda'):
        try:
            import nvdiffrast.torch as dr
            self.ctx = dr.RasterizeCudaContext(device=device)
        except Exception as e:      # no package, no CUDA context, no driver
            raise RendererUnavailable(
                f"nvdiffrast unusable ({type(e).__name__}: {e}). Install with:\n"
                f"  pip install --no-build-isolation "
                f"git+https://github.com/NVlabs/nvdiffrast.git") from e
        self.H = H
        self.W = W
        self.device = device

    def render(self, vertices: np.ndarray, faces: np.ndarray,
               cam_pos: np.ndarray, fov_deg: float = 60.0,
               normals: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render mesh from cam_pos.
        vertices: (V, 3)  faces: (F, 3)
        Returns: rgb (H, W, 3) uint8, depth (H, W) float32
        """
        import nvdiffrast.torch as dr

        V_np = vertices.astype(np.float32)
        F_np = faces.astype(np.int32)

        MV = look_at_matrix(cam_pos)
        P = perspective_matrix(fov_deg, aspect=self.W / self.H)
        MVP = (P @ MV).astype(np.float32)

        # Transform vertices to clip space
        V_h = np.concatenate([V_np, np.ones((len(V_np), 1), dtype=np.float32)], axis=1)  # (V,4)
        V_clip = V_h @ MVP.T  # (V, 4)
        # Camera-space z for the depth map. look_at_matrix is OpenGL-style
        # (camera looks down -z), so visible points have z_cam < 0 and the
        # metric distance along the view axis is -z_cam.
        z_cam = (V_h @ MV.T)[:, 2:3].astype(np.float32)  # (V, 1)

        pos = torch.from_numpy(V_clip).unsqueeze(0).to(self.device)  # (1,V,4)
        tri = torch.from_numpy(F_np).to(self.device)

        rast, _ = dr.rasterize(self.ctx, pos, tri, resolution=[self.H, self.W])  # (1,H,W,4)

        # Compute normals per vertex for shading
        if normals is None:
            normals = self._compute_vertex_normals(V_np, F_np)
        nrm_t = torch.from_numpy(normals.astype(np.float32)).unsqueeze(0).to(self.device)
        nrm_interp, _ = dr.interpolate(nrm_t, rast, tri)  # (1,H,W,3)

        # Simple Lambertian shading
        light_dir = torch.tensor(-cam_pos / (np.linalg.norm(cam_pos) + 1e-8),
                                 dtype=torch.float32, device=self.device)
        nrm_n = torch.nn.functional.normalize(nrm_interp[0], dim=-1)
        shading = torch.clamp((nrm_n * light_dir).sum(dim=-1, keepdim=True), 0.1, 1.0)
        base_color = torch.tensor([0.7, 0.62, 0.55], device=self.device)
        color = (shading * base_color).clamp(0, 1)  # (H,W,3)

        # Mask (where rast[...,3] > 0 means hit)
        hit_mask = rast[0, :, :, 3] > 0  # (H,W)
        color[~hit_mask] = 1.0  # white background

        rgb_np = (color.cpu().numpy() * 255).astype(np.uint8)

        # Metric depth: interpolate camera-space z per pixel (rast[...,3] is the
        # TRIANGLE ID, not depth — the previous 1/(id) "depth" was garbage as a
        # depth map and only usable as a >0 hit test).
        zcam_t = torch.from_numpy(z_cam).unsqueeze(0).to(self.device)  # (1,V,1)
        zcam_interp, _ = dr.interpolate(zcam_t, rast, tri)             # (1,H,W,1)
        depth_t = torch.where(hit_mask, -zcam_interp[0, :, :, 0],
                              torch.zeros_like(zcam_interp[0, :, :, 0]))
        depth_np = depth_t.cpu().numpy().astype(np.float32)

        return rgb_np, depth_np

    @staticmethod
    def _compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        V = len(vertices)
        norms = np.zeros((V, 3), dtype=np.float64)
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        for i in range(3):
            np.add.at(norms, faces[:, i], fn)
        n_len = np.linalg.norm(norms, axis=1, keepdims=True) + 1e-8
        return (norms / n_len).astype(np.float32)


class TrimeshRenderer:
    """CPU ray-casting renderer. Always available; ~100x slower than the GPU
    rasteriser, and its images are NOT pixel-equivalent to nvdiffrast's."""

    source = 'trimesh'

    def __init__(self, H: int = 480, W: int = 480):
        self.H = H
        self.W = W

    def render(self, vertices: np.ndarray, faces: np.ndarray,
               cam_pos: np.ndarray, fov_deg: float = 60.0,
               normals: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        import trimesh
        H, W = self.H, self.W

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.fix_normals()

        forward = -cam_pos / (np.linalg.norm(cam_pos) + 1e-8)
        up_ref = np.array([0., 1., 0.]) if abs(forward[1]) < 0.9 else np.array([1., 0., 0.])
        right = np.cross(forward, up_ref)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        fy = fx = (H / 2) / math.tan(math.radians(fov_deg) / 2)
        cx, cy = W / 2.0, H / 2.0

        py_g, px_g = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        dx = (px_g.flatten() - cx) / fx
        dy = (py_g.flatten() - cy) / fy
        dirs = dx[:, None] * right + dy[:, None] * up + forward[None]
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

        origins = np.tile(cam_pos, (H * W, 1))
        locs, idx_ray, idx_tri = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)

        rgb = np.full((H * W, 3), 200, dtype=np.uint8)
        depth_map = np.zeros(H * W, dtype=np.float32)

        if len(locs) > 0:
            fn = mesh.face_normals[idx_tri]
            shading = np.clip((-fn * forward).sum(axis=1), 0.1, 1.0)
            base = np.array([178, 158, 140], dtype=np.float32)
            rgb[idx_ray] = (base * shading[:, None]).astype(np.uint8)
            # Camera-axis z, not Euclidean ray length — same convention as
            # NvdiffrastRenderer, so the two depths backproject identically.
            depth_map[idx_ray] = (locs - cam_pos) @ forward

        return rgb.reshape(H, W, 3), depth_map.reshape(H, W)


def get_renderer(H: int = 480, W: int = 480, device: str = 'cuda',
                 backend: str = 'auto') -> object:
    """Build a renderer. `backend` is 'nvdiffrast' | 'trimesh' | 'auto'.

    The two renderers do NOT produce the same image — different rasterisation
    and shading — so they do not produce the same DINOv2 features from a CAD
    render. Whichever ran is therefore part of your experiment's configuration:
    read it back from `renderer.source` and put it in the cache key
    (examples/bop_eval.py does; see cache.py on why an unkeyed upstream knob
    silently poisons entries).

    'auto' still prefers the GPU and drops to the CPU ray-caster, but that is
    now the CALLER asking for a preference, not an implementation hiding a
    substitution. Name a backend explicitly when a run must be reproducible —
    it then raises RendererUnavailable instead of quietly going ~100x slower on
    a different method.
    """
    if backend not in ('auto', 'nvdiffrast', 'trimesh'):
        raise ValueError(f"unknown renderer backend: {backend!r}")

    if backend == 'trimesh':
        return TrimeshRenderer(H, W)
    if backend == 'nvdiffrast':
        return NvdiffrastRenderer(H, W, device)     # raises if unusable

    try:
        r = NvdiffrastRenderer(H, W, device)
        print("Using nvdiffrast GPU renderer.")
        return r
    except RendererUnavailable as e:
        print(f"[renderer] nvdiffrast unavailable -> {e}")
        print("[renderer] backend='auto' -> falling back to the trimesh CPU "
              "ray-caster. Renders (and any DINO features derived from them) "
              "will DIFFER from a GPU run. Pass backend='nvdiffrast' to make "
              "this an error instead.")
        return TrimeshRenderer(H, W)


def load_mesh_for_rendering(mesh_path: str, target_diameter: float = 0.1):
    """Load mesh, normalise to target diameter, return vertices/faces/normals."""
    import trimesh
    mesh = trimesh.load(mesh_path, force='mesh')
    mesh.fix_normals()
    diameter = np.linalg.norm(mesh.extents)
    scale = target_diameter / (diameter + 1e-8)
    mesh.apply_scale(scale)
    center = mesh.centroid
    mesh.apply_translation(-center)
    V = np.array(mesh.vertices, dtype=np.float32)
    F = np.array(mesh.faces, dtype=np.int32)
    N = NvdiffrastRenderer._compute_vertex_normals(V, F)
    return V, F, N, scale, center

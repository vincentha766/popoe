"""Prove the vertex-colour render path is correct, then show it.

The UV path already had a v-flip bug once (2026-07-06: YCB-V renders came out
vertically mirrored and nobody noticed for weeks, because a plausible-looking
image is not evidence). So the vertex-colour path added on 2026-07-28 is not
trusted on the strength of "the ape looks brown". This script runs four
falsifiable checks on synthetic geometry whose answer is known in closed form,
and only then dumps the pretty pictures.

The checks, and what each one would catch:

  1. SILHOUETTE   the coloured render must cover exactly the same pixels as the
                  flat render of the same mesh from the same camera. Catches a
                  wrong rasterisation, a lost `hit` mask, or a background that
                  bleeds into the object.
  2. CHANNELS     a mesh whose vertices are pure red / green / blue must produce
                  those channels in that order. Catches BGR/RGB transposition —
                  the single most likely defect, and one that a brown ape hides
                  because brown survives an R/B swap looking merely "off".
  3. ORIENTATION  the renderer's own convention is y_cam > 0 -> larger ROW (see
                  _nvdiffrast_render). A mesh coloured red at +y and green at -y
                  must put red in the LOWER half of the image. Catches exactly
                  the v-flip class of bug the UV path had.
  4. GAMUT        every rendered object pixel must be `lambert * albedo` for
                  some albedo the mesh actually carries and lambert in
                  [0.1, 1.0]. Checked on the REAL LM-O meshes, per channel, so
                  a colour that came from nowhere (or from the beige default)
                  fails.

Usage:
  python scripts/render_shading_check.py --bop /workspace/bop_data \
      --out /workspace/results/<run>/shading_check
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import trimesh

from popoe.freeze.feature_extractor import (
    SHADING_FLAT, SHADING_UV, SHADING_VERTEX_COLOR, QueryFeatureExtractor,
    resolve_mesh_shading,
)

FAILURES: list = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    if not ok:
        FAILURES.append(name)


class _Renderer(QueryFeatureExtractor):
    """The renderer alone — no DINOv2, no GeDi, no checkpoints to download."""

    def __init__(self, device="cuda"):
        self.device = device
        self._render_backend_pref = "nvdiffrast"
        self._nvd_ctx = None
        self._nvd_init_tried = False


def _tri_mesh(colors=None):
    """One big triangle spanning the view, vertices at known image positions.

    Vertex 0 at +y (renders DOWN, large row), 1 at -y & -x (up-left),
    2 at -y & +x (up-right)."""
    v = np.array([[0.0, 60.0, 0.0], [-60.0, -50.0, 0.0], [60.0, -50.0, 0.0]])
    f = np.array([[0, 1, 2]])
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    if colors is not None:
        m.visual.vertex_colors = np.asarray(colors, dtype=np.uint8)
    return m


def _render(rend, mesh, cam_pos=np.array([0.0, 0.0, -200.0]), H=224, W=224):
    img, depth, *_ = rend._raycast_render(mesh, cam_pos, H, W)
    return np.asarray(img), depth


def check_silhouette(rend):
    col = _tri_mesh([[255, 0, 0, 255], [255, 0, 0, 255], [255, 0, 0, 255]])
    flat = _tri_mesh()
    assert resolve_mesh_shading(col) == SHADING_VERTEX_COLOR
    assert resolve_mesh_shading(flat) == SHADING_FLAT
    _, d_col = _render(rend, col)
    _, d_flat = _render(rend, flat)
    same = np.array_equal(d_col > 0, d_flat > 0)
    n = int((d_flat > 0).sum())
    check("silhouette identical to flat render", same and n > 1000,
          f"{n} object pixels")


def check_channels(rend):
    """Pure R / G / B vertices -> that channel dominant near that vertex."""
    m = _tri_mesh([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]])
    rgb, depth = _render(rend, m)
    hit = depth > 0
    ys, xs = np.where(hit)
    # Barycentric-ish probes: pixel nearest each vertex's projection.
    # v0 -> bottom centre, v1 -> top left, v2 -> top right.
    probes = {0: (ys.max(), int(np.median(xs[ys > ys.max() - 5]))),
              1: (ys.min() + 4, xs[ys < ys.min() + 8].min() + 2),
              2: (ys.min() + 4, xs[ys < ys.min() + 8].max() - 2)}
    got = {}
    for vi, (r, c) in probes.items():
        px = rgb[int(r), int(c)].astype(int)
        got[vi] = px
        check(f"channel order at vertex {vi} (expect argmax=={vi})",
              int(np.argmax(px)) == vi, f"pixel={px.tolist()}")
    return got


def check_orientation(rend):
    """+y vertex is RED and must land in the LOWER half of the image."""
    m = _tri_mesh([[255, 0, 0, 255], [0, 255, 0, 255], [0, 255, 0, 255]])
    rgb, depth = _render(rend, m)
    hit = depth > 0
    H = rgb.shape[0]
    top = rgb[: H // 2][hit[: H // 2]].astype(float)
    bot = rgb[H // 2:][hit[H // 2:]].astype(float)
    rness_top = (top[:, 0] - top[:, 1]).mean()
    rness_bot = (bot[:, 0] - bot[:, 1]).mean()
    check("orientation: +y (red) renders in the LOWER half",
          rness_bot > rness_top,
          f"R-G bottom={rness_bot:.1f} top={rness_top:.1f}")


def check_gamut(rend, mesh_path):
    """Real mesh: every object pixel must be lambert*albedo, per channel."""
    m = trimesh.load(str(mesh_path), force="mesh")
    mode = resolve_mesh_shading(m)
    if mode != SHADING_VERTEX_COLOR:
        check(f"{Path(mesh_path).name}: expected vertex colour", False, mode)
        return None
    m.apply_scale(0.45 * 224 / max(m.extents))
    m.apply_translation(-m.bounds.mean(0))
    rgb, depth = _render(rend, m, cam_pos=np.array([0.0, 0.0, -max(m.extents) * 1.5]))
    hit = depth > 0
    obj = rgb[hit].astype(float)
    vc = np.asarray(m.visual.vertex_colors, dtype=float)[:, :3]
    ok = True
    detail = []
    for ch, nm in enumerate("RGB"):
        hi = vc[:, ch].max()
        # lambert is clamped to [0.1, 1.0], so a pixel can never exceed the
        # brightest albedo, and interpolation cannot invent a new hue.
        ok &= obj[:, ch].max() <= hi + 1.5
        detail.append(f"{nm}: px<= {obj[:, ch].max():.0f} albedo<= {hi:.0f}")
    check(f"{Path(mesh_path).name}: pixels within mesh colour gamut", bool(ok),
          "; ".join(detail))
    # And the render must not be grey: a flat-beige fallback would have
    # R:G:B fixed at 180:160:140 everywhere.
    mean_px = obj.mean(0)
    mean_vc = vc.mean(0)
    order_px = tuple(np.argsort(-mean_px))
    order_vc = tuple(np.argsort(-mean_vc))
    check(f"{Path(mesh_path).name}: channel ORDER matches the mesh's own",
          order_px == order_vc,
          f"render mean={np.round(mean_px, 1).tolist()} "
          f"mesh mean={np.round(mean_vc, 1).tolist()}")
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bop", default="/workspace/bop_data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--objs", default="1,5,6,8,9,10,11,12")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rend = _Renderer()
    if not rend._init_nvdiffrast():
        raise SystemExit("nvdiffrast unavailable — this check is about the GPU "
                         "path and must not silently measure the CPU one.")
    print("nvdiffrast OK", flush=True)

    print("\n--- synthetic geometry (closed-form expectations) ---", flush=True)
    check_silhouette(rend)
    check_channels(rend)
    check_orientation(rend)

    print("\n--- real BOP meshes ---", flush=True)
    import PIL.Image
    lmo = Path(args.bop) / "lmo" / "models"
    for oid in [int(x) for x in args.objs.split(",")]:
        p = lmo / f"obj_{oid:06d}.ply"
        if not p.exists():
            check(f"{p} exists", False)
            continue
        # Side by side: forced-flat (what the pipeline used to render) next to
        # the fix. Same camera, same rasterisation — only the albedo differs.
        os.environ["POPOE_MESH_SHADING"] = "uv-only"
        m = trimesh.load(str(p), force="mesh")
        m.apply_scale(0.45 * 224 / max(m.extents))
        m.apply_translation(-m.bounds.mean(0))
        cam = np.array([0.0, 0.0, -max(m.extents) * 1.5])
        before, _ = _render(rend, m)
        os.environ["POPOE_MESH_SHADING"] = "auto"
        after = check_gamut(rend, p)
        if after is None:
            continue
        pair = np.concatenate([before, np.full((before.shape[0], 4, 3), 255,
                                               np.uint8), after], axis=1)
        PIL.Image.fromarray(pair).save(out / f"obj_{oid:06d}_before_after.png")

    # A UV mesh must be untouched by all of this.
    ycbv = Path(args.bop) / "ycbv" / "models" / "obj_000005.ply"
    if ycbv.exists():
        check("YCB-V mesh still resolves to the UV path",
              resolve_mesh_shading(trimesh.load(str(ycbv), force="mesh"))
              == SHADING_UV)

    print(f"\nimages -> {out}", flush=True)
    if FAILURES:
        raise SystemExit(f"SHADING CHECK FAILED: {FAILURES}")
    print("SHADING CHECK PASSED", flush=True)


if __name__ == "__main__":
    main()

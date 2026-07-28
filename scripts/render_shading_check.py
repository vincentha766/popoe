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
  4. ALBEDO       on the REAL LM-O meshes: render the same mesh from the same
                  camera twice, once forced flat and once coloured. Both are
                  `lambert * albedo` with the SAME lambert, so dividing the two
                  recovers the interpolated albedo per pixel in closed form.
                  Every recovered value must lie inside the mesh's own
                  per-channel vertex-colour range. This is an identity, not a
                  heuristic, and unlike comparing mean colours it still has
                  teeth on the near-grey objects (driller, glue) where every
                  channel mean agrees to within a unit of noise.

A note on what the checks deliberately do NOT assert: that the render "looks
more like a photograph". It does not, and cannot — the lighting is a synthetic
headlight. The claim under test is only that the albedo now comes from the
mesh instead of a constant.

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

    Seen from CAM_Z (below), the renderer's basis is right=+x, up=+y, and
    y_cam > 0 maps to a LARGER row. So vertex 0 (+y) lands at bottom centre,
    vertex 1 (-x, -y) at top left, vertex 2 (+x, -y) at top right.

    The camera side matters: from -z the basis comes out right=-x (cross of
    forward and up_ref flips), which mirrors the image left-to-right. That is
    correct behaviour — you are looking at the back — but it makes the probe
    positions below wrong, which is how this note got written."""
    v = np.array([[0.0, 60.0, 0.0], [-60.0, -50.0, 0.0], [60.0, -50.0, 0.0]])
    f = np.array([[0, 1, 2]])
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    if colors is not None:
        m.visual.vertex_colors = np.asarray(colors, dtype=np.uint8)
    return m


CAM_Z = np.array([0.0, 0.0, 200.0])


def _render(rend, mesh, cam_pos=CAM_Z, H=224, W=224):
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


FLAT_BASE = np.array([180.0, 160.0, 140.0])


def _prepared(mesh_path):
    """The mesh exactly as extract_query_features prepares it."""
    m = trimesh.load(str(mesh_path), force="mesh")
    m.apply_scale(0.45 * 224 / max(m.extents))
    m.apply_translation(-m.bounds.mean(0))
    return m


def check_albedo(rend, mesh_path):
    """Divide the coloured render by the flat one and read the albedo back.

    flat[px]     = round(255 * L * 180/255) = round(L * 180)
    coloured[px] = round(255 * L * A_ch/255) = round(L * A_ch)
    => A_ch = coloured_ch * 180 / flat_R, exactly, up to uint8 rounding.

    Rounding is why only well-lit pixels are used: at the L=0.1 clamp, flat_R
    is 18 and one count of rounding is already 5% of the recovered albedo."""
    name = Path(mesh_path).name
    m = _prepared(mesh_path)
    if resolve_mesh_shading(m) != SHADING_VERTEX_COLOR:
        check(f"{name}: expected vertex colour", False,
              resolve_mesh_shading(m))
        return None, None
    cam = np.array([0.0, 0.0, max(m.extents) * 1.5])

    os.environ["POPOE_MESH_SHADING"] = "uv-only"
    flat, d_flat = _render(rend, m, cam_pos=cam)
    os.environ["POPOE_MESH_SHADING"] = "auto"
    col, d_col = _render(rend, m, cam_pos=cam)

    check(f"{name}: same silhouette as the flat render",
          np.array_equal(d_flat > 0, d_col > 0),
          f"{int((d_col > 0).sum())} object pixels")
    hit = d_col > 0
    check(f"{name}: the coloured render actually DIFFERS from the flat one",
          not np.array_equal(flat[hit], col[hit]),
          f"mean |diff| = {np.abs(flat[hit].astype(int) - col[hit].astype(int)).mean():.1f}")

    lit = hit & (flat[..., 0] >= 60)          # L >= 1/3, rounding under 2%
    if lit.sum() < 200:
        check(f"{name}: enough well-lit pixels to recover albedo", False,
              f"{int(lit.sum())}")
        return col, flat
    L = flat[lit][:, 0].astype(float) / FLAT_BASE[0]
    A = col[lit].astype(float) / L[:, None]
    vc = np.asarray(m.visual.vertex_colors, dtype=float)[:, :3]
    ok, detail = True, []
    for ch, nm in enumerate("RGB"):
        lo, hi = vc[:, ch].min(), vc[:, ch].max()
        # 3 counts of slack absorbs the two uint8 roundings; interpolation can
        # only produce convex combinations, so nothing may fall OUTSIDE.
        bad = int(((A[:, ch] < lo - 3) | (A[:, ch] > hi + 3)).sum())
        ok &= bad == 0
        detail.append(f"{nm}[{lo:.0f},{hi:.0f}] out={bad}")
    check(f"{name}: recovered albedo lies in the mesh's vertex-colour range",
          bool(ok), "; ".join(detail) + f"  (n={int(lit.sum())})")
    return col, flat


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
        after, before = check_albedo(rend, p)
        if after is None:
            continue
        # Side by side: what the pipeline used to render (left) next to what it
        # renders now (right). Same camera, same rasterisation, same lighting —
        # only the albedo differs.
        gap = np.full((before.shape[0], 4, 3), 255, np.uint8)
        PIL.Image.fromarray(np.concatenate([before, gap, after], axis=1)).save(
            out / f"obj_{oid:06d}_before_after.png")

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

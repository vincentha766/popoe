"""Render-appearance re-rank of pose hypotheses (knife-4 / FreeZe SAR core).

Offline measurement (gedi knife 4, 2026-07-29): re-selecting among PCA-axis
rotational variants by DINOv2 render-vs-scene patch cosine (`sar_ti`) lifts
YCB-V combo_sym full BOP AR (flat) 0.8275 → **0.8605** (> FreeZe(CNOS) 0.853).
This module is the **in-pipeline** form of that re-rank.

Public API
----------
* ``pca_flip_variants(R, pts)`` — pure numpy; same variant set as
  ``scripts/flip_rescore_ab.variants``.
* ``pick_by_scores(variants, scores)`` — pure argmax helper.
* ``RenderAppearanceReranker`` — ``PoseRefiner`` that expands the champion,
  scores each variant with DINOv2 render-vs-scene, returns the best.

GPU + nvdiffrast + DINOv2 are required for ``refine``; construction and the
pure helpers do not. Missing backends raise ``BackendUnavailable`` (no silent
fallback — see ``interfaces.BackendUnavailable``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from popoe.interfaces import (
    BackendUnavailable,
    ObjectModel,
    PointFeatures,
    PoseHypothesis,
    Scene,
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without GPU)
# ---------------------------------------------------------------------------

def rot_about(axis: np.ndarray, deg: float) -> np.ndarray:
    a = np.asarray(axis, float).reshape(3)
    a = a / (np.linalg.norm(a) + 1e-12)
    th = np.radians(float(deg))
    K = np.array([[0, -a[2], a[1]],
                  [a[2], 0, -a[0]],
                  [-a[1], a[0], 0]], dtype=float)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def pca_flip_variants(
    R: np.ndarray,
    pts: np.ndarray,
    *,
    include_azimuth: bool = True,
) -> list[tuple[str, np.ndarray]]:
    """Champion + rotational alternates expressed in the **model** frame.

    Axes = principal directions of ``pts`` (query cloud, metres). Byte-same
    variant *set* as ``scripts/flip_rescore_ab.variants`` (the set knife 4
    measured): champion, flip0/1/2 (180° about each PCA axis), optional
    az90/az270 about the first axis.
    """
    R = np.asarray(R, float).reshape(3, 3)
    pts = np.asarray(pts, float).reshape(-1, 3)
    if pts.shape[0] < 3:
        return [("champion", R.copy())]
    c = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    out: list[tuple[str, np.ndarray]] = [("champion", R.copy())]
    for i in range(min(3, vt.shape[0])):
        out.append((f"flip{i}", R @ rot_about(vt[i], 180.0)))
    if include_azimuth and vt.shape[0] >= 1:
        for ang in (90.0, 270.0):
            out.append((f"az{int(ang)}", R @ rot_about(vt[0], ang)))
    return out


def pick_by_scores(
    variants: list[tuple[str, np.ndarray]],
    scores: dict[str, float],
) -> tuple[str, np.ndarray, float]:
    """Argmax over ``scores`` among names present in ``variants``."""
    if not variants:
        raise ValueError("empty variants")
    best_name, best_R = variants[0]
    best_s = float(scores.get(best_name, float("-inf")))
    for name, R in variants:
        s = float(scores.get(name, float("-inf")))
        if s > best_s:
            best_name, best_R, best_s = name, R, s
    return best_name, best_R, best_s


def bbox_from_mask(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


# ---------------------------------------------------------------------------
# PoseRefiner
# ---------------------------------------------------------------------------

# Process-wide shared backends so ``stages_for_object`` per object does not
# re-load DINOv2 / nvdiffrast once per obj_id (Codex P2 / multi-object BOP).
_SHARED: dict = {}


class RenderAppearanceReranker:
    """Re-rank PCA flip variants by DINOv2 render-vs-scene patch cosine.

    Implements the ``PoseRefiner`` protocol. Place **after** ICP in
    ``Pipeline.refiners`` so each geometric champion is re-oriented by
    appearance before the scorer / selector see it.

    ``enabled=False`` makes ``refine`` a pure identity (recipes can leave the
    stage wired and flip a flag).
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        include_azimuth: bool = True,
        device: str = "cuda",
        crop_size: int = 224,
        re_icp: bool = True,
    ):
        self.enabled = enabled
        self.include_azimuth = include_azimuth
        self.device = device
        self.crop_size = crop_size
        self.re_icp = re_icp
        self._ready = False
        self._dino = None
        self._dino_layer = None
        self._pose_renderer = None
        self._asset_cache: dict = {}

    def _sar_mod(self):
        """Load scripts/sar_render_compare.py by path (not a package)."""
        import importlib.util
        from pathlib import Path
        key = "sar_mod"
        if key in _SHARED:
            return _SHARED[key]
        path = Path(__file__).resolve().parents[2] / "scripts" / "sar_render_compare.py"
        if not path.is_file():
            raise BackendUnavailable(
                f"sar_render_compare.py not found at {path}")
        spec = importlib.util.spec_from_file_location("popoe_sar_render_compare", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        _SHARED[key] = mod
        return mod

    def _ensure(self) -> None:
        if self._ready:
            return
        try:
            import torch
            if str(self.device).startswith("cuda") and not torch.cuda.is_available():
                raise BackendUnavailable(
                    "RenderAppearanceReranker needs CUDA; none available")
            dkey = f"dino:{self.device}"
            if dkey not in _SHARED:
                from popoe.freeze.feature_extractor import load_dinov2, _dino_layer
                # load_dinov2(device=...) only — FreeZe ViT-g, same as SAR script.
                dino = load_dinov2(device=self.device)
                _SHARED[dkey] = (dino, _dino_layer(dino))
            self._dino, self._dino_layer = _SHARED[dkey]
            rkey = f"renderer:{self.device}"
            if rkey not in _SHARED:
                mod = self._sar_mod()
                _SHARED[rkey] = mod.PoseRenderer(device=self.device)
            self._pose_renderer = _SHARED[rkey]
            self._ready = True
        except BackendUnavailable:
            raise
        except Exception as e:
            raise BackendUnavailable(
                f"RenderAppearanceReranker backend failed: {type(e).__name__}: {e}"
            ) from e

    def _load_asset(self, mesh_path: str):
        cache = _SHARED.setdefault("assets", {})
        if mesh_path in cache:
            return cache[mesh_path]
        mod = self._sar_mod()
        asset = mod.load_asset(mesh_path, device=self.device)
        cache[mesh_path] = asset
        return asset

    def _embed_crop(self, crop_rgb01: np.ndarray):
        import torch
        import cv2

        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        im = cv2.resize(crop_rgb01, (self.crop_size, self.crop_size),
                        interpolation=cv2.INTER_LINEAR)
        x = (im.astype(np.float32) - mean) / std
        t = torch.from_numpy(x.transpose(2, 0, 1)[None]).float().to(self.device)
        with torch.no_grad():
            # Same intermediate layer as FreeZe / scripts/sar_render_compare.
            toks = self._dino.get_intermediate_layers(
                t, n=[self._dino_layer], return_class_token=False)[0]
        toks = toks[0]
        toks = toks / (toks.norm(dim=-1, keepdim=True) + 1e-8)
        return toks

    def _sar_ti(
        self,
        scene: Scene,
        obj: ObjectModel,
        R: np.ndarray,
        t_m: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> float:
        """Mean patch cosine of rendered object vs scene crop."""
        import torch

        self._ensure()
        H, W = int(scene.rgb.shape[0]), int(scene.rgb.shape[1])
        x0, y0, x1, y1 = bbox
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            return float("-inf")

        t_mm = np.asarray(t_m, float).reshape(3) * 1000.0
        asset = self._load_asset(obj.mesh_path)
        # Calibrate y_sign once per process — but latch ONLY on a reliable
        # calibration. The sign is chosen by rendering the FIRST candidate
        # pose both ways and taking the higher mask-IoU vs a numpy pinhole
        # projection; on a degenerate pose (empty or garbage render) both
        # IoUs are ~0 and the argmax is noise — latching it silently mirrors
        # every later render for the whole run (triage B1-F3). Gate at 0.3
        # (a sane pose scores well above; the wrong sign scores ~0), retry on
        # the next candidate, and always log the outcome.
        if not getattr(self._pose_renderer, "_calibrated", False):
            sign, iou = self._pose_renderer.calibrate(
                asset, R, t_mm, scene.K, H, W)
            if iou >= 0.3:
                self._pose_renderer._calibrated = True  # type: ignore[attr-defined]
                print(f"[rerank] y_sign latched {sign:+.0f} "
                      f"(calibration IoU {iou:.3f}, obj {obj.obj_id})",
                      flush=True)
            else:
                self._pose_renderer.y_sign = 1.0
                print(f"[rerank] y_sign calibration UNRELIABLE "
                      f"(IoU {iou:.3f} < 0.3, obj {obj.obj_id}) — not "
                      f"latched; default +1 for this render, will retry",
                      flush=True)

        rgb_r, _mask, _depth = self._pose_renderer.render(
            asset, R, t_mm, scene.K, H, W)
        crop_s = scene.rgb[y0:y1, x0:x1]
        crop_r = rgb_r[y0:y1, x0:x1]
        if crop_s.dtype != np.float32:
            crop_s = crop_s.astype(np.float32) / 255.0
        if crop_r.max() > 1.5:
            crop_r = crop_r.astype(np.float32) / 255.0

        a = self._embed_crop(crop_s)
        b = self._embed_crop(crop_r)
        return float((a * b).sum(dim=-1).mean().item())

    @staticmethod
    def _tau_icp(pose: PoseHypothesis, query: PointFeatures) -> float:
        """ICP inlier threshold, absolute, in the clouds' units (metres).

        ICPRefiner records the tau it used; reuse it, never guess. The previous
        ``breakdown.get("tau_icp", 0.03)`` never found that key (nothing wrote
        it) and so always meant 0.03 **metres** = 30 mm — clouds are metres
        (adapters.py converts BOP mm), where the real 3%-of-extent tau is
        3-8 mm. Its `if tau > 0.5: expand` guard could not help: it separates
        relative from absolute by magnitude, and 0.03 is exactly the value both
        readings claim.
        """
        tau = pose.breakdown.get("tau_icp")
        if tau is not None:
            return float(tau)
        # No ICP upstream in this chain — rebuild the same rule ICPRefiner uses.
        from popoe.freeze.recipes import TAU_FRAC
        return TAU_FRAC * float(np.ptp(query.pts, axis=0).max())

    def refine(
        self,
        pose: PoseHypothesis,
        scene: Scene,
        obj: ObjectModel,
        query: PointFeatures,
        target: PointFeatures,
    ) -> PoseHypothesis:
        if not self.enabled:
            return pose

        bd = dict(pose.breakdown)
        bd["R_prererank"] = pose.R.copy()
        bd["t_prererank"] = pose.t.copy()

        det = target.meta.get("detection")
        bbox = None
        if det is not None and getattr(det, "bbox", None) is not None:
            bbox = tuple(int(v) for v in det.bbox)  # type: ignore[arg-type]
        elif "bbox" in target.meta:
            bbox = tuple(int(v) for v in target.meta["bbox"])
        elif det is not None and getattr(det, "mask", None) is not None:
            bbox = bbox_from_mask(det.mask)
        if bbox is None:
            bd["render_rerank"] = "skipped_no_bbox"
            return PoseHypothesis(pose.R, pose.t, pose.score, bd)

        variants = pca_flip_variants(
            pose.R, query.pts, include_azimuth=self.include_azimuth)
        scores: dict[str, float] = {}
        for name, Rv in variants:
            scores[name] = self._sar_ti(scene, obj, Rv, pose.t, bbox)

        name, R_best, s_best = pick_by_scores(variants, scores)
        t_best = np.asarray(pose.t, float).copy()
        bd["pre_rerank_score"] = float(pose.score)
        bd["render_rerank"] = name
        bd["sar_ti"] = float(s_best)
        bd["sar_ti_by_variant"] = {k: float(v) for k, v in scores.items()}
        # Model-frame delta of the chosen variant (Rv = R @ D), captured
        # BEFORE re-ICP moves R_best off the pure flip. Applied to the coarse
        # pose below so ChampionScorer's S_coarse measures the orientation
        # family actually being scored — leaving the pre-flip R_coarse there
        # made the arbitration factor penalise exactly the flips this stage
        # exists to rescue (s_coarse × rerank ordering, triage B1-F2).
        flip_delta = (None if name == "champion"
                      else np.asarray(pose.R, float).T @ np.asarray(R_best, float))

        # Re-ICP so translation / s_icp match the chosen R (ChampionScorer then
        # re-scores on consistent geometry). The champion goes through it TOO,
        # even though its R is unchanged: a variant that just had an ICP pass is
        # not comparable to one that did not, and s_icp reaches the scorer as a
        # multiplicative factor. Re-ICP'ing only the flips inflated their s_icp
        # ~4x, so the selector preferred a flip on measurement asymmetry alone
        # (LM-O AR(2/3) 0.77 -> 0.25). scripts/freezev2_flip_post.py in the gedi
        # archive hit the same trap offline and guards it the same way.
        if self.re_icp:
            dense = target.pts_dense if target.pts_dense is not None else target.pts
            if dense is not None and len(dense) >= 4 and len(query.pts) >= 4:
                try:
                    from popoe.registration import icp_refinement
                    tau = self._tau_icp(pose, query)
                    R_f, t_f, fit = icp_refinement(
                        query.pts, dense, R_best, t_best, tau_icp=tau)
                    R_best, t_best = R_f, t_f
                    bd["s_icp"] = float(fit)
                    bd["render_rerank_re_icp"] = True
                except Exception as e:
                    bd["render_rerank_re_icp"] = f"failed:{type(e).__name__}"
                    if name != "champion":
                        # A flipped R carrying the champion's pre-flip s_icp is
                        # not a scoreable candidate (the 07-31 asymmetry trap
                        # in reverse). Without a consistent re-measurement,
                        # keep the champion pose.
                        bd["render_rerank"] = f"reverted:{name}"
                        name = "champion"
                        flip_delta = None
                        R_best = np.asarray(pose.R, float).copy()
                        t_best = np.asarray(pose.t, float).copy()

        if flip_delta is not None and "R_coarse" in bd:
            bd["R_coarse"] = np.asarray(bd["R_coarse"], float) @ flip_delta

        return PoseHypothesis(R_best.copy(), t_best, float(s_best), bd)

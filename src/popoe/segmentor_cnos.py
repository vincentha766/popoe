"""
CNOS-style segmentor: template matching with DINOv2 + SAM mask proposal.
Reproduces the core idea from:
  CNOS: A Strong Baseline for CAD-novel Object Segmentation (ICCV-W 2023)

Pipeline:
  1. Render N templates of query object from icosphere viewpoints
  2. Extract DINOv2 features from each template
  3. Slide over scene with DINOv2 feature crops
  4. Find top-K matching regions by cosine similarity
  5. Use SAM (if available) or depth-based method to produce final masks
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from typing import List, Tuple, Optional


torch.backends.cudnn.enabled = False


def _get_dino(device='cuda'):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg',
                            pretrained=True, source='local'
                            if _local_dino_exists() else 'github')
    return model.to(device).eval()


def _local_dino_exists():
    import os
    return os.path.exists('/root/.cache/torch/hub/facebookresearch_dinov2_main')


class CNOSSegmentor:
    """
    CNOS-style zero-shot instance segmentor.
    Uses DINOv2 ViT-B (faster than ViT-G) for template matching,
    then SAM2 (or depth fallback) for mask refinement.
    """

    def __init__(self, renderer, device='cuda', n_templates=42,
                 top_k_regions=5, sam_model_size='small'):
        self.renderer = renderer
        self.device = device
        self.n_templates = n_templates
        self.top_k_regions = top_k_regions
        self.sam_model_size = sam_model_size
        self._dino = None
        self._sam = None
        self._template_feats = None  # (N_templates, D)
        self._template_masks = None  # (N_templates, H, W)

    def _load_dino(self):
        if self._dino is None:
            self._dino = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=True
            ).to(self.device).eval()

    def _load_sam(self):
        if self._sam is not None:
            return True
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            import os
            ckpt_dir = os.environ.get('POPOE_SAM2_CKPT', '/workspace/sam2_checkpoints')
            # SAM2.1 configs (official sam-2 >= 1.0, 092824 checkpoints)
            cfg_map = {
                'tiny':  ('configs/sam2.1/sam2.1_hiera_t.yaml',  'sam2.1_hiera_tiny.pt'),
                'small': ('configs/sam2.1/sam2.1_hiera_s.yaml',  'sam2.1_hiera_small.pt'),
                'base':  ('configs/sam2.1/sam2.1_hiera_b+.yaml', 'sam2.1_hiera_base_plus.pt'),
                'large': ('configs/sam2.1/sam2.1_hiera_l.yaml',  'sam2.1_hiera_large.pt'),
            }
            cfg, ckpt = cfg_map[self.sam_model_size]
            ckpt_path = os.path.join(ckpt_dir, ckpt)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"SAM2 checkpoint not found: {ckpt_path}\n"
                    f"  wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/{ckpt}"
                    f" -P {ckpt_dir}/"
                )
            self._sam = SAM2ImagePredictor(
                build_sam2(cfg, ckpt_path, device=self.device, apply_postprocessing=False)
            )
            print(f"SAM2.1-{self.sam_model_size} loaded.")
            return True
        except Exception as e:
            print(f"SAM2 load failed: {e}")
            return False

    @torch.no_grad()
    def precompute_templates(self, mesh_path: str, fov_deg: float = 60.0):
        """
        Render N templates and extract DINOv2 features.
        Must be called once per query object before segmenting.

        CAVEAT — templates are rendered with the flat Lambertian, *untextured*
        renderer (renderer.NvdiffrastRenderer: single light, constant base_color,
        no albedo/UV). This is FAITHFUL for LMO (LineMod CAD has no UV texture, so
        the paper renders it Lambertian too) — LMO self-segmentation via this path
        legitimately matches the paper's CNOS line. It is INSUFFICIENT for textured
        objects (YCB-V, HOPE, ...): DINOv2 then matches on silhouette only, losing
        logo/label/color cues, so appearance matching underperforms. For textured
        datasets use A-path official CNOS-FastSAM detections (read the BOP detection
        JSON) or a textured-template source instead of self-rendering here.
        """
        from popoe.renderer import load_mesh_for_rendering, fibonacci_viewpoints
        from torchvision import transforms

        self._load_dino()

        V, F, N, scale, center = load_mesh_for_rendering(mesh_path)
        radius = np.linalg.norm(np.ptp(V, axis=0)) * 1.5
        cam_positions = fibonacci_viewpoints(self.n_templates, radius)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        H = W = self.renderer.H
        template_feats = []
        template_masks = []

        print(f"Rendering {self.n_templates} templates...")
        for i, cam_pos in enumerate(cam_positions):
            rgb, depth = self.renderer.render(V, F, cam_pos, fov_deg=fov_deg, normals=N)
            hit = depth > 0
            template_masks.append(hit)

            # Skip blank renders
            if hit.sum() < 100:
                template_feats.append(None)
                continue

            import PIL.Image
            H_r = (H // 14) * 14
            W_r = (W // 14) * 14
            img_pil = PIL.Image.fromarray(rgb).resize((W_r, H_r))
            img_t = transform(img_pil).unsqueeze(0).to(self.device)  # (1,3,H_r,W_r)

            # ViT-B/14 patch size = 14
            n_ph, n_pw = H_r // 14, W_r // 14
            out = self._dino.get_intermediate_layers(
                img_t, n=[self._dino.n_blocks - 1], return_class_token=True
            )
            # cls token for global descriptor
            cls_token = out[0][1].squeeze(0).cpu().numpy()  # (D,)
            template_feats.append(cls_token)

        # Stack valid templates
        valid = [f for f in template_feats if f is not None]
        self._template_feats = np.stack(valid, axis=0)  # (N_valid, D)
        self._template_masks = template_masks
        print(f"Templates ready: {len(valid)}/{self.n_templates} valid")

    @torch.no_grad()
    def _load_sam_generator(self):
        """Lazy-load SAM2AutomaticMaskGenerator for CNOS-style proposal pipeline."""
        if getattr(self, '_sam_amg', None) is not None:
            return self._sam_amg
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        import os
        ckpt_dir = os.environ.get('POPOE_SAM2_CKPT', '/workspace/sam2_checkpoints')
        cfg_map = {
            'tiny':  ('configs/sam2.1/sam2.1_hiera_t.yaml',  'sam2.1_hiera_tiny.pt'),
            'small': ('configs/sam2.1/sam2.1_hiera_s.yaml',  'sam2.1_hiera_small.pt'),
            'base':  ('configs/sam2.1/sam2.1_hiera_b+.yaml', 'sam2.1_hiera_base_plus.pt'),
            'large': ('configs/sam2.1/sam2.1_hiera_l.yaml',  'sam2.1_hiera_large.pt'),
        }
        cfg, ckpt = cfg_map[self.sam_model_size]
        model = build_sam2(cfg, os.path.join(ckpt_dir, ckpt), device=self.device,
                           apply_postprocessing=False)
        self._sam_amg = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=32,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.85,
            box_nms_thresh=0.7,
            min_mask_region_area=50,
        )
        print(f'SAM2.1-{self.sam_model_size} AMG loaded.')
        return self._sam_amg

    @torch.no_grad()
    def _dino_feat_of_crop(self, rgb, mask):
        """Extract a single DINOv2 CLS-like feature (mean of patch tokens) for the
        image region under mask. Uses a 224x224 masked crop."""
        from torchvision import transforms
        import PIL.Image
        ys, xs = np.where(mask)
        if len(ys) < 10:
            return None
        y0, y1 = ys.min(), ys.max()+1
        x0, x1 = xs.min(), xs.max()+1
        # Pad bbox to square to preserve aspect
        h, w = y1-y0, x1-x0
        side = max(h, w)
        cy, cx = (y0+y1)//2, (x0+x1)//2
        H_img, W_img = rgb.shape[:2]
        y0p = max(0, cy - side//2); y1p = min(H_img, y0p + side)
        x0p = max(0, cx - side//2); x1p = min(W_img, x0p + side)
        crop_rgb = rgb[y0p:y1p, x0p:x1p].copy()
        crop_mask = mask[y0p:y1p, x0p:x1p]
        # Black out non-mask pixels (CNOS standard trick)
        crop_rgb[~crop_mask] = 0
        img_pil = PIL.Image.fromarray(crop_rgb).resize((224, 224))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        img_t = transform(img_pil).unsqueeze(0).to(self.device)
        out = self._dino.get_intermediate_layers(
            img_t, n=[self._dino.n_blocks - 1], return_class_token=True
        )
        # CLS token matches template feat format (global descriptor)
        cls_token = out[0][1].squeeze(0).cpu().numpy()  # (D,)
        return cls_token

    @torch.no_grad()
    def segment(self, rgb: np.ndarray, depth: Optional[np.ndarray] = None,
                n_masks: int = 5, conf_threshold: float = 0.3
                ) -> List[Tuple[np.ndarray, float]]:
        """Proper CNOS: SAM2 AMG proposals -> DINOv2 re-rank against templates.
        Falls back to v0 if SAM2 unavailable."""
        if self._template_feats is None:
            raise RuntimeError('Call precompute_templates() first.')
        self._load_dino()
        try:
            amg = self._load_sam_generator()
        except Exception as e:
            print(f'SAM2 AMG load failed: {e}; falling back to v0')
            return self._segment_v0(rgb, depth, n_masks, conf_threshold)

        proposals = amg.generate(rgb)
        if not proposals:
            return []
        # Normalise template features
        T_norm = self._template_feats / (
            np.linalg.norm(self._template_feats, axis=1, keepdims=True) + 1e-8
        )
        scored = []
        for p in proposals:
            m = p['segmentation'].astype(bool)
            if m.sum() < 50:
                continue
            feat = self._dino_feat_of_crop(rgb, m)
            if feat is None:
                continue
            fn = feat / (np.linalg.norm(feat) + 1e-8)
            sim = float((T_norm @ fn).max())
            scored.append((m, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = [s for s in scored if s[1] >= conf_threshold]
        return scored[:n_masks]

    def _segment_v0(self, rgb: np.ndarray, depth: Optional[np.ndarray] = None,
                    n_masks: int = 5, conf_threshold: float = 0.3
                    ) -> List[Tuple[np.ndarray, float]]:
        """
        Generate candidate masks for the query object in the scene.
        Returns list of (mask, confidence) sorted by confidence desc.
        """
        if self._template_feats is None:
            raise RuntimeError("Call precompute_templates() first.")

        self._load_dino()
        H_img, W_img = rgb.shape[:2]

        # Extract scene DINOv2 features (patch-level)
        scene_feat_map = self._extract_scene_features(rgb)  # (n_ph, n_pw, D)
        n_ph, n_pw = scene_feat_map.shape[:2]

        # Find top-K matching regions using sliding window at multiple scales
        proposals = self._find_top_regions(scene_feat_map, H_img, W_img, n_masks * 3)

        # Generate masks for each proposal
        masks_with_scores = []
        sam_ok = self._load_sam()

        for (y0, x0, y1, x1), sim_score in proposals:
            if sim_score < conf_threshold:
                continue

            if sam_ok:
                mask = self._sam_mask_from_box(rgb, x0, y0, x1, y1)
            else:
                mask = self._depth_mask_from_box(depth, x0, y0, x1, y1, H_img, W_img)

            if mask is not None and mask.sum() > 200:
                masks_with_scores.append((mask, float(sim_score)))

        # Also add depth-based fallback
        if depth is not None and len(masks_with_scores) < n_masks:
            fallback = self._depth_fallback(depth, n=n_masks - len(masks_with_scores))
            masks_with_scores.extend(fallback)

        masks_with_scores.sort(key=lambda x: x[1], reverse=True)
        return masks_with_scores[:n_masks]

    @torch.no_grad()
    def _extract_scene_features(self, rgb: np.ndarray) -> np.ndarray:
        from torchvision import transforms
        import PIL.Image

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        H, W = rgb.shape[:2]
        new_h = (H // 14) * 14
        new_w = (W // 14) * 14
        img_pil = PIL.Image.fromarray(rgb).resize((new_w, new_h))
        img_t = transform(img_pil).unsqueeze(0).to(self.device)

        n_ph, n_pw = new_h // 14, new_w // 14
        out = self._dino.get_intermediate_layers(
            img_t, n=[self._dino.n_blocks - 1], return_class_token=False
        )[0]  # (1, n_ph*n_pw, D)
        feat_map = out[0].reshape(n_ph, n_pw, -1).cpu().numpy()
        return feat_map

    def _find_top_regions(self, feat_map: np.ndarray, H_img: int, W_img: int,
                           n: int) -> list:
        """
        Slide windows of various sizes over the feature map.
        Score each window by max cosine similarity to templates.
        Returns list of ((y0,x0,y1,x1)_image_coords, score).
        """
        n_ph, n_pw, D = feat_map.shape

        # Normalise template and patch features
        T_norm = self._template_feats / (
            np.linalg.norm(self._template_feats, axis=1, keepdims=True) + 1e-8
        )  # (N_t, D)

        # Window sizes in patch units
        win_sizes = [
            (max(1, n_ph // 4), max(1, n_pw // 4)),
            (max(1, n_ph // 3), max(1, n_pw // 3)),
            (max(1, n_ph // 2), max(1, n_pw // 2)),
        ]

        proposals = []
        for (wh, ww) in win_sizes:
            for r in range(0, n_ph - wh + 1, max(1, wh // 2)):
                for c in range(0, n_pw - ww + 1, max(1, ww // 2)):
                    patch = feat_map[r:r+wh, c:c+ww].reshape(-1, D).mean(0)
                    patch_n = patch / (np.linalg.norm(patch) + 1e-8)
                    sims = T_norm @ patch_n
                    score = float(sims.max())

                    # Convert to image coordinates
                    y0 = int(r * H_img / n_ph)
                    x0 = int(c * W_img / n_pw)
                    y1 = int((r + wh) * H_img / n_ph)
                    x1 = int((c + ww) * W_img / n_pw)
                    proposals.append(((y0, x0, y1, x1), score))

        proposals.sort(key=lambda x: x[1], reverse=True)
        # NMS
        kept = []
        for box, score in proposals:
            if not any(self._iou(box, k[0]) > 0.5 for k in kept):
                kept.append((box, score))
            if len(kept) >= n:
                break
        return kept

    @staticmethod
    def _iou(a, b):
        y0 = max(a[0], b[0]); x0 = max(a[1], b[1])
        y1 = min(a[2], b[2]); x1 = min(a[3], b[3])
        inter = max(0, y1-y0) * max(0, x1-x0)
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (b[2]-b[0]) * (b[3]-b[1])
        return inter / (area_a + area_b - inter + 1e-8)

    def _sam_mask_from_box(self, rgb, x0, y0, x1, y1):
        try:
            import torch as _torch
            self._sam.set_image(rgb)
            box = np.array([x0, y0, x1, y1])
            masks, scores, _ = self._sam.predict(box=box, multimask_output=True)
            best = masks[scores.argmax()]
            return best.astype(bool)
        except Exception as e:
            return None

    def _depth_mask_from_box(self, depth, x0, y0, x1, y1, H, W):
        if depth is None:
            return None
        roi = depth[y0:y1, x0:x1]
        valid = roi[roi > 0]
        if len(valid) < 10:
            return None
        d_med = np.median(valid)
        d_lo, d_hi = d_med * 0.8, d_med * 1.2
        full_mask = (depth >= d_lo) & (depth <= d_hi)
        roi_mask = np.zeros((H, W), dtype=bool)
        roi_mask[y0:y1, x0:x1] = full_mask[y0:y1, x0:x1]
        kernel = np.ones((7, 7), np.uint8)
        cleaned = cv2.morphologyEx(roi_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        return cleaned.astype(bool)

    def _depth_fallback(self, depth, n=3):
        if depth is None:
            return []
        valid = depth > 0
        if valid.sum() < 100:
            return []
        d_med = np.median(depth[valid])
        base_mask = valid & (depth > d_med * 0.7) & (depth < d_med * 1.3)
        kernel = np.ones((7, 7), np.uint8)
        cleaned = cv2.morphologyEx(base_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
        results = []
        order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1]
        for lbl in order[:n] + 1:
            m = (labels == lbl).astype(bool)
            if m.sum() > 200:
                area = float(m.sum()) / (depth.shape[0] * depth.shape[1])
                results.append((m, area))
        return results

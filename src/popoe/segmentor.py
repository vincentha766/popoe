"""
Zero-shot instance segmentation wrappers.
Primary: SAM2.1 automatic mask generator.
Fallback: depth-based connected-component segmentation.

Installation:
    pip install git+https://github.com/facebookresearch/sam2.git
    wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt \
         -P /workspace/sam2_checkpoints/
"""

import os
import numpy as np
import torch
import cv2

torch.backends.cudnn.enabled = False

# SAM2.1 configs (official sam-2 >= 1.0) and matching checkpoints (092824 release)
_SAM2_CONFIGS = {
    'tiny':  ('configs/sam2.1/sam2.1_hiera_t.yaml',  'sam2.1_hiera_tiny.pt'),
    'small': ('configs/sam2.1/sam2.1_hiera_s.yaml',  'sam2.1_hiera_small.pt'),
    'base':  ('configs/sam2.1/sam2.1_hiera_b+.yaml', 'sam2.1_hiera_base_plus.pt'),
    'large': ('configs/sam2.1/sam2.1_hiera_l.yaml',  'sam2.1_hiera_large.pt'),
}
_CKPT_DIR = os.environ.get('POPOE_SAM2_CKPT', '/workspace/sam2_checkpoints')


def _load_sam2(model_size='small', device='cuda'):
    """Load SAM2.1 model. Returns (model, generator_class) or (None, None)."""
    import os
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    cfg, ckpt_name = _SAM2_CONFIGS[model_size]
    ckpt_path = os.path.join(_CKPT_DIR, ckpt_name)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {ckpt_path}\n"
            f"Download with:\n"
            f"  mkdir -p {_CKPT_DIR}\n"
            f"  wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/{ckpt_name}"
            f" -P {_CKPT_DIR}/"
        )

    model = build_sam2(cfg, ckpt_path, device=device, apply_postprocessing=False)
    return model, SAM2AutomaticMaskGenerator


class SAMSegmentor:
    """SAM2.1-based zero-shot segmentor (automatic mask generator)."""

    def __init__(self, device='cuda', model_size='small'):
        self.device = device
        self.model_size = model_size
        self._generator = None   # SAM2AutomaticMaskGenerator instance
        self._load_error = None  # cache error to avoid repeated attempts

    def _load(self):
        if self._generator is not None or self._load_error is not None:
            return
        try:
            model, GeneratorCls = _load_sam2(self.model_size, self.device)
            self._generator = GeneratorCls(
                model,
                points_per_side=32,
                pred_iou_thresh=0.7,
                stability_score_thresh=0.85,
                box_nms_thresh=0.7,
                min_mask_region_area=200,
            )
            print(f"SAM2.1-{self.model_size} loaded.")
        except Exception as e:
            print(f"SAM2 load failed: {e}. Using depth fallback.")
            self._load_error = str(e)

    def segment(self, rgb, depth=None, n_masks=5, conf_threshold=0.4):
        """
        Returns list of (mask bool H×W, confidence float) sorted by confidence desc.
        """
        self._load()
        if self._generator is None:
            return self._fallback_segment(rgb, depth)

        try:
            masks_data = self._generator.generate(rgb)
            results = []
            for m in sorted(masks_data, key=lambda x: x['predicted_iou'], reverse=True):
                if float(m['predicted_iou']) < conf_threshold:
                    continue
                results.append((m['segmentation'].astype(bool), float(m['predicted_iou'])))
                if len(results) >= n_masks:
                    break
            return results if results else self._fallback_segment(rgb, depth)
        except Exception as e:
            print(f"SAM2 generate failed: {e}")
            return self._fallback_segment(rgb, depth)

    def _fallback_segment(self, rgb, depth):
        """Depth-based connected-component fallback."""
        if depth is None:
            return []
        valid = depth > 0
        if not valid.any():
            return []
        median_d = float(np.median(depth[valid]))
        mask = valid & (depth < median_d * 1.3) & (depth > median_d * 0.7)

        kernel = np.ones((5, 5), np.uint8)
        mask_u8 = mask.astype(np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8)
        h, w = depth.shape
        results = []
        for lbl in range(1, n_labels):
            comp = (labels == lbl).astype(bool)
            if comp.sum() > 100:
                results.append((comp, float(comp.sum()) / (h * w)))
        return sorted(results, key=lambda x: x[1], reverse=True)[:5]


def get_dense_target_pcd(depth, mask, intrinsics):
    """Build dense point cloud from depth map within mask."""
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    ys, xs = np.where(mask & (depth > 0))
    d = depth[ys, xs]
    X = (xs - cx) * d / fx
    Y = (ys - cy) * d / fy
    return np.stack([X, Y, d], axis=1).astype(np.float32)

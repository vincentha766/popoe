"""
Feature fusion — the first extracted swappable stage.

Historically this logic was copy-pasted into both QueryFeatureExtractor and
TargetFeatureExtractor as a private `_fuse_features` method. That duplication
(and the DINOv2+GeDi concat being hard-wired inside the extractors) was the #1
obstacle to swapping the fusion rule for ablations. It now lives here as a
`FeatureFusion` component (see popoe/interfaces.py) that both extractors
delegate to.

Behaviour is byte-identical to the old `_fuse_features` when `vis_dim` /
`vis_weight` are left as None (they then fall back to the same env vars).
Passing them explicitly is what enables one-line ablations, e.g.

    DinoGeDiFusion(vis_weight=0.0)   # pure-geometric  (== POPOE_VIS_WEIGHT=0)
    DinoGeDiFusion(vis_weight=1.0)   # balanced 50/50
"""

import os
import numpy as np
from sklearn.decomposition import PCA


class DinoGeDiFusion:
    """FreeZe v2 default fusion (paper Eq. 1/2):

        fused = [ vis_weight * L2(PCA(f_vis)) ,  L2(f_geo) ]

    The visual PCA is fit ONCE (on the query object's features) and then reused
    on the target side so query/target live in the same reduced space. Share a
    single instance across the query and target extractors, or copy `.pca_vis`
    from the query fusion onto the target fusion (this is what the pipeline
    does today via `target_extractor._pca_vis = query_extractor._pca_vis`).

    Args:
        vis_dim:    PCA target dim for the visual branch. None -> match the
                    geometric dim at fuse() time (env POPOE_VIS_DIM override).
        vis_weight: scale on the L2-normed visual branch. In cosine matching the
                    effective visual fraction is w^2/(1+w^2): w=1 -> 50/50,
                    w<1 -> geometry-led, w=0 -> pure geometric. None -> env
                    POPOE_VIS_WEIGHT (default 0.5).
        pca_vis:    pre-fit PCA to reuse (target side); None -> fit lazily.
    """

    def __init__(self, vis_dim=None, vis_weight=None, pca_vis=None):
        self.vis_dim = vis_dim
        self.vis_weight = vis_weight
        self.pca_vis = pca_vis

    def fuse(self, vis_feats, geo_feats, apply_skip_vis=False):
        """Fuse per-point visual and geometric features into one array.

        Args:
            vis_feats: (N, D_vis) raw visual (DINOv2) features.
            geo_feats: (N, D_geo) geometric (GeDi) features; NaN rows = invalid.
            apply_skip_vis: if True, honour POPOE_SKIP_VIS=1 by zeroing the
                visual branch. Only the QUERY side sets this (preserves the
                original asymmetry — see popoe/freeze/feature_extractor.py).
        Returns:
            (N, D_vis' + D_geo) float32 fused features.
        """
        if apply_skip_vis and os.environ.get("POPOE_SKIP_VIS") == "1":
            vis_feats = vis_feats * 0.0

        valid = ~np.isnan(geo_feats).any(axis=1)
        n_vis = vis_feats.shape[1]

        # Match visual PCA dim to the geometric dim so fused = 2xD_geo (balanced
        # contribution, paper Eq.1/2). Two-scale GeDi -> 64D geo -> 64D vis -> 128D.
        vis_dim = self.vis_dim
        if vis_dim is None:
            vis_dim = int(os.environ.get("POPOE_VIS_DIM", str(geo_feats.shape[1])))

        if self.pca_vis is None and valid.sum() > vis_dim and n_vis > vis_dim:
            self.pca_vis = PCA(n_components=vis_dim)
            self.pca_vis.fit(vis_feats[valid])
            # Canonicalise component SIGNS (largest-|loading| entry positive).
            # PCA signs are arbitrary per fit; two fits on slightly different
            # query samples agree up to sign flips, and a flipped TOP component
            # scrambles cosine similarity against features projected with the
            # other fit (measured: flipped-variance-mass 29-48% <-> AR 0.16-0.25
            # vs 3-5% <-> AR 0.79-0.85 on YCB-V obj8). With canonical signs any
            # two fits of the same object produce compatible bases, so cached
            # target features stay valid across runs.
            comps = self.pca_vis.components_
            signs = np.sign(comps[np.arange(len(comps)),
                                  np.abs(comps).argmax(axis=1)])
            signs[signs == 0] = 1.0
            self.pca_vis.components_ = comps * signs[:, None]

        # Reduction is PCA projection or nothing. It used to fall back to a raw
        # slice of the first vis_dim DINO dims (or zero-padding) whenever the
        # PCA was absent or its width disagreed — a materially different visual
        # representation served under the same name, with nothing recorded in
        # the cache key. That is the substitution popoe.interfaces'
        # availability contract exists to forbid, and the one place it would
        # bias every number in the study rather than one stage's output.
        if self.pca_vis is not None:
            if n_vis != self.pca_vis.n_features_in_:
                raise ValueError(
                    f"visual features are {n_vis}-D but the installed PCA was "
                    f"fitted on {self.pca_vis.n_features_in_}-D. This is the "
                    f"wrong basis for these features (a stale snapshot, or a "
                    f"changed DINO layer); slicing to {vis_dim}-D instead would "
                    f"quietly swap PCA projection for raw DINO dims. Install "
                    f"the matching PCA, or re-encode.")
            vis_reduced = self.pca_vis.transform(vis_feats)
        elif n_vis == vis_dim:
            # Nothing to reduce: the visual branch already has the target width,
            # so passing it through is the identity — not a substituted method.
            vis_reduced = vis_feats
        else:
            why = (f"only {int(valid.sum())} of {len(valid)} points have valid "
                   f"geometric features, which is not more than vis_dim "
                   f"{vis_dim}" if n_vis > vis_dim else
                   f"vis_dim {vis_dim} exceeds the {n_vis}-D visual features")
            raise ValueError(
                f"cannot reduce {n_vis}-D visual features to {vis_dim}-D: no "
                f"PCA could be fitted ({why}). Truncating or zero-padding here "
                f"would substitute a different visual representation under the "
                f"same name and leave no trace in the cache key. Fix the input "
                f"(a cloud this degenerate is not describable), or install a "
                f"fitted PCA.")

        def l2norm(x):
            norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
            return x / norms

        vis_w = self.vis_weight
        if vis_w is None:
            vis_w = float(os.environ.get("POPOE_VIS_WEIGHT", "0.5"))

        geo_safe = geo_feats.copy()
        geo_safe[~valid] = 0
        fused = np.concatenate([vis_w * l2norm(vis_reduced), l2norm(geo_safe)], axis=1)
        return fused.astype(np.float32)

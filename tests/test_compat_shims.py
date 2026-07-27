"""The freeze/ extraction kept the old import paths alive as shims — lock that.

Only the LIGHT shims are checked. popoe.feature_extractor is excluded on
purpose: importing it pulls torch + GeDi (same as before the move), which the
offline suite does not have.
"""


def test_fusion_shim():
    from popoe.freeze.fusion import DinoGeDiFusion as new
    from popoe.fusion import DinoGeDiFusion as old
    assert old is new


def test_recipes_shim():
    import popoe.freeze.recipes as new
    import popoe.recipes as old
    for name in ("SOLVERS", "TAU_FRAC", "WEIGHTS", "YCBV_MERGE_LABELS",
                 "best_encoders", "best_segmentor", "scale_vis",
                 "stages_for_object"):
        assert getattr(old, name) is getattr(new, name), name


def test_freeze_adapters_reexported_from_adapters():
    import popoe.adapters as old
    import popoe.freeze.adapters as new
    for name in ("FreeZeQueryEncoder", "FreeZeTargetEncoder", "FreeZeScorer",
                 "make_freeze_encoders"):
        assert getattr(old, name) is getattr(new, name), name


def test_pose_estimator_shim():
    import popoe.pose_estimator as old
    import popoe.registration as new
    for name in ("top_k_correspondences", "feature_aware_score",
                 "ransac_pose_estimation", "icp_refinement", "final_score"):
        assert getattr(old, name) is getattr(new, name), name


def test_freeze_package_exports():
    import popoe.freeze as fz
    for name in fz.__all__:
        assert getattr(fz, name) is not None, name


def test_segmentor_cnos_v3_shim():
    """segmentor_cnos_v3 -> segmentor_cnos_lab rename (the 'v3' read as an
    official CNOS release; it was gedi's cnos_match3.py iteration count)."""
    import popoe.segmentor_cnos_lab as new
    import popoe.segmentor_cnos_v3 as old
    for name in ("DepthSizeGate", "DiameterSizeModel",
                 "DinoV2ForegroundPatchExtractor", "PatchForegroundScorer",
                 "SAM2AMGMaskProposer", "TemplateDirPatchBank",
                 "select_by_nearest_diameter", "select_by_soft_affinity",
                 "square_crop"):
        assert getattr(old, name) is getattr(new, name), name
    assert old.CNOSv3Segmentor is new.CNOSLabSegmentor
    # the constant follows the rename: old importers now emit the new label
    assert old.CNOS_V3_SOURCE == new.CNOS_LAB_SOURCE == "cnos-lab"

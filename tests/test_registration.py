"""registration.py numeric contracts — GPU-free (numpy + sklearn only).

These pin the ONE thing about `feature_aware_score` that a reader is most
likely to "fix" in the wrong direction: its denominator.
"""
import numpy as np

from popoe.registration import feature_aware_score


def _fixture():
    """Two of four target points fall inside tau; their cosines are 1.0 and 0.5.

    So the two candidate normalisations are far apart and cannot be confused:
        mean over inliers   = 1.5 / 2 = 0.75      <- what this function returns
        Eq.5, fixed |P_T|   = 1.5 / 4 = 0.375     <- what the docstring used to
                                                     claim, and what the GPU
                                                     solver's fitness computes
    """
    pts_query = np.array([[0., 0, 0], [1, 0, 0], [10, 0, 0], [11, 0, 0]])
    pts_target = np.array([[0., 0, 0], [1, 0, 0], [100, 0, 0], [101, 0, 0]])
    feats_query = np.tile(np.array([[1., 0.]]), (4, 1))
    feats_target = np.array([[1., 0.],                    # cos vs query0 = 1.0
                             [0.5, np.sqrt(0.75)],        # cos vs query1 = 0.5
                             [1., 0.], [1., 0.]])         # outliers, not scored
    return pts_query, pts_target, feats_query, feats_target


def test_denominator_is_the_inlier_count_not_the_target_count():
    """Eq.5 divides by |P_T^sparse|; this function divides by |I|, on purpose.

    Confusing the two was measured at -31 pt (ISSUES.md, B-layer): a fixed
    denominator rewards inlier quantity x quality, the mean rewards quality
    alone. Callers here supply the quantity term separately as s_icp, so the
    mean is the correct half — but only as long as it stays the mean.
    """
    pts_q, pts_t, fq, ft = _fixture()
    score, inliers = feature_aware_score(
        np.eye(3), np.zeros(3), pts_q, pts_t, fq, ft, tau_inlier=0.5)

    assert inliers.sum() == 2, "fixture must produce exactly 2 inliers"
    assert np.isclose(score, 0.75), "score must be the MEAN cosine over inliers"
    assert not np.isclose(score, 0.375), (
        "score divided by |P_T| — that is the Eq.5 fitness, which belongs to "
        "solvers.gpu_ransac(fitness='feature'), not here")


def test_inlier_mask_is_over_target_points():
    pts_q, pts_t, fq, ft = _fixture()
    _, inliers = feature_aware_score(
        np.eye(3), np.zeros(3), pts_q, pts_t, fq, ft, tau_inlier=0.5)
    assert inliers.shape == (len(pts_t),)
    assert list(inliers) == [True, True, False, False]


def test_no_inliers_scores_zero():
    # Shrinking tau cannot empty the inlier set here: two target points sit at
    # distance EXACTLY 0 from a query point, and 0 < tau for every tau > 0.
    # Translate the query away instead.
    pts_q, pts_t, fq, ft = _fixture()
    score, inliers = feature_aware_score(
        np.eye(3), np.array([1e3, 0., 0.]), pts_q, pts_t, fq, ft, tau_inlier=0.5)
    assert inliers.sum() == 0
    assert score == 0.0

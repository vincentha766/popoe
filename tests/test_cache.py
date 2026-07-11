"""StageCache: fingerprint stability, key sensitivity, and the two invariants
(fitted-state provenance in the key; content addressing)."""

import numpy as np

from popoe.cache import StageCache, fingerprint


def test_fingerprint_stability_and_sensitivity():
    a = {"grid": 32, "layer": 30, "arr": np.arange(6).reshape(2, 3)}
    b = {"layer": 30, "arr": np.arange(6).reshape(2, 3), "grid": 32}  # reordered
    assert fingerprint(a) == fingerprint(b)                # order-insensitive
    assert fingerprint(a) != fingerprint({**a, "grid": 16})       # config change
    c = dict(a); c["arr"] = np.arange(6).reshape(3, 2)            # shape change
    assert fingerprint(a) != fingerprint(c)
    d = dict(a); d["arr"] = a["arr"].copy(); d["arr"][0, 0] = 99  # content change
    assert fingerprint(a) != fingerprint(d)


def test_mask_content_addressing():
    """Invariant 2: identity follows pixels, not list position."""
    m1 = np.zeros((8, 8), bool); m1[:4] = True
    m2 = np.zeros((8, 8), bool); m2[4:] = True
    cfg = {"grid": 32}
    k_pos0_m1 = fingerprint("target", cfg, 1, 7, m1)
    k_pos0_m2 = fingerprint("target", cfg, 1, 7, m2)
    assert k_pos0_m1 != k_pos0_m2          # different masks never alias
    assert k_pos0_m1 == fingerprint("target", cfg, 1, 7, m1)  # stable


def test_query_key_in_target_key():
    """Invariant 1: a different query fit invalidates its targets."""
    cfg = {"grid": 32}
    m = np.ones((4, 4), bool)
    qk_a = fingerprint("query", cfg, "mesh-hash-A", 3000, 8)
    qk_b = fingerprint("query", cfg, "mesh-hash-A", 3000, 9)   # different seed
    assert fingerprint("target", cfg, 1, 7, m, qk_a) != \
           fingerprint("target", cfg, 1, 7, m, qk_b)


def test_roundtrip(tmp_path):
    c = StageCache(str(tmp_path))
    key = fingerprint({"x": 1})
    assert c.get_arrays("query", key) is None
    pts = np.random.default_rng(0).standard_normal((10, 3))
    c.put_arrays("query", key, pts=pts, feats=pts * 2)
    out = c.get_arrays("query", key)
    assert np.allclose(out["pts"], pts) and np.allclose(out["feats"], pts * 2)
    c.put_pickle("query", key, {"fit": [1, 2, 3]})
    assert c.get_pickle("query", key) == {"fit": [1, 2, 3]}
    assert c.get_pickle("query", "nope") is None

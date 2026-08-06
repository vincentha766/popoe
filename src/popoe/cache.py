"""Config-addressed stage caching.

Every stage output is stored under a key that fingerprints (a) the stage's
CONFIGURATION and (b) the CONTENT of its inputs — so a rerun with the same
config reuses results automatically, and changing any upstream knob
invalidates exactly the entries it should. Measured payoff in the
reproduction study: feature extraction skipped entirely on reruns
(registration-only iterations), selection-rule changes with zero GPU, and
the whole obj8/obj21 investigation ran offline from caches.

Two invariants, both learned the hard way (see ISSUES.md):

1. **Fitted state is part of the key.** Anything FIT during a stage (the
   visual PCA, normalisation stats) makes downstream outputs functions of
   that fit. Cached target features are only valid together with the query
   fit that produced their basis — so the target key includes the QUERY key.
   (Violation symptom: silently scrambled cosines, texture-reliant objects
   crater across runs.)

2. **Content-addressed inputs, not positional indices.** A mask's cache
   identity is a hash of its pixels, not its index in a detection list —
   list order changes (e.g. label pooling) must not alias entries.
   (Violation symptom: features of a *different* mask load successfully.)

Storage layout: ``<root>/<stage>_<key>.npz`` for arrays plus optional
``.pkl`` sidecar for fitted objects (e.g. the PCA).
"""

from __future__ import annotations

import hashlib
import os
import pickle
from typing import Optional

import numpy as np


CONDITIONAL_ENC_KEYS = {
    # Paper-fidelity feature knobs: each enters enc_cfg ONLY at a non-default
    # value, so every cache built before a knob existed keeps its exact key.
    # THE single authority — bop_eval and scripts/flip_rescore_ab both build
    # their conditional entries from here. The mirrored copy in flip_rescore
    # drifted THREE times (missing keys -> faithful lookups miss, or worse,
    # hit legacy-render features under a stale key); a shared table is the
    # only fix that stays fixed.
    "query_canon": ("POPOE_QUERY_CANON", "224"),
    "query_fill": ("POPOE_QUERY_FILL", "0.45"),
    "query_fill_mode": ("POPOE_QUERY_FILL_MODE", "legacy"),
    "query_min_views": ("POPOE_QUERY_MIN_VIEWS", "0"),
    "canon_basis": ("POPOE_CANON_BASIS", "extent"),
    "query_views": ("POPOE_QUERY_VIEWS", "spiral"),
    "target_dense": ("POPOE_TARGET_DENSE", "0"),
    "target_paper_grid": ("POPOE_TARGET_PAPER_GRID", "0"),
}


def conditional_enc_entries() -> dict:
    """{cfg_key: env_value} for every CONDITIONAL_ENC_KEYS knob currently at a
    NON-default value. Callers merge this into their enc_cfg."""
    out = {}
    for key, (env, dflt) in CONDITIONAL_ENC_KEYS.items():
        val = os.environ.get(env, dflt)
        if val != dflt:
            out[key] = val
    return out


def fingerprint(*parts) -> str:
    """Stable content hash of nested dicts / sequences / arrays / scalars.
    Dicts are order-insensitive; arrays hash dtype+shape+bytes."""
    h = hashlib.sha256()

    def feed(x):
        if isinstance(x, dict):
            h.update(b"{")
            for k in sorted(x, key=repr):
                feed(k); feed(x[k])
            h.update(b"}")
        elif isinstance(x, (list, tuple)):
            h.update(b"[")
            for v in x:
                feed(v)
            h.update(b"]")
        elif isinstance(x, np.ndarray):
            h.update(str(x.dtype).encode())
            h.update(str(x.shape).encode())
            h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, (bytes, bytearray)):
            h.update(bytes(x))
        else:
            h.update(repr(x).encode())
        h.update(b"|")

    for p in parts:
        feed(p)
    return h.hexdigest()[:24]


def file_fingerprint(path: str) -> str:
    """Content hash of a file (e.g. a mesh) — cheap enough for CAD models."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:24]


class StageCache:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, stage: str, key: str, ext: str) -> str:
        return os.path.join(self.root, f"{stage}_{key}.{ext}")

    def get_arrays(self, stage: str, key: str) -> Optional[dict]:
        p = self._path(stage, key, "npz")
        if not os.path.exists(p):
            return None
        z = np.load(p)
        return {k: z[k] for k in z.files}

    def put_arrays(self, stage: str, key: str, **arrays) -> None:
        # np.savez appends ".npz" when missing — keep the tmp name ending in it.
        tmp = self._path(stage, key + ".tmp", "npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, self._path(stage, key, "npz"))   # atomic publish

    def get_pickle(self, stage: str, key: str):
        p = self._path(stage, key, "pkl")
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            return pickle.load(f)

    def put_pickle(self, stage: str, key: str, obj) -> None:
        tmp = self._path(stage, key, "pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f)
        os.replace(tmp, self._path(stage, key, "pkl"))

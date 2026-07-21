"""Deprecated shim — the DINOv2+GeDi extractors are FreeZe-specific and moved
to popoe.freeze.feature_extractor. Import from there; this path keeps old
imports working. (Heavy: importing this pulls torch and GeDi, as before.)"""
from popoe.freeze.feature_extractor import *  # noqa: F401,F403
from popoe.freeze.feature_extractor import (  # noqa: F401
    QueryFeatureExtractor, TargetFeatureExtractor, load_dinov2, load_gedi,
)

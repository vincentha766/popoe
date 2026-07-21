"""Deprecated shim — the DINOv2+GeDi fusion is FreeZe-specific and moved to
popoe.freeze.fusion. Import from there; this path keeps old imports working."""
from popoe.freeze.fusion import DinoGeDiFusion  # noqa: F401

__all__ = ["DinoGeDiFusion"]

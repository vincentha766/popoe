"""Dataset helpers.

`popoe.datasets.bop` loads BOP test-split RGB-D + GT. `frames` loads generic
RGB-D frame manifests for real captures.
"""

from popoe.datasets.frames import (
    FrameManifest, load_frame_manifest, load_scene_from_manifest,
)

__all__ = ["FrameManifest", "load_frame_manifest", "load_scene_from_manifest"]

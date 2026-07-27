"""Import shim: ``segmentor_cnos_v3`` was renamed to ``segmentor_cnos_lab``.

The "v3" in the old name was the iteration count of gedi's ``cnos_match3.py``
lab script; outside readers reliably misread it as an official CNOS release,
so the recipe is now ``cnos-lab`` everywhere. This module keeps old imports
working (same pattern as the freeze/ extraction shims — locked in
tests/test_compat_shims.py).

Note ``CNOS_V3_SOURCE`` re-exports the RENAMED label ``"cnos-lab"``: the
constant names the recipe, and the recipe's label changed. Readers of old
artifacts with ``source="cnos-v3"`` should treat the two labels as the same
recipe.
"""
from popoe.segmentor_cnos_lab import *  # noqa: F401,F403
from popoe.segmentor_cnos_lab import (  # noqa: F401
    CNOS_LAB_SOURCE,
    CNOSLabSegmentor,
    GRID,
)

# Old names, kept importable.
CNOSv3Segmentor = CNOSLabSegmentor
CNOS_V3_SOURCE = CNOS_LAB_SOURCE

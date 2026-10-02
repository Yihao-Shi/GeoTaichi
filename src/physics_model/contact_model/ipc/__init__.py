"""Shared Incremental Potential Contact building blocks.

The submodules separate constitutive laws, closest-point geometry, distance
derivatives, mollifiers, geometric measures, matrix assembly, and NURBS
contact utilities.  The constitutive symbols remain available directly from
this package as the stable public API.
"""

from .IPC import *  # noqa: F401,F403
from .IPC import __all__ as _constitutive_all
from .LevelSetAffine import *  # noqa: F401,F403
from .LevelSetAffine import __all__ as _levelset_affine_all
from .AffineDifferentiable import *  # noqa: F401,F403
from .AffineDifferentiable import __all__ as _affine_diff_all

__all__ = (
    list(_constitutive_all)
    + list(_levelset_affine_all)
    + list(_affine_diff_all)
)

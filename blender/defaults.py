"""Blender-only solver metadata and add-on defaults.

This module deliberately contains presentation/routing defaults only.  The
numerical defaults remain owned by each GeoTaichi solver and its model script.
"""

from __future__ import annotations

import os


PHASE_ITEMS = [
    ("IDLE", "Idle", "No exported SolverJob"),
    ("EXPORTED", "Exported", "SolverJob is ready for validation"),
    ("VALIDATED", "Validated", "SolverJob and referenced manifests are valid"),
    ("SUBMITTED", "Submitted", "Task is waiting for its worker"),
    ("RUNNING", "Running", "GeoTaichi worker is active"),
    ("COMPLETED", "Completed", "Task completed successfully"),
    ("FAILED", "Failed", "Task failed"),
    ("INTERRUPTED", "Interrupted", "Task was interrupted"),
]

ROLE_ITEMS = [
    ("REFERENCE", "Reference", "Export metadata only"),
    ("COLLIDER", "Collider", "Static or kinematic collision geometry"),
    ("FEM", "FEM", "Finite-element geometry"),
    ("MPM", "MPM", "Material-point source or boundary geometry"),
    ("DEM", "DEM", "Discrete-element source or rigid geometry"),
    ("IGA", "IGA", "Isogeometric source or boundary geometry"),
]

SOLVER_ITEMS = [
    ("MPM", "MPM", "Material Point Method"),
    ("DEM", "DEM", "Discrete Element Method"),
    ("MPDEM", "MPDEM", "MPM-DEM coupling"),
    ("CFDEM", "CFDEM", "CFD-DEM coupling"),
    ("FEM", "FEM", "Finite Element Method"),
    ("FEDEM", "FEDEM", "FEM-DEM coupling"),
    ("FEMPM", "FEMPM", "FEM-MPM coupling"),
    ("IGA", "IGA", "Isogeometric Analysis"),
    ("IGAMPM", "IGAMPM", "IGA-MPM coupling"),
]

# The values are routing metadata, not a second copy of solver parameters.
# Reusing a value such as ``EXPLICIT_CONTACT`` across compatible families keeps
# the manifest vocabulary small while the family identifies the implementation.
SOLVER_MODE_ITEMS = [
    ("EXPLICIT", "Explicit", "Explicit time integration"),
    ("IMPLICIT", "Implicit", "Implicit time integration"),
    ("INCOMPRESSIBLE", "Incompressible", "Incompressible pressure projection"),
    (
        "TWO_LAYER_INCOMPRESSIBLE",
        "Two-layer incompressible",
        "Separate solid and fluid material points",
    ),
    (
        "TWO_PHASE_SINGLE_POINT",
        "Single-point two-phase",
        "One material point carries both phases",
    ),
    (
        "TWO_PHASE_SINGLE_POINT_SEMI_IMPLICIT",
        "Semi-implicit single-point",
        "Semi-implicit pressure solve for single-point two-phase MPM",
    ),
    ("LSDEM", "LSDEM", "Level-set discrete elements"),
    ("ABD", "ABD", "Affine-body dynamics"),
    ("EXPLICIT_CONTACT", "Explicit contact", "Explicit cross-solver contact"),
    ("IPC", "IPC implicit", "Barrier contact with CCD and implicit integration"),
    ("CLOTH", "Cloth", "Membrane or shell formulation"),
]

DIMENSION_ITEMS = [
    ("2", "2D", "Two-dimensional model"),
    ("3", "3D", "Three-dimensional model"),
    ("AXISYMMETRIC", "Axisymmetric", "Axisymmetric two-dimensional model"),
]

SOLVER_MODES_BY_FAMILY = {
    "MPM": (
        "EXPLICIT",
        "IMPLICIT",
        "INCOMPRESSIBLE",
        "TWO_LAYER_INCOMPRESSIBLE",
        "TWO_PHASE_SINGLE_POINT",
        "TWO_PHASE_SINGLE_POINT_SEMI_IMPLICIT",
        "IPC",
    ),
    "DEM": ("EXPLICIT", "LSDEM", "ABD", "IPC"),
    "MPDEM": ("EXPLICIT_CONTACT", "LSDEM", "ABD", "INCOMPRESSIBLE", "IPC"),
    "CFDEM": ("INCOMPRESSIBLE",),
    "FEM": ("EXPLICIT", "IMPLICIT", "CLOTH"),
    "FEDEM": ("EXPLICIT_CONTACT", "LSDEM", "ABD", "IPC"),
    "FEMPM": ("EXPLICIT_CONTACT", "IPC"),
    "IGA": ("EXPLICIT", "IMPLICIT"),
    "IGAMPM": ("EXPLICIT_CONTACT", "IPC"),
}

DEFAULT_PHASE = "IDLE"
DEFAULT_OBJECT_ROLE = "REFERENCE"
DEFAULT_SOLVER_FAMILY = "MPM"
DEFAULT_SOLVER_MODE = "EXPLICIT"
DEFAULT_DIMENSION = "3"
DEFAULT_EXPORT_DIRECTORY = "//geotaichi_jobs"
DEFAULT_CACHE_ROOT = "//geotaichi_cache"
DEFAULT_USE_STANDARD_ARGUMENTS = True
DEFAULT_RESULT_START_FRAME = 1
DEFAULT_RESULT_FRAME_STEP = 1
DEFAULT_RESULT_POINT_RADIUS = 0.005
DEFAULT_RESULT_RECURSIVE = True
DEFAULT_RESULT_REPLACE_EXISTING = True
DEFAULT_RESULT_UPDATE_TIMELINE = True


def default_python_executable() -> str:
    return "python" if os.name == "nt" else "python3"


def validate_solver_route(family: str, mode: str) -> tuple[str, str]:
    """Return a canonical Blender route or reject an incompatible pair."""
    family = str(family).strip().upper()
    mode = str(mode).strip().upper()
    if family not in SOLVER_MODES_BY_FAMILY:
        raise ValueError("unsupported Blender solver family: %s" % family)
    if mode not in SOLVER_MODES_BY_FAMILY[family]:
        raise ValueError("solver mode %s is not available for Blender family %s" % (mode, family))
    return family, mode

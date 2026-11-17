from __future__ import annotations
from src.utils.linalg import _euler_to_matrix
from src.utils.errors import ConfigurationError
import numpy as np
from src.dem.engines.IPCWallBoundary import IPCWallBoundary


def _normalize_affine_assemble_type(assemble_type):
    key = str(assemble_type).replace("_", "").replace("-", "").lower()
    aliases = {
        "matrixfree": "MatrixFree",
        "coo": "COO",
        "csr": "COO",
        "coordinate": "COO",
        "coordinatematrix": "COO",
        "hashtriplet": "HashTriplet",
        "triplet": "HashTriplet",
    }
    if key not in aliases:
        raise ConfigurationError("AffineBody assemble_type must be one of ['MatrixFree', 'COO', 'HashTriplet']")
    return aliases[key]


def _asarray3(value, default=None):
    if value is None:
        value = default if default is not None else [0.0, 0.0, 0.0]
    return np.asarray(value, dtype=np.float64).reshape(3)


def _field_scalar(field_numpy, name, index, default=0.0):
    if field_numpy is None:
        return default
    try:
        return float(field_numpy[name][index])
    except (KeyError, IndexError, TypeError, ValueError):
        return default


def ipc_wall_boundary(scene, sims):
    boundary = getattr(scene, "ipc_wall_boundary", None)
    if boundary is None:
        boundary = IPCWallBoundary(scene, sims)
        scene.ipc_wall_boundary = boundary
    return boundary

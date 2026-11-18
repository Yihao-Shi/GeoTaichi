"""Reference-space operators for triangular cloth finite elements."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ClothElement:
    connectivity: np.ndarray
    inverse_reference_jacobian: np.ndarray
    inverse_reference_metric: np.ndarray
    integration_weight: np.ndarray
    characteristic_length: np.ndarray
    bending_connectivity: np.ndarray
    bending_stiffness: np.ndarray
    bending_edge_length: np.ndarray
    bending_height: np.ndarray
    bending_rest_angle: np.ndarray

"""Preprocessing for optional cloth energies.

The resulting arrays are immutable device-input data.  Energy, gradient and
Hessian evaluation is implemented by :mod:`src.fem.cloth.ClothAssembler`.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ClothEnergyData:
    stitch_nodes: np.ndarray
    stitch_ratio: np.ndarray
    stitch_stiffness: np.ndarray
    spring_nodes: np.ndarray
    spring_target: np.ndarray
    spring_stiffness: np.ndarray
    sdf_nodes: np.ndarray
    sdf_target: np.ndarray
    sdf_normal: np.ndarray
    sdf_stiffness: np.ndarray
    sdf_dhat: np.ndarray
    node_area: np.ndarray

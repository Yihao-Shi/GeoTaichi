"""Surface extraction and broad-phase candidates for FEM contact."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ContactSurface:
    faces: np.ndarray
    edges: np.ndarray
    vertices: np.ndarray
    node_area: np.ndarray
    edge_area: np.ndarray
    face_body: np.ndarray
    edge_body: np.ndarray
    vertex_body: np.ndarray

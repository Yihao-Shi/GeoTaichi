from __future__ import annotations
from src.dem.neighbor.AffineBodyBoundingVolumeHierarchy import AffineBodyBoundingVolumeHierarchy
import numpy as np


def _coordination_value(value, default=16):
    if isinstance(value, (list, tuple, np.ndarray)):
        return max(int(v) for v in value) if len(value) > 0 else int(default)
    return int(value) if value is not None else int(default)


def _as_domain(domain):
    return np.maximum(np.asarray(domain, dtype=np.float64).reshape(3), 1.0e-6)


def make_affine_body_neighbor(sims, vertex_num, face_num, edge_num, dhat):
    from src.dem.neighbor.AffineBodyHierarchicalLinkedCell import AffineBodyHierarchicalLinkedCell
    from src.dem.neighbor.AffineBodyLinkedCell import AffineBodyLinkedCell

    search = str(getattr(sims, "search", "LinkedCell"))
    if search == "BVH":
        return AffineBodyBoundingVolumeHierarchy(sims, vertex_num, face_num, edge_num, dhat)
    if search == "HierarchicalLinkedCell":
        return AffineBodyHierarchicalLinkedCell(sims, vertex_num, face_num, edge_num, dhat)
    return AffineBodyLinkedCell(sims, vertex_num, face_num, edge_num, dhat)

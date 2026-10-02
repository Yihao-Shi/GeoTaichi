"""Persistent FEM--DEM contact records."""

import taichi as ti


@ti.dataclass
class FEDEMContact:
    particle_id: ti.i32
    face_id: ti.i32
    active: ti.i32
    normal_force: ti.types.vector(3, float)
    tangential_force: ti.types.vector(3, float)
    old_tangential_overlap: ti.types.vector(3, float)


@ti.dataclass
class FEDEMHistoryContact:
    face_id: ti.i32
    old_tangential_overlap: ti.types.vector(3, float)


@ti.dataclass
class FEMLevelSetContact:
    """One FEM boundary node querying one rigid level-set body."""

    rigid_id: ti.i32
    node_id: ti.i32
    active: ti.i32
    normal_force: ti.types.vector(3, float)
    tangential_force: ti.types.vector(3, float)
    old_tangential_overlap: ti.types.vector(3, float)
    normal_gap: float


@ti.dataclass
class FEMFacetWallContact:
    """Persistent state for one FEM surface node--DEM facet pair."""

    active: ti.i32
    normal_gap: float
    normal_force: ti.types.vector(3, float)
    tangential_force: ti.types.vector(3, float)
    old_tangential_overlap: ti.types.vector(3, float)


__all__ = [
    "FEDEMContact",
    "FEDEMHistoryContact",
    "FEMFacetWallContact",
    "FEMLevelSetContact",
]

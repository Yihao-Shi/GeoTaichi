"""Persistent FEM--MPM contact records."""

import taichi as ti


@ti.dataclass
class FEMPMContact:
    particle_id: ti.i32
    face_id: ti.i32
    active: ti.i32
    normal_force: ti.types.vector(3, float)
    tangential_force: ti.types.vector(3, float)
    old_tangential_overlap: ti.types.vector(3, float)


@ti.dataclass
class FEMPMHistoryContact:
    face_id: ti.i32
    old_tangential_overlap: ti.types.vector(3, float)


__all__ = ["FEMPMContact", "FEMPMHistoryContact"]

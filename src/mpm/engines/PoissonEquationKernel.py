"""Taichi kernels for MPM Poisson assembly."""

import taichi as ti


@ti.kernel
def build_coo_jacobi_diagonal(
    active_dofs: ti.i32,
    nonzeros: ti.i32,
    rows: ti.template(),
    columns: ti.template(),
    values: ti.template(),
    diagonal: ti.template(),
):
    for dof in range(active_dofs):
        diagonal[dof] = 0.0
    for entry in range(nonzeros):
        if rows[entry] == columns[entry] and rows[entry] < active_dofs:
            ti.atomic_add(diagonal[rows[entry]], values[entry])
    for dof in range(active_dofs):
        if ti.abs(diagonal[dof]) < 1.0e-30:
            diagonal[dof] = 1.0

"""Production-kernel regression for open boundaries in the MG level-0 matrix."""

import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from src.mpm.engines.EngineKernel import (
    kernel_assemble_incompressible_mg_A_level0,
    kernel_assemble_incompressible_mg_A_level0_cut_cell,
)

pytestmark = [pytest.mark.verification, pytest.mark.mpm, pytest.mark.cpu]


@ti.data_oriented
class FluidProperties:
    def __init__(self, density):
        self.density = density


@ti.kernel
def initialize_fields(
    cell_type: ti.template(),
    grid_type: ti.template(),
    fluid_sdf: ti.template(),
    face_fraction0: ti.template(),
    face_fraction1: ti.template(),
):
    for I in ti.grouped(cell_type):
        cell_type[I] = 2
        fluid_sdf[I] = 1.0
    for I in ti.grouped(grid_type):
        grid_type[I] = 1
    for I in ti.grouped(face_fraction0):
        face_fraction0[I] = 1.0
        face_fraction1[I] = 1.0
    for j in range(2):
        cell_type[-1, j] = 0
        for i in range(2):
            cell_type[i, j] = 1


def test_level0_matrix_uses_air_ghost_cells(taichi_runtime):
    del taichi_runtime
    ghost = 1
    cnum = ti.Vector([4, 4])
    offset = (-ghost, -ghost)
    cell_type = ti.field(int, shape=(4, 4), offset=offset)
    fluid_sdf = ti.field(float, shape=(4, 4), offset=offset)
    solid_sdf = ti.field(float, shape=(4, 4), offset=offset)
    solid_sdf.fill(1.0)
    grid_type = ti.field(int, shape=(2, 2))
    face_fraction0 = ti.field(float, shape=(5, 5), offset=offset)
    face_fraction1 = ti.field(float, shape=(5, 5), offset=offset)
    diagonal = ti.field(float, shape=(2, 2))
    positive_axis = ti.Vector.field(2, float, shape=(2, 2))
    dt = ti.field(float, shape=())
    dt[None] = 0.1
    initialize_fields(cell_type, grid_type, fluid_sdf, face_fraction0, face_fraction1)

    arguments = (
        ghost,
        cnum,
        dt,
        ti.Vector([1.0, 1.0]),
        ti.Vector([1.0, 1.0]),
        FluidProperties(density=2.0),
        grid_type,
        cell_type,
        fluid_sdf,
        False,
    )
    kernel_assemble_incompressible_mg_A_level0(*arguments, diagonal, positive_axis)
    np.testing.assert_allclose(diagonal.to_numpy()[0], [0.15, 0.15])

    kernel_assemble_incompressible_mg_A_level0_cut_cell(
        ghost,
        cnum,
        dt,
        ti.Vector([1.0, 1.0]),
        ti.Vector([1.0, 1.0]),
        FluidProperties(density=2.0),
        grid_type,
        cell_type,
        fluid_sdf,
        solid_sdf,
        False,
        face_fraction0,
        face_fraction1,
        face_fraction1,
        diagonal,
        positive_axis,
    )
    np.testing.assert_allclose(diagonal.to_numpy()[0], [0.15, 0.15])

    solid_sdf[-1, 0] = -1.0
    kernel_assemble_incompressible_mg_A_level0_cut_cell(
        ghost,
        cnum,
        dt,
        ti.Vector([1.0, 1.0]),
        ti.Vector([1.0, 1.0]),
        FluidProperties(density=2.0),
        grid_type,
        cell_type,
        fluid_sdf,
        solid_sdf,
        False,
        face_fraction0,
        face_fraction1,
        face_fraction1,
        diagonal,
        positive_axis,
    )
    np.testing.assert_allclose(diagonal.to_numpy()[0], [0.10, 0.15])

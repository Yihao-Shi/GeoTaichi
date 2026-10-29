"""Mixture continuity uses a single flux shared by the two adjacent cells."""

import numpy as np
import pytest
import taichi as ti

from src.mpm.engines import TwoPhaseDoubleLayerKernel as kernels
from src.mpm.structs.GridNode import NodeTwoPhase, NodeTwoPhase2D


@pytest.mark.parametrize("dimension", [2, 3])
def test_variable_porosity_rhs_conserves_closed_domain_flux(taichi_runtime, dimension):
    shape = (4,) * dimension
    spacing = np.array([0.1, 0.13, 0.17][:dimension])
    kinds = ti.field(ti.i32, shape=shape)
    kinds.fill(1)
    rhs = ti.field(ti.f64, shape=shape)
    rng = np.random.default_rng(372)
    porosity, fluid, solid = [], [], []
    expected = np.zeros(shape)
    for axis in range(dimension):
        face_shape = tuple(n + int(d == axis) for d, n in enumerate(shape))
        values = [rng.uniform(0.2, 0.8, face_shape), rng.normal(size=face_shape), rng.normal(size=face_shape)]
        boundary = [slice(None)] * dimension
        for side in (0, -1):
            boundary[axis] = side
            for velocity in values[1:]:
                velocity[tuple(boundary)] = 0.0
        flux = values[0] * values[1] + (1.0 - values[0]) * values[2]
        expected -= np.diff(flux, axis=axis) / spacing[axis]
        for fields, value in zip((porosity, fluid, solid), values):
            field = ti.field(ti.f64, shape=face_shape)
            field.from_numpy(value)
            fields.append(field)
    assemble = (
        kernels.kernel_assemble_double_layer_pressure_rhs
        if dimension == 2
        else kernels.kernel_assemble_double_layer_pressure_rhs3d
    )
    assemble(ti.Vector(spacing), kinds, *porosity, *fluid, *solid, rhs)
    np.testing.assert_allclose(rhs.to_numpy(), expected, rtol=1e-13, atol=1e-13)
    assert abs(rhs.to_numpy().sum()) < 1e-12


@pytest.mark.parametrize("dimension", [2, 3])
def test_solid_mac_projection_obeys_domain_normal_walls(taichi_runtime, dimension):
    cells = (4,) * dimension
    kinds = ti.field(ti.i32, shape=cells)
    kinds.fill(1)
    node_type = NodeTwoPhase2D if dimension == 2 else NodeTwoPhase
    node = node_type.field(shape=(5**dimension, 1))
    node.ms.fill(1.0)
    node.momentums.from_numpy(np.full((5**dimension, 1, dimension), 0.37))
    length = ti.Vector.field(dimension, ti.f64, shape=1)
    length.fill(0.0)
    mass, velocity = [], []
    for axis in range(dimension):
        shape = tuple(n + int(d == axis) for d, n in enumerate(cells))
        mass.append(ti.field(ti.f64, shape=shape))
        mass[-1].fill(1.0)
        velocity.append(ti.field(ti.f64, shape=shape))
    project = getattr(kernels, f"kernel_project_solid_grid_velocity_to_mac{dimension}d")
    project(
        1e-12, ti.Vector([0.1] * dimension), ti.Vector([5] * dimension), 0, 2, node, length, kinds, *mass, *velocity
    )
    for axis, field in enumerate(velocity):
        expected = np.full(field.shape, 0.37)
        for side in (0, -1):
            boundary = [slice(None)] * dimension
            boundary[axis] = side
            expected[tuple(boundary)] = 0.0
        np.testing.assert_allclose(field.to_numpy(), expected, rtol=1e-13, atol=1e-13)

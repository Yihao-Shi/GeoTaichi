"""Shifting corrects interior deficits without amplifying immovable errors."""

import numpy as np
import pytest
import taichi as ti

from src.mpm.engines.EngineKernel import (
    kernel_apply_particle_shifting_gradient,
    kernel_compute_particle_shifting_gradient_fdm_mac,
    particle_shifting_node_residual,
)

pytestmark = [pytest.mark.unit, pytest.mark.mpm, pytest.mark.cpu]


@pytest.mark.parametrize("dimension", [2, pytest.param(3, marks=pytest.mark.isolated_dimension(3))])
@pytest.mark.parametrize("errors", [(0.01, -0.01), (0.0, -0.01)])
def test_cauchy_shift_reduces_reachable_error_independently_of_remote_error(taichi_runtime, dimension, errors):
    n, ghost = 8, 1
    particle = ti.Struct.field(
        dict(
            active=ti.u8,
            materialID=ti.u8,
            bodyID=ti.u8,
            vol=float,
            x=ti.types.vector(dimension, float),
            v=ti.types.vector(dimension, float),
            grad_E2=ti.types.vector(dimension, float),
            fix_v=ti.types.vector(dimension, ti.u8),
            velocity_gradient=ti.types.matrix(dimension, dimension, float),
        ),
        shape=1,
    )
    particle.active.fill(1)
    particle.materialID.fill(1)
    particle.vol.fill(1)
    volume = ti.field(float, shape=(n**dimension, 1))
    reference = ti.field(float, shape=volume.shape)
    reference.fill(1)
    cell_type = ti.field(int, shape=(n - 1,) * dimension, offset=(-ghost,) * dimension)
    cell_type.fill(1)
    node_ids = ti.field(int, shape=2)
    node0 = sum(3 * n**d for d in range(dimension))
    node_ids.from_numpy(np.array([node0, node0 + 1], np.int32))
    gradient = ti.Vector.field(dimension, float, shape=2)
    jacobian = np.zeros((2, dimension))
    jacobian[:, 0] = [1, -1]
    gradient.from_numpy(jacobian)
    node_size = ti.field(ti.u8, shape=1)
    node_size.fill(2)
    cnum, spacing = ti.Vector([n - 1] * dimension), ti.Vector([1.0] * dimension)
    arguments = (2, 1, ghost, cnum, spacing, reference, volume, cell_type, particle, node_ids, gradient, node_size)
    shifts = []
    for remote_error in (0.0, 100.0):
        particle.x[0] = [2.5] * dimension
        volume.fill(1)
        volume[node0, 0], volume[node0 + 1, 0] = 1 + errors[0], 1 + errors[1]
        volume[0, 0] = 1 + remote_error
        den = kernel_compute_particle_shifting_gradient_fdm_mac(*arguments, False)
        curvature = kernel_compute_particle_shifting_gradient_fdm_mac(*arguments, True)
        kernel_apply_particle_shifting_gradient(
            1, ghost, cnum, spacing, 0.05, den / (2 * curvature), cell_type, particle
        )
        shift = particle.x.to_numpy()[0] - 2.5
        assert np.linalg.norm(np.asarray(errors) + jacobian @ shift) < np.linalg.norm(errors)
        shifts.append(shift)
    np.testing.assert_allclose(shifts[0], shifts[1], atol=1e-14)


@pytest.mark.parametrize("dimension", [2, pytest.param(3, marks=pytest.mark.isolated_dimension(3))])
def test_deficits_only_corrected_inside_fluid(taichi_runtime, dimension):
    n = 8
    volume = ti.field(float, shape=(n**dimension, 1))
    reference = ti.field(float, shape=volume.shape)
    reference.fill(1)
    types = ti.field(int, shape=(n - 1,) * dimension, offset=(-1,) * dimension)
    types.fill(1)
    node_id = sum(3 * n**d for d in range(dimension))
    result = ti.field(float, shape=())

    @ti.kernel
    def residual():
        result[None] = particle_shifting_node_residual(
            node_id, 0, 1, ti.Vector([n - 1] * dimension), types, volume, reference
        )

    residual()
    assert result[None] == -1
    types[(2,) * dimension] = 0
    residual()
    assert result[None] == 0
    types[(2,) * dimension] = 2
    residual()
    assert result[None] == -1

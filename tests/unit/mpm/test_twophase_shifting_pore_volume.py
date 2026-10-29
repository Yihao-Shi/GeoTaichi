import numpy as np
import pytest
import taichi as ti

from geotaichi import init


@pytest.mark.parametrize("dimension", [2, pytest.param(3, marks=pytest.mark.isolated_dimension(3))])
def test_saturated_porosity_interface_does_not_move_water(dimension):
    from src.mpm.engines.TwoPhaseDoubleLayerKernel import (
        kernel_volume_p2g_double_layer_fluid2d,
        kernel_volume_p2g_double_layer_fluid3d,
        kernel_delta_correct_double_layer_fluid2d,
        kernel_delta_correct_double_layer_fluid3d,
    )
    from src.mpm.structs.Particle import ParticleCloudTwoPhase, ParticleCloudTwoPhase2D

    init(dim=dimension, arch="cpu", default_fp="float64", cpu_max_num_threads=2, offline_cache=False, log=False)
    dx, count = 0.125, 4
    volume = dx**dimension
    node = ti.Struct.field({"vol": float, "porosity": float}, shape=((count + 1) ** dimension, 1))
    reference = ti.field(float, shape=node.shape)
    residual = ti.field(float, shape=node.shape)
    template = ParticleCloudTwoPhase2D if dimension == 2 else ParticleCloudTwoPhase
    particle = ti.Struct.field({**template.members, "grad_E2": ti.types.vector(dimension, float)}, shape=2)
    particle.active.fill(1)
    particle.materialID.fill(1)
    particle.phase.fill(2)
    targets = np.array(
        [sum(2 * (count + 1) ** d for d in range(dimension)), 1 + sum(2 * (count + 1) ** d for d in range(dimension))]
    )
    weights = np.array([[0.75, 0.25], [0.25, 0.75]])
    phi = np.array([0.4, 1.0])
    water = np.array([0.1, 1.3]) * volume
    particle_phi = weights @ phi
    for target, porosity in zip(targets, phi):
        node.porosity[int(target), 0] = porosity
        reference[int(target), 0] = volume
    particle.porosity.from_numpy(particle_phi)
    particle.vol.from_numpy(water / particle_phi)
    particle.mf.from_numpy(1000 * water)
    position = np.full((2, dimension), 2.25 * dx)
    position[1, 0] += 0.5 * dx
    particle.x.from_numpy(position)
    lnid = ti.field(int, shape=4)
    lnid.from_numpy(np.tile(targets, 2).astype(np.int32))
    shape = ti.field(float, shape=4)
    shape.from_numpy(weights.ravel())
    derivative = ti.Vector.field(dimension, float, shape=4)
    gradients = np.zeros((4, dimension))
    gradients[:, 0] = np.tile([-1 / dx, 1 / dx], 2)
    derivative.from_numpy(gradients)
    size = ti.field(int, shape=2)
    size.fill(2)
    cells = ti.field(int, shape=(count,) * dimension)
    cells.fill(1)
    scatter = kernel_volume_p2g_double_layer_fluid2d if dimension == 2 else kernel_volume_p2g_double_layer_fluid3d
    correct = kernel_delta_correct_double_layer_fluid2d if dimension == 2 else kernel_delta_correct_double_layer_fluid3d
    scatter(2, 2, node, particle, lnid, shape, size)
    np.testing.assert_allclose(node.vol.to_numpy()[targets, 0], phi * volume, rtol=1e-6)
    correct(
        2,
        2,
        ti.Vector([count * dx] * dimension),
        ti.Vector([dx] * dimension),
        ti.Vector([count] * dimension),
        1.0,
        cells,
        node,
        particle,
        lnid,
        derivative,
        size,
        reference,
        residual,
    )
    np.testing.assert_allclose(particle.x.to_numpy(), position, atol=1e-7)
    np.testing.assert_allclose(particle.mf.to_numpy(), 1000 * water)

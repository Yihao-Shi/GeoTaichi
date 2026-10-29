import numpy as np
import pytest
import taichi as ti

from geotaichi import init
from src.mpm.engines.TwoPhaseDoubleLayerKernel import (
    kernel_remap_double_layer_shift_velocity2d,
    kernel_remap_double_layer_shift_velocity3d,
)
from src.mpm.structs.Particle import ParticleCloudTwoPhase, ParticleCloudTwoPhase2D


@pytest.mark.parametrize("dimension", [2, pytest.param(3, marks=pytest.mark.isolated_dimension(3))])
@pytest.mark.parametrize("curved", [False, True])
def test_shift_remap_preserves_affine_fields_and_resamples_curved_fields(dimension, curved):
    init(dim=dimension, arch="cpu", default_fp="float64", cpu_max_num_threads=2, offline_cache=False, log=False)
    n, dx = 8, 0.125
    coefficients = np.eye(dimension) * 0.2 + 0.03
    constant = np.arange(dimension) * 0.04 + 0.1
    fields, coordinates, values = [], [], []
    for axis in range(dimension):
        shape = [n] * dimension
        shape[axis] += 1
        coord = np.moveaxis(np.indices(shape), 0, -1).astype(float)
        stagger = np.full(dimension, 0.5)
        stagger[axis] = 0
        coord = (coord + stagger) * dx
        value = constant[axis] + np.sum(coord * coefficients[axis], axis=-1)
        if curved:
            value += 0.2 * np.sin(2 * coord[..., 0]) + 0.1 * coord[..., -1] ** 2
        field = ti.field(float, shape=shape)
        field.from_numpy(value)
        fields.append(field)
        coordinates.append(coord)
        values.append(value)

    def sample(position):
        result = []
        for coord, value in zip(coordinates, values):
            r = np.abs((coord - position) / dx)
            weights = np.where(r < 0.5, 0.75 - r**2, np.where(r < 1.5, 0.5 * (1.5 - r) ** 2, 0)).prod(axis=-1)
            result.append(np.sum(weights * value) / weights.sum())
        return np.array(result)

    particle = (ParticleCloudTwoPhase2D if dimension == 2 else ParticleCloudTwoPhase).field(shape=2)
    particle.active.fill(1)
    particle.materialID.fill(1)
    particle.phase.from_numpy(np.array([2, 1], dtype=np.uint8))
    particle.mf.fill(0.25)
    lengths = ti.Vector.field(dimension, float, shape=1)
    lengths.fill(dx / 2)
    residual = ti.Vector.field(dimension, float, shape=2)
    remap = kernel_remap_double_layer_shift_velocity2d if dimension == 2 else kernel_remap_double_layer_shift_velocity3d
    old = np.arange(dimension) * 0.1 * dx + 3.2 * dx
    new = old + np.array([(-1) ** d * 0.05 * dx for d in range(dimension)])
    for flip_offset, fixed in ((0.0, False), (0.02, False), (0.02, True)):
        particle.x.from_numpy(np.stack([old, old]))
        before = sample(old) + flip_offset
        particle.vf.from_numpy(np.stack([before, before]))
        particle.v.from_numpy(np.stack([before, before]))
        particle.fluid_velocity_gradient.from_numpy(np.stack([np.eye(dimension) * 1000, coefficients]))
        flags = np.zeros((2, dimension), dtype=np.uint8)
        flags[0, 0] = fixed
        particle.fix_v.from_numpy(flags)
        remap(2, 1, 2, 3, 1, ti.Vector([dx] * dimension), particle, lengths, tuple(fields), residual)
        particle.x.from_numpy(np.stack([new, old]))
        remap(2, 0, 2, 3, 1, ti.Vector([dx] * dimension), particle, lengths, tuple(fields), residual)
        expected = sample(new) + flip_offset
        if fixed:
            expected[0] = before[0]
        np.testing.assert_allclose(particle.vf.to_numpy()[0], expected, atol=1e-7)
        np.testing.assert_allclose(particle.v.to_numpy()[0], expected, atol=1e-7)
        np.testing.assert_allclose(particle.vf.to_numpy()[1], before, atol=1e-7)
        np.testing.assert_allclose(particle.fluid_velocity_gradient.to_numpy()[1], coefficients, atol=1e-7)
        assert particle.mf[0] == particle.mf[1] == 0.25
        if not curved:
            np.testing.assert_allclose(particle.fluid_velocity_gradient.to_numpy()[0], coefficients, atol=1e-6)
        else:
            assert np.linalg.norm(particle.fluid_velocity_gradient.to_numpy()[0]) < 10

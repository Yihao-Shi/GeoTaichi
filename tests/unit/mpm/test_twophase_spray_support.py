"""Regression checks: unresolved spray must not invent pressure or lose free motion."""

import numpy as np
import pytest
import taichi as ti
from geotaichi import init
from src.mpm.engines.TwoPhaseDoubleLayerKernel import (
    kernel_classify_double_layer_fluid_cells2d,
    kernel_classify_double_layer_fluid_cells3d,
    kernel_extrapolate_double_layer_ghost_faces,
)
from src.mpm.structs.Particle import ParticleCloudTwoPhase, ParticleCloudTwoPhase2D


@pytest.mark.parametrize("dimension", [2, pytest.param(3, marks=pytest.mark.isolated_dimension(3))])
@pytest.mark.parametrize("attached", [False, True])
@pytest.mark.parametrize("porous", [False, True])
def test_pressure_support_distinguishes_isolated_spray_from_attached_surface(dimension, attached, porous):
    init(dim=dimension, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    shape = (6,) * dimension
    particle = (ParticleCloudTwoPhase2D if dimension == 2 else ParticleCloudTwoPhase).field(shape=1)
    particle.active[0] = 1
    particle.materialID[0] = 1
    particle.phase[0] = 2
    particle.x[0] = [0.025] * dimension
    mass, density, porosity = [ti.field(float, shape=shape) for _ in range(3)]
    types = ti.field(int, shape=shape)
    density.fill(1000.0)
    porosity.fill(1.0)
    if porous:
        porosity[(2,) * dimension] = 0.7
    mass[(2,) * dimension] = 0.04 * 1000 * 0.01**dimension
    if attached:
        mass[(3,) * dimension] = 0.9 * 1000 * 0.01**dimension
    classify = (
        kernel_classify_double_layer_fluid_cells2d if dimension == 2 else kernel_classify_double_layer_fluid_cells3d
    )
    classify(1, ti.Vector([0.01] * dimension), particle, mass, density, porosity, types)
    assert types[(2,) * dimension] == (1 if attached or porous else 0)
    if not attached and not porous:
        assert not np.any(types.to_numpy() == 1)
    assert particle.active[0] == 1 and particle.phase[0] == 2


@pytest.mark.parametrize("dimension", [2, pytest.param(3, marks=pytest.mark.isolated_dimension(3))])
def test_pressure_free_spray_preserves_uniform_tangential_motion(dimension):
    init(dim=dimension, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    shape = (6,) * dimension
    types = ti.field(int, shape=shape)
    array = np.zeros(shape, dtype=np.int32)
    array[..., 0] = 2
    types.from_numpy(array)
    velocity, acceleration, solid, mass = [], [], [], []
    for axis in range(dimension):
        face_shape = tuple(n + (axis == d) for d, n in enumerate(shape))
        for group, value in ((velocity, 1.7 if axis == 0 else 0.0), (acceleration, 0.0), (solid, 0.0), (mass, 1e-6)):
            field = ti.field(float, shape=face_shape)
            field.fill(value)
            group.append(field)
    dt = ti.field(float, shape=())
    dt[None] = 2e-5
    before = [v.to_numpy() for v in velocity]
    kernel_extrapolate_double_layer_ghost_faces(
        types, tuple(velocity), tuple(acceleration), tuple(solid), tuple(mass), 1e-12, 2, dt
    )
    for old, new in zip(before, velocity):
        np.testing.assert_array_equal(new.to_numpy(), old)

"""Wall support kernels must not read outside the mesh while searching donors."""

import numpy as np
import pytest
import taichi as ti
from src.mpm.engines.TwoPhaseDoubleLayerKernel import (
    kernel_extend_double_layer_wall_pressure,
    kernel_extend_double_layer_face_support_at_solid_walls,
)


@pytest.fixture
def checked_runtime():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, debug=True, cpu_max_num_threads=1, offline_cache=False)
    yield
    ti.reset()


@pytest.mark.parametrize("dimension", [2, 3])
def test_pressure_extension_checks_outer_mesh_bounds(checked_runtime, dimension):
    kind = ti.field(ti.i32, shape=(4,) * dimension)
    kind.fill(2)
    kind[(2,) * dimension] = 1
    pressure = ti.field(ti.f64, shape=kind.shape)
    pressure.fill(0.0)
    pressure[(2,) * dimension] = 1000.0
    kernel_extend_double_layer_wall_pressure(kind, pressure, 2)
    np.testing.assert_allclose(pressure.to_numpy(), 1000.0, atol=0, rtol=0)


@pytest.mark.parametrize("dimension", [2, 3])
def test_face_extension_checks_outer_mesh_bounds(checked_runtime, dimension):
    kind = ti.field(ti.i32, shape=(4,) * dimension)
    kind.fill(1)
    particle = ti.Struct.field(
        {
            "active": ti.i32,
            "materialID": ti.i32,
            "phase": ti.i32,
            "bodyID": ti.i32,
            "x": ti.types.vector(dimension, ti.f64),
            "vol": ti.f64,
            "porosity": ti.f64,
        },
        shape=1,
    )
    particle.active.fill(1)
    particle.materialID.fill(1)
    particle.phase.fill(1)
    particle.bodyID.fill(0)
    particle.x.fill(0.025)
    particle.vol.fill(0.1**dimension / 2**dimension)
    particle.porosity.fill(0.3)
    faces = tuple(ti.field(ti.f64, shape=tuple(4 + (d == a) for d in range(dimension))) for a in range(dimension))
    for f in faces:
        f.fill(1.0)
    lengths = ti.Vector.field(dimension, ti.f64, shape=1)
    lengths.fill(0.05)
    kernel_extend_double_layer_face_support_at_solid_walls(
        1, ti.Vector([0.1] * dimension), particle, kind, faces, lengths, 2, 3
    )
    for f in faces:
        np.testing.assert_array_equal(f.to_numpy(), 1.0)

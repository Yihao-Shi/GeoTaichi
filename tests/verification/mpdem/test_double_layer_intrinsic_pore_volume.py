"""Intrinsic fluid volume excludes missing interpolation support at material edges."""

import numpy as np
import pytest
import taichi as ti

from src.mpm.engines.TwoPhaseDoubleLayerKernel import (
    kernel_map_double_layer_phase_volume_support,
    kernel_update_double_layer_fluid_volume2d,
    kernel_update_double_layer_fluid_volume3d,
)


def fields(solid_bulk, fill, bodies=1):
    particle = ti.Struct.field(
        {
            "active": ti.i32,
            "materialID": ti.i32,
            "bodyID": ti.i32,
            "phase": ti.i32,
            "vol": ti.f64,
            "porosity": ti.f64,
            "mf": ti.f64,
            "m": ti.f64,
            "rad": ti.f64,
        },
        shape=2 * bodies,
    )
    node = ti.Struct.field({"porosity": ti.f64}, shape=(2, bodies))
    node.porosity.fill(0.99)
    support = ti.Vector.field(3, ti.f64, shape=(2, bodies))
    lnid = ti.field(ti.i32, shape=4 * bodies)
    shape = ti.field(ti.f64, shape=4 * bodies)
    size = ti.field(ti.i32, shape=2 * bodies)
    mapping = ti.field(ti.i32, shape=2 * bodies)
    lnid.from_numpy(np.tile([0, 1], 2 * bodies).astype(np.int32))
    shape.from_numpy(np.tile([0.25 * fill, 0.75 * fill], 2 * bodies))
    size.fill(2)
    mapping.from_numpy(np.arange(2 * bodies, dtype=np.int32))
    for body in range(bodies):
        for phase in (1, 2):
            p = 2 * body + phase - 1
            particle[p].active = int(phase == 2 or solid_bulk > 0)
            particle[p].materialID = 1
            particle[p].bodyID = body
            particle[p].phase = phase
            particle[p].vol = solid_bulk if phase == 1 else 1.0
            particle[p].porosity = 0.3 + 0.1 * body
    return particle, node, support, lnid, shape, size, mapping


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("fill", [1.0, 0.1])
@pytest.mark.parametrize("solid_bulk", [0.0, 0.5, 1.0])
def test_intrinsic_mass_and_volume_ignore_geometric_underfill(taichi_runtime, dimension, fill, solid_bulk):
    particle, node, support, lnid, shape, size, mapping = fields(solid_bulk, fill)
    update = kernel_update_double_layer_fluid_volume2d if dimension == 2 else kernel_update_double_layer_fluid_volume3d
    kernel_map_double_layer_phase_volume_support(2, 2, particle, lnid, shape, size, support)
    first = support.to_numpy()
    kernel_map_double_layer_phase_volume_support(2, 2, particle, lnid, shape, size, support)
    np.testing.assert_array_equal(support.to_numpy(), first)
    update(2, 0, 2, 1, 1000.0, node, particle, mapping, lnid, shape, size, support)
    expected_phi = 1.0 - 0.7 * solid_bulk
    assert float(particle[1].mf) == pytest.approx(1000.0 * expected_phi, rel=1e-13)
    assert float(particle[1].porosity) == pytest.approx(expected_phi, abs=1e-13)
    assert float(particle[1].vol) == 1.0
    assert float(particle[0].vol) == solid_bulk
    np.testing.assert_array_equal(node.porosity.to_numpy(), 0.99)

    particle[0].porosity = 0.4
    kernel_map_double_layer_phase_volume_support(2, 2, particle, lnid, shape, size, support)
    update(2, 0, 2, 0, 1000.0, node, particle, mapping, lnid, shape, size, support)
    assert float(particle[1].mf) == pytest.approx(1000.0 * expected_phi, rel=1e-13)
    assert float(particle[1].vol) == pytest.approx(expected_phi / (1.0 - 0.6 * solid_bulk), rel=1e-13)


@pytest.mark.parametrize("dimension", [2, 3])
def test_intrinsic_support_keeps_bodies_separate(taichi_runtime, dimension):
    particle, node, support, lnid, shape, size, mapping = fields(1.0, 0.1, bodies=2)
    kernel_map_double_layer_phase_volume_support(4, 2, particle, lnid, shape, size, support)
    update = kernel_update_double_layer_fluid_volume2d if dimension == 2 else kernel_update_double_layer_fluid_volume3d
    update(2, 0, 4, 1, 1000.0, node, particle, mapping, lnid, shape, size, support)
    assert float(particle[1].mf) == pytest.approx(300.0, abs=1e-10)
    assert float(particle[3].mf) == pytest.approx(400.0, abs=1e-10)

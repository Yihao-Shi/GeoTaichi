"""A uniform nonlinear drag exchange must conserve momentum in every direction."""

import numpy as np
import pytest
import taichi as ti

from src.mpm.engines.TwoPhaseDoubleLayerKernel import kernel_predict_double_layer2d, kernel_predict_double_layer3d
from src.mpm.structs.GridNode import NodeTwoPhase, NodeTwoPhase2D


@pytest.mark.parametrize("occupancy", [1.0, 0.125])
@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("drag_model", [0.0, 1.0, 2.0])
def test_uniform_drag_is_conservative_and_rotation_invariant(taichi_runtime, dimension, drag_model, occupancy):
    size, dx, phi, rho_s, rho_f, timestep = 8, 0.125, 0.25, 2650.0, 1000.0, 1e-5
    ms, mf = rho_s * (1.0 - phi) * dx**dimension, rho_f * phi * dx**dimension
    ms, mf = occupancy * ms, occupancy * mf
    geometric_porosity = 1.0 - occupancy * (1.0 - phi)
    node_type = NodeTwoPhase2D if dimension == 2 else NodeTwoPhase
    node = node_type.field(shape=((size + 1) ** dimension, 1))
    shapes = [tuple(size + (d == axis) for d in range(dimension)) for axis in range(dimension)]

    def field(shape, value=0.0):
        f = ti.field(ti.f64, shape=shape)
        f.fill(value)
        return f

    mass = [field(s, mf) for s in shapes]
    velocity, acceleration, solid_velocity = [[field(s) for s in shapes] for _ in range(3)]
    properties = [geometric_porosity, rho_s, rho_f, 0.001, 0.0005, 0.001, 9800.0, drag_model]
    face_properties = [[field(s, value) for s in shapes] for value in properties]
    node_properties = [field(node.shape, value) for value in properties[1:]]
    calLength = ti.Vector.field(dimension, ti.f64, shape=1)
    calLength.fill(0.5 * dx)
    dt = field((), timestep)
    initial_s = ti.Vector.field(dimension, ti.f64, shape=())
    initial_f = ti.Vector.field(dimension, ti.f64, shape=())

    @ti.kernel
    def initialize():
        for i, j in node:
            node[i, j].ms, node[i, j].mf = ms, mf
            node[i, j].porosity = geometric_porosity
            node[i, j].force = ti.Vector.zero(ti.f64, dimension)
            node[i, j].momentums, node[i, j].momentumf = initial_s[None], initial_f[None]

    solid_mass = tuple(field(shape, ms) for shape in shapes)
    predictor = kernel_predict_double_layer2d if dimension == 2 else kernel_predict_double_layer3d
    final_speeds = []
    for diagonal in [False, True]:
        relative = (
            np.full(dimension, 0.03 / np.sqrt(dimension)) if diagonal else np.array([0.03] + [0.0] * (dimension - 1))
        )
        vs, vf = -mf / (ms + mf) * relative, ms / (ms + mf) * relative
        initial_s[None], initial_f[None] = vs, vf
        initialize()
        for axis in range(dimension):
            velocity[axis].fill(vf[axis])
            solid_velocity[axis].fill(vs[axis])
        predictor(
            1e-12,
            ti.Vector([0.0] * 3),
            ti.Vector([dx] * dimension),
            dt,
            0.0,
            node,
            *mass,
            *velocity,
            *acceleration,
            *solid_velocity,
            *(f for group in face_properties for f in group),
            *node_properties,
            0,
            2,
            calLength,
            None,
            None,
            None,
            solid_mass,
        )
        center = size // 2
        node_id = sum(center * (size + 1) ** d for d in range(dimension))
        solid = np.array(node[node_id, 0].momentums)
        fluid = np.array([float(v[(center,) * dimension]) for v in velocity])
        np.testing.assert_allclose(ms * solid + mf * fluid, np.zeros(dimension), atol=1e-12)
        final_speeds.append(np.linalg.norm(fluid - solid))
    assert final_speeds[0] == pytest.approx(final_speeds[1], rel=1e-12, abs=1e-14)

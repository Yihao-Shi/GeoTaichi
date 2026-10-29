"""The production predictor must preserve a homogeneous forced steady slip."""

import numpy as np
import pytest
import taichi as ti

from src.mpm.engines.TwoPhaseDoubleLayerKernel import kernel_predict_double_layer2d, kernel_predict_double_layer3d
from src.mpm.structs.GridNode import NodeTwoPhase, NodeTwoPhase2D


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("shape_type", [0, 2])
def test_forced_predictor_preserves_coupled_momentum_and_slip(taichi_runtime, dimension, shape_type):
    size, dx, phi, rho_s, rho_f = 8, 0.125, 0.25, 2650.0, 1000.0
    permeability, unit_weight = 0.001, 9800.0
    ms, mf = rho_s * (1.0 - phi) * dx**dimension, rho_f * phi * dx**dimension
    beta = phi**2 * unit_weight / permeability
    a, b = beta / (rho_s * (1.0 - phi)), beta / (rho_f * phi)
    node_type = NodeTwoPhase2D if dimension == 2 else NodeTwoPhase
    node = node_type.field(shape=((size + 1) ** dimension, 1))
    shapes = [tuple(size + (d == axis) for d in range(dimension)) for axis in range(dimension)]

    def field(shape, value=0.0):
        out = ti.field(ti.f64, shape=shape)
        out.fill(value)
        return out

    mass = [field(shape, mf) for shape in shapes]
    velocity, acceleration, solid_velocity = [[field(shape) for shape in shapes] for _ in range(3)]
    properties = [phi, rho_s, rho_f, 0.001, 0.0005, permeability, unit_weight, 1.0]
    face_properties = [[field(shape, value) for shape in shapes] for value in properties]
    node_properties = [field(node.shape, value) for value in properties[1:]]
    calLength = ti.Vector.field(dimension, ti.f64, shape=1)
    calLength.fill(0.5 * dx)
    dt = field(())
    solid_pressure_factor = field(node.shape)
    fluid_mobility = tuple(field(shape) for shape in shapes)
    mixture_mobility = tuple(field(shape) for shape in shapes)

    @ti.kernel
    def initialize(vs: ti.f64, vf: ti.f64):
        for i, j in node:
            node[i, j].ms, node[i, j].mf = ms, mf
            node[i, j].porosity = phi
            node[i, j].force = ti.Vector.unit(dimension, dimension - 1) * ms
            node[i, j].momentums = ti.Vector.unit(dimension, dimension - 1) * vs
            node[i, j].momentumf = ti.Vector.unit(dimension, dimension - 1) * vf

    solid_mass = tuple(field(shape, ms) for shape in shapes)
    predictor = kernel_predict_double_layer2d if dimension == 2 else kernel_predict_double_layer3d
    for timestep in [1e-5, 5e-6, 1e-6]:
        for steady in [False, True]:
            slip = -1.0 / (a + b) if steady else 0.0
            vs, vf = -mf / (ms + mf) * slip, ms / (ms + mf) * slip
            initialize(vs, vf)
            # A below-cutoff deposit must not enter force/mass dynamics or leak force into FLIP.
            node.ms[0, 0], node.mf[0, 0] = 5e-13, 0.0
            node.forces[0, 0] = np.ones(dimension)
            inactive_velocity = np.array(node.momentums[0, 0])
            dt[None] = timestep
            for axis in range(dimension):
                velocity[axis].fill(vf if axis == dimension - 1 else 0.0)
                solid_velocity[axis].fill(vs if axis == dimension - 1 else 0.0)
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
                *(f for fields in face_properties for f in fields),
                *node_properties,
                shape_type,
                2 if shape_type == 0 else 3,
                calLength,
                solid_pressure_factor,
                fluid_mobility,
                mixture_mobility,
                solid_mass,
            )
            center = size // 2
            np.testing.assert_array_equal(np.array(node.momentums[0, 0]), inactive_velocity)
            np.testing.assert_array_equal(np.array(node.forces[0, 0]), np.zeros(dimension))
            node_id = sum(center * (size + 1) ** d for d in range(dimension))
            actual = np.array(
                [float(node[node_id, 0].momentums[dimension - 1]), float(velocity[-1][(center,) * dimension])]
            )
            matrix = np.array([[1.0 + timestep * a, -timestep * a], [-timestep * b, 1.0 + timestep * b]])
            expected = np.linalg.solve(matrix, [vs + timestep, vf])
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-14)
            mu_s, mu_f = np.linalg.solve(matrix, [1.0 / rho_s, 1.0 / rho_f])
            assert float(solid_pressure_factor[node_id, 0]) == pytest.approx(rho_s * mu_s, rel=1e-12)
            for axis in range(dimension):
                assert float(fluid_mobility[axis][(center,) * dimension]) == pytest.approx(mu_f, rel=1e-12)
                assert float(mixture_mobility[axis][(center,) * dimension]) == pytest.approx(
                    (1.0 - phi) * mu_s + phi * mu_f, rel=1e-12
                )
            assert abs(ms * actual[0] + mf * actual[1] - ms * timestep) < 1e-12
            if steady:
                assert abs(actual[1] - actual[0] - slip) < 1e-14


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("taichi_runtime", [{"fp": "f64"}], indirect=True)
def test_small_mass_support_preserves_solid_velocity_gradient(taichi_runtime, dimension, affine):
    from src.mpm.engines import TwoPhaseDoubleLayerKernel as kernels
    from src.mpm.structs.Particle import ParticleCloudTwoPhase, ParticleCloudTwoPhase2D
    from src.utils.ShapeFunctions import ShapeBsplineQ, GShapeBsplineQ

    dx, size, cutoff = 0.0025, 9, 3.6015625e-11
    particle = (ParticleCloudTwoPhase2D if dimension == 2 else ParticleCloudTwoPhase).field(shape=1)
    node = (NodeTwoPhase2D if dimension == 2 else NodeTwoPhase).field(shape=(size**dimension, 1))
    reference = ti.field(ti.f64, shape=node.shape)
    count = ti.field(ti.i32, shape=1)
    ids = ti.field(ti.i32, shape=3**dimension)
    weights = ti.field(ti.f64, shape=3**dimension)
    gradients = ti.Vector.field(dimension, ti.f64, shape=3**dimension)
    position = np.full(dimension, 3.31 * dx)
    position[-1] = (size - 1) * dx - 1e-8
    velocity = np.arange(1, dimension + 1, dtype=float)
    expected = np.arange(dimension**2, dtype=float).reshape(dimension, dimension) * float(affine)
    particle.active[0], particle.materialID[0], particle.phase[0] = 1, 1, 1
    particle.ms[0], particle.vol[0], particle.porosity[0] = 2.5e-5, dx**dimension, 0.36
    particle.x[0], particle.vs[0], particle.solid_velocity_gradient[0] = position, velocity, expected

    @ti.kernel
    def basis():
        base = ti.cast(ti.floor(particle[0].x / dx - 0.5), ti.i32)
        for offset in ti.grouped(ti.ndrange(*((3,) * dimension))):
            index = base + offset
            inside = True
            for axis in ti.static(range(dimension)):
                inside = inside and 0 <= index[axis] < size
            if inside:
                q = ti.atomic_add(count[0], 1)
                w = ti.Vector.zero(ti.f64, dimension)
                g = ti.Vector.zero(ti.f64, dimension)
                for axis in ti.static(range(dimension)):
                    ids[q] += index[axis] * size**axis
                    boundary = kernels._mac_bspline_boundary_type(index[axis], size)
                    w[axis] = ShapeBsplineQ(particle[0].x[axis], index[axis] * dx, 1 / dx, boundary)
                    g[axis] = GShapeBsplineQ(particle[0].x[axis], index[axis] * dx, 1 / dx, boundary)
                weight = 1.0
                for axis in ti.static(range(dimension)):
                    weight *= w[axis]
                weights[q] = weight
                for axis in ti.static(range(dimension)):
                    value = g[axis]
                    for other in ti.static(range(dimension)):
                        if ti.static(other != axis):
                            value *= w[other]
                    gradients[q][axis] = value

    @ti.kernel
    def gradient() -> ti.types.matrix(dimension, dimension, ti.f64):
        result = ti.Matrix.zero(ti.f64, dimension, dimension)
        for q in range(count[0]):
            result += node[ids[q], 0].momentums.outer_product(gradients[q])
        return result

    basis()
    p2g = getattr(kernels, f"kernel_p2g_double_layer_mass{dimension}d")
    normalize = getattr(kernels, f"kernel_normalize_double_layer_nodes{dimension}d")
    p2g(
        3**dimension,
        1,
        ti.Vector([dx] * dimension),
        ti.Vector([size] * dimension),
        1,
        node,
        particle,
        ids,
        weights,
        count,
    )
    masses = node.ms.to_numpy()[:, 0]
    assert np.any((masses > 0) & (masses <= cutoff))
    assert np.any(masses > cutoff)
    normalize(cutoff, node, reference)
    np.testing.assert_allclose(np.array(gradient()), expected, rtol=1e-10, atol=1e-9)
    support = ids.to_numpy()[: count[0]]
    pic = weights.to_numpy()[: count[0]] @ node.momentums.to_numpy()[support, 0]
    np.testing.assert_allclose(pic, velocity, rtol=1e-12, atol=1e-12)
    assert np.isfinite(node.momentums.to_numpy()).all()

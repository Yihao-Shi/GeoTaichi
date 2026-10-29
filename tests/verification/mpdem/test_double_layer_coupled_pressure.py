"""Pressure assembly and both correctors must use the frozen drag inverse."""

import itertools

import numpy as np
import pytest
import taichi as ti

from src.mpm.engines import TwoPhaseDoubleLayerKernel as kernels
from src.mpm.structs.GridNode import NodeTwoPhase, NodeTwoPhase2D


@pytest.mark.parametrize("dimension", [2, 3])
def test_coupled_pressure_operator_and_phase_correctors(taichi_runtime, dimension):
    size, dx, timestep, phi, rho_s, rho_f = 6, 0.125, 0.1, 0.25, 2650.0, 1000.0
    beta = phi**2 * 9800.0 / 0.001
    a, b = beta / ((1.0 - phi) * rho_s), beta / (phi * rho_f)
    matrix = np.array([[1.0 + timestep * a, -timestep * a], [-timestep * b, 1.0 + timestep * b]])
    mu_s, mu_f = np.linalg.solve(matrix, [1.0 / rho_s, 1.0 / rho_f])
    mixture_mu = (1.0 - phi) * mu_s + phi * mu_f
    shape = (size,) * dimension
    face_shapes = [tuple(size + (d == axis) for d in range(dimension)) for axis in range(dimension)]

    def field(shape, value=0.0, dtype=ti.f64):
        f = ti.field(dtype, shape=shape)
        f.fill(value)
        return f

    kind, pressure, sdf = field(shape, 1, ti.i32), field(shape), field(shape, -1.0)
    gradient = np.arange(1.0, dimension + 1.0) * 173.0
    coordinates = np.indices(shape)
    pressure.from_numpy(sum(gradient[d] * dx * (coordinates[d] + 0.5) for d in range(dimension)))
    porosity = [field(s, phi) for s in face_shapes]
    mass = [field(s, 1.0) for s in face_shapes]
    velocity, acceleration = [[field(s) for s in face_shapes] for _ in range(2)]
    density = [field(s, rho_f) for s in face_shapes]
    mobility = tuple(field(s, mu_f) for s in face_shapes)
    mixture_faces = tuple(field(s, mixture_mu) for s in face_shapes)
    cell_mobility = field(shape)
    kernels.kernel_average_double_layer_pressure_mobility(mixture_faces, cell_mobility)
    np.testing.assert_allclose(cell_mobility.to_numpy(), mixture_mu, rtol=1e-13)
    dt = field((), timestep)
    solid_density, fluid_density = field(shape, rho_s), field(shape, rho_f)
    diag = field(shape)
    offdiag = ti.Vector.field(dimension, ti.f64, shape=shape)
    assembly = (
        kernels.kernel_assemble_double_layer_pressure_A
        if dimension == 2
        else kernels.kernel_assemble_double_layer_pressure_A3d
    )
    assembly(
        ti.Vector([dx] * dimension),
        dt,
        kind,
        *porosity,
        solid_density,
        fluid_density,
        sdf,
        diag,
        offdiag,
        cell_mobility,
    )
    center = (size // 2,) * dimension
    scale = timestep * mixture_mu / dx**2
    assert float(diag[center]) == pytest.approx(2.0 * dimension * scale + 1e-12, rel=1e-12)
    np.testing.assert_allclose(np.array(offdiag[center]), -scale, rtol=1e-12)
    coarse_kind = field((3,) * dimension, 1, ti.i32)
    coarse_diag = field((3,) * dimension)
    coarse_offdiag = ti.Vector.field(dimension, ti.f64, shape=(3,) * dimension)
    assembly_mg = (
        kernels.kernel_assemble_double_layer_pressure_mg_A
        if dimension == 2
        else kernels.kernel_assemble_double_layer_pressure_mg_A3d
    )
    assembly_mg(
        ti.Vector([dx] * dimension),
        dt,
        2,
        coarse_kind,
        *porosity,
        solid_density,
        fluid_density,
        sdf,
        coarse_diag,
        coarse_offdiag,
        cell_mobility,
    )
    assert float(coarse_diag[(1,) * dimension]) == pytest.approx(2.0 * dimension * scale / 4.0 + 1e-12, rel=1e-12)
    correction = (
        kernels.kernel_correct_double_layer_velocity2d
        if dimension == 2
        else kernels.kernel_correct_double_layer_velocity3d
    )
    correction(
        1e-12, ti.Vector([dx] * dimension), dt, kind, pressure, sdf, *mass, *velocity, *acceleration, *density, mobility
    )
    for d in range(dimension):
        assert float(velocity[d][center]) == pytest.approx(-timestep * mu_f * gradient[d], rel=1e-12)

    count = 2**dimension
    node_type = NodeTwoPhase2D if dimension == 2 else NodeTwoPhase
    node = node_type.field(shape=(count, 1))
    particle_type = ti.types.struct(
        active=ti.i32, materialID=ti.i32, phase=ti.i32, bodyID=ti.i32, vol=ti.f64, porosity=ti.f64
    )
    particle = particle_type.field(shape=1)
    particle[0].active, particle[0].materialID, particle[0].phase = 1, 1, 1
    particle[0].vol, particle[0].porosity = 1.0, phi
    mapping, ids = field(1, 0, ti.i32), field(count, 0, ti.i32)
    ids.from_numpy(np.arange(count, dtype=np.int32))
    weights, node_size = field(count, 1.0 / count), field(1, count, ti.i32)
    derivatives = ti.Vector.field(dimension, ti.f64, shape=count)
    corners = np.array(list(itertools.product([0, 1], repeat=dimension)))
    derivatives.from_numpy((2.0 * corners - 1.0) / (2 ** (dimension - 1) * dx))
    nodal_pressure = corners @ (gradient * dx)
    for i in range(count):
        node[i, 0].ms = rho_s * (1.0 - phi) / count
        node[i, 0].pressure = nodal_pressure[i]
    factor = field(node.shape, rho_s * mu_s)
    solid_correction = (
        kernels.kernel_correct_double_layer_solid_velocity_paper2d
        if dimension == 2
        else kernels.kernel_correct_double_layer_solid_velocity_paper3d
    )
    solid_correction(count, 0, 1, 1e-12, dt, node, particle, mapping, ids, weights, derivatives, node_size, factor)
    np.testing.assert_allclose(
        node.momentums.to_numpy()[:, 0], np.tile(-timestep * mu_s * gradient, (count, 1)), rtol=1e-12, atol=1e-13
    )

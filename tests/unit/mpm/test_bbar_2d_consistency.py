import numpy as np
import pytest
import taichi as ti

from src.mpm.engines.EngineKernel import (
    bbar_internal_force_2d,
    bbar_velocity_gradient_2d,
    kernel_update_displacement_gradient_bbar_2D,
    kernel_update_velocity_gradient_bbar_twophase_2D,
    kernel_update_velocity_gradient_bbar_2D,
)
from src.physics_model.consititutive_model.infinitesimal_strain.InfinitesimalStrainModel import (
    InfinitesimalStrainModel,
)
from src.utils.TypeDefination import mat2x2, real, vec2f, vec4f, vec6f


pytestmark = [
    pytest.mark.unit,
    pytest.mark.mpm,
    pytest.mark.materials,
    pytest.mark.assembly,
    pytest.mark.cpu,
]


@ti.dataclass
class BBarNodeFixture:
    momentum: vec2f
    displacement: vec2f


@ti.dataclass
class BBarParticleFixture:
    active: ti.u8
    bodyID: ti.u8
    velocity_gradient: mat2x2
    vol: real
    vol0: real


@ti.dataclass
class BBarTwoPhaseNodeFixture:
    momentums: vec2f
    momentumf: vec2f


@ti.dataclass
class BBarTwoPhaseParticleFixture:
    active: ti.u8
    bodyID: ti.u8
    porosity: real
    vol: real
    ms: real
    mf: real
    m: real
    solid_velocity_gradient: mat2x2
    fluid_velocity_gradient: mat2x2


@ti.kernel
def evaluate_bbar_pair_2d(
    grid_velocity: ti.types.vector(2, real),
    dshape_fn: ti.types.vector(2, real),
    dshape_fnc: ti.types.vector(2, real),
    stress: ti.types.vector(4, real),
) -> ti.types.vector(4, real):
    internal_stress = vec6f(
        stress[0], stress[1], stress[2], stress[3], 0.0, 0.0
    )
    internal_force = bbar_internal_force_2d(
        dshape_fn, dshape_fnc, internal_stress
    )
    velocity_gradient = bbar_velocity_gradient_2d(
        grid_velocity, dshape_fn, dshape_fnc
    )
    nodal_power = grid_velocity.dot(internal_force)
    stress_power = (
        stress[0] * velocity_gradient[0, 0]
        + stress[1] * velocity_gradient[1, 1]
        + stress[3]
        * (velocity_gradient[0, 1] + velocity_gradient[1, 0])
    )
    return vec4f(
        internal_force[0], internal_force[1], nodal_power, stress_power
    )


def projected_gradient_2d(values, gradients, center_gradients):
    result = np.zeros((2, 2), dtype=np.float64)
    for value, gradient, center_gradient in zip(
        values, gradients, center_gradients
    ):
        correction = 0.5 * (center_gradient - gradient)
        result += np.outer(value, gradient)
        result += np.eye(2) * np.dot(correction, value)
    return result


def test_bbar_plane_strain_force_is_adjoint_and_ignores_sigma_zz(
    taichi_runtime,
):
    grid_velocity = ti.Vector([1.2, -0.7])
    dshape_fn = ti.Vector([0.4, -0.2])
    dshape_fnc = ti.Vector([0.1, 0.3])
    in_plane_stress = [11.0, -4.0, 0.0, 2.5]

    without_sigma_zz = np.asarray(
        evaluate_bbar_pair_2d(
            grid_velocity,
            dshape_fn,
            dshape_fnc,
            ti.Vector(in_plane_stress),
        )
    )
    in_plane_stress[2] = 1.0e6
    with_sigma_zz = np.asarray(
        evaluate_bbar_pair_2d(
            grid_velocity,
            dshape_fn,
            dshape_fnc,
            ti.Vector(in_plane_stress),
        )
    )

    np.testing.assert_allclose(
        with_sigma_zz[:2], without_sigma_zz[:2], rtol=0.0, atol=1.0e-13
    )
    assert without_sigma_zz[2] == pytest.approx(
        without_sigma_zz[3], rel=1.0e-13, abs=1.0e-13
    )
    assert with_sigma_zz[2] == pytest.approx(
        with_sigma_zz[3], rel=1.0e-13, abs=1.0e-13
    )


def test_bbar_plane_strain_volume_uses_projected_explicit_and_implicit_gradient(
    taichi_runtime,
):
    node = BBarNodeFixture.field(shape=(2, 1))
    particle = BBarParticleFixture.field(shape=1)
    material_id = ti.field(dtype=ti.i32, shape=1)
    state_vars = ti.field(dtype=real, shape=1)
    lnid = ti.field(dtype=ti.i32, shape=2)
    dshape_fn = ti.Vector.field(2, dtype=real, shape=2)
    dshape_fnc = ti.Vector.field(2, dtype=real, shape=2)
    node_size = ti.field(dtype=ti.i32, shape=1)
    dt = ti.field(dtype=real, shape=())
    material = InfinitesimalStrainModel(
        material_type="Solid", configuration="ULMPM", solver_type="Explicit"
    )

    gradients = np.array([[0.4, -0.2], [-0.1, 0.5]])
    center_gradients = np.array([[0.1, 0.3], [0.25, -0.15]])
    velocities = np.array([[1.2, -0.7], [-0.4, 0.9]])
    displacements = np.array([[0.08, -0.03], [-0.02, 0.05]])
    dshape_fn.from_numpy(gradients)
    dshape_fnc.from_numpy(center_gradients)
    lnid.from_numpy(np.array([0, 1], dtype=np.int32))
    node_size[0] = 2
    material_id[0] = 0
    particle[0].active = 1
    particle[0].bodyID = 0

    initial_volume = 2.3
    dt[None] = 0.2
    for node_id in range(2):
        node[node_id, 0].momentum = velocities[node_id]
        node[node_id, 0].displacement = displacements[node_id]
    particle[0].vol = initial_volume
    particle[0].vol0 = initial_volume

    kernel_update_velocity_gradient_bbar_2D(
        2,
        0,
        1,
        dt,
        node,
        particle,
        material_id,
        material,
        state_vars,
        lnid,
        dshape_fn,
        dshape_fnc,
        node_size,
    )
    explicit_gradient = projected_gradient_2d(
        velocities, gradients, center_gradients
    )
    explicit_volume = initial_volume * np.linalg.det(
        np.eye(2) + float(dt[None]) * explicit_gradient
    )
    np.testing.assert_allclose(
        particle.velocity_gradient.to_numpy()[0],
        explicit_gradient,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert float(particle[0].vol) == pytest.approx(
        explicit_volume, rel=1.0e-13, abs=1.0e-13
    )

    # Deliberately leave unrelated momentum values in place: the implicit
    # path must reconstruct its gradient from nodal displacement.
    particle[0].vol = -1.0
    particle[0].vol0 = initial_volume
    kernel_update_displacement_gradient_bbar_2D(
        2,
        0,
        1,
        dt,
        node,
        particle,
        material_id,
        material,
        state_vars,
        lnid,
        dshape_fn,
        dshape_fnc,
        node_size,
    )
    displacement_gradient = projected_gradient_2d(
        displacements, gradients, center_gradients
    )
    implicit_gradient = displacement_gradient / float(dt[None])
    implicit_volume = initial_volume * np.linalg.det(
        np.eye(2) + displacement_gradient
    )
    np.testing.assert_allclose(
        particle.velocity_gradient.to_numpy()[0],
        implicit_gradient,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert float(particle[0].vol) == pytest.approx(
        implicit_volume, rel=1.0e-13, abs=1.0e-13
    )


def test_twophase_bbar_uses_the_same_projected_solid_gradient(
    taichi_runtime,
):
    node = BBarTwoPhaseNodeFixture.field(shape=(2, 1))
    particle = BBarTwoPhaseParticleFixture.field(shape=1)
    material_id = ti.field(dtype=ti.i32, shape=1)
    state_vars = ti.field(dtype=real, shape=1)
    lnid = ti.field(dtype=ti.i32, shape=2)
    dshape_fn = ti.Vector.field(2, dtype=real, shape=2)
    dshape_fnc = ti.Vector.field(2, dtype=real, shape=2)
    node_size = ti.field(dtype=ti.i32, shape=1)
    dt = ti.field(dtype=real, shape=())
    material = InfinitesimalStrainModel(
        material_type="TwoPhaseSingleLayer",
        configuration="ULMPM",
        solver_type="Explicit",
    )
    material.fluid_density = 1000.0

    gradients = np.array([[0.4, -0.2], [-0.1, 0.5]])
    center_gradients = np.array([[0.1, 0.3], [0.25, -0.15]])
    solid_velocities = np.array([[1.2, -0.7], [-0.4, 0.9]])
    fluid_velocities = np.array([[0.3, -0.1], [0.2, 0.4]])
    dshape_fn.from_numpy(gradients)
    dshape_fnc.from_numpy(center_gradients)
    lnid.from_numpy(np.array([0, 1], dtype=np.int32))
    node_size[0] = 2
    material_id[0] = 0
    dt[None] = 0.2
    for node_id in range(2):
        node[node_id, 0].momentums = solid_velocities[node_id]
        node[node_id, 0].momentumf = fluid_velocities[node_id]

    initial_volume = 2.3
    initial_porosity = 0.35
    solid_mass = 1.2
    particle[0].active = 1
    particle[0].bodyID = 0
    particle[0].vol = initial_volume
    particle[0].porosity = initial_porosity
    particle[0].ms = solid_mass

    kernel_update_velocity_gradient_bbar_twophase_2D(
        2,
        0,
        1,
        dt,
        node,
        particle,
        material_id,
        material,
        state_vars,
        lnid,
        dshape_fn,
        dshape_fnc,
        node_size,
    )

    solid_gradient = projected_gradient_2d(
        solid_velocities, gradients, center_gradients
    )
    fluid_gradient = np.zeros((2, 2), dtype=np.float64)
    fluid_gradient[0, 0] = np.sum(
        center_gradients[:, 0] * fluid_velocities[:, 0]
    )
    fluid_gradient[1, 1] = np.sum(
        center_gradients[:, 1] * fluid_velocities[:, 1]
    )
    volume_ratio = np.linalg.det(
        np.eye(2) + float(dt[None]) * solid_gradient
    )
    expected_volume = initial_volume * volume_ratio
    expected_porosity = 1.0 - (1.0 - initial_porosity) / volume_ratio
    expected_fluid_mass = (
        expected_volume * expected_porosity * material.fluid_density
    )

    np.testing.assert_allclose(
        particle.solid_velocity_gradient.to_numpy()[0],
        solid_gradient,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        particle.fluid_velocity_gradient.to_numpy()[0],
        fluid_gradient,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert float(particle[0].vol) == pytest.approx(
        expected_volume, rel=1.0e-13, abs=1.0e-13
    )
    assert float(particle[0].porosity) == pytest.approx(
        expected_porosity, rel=1.0e-13, abs=1.0e-13
    )
    assert float(particle[0].mf) == pytest.approx(
        expected_fluid_mass, rel=1.0e-13, abs=1.0e-13
    )
    assert float(particle[0].m) == pytest.approx(
        solid_mass + expected_fluid_mass, rel=1.0e-13, abs=1.0e-13
    )

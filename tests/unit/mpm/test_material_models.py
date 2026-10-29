"""Exercise supported Native and Direct explicit materials through their factories."""

import numpy as np
import pytest
import taichi as ti

from tests.helpers.material_cases import (
    DIRECT_EXPLICIT_MPM,
    FINITE_ELASTIC,
    FLUIDS,
    NATIVE_MPM,
    initial_stress,
    material_parameters,
)

pytestmark = [pytest.mark.materials, pytest.mark.cpu]


@pytest.mark.parametrize("dimension", [pytest.param(d, marks=pytest.mark.isolated_dimension(d)) for d in (2, 3)])
@pytest.mark.parametrize(
    "backend,name",
    [
        *[("Native", name) for name in NATIVE_MPM],
        *[("Direct", name) for name in DIRECT_EXPLICIT_MPM],
    ],
)
def test_material_state_and_stress_update(taichi_runtime, dimension, backend, name):
    from src.mpm.MaterialManager import MaterialHandle
    from src.mpm.Simulation import Simulation

    sims = Simulation()
    sims.dimension, sims.max_particle_num = dimension, 2
    sims.configuration, sims.solver_type = "ULMPM", "Explicit"
    sims.material_type = "Fluid" if name in FLUIDS else "Solid"
    sims.stress_integration = "ReturnMapping"
    if backend == "Native":
        materials = MaterialHandle(sims)
        materials.setup(sims, None, name, {"MaterialID": 1, **material_parameters(name)})
        materials.activate_state_variables(sims)
        model, state = materials.matProps[1], materials.stateVars
        schema = materials.stateDict
    else:
        from src.mpm.engines.direct.ExplicitMPM import ExplicitMPM

        solver = ExplicitMPM.__new__(ExplicitMPM)
        solver.configuration = "UL"
        solver.handle_material(material=name, material_parameters=material_parameters(name))
        model = solver.material
        schema = model.get_state_vars()
        if "deformation_gradient" in schema:
            schema["deformation_gradient"] = ti.types.matrix(dimension, dimension, ti.f64)
        state = ti.Struct.field(schema, shape=2)
    particles = ti.Struct.field({"stress": ti.types.vector(6, ti.f64)}, shape=2)
    stress0 = np.asarray(initial_stress(name))
    particles.stress.from_numpy(np.tile(stress0, (2, 1)))
    dt = ti.field(ti.f64, shape=())
    dt[None] = 1e-3
    volumes = ti.field(ti.f64, shape=2)
    volumes.fill(1.0)
    pk1_3d = ti.Matrix.field(3, 3, ti.f64, shape=())
    pk1_2d = ti.Matrix.field(2, 2, ti.f64, shape=())

    @ti.kernel
    def initialize():
        for point in range(2):
            model._initialize_vars(point, particles, state)

    @ti.kernel
    def update(point: int, magnitude: ti.f64):
        gradient = ti.Matrix.zero(ti.f64, dimension, dimension)
        for axis in ti.static(range(dimension)):
            gradient[axis, axis] = -0.001 * magnitude
        gradient[0, 1] = 0.01 * magnitude
        if ti.static(dimension == 2):
            volumes[point] *= model.update_particle_volume_2D(point, gradient, state, dt)
            particles[point].stress = model.ComputeStress2D(point, particles[point].stress, gradient, state, dt)
        else:
            volumes[point] *= model.update_particle_volume(point, gradient, state, dt)
            particles[point].stress = model.ComputeStress(point, particles[point].stress, gradient, state, dt)
        if ti.static(name in FINITE_ELASTIC and dimension == 2):
            deformation = ti.Matrix.identity(ti.f64, 3)
            for i, j in ti.static(ti.ndrange(2, 2)):
                deformation[i, j] = state[point].deformation_gradient[i, j]
            pk1_3d[None] = model.first_piola_stress(deformation)
            pk1_2d[None] = model.ComputePKStress2D(point, ti.Matrix.zero(ti.f64, 2, 2), gradient, state, dt)

    initialize()
    assert model.density == material_parameters(name)["Density"]
    assert set(state.to_numpy()) == set(schema)
    update(1, 0.0)
    np.testing.assert_allclose(particles.stress.to_numpy()[1], stress0, rtol=1e-10, atol=1e-8)
    untouched = {key: value[1].copy() for key, value in state.to_numpy().items()}
    update(0, 1.0)
    first = particles.stress.to_numpy()[0].copy()
    update(0, 1.0)
    stress = particles.stress.to_numpy()
    assert np.isfinite(stress).all()
    assert np.linalg.norm(first - stress0) > 1e-8
    assert np.linalg.norm(stress[0] - first) > 1e-8
    np.testing.assert_allclose(stress[1], stress0, rtol=1e-10, atol=1e-8)
    assert 0.0 < volumes[0] < 1.0
    assert volumes[1] == 1.0
    if name in FINITE_ELASTIC and dimension == 2:
        deformation = np.eye(3)
        deformation[:2, :2] = state.deformation_gradient.to_numpy()[0]
        cauchy = pk1_3d.to_numpy() @ deformation.T / np.linalg.det(deformation)
        expected = cauchy[[0, 1, 2, 0, 1, 0], [0, 1, 2, 1, 2, 2]]
        np.testing.assert_allclose(stress[0], expected, rtol=1e-9, atol=1e-8)
        np.testing.assert_allclose(pk1_2d.to_numpy(), pk1_3d.to_numpy()[:2, :2], rtol=1e-9, atol=1e-8)
    for key, value in state.to_numpy().items():
        assert np.isfinite(value).all(), key
        np.testing.assert_array_equal(value[1], untouched[key], err_msg=key)

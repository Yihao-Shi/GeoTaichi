"""Supported Direct implicit and soft-particle material entry points."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from tests.helpers.material_cases import FINITE_ELASTIC, material_parameters

pytestmark = [pytest.mark.materials, pytest.mark.cpu, pytest.mark.isolated_dimension(3)]


@pytest.mark.parametrize("name", FINITE_ELASTIC)
def test_soft_material_manager_derivatives_and_commit(taichi_runtime, name):
    from src.mpm.MaterialManager import SoftParticleMaterialManager

    soft = ti.Struct.field({"materialID": int}, shape=1)
    soft.materialID.fill(1)
    scene = SimpleNamespace(
        softNum=[1],
        soft=soft,
        material_parameters={1: {"ConstitutiveModel": name, **material_parameters(name)}},
    )
    manager = SoftParticleMaterialManager()
    manager.setup(scene, SimpleNamespace(max_material_num=2, max_material_point_num=1))
    model, state = manager.matProps, manager.stateVars
    energy = ti.field(ti.f64, shape=())
    stress = ti.Vector.field(9, ti.f64, shape=())
    tangent = ti.Matrix.field(9, 9, ti.f64, shape=())
    committed = ti.Matrix.field(3, 3, ti.f64, shape=())
    committed_stress = ti.Matrix.field(3, 3, ti.f64, shape=())

    @ti.kernel
    def evaluate(values: ti.types.ndarray(dtype=ti.f64, ndim=2), commit: int):
        deformation = ti.Matrix.zero(ti.f64, 3, 3)
        for i, j in ti.static(ti.ndrange(3, 3)):
            deformation[i, j] = values[i, j]
        energy[None] = model.Psi(1, deformation)
        stress[None] = model.dPsi_div_dF(1, deformation)
        tangent[None] = model.d2Psi_div_d2F(1, deformation)
        if commit:
            committed[None], committed_stress[None] = model.commit_soft_particle_state(0, 1, deformation, state)

    evaluate(np.eye(3), 0)
    assert abs(energy[None]) < 1e-8
    np.testing.assert_allclose(stress.to_numpy(), 0, atol=1e-8)
    deformation = np.array([[1.08, 0.03, 0], [0, 0.96, 0.02], [0, 0, 1.02]])
    direction = np.arange(1, 10, dtype=float).reshape(3, 3) / 9
    evaluate(deformation, 0)
    reference_stress, reference_tangent = stress.to_numpy().copy(), tangent.to_numpy().copy()
    epsilon = 1e-6
    evaluate(deformation + epsilon * direction, 0)
    plus_energy, plus_stress = energy[None], stress.to_numpy().copy()
    evaluate(deformation - epsilon * direction, 0)
    assert (plus_energy - energy[None]) / (2 * epsilon) == pytest.approx(
        reference_stress @ direction.flatten(order="F"),
        rel=1e-7,
        abs=1e-4,
    )
    np.testing.assert_allclose(
        (plus_stress - stress.to_numpy()) / (2 * epsilon),
        reference_tangent @ direction.flatten(order="F"),
        rtol=1e-7,
        atol=1e-4,
    )
    evaluate(deformation, 1)
    np.testing.assert_allclose(committed.to_numpy(), deformation)
    np.testing.assert_allclose(committed_stress.to_numpy().flatten(order="F"), reference_stress)
    assert np.isfinite(state.estress[0]) and state.estress[0] > 0


@pytest.mark.parametrize(
    "name",
    [
        "NeoHookean",
        "DruckerPrager",
        "VonMises",
        "ModifiedCamClay",
        "StateDependentDruckerPrager",
    ],
)
def test_direct_implicit_material_response_and_history(taichi_runtime, tmp_path, name):
    from src.mpm.engines.direct.ImplicitULMPM import ImplicitULMPM
    from src.mpm.generator.Body import Body

    parameters = material_parameters("StateDependentMohrCoulomb" if name == "StateDependentDruckerPrager" else name)
    if name == "VonMises":
        parameters.update(YieldStress=1200.0, HardeningModulus=500.0)
    if name == "ModifiedCamClay":
        parameters.update(pc0=2e5, OverConsolidationRatio=2.0)
    bodies = Body()
    bodies.add_particles(np.array([[0.45, 0.45, 0.45]]), volume=1e-3, init_v=[0.0] * 3, xmin=[0.0] * 3, xmax=[1.0] * 3)
    solver = ImplicitULMPM(
        bodies=bodies,
        material=name,
        material_parameters=parameters,
        domain=[1.0] * 3,
        dx=0.25,
        dt=1e-3,
        density=1800.0,
        young_modulus=2e5,
        poisson_ratio=0.3,
        gravity=[0.0] * 3,
        shape_function="linear",
        step=1,
        interval=1,
        visualize=False,
        path=str(tmp_path),
    )
    model = solver.material
    plastic = solver.is_finite_strain_plastic
    stress = ti.Matrix.field(3, 3, ti.f64, shape=())
    tangent = ti.Matrix.field(9, 9, ti.f64, shape=())

    @ti.kernel
    def evaluate(values: ti.types.ndarray(dtype=ti.f64, ndim=2), commit: int):
        deformation = ti.Matrix.zero(ti.f64, 3, 3)
        for i, j in ti.static(ti.ndrange(3, 3)):
            deformation[i, j] = values[i, j]
        if ti.static(plastic):
            stress[None] = model.total_first_piola_stress_at(0, deformation)
            tangent[None] = model.total_first_piola_tangent_at(0, deformation)
            if commit:
                ignored = model.commit_total_state(0, deformation)
        else:
            stress[None] = model.first_piola_stress(deformation)
            tangent[None] = model.first_piola_tangent(deformation)

    deformation = np.array([[0.98, 0.06, 0], [0, 0.99, 0.01], [0, 0, 1.01]])
    direction = np.arange(1, 10, dtype=float).reshape(3, 3) / 9
    if solver.has_lagged_material:

        @ti.kernel
        def refresh(values: ti.types.ndarray(dtype=ti.f64, ndim=2)) -> ti.f64:
            total = ti.Matrix.zero(ti.f64, 3, 3)
            for i, j in ti.static(ti.ndrange(3, 3)):
                total[i, j] = values[i, j]
            return model.refresh_lagged_incremental_potential(0, total)

        solver.begin_lagged_material_state()
        for _ in range(solver.material_lagged_max_iterations):
            error = refresh(deformation)
            if error <= solver.material_lagged_tolerance:
                break
        assert error <= solver.material_lagged_tolerance
    evaluate(deformation, 0)
    reference, jacobian = stress.to_numpy().copy(), tangent.to_numpy().copy()
    assert np.isfinite(reference).all() and np.linalg.norm(reference) > 0
    epsilon = 1e-6
    evaluate(deformation + epsilon * direction, 0)
    plus = stress.to_numpy().copy()
    evaluate(deformation - epsilon * direction, 0)
    numerical = ((plus - stress.to_numpy()) / (2 * epsilon)).flatten(order="F")
    np.testing.assert_allclose(jacobian @ direction.flatten(order="F"), numerical, rtol=2e-5, atol=2e-3)
    evaluate(deformation, 1)
    evaluate(deformation, 0)
    np.testing.assert_allclose(stress.to_numpy(), reference, rtol=2e-7, atol=1e-5)

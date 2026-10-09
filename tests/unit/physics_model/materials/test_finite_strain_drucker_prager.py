"""Finite-strain Drucker--Prager parameter and analytic-tangent tests."""

import math

import numpy as np
import pytest
import taichi as ti

from src.physics_model.consititutive_model.finite_strain.DruckerPrager import (
    FiniteStrainDruckerPragerModel,
)

pytestmark = [pytest.mark.materials, pytest.mark.cpu]


def _model(**overrides):
    parameters = {
        "Density": 1800.0,
        "YoungModulus": 2.0e5,
        "PoissonRatio": 0.3,
        "FrictionAngle": 30.0,
        "DilationAngle": 30.0,
        "Cohesion": 1200.0,
        "dpType": "Circumscribed",
    }
    parameters.update(overrides)
    model = FiniteStrainDruckerPragerModel()
    model.model_initialize(parameters)
    model.allocate_state(1)
    return model


def _evaluator(model):
    energy = ti.field(ti.f64, shape=())
    stress = ti.Matrix.field(3, 3, ti.f64, shape=())
    tangent = ti.Matrix.field(9, 9, ti.f64, shape=())
    projected = ti.Matrix.field(3, 3, ti.f64, shape=())

    @ti.kernel
    def evaluate_kernel(values: ti.types.ndarray(dtype=ti.f64, ndim=2)):
        deformation = ti.Matrix.zero(ti.f64, 3, 3)
        for row, column in ti.static(ti.ndrange(3, 3)):
            deformation[row, column] = values[row, column]
        energy[None] = model.Psi_at(0, deformation)
        stress[None] = model.first_piola_stress_at(0, deformation)
        tangent[None] = model.first_piola_tangent_at(0, deformation)
        projected[None] = model.project_elastic_deformation(0, deformation)

    def evaluate(deformation):
        evaluate_kernel(np.ascontiguousarray(deformation, dtype=np.float64))
        return (
            float(energy[None]),
            stress.to_numpy()[()],
            tangent.to_numpy()[()],
            projected.to_numpy()[()],
        )

    return evaluate


def _flatten_column_major(matrix):
    return np.asarray(matrix, dtype=np.float64).T.reshape(-1)


def test_classical_circumscribed_cone_parameters(taichi_material_cpu):
    model = _model()
    phi = math.radians(30.0)
    expected_q = 6.0 * math.sin(phi) / (math.sqrt(3.0) * (3.0 - math.sin(phi)))
    expected_k = 6.0 * 1200.0 * math.cos(phi) / (math.sqrt(3.0) * (3.0 - math.sin(phi)))
    assert model.q_friction == pytest.approx(expected_q)
    assert model.k_cohesion == pytest.approx(expected_k)
    assert model.alpha == pytest.approx(math.sqrt(2.0) * expected_q / 3.0)
    assert model.cohesive_yield_stress == pytest.approx(math.sqrt(2.0) * expected_k)


def test_independent_dilation_preserves_friction_cone(taichi_material_cpu):
    model = _model(DilationAngle=5.0)
    associated = _model()
    assert model.alpha == associated.alpha
    assert model.reference_yield_intercept == associated.reference_yield_intercept
    assert 0.0 < model.beta < model.alpha
    assert not model.has_symmetric_tangent
    assert not model.has_physical_incremental_potential
    assert model.has_incremental_potential  # frozen inner potential


@pytest.mark.parametrize("dilation", [0.0, 15.0])
@pytest.mark.parametrize("branch", ["elastic", "cone", "apex"])
def test_direct_nonassociated_total_tangent_and_commit_stress(taichi_material_cpu, dilation, branch):
    model = _model(DilationAngle=dilation)
    model.use_direct_nonassociated_solve = True
    model.has_incremental_potential = False
    model.plastic_deformation_inverse[0] = np.diag(np.exp([-0.03, 0.01, -0.02]))
    strain = {
        "elastic": [-0.001, 0.0005, -0.002],
        "cone": [0.20, -0.12, -0.25],
        "apex": [0.04, 0.05, 0.06],
    }[branch]
    theta = 0.37
    left = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    right = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    deformation = left @ np.diag(np.exp(strain)) @ right.T @ np.linalg.inv(model.plastic_deformation_inverse[0])
    total = ti.Matrix.field(3, 3, ti.f64, shape=())
    stress = ti.Matrix.field(3, 3, ti.f64, shape=())
    tangent = ti.Matrix.field(9, 9, ti.f64, shape=())

    @ti.kernel
    def evaluate():
        stress[None] = model.total_first_piola_stress_at(0, total[None])
        tangent[None] = model.total_first_piola_tangent_at(0, total[None])

    @ti.kernel
    def commit():
        ignored = model.commit_total_state(0, total[None])

    total[None] = deformation
    evaluate()
    physical_stress, analytic = stress.to_numpy()[()], tangent.to_numpy()[()]
    old_inverse = model.plastic_deformation_inverse.to_numpy().copy()
    numerical = np.zeros((9, 9))
    step = 2e-5
    for column in range(9):
        plus, minus = deformation.copy(), deformation.copy()
        plus[column % 3, column // 3] += step
        minus[column % 3, column // 3] -= step
        total[None] = plus
        evaluate()
        plus_stress = stress.to_numpy()[()]
        total[None] = minus
        evaluate()
        numerical[:, column] = _flatten_column_major(plus_stress - stress.to_numpy()[()]) / (2 * step)
    assert np.linalg.norm(analytic - numerical) / np.linalg.norm(analytic) < 3e-6
    np.testing.assert_array_equal(model.plastic_deformation_inverse.to_numpy(), old_inverse)
    if branch == "cone":
        assert np.linalg.norm(analytic - analytic.T) > 0.01 * np.linalg.norm(analytic)
    total[None] = deformation
    commit()
    evaluate()
    # SVD roundoff at the cone/apex boundary can populate zero shear entries;
    # compare against the stress scale rather than those individual zeros.
    assert np.linalg.norm(stress.to_numpy()[()] - physical_stress) / np.linalg.norm(physical_stress) < 1e-9


@pytest.mark.parametrize(
    "deformation",
    [
        np.diag(np.exp([-0.001, 0.0005, -0.002])),
        np.array([[1.12, 0.06, -0.01], [0.02, 0.92, 0.04], [0.01, -0.03, 0.84]]),
        np.eye(3) * np.exp(0.05),
    ],
    ids=["elastic", "cone", "apex"],
)
def test_associated_incremental_energy_has_analytic_symmetric_tangent(
    taichi_material_cpu,
    deformation,
):
    model = _model()
    evaluate = _evaluator(model)
    energy, pk1, tangent, projected = evaluate(deformation)

    assert np.isfinite(energy)
    assert np.linalg.det(projected) > 0.0
    np.testing.assert_allclose(tangent, tangent.T, rtol=2.0e-10, atol=2.0e-8)

    step = 2.0e-7
    numerical_energy_gradient = np.zeros(9, dtype=np.float64)
    numerical = np.zeros((9, 9), dtype=np.float64)
    for column in range(9):
        component = column // 3
        row = column % 3
        plus = deformation.copy()
        minus = deformation.copy()
        plus[row, component] += step
        minus[row, component] -= step
        plus_energy, plus_stress = evaluate(plus)[:2]
        minus_energy, minus_stress = evaluate(minus)[:2]
        numerical_energy_gradient[column] = (plus_energy - minus_energy) / (2.0 * step)
        numerical[:, column] = (_flatten_column_major(plus_stress) - _flatten_column_major(minus_stress)) / (2.0 * step)
    np.testing.assert_allclose(
        _flatten_column_major(pk1),
        numerical_energy_gradient,
        rtol=5.0e-7,
        atol=2.0e-4,
    )
    np.testing.assert_allclose(tangent, numerical, rtol=3.0e-4, atol=2.0e-2)
    assert np.linalg.norm(pk1) > 0.0


def test_returned_stress_satisfies_classical_dp_cone(taichi_material_cpu):
    model = _model()
    evaluate = _evaluator(model)
    deformation = np.diag(np.exp([0.20, -0.12, -0.25]))
    _, _, _, projected = evaluate(deformation)

    singular_values = np.linalg.svd(projected, compute_uv=False)
    strain = np.log(singular_values)
    trace = strain.sum()
    deviatoric = strain - trace / 3.0
    kirchhoff = 2.0 * model.shear * deviatoric + model.bulk * trace
    mean = kirchhoff.mean()
    sqrt_j2 = np.linalg.norm(kirchhoff - mean) / math.sqrt(2.0)
    yield_value = sqrt_j2 + model.q_friction * mean - model.k_cohesion
    assert yield_value <= 1.0e-7 * max(model.k_cohesion, 1.0)


def test_hydrostatic_apex_commit_uses_associated_multiplier(
    taichi_material_cpu,
):
    model = _model()
    committed = ti.Matrix.field(3, 3, ti.f64, shape=())
    trial_log_stretch = 0.05

    @ti.kernel
    def commit():
        trial = ti.Matrix.identity(ti.f64, 3) * ti.exp(trial_log_stretch)
        committed[None] = model.commit_state(0, trial)

    commit()
    projected_singular_values = np.linalg.svd(committed.to_numpy()[()], compute_uv=False)
    projected_trace = float(np.log(projected_singular_values).sum())
    trial_trace = 3.0 * trial_log_stretch
    expected_multiplier = (trial_trace - model.trace_apex) / (3.0 * model.alpha)

    assert projected_trace == pytest.approx(model.trace_apex, abs=1.0e-12)
    assert float(model.equivalent_plastic_strain[0]) == pytest.approx(math.sqrt(2.0 / 3.0) * expected_multiplier)
    assert float(model.volumetric_plastic_strain[0]) == pytest.approx(trial_trace - model.trace_apex)


def test_total_deformation_contract_preserves_plastic_part_and_tangent(
    taichi_material_cpu,
):
    model = _model()
    committed_total = ti.Matrix.field(3, 3, ti.f64, shape=())
    energy = ti.field(ti.f64, shape=())
    stress = ti.Matrix.field(3, 3, ti.f64, shape=())
    tangent = ti.Matrix.field(9, 9, ti.f64, shape=())

    @ti.kernel
    def commit(values: ti.types.ndarray(dtype=ti.f64, ndim=2)):
        total = ti.Matrix.zero(ti.f64, 3, 3)
        for row, column in ti.static(ti.ndrange(3, 3)):
            total[row, column] = values[row, column]
        committed_total[None] = model.commit_total_state(0, total)

    @ti.kernel
    def evaluate(values: ti.types.ndarray(dtype=ti.f64, ndim=2)):
        total = ti.Matrix.zero(ti.f64, 3, 3)
        for row, column in ti.static(ti.ndrange(3, 3)):
            total[row, column] = values[row, column]
        energy[None] = model.total_strain_energy_density_at(0, total)
        stress[None] = model.total_first_piola_stress_at(0, total)
        tangent[None] = model.total_first_piola_tangent_at(0, total)

    first_total = np.diag(np.exp([0.20, -0.12, -0.25]))
    commit(np.ascontiguousarray(first_total))
    np.testing.assert_allclose(committed_total[None], first_total, atol=1.0e-14)
    plastic_inverse = model.plastic_deformation_inverse.to_numpy()[0]
    assert not np.allclose(plastic_inverse, np.eye(3))

    trial_total = first_total @ np.array([[1.01, 0.015, 0.0], [0.0, 0.995, 0.01], [0.0, 0.0, 1.0]])
    evaluate(np.ascontiguousarray(trial_total))
    analytic = tangent.to_numpy()[()]
    total_pk1 = stress.to_numpy()[()]

    step = 2.0e-7
    numerical = np.zeros((9, 9), dtype=np.float64)
    numerical_energy_gradient = np.zeros(9, dtype=np.float64)
    for column in range(9):
        material_axis = column // 3
        spatial_axis = column % 3
        plus = trial_total.copy()
        minus = trial_total.copy()
        plus[spatial_axis, material_axis] += step
        minus[spatial_axis, material_axis] -= step
        evaluate(np.ascontiguousarray(plus))
        plus_energy = float(energy[None])
        plus_stress = stress.to_numpy()[()].copy()
        evaluate(np.ascontiguousarray(minus))
        minus_energy = float(energy[None])
        minus_stress = stress.to_numpy()[()].copy()
        numerical_energy_gradient[column] = (plus_energy - minus_energy) / (2.0 * step)
        numerical[:, column] = (_flatten_column_major(plus_stress) - _flatten_column_major(minus_stress)) / (2.0 * step)
    np.testing.assert_allclose(analytic, numerical, rtol=5.0e-4, atol=3.0e-2)
    np.testing.assert_allclose(_flatten_column_major(total_pk1), numerical_energy_gradient, rtol=5.0e-7, atol=2.0e-4)

    elastic_trial = trial_total @ plastic_inverse
    evaluate_elastic = _evaluator(model)
    elastic_pk1 = evaluate_elastic(elastic_trial)[1]
    total_cauchy = total_pk1 @ trial_total.T / np.linalg.det(trial_total)
    elastic_cauchy = elastic_pk1 @ elastic_trial.T / np.linalg.det(elastic_trial)
    np.testing.assert_allclose(total_cauchy, elastic_cauchy, rtol=2.0e-12, atol=2.0e-10)


def test_lagged_plastic_volume_preserves_inner_derivatives_and_commit_stress(taichi_material_cpu):
    model = _model(YoungModulus=60.0e6, Density=1600.0, Cohesion=3000.0)
    total = ti.Matrix.field(3, 3, ti.f64, shape=())
    energy = ti.field(ti.f64, shape=())
    stress = ti.Matrix.field(3, 3, ti.f64, shape=())
    tangent = ti.Matrix.field(9, 9, ti.f64, shape=())
    history = ti.Vector.field(11, ti.f64, shape=())
    error = ti.field(ti.f64, shape=())

    @ti.kernel
    def begin():
        history[None] = model.get_history_state(0)
        model.begin_lagged_incremental_potential(0)

    @ti.kernel
    def refresh():
        error[None] = model.refresh_lagged_incremental_potential(0, total[None])

    @ti.kernel
    def evaluate():
        energy[None] = model.total_strain_energy_density_at(0, total[None])
        stress[None] = model.total_first_piola_stress_at(0, total[None])
        tangent[None] = model.total_first_piola_tangent_at(0, total[None])

    @ti.kernel
    def commit():
        ignored = model.commit_total_state(0, total[None])

    @ti.kernel
    def restore():
        model.set_history_state(0, history[None])

    deformations = [
        np.diag(np.exp([-0.0015, -0.001, -0.0005])),
        np.diag(np.exp([0.20, -0.12, -0.25])),
        np.diag(np.exp([0.04, 0.05, 0.06])),
        np.diag([3.0, 0.8, 14.0]),
    ]
    for branch, deformation in enumerate(deformations):
        total[None] = deformation
        begin()
        old_inverse = model.plastic_deformation_inverse.to_numpy().copy()
        refresh()
        refresh()
        assert error[None] < model.lagged_tolerance
        np.testing.assert_array_equal(model.plastic_deformation_inverse.to_numpy(), old_inverse)
        assert model.equivalent_plastic_strain[0] == 0.0
        evaluate()
        pk1, analytic = stress.to_numpy()[()], tangent.to_numpy()[()]
        if branch < 3:
            # Differentiate the inner potential at fixed history AND frozen weight.
            step = 1.0e-6
            numeric_p, numeric_h = np.zeros(9), np.zeros((9, 9))
            for column in range(9):
                row, axis = column % 3, column // 3
                plus, minus = deformation.copy(), deformation.copy()
                plus[row, axis] += step
                minus[row, axis] -= step
                total[None] = plus
                evaluate()
                plus_energy, plus_stress = energy[None], stress.to_numpy()[()]
                total[None] = minus
                evaluate()
                numeric_p[column] = (plus_energy - energy[None]) / (2 * step)
                numeric_h[:, column] = _flatten_column_major(plus_stress - stress.to_numpy()[()]) / (2 * step)
            np.testing.assert_allclose(_flatten_column_major(pk1), numeric_p, rtol=2.0e-6, atol=0.02)
            np.testing.assert_allclose(analytic, numeric_h, rtol=1.0e-4, atol=0.2)
        total[None] = deformation
        commit()
        assert model.lagged_plastic_volume_active[0] == 0
        actual_jp = 1.0 / np.linalg.det(model.plastic_deformation_inverse.to_numpy()[0])
        assert model.lagged_plastic_jacobian[0] == pytest.approx(actual_jp, rel=1.0e-10)
        evaluate()
        np.testing.assert_allclose(stress.to_numpy()[()], pk1, rtol=1.0e-7, atol=0.002)
        # Retry/history restore must discard the predictor, including before commit.
        restore()
        begin()
        refresh()
        restore()
        assert model.lagged_plastic_volume_active[0] == 0
        np.testing.assert_array_equal(model.plastic_deformation_inverse.to_numpy(), old_inverse)


@pytest.mark.parametrize("dilation", [0.0, 15.0])
@pytest.mark.parametrize("branch", ["elastic", "cone", "apex"])
def test_nonassociated_physical_return_and_frozen_symmetric_potential(taichi_material_cpu, dilation, branch):
    model = _model(DilationAngle=dilation)
    strain = {
        "elastic": [-0.001, 0.0005, -0.002],
        "cone": [0.20, -0.12, -0.25],
        "apex": [0.04, 0.05, 0.06],
    }[branch]
    # Independent left/right rotations exercise spectral off-diagonal derivatives.
    theta = 0.37
    left = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    right = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    deformation = left @ np.diag(np.exp(strain)) @ right.T
    total = ti.Matrix.field(3, 3, ti.f64, shape=())
    physical_p = ti.Matrix.field(3, 3, ti.f64, shape=())
    physical_h = ti.Matrix.field(9, 9, ti.f64, shape=())
    error = ti.field(ti.f64, shape=())

    @ti.kernel
    def physical():
        physical_p[None] = model.total_first_piola_stress_at(0, total[None])
        physical_h[None] = model.total_first_piola_tangent_at(0, total[None])

    @ti.kernel
    def begin():
        model.begin_lagged_incremental_potential(0)

    @ti.kernel
    def refresh():
        error[None] = model.refresh_lagged_incremental_potential(0, total[None])

    @ti.kernel
    def commit():
        ignored = model.commit_total_state(0, total[None])

    total[None] = deformation
    physical()
    true_h = physical_h.to_numpy()[()]
    # Rotated, nearly repeated stretches amplify Taichi SVD roundoff for
    # tiny perturbations. Use a resolvable step and the operator norm error.
    step = 2.0e-5
    numerical = np.zeros((9, 9))
    for column in range(9):
        plus, minus = deformation.copy(), deformation.copy()
        plus[column % 3, column // 3] += step
        minus[column % 3, column // 3] -= step
        total[None] = plus
        physical()
        plus_p = physical_p.to_numpy()[()]
        total[None] = minus
        physical()
        numerical[:, column] = _flatten_column_major(plus_p - physical_p.to_numpy()[()]) / (2 * step)
    assert np.linalg.norm(true_h - numerical) / np.linalg.norm(true_h) < 3e-6
    if branch == "cone":
        assert np.linalg.norm(true_h - true_h.T) > 0.01 * np.linalg.norm(true_h)

    total[None] = deformation
    begin()
    refresh()
    refresh()
    assert error[None] < model.lagged_tolerance
    assert model.equivalent_plastic_strain[0] == 0
    evaluate = _evaluator(model)
    energy, inner_p, inner_h, projected = evaluate(deformation)
    np.testing.assert_allclose(inner_h, inner_h.T, rtol=1e-10, atol=2e-8)
    numeric_p, numeric_h = np.zeros(9), np.zeros((9, 9))
    for column in range(9):
        plus, minus = deformation.copy(), deformation.copy()
        plus[column % 3, column // 3] += step
        minus[column % 3, column // 3] -= step
        plus_e, plus_p = evaluate(plus)[:2]
        minus_e, minus_p = evaluate(minus)[:2]
        numeric_p[column] = (plus_e - minus_e) / (2 * step)
        numeric_h[:, column] = _flatten_column_major(plus_p - minus_p) / (2 * step)
    jp = model.lagged_plastic_jacobian[0]
    np.testing.assert_allclose(_flatten_column_major(inner_p), numeric_p, rtol=2e-6, atol=0.001)
    assert np.linalg.norm(inner_h - numeric_h) / np.linalg.norm(inner_h) < 3e-6
    total[None] = deformation
    commit()
    physical()
    assert np.linalg.norm(physical_p.to_numpy()[()] - jp * inner_p) / np.linalg.norm(jp * inner_p) < 1e-9
    trace = float(np.log(np.linalg.svd(projected, compute_uv=False)).sum())
    assert model.volumetric_plastic_strain[0] == pytest.approx(sum(strain) - trace, abs=1e-12)
    if branch == "cone":
        assert model.equivalent_plastic_strain[0] > 0
        assert model.volumetric_plastic_strain[0] == pytest.approx(
            3.0 * model.beta * model.equivalent_plastic_strain[0] / math.sqrt(2 / 3),
            abs=1e-12,
        )
        if dilation == 0:
            assert model.volumetric_plastic_strain[0] == pytest.approx(0, abs=1e-12)


def test_nonassociated_apex_fixed_point_checks_stress_not_shift(taichi_material_cpu):
    model = _model(DilationAngle=0.0)
    deformation = np.diag(np.exp([0.04, 0.05, 0.06]))
    error = ti.field(ti.f64, shape=())

    @ti.kernel
    def refresh():
        total = ti.Matrix.zero(ti.f64, 3, 3)
        for row, column in ti.static(ti.ndrange(3, 3)):
            total[row, column] = deformation[row, column]
        error[None] = model.refresh_lagged_incremental_potential(0, total)

    @ti.kernel
    def begin():
        model.begin_lagged_incremental_potential(0)

    begin()
    refresh()
    # Several shifts give the identical capped stress. A changed shift alone
    # must not demand another global equilibrium solve in this flat branch.
    model.lagged_flow_shift[0] += 0.02
    evaluate = _evaluator(model)
    before = evaluate(deformation)[1]
    refresh()
    after = evaluate(deformation)[1]
    np.testing.assert_allclose(before, after, rtol=1e-12, atol=1e-9)
    assert error[None] <= model.lagged_tolerance

    # A shift that leaves the cap really changes the stress and must fail.
    model.lagged_flow_shift[0] = -0.1
    refresh()
    assert error[None] > model.lagged_tolerance


def test_nonassociated_lagged_solve_converges_for_constrained_uniaxial_load(
    taichi_material_cpu,
):
    model = _model(DilationAngle=0.0)
    direction = math.sqrt(2.0 / 3.0)
    axial_strain, lateral_strain = 0.2, -0.1
    norm = direction * (axial_strain - lateral_strain)
    multiplier = norm - model.cohesive_yield_stress / (2.0 * model.shear)
    shift_at_equilibrium = model.alpha * multiplier
    load = direction * model.cohesive_yield_stress
    stress_direction = 2.0 * model.shear * direction + 3.0 * model.alpha * model.bulk
    denominator = 2.0 * model.shear + 9.0 * model.alpha**2 * model.bulk
    inner_modulus = 4.0 * model.shear / 3.0 + model.bulk - stress_direction**2 / denominator
    shift_modulus = 3.0 * model.bulk - stress_direction * 9.0 * model.alpha * model.bulk / denominator
    physical_modulus = model.bulk * (1.0 - 3.0 * model.alpha * direction)
    assert inner_modulus > 0 and physical_modulus > 0
    # Both equilibrium tangents are positive, but a full material update
    # overshoots: this failure exists without contact or an indefinite PCG solve.
    assert 1.0 - physical_modulus / inner_modulus < -1.0
    total = ti.Matrix.field(3, 3, ti.f64, shape=())
    stress = ti.Matrix.field(3, 3, ti.f64, shape=())
    error = ti.field(ti.f64, shape=())

    @ti.kernel
    def begin():
        model.begin_lagged_incremental_potential(0)

    @ti.kernel
    def refresh():
        stress[None] = model.total_first_piola_stress_at(0, total[None]) @ total[None].transpose()
        error[None] = model.refresh_lagged_incremental_potential(0, total[None])

    begin()
    model.lagged_flow_shift[0] = shift_at_equilibrium + 1e-7
    for _ in range(5):
        shift = float(model.lagged_flow_shift[0])
        # Exact frozen-inner equilibrium under constant Kirchhoff axial load,
        # with both lateral logarithmic strains constrained.
        axial = axial_strain - shift_modulus * (shift - shift_at_equilibrium) / inner_modulus
        total[None] = np.diag(np.exp([axial, lateral_strain, lateral_strain]))
        refresh()
        assert stress[None][0, 0] == pytest.approx(load, abs=5e-8)
        if error[None] <= model.lagged_tolerance:
            break
    assert error[None] <= model.lagged_tolerance
    assert axial == pytest.approx(axial_strain, abs=1e-9)

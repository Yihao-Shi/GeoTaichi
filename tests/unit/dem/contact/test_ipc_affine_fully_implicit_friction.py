import os
import importlib
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

pytestmark = [
    pytest.mark.unit,
    pytest.mark.dem,
    pytest.mark.ipc,
    pytest.mark.contact,
    pytest.mark.cpu,
    pytest.mark.serial,
]

from test_ipc_affine_friction_assembly import _build_affine_contact
from src.dem.engines.AffineBodyEngine import AffineBodyEngine
from src.dem.engines.AffineBodyOperator import MATRIX_COO, TaichiAffineBodyOperator
from src.physics_model.contact_model.ipc.IPC import (
    ipc_barrier_distance_terms_py,
    ipc_fully_implicit_scalar_law_py,
)


def _full_column_finite_difference(evaluate, positions, step):
    positions = np.asarray(positions, dtype=np.float64).reshape(-1)
    jacobian = np.zeros((positions.size, positions.size), dtype=np.float64)
    for column in range(positions.size):
        delta = np.zeros_like(positions)
        delta[column] = step
        plus = evaluate(positions + delta)[0]
        minus = evaluate(positions - delta)[0]
        jacobian[:, column] = (plus - minus) / (2.0 * step)
    return jacobian


def _relative_matrix_error(actual, expected):
    return np.linalg.norm(actual - expected, ord=np.inf) / max(
        np.linalg.norm(actual, ord=np.inf),
        np.linalg.norm(expected, ord=np.inf),
        1.0e-14,
    )


def test_affine_fully_implicit_residual_only_source_omits_large_derivatives():
    residual_source = inspect.getsource(TaichiAffineBodyOperator._fully_implicit_local_friction_residual)
    for forbidden in (
        "distance_hessian",
        "barrier_hessian",
        "mollifier_gradient",
        "local_jacobian",
        "Matrix.zero(float, 12, 12)",
    ):
        assert forbidden not in residual_source

    pt_source = inspect.getsource(TaichiAffineBodyOperator._evaluate_fully_implicit_pt_local_residual_kernel)
    assert "point_triangle_distance_grad(" in pt_source
    assert "point_triangle_distance_grad_hess" not in pt_source

    ee_source = inspect.getsource(TaichiAffineBodyOperator._evaluate_fully_implicit_ee_local_residual_kernel)
    assert "edge_edge_distance_grad(" in ee_source
    assert "edge_edge_distance_grad_hess" not in ee_source
    assert "edge_edge_mollifier_grad" not in ee_source

    operator_source = inspect.getsource(TaichiAffineBodyOperator)
    assert "ti.static(range(4))" not in operator_source


def test_affine_fully_implicit_non_degenerate_pt_and_mollified_ee_full_fd(tmp_path):
    """Every local column includes dT/dq, d(lambda)/dq, and dm/dq."""
    os.environ.setdefault("GEOTAICHI_REAL_DTYPE", "float64")
    ti.init(arch=ti.cpu, default_fp=ti.f64, debug=False, offline_cache=False)
    dem, unused_hat = _build_affine_contact(
        tmp_path / "fully-implicit-pt-ee",
        friction_mode="fully_implicit",
    )
    operator = dem.enginer.operator

    # Exercise the complete paper law, rather than the equal-mu compatibility
    # special case.  These are compile-time parameters of this operator and
    # are set before the first fully-implicit kernel invocation.
    operator.fully_mu_dynamic = 0.32
    operator.fully_mu_static = 0.58
    operator.fully_mu_viscous = 0.015
    operator.fully_stribeck_velocity = 0.20
    operator.fully_profile_id = 0

    pt = np.array(
        [
            [0.23, 0.31, 0.075],
            [0.00, 0.00, 0.000],
            [1.10, 0.05, 0.010],
            [0.10, 1.00, -0.015],
        ],
        dtype=np.float64,
    ).reshape(-1)
    pt_hat = pt.copy().reshape((4, 3))
    pt_hat[0] -= np.array([1.3e-4, -0.7e-4, 0.2e-5])
    pt_hat[1] -= np.array([-0.2e-4, 0.1e-4, 0.0])
    pt_hat[2] -= np.array([0.1e-4, -0.1e-4, 0.0])
    pt_hat[3] -= np.array([0.0, 0.2e-4, 0.0])
    pt_hat = pt_hat.reshape(-1)

    def evaluate_pt(value, need_matrix=True):
        return operator.evaluate_fully_implicit_contact_local(
            "pt",
            value,
            pt_hat,
            dhat=0.20,
            kappa=2.0e4,
            pair_mu=0.4,
            area=0.73,
            need_matrix=need_matrix,
        )

    pt_residual, pt_jacobian = evaluate_pt(pt, need_matrix=True)
    pt_residual_only, no_pt_jacobian = evaluate_pt(pt, need_matrix=False)
    assert no_pt_jacobian is None
    np.testing.assert_array_max_ulp(pt_residual_only, pt_residual, maxulp=32)
    pt_fd = _full_column_finite_difference(
        lambda value: evaluate_pt(value, need_matrix=False),
        pt,
        2.0e-8,
    )
    assert np.linalg.norm(pt_residual) > 0.0
    np.testing.assert_allclose(
        pt_residual.reshape((4, 3)).sum(axis=0),
        np.zeros(3),
        rtol=0.0,
        atol=1.0e-10 * max(np.linalg.norm(pt_residual), 1.0),
    )
    assert _relative_matrix_error(pt_jacobian, pt_fd) < 2.0e-5
    assert np.linalg.norm(pt_jacobian - pt_jacobian.T) > 1.0e-6

    # The production plane-wall branch calls the shared IPC point-plane
    # kernel and then applies the affine time/area scale.  Differentiate the
    # resulting wall residual with respect to every moving-point coordinate.
    plane_position = np.array([0.31, -0.17, 0.075], dtype=np.float64)
    plane_hat = plane_position - np.array([1.3e-4, -0.8e-4, 0.2e-5], dtype=np.float64)

    def evaluate_plane(value):
        return operator.evaluate_fully_implicit_plane_local(
            value,
            plane_hat,
            plane_point=np.zeros(3),
            normal=np.array([0.0, 0.0, 1.0]),
            dhat=0.20,
            kappa=2.0e4,
            pair_mu=0.4,
            area=0.67,
        )

    plane_residual, plane_jacobian = evaluate_plane(plane_position)
    plane_fd = _full_column_finite_difference(evaluate_plane, plane_position, 2.0e-8)
    assert np.linalg.norm(plane_residual) > 0.0
    assert _relative_matrix_error(plane_jacobian, plane_fd) < 2.0e-5
    assert np.linalg.norm(plane_jacobian - plane_jacobian.T) > 1.0e-6

    # Cross-backend constitutive convention: contact measure belongs to
    # lambda, while the public viscous coefficient does not acquire another
    # area factor.  Rebuild the shared MPM/IGA scalar law analytically.
    unused_energy, barrier_gradient, unused_hessian = ipc_barrier_distance_terms_py(
        plane_position[2], 0.20, kappa=2.0e4
    )
    plane_area = 0.67
    normal_force = plane_area * max(-barrier_gradient, 0.0)
    velocity = (plane_position - plane_hat) / operator.dt
    velocity[2] = 0.0
    radial_factor, unused_speed_derivative, unused_force_derivative = ipc_fully_implicit_scalar_law_py(
        np.linalg.norm(velocity),
        normal_force,
        operator.fully_mu_dynamic,
        operator.fully_mu_static,
        operator.fully_mu_viscous,
        operator.fully_stribeck_velocity,
        operator.epsv,
        profile="quadratic",
    )
    np.testing.assert_allclose(
        plane_residual,
        operator.dt * operator.dt * radial_factor * velocity,
        rtol=2.0e-10,
        atol=2.0e-12,
    )

    # A facet wall is the same exact PT law with the triangle held fixed.
    # The wall assembler scatters precisely this leading 3x3 block.
    facet_positions = pt.copy().reshape((4, 3))
    facet_hats = facet_positions.copy()
    facet_hats[0] -= np.array([1.1e-4, -0.6e-4, 0.1e-5])
    facet_hats = facet_hats.reshape(-1)

    def evaluate_facet_point(value, need_matrix=True):
        positions = facet_positions.copy()
        positions[0] = np.asarray(value, dtype=np.float64)
        residual, jacobian = operator.evaluate_fully_implicit_contact_local(
            "pt",
            positions.reshape(-1),
            facet_hats,
            dhat=0.20,
            kappa=2.0e4,
            pair_mu=0.4,
            area=0.69,
            need_matrix=need_matrix,
        )
        return (
            residual[:3],
            jacobian[:3, :3] if jacobian is not None else None,
        )

    facet_residual, facet_jacobian = evaluate_facet_point(facet_positions[0], need_matrix=True)
    facet_residual_only, no_facet_jacobian = evaluate_facet_point(facet_positions[0], need_matrix=False)
    assert no_facet_jacobian is None
    np.testing.assert_array_max_ulp(facet_residual_only, facet_residual, maxulp=32)
    facet_fd = _full_column_finite_difference(
        lambda value: evaluate_facet_point(value, need_matrix=False),
        facet_positions[0],
        2.0e-8,
    )
    assert np.linalg.norm(facet_residual) > 0.0
    assert _relative_matrix_error(facet_jacobian, facet_fd) < 2.0e-5

    # Current edges are non-parallel but lie inside IPC's near-parallel
    # mollifier region.  Rest edges set eps_x exactly as the production code.
    angle = np.deg2rad(1.05)
    ee = np.array(
        [
            [-0.52, 0.00, 0.000],
            [0.48, 0.00, 0.000],
            [-0.49 * np.cos(angle), -0.49 * np.sin(angle), 0.052],
            [0.51 * np.cos(angle), 0.51 * np.sin(angle), 0.052],
        ],
        dtype=np.float64,
    ).reshape(-1)
    ee_rest = np.array(
        [
            [-0.50, 0.00, 0.00],
            [0.50, 0.00, 0.00],
            [-0.50, 0.04, 0.00],
            [0.50, 0.04, 0.00],
        ],
        dtype=np.float64,
    ).reshape(-1)
    edge_a = ee.reshape((4, 3))[1] - ee.reshape((4, 3))[0]
    edge_b = ee.reshape((4, 3))[3] - ee.reshape((4, 3))[2]
    cross2 = np.dot(np.cross(edge_a, edge_b), np.cross(edge_a, edge_b))
    rest_a = ee_rest.reshape((4, 3))[1] - ee_rest.reshape((4, 3))[0]
    rest_b = ee_rest.reshape((4, 3))[3] - ee_rest.reshape((4, 3))[2]
    eps_x = 1.0e-3 * np.dot(rest_a, rest_a) * np.dot(rest_b, rest_b)
    ratio = cross2 / eps_x
    mollifier = ratio * (2.0 - ratio)
    assert 0.0 < mollifier < 1.0

    ee_hat = ee.copy().reshape((4, 3))
    ee_hat[0] -= np.array([0.2e-4, 1.1e-4, -0.1e-5])
    ee_hat[1] -= np.array([-0.1e-4, 0.9e-4, 0.2e-5])
    ee_hat[2] -= np.array([0.1e-4, -0.8e-4, 0.0])
    ee_hat[3] -= np.array([-0.2e-4, -1.0e-4, 0.1e-5])
    ee_hat = ee_hat.reshape(-1)

    def evaluate_ee(value, need_matrix=True):
        return operator.evaluate_fully_implicit_contact_local(
            "ee",
            value,
            ee_hat,
            dhat=0.20,
            kappa=2.0e4,
            pair_mu=0.4,
            area=0.61,
            rest_positions=ee_rest,
            need_matrix=need_matrix,
        )

    ee_residual, ee_jacobian = evaluate_ee(ee, need_matrix=True)
    ee_residual_only, no_ee_jacobian = evaluate_ee(ee, need_matrix=False)
    assert no_ee_jacobian is None
    np.testing.assert_array_max_ulp(ee_residual_only, ee_residual, maxulp=32)
    ee_fd = _full_column_finite_difference(
        lambda value: evaluate_ee(value, need_matrix=False),
        ee,
        5.0e-9,
    )
    assert np.linalg.norm(ee_residual) > 0.0
    np.testing.assert_allclose(
        ee_residual.reshape((4, 3)).sum(axis=0),
        np.zeros(3),
        rtol=0.0,
        atol=1.0e-10 * max(np.linalg.norm(ee_residual), 1.0),
    )
    assert _relative_matrix_error(ee_jacobian, ee_fd) < 8.0e-5
    assert np.linalg.norm(ee_jacobian - ee_jacobian.T) > 1.0e-6


def test_affine_fully_implicit_global_residual_only_matches_matrix_assembly(tmp_path):
    """Compile both production branches and compare their assembled residual."""
    os.environ.setdefault("GEOTAICHI_REAL_DTYPE", "float64")
    ti.init(arch=ti.cpu, default_fp=ti.f64, debug=False, offline_cache=False)
    dem, hat_y = _build_affine_contact(
        tmp_path / "fully-implicit-production-branches",
        with_wall=True,
        friction_mode="fully_implicit",
    )
    engine = dem.enginer
    displacement = np.zeros_like(hat_y).reshape((2, 4, 3))
    displacement[0, :, 1] = 7.0e-5
    displacement[1, :, 1] = -5.0e-5
    y = hat_y + displacement.reshape(-1)

    matrix_energy, matrix_residual = engine._assemble_system(dem.sims, y, need_matrix=True)
    residual_energy, residual_only = engine._assemble_system(dem.sims, y, need_matrix=False)

    assert np.all(np.isfinite(matrix_residual))
    assert np.linalg.norm(matrix_residual) > 0.0
    np.testing.assert_allclose(
        residual_energy,
        matrix_energy,
        rtol=5.0e-13,
        atol=5.0e-13 * max(abs(matrix_energy), 1.0),
    )
    np.testing.assert_allclose(
        residual_only,
        matrix_residual,
        rtol=5.0e-13,
        atol=5.0e-13 * max(np.linalg.norm(matrix_residual), 1.0),
    )

    operator = engine.operator
    controls = y.reshape((operator.control_num, 3))
    expected_vertices = np.sum(
        operator.basis_np[:, :, None] * controls.reshape((operator.body_num, 4, 3))[operator.node2body_np],
        axis=1,
    )
    np.testing.assert_allclose(
        operator.x.to_numpy()[: operator.vertex_num],
        expected_vertices,
        rtol=2.0e-14,
        atol=2.0e-14,
    )

    direction = np.linspace(
        -3.0e-3,
        4.0e-3,
        operator.control_num * 3,
        dtype=np.float64,
    ).reshape((operator.control_num, 3))
    operator.direction_y.from_numpy(direction)
    operator._reconstruct_vertex_directions()
    expected_direction = np.sum(
        operator.basis_np[:, :, None] * direction.reshape((operator.body_num, 4, 3))[operator.node2body_np],
        axis=1,
    )
    np.testing.assert_allclose(
        operator.dx.to_numpy()[: operator.vertex_num],
        expected_direction,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    assert operator.device_surface_direction_inf_norm() == pytest.approx(
        np.max(np.abs(expected_direction)),
        rel=2.0e-14,
        abs=2.0e-14,
    )

    gravity = np.ascontiguousarray(np.asarray(engine.state.gravity, dtype=np.float64))
    mass = operator.mass.to_numpy()[: operator.body_num]
    lumped_mass = np.sum(mass, axis=2)
    expected_body_force = (-operator.scale * lumped_mass[:, :, None] * gravity[None, None, :]).reshape(
        (operator.control_num, 3)
    )
    expected_body_force_energy = float(np.sum(expected_body_force * controls))

    operator._clear_system(False, int(MATRIX_COO))
    operator._assemble_body_force(gravity, False)
    np.testing.assert_allclose(
        operator.grad.to_numpy()[: operator.control_num],
        expected_body_force,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    assert float(operator.energy[None]) == pytest.approx(
        expected_body_force_energy,
        rel=2.0e-14,
        abs=2.0e-14,
    )

    operator._clear_system(False, int(MATRIX_COO))
    operator._assemble_body_force_device(False)
    np.testing.assert_allclose(
        operator.grad.to_numpy()[: operator.control_num],
        expected_body_force,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    assert float(operator.energy[None]) == pytest.approx(
        expected_body_force_energy,
        rel=2.0e-14,
        abs=2.0e-14,
    )


def test_affine_fully_implicit_friction_adjoint_is_rejected(tmp_path):
    os.environ.setdefault("GEOTAICHI_REAL_DTYPE", "float64")
    ti.init(arch=ti.cpu, default_fp=ti.f64, debug=False, offline_cache=False)
    dem, _ = _build_affine_contact(
        tmp_path / "fully-implicit-adjoint",
        friction_mode="fully_implicit",
    )
    engine = dem.enginer
    seed = np.linspace(-0.4, 0.7, engine.operator.dof)
    with pytest.raises(ValueError, match="fully_implicit friction"):
        engine.solve_adjoint(seed)


def test_official_lagged_ee_contract_still_skips_the_mollifier_region():
    """Document the distinction retained by the lagged path."""
    angle = np.deg2rad(1.0)
    edge_a = np.array([1.0, 0.0, 0.0])
    edge_b = np.array([np.cos(angle), np.sin(angle), 0.0])
    cross2 = np.dot(np.cross(edge_a, edge_b), np.cross(edge_a, edge_b))
    eps_x = 1.0e-3 * np.dot(edge_a, edge_a) * np.dot(edge_b, edge_b)
    ratio = cross2 / eps_x
    mollifier = ratio * (2.0 - ratio)
    assert 0.0 < mollifier < 1.0
    # AffineBodyEngine._initialize_lagged_mesh_friction stores an EE
    # tangential collision only for mollifier >= 1.
    assert not (mollifier >= 1.0)


class _ScalarField:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, _index):
        return self.value


class _DispatchState:
    def __init__(self):
        self.hat_y = np.zeros((1, 4, 3), dtype=np.float64)
        self.tilde_y = self.hat_y.copy()
        self.accepted = None

    def begin_step(self, _dt):
        self.tilde_y = self.hat_y.copy()

    def pack(self, values=None):
        if values is None:
            values = self.hat_y
        return np.asarray(values, dtype=np.float64).reshape(-1)

    def unpack(self, values):
        return np.asarray(values, dtype=np.float64).reshape((1, 4, 3))

    def accept_step(self, values, _dt):
        self.accepted = np.asarray(values).copy()


def _minimal_operator_sims(**overrides):
    values = {
        "max_material_num": 1,
        "dt": _ScalarField(0.01),
        "affine_dhat": 0.1,
        "affine_barrier_stiffness": 1.0e4,
        "affine_contact_damping_stiffness": 0.0,
        "affine_friction_epsv": 1.0e-3,
        "affine_hessian_shift": 0.0,
        "affine_friction_mode": "fully_implicit",
        "affine_dynamic_friction": -1.0,
        "affine_static_friction": -1.0,
        "affine_viscous_friction": 0.0,
        "affine_stribeck_velocity": -1.0,
        "affine_friction_profile": "quadratic",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _minimal_operator_state():
    return SimpleNamespace(
        body_num=1,
        control_num=4,
        bodies=[{"mass_matrix": np.eye(4, dtype=np.float64)}],
    )


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("affine_dynamic_friction", np.nan),
        ("affine_static_friction", np.inf),
        ("affine_viscous_friction", np.nan),
        ("affine_stribeck_velocity", -np.inf),
        ("affine_friction_epsv", np.inf),
        ("affine_dhat", np.nan),
        ("affine_barrier_stiffness", np.inf),
        ("affine_fully_implicit_jacobian_shift", np.nan),
        ("dt", _ScalarField(np.nan)),
    ],
)
def test_affine_operator_rejects_nonfinite_solver_inputs(attribute, value):
    sims = _minimal_operator_sims(**{attribute: value})
    state = _minimal_operator_state()
    with pytest.raises(ValueError, match="finite"):
        TaichiAffineBodyOperator(state, sims, scene=None)


def test_affine_operator_rejects_unrecognized_negative_stribeck_sentinel():
    sims = _minimal_operator_sims(affine_stribeck_velocity=-2.0)
    state = _minimal_operator_state()
    with pytest.raises(ValueError, match="non-negative or -1"):
        TaichiAffineBodyOperator(state, sims, scene=None)


@pytest.mark.parametrize("iterations", [-1, 0, 2])
def test_affine_fully_implicit_rejects_lagged_outer_iterations(iterations):
    sims = _minimal_operator_sims(affine_friction_iterations=iterations)
    state = _minimal_operator_state()
    with pytest.raises(ValueError, match="friction_iterations=1"):
        TaichiAffineBodyOperator(state, sims, scene=None)


def test_affine_operator_rejects_nonfinite_pair_contact_properties():
    operator = object.__new__(TaichiAffineBodyOperator)
    operator.material_num = 1
    operator.default_dhat = 0.1
    operator.default_kappa = 1.0e4
    operator.default_contact_damping_stiffness = 0.0
    scene = SimpleNamespace(affine_contact_properties={(0, 0, "particle-particle"): {"Friction": np.nan}})
    with pytest.raises(ValueError, match="friction.*finite"):
        operator._build_contact_property_arrays(scene, walls=[])


def test_affine_fully_implicit_default_ignores_lagged_hessian_shift():
    operator = object.__new__(TaichiAffineBodyOperator)
    operator.fully_implicit = True
    operator.hessian_shift = 3.5
    operator.fully_implicit_jacobian_shift = 0.0
    assert operator._active_jacobian_shift() == 0.0

    operator.fully_implicit_jacobian_shift = 2.0e-8
    assert operator._active_jacobian_shift() == pytest.approx(2.0e-8)

    operator.fully_implicit = False
    assert operator._active_jacobian_shift() == pytest.approx(3.5)


@pytest.mark.parametrize("mode", ["fully_implicit", "fullyimplicit", "fully-implicit"])
def test_affine_fully_implicit_dispatch_bypasses_lagged_outer_loop(mode):
    engine = AffineBodyEngine()
    engine.state = _DispatchState()
    engine.operator = SimpleNamespace(fully_implicit=True)
    events = []
    engine._step_fully_implicit = lambda sims, y, dt: events.append((sims, np.asarray(y).copy(), dt))
    engine._solve_lagged_inner = lambda *args: pytest.fail("fully implicit mode entered the lagged fixed-point loop")
    sims = SimpleNamespace(
        dt=_ScalarField(0.02),
        delta=0.02,
        current_step=0,
        current_time=0.0,
        affine_friction_mode=mode,
    )
    engine.step(sims, scene=None)
    assert len(events) == 1
    assert events[0][2] == pytest.approx(0.02)


def test_affine_fully_implicit_line_search_uses_residual_not_energy():
    engine = AffineBodyEngine()
    engine._ccd_step_size = lambda _sims, _y, _direction: 1.0
    calls = []

    def assemble(_sims, trial, need_matrix=True):
        calls.append((float(np.asarray(trial)[0]), need_matrix))
        # Deliberately increase the conservative diagnostic energy while the
        # nonconservative residual decreases.  A potential Armijo test would
        # reject this; the paper's residual-merit Armijo must accept it.
        return 1.0e12, np.array([0.25], dtype=np.float64)

    engine._assemble_system = assemble
    sims = SimpleNamespace(
        affine_line_search_max_iteration=4,
        affine_fully_implicit_armijo=1.0e-4,
        affine_fully_implicit_line_search_contraction=0.5,
    )
    alpha, residual = engine._fully_implicit_line_search(sims, np.zeros(1), np.ones(1), np.ones(1), -np.ones(1))
    assert alpha == pytest.approx(1.0)
    np.testing.assert_allclose(residual, [0.25])
    assert calls == [(1.0, False), (1.0, True)]


def test_affine_fully_implicit_armijo_uses_actual_jacobian_slope():
    engine = AffineBodyEngine()
    engine._ccd_step_size = lambda _sims, _y, _direction: 1.0
    calls = []

    def assemble(_sims, _trial, need_matrix=True):
        calls.append(need_matrix)
        return 0.0, np.array([0.99995], dtype=np.float64)

    engine._assemble_system = assemble
    sims = SimpleNamespace(
        affine_line_search_max_iteration=3,
        affine_fully_implicit_armijo=1.0e-4,
        affine_fully_implicit_line_search_contraction=0.5,
    )
    # This represents a max-step-clamped Newton correction: R^T Jp=-0.25,
    # not the unscaled -||R||^2.  The exact merit Armijo accepts alpha=1;
    # the old hard-coded residual-norm bound would reject it.
    alpha, residual = engine._fully_implicit_line_search(
        sims,
        np.zeros(1),
        np.ones(1),
        np.ones(1),
        np.array([-0.25]),
    )
    assert alpha == pytest.approx(1.0)
    np.testing.assert_allclose(residual, [0.99995])
    assert engine.last_fully_implicit_merit_slope == pytest.approx(-0.25)
    assert calls == [False, True]


def test_affine_fully_implicit_rejects_non_descent_merit_direction():
    engine = AffineBodyEngine()
    engine._ccd_step_size = lambda _sims, _y, _direction: 1.0
    engine._assemble_system = lambda *_args, **_kwargs: pytest.fail(
        "trial assembly ran for a non-descent merit direction"
    )
    sims = SimpleNamespace(
        affine_line_search_max_iteration=3,
        affine_fully_implicit_armijo=1.0e-4,
        affine_fully_implicit_line_search_contraction=0.5,
    )
    with pytest.raises(RuntimeError, match="not a descent direction"):
        engine._fully_implicit_line_search(
            sims,
            np.zeros(1),
            np.ones(1),
            np.ones(1),
            np.ones(1),
        )
    assert engine.last_inner_failure_reason == "non_descent_residual_merit_direction"


def test_affine_merit_slope_removes_explicit_jacobian_shift():
    engine = AffineBodyEngine()
    engine.operator = SimpleNamespace(dof=2, _active_jacobian_shift=lambda: 2.0)
    engine.coo_matrix = SimpleNamespace(_to_scipy=lambda: 3.0 * np.eye(2))
    direction = np.array([0.4, -0.7])
    product = engine._fully_implicit_jacobian_direction(SimpleNamespace(affine_assemble_type="COO"), direction)
    # The solve used (J + 2I), while merit differentiation must use J.
    np.testing.assert_allclose(product, direction)


def test_affine_fully_implicit_failure_does_not_accept_state():
    engine = AffineBodyEngine()
    engine.state = _DispatchState()
    engine.operator = SimpleNamespace(initialize_contact_damping=lambda _y, _hat: None)
    assembled_configurations = []

    def assemble(_sims, y, need_matrix=True):
        assembled_configurations.append(np.asarray(y).copy())
        return 0.0, np.ones(12, dtype=np.float64)

    engine._assemble_system = assemble
    engine._solve_direction = lambda _sims, residual: -np.asarray(residual)
    engine._fully_implicit_jacobian_direction = lambda _sims, direction: np.asarray(direction)

    def fail_after_trial(sims, *_args):
        engine._assemble_system(sims, np.full(12, 7.0, dtype=np.float64), need_matrix=False)
        raise RuntimeError("synthetic line-search failure")

    engine._fully_implicit_line_search = fail_after_trial
    engine._ccd_step_size = lambda *_args: 1.0
    sims = SimpleNamespace(
        affine_newton_tolerance=1.0e-12,
        affine_max_newton_iteration=2,
        affine_max_step=1.0,
    )
    with pytest.raises(RuntimeError, match="synthetic line-search failure"):
        engine._step_fully_implicit(sims, engine.state.pack(), dt=0.01)
    assert engine.state.accepted is None
    np.testing.assert_allclose(assembled_configurations[-2], 7.0)
    np.testing.assert_allclose(assembled_configurations[-1], 0.0)


def test_affine_fully_implicit_predictor_rejects_zero_ccd_step():
    engine = AffineBodyEngine()
    engine.state = _DispatchState()
    engine.state.tilde_y.fill(1.0)
    engine.operator = SimpleNamespace(initialize_contact_damping=lambda _y, _hat: None)
    engine._ccd_step_size = lambda *_args: 0.0
    rollback = []
    engine._assemble_system = lambda _sims, y, need_matrix=True: (
        rollback.append(np.asarray(y).copy()) or 0.0,
        np.zeros(12, dtype=np.float64),
    )
    with pytest.raises(RuntimeError, match="non-positive step"):
        engine._step_fully_implicit(SimpleNamespace(), engine.state.pack(), dt=0.01)
    assert engine.state.accepted is None
    np.testing.assert_allclose(rollback[-1], 0.0)


def test_affine_cuda_failure_rollback_stays_on_device_path():
    engine = AffineBodyEngine()
    events = []
    engine.operator = SimpleNamespace(device_restore_step_start=lambda: events.append("restore-device"))

    def fail(*_args):
        engine.last_device_nonlinear_path = True
        raise RuntimeError("synthetic CUDA failure")

    engine._step_fully_implicit_impl = fail
    engine._assemble_system_device = lambda _sims, need_matrix=False: events.append("assemble-device")
    engine._assemble_system = lambda *_args, **_kwargs: pytest.fail(
        "CUDA rollback transferred the nonlinear vector through NumPy"
    )
    with pytest.raises(RuntimeError, match="synthetic CUDA failure"):
        engine._step_fully_implicit(SimpleNamespace(), np.zeros(12), dt=0.01)
    assert events == ["restore-device", "assemble-device"]


def test_affine_cuda_nonlinear_hot_loop_has_no_full_vector_numpy_roundtrip():
    methods = (
        AffineBodyEngine._assemble_system_device,
        AffineBodyEngine._solve_hash_direction_device,
        AffineBodyEngine._step_lagged_cuda,
        AffineBodyEngine._solve_lagged_inner_cuda,
        AffineBodyEngine._line_search_cuda,
        AffineBodyEngine._step_fully_implicit_cuda,
        AffineBodyEngine._fully_implicit_line_search_cuda,
    )
    for method in methods:
        source = inspect.getsource(method)
        assert ".to_numpy(" not in source, method.__qualname__
        assert ".from_numpy(" not in source, method.__qualname__


def test_affine_fully_implicit_driver_converges_and_accepts_once():
    engine = AffineBodyEngine()
    engine.state = _DispatchState()
    engine.operator = SimpleNamespace(initialize_contact_damping=lambda _y, _hat: None)
    target = np.full(12, 0.125, dtype=np.float64)
    engine._assemble_system = lambda _sims, y, need_matrix=True: (
        1.0e20,
        np.asarray(y, dtype=np.float64) - target,
    )
    engine._solve_direction = lambda _sims, residual: -np.asarray(residual)
    engine._fully_implicit_jacobian_direction = lambda _sims, direction: np.asarray(direction)
    engine._ccd_step_size = lambda *_args: 1.0
    sims = SimpleNamespace(
        affine_newton_tolerance=1.0e-12,
        affine_max_newton_iteration=4,
        affine_max_step=1.0,
        affine_line_search_max_iteration=4,
        affine_fully_implicit_armijo=1.0e-4,
        affine_fully_implicit_line_search_contraction=0.5,
    )
    engine._step_fully_implicit(sims, engine.state.pack(), dt=0.01)
    assert engine.last_inner_converged
    assert engine.last_friction_converged
    assert engine.last_newton_iterations == 1
    np.testing.assert_allclose(engine.state.accepted.reshape(-1), target)


def test_affine_fully_implicit_force_tolerance_is_not_velocity_tolerance():
    sims = SimpleNamespace(
        affine_newton_tolerance=7.0,
        affine_fully_implicit_force_atol=2.0e-6,
        affine_fully_implicit_force_rtol=3.0e-4,
    )

    assert AffineBodyEngine._fully_implicit_force_target(sims, 10.0) == pytest.approx(3.002e-3)


def test_affine_fully_implicit_selects_nonsymmetric_linear_backends(monkeypatch):
    module = importlib.import_module("src.dem.engines.AffineBodyEngine")
    created = []

    class _DummyMatrix:
        def __init__(self, *args, **kwargs):
            created.append(("coo", args, kwargs))

    class _DummyTriplet:
        def __init__(self, *args, **kwargs):
            created.append(("hash", args, kwargs))

    operator = SimpleNamespace(
        fully_implicit=True,
        max_coo_entries=144,
        max_hash_triplets=512,
        dof=12,
        bind_coo_matrix=lambda _matrix: None,
        bind_hash_triplet=lambda _matrix: None,
    )
    monkeypatch.setattr(module, "CoordinateSparseMatrix", _DummyMatrix)
    monkeypatch.setattr(module, "BuildTriplet", _DummyTriplet)
    monkeypatch.setattr(module, "current_cfg", lambda: SimpleNamespace(arch=ti.cuda))

    coo_engine = AffineBodyEngine()
    coo_engine.operator = operator
    coo_engine._ensure_linear_solver(SimpleNamespace(affine_assemble_type="COO"))
    assert created[-1][2]["symmetry"] is False

    hash_engine = AffineBodyEngine()
    hash_engine.operator = operator
    hash_engine._ensure_linear_solver(SimpleNamespace(affine_assemble_type="HashTriplet"))
    assert created[-1][2]["solver"] == "BiCGSTAB"
    assert created[-1][2]["matrix_symmetric"] is False

    # Official lagged IPC spectrally clamps every non-inertial local block,
    # so its symmetric global system is SPD and uses PCG.
    operator.fully_implicit = False
    lagged_hash_engine = AffineBodyEngine()
    lagged_hash_engine.operator = operator
    lagged_hash_engine._ensure_linear_solver(SimpleNamespace(affine_assemble_type="HashTriplet"))
    assert created[-1][2]["solver"] == "PCG"
    assert created[-1][2]["matrix_symmetric"] is True

    lagged_coo_engine = AffineBodyEngine()
    lagged_coo_engine.operator = operator
    lagged_coo_engine._ensure_linear_solver(SimpleNamespace(affine_assemble_type="COO"))
    assert created[-1][2]["symmetry"] is True


@pytest.mark.parametrize(
    ("fully_implicit", "expected"),
    [
        # A one-body solve has no body/body contact. The 256-entry floor covers
        # its 52 base blocks and five wall contacts (2 * 16 blocks each).
        (True, 256),
        (False, 256),
    ],
)
def test_affine_hash_capacity_bounds_raw_contact_scatter(monkeypatch, fully_implicit, expected):
    monkeypatch.setenv("GT_AFFINE_HASH_TRIPLET_SAFETY", "1.0")
    operator = object.__new__(TaichiAffineBodyOperator)
    operator.fully_implicit = fully_implicit
    operator.neighbor = SimpleNamespace(candidate_capacity=2, edge_candidate_capacity=3)
    operator.wall_num = 1
    operator.vertex_num = 5
    operator.body_num = 1
    operator.dof = 12
    operator.contact_damping_stiffness = 0.0
    sims = SimpleNamespace(wall_coordination_number=1)
    assert operator._estimate_hash_triplet_capacity(sims) == expected


def test_affine_hash_capacity_uses_configured_contact_block_capacity(monkeypatch):
    monkeypatch.setenv("GT_AFFINE_HASH_TRIPLET_SAFETY", "1.0")
    operator = object.__new__(TaichiAffineBodyOperator)
    operator.fully_implicit = False
    operator.neighbor = SimpleNamespace(candidate_capacity=2, edge_candidate_capacity=3)
    operator.wall_num = 0
    operator.vertex_num = 10
    operator.body_num = 2
    operator.dof = 24
    operator.contact_damping_stiffness = 0.0
    sims = SimpleNamespace(
        wall_coordination_number=0,
        affine_contact_block_capacity=128,
    )
    assert operator._estimate_hash_triplet_capacity(sims, safety_factor=1.0, minimum_capacity=1) == 872


def test_affine_hash_overflow_is_reported_before_linear_solve():
    engine = AffineBodyEngine()
    engine.operator = SimpleNamespace(dof=3, fully_implicit=True)

    class _OverflowTriplet:
        def __init__(self):
            self.solve_called = False

        def finalize_taichi_assembly(self):
            raise RuntimeError("BuildTriplet buffer overflow")

        def solve(self, **_kwargs):
            self.solve_called = True
            pytest.fail("linear solve ran after triplet overflow")

    engine.hash_triplet = _OverflowTriplet()
    sims = SimpleNamespace(
        affine_linear_tolerance=1.0e-9,
        affine_linear_max_iteration=20,
    )
    with pytest.raises(RuntimeError, match="buffer overflow"):
        engine._solve_hash_triplet(sims, np.ones(3))
    assert not engine.hash_triplet.solve_called


def test_affine_fully_implicit_plane_and_facet_wall_ccd_prevent_crossing(tmp_path):
    os.environ.setdefault("GEOTAICHI_REAL_DTYPE", "float64")
    ti.init(arch=ti.cpu, default_fp=ti.f64, debug=False, offline_cache=False)
    dem, unused_hat = _build_affine_contact(
        tmp_path / "fully-implicit-wall-ccd", with_wall=True, friction_mode="fully_implicit"
    )
    operator = dem.enginer.operator
    state = dem.enginer.state
    y = state.pack()
    direction = np.zeros_like(y).reshape((-1, 3))
    direction[:, 2] = -1.0
    direction = direction.reshape(-1)
    wall_height = 0.75

    # The unit trial crosses the plane; CCD must leave a strictly positive
    # signed gap.  This same init_step_size path is used for the predictor and
    # every residual line-search trial.
    operator.wall_type[0] = 0
    plane_alpha = operator.init_step_size(y, direction, ccd_type="ccd", eta=0.2)
    x0 = operator.x.to_numpy()[: operator.vertex_num]
    dx = operator.dx.to_numpy()[: operator.vertex_num]
    assert np.min(x0[:, 2] + dx[:, 2] - wall_height) < 0.0
    assert 0.0 < plane_alpha < 1.0
    assert np.min(x0[:, 2] + plane_alpha * dx[:, 2] - wall_height) > 0.0

    # Reuse the allocated wall slot as a large static facet.  The continuous
    # point-triangle query must likewise stop before the swept point reaches
    # the triangle, rather than accepting a finite endpoint on its far side.
    operator.wall_type[0] = 1
    operator.wall_v0[0] = ti.Vector([-10.0, -10.0, wall_height])
    operator.wall_v1[0] = ti.Vector([10.0, -10.0, wall_height])
    operator.wall_v2[0] = ti.Vector([0.0, 10.0, wall_height])
    facet_alpha = operator.init_step_size(y, direction, ccd_type="ccd", eta=0.2)
    x0 = operator.x.to_numpy()[: operator.vertex_num]
    dx = operator.dx.to_numpy()[: operator.vertex_num]
    assert 0.0 < facet_alpha < 1.0
    assert np.min(x0[:, 2] + facet_alpha * dx[:, 2] - wall_height) > 0.0

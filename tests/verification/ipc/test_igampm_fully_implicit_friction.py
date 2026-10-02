"""Verification of fully-implicit IGA-MPM friction derivatives and solves."""

from importlib import import_module
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti
import importlib

from src.igampm.engines import Engine
from src.igampm.ContactManager import _validate_iga_friction_configuration
from tests.helpers.igampm_fully_implicit_friction_reference import (
    point_nurbs_friction_residual_jacobian,
)


def test_exact_point_nurbs_fully_implicit_device_path_requires_full_bicgstab_state():
    """Never route device FI through a symmetric or incomplete approximation."""
    engine_module = importlib.import_module("src.igampm.engines")
    engine = object.__new__(engine_module.Engine)
    engine.friction_mode = "fully_implicit"
    engine.monolithic_hash_matrix = SimpleNamespace(
        solver="BiCGSTAB",
        symmetric=False,
        matrix_symmetric=False,
    )
    engine.monolithic_tangent_product = None
    engine.iga = SimpleNamespace(assemble_type="Hash", hash_matrix=object())
    engine.mpm = SimpleNamespace(hash_matrix=object())
    assert engine._device_monolithic_available(include_friction=True) is False
    assert engine._device_monolithic_available(include_friction=False) is False

    engine.monolithic_tangent_product = object()
    assert engine._device_monolithic_available(include_friction=True) is True
    assert engine._device_monolithic_available(include_friction=False) is True

    engine.monolithic_hash_matrix.solver = "PCG"
    assert engine._device_monolithic_available(include_friction=True) is False


def test_fully_implicit_device_hot_loop_has_no_full_vector_host_transfer():
    """Keep Newton/Krylov/Armijo vectors resident in Taichi fields."""
    hot_functions = (
        Engine.assemble_monolithic_newton_system,
        Engine._assemble_device_physical_tangent_product,
        Engine.fully_implicit_residual_armijo_device,
        Engine._solve_fully_implicit_newton_device,
        Engine._solve_monolithic_linear_system,
        Engine.assemble_fully_implicit_friction_system_taichi,
    )
    source = "\n".join(inspect.getsource(function) for function in hot_functions)
    assert ".to_numpy(" not in source
    assert ".from_numpy(" not in source
    assert "solve_flat_system" in source
    assert "fully_implicit_residual_armijo_device" in source


from src.nurbs.NurbsBasis import (
    NurbsBasis2ndDers1d,
    NurbsBasis2ndDers2d,
    NurbsBasisInterpolations2ndDers1d,
    NurbsBasisInterpolations2ndDers2d,
)
from src.physics_model.contact_model.ipc.NurbsContact import closest_curve_point_py


def _friction_parameters():
    return {
        "area": 0.7,
        "dhat": 0.4,
        "dmin": 0.0,
        "kappa": 3.0,
        "use_physical_barrier": False,
        "mu_dynamic": 0.35,
        "mu_static": 0.62,
        "mu_viscous": 0.03,
        "stribeck_velocity": 0.45,
        "epsv": 0.025,
        "friction_profile": "quadratic",
    }


def _fd_jacobian(function, value, step=2.0e-7):
    result = np.empty((value.size, value.size), dtype=np.float64)
    for column in range(value.size):
        perturbation = np.zeros_like(value)
        perturbation[column] = step
        result[:, column] = (function(value + perturbation) - function(value - perturbation)) / (2.0 * step)
    return result


def _refine_curve_parameter(knots, weights, control, point, seed):
    parameter = float(seed)
    for _ in range(20):
        position, tangent, curvature = NurbsBasisInterpolations2ndDers1d(parameter, 2, knots, control, weights)
        residual = position - point
        tangent = tangent[0]
        curvature = curvature[0]
        gradient = float(np.dot(residual, tangent))
        hessian = float(np.dot(tangent, tangent) + np.dot(residual, curvature))
        update = gradient / hessian
        parameter = float(np.clip(parameter - update, 0.0, 1.0))
        if abs(update) < 1.0e-14:
            break
    return parameter


def test_curve_fully_implicit_jacobian_differentiates_closest_parameter():
    knots = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    weights = np.asarray([1.0, 0.83, 1.12])
    reference_control = np.asarray([[-0.8, 0.02], [0.0, 0.31], [0.9, -0.04]])
    reference_point = np.asarray([0.18, -0.105])
    control_dofs = np.arange(6, dtype=np.int64).reshape((3, 2))
    point_dofs = np.asarray([[6, 7]], dtype=np.int64)
    velocity_offset = np.asarray([-0.04, 0.03, 0.02, -0.01, 0.06, 0.025, 0.17, -0.035])
    velocity_scale = np.asarray([7.0, 6.0, 8.0, 7.0, 6.5, 8.5, 9.0, 7.5])
    displacement = np.asarray([0.006, -0.004, -0.003, 0.005, 0.004, -0.002, 0.008, 0.003])

    def evaluate(value, with_jacobian=False):
        control = reference_control + value[:6].reshape((3, 2))
        point = reference_point + value[6:]
        seed, _ = closest_curve_point_py(2, knots, control, weights, point)
        parameter = _refine_curve_parameter(knots, weights, control, point, seed)
        shape, derivative, second = NurbsBasis2ndDers1d(parameter, 2, knots, weights)
        force, matrix, diagnostics = point_nurbs_friction_residual_jacobian(
            control_positions=control,
            point_position=point,
            shape_values=shape,
            shape_first_derivatives=derivative[:, None],
            shape_second_derivatives=second[:, None, None],
            control_dofs=control_dofs,
            point_dofs=point_dofs,
            point_weights=[1.0],
            endpoint_velocity=velocity_offset + velocity_scale * value,
            endpoint_velocity_displacement_scale=velocity_scale,
            free_parameters=[True],
            **_friction_parameters(),
        )
        return (force, matrix, diagnostics) if with_jacobian else force

    force, matrix, diagnostics = evaluate(displacement, with_jacobian=True)
    finite_difference = _fd_jacobian(evaluate, displacement)
    resultant = np.sum(force[control_dofs], axis=0) + np.sum(force[point_dofs], axis=0)
    assert np.linalg.norm(diagnostics["parameter_jacobian"]) > 1.0e-3
    assert np.linalg.norm(force) > 0.0
    assert np.allclose(resultant, 0.0, rtol=0.0, atol=1.0e-12)
    assert np.allclose(-matrix, finite_difference, rtol=4.0e-5, atol=2.0e-6)
    assert np.linalg.norm(matrix - matrix.T, ord=np.inf) > 1.0e-4


def _refine_surface_parameter(knots, weights, control, point, seed):
    parameter = np.asarray(seed, dtype=np.float64).copy()
    for _ in range(30):
        position, tangent, curvature = NurbsBasisInterpolations2ndDers2d(
            parameter[0], parameter[1], 2, 2, knots, knots, control, weights
        )
        residual = position - point
        tangent_u, tangent_v = tangent
        curvature_uu, curvature_vv, curvature_uv = curvature
        gradient = np.asarray([np.dot(residual, tangent_u), np.dot(residual, tangent_v)])
        hessian = np.asarray(
            [
                [
                    np.dot(tangent_u, tangent_u) + np.dot(residual, curvature_uu),
                    np.dot(tangent_u, tangent_v) + np.dot(residual, curvature_uv),
                ],
                [
                    np.dot(tangent_v, tangent_u) + np.dot(residual, curvature_uv),
                    np.dot(tangent_v, tangent_v) + np.dot(residual, curvature_vv),
                ],
            ]
        )
        update = np.linalg.solve(hessian, gradient)
        parameter = np.clip(parameter - update, 0.0, 1.0)
        if np.linalg.norm(update, ord=np.inf) < 1.0e-14:
            break
    return parameter


def test_surface_fully_implicit_jacobian_differentiates_two_parameters():
    knots = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    control = []
    weights = []
    for v in np.linspace(-0.7, 0.7, 3):
        for u in np.linspace(-0.8, 0.8, 3):
            control.append([u, v, 0.16 * (1.0 - u * u) + 0.05 * u * v])
            weights.append(1.0 + 0.08 * u - 0.04 * v)
    reference_control = np.asarray(control)
    weights = np.asarray(weights)
    reference_point = np.asarray([0.16, -0.11, -0.105])
    dof_count = 30
    control_dofs = np.arange(27, dtype=np.int64).reshape((9, 3))
    point_dofs = np.asarray([[27, 28, 29]], dtype=np.int64)
    rng = np.random.default_rng(91827)
    displacement = rng.normal(scale=1.5e-3, size=dof_count)
    velocity_offset = rng.normal(scale=0.045, size=dof_count)
    velocity_offset[27:] += np.asarray([0.14, -0.08, 0.035])
    velocity_scale = rng.uniform(5.0, 9.0, size=dof_count)

    def evaluate(value, with_jacobian=False):
        current_control = reference_control + value[:27].reshape((9, 3))
        point = reference_point + value[27:]
        parameter = _refine_surface_parameter(knots, weights, current_control, point, [0.6, 0.42])
        shape_data = NurbsBasis2ndDers2d(parameter[0], parameter[1], 2, 2, knots, knots, weights)
        shape, derivative_u, derivative_v, second_uu, second_vv, second_uv = shape_data
        first = np.column_stack([derivative_u, derivative_v])
        second = np.empty((9, 2, 2))
        second[:, 0, 0] = second_uu
        second[:, 1, 1] = second_vv
        second[:, 0, 1] = second_uv
        second[:, 1, 0] = second_uv
        force, matrix, diagnostics = point_nurbs_friction_residual_jacobian(
            control_positions=current_control,
            point_position=point,
            shape_values=shape,
            shape_first_derivatives=first,
            shape_second_derivatives=second,
            control_dofs=control_dofs,
            point_dofs=point_dofs,
            point_weights=[1.0],
            endpoint_velocity=velocity_offset + velocity_scale * value,
            endpoint_velocity_displacement_scale=velocity_scale,
            free_parameters=[True, True],
            **_friction_parameters(),
        )
        return (force, matrix, diagnostics) if with_jacobian else force

    force, matrix, diagnostics = evaluate(displacement, with_jacobian=True)
    finite_difference = _fd_jacobian(evaluate, displacement, step=1.0e-7)
    resultant = np.sum(force[control_dofs], axis=0) + np.sum(force[point_dofs], axis=0)
    assert np.linalg.matrix_rank(diagnostics["parameter_jacobian"]) == 2
    assert np.linalg.norm(force) > 0.0
    assert np.allclose(resultant, 0.0, rtol=0.0, atol=1.0e-12)
    assert np.allclose(-matrix, finite_difference, rtol=8.0e-5, atol=5.0e-6)
    assert np.linalg.norm(matrix - matrix.T, ord=np.inf) > 1.0e-4


def test_production_iga_mpm_friction_all_columns_match_force_fd(
    taichi_runtime,
    tmp_path,
):
    import src.igampm.config as config
    from src.igampm import IGAMPM
    from tests.integration.igampm.test_igampm_coupling_friction_2d import (
        _build_iga_rectangle,
        _build_mpm_particle,
        _set_coupled_displacement,
    )

    config.set_dimension(2)
    iga = _build_iga_rectangle(tmp_path / "iga")
    mpm = _build_mpm_particle(tmp_path / "mpm")
    iga.dt = mpm.dt
    engine = IGAMPM(
        iga,
        mpm,
        kappa=1.0e4,
        dhat=0.08,
        dynamic_friction=0.4,
        static_friction=0.6,
        viscous_friction=0.01,
        stribeck_velocity=0.2,
        epsv=0.01,
        friction_profile="quadratic",
        activate_friction=True,
        friction_mode="fully_implicit",
    ).build()
    active_dof = iga.degree_of_freedom + mpm.active_dof
    rng = np.random.default_rng(4217)
    displacement = rng.normal(scale=2.0e-5, size=active_dof)
    displacement[iga.degree_of_freedom :: 2] += 1.0e-4

    def evaluate(value, with_matrix=False):
        _set_coupled_displacement(iga, mpm, value)
        engine.initialize_barrier(mpm.grid_disp, iga.grid_disp)
        engine.assemble_fully_implicit_friction_system_taichi(need_matrix=with_matrix)
        force = engine.friction_grad.to_numpy()[:active_dof].copy()
        if with_matrix:
            engine.friction_hash_matrix.finalize_taichi_assembly()
            matrix = engine.friction_hash_matrix.to_scipy(active_dof // 2).toarray()[:active_dof, :active_dof]
            return force, matrix
        return force

    force, matrix = evaluate(displacement, with_matrix=True)

    # Reassemble the production Taichi path and verify its stable sparse stream.
    # The independent oracle below is force finite difference; production no
    # longer exposes a NumPy/SciPy friction-assembly backend.
    _set_coupled_displacement(iga, mpm, displacement)
    engine.initialize_barrier(mpm.grid_disp, iga.grid_disp)
    engine.assemble_fully_implicit_friction_system_taichi(need_matrix=True)
    device_force = engine.friction_grad.to_numpy()[:active_dof].copy()
    engine.friction_hash_matrix.finalize_taichi_assembly()
    device_matrix = engine.friction_hash_matrix.to_scipy(active_dof // 2).toarray()[:active_dof, :active_dof]
    raw_count = int(engine.friction_hash_matrix.raw_non_diag_count[0])
    raw_i = engine.friction_hash_matrix.non_diag.blockI.to_numpy()[:raw_count].copy()
    raw_j = engine.friction_hash_matrix.non_diag.blockJ.to_numpy()[:raw_count].copy()

    # A second assembly must own exactly the same raw slots.  This is the
    # topology-stable stream consumed by the persistent GPU pattern cache.
    engine.assemble_fully_implicit_friction_system_taichi(need_matrix=True)
    assert int(engine.friction_hash_matrix.raw_non_diag_count[0]) == raw_count
    np.testing.assert_array_equal(
        engine.friction_hash_matrix.non_diag.blockI.to_numpy()[:raw_count],
        raw_i,
    )
    np.testing.assert_array_equal(
        engine.friction_hash_matrix.non_diag.blockJ.to_numpy()[:raw_count],
        raw_j,
    )
    engine.friction_hash_matrix.finalize_taichi_assembly()
    assert engine.friction_hash_matrix.non_diag.pattern_cache_statistics()["pattern_hits"] >= 1
    np.testing.assert_allclose(device_force, force, rtol=2.0e-10, atol=2.0e-11)
    np.testing.assert_allclose(device_matrix, matrix, rtol=2.0e-9, atol=2.0e-9)

    finite_difference = _fd_jacobian(evaluate, displacement, step=2.0e-7)
    scale = max(
        np.linalg.norm(matrix, ord=np.inf),
        np.linalg.norm(finite_difference, ord=np.inf),
        1.0,
    )
    assert engine.curr_friction_contact_num == 1
    assert np.linalg.norm(force) > 0.0
    iga_resultant = force[: iga.degree_of_freedom].reshape((-1, 2)).sum(axis=0)
    mpm_resultant = force[iga.degree_of_freedom :].reshape((-1, 2)).sum(axis=0)
    assert np.allclose(iga_resultant + mpm_resultant, 0.0, rtol=0.0, atol=1.0e-11)
    assert np.linalg.norm(finite_difference + matrix, ord=np.inf) / scale < 3.0e-5
    assert np.linalg.norm(matrix - matrix.T, ord=np.inf) > 1.0e-5


def test_production_monolithic_residual_probe_preserves_gpu_matrices(
    taichi_runtime,
    tmp_path,
):
    import src.igampm.config as config
    from src.igampm import IGAMPM
    from tests.integration.igampm.test_igampm_coupling_friction_2d import (
        _build_iga_rectangle,
        _build_mpm_particle,
    )

    config.set_dimension(2)
    iga = _build_iga_rectangle(tmp_path / "iga")
    mpm = _build_mpm_particle(tmp_path / "mpm")
    iga.dt = mpm.dt
    engine = IGAMPM(
        iga,
        mpm,
        kappa=1.0e4,
        dhat=0.08,
        dynamic_friction=0.4,
        static_friction=0.6,
        viscous_friction=0.01,
        stribeck_velocity=0.2,
        epsv=0.01,
        friction_profile="quadratic",
        activate_friction=True,
        friction_mode="fully_implicit",
    ).build()

    full = engine.assemble_monolithic_newton_system(include_friction=True, need_matrix=True)
    active_dof = full["active_dof"]
    full_residual = full["unconstrained_rhs"].to_numpy()[:active_dof].copy()
    matrices = (
        iga.hash_matrix,
        mpm.hash_matrix,
        engine.barrier_hash_matrix,
    )

    def snapshot(matrix):
        return (
            int(matrix.raw_non_diag_count[0]),
            int(matrix.non_diag.element_pair_num[0]),
            matrix.diag.to_numpy().copy(),
        )

    before = tuple(snapshot(matrix) for matrix in matrices)
    probe = engine.assemble_monolithic_newton_system(include_friction=True, need_matrix=False)
    probe_residual = probe["unconstrained_rhs"].to_numpy()[:active_dof].copy()
    after = tuple(snapshot(matrix) for matrix in matrices)

    assert probe["matrix"] is None
    np.testing.assert_allclose(
        probe_residual,
        full_residual,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    for before_matrix, after_matrix in zip(before, after):
        assert before_matrix[:2] == after_matrix[:2]
        assert np.array_equal(before_matrix[2], after_matrix[2])


def _run_production_3d_surface_fully_implicit_assembly_fd(tmp_path):
    """Exercise the complete 3D point--surface production assembly path."""
    import src.igampm.config as config
    from src.iga import Cube, ImplicitIGA, Primitives
    from src.igampm import IGAMPM
    from tests.integration.igampm.test_igampm_coupling_friction_2d import (
        _set_coupled_displacement,
    )
    from tests.integration.igampm.test_igampm_coupling_friction_3d import (
        _build_mpm_particle,
    )

    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)
    config.set_dimension(3)

    # A degree-one volume gives the smallest real 3D IGA production stencil:
    # four rational surface controls and eight linear-MPM grid nodes.  Unequal
    # weights make this exercise rational basis extraction, not just B-splines.
    cube = Cube()
    cube.set_parameters(start_point=[0.0, 0.0, 0.0], size=[1.0, 1.0, 0.2])
    cube.generate_knot_u(degree=1, num_ctrlpts=2)
    cube.generate_knot_v(degree=1, num_ctrlpts=2)
    cube.generate_knot_w(degree=1, num_ctrlpts=2)
    cube.generate_ctrlpts()
    cube.generate_weights()
    cube.weights = np.asarray(
        [1.0, 0.91, 1.08, 0.96, 1.03, 0.94, 1.06, 0.98],
        dtype=np.float64,
    )

    primitives = Primitives()
    primitives.append(cube, "cube")
    primitives.finialize()
    iga = ImplicitIGA(
        primitives=primitives,
        newmark=[1.0, 0.5, 1.0],
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        density=1000.0,
        gravity=[0.0, 0.0, 0.0],
        residual=1.0e-8,
        interval=1,
        step=1,
        degree=[1, 1, 1],
        path=str(tmp_path / "iga"),
    )
    mpm = _build_mpm_particle(tmp_path / "mpm")
    iga.dt = mpm.dt
    engine = IGAMPM(
        iga,
        mpm,
        kappa=1.0e4,
        dhat=0.08,
        dynamic_friction=0.4,
        static_friction=0.6,
        viscous_friction=0.01,
        stribeck_velocity=0.2,
        epsv=0.01,
        friction_profile="quadratic",
        activate_friction=True,
        friction_mode="fully_implicit",
        # Face zero is the bottom u-v NURBS surface adjacent to the particle.
        # Selecting it avoids unrelated far-side closest-point work while
        # retaining the real contact discovery and global scattering path.
        contact_surface_include=[(0, 0)],
    ).build()
    active_dof = iga.degree_of_freedom + mpm.active_dof
    rng = np.random.default_rng(73129)
    displacement = rng.normal(scale=1.5e-5, size=active_dof)
    displacement[iga.degree_of_freedom :: 3] += 8.0e-5

    def evaluate(value, with_matrix=False):
        _set_coupled_displacement(iga, mpm, value)
        engine.initialize_barrier(mpm.grid_disp, iga.grid_disp)
        # Every finite-difference sample traverses the production device
        # contact fields, rational surface basis lookup, active MPM support
        # mapping, and global nonsymmetric scattering.
        engine.assemble_fully_implicit_friction_system_taichi(need_matrix=with_matrix)
        force = engine.friction_grad.to_numpy()[:active_dof].copy()
        if with_matrix:
            engine.friction_hash_matrix.finalize_taichi_assembly()
            matrix = engine.friction_hash_matrix.to_scipy(active_dof // 3).toarray()[:active_dof, :active_dof]
            return force, matrix
        return force

    force, matrix = evaluate(displacement, with_matrix=True)
    base_contact_num = int(engine.curr_friction_contact_num)
    base_barrier_contacts = int(engine.curr_barrier_contact_num)
    base_distances = engine.contacts.distance.to_numpy().copy()
    assert base_contact_num == 1, {
        "barrier_contacts": base_barrier_contacts,
        "distances": base_distances,
        "point": mpm.p_temp.to_numpy()[: mpm.total_surface_num],
        "surface_keys": engine.contact_surface.surface_keys,
        "control_points": engine.contact_surface.control_points_hat.to_numpy(),
        "weights": engine.contact_surface.weights.to_numpy(),
        "control_ids": engine.contact_surface.control_points_id.to_numpy(),
        "parameters": engine.contacts.knot_value.to_numpy(),
        "force_norm": np.linalg.norm(force),
    }

    # Re-run the complete production Taichi point--surface assembler and check
    # every full generally nonsymmetric block for deterministic reassembly.
    _set_coupled_displacement(iga, mpm, displacement)
    engine.initialize_barrier(mpm.grid_disp, iga.grid_disp)
    engine.assemble_fully_implicit_friction_system_taichi(need_matrix=True)
    device_force = engine.friction_grad.to_numpy()[:active_dof].copy()
    engine.friction_hash_matrix.finalize_taichi_assembly()
    device_matrix = engine.friction_hash_matrix.to_scipy(active_dof // 3).toarray()[:active_dof, :active_dof]
    np.testing.assert_allclose(device_force, force, rtol=3.0e-10, atol=3.0e-11)
    np.testing.assert_allclose(device_matrix, matrix, rtol=3.0e-9, atol=3.0e-9)
    raw_count = int(engine.friction_hash_matrix.raw_non_diag_count[0])
    raw_i = engine.friction_hash_matrix.non_diag.blockI.to_numpy()[:raw_count].copy()
    raw_j = engine.friction_hash_matrix.non_diag.blockJ.to_numpy()[:raw_count].copy()
    engine.assemble_fully_implicit_friction_system_taichi(need_matrix=True)
    assert int(engine.friction_hash_matrix.raw_non_diag_count[0]) == raw_count
    np.testing.assert_array_equal(
        engine.friction_hash_matrix.non_diag.blockI.to_numpy()[:raw_count],
        raw_i,
    )
    np.testing.assert_array_equal(
        engine.friction_hash_matrix.non_diag.blockJ.to_numpy()[:raw_count],
        raw_j,
    )
    engine.friction_hash_matrix.finalize_taichi_assembly()
    assert engine.friction_hash_matrix.non_diag.pattern_cache_statistics()["pattern_hits"] >= 1

    finite_difference = _fd_jacobian(evaluate, displacement, step=1.0e-7)
    scale = max(
        np.linalg.norm(matrix, ord=np.inf),
        np.linalg.norm(finite_difference, ord=np.inf),
        1.0,
    )
    iga_resultant = force[: iga.degree_of_freedom].reshape((-1, 3)).sum(axis=0)
    mpm_resultant = force[iga.degree_of_freedom :].reshape((-1, 3)).sum(axis=0)

    assert np.linalg.norm(force) > 0.0
    assert np.allclose(iga_resultant + mpm_resultant, 0.0, rtol=0.0, atol=2.0e-11)
    assert np.linalg.norm(finite_difference + matrix, ord=np.inf) / scale < 8.0e-5
    assert np.linalg.norm(matrix - matrix.T, ord=np.inf) > 1.0e-5


@pytest.mark.isolated_dimension(3)
def test_production_3d_surface_fully_implicit_assembly_matches_all_columns_fd(
    tmp_path,
):
    _run_production_3d_surface_fully_implicit_assembly_fd(tmp_path)


class _ArrayField:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float64)

    def to_numpy(self):
        return self.value.copy()

    def from_numpy(self, value):
        self.value = np.asarray(value, dtype=np.float64).copy()


def _transaction_driver(raise_in_solve=False):
    events = []
    engine = SimpleNamespace(
        activate_fric=True,
        friction_mode="fully_implicit",
        mpm=SimpleNamespace(grid_disp=_ArrayField([0.0])),
    )
    engine.begin_implicit_ipc_step = lambda: events.append("begin")

    def solve(**kwargs):
        events.append("fully")
        if raise_in_solve:
            raise RuntimeError("nonlinear failure")
        return {"converged": True, "residual": 0.0}

    engine.solve_fully_implicit_friction_newton = solve
    engine.solve_lagged_friction_fixed_point = lambda *args, **kwargs: events.append("lagged")
    engine.accept_implicit_ipc_step = lambda: {
        "step": 0,
        "iga_increment": np.zeros(1),
        "mpm_increment": np.zeros(1),
        "minimum_distance": 0.1,
    }
    engine.abort_implicit_ipc_step = lambda: events.append("abort")
    return engine, events


def test_fully_implicit_substep_dispatches_without_lagged_outer_iteration():
    engine, events = _transaction_driver()
    result = Engine.implicit_ipc_substep(engine, include_friction=True)
    assert result["accepted"] is True
    assert events == ["begin", "fully"]


def test_fully_implicit_substep_rolls_back_on_nonlinear_failure():
    engine, events = _transaction_driver(raise_in_solve=True)
    with pytest.raises(RuntimeError, match="nonlinear failure"):
        Engine.implicit_ipc_substep(engine, include_friction=True)
    assert events == ["begin", "fully", "abort"]


def test_fully_implicit_solver_has_no_host_numerical_fallback():
    source = inspect.getsource(Engine.solve_fully_implicit_friction_newton)
    assert "_solve_fully_implicit_newton_device" in source
    assert "assemble_monolithic_newton_system" not in source
    assert "fully_implicit_residual_armijo(" not in source
    assert ".to_numpy(" not in source
    assert ".from_numpy(" not in source


@pytest.mark.parametrize(
    "parameters",
    (
        {"dynamic_friction": 0.3, "static_friction": 0.5},
        {"mu": 0.4, "dynamic_friction": 0.3},
        {"mu": 0.3, "viscous_friction": 0.01},
        {"mu": 0.3, "friction_profile": "stabilized"},
    ),
)
def test_iga_lagged_mode_rejects_fully_implicit_law_parameters(parameters):
    with pytest.raises(RuntimeError, match="requires friction_mode"):
        _validate_iga_friction_configuration({"friction_mode": "lagged", **parameters})


def test_iga_fully_implicit_mode_requires_positive_stribeck_transition():
    with pytest.raises(RuntimeError, match="positive stribeck_velocity"):
        _validate_iga_friction_configuration(
            {
                "friction_mode": "fully_implicit",
                "dynamic_friction": 0.3,
                "static_friction": 0.5,
                "stribeck_velocity": 0.0,
            }
        )


@pytest.mark.parametrize("iterations", (0, -1, 2))
def test_iga_fully_implicit_mode_rejects_lagged_outer_iterations(iterations):
    with pytest.raises(RuntimeError, match="only defined for lagged"):
        _validate_iga_friction_configuration(
            {
                "friction_mode": "fully_implicit",
                "friction_iterations": iterations,
            }
        )


def test_igampm_rejects_friction_mode_hot_switch_after_build(monkeypatch):
    main_module = import_module("src.igampm.mainIGAMPM")
    old_contactor = SimpleNamespace(contact_model="IPC", friction_mode="lagged")
    wrapper = SimpleNamespace(
        contact_kwargs={"friction_mode": "lagged"},
        contactor=old_contactor,
        engine=SimpleNamespace(
            contactor=old_contactor,
            friction_mode="lagged",
            implicit_step_in_progress=False,
        ),
        sims=SimpleNamespace(contact_model="IPC"),
    )
    monkeypatch.setattr(
        main_module,
        "ContactManager",
        lambda **kwargs: SimpleNamespace(contact_model="IPC", friction_mode=kwargs["friction_mode"]),
    )
    with pytest.raises(RuntimeError, match="cannot be changed after build"):
        main_module.IGAMPM.choose_contact_model(wrapper, friction_mode="fully_implicit")
    assert wrapper.contactor is old_contactor
    assert wrapper.contact_kwargs == {"friction_mode": "lagged"}


def test_set_configuration_uses_guarded_contact_reconfiguration(monkeypatch):
    main_module = import_module("src.igampm.mainIGAMPM")
    calls = []
    wrapper = SimpleNamespace(
        sims=SimpleNamespace(set_configuration=lambda **kwargs: calls.append(("sims", kwargs))),
        engine=None,
    )

    def choose_contact_model(contact_model="IPC", **kwargs):
        calls.append(
            (
                "contact",
                {"contact_model": contact_model, **kwargs},
            )
        )

    wrapper.choose_contact_model = choose_contact_model
    result = main_module.IGAMPM.set_configuration(
        wrapper,
        dimension=3,
        contact_model="IPC",
        activate_friction=True,
        log=False,
    )

    assert result is wrapper
    assert calls == [
        (
            "sims",
            {
                "dimension": 3,
                "coupling_scheme": "IGAMPM",
                "contact_model": "IPC",
                "activate_friction": True,
                "axisymmetric": False,
                "axis_offset": 0.0,
            },
        ),
        (
            "contact",
            {"contact_model": "IPC", "activate_friction": True},
        ),
    ]


def test_igampm_rejects_contact_model_hot_switch_after_build(monkeypatch):
    main_module = import_module("src.igampm.mainIGAMPM")
    old_contactor = SimpleNamespace(contact_model="IPC", friction_mode="lagged")
    wrapper = SimpleNamespace(
        contact_kwargs={"friction_mode": "lagged"},
        contactor=old_contactor,
        engine=SimpleNamespace(
            contactor=old_contactor,
            friction_mode="lagged",
            implicit_step_in_progress=False,
        ),
        sims=SimpleNamespace(contact_model="IPC"),
    )
    monkeypatch.setattr(
        main_module,
        "ContactManager",
        lambda **kwargs: SimpleNamespace(contact_model="DEM", friction_mode="dem"),
    )

    with pytest.raises(RuntimeError, match="contact_model cannot be changed"):
        main_module.IGAMPM.choose_contact_model(wrapper, contact_model="DEM")

    assert wrapper.contactor is old_contactor
    assert wrapper.engine.contactor is old_contactor
    assert wrapper.contact_kwargs == {"friction_mode": "lagged"}


def test_igampm_rejects_dimension_hot_switch_after_build():
    main_module = import_module("src.igampm.mainIGAMPM")
    wrapper = SimpleNamespace(
        sims=SimpleNamespace(dimension=2),
        engine=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="dimension cannot be changed"):
        main_module.IGAMPM.set_configuration(wrapper, dimension=3)

    assert wrapper.sims.dimension == 2

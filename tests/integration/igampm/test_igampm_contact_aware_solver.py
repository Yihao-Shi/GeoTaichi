"""Integration checks for contact-aware monolithic IGA-MPM solves."""

import inspect
import numpy as np
import pytest
import taichi as ti
from types import SimpleNamespace

import src.igampm.config as config


@pytest.fixture(autouse=True)
def _isolated_taichi_runtime(taichi_runtime):
    config.set_dimension(2)


@pytest.mark.parametrize("forcing", [None, 1.0e-3])
def test_monolithic_hash_solver_receives_relative_tolerance(forcing):
    from src.igampm.engines.ImplicitEngine import ImplicitEngineMixin

    class Matrix:
        def solve_flat_system(self, *args, **kwargs):
            self.kwargs = kwargs
            return {"converged": True}

    engine = object.__new__(ImplicitEngineMixin)
    engine.assemble_type = "HashTriplet"
    engine.monolithic_hash_matrix = Matrix()
    engine.monolithic_rhs = object()
    engine.monolithic_correction = object()
    engine.monolithic_linear_solver_tolerance = 1.0e-9
    engine.monolithic_linear_solver_relative_tolerance = 2.0e-7
    engine.monolithic_linear_solver_max_iters = 123

    system = {"active_nodes": 4, "active_dof": 8}
    if forcing is not None:
        system["linear_relative_tolerance"] = forcing
    engine._solve_monolithic_linear_system(system)

    assert engine.monolithic_hash_matrix.kwargs["tol"] == 1.0e-9
    assert engine.monolithic_hash_matrix.kwargs["rel_tol"] == (2.0e-7 if forcing is None else forcing)
    assert engine.monolithic_hash_matrix.kwargs["maxiter"] == 123


def test_inexact_forcing_tightens_with_nonlinear_progress():
    from src.igampm.engines.ImplicitEngine import ImplicitEngineMixin

    engine = object.__new__(ImplicitEngineMixin)
    engine.monolithic_linear_solver_relative_tolerance = 1e-7
    assert engine._newton_linear_tolerance(10.0, None) == 0.01
    assert engine._newton_linear_tolerance(10.0, 0.0) == 0.01
    assert engine._newton_linear_tolerance(10.0, 100.0) == 0.01
    assert engine._newton_linear_tolerance(1.0, 100.0) == pytest.approx(9e-4)
    assert engine._newton_linear_tolerance(1e-10, 100.0) == 1e-7


@pytest.mark.parametrize("accepted", [True, False])
@pytest.mark.parametrize("prepare_contacts", [True, False])
def test_armijo_accepts_a_trial_within_energy_roundoff(accepted, prepare_contacts):
    from src.igampm.engines.ImplicitEngine import ImplicitEngineMixin

    engine = object.__new__(ImplicitEngineMixin)
    engine.armijo_c1 = 1.0e-4
    engine.armijo_max_backtracks = 2
    engine.line_search_energy_rtol = 1.0e-12
    engine.line_search_energy_atol = 1.0e-14
    engine.contact_ccd_safety = 0.9
    engine.contact_ccd_min_step = 1.0e-12
    engine.mpm = SimpleNamespace(grid_disp_temp=object())
    engine.iga = SimpleNamespace(grid_disp_temp=object())
    engine._set_device_trial_displacements = lambda alpha: None
    queries = []
    syncs = []
    engine.initialize_barrier = lambda mpm, iga: queries.append((mpm, iga))
    engine._conservative_contact_step_device_impl = lambda **kwargs: 1.0
    engine._accept_device_trial_displacements = lambda: None
    engine._sync_device_trial_displacements = lambda: syncs.append("accepted")
    engine._synchronize_device_trial_state_with_accepted = lambda: syncs.append("restored")
    engine.minimum_contact_distance = lambda: 0.25

    energies = iter([1.0, 1.0 if accepted else 2.0, 2.0])
    result = engine.contact_aware_armijo_device(
        -1.0e-10,
        energy_function=lambda current: next(energies),
        prepare_contacts=prepare_contacts,
    )

    assert result["accepted"] is accepted
    assert result["step"] == (1.0 if accepted else 0.0)
    assert len(queries) == (1 if accepted else 2) + int(prepare_contacts)
    assert syncs == (["accepted"] if accepted else ["restored"])


def test_linear_solve_tolerance_does_not_relax_prescribed_displacements():
    from src.igampm.engines.ImplicitEngine import ImplicitEngineMixin

    @ti.data_oriented
    class DirectionEngine(ImplicitEngineMixin):
        pass

    engine = object.__new__(DirectionEngine)
    engine.iga = SimpleNamespace(degree_of_freedom=4, incre_resolution=ti.field(ti.f64, shape=4))
    engine.mpm = SimpleNamespace(incre_resolution=ti.field(ti.f64, shape=6))
    engine.monolithic_correction = ti.field(ti.f64, shape=10)
    engine.monolithic_fixed = ti.field(ti.i32, shape=10)
    engine.monolithic_fixed_correction = ti.field(ti.f64, shape=10)
    approximate = np.asarray([2e-6, 0, 4e-6, 3e-5, -2e-6, 0, 7e-6, 1e-6, 99, 99], dtype=np.float64)
    prescribed = np.zeros(10, dtype=np.float64)
    prescribed[[1, 5]] = [-5e-5, 3e-3]
    fixed = np.zeros(10, dtype=np.int32)
    fixed[[1, 5]] = 1
    engine.monolithic_correction.from_numpy(approximate)
    engine.monolithic_fixed.from_numpy(fixed)
    engine.monolithic_fixed_correction.from_numpy(prescribed)
    engine._split_device_monolithic_correction(4)

    expected = approximate.copy()
    expected[[1, 5]] = prescribed[[1, 5]]
    np.testing.assert_array_equal(engine.iga.incre_resolution.to_numpy(), expected[:4])
    np.testing.assert_array_equal(engine.mpm.incre_resolution.to_numpy(), np.r_[expected[4:8], 0, 0])


def test_monolithic_newton_does_not_solve_an_already_balanced_rhs():
    from src.igampm.engines.ImplicitEngine import ImplicitEngineMixin

    engine = object.__new__(ImplicitEngineMixin)
    engine.nonassociated_newton = False
    engine.monolithic_force_atol = 1.0e-10
    engine.monolithic_force_rtol = 5.0e-4
    engine.monolithic_dirichlet_tolerance = 1.0e-12
    engine.is_semi = False
    engine.semi_contact_converged = lambda: True

    def residual_only(**kwargs):
        assert kwargs["need_matrix"] is False, "an equilibrated RHS must not assemble a Hessian"
        return {"active_dof": 3}

    engine.assemble_monolithic_newton_system = residual_only
    engine._device_monolithic_free_rhs_norm = lambda active_dof: 4.3e-7
    engine._device_dirichlet_residual = lambda active_dof: 0.0
    engine._solve_monolithic_linear_system = lambda *args, **kwargs: pytest.fail(
        "an equilibrated RHS must not be sent to PCG"
    )

    result = engine._solve_monolithic_newton_device(
        include_friction=False,
        max_iterations=2,
        tolerance=1.0e-3,
        energy_function=None,
        verbose=False,
    )

    assert result["converged"] is True
    assert result["convergence_reason"] == "force_residual"
    assert result["force_residual"] == pytest.approx(4.3e-7)


@pytest.mark.parametrize(
    ("has_incremental_potential", "prescribed_motion"),
    [(False, False), (True, False), (True, True)],
)
def test_plastic_monolithic_newton_selects_material_line_search(has_incremental_potential, prescribed_motion):
    from src.igampm.engines.ImplicitEngine import ImplicitEngineMixin

    engine = object.__new__(ImplicitEngineMixin)
    engine.nonassociated_newton = False
    engine.mpm_has_plastic_history = True
    engine.is_semi = False
    engine.semi_contact_converged = lambda: True
    engine.monolithic_force_atol = 1.0e-10
    engine.monolithic_force_rtol = 5.0e-4
    engine.monolithic_dirichlet_tolerance = 1.0e-12
    engine.monolithic_solver_name = "PCG"
    engine.iga = SimpleNamespace(dt=0.1)
    engine.mpm = SimpleNamespace(dt=0.1, material=SimpleNamespace(has_incremental_potential=has_incremental_potential))
    force_residuals = iter((2.0, 0.0))
    engine.assemble_monolithic_newton_system = lambda **kwargs: {
        "active_dof": 3,
        "active_mpm_dof": 3,
    }
    engine._device_monolithic_free_rhs_norm = lambda active_dof: next(force_residuals)
    dirichlet_residuals = iter((0.001 if prescribed_motion else 0.0, 0.0))
    engine._device_dirichlet_residual = lambda active_dof: next(dirichlet_residuals)
    engine._solve_monolithic_linear_system = lambda *args, **kwargs: {
        "converged": True,
        "residual": 0.0,
        "iterations": 1,
    }
    engine._split_device_monolithic_correction = lambda active_dof: None
    engine._device_monolithic_correction_residual = lambda *args: 1.0
    engine._material_feasible_step_device = lambda: 1.0
    engine._assemble_device_physical_tangent_product = lambda **kwargs: -4.0
    engine._device_monolithic_directional_derivative = lambda active_dof: -2.0
    calls = []
    use_energy = has_incremental_potential and not prescribed_motion

    def residual_armijo(current_residual, merit_slope, **kwargs):
        assert not use_energy
        calls.append((current_residual, merit_slope, kwargs))
        return {"accepted": True, "step": 0.5, "residual": 0.1}

    def energy_armijo(directional_derivative, **kwargs):
        assert use_energy
        calls.append((directional_derivative, kwargs))
        return {"accepted": True, "step": 0.5, "energy": 0.1}

    engine.fully_implicit_residual_armijo_device = residual_armijo
    engine.contact_aware_armijo_device = energy_armijo

    result = engine._solve_monolithic_newton_device(
        include_friction=False,
        max_iterations=2,
        tolerance=1.0e-3,
        energy_function=None,
        verbose=False,
    )

    assert result["converged"] is True
    assert result["convergence_reason"] == "force_residual"
    if use_energy:
        assert calls[0][0] == pytest.approx(-2.0)
        assert calls[0][1]["include_friction"] is False
    else:
        assert calls[0][0] == pytest.approx(2.0)
        assert calls[0][1] == pytest.approx(-4.0)
        assert calls[0][2]["include_friction"] is False


def test_residual_merit_armijo_backtracks_a_failed_trial_query():
    from src.igampm.engines.FrictionEngine import FrictionEngineMixin

    engine = object.__new__(FrictionEngineMixin)
    engine.armijo_c1 = 1.0e-4
    engine.armijo_max_backtracks = 3
    engine.contact_ccd_min_step = 1.0e-12
    engine.fully_implicit_armijo_reduction = 0.5
    engine.mpm = SimpleNamespace(grid_disp=object())
    engine.iga = SimpleNamespace(grid_disp=object())
    accd_calls = []
    engine._conservative_contact_step_device_impl = lambda *args, **kwargs: (accd_calls.append((args, kwargs)) or 1.0)
    engine.conservative_contact_step_device = lambda **kwargs: pytest.fail(
        "the residual Armijo loop owns trial validation"
    )
    engine._sync_device_trial_displacements = lambda: None
    engine._set_device_current_from_trial_base = lambda alpha: None
    engine._restore_device_current_from_trial_base = lambda: None
    attempts = iter((RuntimeError("trial closest query failed"), {"active_dof": 1}))

    def assemble(**kwargs):
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    engine.assemble_monolithic_newton_system = assemble
    engine._device_physical_residual_squared = lambda active_dof: 0.25
    engine.initialize_barrier = lambda *args: None
    engine.minimum_contact_distance = lambda: 0.02

    result = engine.fully_implicit_residual_armijo_device(
        1.0,
        -1.0,
        initial_step=1.0,
        include_friction=False,
    )

    assert result["accepted"] is True
    assert result["step"] == pytest.approx(0.5)
    assert result["residual"] == pytest.approx(0.5)
    assert accd_calls == [((1.0, None, False), {"prepare_contacts": True})]


def _build_iga_rectangle(output_path):
    from src.iga import ImplicitIGA, Primitives, Rectangle

    rectangle = Rectangle()
    rectangle.set_parameters(start_point=[0.0, 0.0], size=[1.0, 0.2])
    rectangle.generate_knot_u(degree=2, num_ctrlpts=3)
    rectangle.generate_knot_v(degree=2, num_ctrlpts=3)
    rectangle.generate_ctrlpts()
    rectangle.generate_weights()
    primitives = Primitives()
    primitives.append(rectangle, "rectangle")
    primitives.finialize()
    return ImplicitIGA(
        primitives=primitives,
        newmark=[1.0, 0.5, 1.0],
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        density=1000.0,
        gravity=[0.0, 0.0],
        residual=1.0e-8,
        interval=1,
        step=1,
        degree=[2, 2],
        path=str(output_path),
    )


def _build_mpm_particle(output_path):
    from src.mpm.engines.direct.ImplicitULMPM import ImplicitULMPM
    from src.mpm.generator.Body import Body

    body = Body()
    body.add_particles(
        [[0.5, -0.031]],
        volume=1.0e-3,
        xmin=[-0.1, -0.1],
        xmax=[1.1, 0.3],
        boundary_ids=[0],
    )
    mpm = ImplicitULMPM(
        domain=[1.2, 0.5],
        dx=0.1,
        dt=1.0e-3,
        bodies=body,
        newmark=[1.0, 0.5, 1.0],
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        density=1000.0,
        gravity=[0.0, 0.0],
        residual=1.0e-8,
        interval=1,
        step=1,
        scale=1.0,
        line_search=False,
        shape_function="linear",
        visualize=False,
        path=str(output_path),
    )
    mpm.init_F0()
    mpm.mass_vec.fill(0.0)
    mpm.grid_reset()
    mpm.compute_shapefn()
    mpm.mass_vel_acc_p2g()
    mpm.find_active_node()
    mpm.prefix_sum_executor.run(mpm.node2dof)
    mpm.active_dof = mpm.set_active_dof()
    mpm.compute_nodal_vel_acc()
    return mpm


def test_compact_contact_candidates_preserve_barrier_and_distant_ccd(tmp_path):
    from src.igampm import IGAMPM

    iga = _build_iga_rectangle(tmp_path / "iga")
    mpm = _build_mpm_particle(tmp_path / "mpm")
    engine = IGAMPM(iga, mpm, kappa=1e4, dhat=0.08, dmin=0.005, barrier_nnz=20_000, project_pd=True).build()
    engine.begin_implicit_ipc_step()
    counts = engine.contact_candidate_count.to_numpy()
    assert 0 < counts.sum() < engine.contact_pair_count
    assert np.all(np.isfinite(engine.contacts.distance.to_numpy()[: engine.contact_pair_count]))
    engine.assemble_barrier_system()
    expected_force = engine.barrier_grad.to_numpy().copy()
    engine.barrier_hash_matrix.finalize_taichi_assembly()
    expected_matrix = engine.barrier_hash_matrix.to_scipy().toarray()

    # Replaying the full face/sample product is the uncompressed oracle.
    faces, samples = engine.contact_surface.num_surfaces, mpm.total_surface_num
    ids = np.arange(samples, dtype=np.int32)[None, :] * faces + np.arange(faces, dtype=np.int32)[:, None]
    engine.contact_candidates.from_numpy(ids)
    engine.contact_candidate_count.from_numpy(np.full(faces, samples, dtype=np.int32))
    engine.assemble_barrier_system()
    engine.barrier_hash_matrix.finalize_taichi_assembly()
    np.testing.assert_allclose(engine.barrier_grad.to_numpy(), expected_force, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(engine.barrier_hash_matrix.to_scipy().toarray(), expected_matrix, rtol=1e-13, atol=1e-13)

    # A presently inactive point can approach the solid during the proposed step.
    mpm.grid_disp.fill(0.0)
    moving = np.zeros(mpm.degree_of_freedom)
    moving[: mpm.active_dof].reshape(-1, 2)[:, 1] = -0.25
    mpm.grid_disp.from_numpy(moving)
    engine.initialize_barrier()
    assert engine.contact_candidate_count.to_numpy().sum() == 0
    moving[: mpm.active_dof].reshape(-1, 2)[:, 1] = 0.5
    mpm.incre_resolution.from_numpy(moving)
    iga.incre_resolution.fill(0.0)
    alpha = engine.conservative_contact_step_device(verify=True)
    assert 0.0 < alpha < 1.0


def test_newton_reuses_contact_queries_without_changing_solution(tmp_path, monkeypatch):
    from src.igampm import IGAMPM

    iga = _build_iga_rectangle(tmp_path / "iga-reuse")
    mpm = _build_mpm_particle(tmp_path / "mpm-reuse")
    engine = IGAMPM(iga, mpm, kappa=1e4, dhat=0.08, dmin=0.005, barrier_nnz=20_000, project_pd=True).build()
    engine.begin_implicit_ipc_step()
    query = engine.initialize_barrier
    assemble = engine.assemble_monolithic_newton_system
    armijo = engine.contact_aware_armijo_device
    queries = []

    def counted_query(*args, **kwargs):
        queries.append(1)
        return query(*args, **kwargs)

    monkeypatch.setattr(engine, "initialize_barrier", counted_query)
    reference = None
    for refresh in (True, False):

        def assemble_current(*args, **kwargs):
            if refresh:
                kwargs["prepare_contacts"] = True
            return assemble(*args, **kwargs)

        def search_current(*args, **kwargs):
            if refresh:
                kwargs["prepare_contacts"] = True
            return armijo(*args, **kwargs)

        monkeypatch.setattr(engine, "assemble_monolithic_newton_system", assemble_current)
        monkeypatch.setattr(engine, "contact_aware_armijo_device", search_current)
        iga.grid_disp.fill(0.0)
        mpm.grid_disp.fill(0.0)
        queries.clear()
        result = engine.solve_monolithic_newton(include_friction=False, max_iterations=30, tolerance=1e-10)
        assert result["converged"] and result["iterations"] > 0
        assert engine.minimum_contact_distance() > 0.005
        actual = (
            iga.grid_disp.to_numpy(),
            mpm.grid_disp.to_numpy(),
            engine.coupled_potential_energy(include_friction=False),
        )
        if refresh:
            reference = actual
            reference_queries = len(queries)
        else:
            for current, expected in zip(actual, reference):
                np.testing.assert_allclose(current, expected, rtol=1e-9, atol=1e-12)
            assert len(queries) < reference_queries
    engine.abort_implicit_ipc_step()


def test_conservative_point_nurbs_step_and_armijo_preserve_dmin(tmp_path):
    from src.igampm import IGAMPM

    iga = _build_iga_rectangle(tmp_path / "iga")
    mpm = _build_mpm_particle(tmp_path / "mpm")
    coupling = IGAMPM(
        iga,
        mpm,
        kappa=1.0e4,
        dhat=0.08,
        dmin=0.005,
        barrier_nnz=20_000,
        contact_ccd_safety=0.9,
    )
    engine = coupling.build()
    assert engine.friction_nnz_capacity == 1
    engine.initialize_barrier()

    # Every sample--boundary pair has a stored closest distance, including
    # pairs outside the barrier activation distance.
    expected_pairs = mpm.total_surface_num * engine.contact_surface.num_surfaces
    distances = engine.contacts.distance.to_numpy()[:expected_pairs]
    assert np.all(np.isfinite(distances))
    start_distance = float(np.min(distances))
    assert start_distance > 0.005

    monolithic = engine.assemble_monolithic_newton_system(include_friction=False)
    total_active = iga.degree_of_freedom + mpm.active_dof
    output_matrix = monolithic["matrix"].to_scipy(monolithic["active_nodes"])
    output_rhs = monolithic["rhs"].to_numpy()[:total_active]
    assert output_matrix.shape == (total_active, total_active)
    assert output_rhs.shape == (total_active,)
    assert np.all(np.isfinite(output_matrix.data))
    assert np.all(np.isfinite(output_rhs))
    zero_solve = engine.solve_monolithic_newton(
        include_friction=False,
        max_iterations=1,
        tolerance=1.0e-12,
        linear_solve=lambda matrix, rhs: np.zeros_like(rhs),
    )
    assert zero_solve["converged"] is True
    assert zero_solve["residual"] == 0.0

    correction = np.zeros(iga.degree_of_freedom + mpm.active_dof, dtype=np.float64)
    correction[iga.degree_of_freedom :].reshape((-1, 2))[:, 1] = 0.1
    alpha = engine.conservative_contact_step(correction)
    expected_bound = 0.9 * (start_distance - 0.005) / 0.1
    assert 0.0 < alpha <= expected_bound * (1.0 + 1.0e-10)

    iga_base = iga.grid_disp.to_numpy().copy()
    mpm_base = mpm.grid_disp.to_numpy().copy()
    iga_delta = correction[: iga.degree_of_freedom]
    mpm_delta = np.zeros(mpm.degree_of_freedom, dtype=np.float64)
    mpm_delta[: mpm.active_dof] = correction[iga.degree_of_freedom :]

    def quadratic_target(current):
        iga_error = current.iga.grid_disp_temp.to_numpy() - (iga_base + iga_delta)
        mpm_error = current.mpm.grid_disp_temp.to_numpy() - (mpm_base + mpm_delta)
        return 0.5 * (float(np.dot(iga_error, iga_error)) + float(np.dot(mpm_error, mpm_error)))

    result = engine.contact_aware_armijo(
        correction,
        directional_derivative=-float(np.dot(correction, correction)),
        energy_function=quadratic_target,
    )
    assert result["accepted"] is True
    assert 0.0 < result["step"] <= alpha * (1.0 + 1.0e-10)
    assert result["minimum_distance"] > 0.005
    assert np.allclose(iga.grid_disp.to_numpy(), iga_base)
    assert np.linalg.norm(mpm.grid_disp.to_numpy() - mpm_base) > 0.0

    # A rejected search must restore both accepted and temporary fields, plus
    # every geometry cache derived from them.
    accepted_iga = iga.grid_disp.to_numpy().copy()
    accepted_mpm = mpm.grid_disp.to_numpy().copy()
    accepted_points = mpm.p_temp.to_numpy().copy()
    accepted_ctrlpts = engine.contact_surface.control_points_hat.to_numpy().copy()

    def uphill_energy(current):
        iga_delta_trial = current.iga.grid_disp_temp.to_numpy() - accepted_iga
        mpm_delta_trial = current.mpm.grid_disp_temp.to_numpy() - accepted_mpm
        return float(np.dot(iga_delta_trial, iga_delta_trial) + np.dot(mpm_delta_trial, mpm_delta_trial))

    rejected = engine.contact_aware_armijo(
        correction,
        directional_derivative=-float(np.dot(correction, correction)),
        energy_function=uphill_energy,
        max_backtracks=2,
    )
    assert rejected["accepted"] is False
    assert np.allclose(iga.grid_disp.to_numpy(), accepted_iga)
    assert np.allclose(mpm.grid_disp.to_numpy(), accepted_mpm)
    assert np.allclose(iga.grid_disp_temp.to_numpy(), accepted_iga)
    assert np.allclose(mpm.grid_disp_temp.to_numpy(), accepted_mpm)
    assert np.allclose(mpm.p_temp.to_numpy(), accepted_points)
    assert np.allclose(engine.contact_surface.control_points_hat.to_numpy(), accepted_ctrlpts)

    calls = 0

    def exceptional_energy(current):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 0.0
        raise TypeError("custom trial energy failed")

    with pytest.raises(TypeError, match="custom trial energy failed"):
        engine.contact_aware_armijo(
            correction,
            directional_derivative=-float(np.dot(correction, correction)),
            energy_function=exceptional_energy,
        )
    assert np.allclose(iga.grid_disp.to_numpy(), accepted_iga)
    assert np.allclose(mpm.grid_disp.to_numpy(), accepted_mpm)
    assert np.allclose(iga.grid_disp_temp.to_numpy(), accepted_iga)
    assert np.allclose(mpm.grid_disp_temp.to_numpy(), accepted_mpm)
    assert np.allclose(mpm.p_temp.to_numpy(), accepted_points)
    assert np.allclose(engine.contact_surface.control_points_hat.to_numpy(), accepted_ctrlpts)

    # Exercise the real Taichi begin/accept lifecycle.  A common rigid
    # translation preserves every contact gap while proving that physical IGA
    # control points and MPM particles advance exactly once and that local
    # displacement fields are canonicalized back to zero afterwards.
    control_points_before = iga.patch.control_points.to_numpy().copy()
    particles_before = mpm.particle.x.to_numpy().copy()
    begin = engine.begin_implicit_ipc_step()
    assert begin["active_mpm_dof"] == mpm.active_dof

    rigid_translation = 2.0e-6
    iga_increment = np.zeros(iga.degree_of_freedom, dtype=np.float64)
    iga_increment.reshape((-1, config.DIM))[:, 0] = rigid_translation
    mpm_increment = np.zeros(mpm.degree_of_freedom, dtype=np.float64)
    mpm_increment[: mpm.active_dof].reshape((-1, config.DIM))[:, 0] = rigid_translation
    iga.grid_disp.from_numpy(iga_increment)
    mpm.grid_disp.from_numpy(mpm_increment)
    accepted_step = engine.accept_implicit_ipc_step()

    np.testing.assert_allclose(
        iga.patch.control_points.to_numpy() - control_points_before,
        np.tile([rigid_translation, 0.0], (control_points_before.shape[0], 1)),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        mpm.particle.x.to_numpy() - particles_before,
        np.tile([rigid_translation, 0.0], (particles_before.shape[0], 1)),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(iga.grid_disp.to_numpy(), np.zeros(iga.degree_of_freedom))
    np.testing.assert_array_equal(mpm.grid_disp.to_numpy(), np.zeros(mpm.degree_of_freedom))
    assert accepted_step["minimum_distance"] > engine.barrier.minimum_distance


def test_taichi_point_nurbs_accd_two_sided_motion_and_coo_assembly(tmp_path):
    """ACCD and coupled COO assembly keep numerical arrays on the device."""
    from src.igampm import IGAMPM

    iga = _build_iga_rectangle(tmp_path / "iga")
    mpm = _build_mpm_particle(tmp_path / "mpm")
    engine = IGAMPM(
        iga,
        mpm,
        kappa=1.0e4,
        dhat=0.08,
        dmin=0.005,
        barrier_nnz=20_000,
        contact_ccd_safety=0.9,
        assemble_type="COO",
    ).build()
    engine.initialize_barrier()
    expected_pairs = int(mpm.total_surface_num) * int(engine.contact_surface.num_surfaces)
    cpu_distances = engine.contacts.distance.to_numpy()[:expected_pairs].copy()

    engine.initialize_barrier()
    device_distances = engine.contacts.distance.to_numpy()[:expected_pairs]
    np.testing.assert_allclose(device_distances, cpu_distances, rtol=1.0e-10, atol=1.0e-12)

    iga.incre_resolution.fill(0.0)
    mpm_increment = np.zeros(mpm.degree_of_freedom, dtype=np.float64)
    mpm_increment[: mpm.active_dof].reshape((-1, 2))[:, 1] = 0.1
    mpm.incre_resolution.from_numpy(mpm_increment)

    minimum = engine.minimum_contact_distance()
    clearance = 0.005 + engine.strict_feasibility_tolerance
    control_points_before = engine.contact_surface.control_points_hat.to_numpy().copy()
    device_bound = engine.conservative_contact_step_device(max_step=1.0, safety=0.9, verify=False)
    expected_bound = min(1.0, 0.9 * (minimum - clearance) / 0.1)

    assert int(engine.contact_query_status[None]) == 0
    assert minimum > clearance
    assert np.isclose(device_bound, expected_bound, rtol=1.0e-10, atol=1.0e-12)
    np.testing.assert_array_equal(
        engine.contact_surface.control_points_hat.to_numpy(),
        control_points_before,
    )
    # Point and NURBS control points move on equal footing. A common
    # translation has zero relative-motion bound even though both directions
    # are nonzero; changing only the point by 0.1 recovers the same bound as
    # the one-sided case above.
    common_direction = np.zeros(iga.degree_of_freedom, dtype=np.float64)
    common_direction.reshape((-1, 2))[:, 1] = 0.04
    iga.incre_resolution.from_numpy(common_direction)
    mpm_increment[: mpm.active_dof].reshape((-1, 2))[:, 1] = 0.04
    mpm.incre_resolution.from_numpy(mpm_increment)
    common_translation_step = engine.conservative_contact_step_device(max_step=1.0, safety=0.9, verify=False)
    assert np.isclose(common_translation_step, 1.0, atol=1.0e-12)

    mpm_increment[: mpm.active_dof].reshape((-1, 2))[:, 1] = 0.14
    mpm.incre_resolution.from_numpy(mpm_increment)
    translated_relative_bound = min(1.0, 0.9 * (minimum - clearance) / 0.1)
    accumulated_step = engine.conservative_contact_step_device(max_step=1.0, safety=0.9)
    assert np.isclose(
        accumulated_step,
        translated_relative_bound,
        rtol=1.0e-9,
        atol=1.0e-11,
    )
    engine._set_device_trial_displacements(accumulated_step)
    engine.initialize_barrier(mpm.grid_disp_temp, iga.grid_disp_temp)
    retained_distance = engine.minimum_contact_distance()
    assert retained_distance > clearance
    assert retained_distance - clearance >= (0.1 * (minimum - clearance) - 1.0e-11)
    engine._synchronize_device_trial_state_with_accepted()

    system = engine.assemble_monolithic_newton_system(include_friction=False)
    active_dof = int(system["active_dof"])
    coo = system["matrix"]._to_scipy().tocsr()[:active_dof, :active_dof]
    assert system["assemble_type"] == "COO"
    assert system["backend"] == "taichi_device_coo_lagged_pcg"
    assert coo.shape == (active_dof, active_dof)
    assert np.all(np.isfinite(coo.data))


def test_default_monolithic_solve_routes_to_taichi_device_backend():
    from src.igampm.engines import Engine

    engine = object.__new__(Engine)
    engine.activate_fric = True
    engine.friction_mode = "lagged"
    engine.monolithic_max_iterations = 7
    engine.monolithic_tolerance = 2.0e-6
    engine._device_monolithic_available = lambda include_friction: True
    captured = {}

    def device_solve(**kwargs):
        captured.update(kwargs)
        return {"backend": "taichi_device_lagged_pcg"}

    engine._solve_monolithic_newton_device = device_solve
    result = engine.solve_monolithic_newton()

    assert result["backend"] == "taichi_device_lagged_pcg"
    assert captured == {
        "include_friction": True,
        "max_iterations": 7,
        "tolerance": 2.0e-6,
        "energy_function": None,
        "verbose": False,
        "linear_solve": None,
    }


def test_taichi_monolithic_assembly_merges_blocks_and_eliminates_dirichlet(
    tmp_path,
):
    """Compile device kernels and check their coupled scalar result."""
    from src.igampm import IGAMPM
    from src.linear_solver.BuildTriplet import BuildTriplet

    iga = _build_iga_rectangle(tmp_path / "iga")
    mpm = _build_mpm_particle(tmp_path / "mpm")
    engine = IGAMPM(
        iga,
        mpm,
        kappa=1.0e4,
        dhat=0.08,
        dmin=0.005,
        barrier_nnz=20_000,
    ).build()

    # Add absolute IGA displacement constraints at both ends of the block
    # ordering. Their nonzero Newton corrections exercise both the stored
    # upper block and its transposed mirror during GPU row/column elimination.
    iga.dirichlet.node = ti.field(ti.i32, shape=iga.degree_of_freedom)
    iga.dirichlet.value = ti.field(ti.f64, shape=iga.degree_of_freedom)
    fixed_dofs = np.asarray([0, iga.degree_of_freedom - 1], dtype=np.int64)
    fixed_corrections = np.asarray([3.0e-5, -2.0e-5])
    for fixed_dof, target in zip(fixed_dofs, fixed_corrections):
        iga.dirichlet.node[int(fixed_dof)] = 1
        iga.dirichlet.value[int(fixed_dof)] = float(target)
    # First assemble the exact same physical state without applying DBC so a
    # dense oracle can independently form b-A[:,F]c and the eliminated matrix.
    iga.dirichlet.num = 0

    total_dof = iga.degree_of_freedom + mpm.degree_of_freedom
    sources = (
        iga.hash_matrix,
        mpm.hash_matrix,
        engine.barrier_hash_matrix,
        engine.friction_hash_matrix,
    )
    engine.monolithic_hash_matrix = BuildTriplet(
        dim=2,
        max_pairs_num=sum(matrix.non_diag.max_pairs_num for matrix in sources),
        max_nonzeros=sum(matrix.max_nonzeros for matrix in sources),
        max_active_nodes=total_dof // 2,
        symmetric=False,
        solver="PCG",
        matrix_symmetric=True,
        full_symmetric_input=True,
    )
    coordinates, _ = iga.fixed_block_coordinates(upper_triangle=True)
    engine.monolithic_hash_matrix.install_fixed_pattern(coordinates)
    engine.monolithic_rhs = ti.field(ti.f64, shape=total_dof)
    engine.monolithic_physical_rhs = ti.field(ti.f64, shape=total_dof)
    engine.monolithic_correction = ti.field(ti.f64, shape=total_dof)
    engine.monolithic_fixed = ti.field(ti.i32, shape=total_dof)
    engine.monolithic_fixed_correction = ti.field(ti.f64, shape=total_dof)
    engine.contact_step_alpha = ti.field(ti.f64, shape=())
    engine.contact_query_status = ti.field(ti.i32, shape=())
    engine._device_monolithic_available = lambda include_friction: True

    def scipy_source_conversion_forbidden(*_args, **_kwargs):
        raise AssertionError("device monolithic assembly converted a source to SciPy")

    iga.hash_matrix.to_scipy = scipy_source_conversion_forbidden
    mpm.hash_matrix.to_scipy = scipy_source_conversion_forbidden
    engine.barrier_hash_matrix.to_scipy = scipy_source_conversion_forbidden
    unconstrained = engine.assemble_monolithic_newton_system(include_friction=False)
    unconstrained_matrix = engine.monolithic_hash_matrix.to_scipy(unconstrained["active_nodes"]).toarray()
    unconstrained_rhs = engine.monolithic_rhs.to_numpy()[: unconstrained["active_dof"]].copy()
    np.testing.assert_array_equal(unconstrained_matrix, unconstrained_matrix.T)

    iga.dirichlet.num = int(fixed_dofs.size)
    system = engine.assemble_monolithic_newton_system(include_friction=False)
    matrix = engine.monolithic_hash_matrix.to_scipy(system["active_nodes"]).toarray()
    rhs = engine.monolithic_rhs.to_numpy()[: system["active_dof"]]

    expected_rhs = unconstrained_rhs - (unconstrained_matrix[:, fixed_dofs] @ fixed_corrections)
    expected_matrix = unconstrained_matrix.copy()
    expected_matrix[:, fixed_dofs] = 0.0
    expected_matrix[fixed_dofs, :] = 0.0
    expected_matrix[fixed_dofs, fixed_dofs] = 1.0
    expected_rhs[fixed_dofs] = fixed_corrections
    np.testing.assert_array_equal(matrix, matrix.T)
    np.testing.assert_allclose(matrix, expected_matrix, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(rhs, expected_rhs, rtol=0.0, atol=1.0e-11)

    projected_newton_solve = engine.monolithic_hash_matrix.solve_flat_system(
        engine.monolithic_rhs,
        engine.monolithic_correction,
        active_nodes=system["active_nodes"],
        tol=1.0e-10,
        maxiter=1000,
        return_solution=False,
    )
    assert projected_newton_solve["converged"]
    assert np.isfinite(projected_newton_solve["residual"])
    expected_solution = np.linalg.solve(expected_matrix, expected_rhs)
    np.testing.assert_allclose(
        engine.monolithic_correction.to_numpy()[: system["active_dof"]],
        expected_solution,
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    barrier_raw_count = int(engine.barrier_hash_matrix.raw_non_diag_count[0])
    barrier_raw_i = engine.barrier_hash_matrix.non_diag.blockI.to_numpy()[:barrier_raw_count].copy()
    barrier_raw_j = engine.barrier_hash_matrix.non_diag.blockJ.to_numpy()[:barrier_raw_count].copy()
    engine.monolithic_correction.fill(0.0)
    engine.monolithic_correction[1] = 2.0e-5
    engine._split_device_monolithic_correction(system["active_mpm_dof"])
    correction_residual = engine._device_monolithic_correction_residual(system["active_mpm_dof"], iga.dt, mpm.dt)
    directional_derivative = engine._device_monolithic_directional_derivative(system["active_dof"])
    engine._set_device_trial_displacements(0.5)
    assert np.isclose(iga.grid_disp_temp[1], 1.0e-5)
    engine._sync_device_trial_displacements()

    assert system["backend"] == "taichi_device_hashtriplet_lagged_pcg"
    assert engine.monolithic_hash_matrix.solver == "PCG"
    assert np.all(np.isfinite(matrix))
    assert np.all(np.isfinite(rhs))
    assert np.isclose(correction_residual, max(2.0e-5, np.max(np.abs(fixed_corrections))) / iga.dt)
    assert np.isfinite(directional_derivative)
    identity = np.eye(matrix.shape[0])
    for fixed_dof, correction in zip(fixed_dofs, fixed_corrections):
        np.testing.assert_allclose(matrix[fixed_dof], identity[fixed_dof])
        np.testing.assert_allclose(matrix[:, fixed_dof], identity[:, fixed_dof])
        assert rhs[fixed_dof] == correction

    # Residual-only Armijo probes must preserve both subsystem directions;
    # otherwise the second backtrack silently evaluates the base state.
    iga_direction = np.linspace(-2.0e-5, 3.0e-5, iga.degree_of_freedom)
    mpm_direction = np.linspace(4.0e-5, -1.0e-5, mpm.degree_of_freedom)
    iga.incre_resolution.from_numpy(iga_direction)
    mpm.incre_resolution.from_numpy(mpm_direction)
    for _ in range(2):
        probe = engine.assemble_monolithic_newton_system(include_friction=False, need_matrix=False)
        assert probe["need_matrix"] is False
        np.testing.assert_array_equal(iga.incre_resolution.to_numpy(), iga_direction)
        np.testing.assert_array_equal(mpm.incre_resolution.to_numpy(), mpm_direction)

    # Reassembling the matrix at unchanged topology must reproduce the exact
    # contact/local-pair slot stream used by the persistent pattern cache.
    engine.assemble_monolithic_newton_system(include_friction=False, need_matrix=True)
    assert int(engine.barrier_hash_matrix.raw_non_diag_count[0]) == (barrier_raw_count)
    np.testing.assert_array_equal(
        engine.barrier_hash_matrix.non_diag.blockI.to_numpy()[:barrier_raw_count],
        barrier_raw_i,
    )
    np.testing.assert_array_equal(
        engine.barrier_hash_matrix.non_diag.blockJ.to_numpy()[:barrier_raw_count],
        barrier_raw_j,
    )


@pytest.mark.parametrize("invalid_distance", [np.nan, np.inf, -np.inf])
def test_nonfinite_expected_pair_is_never_treated_as_feasible(invalid_distance):
    from src.igampm.engines import Engine

    engine = object.__new__(Engine)
    engine.mpm = SimpleNamespace(total_surface_num=1)
    engine.contact_surface = SimpleNamespace(num_surfaces=2)
    contact_type = ti.types.struct(distance=ti.f64)
    engine.contacts = contact_type.field(shape=3)
    engine.contacts[0].distance = 0.1
    engine.contacts[1].distance = invalid_distance
    # The trailing infinity models spare storage and is not a real pair.
    engine.contacts[2].distance = np.inf
    engine.contact_query_status = ti.field(ti.i32, shape=())

    with pytest.raises(RuntimeError, match="non-finite distance"):
        engine.minimum_contact_distance()


def test_no_contact_pairs_ignore_dummy_storage():
    from src.igampm.engines import Engine

    class DistanceField:
        def to_numpy(self):
            return np.array([np.inf], dtype=np.float64)

    engine = object.__new__(Engine)
    engine.mpm = SimpleNamespace(total_surface_num=1)
    engine.contact_surface = SimpleNamespace(num_surfaces=0)
    engine.contacts = SimpleNamespace(distance=DistanceField())
    engine.barrier = SimpleNamespace(minimum_distance=0.0)
    engine.strict_feasibility_tolerance = 1.0e-14

    assert engine.minimum_contact_distance() == np.inf


def test_public_contact_step_and_armijo_only_upload_then_use_taichi_backend():
    from src.igampm.engines import Engine

    contact_source = inspect.getsource(Engine.conservative_contact_step)
    device_contact_source = inspect.getsource(Engine.conservative_contact_step_device)
    advance_accd_source = inspect.getsource(Engine._advance_point_nurbs_accd)
    armijo_source = inspect.getsource(Engine.contact_aware_armijo)
    device_armijo_source = inspect.getsource(Engine.contact_aware_armijo_device)
    assert "_load_external_correction" in contact_source
    assert "conservative_contact_step_device" in contact_source
    assert ".to_numpy(" not in contact_source
    assert "range(self.contact_ccd_max_iterations)" not in device_contact_source
    assert ".to_numpy(" not in advance_accd_source
    assert "control_points_hat[" not in advance_accd_source
    assert "_load_external_correction" in armijo_source
    assert "contact_aware_armijo_device" in armijo_source
    assert ".to_numpy(" not in armijo_source
    assert "verify=False" in device_armijo_source
    assert "prepare_contacts=False" in device_armijo_source

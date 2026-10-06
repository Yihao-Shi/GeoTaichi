"""Finite-strain plastic Direct ULMPM inside monolithic IGA-MPM IPC."""

import numpy as np
import pytest

import src.igampm.config as config

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.serial,
    pytest.mark.materials,
    pytest.mark.coupling,
    pytest.mark.contact,
    pytest.mark.assembly,
    pytest.mark.isolated_dimension(2),
]


def _build_iga_rectangle(output_path, axisymmetric=False):
    from src.iga import ImplicitIGA, Primitives, Rectangle

    rectangle = Rectangle()
    rectangle.set_parameters(
        start_point=[0.5, 0.0] if axisymmetric else [0.0, 0.0],
        size=[1.0, 0.2],
    )
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
        axisymmetric=axisymmetric,
        axis_offset=0.0,
        path=str(output_path),
    )


def _build_plastic_mpm_particle(output_path, material_name, axisymmetric=False, damping=0.0, dilation=30.0):
    from src.mpm.engines.direct.ImplicitULMPM import ImplicitULMPM
    from src.mpm.generator.Body import Body

    body = Body()
    body.add_particles(
        [[1.0 if axisymmetric else 0.5, -0.031]],
        volume=1.0e-3,
        xmin=[-0.1, -0.1],
        xmax=[1.1, 0.3],
        boundary_ids=[0],
    )
    material_parameters = {
        "material": material_name,
        "young_modulus": 1.0e5,
        "poisson_ratio": 0.3,
        "density": 1000.0,
    }
    if material_name == "DruckerPrager":
        material_parameters.update(
            {
                "FrictionAngle": 30.0,
                "DilationAngle": dilation,
                "Cohesion": 1.0,
                "dpType": "Circumscribed",
            }
        )
    else:
        material_parameters.update({"YieldStress": 1.0, "HardeningModulus": 50.0})
    mpm = ImplicitULMPM(
        domain=[1.2, 0.5],
        dx=0.1,
        dt=1.0e-3,
        bodies=body,
        newmark=[1.0, 0.5, 1.0],
        gravity=[0.0, 0.0],
        damping=damping,
        residual=1.0e-8,
        interval=1,
        step=1,
        scale=1.0,
        shape_function="linear",
        visualize=False,
        axisymmetric=axisymmetric,
        axis_offset=0.0,
        path=str(output_path),
        **material_parameters,
    )
    mpm.init_F0()
    return mpm


@pytest.mark.parametrize("axisymmetric", [False, True])
def test_cpt_total_potential_force_and_exact_hessian_by_fd(taichi_runtime, tmp_path, monkeypatch, axisymmetric):
    """Differentiate the actual coupled potential with frozen material history."""
    config.set_dimension(2)
    from src.igampm import IGAMPM
    import src.physics_model.contact_model.ipc.NurbsContact as nurbs_contact

    # Resolve the closest point below the finite-difference perturbation scale.
    monkeypatch.setattr(nurbs_contact, "CLOSEST_POINT_STATIONARITY_TOL", 1e-13)

    iga = _build_iga_rectangle(tmp_path / "iga", axisymmetric=axisymmetric)
    mpm = _build_plastic_mpm_particle(tmp_path / "mpm", "DruckerPrager", axisymmetric=axisymmetric, damping=0.05)
    engine = IGAMPM(
        iga,
        mpm,
        kappa=1e4,
        dhat=0.08,
        dmin=0.005,
        barrier_nnz=20_000,
        axisymmetric=axisymmetric,
        axis_offset=0.0,
        use_physical_barrier=True,
    ).build()
    # The exact Hessian must be checked before the production PSD projection.
    engine.project_lagged_hessians = False
    exact_material = mpm.assemble_stiffness_matrix_hash
    monkeypatch.setattr(
        mpm,
        "assemble_stiffness_matrix_hash",
        lambda *args, **kwargs: exact_material(*args, project_spd=False, exact_plastic_tangent=True),
    )
    engine.begin_implicit_ipc_step()
    mpm.F0.from_numpy(np.array([np.diag([1.18, 0.82, 0.90])]))
    mpm.init_particle_pressure(np.array([0]), [0.0, -150.0], np.array([0.01]))
    mpm.traction_p2g()
    iga.patch.velocitys.from_numpy(np.tile([0.03, -0.02], (iga.degree_of_freedom // 2, 1)))
    iga.patch.accelerations.from_numpy(np.tile([0.2, -0.1], (iga.degree_of_freedom // 2, 1)))
    system = engine.assemble_monolithic_newton_system(include_friction=False)
    assert engine.curr_barrier_contact_num > 0
    count = system["active_dof"]
    base_mpm = mpm.grid_disp.to_numpy()
    base = np.linspace(-2e-5, 3e-5, count)

    def evaluate(values, need_matrix=False):
        iga.grid_disp.from_numpy(values[: iga.degree_of_freedom])
        displacement = base_mpm.copy()
        displacement[: mpm.active_dof] = values[iga.degree_of_freedom :]
        mpm.grid_disp.from_numpy(displacement)
        assembled = engine.assemble_monolithic_newton_system(include_friction=False, need_matrix=need_matrix)
        energy = engine.coupled_potential_energy(include_friction=False)
        gradient = -engine.monolithic_physical_rhs.to_numpy()[:count]
        matrix = assembled["matrix"].to_scipy(assembled["active_nodes"]).toarray() if need_matrix else None
        return energy, gradient.copy(), matrix

    _, gradient, hessian = evaluate(base, True)
    step = 2e-6
    fd_gradient, fd_hessian = np.zeros_like(gradient), np.zeros_like(hessian)
    for column in range(count):
        delta = np.zeros_like(base)
        delta[column] = step
        plus, minus = evaluate(base + delta), evaluate(base - delta)
        fd_gradient[column] = (plus[0] - minus[0]) / (2 * step)
        fd_hessian[:, column] = (plus[1] - minus[1]) / (2 * step)
    np.testing.assert_allclose(gradient, fd_gradient, rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(hessian, fd_hessian, rtol=3e-4, atol=3e-2)
    np.testing.assert_allclose(hessian, hessian.T, rtol=1e-11, atol=1e-7)


@pytest.mark.parametrize(
    ("material_name", "assemble_type"),
    [("DruckerPrager", "HashTriplet"), ("VonMises", "COO")],
)
def test_plastic_ulmpm_uses_monolithic_ipc_accd_and_transactional_history(
    taichi_runtime,
    tmp_path,
    monkeypatch,
    material_name,
    assemble_type,
):
    config.set_dimension(2)

    from src.igampm import IGAMPM

    iga = _build_iga_rectangle(tmp_path / "iga")
    mpm = _build_plastic_mpm_particle(tmp_path / "mpm", material_name)
    trial_deformation = np.array([np.diag([1.30, 0.70, 1.0])], dtype=np.float64)
    mpm.F0.from_numpy(trial_deformation)

    engine = IGAMPM(
        iga,
        mpm,
        kappa=1.0e4,
        dhat=0.08,
        dmin=0.005,
        barrier_nnz=20_000,
        assemble_type=assemble_type,
    ).build()
    begin = engine.begin_implicit_ipc_step()
    system = engine.assemble_monolithic_newton_system(include_friction=False)

    assert engine.mpm_has_plastic_history
    assert mpm.is_plane_strain
    assert mpm.material_dimension == 3
    assert mpm.F0.to_numpy().shape == (1, 3, 3)
    assert begin["active_mpm_dof"] > 0
    assert engine.curr_barrier_contact_num > 0
    assert system["assemble_type"] == assemble_type
    assert np.all(np.isfinite(system["rhs"].to_numpy()[: system["active_dof"]]))
    if assemble_type == "COO":
        assert int(engine.monolithic_coo_count[None]) > 0
    else:
        assert int(system["matrix"].raw_non_diag_count[0]) > 0

    correction = np.zeros(system["active_dof"], dtype=np.float64)
    correction[iga.degree_of_freedom :].reshape((-1, 2))[:, 1] = 0.1
    contact_step = engine.conservative_contact_step(correction)
    assert 0.0 < contact_step < 1.0
    assert engine.last_contact_ccd_step == pytest.approx(contact_step)

    equivalent_before = mpm.material.equivalent_plastic_strain.to_numpy().copy()
    volumetric_before = mpm.material.volumetric_plastic_strain.to_numpy().copy()
    plastic_inverse_before = mpm.material.plastic_deformation_inverse.to_numpy().copy()
    deformation_before = mpm.F0.to_numpy().copy()
    grid_mass_before = mpm.grid.m.to_numpy().copy()

    initialize_barrier = engine.initialize_barrier
    barrier_calls = 0

    def fail_after_material_commit(*args):
        nonlocal barrier_calls
        barrier_calls += 1
        if barrier_calls == 2:
            mpm.grid.m.fill(13.0)
            raise RuntimeError("post-commit contact refresh failed")
        return initialize_barrier(*args)

    monkeypatch.setattr(engine, "initialize_barrier", fail_after_material_commit)
    with pytest.raises(RuntimeError, match="post-commit contact refresh failed"):
        engine.accept_implicit_ipc_step()

    np.testing.assert_array_equal(mpm.F0.to_numpy(), deformation_before)
    np.testing.assert_array_equal(mpm.grid.m.to_numpy(), grid_mass_before)
    np.testing.assert_array_equal(mpm.material.equivalent_plastic_strain.to_numpy(), equivalent_before)
    np.testing.assert_array_equal(mpm.material.volumetric_plastic_strain.to_numpy(), volumetric_before)
    np.testing.assert_array_equal(mpm.material.plastic_deformation_inverse.to_numpy(), plastic_inverse_before)
    assert not engine.implicit_step_in_progress

    monkeypatch.setattr(engine, "initialize_barrier", initialize_barrier)
    engine.begin_implicit_ipc_step()
    accepted = engine.accept_implicit_ipc_step()

    equivalent_after = mpm.material.equivalent_plastic_strain.to_numpy()
    assert equivalent_after[0] > equivalent_before[0]
    assert np.linalg.det(mpm.F0.to_numpy()[0]) > 0.0
    assert accepted["minimum_distance"] > engine.barrier.minimum_distance


@pytest.mark.parametrize("material_name", ["DruckerPrager", "VonMises"])
def test_axisymmetric_plastic_ulmpm_uses_three_dimensional_material_map(
    taichi_runtime,
    tmp_path,
    material_name,
):
    config.set_dimension(2)

    from src.igampm import IGAMPM

    iga = _build_iga_rectangle(tmp_path / f"iga-axis-{material_name}", axisymmetric=True)
    mpm = _build_plastic_mpm_particle(
        tmp_path / f"mpm-axis-{material_name}",
        material_name,
        axisymmetric=True,
    )
    mpm.F0.from_numpy(np.array([np.diag([1.22, 0.78, 1.16])], dtype=np.float64))
    engine = IGAMPM(
        iga,
        mpm,
        kappa=1.0e4,
        dhat=0.08,
        dmin=0.005,
        barrier_nnz=20_000,
        assemble_type="HashTriplet",
        axisymmetric=True,
        axis_offset=0.0,
    ).build()

    begin = engine.begin_implicit_ipc_step()
    system = engine.assemble_monolithic_newton_system(include_friction=False)

    assert engine.is_axisymmetric
    assert engine.mpm_has_plastic_history
    assert mpm.material_dimension == 3
    assert mpm.F0.to_numpy().shape == (1, 3, 3)
    assert mpm.particle.vol0.to_numpy()[0] == pytest.approx(2.0 * np.pi * 1.0e-3)
    assert begin["active_mpm_dof"] > 0
    assert engine.curr_barrier_contact_num > 0
    assert np.all(np.isfinite(system["rhs"].to_numpy()[: system["active_dof"]]))
    assert int(system["matrix"].raw_non_diag_count[0]) > 0

    equivalent_before = mpm.material.equivalent_plastic_strain.to_numpy().copy()
    engine.accept_implicit_ipc_step()
    equivalent_after = mpm.material.equivalent_plastic_strain.to_numpy()
    assert equivalent_after[0] > equivalent_before[0]
    assert np.linalg.det(mpm.F0.to_numpy()[0]) > 0.0


@pytest.mark.parametrize("axisymmetric", [False, True])
@pytest.mark.parametrize("dilation", [0.0, 15.0, 30.0])
def test_plastic_volume_fixed_point_preserves_assembled_force_at_commit(
    taichi_runtime, tmp_path, axisymmetric, dilation
):
    import taichi as ti

    mpm = _build_plastic_mpm_particle(tmp_path / "mpm", "DruckerPrager", axisymmetric, dilation=dilation)
    assert mpm.has_lagged_material
    mpm.F0.from_numpy(np.array([np.diag([1.22, 0.78, 1.16])]))
    mpm.grid_reset()
    mpm.compute_shapefn()
    mpm.mass_vel_acc_p2g()
    mpm.find_active_node()
    mpm.prefix_sum_executor.run(mpm.node2dof)
    active = mpm.set_active_dof()
    mpm.grid_disp.fill(0.05)
    mpm.begin_lagged_material_state()
    assert mpm.refresh_lagged_material_state(mpm.grid_disp) > mpm.material_lagged_tolerance
    assert mpm.refresh_lagged_material_state(mpm.grid_disp) <= mpm.material_lagged_tolerance
    mpm.rhs.fill(0.0)
    mpm.assemble_material_force(active, mpm.grid_disp)
    before = mpm.rhs.to_numpy().copy()

    @ti.kernel
    def commit_history_at_same_displacement():
        for i in range(mpm.particleNum[0]):
            incremental = ti.Matrix.identity(ti.f64, 3)
            if ti.static(axisymmetric):
                incremental = mpm.get_axisymmetric_incremental_map(i, mpm.grid_disp)
            else:
                incremental = mpm.get_plane_strain_incremental_map(i, mpm.grid_disp)
            ignored = mpm.material.commit_total_state(i, incremental @ mpm.F0[i])

    # Retain the original geometry, F_n and basis: only the material state changes.
    commit_history_at_same_displacement()
    mpm.rhs.fill(0.0)
    mpm.assemble_material_force(active, mpm.grid_disp)
    assert mpm.material.lagged_plastic_volume_active[0] == 0
    np.testing.assert_allclose(mpm.rhs.to_numpy(), before, rtol=1.0e-8, atol=1.0e-8)


@pytest.mark.parametrize("axisymmetric", [False, True])
def test_nonassociated_ipc_solves_with_pcg_and_physical_flow(taichi_runtime, tmp_path, axisymmetric):
    from src.igampm import IGAMPM

    iga = _build_iga_rectangle(tmp_path / "iga", axisymmetric)
    mpm = _build_plastic_mpm_particle(tmp_path / "mpm", "DruckerPrager", axisymmetric, dilation=15.0)
    mpm.F0.from_numpy(np.array([np.diag([1.18, 0.82, 0.90])]))
    engine = IGAMPM(
        iga,
        mpm,
        kappa=1e4,
        dhat=0.08,
        dmin=0.005,
        barrier_nnz=20_000,
        assemble_type="HashTriplet",
        axisymmetric=axisymmetric,
        axis_offset=0.0,
        monolithic_max_iterations=100,
        monolithic_tolerance=1e-8,
        monolithic_force_rtol=1e-8,
        monolithic_force_atol=1e-8,
    ).build()
    assert engine.monolithic_hash_matrix.solver == "PCG"
    assert mpm.hash_matrix.solver == "PCG"
    result = engine.implicit_ipc_substep(verbose=False)
    assert result["converged"]
    assert mpm.last_material_lagged_error <= mpm.material_lagged_tolerance
    assert engine.last_monolithic_force_residual <= 1e-6
    assert result["minimum_distance"] > engine.barrier.minimum_distance
    equivalent = mpm.material.equivalent_plastic_strain[0]
    assert equivalent > 0
    assert mpm.material.volumetric_plastic_strain[0] == pytest.approx(
        3 * mpm.material.beta * equivalent / np.sqrt(2 / 3),
        abs=1e-9,
    )

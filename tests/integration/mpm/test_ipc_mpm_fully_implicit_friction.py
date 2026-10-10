import os

import numpy as np
import pytest
import taichi as ti

pytestmark = [pytest.mark.slow, pytest.mark.serial]


@pytest.fixture(scope="module", autouse=True)
def _initialize_taichi():
    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    arch = ti.gpu if os.environ.get("GEOTAICHI_TEST_ARCH") == "gpu" else ti.cpu
    ti.init(
        arch=arch,
        default_fp=ti.f64,
        debug=os.environ.get("GEOTAICHI_TEST_DEBUG") == "1",
        offline_cache=False,
    )


def _build_point_plane_contact(
    tmp_path,
    *,
    ipc_model="BarrierIPC",
    friction_mode="fully_implicit_experimental",
    activate_friction=True,
    gravity=(2.0, -9.81),
    newmark=(1.0, 0.5, 1.0),
    material="neoHookean",
    material_parameters=None,
):
    import src.mpm.config as config
    from src.mpm.engines.direct.ImplicitULMPM import ImplicitULMPM
    from src.mpm.generator.Body import Body
    from src.mpm.generator.Ground import Ground
    from src.mpm.soft_particle.IPCMPM import IPCMPM

    config.set_dimension(2)
    bodies = Body()
    bodies.add_particles(
        [[0.45, 0.04]],
        volume=1.0e-3,
        boundary_ids=[0],
        xmin=[0.0, 0.0],
        xmax=[1.0, 1.0],
    )
    material_parameters = {} if material_parameters is None else dict(material_parameters)
    mpm = ImplicitULMPM(
        domain=[1.0, 1.0],
        dx=0.1,
        dt=1.0e-3,
        bodies=bodies,
        newmark=newmark,
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        density=1000.0,
        material=material,
        gravity=gravity,
        residual=1.0e-5,
        max_iters=12,
        interval=1,
        step=1,
        scale=1.0,
        line_search=False,
        shape_function="linear",
        visualize=False,
        path=str(tmp_path / "mpm"),
        **material_parameters,
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

    ground = Ground()
    ground.append([0.0, 0.0], [0.0, 1.0])
    contact = IPCMPM(
        mpm,
        ground,
        ipc_model=ipc_model,
        kappa=1.0e4,
        dhat=0.08,
        mu=0.5 if activate_friction else 0.0,
        epsv=0.1,
        activate_friction=activate_friction,
        friction_mode=friction_mode,
        coordination_number=[1, 1],
        friction_set=[4, 4],
        barrier_set=[4, 4],
    )
    contact.initial_surface_temp()
    return config, mpm, contact


def test_elastic_barrier_ipc_adjoint_matches_resolved_gravity_fd(tmp_path):
    seed = np.array([0.3, -0.7])

    def solve(gravity_y, young=1.0e5, need_adjoint=False):
        config, mpm, contact = _build_point_plane_contact(
            tmp_path,
            friction_mode="lagged",
            activate_friction=False,
            gravity=(2.0, gravity_y),
        )
        mpm.material.add_material(mpm.material.density, young, mpm.material.poisson)
        mpm.compute_mass_list(mpm.integration)
        mpm.grid_disp.fill(0.0)
        contact.solve_friction_step(verbose=False)
        contact.update_particle_pos(mpm.grid_disp)
        objective = float(seed @ mpm.p_temp.to_numpy()[0])
        if not need_adjoint:
            return objective

        rhs = np.zeros(mpm.degree_of_freedom)
        particle = int(mpm.surface_id.to_numpy()[0])
        local_count = int(mpm.offset.to_numpy()[particle])
        local_nodes = mpm.LnID.to_numpy()[particle, :local_count]
        weights = mpm.shape.to_numpy()[particle, :local_count]
        node_to_dof = mpm.node2dof.to_numpy()
        for node, weight in zip(local_nodes, weights):
            block = int(node_to_dof[node]) - 1
            rhs[config.DIM * block : config.DIM * (block + 1)] += weight * seed
        parameters = contact.differentiate_elastic_parameters(rhs)
        assert contact.last_adjoint_result["converged"]
        assert contact.system_hash_matrix.solver == "PCG"
        return objective, parameters

    _, parameters = solve(-9.81, need_adjoint=True)
    gravity_delta = 1.0e-3
    gravity_fd = (solve(-9.81 + gravity_delta) - solve(-9.81 - gravity_delta)) / (2.0 * gravity_delta)
    assert parameters["gravity"][1] == pytest.approx(gravity_fd, rel=2.0e-5, abs=1.0e-10)

    young_delta = 10.0
    young_fd = (solve(-9.81, 1.0e5 + young_delta) - solve(-9.81, 1.0e5 - young_delta)) / (2.0 * young_delta)
    assert parameters["young_modulus"] == pytest.approx(young_fd, rel=3.0e-5, abs=1.0e-12)


def test_fully_implicit_friction_differentiation_is_rejected(tmp_path):
    config, mpm, contact = _build_point_plane_contact(tmp_path)
    mpm.compute_mass_list(mpm.integration)
    displacement = _translation_state(config, mpm, 3.0e-5, -2.0e-3)
    mpm.grid_disp.from_numpy(displacement)
    seed = np.linspace(-0.4, 0.7, mpm.active_dof)

    with pytest.raises(ValueError, match="supports friction_mode='lagged' only"):
        contact.differentiate_elastic_parameters(seed)


def test_lagged_friction_and_plasticity_share_exact_adjoint(tmp_path):
    config, mpm, contact = _build_point_plane_contact(
        tmp_path,
        friction_mode="lagged",
        material="vonMises",
        material_parameters={
            "yield_stress": 500.0,
            "hardening_modulus": 1200.0,
        },
    )
    mpm.compute_mass_list(mpm.integration)
    plastic_inverse = np.array(
        [[1.03, 0.02, 0.0], [-0.01, 0.97, 0.0], [0.0, 0.0, 1.01]],
        dtype=np.float64,
    )
    mpm.material.plastic_deformation_inverse.from_numpy(plastic_inverse[None])
    mpm.material.equivalent_plastic_strain[0] = 0.15
    displacement = _affine_nodal_state(
        config,
        mpm,
        np.diag([0.20, -0.10]),
    ) + _translation_state(config, mpm, 3.0e-5, -2.0e-3)
    mpm.grid_disp.from_numpy(displacement)
    contact.begin_friction_step(mpm.grid_disp)
    matrix = (
        contact.assemble_current_system(
            mpm.grid_disp,
            project_spd=False,
            exact_plastic_tangent=True,
        )
        .to_scipy(mpm.active_dof // config.DIM)
        .toarray()
    )
    seed = np.linspace(0.6, -0.3, mpm.active_dof)

    parameters = contact.differentiate_plastic_equilibrium_parameters(seed)
    adjoint = parameters["adjoint"].to_numpy()[: mpm.active_dof]

    assert int(contact.gfrictionNum[0]) == 1
    assert contact.last_adjoint_result["converged"]
    np.testing.assert_allclose(
        matrix.T @ adjoint,
        seed,
        rtol=2.0e-7,
        atol=1.0e-9,
    )
    expected_friction = adjoint @ (contact.friction_grad.to_numpy()[: mpm.active_dof] / float(contact.friction.mu[0]))
    assert parameters["friction_coefficient"] == pytest.approx(
        expected_friction,
        rel=2.0e-7,
        abs=1.0e-9,
    )

    history_step = 2.0e-7
    accepted_inverse = mpm.material.plastic_deformation_inverse.to_numpy()
    trial = accepted_inverse.copy()
    # Friction has no direct dependence on constitutive history at fixed
    # displacement. Difference only the material residual so the small
    # history signal is not cancelled against the much larger IPC force.
    trial[0, 0, 1] += history_step
    mpm.material.plastic_deformation_inverse.from_numpy(trial)
    mpm.rhs.fill(0.0)
    mpm.assemble_material_force(mpm.active_dof, mpm.grid_disp)
    plus = mpm.rhs.to_numpy()[: mpm.active_dof].copy()
    trial[0, 0, 1] -= 2.0 * history_step
    mpm.material.plastic_deformation_inverse.from_numpy(trial)
    mpm.rhs.fill(0.0)
    mpm.assemble_material_force(mpm.active_dof, mpm.grid_disp)
    minus = mpm.rhs.to_numpy()[: mpm.active_dof].copy()
    mpm.material.plastic_deformation_inverse.from_numpy(accepted_inverse)
    expected = adjoint @ ((plus - minus) / (2.0 * history_step))
    assert parameters["plastic_deformation_inverse"][0, 0, 1] == pytest.approx(
        expected,
        rel=2.0e-5,
        abs=2.0e-8,
    )


def test_lagged_adjoint_cache_is_distinct_from_fixed_point_probe(tmp_path):
    config, mpm, contact = _build_point_plane_contact(
        tmp_path,
        friction_mode="lagged",
    )
    accepted = _translation_state(config, mpm, 3.0e-5, -2.0e-3)
    probe = _translation_state(config, mpm, 3.0e-5, -8.0e-3)

    mpm.grid_disp.from_numpy(accepted)
    contact.begin_friction_step(mpm.grid_disp)
    assert int(contact.gfrictionNum[0]) == 1
    accepted_weight = float(contact.gfriction[0].mu_lambda)
    contact._backup_lagged_friction_for_adjoint()
    mpm.grid_disp_temp.from_numpy(probe)
    contact.refresh_friction_cache(mpm.grid_disp_temp)
    assert float(contact.gfriction[0].mu_lambda) != pytest.approx(accepted_weight)

    contact._restore_lagged_friction_for_adjoint()
    assert int(contact.gfrictionNum[0]) == 1
    assert float(contact.gfriction[0].mu_lambda) == pytest.approx(accepted_weight)


@pytest.mark.parametrize(
    ("material", "material_parameters"),
    [
        (
            "vonMises",
            {"yield_stress": 500.0, "hardening_modulus": 1200.0},
        ),
        (
            "druckerPrager",
            {"cohesion": 100.0, "friction_angle": 20.0},
        ),
    ],
)
def test_plastic_equilibrium_adjoint_uses_exact_consistent_tangent(tmp_path, material, material_parameters):
    config, mpm, contact = _build_point_plane_contact(
        tmp_path,
        friction_mode="lagged",
        activate_friction=False,
        gravity=(0.0, 0.0),
        material=material,
        material_parameters=material_parameters,
    )
    mpm.compute_mass_list(mpm.integration)
    plastic_inverse = np.array(
        [[1.03, 0.02, 0.0], [-0.01, 0.97, 0.0], [0.0, 0.0, 1.01]],
        dtype=np.float64,
    )
    equivalent_plastic_strain = 0.15
    mpm.material.plastic_deformation_inverse.from_numpy(plastic_inverse[None])
    mpm.material.equivalent_plastic_strain[0] = equivalent_plastic_strain
    base = _affine_nodal_state(config, mpm, np.diag([0.20, -0.10]))
    mpm.grid_disp.from_numpy(base)
    matrix = (
        contact.assemble_current_system(
            mpm.grid_disp,
            project_spd=False,
            exact_plastic_tangent=True,
        )
        .to_scipy(mpm.active_dof // config.DIM)
        .toarray()
    )

    step = 2.0e-7
    numerical = np.zeros_like(matrix)
    for column in range(mpm.active_dof):
        perturbation = np.zeros(mpm.degree_of_freedom, dtype=np.float64)
        perturbation[column] = step
        mpm.grid_disp_temp.from_numpy(base + perturbation)
        contact.assemble_current_system(mpm.grid_disp_temp, need_matrix=False)
        plus = mpm.rhs.to_numpy()[: mpm.active_dof].copy()
        mpm.grid_disp_temp.from_numpy(base - perturbation)
        contact.assemble_current_system(mpm.grid_disp_temp, need_matrix=False)
        minus = mpm.rhs.to_numpy()[: mpm.active_dof].copy()
        numerical[:, column] = -(plus - minus) / (2.0 * step)

    scale = max(float(np.linalg.norm(matrix, ord=np.inf)), 1.0)
    assert np.linalg.norm(matrix - numerical, ord=np.inf) / scale < 8.0e-5

    mpm.grid_disp.from_numpy(base)
    seed = np.linspace(-0.4, 0.6, mpm.active_dof)
    parameters = contact.differentiate_plastic_equilibrium_parameters(seed)
    adjoint = parameters["adjoint"].to_numpy()[: mpm.active_dof]
    assert contact.system_hash_matrix.solver == "PCG"
    np.testing.assert_allclose(matrix.T @ adjoint, seed, rtol=2.0e-7, atol=1.0e-9)

    history_step = 2.0e-7
    numerical_inverse_vjp = np.zeros((3, 3), dtype=np.float64)
    for row, column in np.ndindex(3, 3):
        trial = plastic_inverse.copy()
        trial[row, column] += history_step
        mpm.material.plastic_deformation_inverse.from_numpy(trial[None])
        contact.assemble_current_system(mpm.grid_disp, need_matrix=False)
        plus = mpm.rhs.to_numpy()[: mpm.active_dof].copy()
        trial[row, column] -= 2.0 * history_step
        mpm.material.plastic_deformation_inverse.from_numpy(trial[None])
        contact.assemble_current_system(mpm.grid_disp, need_matrix=False)
        minus = mpm.rhs.to_numpy()[: mpm.active_dof].copy()
        numerical_inverse_vjp[row, column] = adjoint @ ((plus - minus) / (2.0 * history_step))

    accepted_total = mpm.F0.to_numpy()
    numerical_total_vjp = np.zeros((3, 3), dtype=np.float64)
    mpm.material.plastic_deformation_inverse.from_numpy(plastic_inverse[None])
    for row, column in np.ndindex(3, 3):
        trial = accepted_total.copy()
        trial[0, row, column] += history_step
        mpm.F0.from_numpy(trial)
        contact.assemble_current_system(mpm.grid_disp, need_matrix=False)
        plus = mpm.rhs.to_numpy()[: mpm.active_dof].copy()
        trial[0, row, column] -= 2.0 * history_step
        mpm.F0.from_numpy(trial)
        contact.assemble_current_system(mpm.grid_disp, need_matrix=False)
        minus = mpm.rhs.to_numpy()[: mpm.active_dof].copy()
        numerical_total_vjp[row, column] = adjoint @ ((plus - minus) / (2.0 * history_step))
    mpm.F0.from_numpy(accepted_total)

    mpm.material.plastic_deformation_inverse.from_numpy(plastic_inverse[None])
    mpm.material.equivalent_plastic_strain[0] = equivalent_plastic_strain + history_step
    contact.assemble_current_system(mpm.grid_disp, need_matrix=False)
    plus = mpm.rhs.to_numpy()[: mpm.active_dof].copy()
    mpm.material.equivalent_plastic_strain[0] = equivalent_plastic_strain - history_step
    contact.assemble_current_system(mpm.grid_disp, need_matrix=False)
    minus = mpm.rhs.to_numpy()[: mpm.active_dof].copy()
    numerical_equivalent_vjp = adjoint @ ((plus - minus) / (2.0 * history_step))
    mpm.material.equivalent_plastic_strain[0] = equivalent_plastic_strain

    np.testing.assert_allclose(
        parameters["plastic_deformation_inverse"][0],
        numerical_inverse_vjp,
        rtol=2.0e-5,
        atol=2.0e-8,
    )
    np.testing.assert_allclose(
        parameters["deformation_gradient"][0],
        numerical_total_vjp,
        rtol=2.0e-5,
        atol=2.0e-8,
    )
    assert parameters["equivalent_plastic_strain"][0] == pytest.approx(
        numerical_equivalent_vjp,
        rel=2.0e-5,
        abs=2.0e-8,
    )
    np.testing.assert_array_equal(
        parameters["volumetric_plastic_strain"],
        np.zeros_like(parameters["volumetric_plastic_strain"]),
    )
    assert parameters["plastic_history"] == "input_vjp"

    accepted_parameters = contact.differentiate_plastic_step_parameters(seed, {})
    assert accepted_parameters["plastic_history"] == "accepted_step_input_vjp"
    for name in (
        "deformation_gradient",
        "plastic_deformation_inverse",
        "equivalent_plastic_strain",
        "volumetric_plastic_strain",
    ):
        np.testing.assert_allclose(accepted_parameters[name], parameters[name], rtol=2.0e-10, atol=2.0e-10)


def test_complete_plastic_particle_trajectory_vjp_matches_directional_fd(tmp_path):
    from src.mpm import DifferentiableMPM
    from src.mpm.soft_particle.IPCULMPM import IPCULMPM

    def build():
        _, mpm, contact = _build_point_plane_contact(
            tmp_path,
            friction_mode="lagged",
            activate_friction=True,
            gravity=(2.0, -9.81),
            newmark=(0.5, 0.25, 0.5),
            material="vonMises",
            material_parameters={
                "yield_stress": 80.0,
                "hardening_modulus": 400.0,
            },
        )
        simulation = object.__new__(IPCULMPM)
        simulation.mpm = mpm
        simulation.ipc = contact
        mpm.F0.from_numpy(np.array([[[1.02, 0.004, 0.0], [0.002, 0.985, 0.0], [0.0, 0.0, 1.0]]]))
        mpm.material.plastic_deformation_inverse.from_numpy(
            np.array([[[1.002, -0.002, 0.0], [0.001, 0.998, 0.0], [0.0, 0.0, 1.0]]])
        )
        mpm.material.equivalent_plastic_strain[0] = 0.04
        direction = {
            # Pure tangential translation preserves the lagged normal-force
            # cache. Plastic/history directions have their own return-map and
            # equilibrium finite differences above; varying them here would
            # differentiate the deliberately stopped cache refresh.
            "position": np.array([[0.15, 0.0]]),
            "velocity": np.array([[0.7, 0.0]]),
            "acceleration": np.array([[-0.2, 0.0]]),
            "deformation_gradient": np.zeros((1, 3, 3)),
            "plastic_deformation_inverse": np.zeros((1, 3, 3)),
            "equivalent_plastic_strain": np.zeros(1),
            "volumetric_plastic_strain": np.zeros(1),
        }
        return simulation, direction

    seed = {
        "position": np.array([[0.3, -0.2]]),
        "velocity": np.array([[-0.04, 0.07]]),
        "acceleration": np.array([[2.0e-5, -3.0e-5]]),
        "deformation_gradient": np.array([[[0.08, -0.05, 0.0], [0.03, 0.06, 0.0], [0.0, 0.0, -0.02]]]),
        "plastic_deformation_inverse": np.array([[[-0.04, 0.02, 0.0], [0.01, 0.03, 0.0], [0.0, 0.0, 0.05]]]),
        "equivalent_plastic_strain": np.array([0.11]),
        "volumetric_plastic_strain": np.array([-0.09]),
    }

    simulation, direction = build()
    mpm = simulation.mpm
    initial = {
        "position": mpm.particle.x.to_numpy(),
        "velocity": mpm.particle.v.to_numpy(),
        "acceleration": mpm.particle.a.to_numpy(),
        "deformation_gradient": mpm.F0.to_numpy(),
        "plastic_deformation_inverse": (mpm.material.plastic_deformation_inverse.to_numpy()),
        "equivalent_plastic_strain": (mpm.material.equivalent_plastic_strain.to_numpy()),
        "volumetric_plastic_strain": (mpm.material.volumetric_plastic_strain.to_numpy()),
    }

    def reset(shift):
        mpm.particle.x.from_numpy(initial["position"] + shift * direction["position"])
        mpm.particle.v.from_numpy(initial["velocity"] + shift * direction["velocity"])
        mpm.particle.a.from_numpy(initial["acceleration"] + shift * direction["acceleration"])
        mpm.F0.from_numpy(initial["deformation_gradient"] + shift * direction["deformation_gradient"])
        mpm.material.plastic_deformation_inverse.from_numpy(
            initial["plastic_deformation_inverse"] + shift * direction["plastic_deformation_inverse"]
        )
        mpm.material.equivalent_plastic_strain.from_numpy(
            initial["equivalent_plastic_strain"] + shift * direction["equivalent_plastic_strain"]
        )
        mpm.material.volumetric_plastic_strain.from_numpy(
            initial["volumetric_plastic_strain"] + shift * direction["volumetric_plastic_strain"]
        )
        mpm.time = 0.0
        mpm.step_count = 0
        mpm.history.clear()

    def objective(shift):
        reset(shift)
        for _ in range(2):
            simulation.step(verbose=False, record_history=False)
        return (
            np.sum(seed["position"] * mpm.particle.x.to_numpy()[:1])
            + np.sum(seed["velocity"] * mpm.particle.v.to_numpy()[:1])
            + np.sum(seed["acceleration"] * mpm.particle.a.to_numpy()[:1])
            + np.sum(seed["deformation_gradient"] * mpm.F0.to_numpy()[:1])
            + np.sum(seed["plastic_deformation_inverse"] * mpm.material.plastic_deformation_inverse.to_numpy()[:1])
            + np.sum(seed["equivalent_plastic_strain"] * mpm.material.equivalent_plastic_strain.to_numpy()[:1])
            + np.sum(seed["volumetric_plastic_strain"] * mpm.material.volumetric_plastic_strain.to_numpy()[:1])
        )

    reset(0.0)
    trajectory = DifferentiableMPM(simulation, steps=2)
    for _ in range(2):
        trajectory.step(verbose=False)
    gradient = trajectory.backward(seed, verbose=False)
    assert int(simulation.ipc.gfrictionNum[0]) == 1
    assert np.isfinite(gradient["friction_coefficient"])
    assert abs(gradient["friction_coefficient"]) > 1.0e-12
    assert mpm.material.equivalent_plastic_strain[0] > (initial["equivalent_plastic_strain"][0] + 1.0e-8)
    analytic = sum(np.sum(gradient[f"initial_{name}"] * direction[name]) for name in direction)
    delta = 2.0e-6
    numerical = (objective(delta) - objective(-delta)) / (2.0 * delta)
    assert analytic == pytest.approx(numerical, rel=2.0e-3, abs=2.0e-7)


def _translation_state(config, mpm, x, y):
    displacement = np.zeros(mpm.degree_of_freedom, dtype=np.float64)
    for dof in range(0, mpm.active_dof, config.DIM):
        displacement[dof] = x
        displacement[dof + 1] = y
    return displacement


def _affine_nodal_state(config, mpm, displacement_gradient):
    """Construct nodal values reproducing one particle's affine gradient."""
    gradient = np.asarray(displacement_gradient, dtype=np.float64)
    local_count = int(mpm.offset.to_numpy()[0])
    shape_gradients = mpm.dshape.to_numpy()[0, :local_count]
    local_values = np.vstack(
        [
            np.linalg.lstsq(
                shape_gradients.T,
                gradient[component],
                rcond=None,
            )[0]
            for component in range(config.DIM)
        ]
    ).T
    state = np.zeros(mpm.degree_of_freedom, dtype=np.float64)
    local_nodes = mpm.LnID.to_numpy()[0, :local_count]
    node_to_dof = mpm.node2dof.to_numpy()
    for local_node, grid_node in enumerate(local_nodes):
        block = int(node_to_dof[grid_node]) - 1
        state[config.DIM * block : config.DIM * (block + 1)] = local_values[local_node]
    return state


def _evaluate_friction(mpm, contact, displacement):
    mpm.grid_disp.from_numpy(displacement)
    contact.update_particle_pos(mpm.grid_disp)
    contact.ground_friction_initialize()
    assert int(contact.gfrictionNum[0]) == 1
    contact.friction_grad.fill(0.0)
    contact.friction_hash_matrix.reset_system()
    contact.assemble_ground_fully_implicit_friction_matrix(mpm.grid_disp)
    force = contact.friction_grad.to_numpy()[: mpm.active_dof].copy()
    jacobian = contact.friction_matrix(mpm.active_dof).toarray()
    return force, jacobian


def test_fully_implicit_point_plane_production_jacobian(tmp_path):
    config, mpm, contact = _build_point_plane_contact(tmp_path)
    last_force = None
    for speed_ratio in (0.0, 0.5, 2.0):
        tangential_displacement = (
            speed_ratio * float(contact.friction.epsv[0]) / contact.endpoint_velocity_displacement_scale
        )
        displacement = _translation_state(config, mpm, tangential_displacement, -2.0e-3)
        force, jacobian = _evaluate_friction(mpm, contact, displacement)

        # At zero speed the C1 (not C2) law gives first-order convergence for
        # a centered Jacobian difference, so use a correspondingly smaller
        # perturbation there.
        step = 2.0e-10 if speed_ratio == 0.0 else 2.0e-8
        finite_difference = np.zeros_like(jacobian)
        for column in range(mpm.active_dof):
            perturbation = np.zeros(mpm.degree_of_freedom, dtype=np.float64)
            perturbation[column] = step
            force_plus, _ = _evaluate_friction(mpm, contact, displacement + perturbation)
            force_minus, _ = _evaluate_friction(mpm, contact, displacement - perturbation)
            finite_difference[:, column] = (force_plus - force_minus) / (2.0 * step)

        scale = max(np.linalg.norm(jacobian, ord=np.inf), 1.0)
        assert np.linalg.norm(finite_difference + jacobian, ord=np.inf) / scale < 2.0e-6
        if speed_ratio > 0.0:
            assert not np.allclose(jacobian, jacobian.T, rtol=1.0e-10, atol=1.0e-10)
        last_force = force

    assert float(np.sum(last_force[0::2])) < 0.0


def test_residual_only_probe_preserves_all_gpu_triplet_buffers(tmp_path):
    config, mpm, contact = _build_point_plane_contact(tmp_path)
    displacement = _translation_state(config, mpm, 3.0e-5, -2.0e-3)
    mpm.grid_disp.from_numpy(displacement)

    contact.assemble_current_system(mpm.grid_disp, need_matrix=True)
    reference_rhs = mpm.rhs.to_numpy()[: mpm.active_dof].copy()

    def snapshot(matrix):
        raw = int(matrix.raw_non_diag_count[0])
        reduced = int(matrix.non_diag.element_pair_num[0])
        return (
            raw,
            reduced,
            matrix.diag.to_numpy().copy(),
            matrix.non_diag.blockI.to_numpy()[:raw].copy(),
            matrix.non_diag.blockJ.to_numpy()[:raw].copy(),
            matrix.non_diag.blockH.to_numpy()[:raw].copy(),
        )

    matrices = (
        mpm.hash_matrix,
        contact.barrier_hash_matrix,
        contact.friction_hash_matrix,
    )
    before = tuple(snapshot(matrix) for matrix in matrices)
    assert contact.assemble_current_system(mpm.grid_disp, need_matrix=False) is None
    probe_rhs = mpm.rhs.to_numpy()[: mpm.active_dof].copy()
    after = tuple(snapshot(matrix) for matrix in matrices)

    np.testing.assert_allclose(probe_rhs, reference_rhs, rtol=0.0, atol=1e-12)
    for before_matrix, after_matrix in zip(before, after):
        assert before_matrix[:2] == after_matrix[:2]
        for before_field, after_field in zip(before_matrix[2:], after_matrix[2:]):
            assert np.array_equal(before_field, after_field)


def test_cuda_monolithic_soft_particle_matrix_matches_cpu_oracle(
    monkeypatch,
    tmp_path,
):
    """Exercise the CUDA production dataflow with Taichi's CPU test backend."""
    from src.linear_solver.BuildTriplet import solve_csr_system
    from src.mpm.boundaries.BoundaryCondition import DirichletBoundary

    config, mpm, contact = _build_point_plane_contact(tmp_path)
    mpm.compute_mass_list(mpm.integration)
    constrained_grid = int(mpm.dof2node.to_numpy()[0])
    dirichlet = DirichletBoundary()
    dirichlet.append(
        [[config.DIM * constrained_grid]],
        [1.0e-5],
    )
    dirichlet.finalize(config.DIM * mpm.total_background_grid_num)
    mpm.dirichlet = dirichlet
    displacement = _translation_state(config, mpm, 4.0e-5, -2.0e-3)
    mpm.grid_disp.from_numpy(displacement)

    reference_matrix = contact.assemble_current_system(mpm.grid_disp).to_scipy(mpm.active_dof // config.DIM).tocsr()
    reference_rhs = mpm.rhs.to_numpy()[: mpm.active_dof].copy()
    reference_solution = solve_csr_system(reference_rhs, reference_matrix)

    device_matrix = contact.assemble_current_system(mpm.grid_disp)
    device_rhs = mpm.rhs.to_numpy()[: mpm.active_dof].copy()
    np.testing.assert_allclose(
        device_matrix.to_scipy(mpm.active_dof // config.DIM).toarray(),
        reference_matrix.toarray(),
        rtol=1.0e-11,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(device_rhs, reference_rhs, rtol=0.0, atol=1.0e-12)
    # Essential data is an absolute displacement.  Newton must solve for the
    # remaining correction, not reapply the absolute value every iteration.
    assert device_rhs[0] == pytest.approx(1.0e-5 - displacement[0])

    def scipy_fallback_forbidden(*_args, **_kwargs):
        raise AssertionError("CUDA SoftParticle solve must not materialize CSR")

    monkeypatch.setattr(contact.system_hash_matrix, "to_scipy", scipy_fallback_forbidden)
    result = contact.solve_current_system(mpm.grid_disp)
    assert result["backend"] == "taichi_cuda_monolithic_bicgstab"
    assert result["converged"]
    np.testing.assert_allclose(
        mpm.incre_resolution.to_numpy()[: mpm.active_dof],
        reference_solution,
        rtol=2.0e-7,
        atol=1.0e-9,
    )
    assert mpm.incre_resolution[0] == pytest.approx(1.0e-5 - displacement[0])

    # The fully implicit Armijo slope is evaluated as -R^T Kp from the
    # device-resident physical residual and the full nonsymmetric direction.
    # Check it against a finite difference of the constrained residual merit,
    # including the nonzero essential condition.
    slope = contact._fully_implicit_cuda_merit_slope(mpm.active_dof)
    direction = mpm.incre_resolution.to_numpy()[: mpm.active_dof].copy()

    def constrained_merit(state):
        trial = np.zeros(mpm.degree_of_freedom, dtype=np.float64)
        trial[: mpm.active_dof] = state
        mpm.grid_disp_temp.from_numpy(trial)
        assert contact.assemble_current_system(mpm.grid_disp_temp, need_matrix=False) is None
        residual = mpm.rhs.to_numpy()[: mpm.active_dof].copy()
        residual[0] = 1.0e-5 - state[0]
        return 0.5 * float(residual @ residual)

    step = 2.0e-7
    finite_difference_slope = (
        constrained_merit(displacement[: mpm.active_dof] + step * direction)
        - constrained_merit(displacement[: mpm.active_dof] - step * direction)
    ) / (2.0 * step)
    assert slope < 0.0
    assert slope == pytest.approx(finite_difference_slope, rel=5.0e-5, abs=1.0e-10)


def test_lagged_material_tangent_is_projected_before_global_scatter(
    tmp_path,
):
    """Official lagged IPC projects each material tangent before scatter."""
    config, mpm, contact = _build_point_plane_contact(tmp_path)
    contact.friction_mode = "lagged"

    displacement = _affine_nodal_state(
        config,
        mpm,
        np.diag([-0.5, 0.0]),
    )
    mpm.grid_disp.from_numpy(displacement)
    active_nodes = mpm.active_dof // config.DIM

    def material_matrix(project_spd):
        mpm.hash_matrix.reset_system()
        mpm.assemble_stiffness_matrix_hash(
            mpm.active_dof,
            mpm.grid_disp,
            project_spd=project_spd,
        )
        mpm.hash_matrix.finalize_taichi_assembly()
        dense = mpm.hash_matrix.to_scipy(active_nodes).toarray()
        return 0.5 * (dense + dense.T)

    exact = material_matrix(False)
    projected = material_matrix(True)
    exact_scale = max(float(np.linalg.norm(exact, ord=2)), 1.0)
    projected_scale = max(float(np.linalg.norm(projected, ord=2)), 1.0)
    assert float(np.linalg.eigvalsh(exact)[0]) < -1.0e-6 * exact_scale
    assert float(np.linalg.eigvalsh(projected)[0]) >= -1.0e-9 * projected_scale


def test_nonassociated_lagged_mpm_uses_physical_jacobian_and_force_balance(tmp_path):
    config, mpm, contact = _build_point_plane_contact(
        tmp_path,
        friction_mode="lagged",
        material="DruckerPrager",
        material_parameters={"Cohesion": 1.0, "FrictionAngle": 30.0, "DilationAngle": 0.0},
    )
    mpm.compute_mass_list(mpm.integration)
    assert contact.nonassociated_newton
    assert not mpm.has_lagged_material
    assert contact._cuda_monolithic_krylov_solver() == "BiCGSTAB"
    assert not contact.system_hash_matrix.matrix_symmetric
    assert not contact.system_hash_matrix.full_symmetric_input
    contact.friction_iterations = -1
    contact.begin_friction_step(mpm.grid_disp)
    displacement = _affine_nodal_state(config, mpm, np.diag([0.03, -0.02]))
    mpm.grid_disp.from_numpy(displacement)
    matrix = contact.assemble_current_system()
    dense = matrix.to_scipy(mpm.active_dof // config.DIM).toarray()
    direction = np.linspace(-0.1, 0.1, mpm.active_dof)
    increment = np.zeros_like(displacement)
    increment[: mpm.active_dof] = direction
    step = 1e-7
    forces = []
    for sign in (1, -1):
        mpm.grid_disp_temp.from_numpy(displacement + sign * step * increment)
        contact.assemble_current_system(mpm.grid_disp_temp, need_matrix=False)
        forces.append(mpm.rhs.to_numpy()[: mpm.active_dof].copy())
    np.testing.assert_allclose(dense @ direction, -(forces[0] - forces[1]) / (2 * step), rtol=2e-5, atol=1e-5)
    mpm.grid_disp.fill(0.0)
    contact.solve_lagged_friction_fixed_point(verbose=False)
    assert contact.last_inner_converged and contact.last_friction_converged
    assert contact.last_newton_residual <= mpm.tol
    assert contact.last_lagged_force_residual <= 1e-10 + 1e-8 * contact._friction_force_reference


def test_cuda_lagged_monolithic_matrix_is_spd_and_uses_pcg(
    monkeypatch,
    tmp_path,
):
    """Projected local blocks plus dynamic mass form the SPD solve."""
    from src.linear_solver.BuildTriplet import solve_csr_system
    from src.mpm.boundaries.BoundaryCondition import DirichletBoundary

    config, mpm, contact = _build_point_plane_contact(tmp_path)
    contact.friction_mode = "lagged"
    contact._validate_lagged_friction_configuration()
    mpm.compute_mass_list(mpm.integration)

    constrained_grid = int(mpm.dof2node.to_numpy()[0])
    dirichlet = DirichletBoundary()
    dirichlet.append([[config.DIM * constrained_grid]], [1.0e-5])
    dirichlet.finalize(config.DIM * mpm.total_background_grid_num)
    mpm.dirichlet = dirichlet
    displacement = _translation_state(config, mpm, 4.0e-5, -2.0e-3)
    mpm.grid_disp.from_numpy(displacement)
    contact.begin_friction_step(mpm.grid_disp)

    reference_matrix = contact.assemble_current_system(mpm.grid_disp).to_scipy(mpm.active_dof // config.DIM).tocsr()
    reference_dense = reference_matrix.toarray()
    symmetry_scale = max(float(np.linalg.norm(reference_dense, ord=np.inf)), 1.0)
    assert float(np.linalg.norm(reference_dense - reference_dense.T, ord=np.inf)) <= 1.0e-12 * symmetry_scale
    eigenvalues = np.linalg.eigvalsh(0.5 * (reference_dense + reference_dense.T))
    assert float(eigenvalues[0]) > 0.0
    reference_rhs = mpm.rhs.to_numpy()[: mpm.active_dof].copy()
    reference_solution = solve_csr_system(reference_rhs, reference_matrix)

    def scipy_fallback_forbidden(*_args, **_kwargs):
        raise AssertionError("CUDA lagged SoftParticle solve must not materialize CSR")

    monkeypatch.setattr(contact.system_hash_matrix, "to_scipy", scipy_fallback_forbidden)
    result = contact.solve_current_system(mpm.grid_disp)

    assert result["backend"] == "taichi_cuda_monolithic_pcg"
    assert contact.system_hash_matrix.solver == "PCG"
    assert result["converged"]
    np.testing.assert_allclose(
        mpm.incre_resolution.to_numpy()[: mpm.active_dof],
        reference_solution,
        rtol=2.0e-7,
        atol=1.0e-9,
    )


def test_surface_measure_scales_barrier_and_friction_once(tmp_path):
    config, mpm, contact = _build_point_plane_contact(tmp_path)
    displacement = _translation_state(config, mpm, 2.0e-5, -1.0e-3)

    def evaluate(measure):
        mpm.surface_measure.fill(measure)
        force, jacobian = _evaluate_friction(mpm, contact, displacement)
        contact.point_ground_distance()
        mpm.energy[None] = 0.0
        contact.get_barrier_energy()
        return force, jacobian, float(mpm.energy[None])

    force_one, jacobian_one, energy_one = evaluate(1.0)
    force_scaled, jacobian_scaled, energy_scaled = evaluate(3.25)

    np.testing.assert_allclose(force_scaled, 3.25 * force_one, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(jacobian_scaled, 3.25 * jacobian_one, rtol=1e-11, atol=1e-11)
    assert energy_scaled == pytest.approx(3.25 * energy_one, rel=1e-12)


def test_production_stribeck_jacobian_in_transition(tmp_path):
    """Verify the public MPM assembly wires every paper coefficient."""
    config, mpm, contact = _build_point_plane_contact(tmp_path)
    contact.friction.epsv[0] = 0.02
    contact.friction.stribeck_velocity[0] = 0.2
    contact.friction.mu_dynamic[0] = 0.31
    contact.friction.mu_static[0] = 0.82
    contact.friction.mu_viscous[0] = 0.07

    tangential_displacement = 0.073 / contact.endpoint_velocity_displacement_scale
    displacement = _translation_state(config, mpm, tangential_displacement, -2.0e-3)
    _, jacobian = _evaluate_friction(mpm, contact, displacement)

    step = 2.0e-8
    finite_difference = np.zeros_like(jacobian)
    for column in range(mpm.active_dof):
        perturbation = np.zeros(mpm.degree_of_freedom, dtype=np.float64)
        perturbation[column] = step
        force_plus, _ = _evaluate_friction(mpm, contact, displacement + perturbation)
        force_minus, _ = _evaluate_friction(mpm, contact, displacement - perturbation)
        finite_difference[:, column] = (force_plus - force_minus) / (2.0 * step)

    scale = max(np.linalg.norm(jacobian, ord=np.inf), 1.0)
    assert np.linalg.norm(finite_difference + jacobian, ord=np.inf) / scale < 3.0e-6
    assert not np.allclose(jacobian, jacobian.T)


def test_complete_fully_implicit_residual_jacobian_by_finite_difference(
    tmp_path,
):
    """Check inertia, material, IPC normal contact, and friction together."""
    config, mpm, contact = _build_point_plane_contact(tmp_path)
    contact.friction.epsv[0] = 0.02
    contact.friction.stribeck_velocity[0] = 0.2
    contact.friction.mu_dynamic[0] = 0.31
    contact.friction.mu_static[0] = 0.82
    contact.friction.mu_viscous[0] = 0.07
    mpm.compute_mass_list(mpm.integration)
    base = _translation_state(
        config,
        mpm,
        0.073 / contact.endpoint_velocity_displacement_scale,
        -2.0e-3,
    )

    def evaluate(displacement):
        mpm.grid_disp.from_numpy(displacement)
        matrix = contact.assemble_current_system(mpm.grid_disp).to_scipy(mpm.active_dof // config.DIM).toarray()
        residual = mpm.rhs.to_numpy()[: mpm.active_dof].copy()
        return residual, matrix

    _, analytic = evaluate(base)
    step = 2.0e-8
    finite_difference = np.zeros_like(analytic)
    for column in range(mpm.active_dof):
        perturbation = np.zeros(mpm.degree_of_freedom, dtype=np.float64)
        perturbation[column] = step
        plus, _ = evaluate(base + perturbation)
        minus, _ = evaluate(base - perturbation)
        finite_difference[:, column] = (plus - minus) / (2.0 * step)

    # GeoTaichi solves A p = rhs with A = -d(rhs)/du.
    scale = max(np.linalg.norm(analytic, ord=np.inf), 1.0)
    assert np.linalg.norm(finite_difference + analytic, ord=np.inf) / scale < 3.0e-6
    assert not np.allclose(analytic, analytic.T)


def test_fully_implicit_mode_rejects_lagged_outer_iterations():
    # Validation is deliberately covered through the public constructor by the
    # production test above; this small check documents the accepted mode name.
    from src.mpm.soft_particle.IPCMPM import _normalize_friction_mode

    assert _normalize_friction_mode("fully_implicit_experimental") == "fully_implicit"
    assert _normalize_friction_mode("lagged") == "lagged"
    with pytest.raises(ValueError, match="friction_mode"):
        _normalize_friction_mode("current-but-inexact")


def test_fully_implicit_point_plane_newton_smoke(tmp_path):
    config, mpm, contact = _build_point_plane_contact(tmp_path)
    mpm.compute_mass_list(mpm.integration)
    mpm.grid_disp.fill(0.0)
    _, correction_residual = contact.solve_friction_step(verbose=False)

    assert np.isfinite(correction_residual)
    assert np.isfinite(contact.last_fully_implicit_residual)
    assert contact.last_newton_iterations > 0
    assert contact.friction_mode == "fully_implicit"
    assert contact.last_friction_converged
    assert contact.last_friction_residual <= (
        contact.fully_implicit_residual_atol
        + contact.fully_implicit_residual_rtol * contact.last_fully_implicit_initial_residual
    )
    contact.update_particle_pos(mpm.grid_disp)
    position = contact.mpm.p_temp.to_numpy()[0]
    assert position[1] > 0.0


def test_fully_implicit_nonconvergence_rolls_back_and_fails(tmp_path):
    _, mpm, contact = _build_point_plane_contact(tmp_path)
    initial_displacement = mpm.grid_disp.to_numpy().copy()
    mpm.max_iters = 0

    with pytest.raises(RuntimeError, match="did not converge.*rolled back"):
        contact.solve_friction_step(verbose=False)

    np.testing.assert_array_equal(mpm.grid_disp.to_numpy(), initial_displacement)
    np.testing.assert_array_equal(mpm.grid_disp_temp.to_numpy(), initial_displacement)


def test_fully_implicit_rejects_invalid_parameters_and_infeasible_gap(
    tmp_path,
):
    config, mpm, contact = _build_point_plane_contact(tmp_path)

    contact.friction.epsv[0] = 0.0
    with pytest.raises(RuntimeError, match="finite positive epsv"):
        contact.solve_friction_step(verbose=False)
    contact.friction.epsv[0] = 0.1

    penetrating = _translation_state(config, mpm, 0.0, -0.05)
    mpm.grid_disp.from_numpy(penetrating)
    with pytest.raises(RuntimeError, match="strictly feasible positive starting gap"):
        contact.solve_friction_step(verbose=False)


def test_direct_mpm_dmin_activation_feasibility_and_ccd(tmp_path):
    config, mpm, contact = _build_point_plane_contact(tmp_path)
    contact.barrier.dmin[0] = 0.02
    contact.barrier.dhat[0] = 0.02

    # The active distance is dmin + dhat, not dhat alone.
    current = _translation_state(config, mpm, 0.0, -1.0e-3)
    mpm.grid_disp.from_numpy(current)
    contact.update_particle_pos(mpm.grid_disp)
    contact.point_ground_distance()
    assert int(contact.gbarrierNum[0]) == 1
    assert float(contact.gbarrier[0].distance) == pytest.approx(0.039)
    assert contact.validate_strict_feasibility(mpm.grid_disp) == pytest.approx(0.039)

    # CCD stops before distance reaches dmin and applies the requested 0.9
    # safety factor to the true impact time.
    correction = _translation_state(config, mpm, 0.0, -0.03)
    mpm.incre_resolution.from_numpy(correction)
    toc = float(contact.ground_ccd(0.9, mpm.incre_resolution))
    assert toc == pytest.approx(0.9 * (0.039 - 0.02) / 0.03)

    infeasible = _translation_state(config, mpm, 0.0, -0.021)
    mpm.grid_disp.from_numpy(infeasible)
    with pytest.raises(RuntimeError, match="above dmin"):
        contact.validate_strict_feasibility(mpm.grid_disp)


def test_direct_mpm_semi_ipc_recovers_dmin_violation_but_keeps_ccd(tmp_path):
    config, mpm, contact = _build_point_plane_contact(
        tmp_path,
        ipc_model="SemiIPC",
        friction_mode="lagged",
    )
    contact.barrier.dmin[0] = 0.02
    contact.barrier.dhat[0] = 0.02
    current = _translation_state(config, mpm, 0.0, -0.021)
    mpm.grid_disp.from_numpy(current)

    assert contact.validate_strict_feasibility(mpm.grid_disp) == pytest.approx(0.019)
    contact.point_ground_distance()
    assert int(contact.gbarrierNum[0]) == 1
    contact.assemble_current_system(mpm.grid_disp, need_matrix=True)
    contact.accept_update()
    assert np.isfinite(float(mpm.energy[None]))
    assert np.max(contact.semi_multiplier.to_numpy()) > 0.0

    correction = _translation_state(config, mpm, 0.0, -0.03)
    mpm.incre_resolution.from_numpy(correction)
    toc = float(contact.ground_ccd(0.9, mpm.incre_resolution))

    assert toc == pytest.approx(0.9 * 0.019 / 0.03)


def test_ul_material_ccd_uses_current_newton_state(tmp_path):
    """The feasibility cap is for u_k + alpha p, not p in isolation."""
    config, mpm, _ = _build_point_plane_contact(tmp_path)
    current_gradient = np.diag([-0.5, 0.0])
    increment_gradient = np.diag([-1.0, 0.0])
    mpm.grid_disp.from_numpy(_affine_nodal_state(config, mpm, current_gradient))
    mpm.incre_resolution.from_numpy(_affine_nodal_state(config, mpm, increment_gradient))

    # det(I + grad(u_k) + alpha grad(p)) = 0.5 - alpha.  With the
    # solver's slackness=0.8 convention, CCD stops at 20% of the current
    # determinant: 0.5 - alpha = 0.1, hence alpha=0.4.
    assert float(mpm.material_ccd(0.8)) == pytest.approx(0.4, abs=1.0e-12)

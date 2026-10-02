import os
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

pytestmark = [
    pytest.mark.integration,
    pytest.mark.dem,
    pytest.mark.ipc,
    pytest.mark.contact,
    pytest.mark.cpu,
    pytest.mark.slow,
    pytest.mark.serial,
]


_TAICHI_READY = False


def _init_once():
    global _TAICHI_READY
    if not _TAICHI_READY:
        from geotaichi import init

        init(
            arch="cpu",
            cpu_max_num_threads=2,
            offline_cache=False,
            log=False,
        )
        _TAICHI_READY = True


def _radial_roughness(direction):
    dx, dy, dz = direction
    return 0.09 * (0.7 * dx * dy + 0.5 * dy * dz - 0.4 * dz * dx) + 0.055 * (dx**3 - 3.0 * dx * dy**2)


def _irregular_signed_distance(x, y, z, axes):
    """Near-surface SDF approximation for a perturbed triaxial grain."""
    axes = np.asarray(axes, dtype=np.float64)
    scaled = np.stack((x / axes[0], y / axes[1], z / axes[2]))
    radius = np.sqrt(np.sum(scaled * scaled, axis=0))
    direction = np.zeros_like(scaled)
    regular = radius > 1.0e-14
    direction[:, regular] = scaled[:, regular] / radius[regular]
    surface_radius = 1.0 + _radial_roughness(direction)
    metric = np.sqrt(
        np.sum(
            (direction / axes.reshape(3, 1, 1, 1)) ** 2,
            axis=0,
        )
    )
    distance = np.empty_like(radius)
    distance[regular] = (radius[regular] - surface_radius[regular]) / metric[regular]
    distance[~regular] = -float(np.min(axes))
    return distance


def _irregular_levelset_object():
    axes = np.array([0.32, 0.23, 0.27], dtype=np.float64)
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    directions = np.asarray(mesh.vertices, dtype=np.float64)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    radial_scale = 1.0 + _radial_roughness(directions.T)
    mesh.vertices = directions * axes * radial_scale[:, None]
    mesh.remove_unreferenced_vertices()

    spacing = 0.04
    shape = np.array([23, 23, 23], dtype=np.int32)
    origin = np.array([-0.443, -0.443, -0.443], dtype=np.float64)
    grid_axes = [origin[d] + spacing * np.arange(shape[d]) for d in range(3)]
    x, y, z = np.meshgrid(*grid_axes, indexing="ij")
    values = _irregular_signed_distance(x, y, z, axes).flatten(order="F")
    grid = SimpleNamespace(
        start_point=origin,
        grid_space=spacing,
        gnum=shape,
        distance_field=values,
    )
    return SimpleNamespace(mesh=mesh, grid=grid, _reset=False), axes


def _rotation_z(degrees):
    angle = np.deg2rad(float(degrees))
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _build_case(
    tmp_path,
    *,
    surface_gap,
    simulation_time,
    auto_initialize=False,
    friction=0.0,
    ccd_type="ccd",
    contact_representation="LevelSet",
):
    from geotaichi import DEM

    levelset_object, _ = _irregular_levelset_object()
    left_angle = 27.0
    right_angle = -19.0
    left_vertices = np.asarray(levelset_object.mesh.vertices) @ _rotation_z(left_angle).T
    right_vertices = np.asarray(levelset_object.mesh.vertices) @ _rotation_z(right_angle).T
    center_distance = float(np.max(left_vertices[:, 0])) - float(np.min(right_vertices[:, 0])) + float(surface_gap)
    midpoint = 1.5
    centers = (
        midpoint - 0.5 * center_distance,
        midpoint + 0.5 * center_distance,
    )

    dem = DEM(log=False)
    dem.set_configuration(
        domain=[3.0, 2.0, 2.0],
        scheme="AffineBody",
        search="LinkedCell",
        gravity=[0.0, 0.0, 0.0],
        visualize=False,
        log=False,
    )
    dem.set_affine_body_parameters(
        assemble_type="COO",
        young_modulus=2.0e6,
        dhat=0.09,
        barrier_stiffness=5.0e6,
        max_newton_iteration=30,
        newton_tolerance=2.0e-7,
        line_search_max_iteration=40,
        direct_hessian_dofs=96,
        max_step=0.03,
        ccd_type=ccd_type,
        levelset_auto_initialize=auto_initialize,
        levelset_initial_anchor_stiffness=1.0,
        levelset_initial_gap_tolerance=1.0e-6,
        levelset_initial_maximum_stages=8,
    )
    dem.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_affine_body_number": 2,
            "surface_node_number": 128,
            "body_coordination_number": 4,
            "wall_coordination_number": 1,
            "compaction_ratio": [1.0, 1.0],
        },
        log=False,
    )
    dem.set_solver(
        {
            "Timestep": 1.0e-3,
            "SimulationTime": float(simulation_time),
            "SaveInterval": max(float(simulation_time), 1.0e-3),
            "SavePath": os.fspath(tmp_path),
        },
        log=False,
    )
    dem.add_attribute(
        materialID=0,
        attribute={
            "Density": 1000.0,
            "ForceLocalDamping": 0.0,
            "TorqueLocalDamping": 0.0,
        },
    )
    template_object = levelset_object
    if contact_representation == "TriangleMesh":
        template_object = SimpleNamespace(mesh=levelset_object.mesh, _reset=False)
    dem.add_template(
        template={
            "Name": "triaxial",
            "TemplateType": "AffineBody",
            "Object": template_object,
            "ContactRepresentation": contact_representation,
        }
    )
    dem.create_body(
        body={
            "BodyType": "AffineBody",
            "Template": [
                {
                    "Name": "triaxial",
                    "GroupID": 0,
                    "MaterialID": 0,
                    "BodyPoint": [centers[0], 1.0, 1.0],
                    "BodyOrientation": [0.0, 0.0, left_angle],
                    "InitialVelocity": [0.25, 0.0, 0.0],
                },
                {
                    "Name": "triaxial",
                    "GroupID": 0,
                    "MaterialID": 0,
                    "BodyPoint": [centers[1], 1.0, 1.0],
                    "BodyOrientation": [0.0, 0.0, right_angle],
                    "InitialVelocity": [-0.25, 0.0, 0.0],
                },
            ],
        }
    )
    dem.add_property(
        materialID1=0,
        materialID2=0,
        property={
            "Dhat": 0.09,
            "BarrierStiffness": 5.0e6,
            "Friction": float(friction),
        },
        dType="all",
    )
    return dem


def _levelset_friction_energy_gradient(operator, controls, reference):
    operator.friction_scale[0] = 1.0
    energy_with, gradient_with = operator.assemble(controls, controls, reference, need_matrix=False)
    active_friction = int(operator.levelset_friction_contacts[None])
    operator.friction_scale[0] = 0.0
    energy_without, gradient_without = operator.assemble(controls, controls, reference, need_matrix=False)
    operator.friction_scale[0] = 1.0
    return (
        energy_with - energy_without,
        gradient_with - gradient_without,
        active_friction,
    )


def _linear_momentum(state):
    momentum = np.zeros(3, dtype=np.float64)
    for body_id, body in enumerate(state.bodies):
        lumped = np.asarray(body["mass_matrix"]).sum(axis=1)
        momentum += np.sum(lumped[:, None] * state.v_y[body_id], axis=0)
    return momentum


def _surface_mean_velocity(state, body_id):
    return np.mean(
        np.asarray(state.bodies[body_id]["basis"]) @ np.asarray(state.v_y[body_id]),
        axis=0,
    )


def test_two_irregular_levelset_particles_collide_without_penetration(
    tmp_path,
):
    _init_once()
    dem = _build_case(
        tmp_path / "collision",
        surface_gap=0.045,
        simulation_time=0.01,
    )
    dem.add_essentials()
    dem.enginer.initialize(dem.sims, dem.scene)
    initial_momentum = _linear_momentum(dem.enginer.state)
    dem.run()

    state = dem.enginer.state
    final_momentum = _linear_momentum(state)
    left_velocity = _surface_mean_velocity(state, 0)
    right_velocity = _surface_mean_velocity(state, 1)
    final_relative_speed = right_velocity[0] - left_velocity[0]

    assert dem.enginer.operator.levelset_contact
    assert dem.enginer.last_candidate_pairs > 0
    assert dem.enginer.operator.levelset_minimum_gap[None] > 0.0
    assert 0.0 <= dem.enginer.last_ccd_step <= 1.0
    assert np.isfinite(state.y).all()
    assert np.isfinite(state.v_y).all()
    # Initial relative speed is -0.5 m/s. Normal IPC must reduce the closing
    # speed while preserving the isolated pair's total linear momentum.
    assert final_relative_speed > -0.5
    np.testing.assert_allclose(
        final_momentum,
        initial_momentum,
        rtol=0.0,
        atol=2.0e-8,
    )


def test_levelset_ccd_retains_gap_fraction_without_locking_motion(tmp_path):
    _init_once()
    dem = _build_case(
        tmp_path / "ccd_gap_fraction",
        surface_gap=0.02,
        simulation_time=0.0,
    )
    dem.add_essentials()
    dem.enginer.initialize(dem.sims, dem.scene)
    operator = dem.enginer.operator
    reference = dem.enginer.state.y.copy()
    operator.assemble(reference, reference, reference, need_matrix=False)
    initial_gap = float(operator.levelset_minimum_gap[None])

    direction = np.zeros_like(reference)
    direction[0, :, 0] = 0.04
    direction[1, :, 0] = -0.04
    alpha = operator.init_step_size(
        reference,
        direction,
        ccd_type="ccd",
        eta=0.2,
    )
    trial = reference + alpha * direction
    operator.assemble(trial, trial, reference, need_matrix=False)
    accepted_gap = float(operator.levelset_minimum_gap[None])

    assert 0.1 < alpha < 1.0
    assert 0.19 * initial_gap <= accepted_gap < 0.4 * initial_gap


def test_levelset_ccd_does_not_tunnel_across_padded_sdf(tmp_path):
    _init_once()
    dem = _build_case(
        tmp_path / "ccd_large_path",
        surface_gap=0.02,
        simulation_time=0.0,
    )
    dem.add_essentials()
    dem.enginer.initialize(dem.sims, dem.scene)
    operator = dem.enginer.operator
    reference = dem.enginer.state.y.copy()
    operator.assemble(reference, reference, reference, need_matrix=False)
    initial_gap = float(operator.levelset_minimum_gap[None])

    direction = np.zeros_like(reference)
    direction[0, :, 0] = 100.0
    alpha = operator.init_step_size(reference, direction, ccd_type="ccd", eta=0.2)
    trial = reference + alpha * direction
    operator.assemble(trial, trial, reference, need_matrix=False)

    assert 0.0 < alpha < 1.0e-3
    assert operator.levelset_minimum_gap[None] >= 0.19 * initial_gap


def test_levelset_signed_overlap_has_infinite_barrier_energy(tmp_path):
    _init_once()
    dem = _build_case(
        tmp_path / "signed_overlap",
        surface_gap=-0.04,
        simulation_time=0.0,
    )
    dem.add_essentials()
    dem.enginer.initialize(dem.sims, dem.scene)
    operator = dem.enginer.operator
    reference = dem.enginer.state.y.copy()
    energy = operator.assemble(reference, reference, reference, need_matrix=False)[0]

    assert operator.levelset_minimum_gap[None] < 0.0
    assert np.isinf(energy)


def test_levelset_normal_barrier_gradient_matches_finite_difference(tmp_path):
    _init_once()
    dem = _build_case(
        tmp_path / "normal_derivative",
        surface_gap=0.005,
        simulation_time=0.0,
    )
    dem.add_essentials()
    dem.enginer.initialize(dem.sims, dem.scene)
    operator = dem.enginer.operator
    reference = dem.enginer.state.y.reshape(-1).copy()
    energy, gradient = operator.assemble(reference, reference, reference, need_matrix=False)
    direction = np.zeros_like(reference)
    direction[:12:3] = 1.0
    epsilon = 2.0e-7
    plus = reference + epsilon * direction
    minus = reference - epsilon * direction
    energy_plus = operator.assemble(plus, plus, reference, need_matrix=False)[0]
    energy_minus = operator.assemble(minus, minus, reference, need_matrix=False)[0]

    assert energy > 0.0
    assert float(gradient @ direction) == pytest.approx(
        (energy_plus - energy_minus) / (2.0 * epsilon),
        rel=3.0e-5,
        abs=1.0e-8,
    )


def test_levelset_accd_conservative_advancement_keeps_positive_gap(tmp_path):
    _init_once()
    dem = _build_case(
        tmp_path / "accd_collision",
        surface_gap=0.02,
        simulation_time=0.006,
        ccd_type="accd",
    )
    dem.run()

    assert dem.enginer.last_ccd_type == "accd"
    assert 0.0 <= dem.enginer.last_ccd_step <= 1.0
    assert dem.enginer.operator.levelset_minimum_gap[None] > 0.0
    assert np.isfinite(dem.enginer.state.y).all()


def test_levelset_lagged_friction_energy_gradient_matches_finite_difference(
    tmp_path,
):
    _init_once()
    dem = _build_case(
        tmp_path / "friction_derivative",
        surface_gap=0.005,
        simulation_time=0.0,
        friction=0.45,
    )
    dem.add_essentials()
    dem.enginer.initialize(dem.sims, dem.scene)
    operator = dem.enginer.operator
    reference = dem.enginer.state.y.reshape(-1).copy()
    operator.initialize_contact_damping(reference, reference)
    controls = reference.copy()
    # Tangentially displace one complete affine body without changing its
    # shape.  The frozen target material coordinate and normal should make
    # the lagged friction potential exactly differentiable in this inner
    # solve.
    for control in range(4):
        controls[3 * control + 1] += 2.5e-5
    energy, gradient, active_friction = _levelset_friction_energy_gradient(operator, controls, reference)
    assert active_friction > 0
    assert energy > 0.0

    direction = np.zeros_like(controls)
    for control in range(4):
        direction[3 * control + 1] = 0.25
    epsilon = 2.0e-7
    plus = _levelset_friction_energy_gradient(operator, controls + epsilon * direction, reference)[0]
    minus = _levelset_friction_energy_gradient(operator, controls - epsilon * direction, reference)[0]
    finite_difference = (plus - minus) / (2.0 * epsilon)
    analytic = float(gradient @ direction)
    assert analytic == pytest.approx(finite_difference, rel=3.0e-5, abs=1.0e-8)


def test_overlapping_irregular_levelsets_are_pushed_to_ipc_feasibility(
    tmp_path,
):
    _init_once()
    dem = _build_case(
        tmp_path / "initialization",
        surface_gap=-0.04,
        simulation_time=0.0,
        auto_initialize=True,
    )
    dem.run()

    result = dem.enginer.last_nonpenetration_result
    assert result is not None
    assert result.success
    assert result.initial_minimum_gap < 0.0
    assert result.minimum_gap >= 1.0e-6
    assert np.linalg.norm(result.translations[0]) > 0.0
    np.testing.assert_allclose(
        result.translations[0],
        -result.translations[1],
        rtol=2.0e-6,
        atol=2.0e-8,
    )


def test_gpu_diffipc_projector_pushes_overlap_without_host_iterate_download(
    tmp_path,
):
    _init_once()
    dem = _build_case(
        tmp_path / "gpu_projection",
        surface_gap=-0.03,
        simulation_time=0.0,
        auto_initialize=False,
    )
    dem.add_essentials()

    from src.dem.engines.AffineBodyOperator import TaichiAffineBodyOperator
    from src.dem.engines.AffineBodyState import AffineBodyState
    from src.dem.engines.AffineDiffIPC import TaichiAffineDiffIPCProjector

    state = AffineBodyState.from_scene(dem.scene, dem.sims)
    operator = TaichiAffineBodyOperator(state, dem.sims, dem.scene, initialization_only=True)
    bounds = np.tile(np.array([-0.5, 0.5]), (2, 3, 1))
    projector = TaichiAffineDiffIPCProjector(operator, state.y, bounds)
    result = projector.solve(
        dhat=0.09,
        kappa=5.0e4,
        anchor_stiffness=1.0,
        continuation_ratio=0.15,
        maximum_stages=6,
        maximum_iterations=40,
        gap_tolerance=1.0e-6,
        gradient_tolerance=1.0e-8,
    )

    assert result["success"]
    assert result["initial_minimum_gap"] < 0.0
    assert result["minimum_gap"] >= 1.0e-6
    left = projector.translation[0]
    right = projector.translation[1]
    np.testing.assert_allclose(
        [left[0], left[1], left[2]],
        [-right[0], -right[1], -right[2]],
        rtol=2.0e-5,
        atol=2.0e-7,
    )
    import taichi as ti

    loss_gradient = ti.Vector.field(3, float, shape=2)
    loss_gradient.from_numpy(np.array([[1.0, 0.2, -0.1], [-0.4, 0.3, 0.2]]))
    adjoint = projector.solve_adjoint(loss_gradient)
    adjoint_numpy = adjoint.to_numpy()
    assert np.isfinite(adjoint_numpy).all()

    # DiffIPC validation must re-solve the inner projection.  Holding the
    # optimized translations fixed would test only the explicit loss term.
    direction = np.array([[0.3, -0.2, 0.1], [-0.15, 0.25, -0.05]], dtype=np.float64)
    loss_numpy = loss_gradient.to_numpy()
    base_controls = state.y.copy()

    def resolved_loss(sign, epsilon):
        controls = base_controls.copy()
        controls += sign * epsilon * direction[:, None, :]
        perturbed = TaichiAffineDiffIPCProjector(operator, controls, bounds)
        perturbed_result = perturbed.solve(
            dhat=0.09,
            kappa=5.0e4,
            anchor_stiffness=1.0,
            continuation_ratio=0.15,
            maximum_stages=6,
            maximum_iterations=40,
            gap_tolerance=1.0e-6,
            gradient_tolerance=1.0e-8,
        )
        assert perturbed_result["success"]
        final_centers = controls.mean(axis=1) + perturbed.translation.to_numpy()
        return float(np.sum(loss_numpy * final_centers))

    epsilon = 2.0e-5
    finite_difference = (resolved_loss(1.0, epsilon) - resolved_loss(-1.0, epsilon)) / (2.0 * epsilon)
    assert float(np.sum(adjoint_numpy * direction)) == pytest.approx(finite_difference, rel=2.0e-3, abs=2.0e-4)


def test_gpu_mesh_untangling_removes_triangle_intersections_before_diffipc(
    tmp_path,
):
    _init_once()
    dem = _build_case(
        tmp_path / "mesh_untangling",
        surface_gap=-0.04,
        simulation_time=0.0,
        contact_representation="TriangleMesh",
    )
    dem.add_essentials()

    from src.dem.engines.AffineBodyOperator import TaichiAffineBodyOperator
    from src.dem.engines.AffineBodyState import AffineBodyState
    from src.dem.engines.AffineDiffIPC import TaichiAffineMeshDiffIPCProjector

    state = AffineBodyState.from_scene(dem.scene, dem.sims)
    operator = TaichiAffineBodyOperator(state, dem.sims, dem.scene, initialization_only=True)
    bounds = np.tile(np.array([-0.5, 0.5]), (2, 3, 1))
    projector = TaichiAffineMeshDiffIPCProjector(operator, state.y, bounds)
    untangle = projector.untangle(
        gaussian_epsilon=0.05,
        gaussian_cutoff=0.2,
        gaussian_weight=1.0,
        minkowski_weight=1.0,
        inclusion_weight=1.0,
        anchor_stiffness=0.1,
        maximum_stages=5,
        maximum_iterations=120,
    )

    assert untangle["initial_overlap_count"] > 0
    assert untangle["success"]
    assert untangle["overlap_count"] == 0
    left = projector.translation[0]
    right = projector.translation[1]
    np.testing.assert_allclose(
        [left[0], left[1], left[2]],
        [-right[0], -right[1], -right[2]],
        rtol=2.0e-4,
        atol=2.0e-6,
    )

    result = projector.solve(
        dhat=0.09,
        kappa=5.0e4,
        anchor_stiffness=0.1,
        maximum_stages=3,
        maximum_iterations=30,
        gap_tolerance=1.0e-7,
        gradient_tolerance=1.0e-8,
        ccd_type="accd",
        accd_tolerance=1.0e-9,
    )
    assert result["success"]
    assert result["minimum_gap"] >= 1.0e-7
    assert result["untangling"]["overlap_count"] == 0

    import taichi as ti

    loss_gradient = ti.Vector.field(3, float, shape=2)
    loss_numpy = np.array([[1.0, 0.2, -0.1], [-0.4, 0.3, 0.2]], dtype=np.float64)
    loss_gradient.from_numpy(loss_numpy)
    adjoint = projector.solve_adjoint(loss_gradient).to_numpy()
    assert np.isfinite(adjoint).all()

    # Re-solve both the untangling basin and the final strict IPC equilibrium;
    # differentiating with translations held fixed would not test DiffIPC.
    direction = np.array([[0.3, -0.2, 0.1], [-0.15, 0.25, -0.05]], dtype=np.float64)
    base_controls = state.y.copy()

    def resolved_mesh_loss(sign, epsilon):
        controls = base_controls.copy()
        controls += sign * epsilon * direction[:, None, :]
        perturbed = TaichiAffineMeshDiffIPCProjector(operator, controls, bounds)
        perturbed_untangle = perturbed.untangle(
            gaussian_epsilon=0.05,
            gaussian_cutoff=0.2,
            gaussian_weight=1.0,
            minkowski_weight=1.0,
            inclusion_weight=1.0,
            anchor_stiffness=0.1,
            maximum_stages=5,
            maximum_iterations=120,
        )
        assert perturbed_untangle["success"]
        perturbed_result = perturbed.solve(
            dhat=0.09,
            kappa=5.0e4,
            anchor_stiffness=0.1,
            maximum_stages=3,
            maximum_iterations=30,
            gap_tolerance=1.0e-7,
            gradient_tolerance=1.0e-8,
            ccd_type="accd",
            accd_tolerance=1.0e-9,
        )
        assert perturbed_result["success"]
        final_centers = controls.mean(axis=1) + perturbed.translation.to_numpy()
        return float(np.sum(loss_numpy * final_centers))

    epsilon = 2.0e-5
    finite_difference = (resolved_mesh_loss(1.0, epsilon) - resolved_mesh_loss(-1.0, epsilon)) / (2.0 * epsilon)
    assert float(np.sum(adjoint * direction)) == pytest.approx(finite_difference, rel=5.0e-3, abs=5.0e-4)


@pytest.mark.parametrize("contact_representation", ["LevelSet", "TriangleMesh"])
def test_affine_distribute_uses_gpu_diffipc_to_meet_porosity_and_feasibility(tmp_path, contact_representation):
    _init_once()
    from geotaichi import DEM

    levelset_object, _ = _irregular_levelset_object()
    template_object = levelset_object
    if contact_representation == "TriangleMesh":
        template_object = SimpleNamespace(mesh=levelset_object.mesh, _reset=False)
    dem = DEM(log=False)
    dem.set_configuration(
        domain=[0.8, 0.8, 0.8],
        scheme="AffineBody",
        search="LinkedCell",
        gravity=[0.0, 0.0, 0.0],
        visualize=False,
        log=False,
    )
    dem.set_affine_body_parameters(
        assemble_type="COO",
        dhat=0.025,
        barrier_stiffness=2.0e4,
    )
    dem.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_affine_body_number": 32,
            "surface_node_number": 2048,
            "body_coordination_number": 16,
            "wall_coordination_number": 1,
            "compaction_ratio": [1.0, 1.0],
        },
        log=False,
    )
    dem.set_solver(
        {
            "Timestep": 1.0e-3,
            "SimulationTime": 0.0,
            "SaveInterval": 1.0,
            "SavePath": os.fspath(tmp_path / "distributed"),
        },
        log=False,
    )
    dem.add_attribute(materialID=0, attribute={"Density": 1000.0})
    dem.add_template(
        template={
            "Name": "grain",
            "TemplateType": "AffineBody",
            "Object": template_object,
            "ContactRepresentation": contact_representation,
        }
    )
    dem.add_property(
        materialID1=0,
        materialID2=0,
        property={
            "Dhat": 0.025,
            "BarrierStiffness": 2.0e4,
            "Friction": 0.0,
        },
        dType="all",
    )
    dem.add_region(
        region={
            "Name": "packing",
            "Type": "Rectangle",
            "BoundingBoxPoint": [0.1, 0.1, 0.1],
            "BoundingBoxSize": [0.6, 0.6, 0.6],
        }
    )
    dem.add_body(
        body={
            "GenerateType": "Distribute",
            "RegionName": "packing",
            "BodyType": "AffineBody",
            # Keep this integration test at two-to-three grains; larger dense
            # packs are covered by the dedicated GPU example/benchmark.
            "Porosity": 0.94,
            "DiffIPC": {
                "Dhat": 0.025,
                "BarrierStiffness": 2.0e4,
                "AnchorStiffness": 1.0,
                "GapTolerance": 1.0e-6,
                "MaximumStages": 8,
                "MaximumIterations": 60,
            },
            "Template": {
                "Name": "grain",
                "GroupID": 0,
                "MaterialID": 0,
                "Fraction": 1.0,
                "Radius": 0.11,
                "BodyOrientation": "uniform",
            },
        }
    )

    result = dem.scene.last_affine_diffipc_result
    assert len(dem.scene.affine_bodies) >= 2
    assert result["success"]
    assert result["minimum_gap"] >= 1.0e-6
    assert result["continuation_stages"] >= 1
    assert dem.scene.affine_bodies[0]["template"].contact_representation == contact_representation
    if contact_representation == "TriangleMesh":
        from src.dem.engines.AffineDiffIPC import TaichiAffineMeshDiffIPCProjector

        assert isinstance(
            dem.scene.last_affine_diffipc_projector,
            TaichiAffineMeshDiffIPCProjector,
        )
        assert result["untangling"]["overlap_count"] == 0
        assert not hasattr(
            dem.scene.affine_bodies[0]["template"],
            "_diffipc_levelset_cache",
        )

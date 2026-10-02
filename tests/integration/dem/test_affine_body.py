import os
import tempfile

import numpy as np
import pytest
import taichi as ti
from taichi.lang.impl import current_cfg

pytestmark = [
    pytest.mark.integration,
    pytest.mark.dem,
    pytest.mark.ipc,
    pytest.mark.contact,
    pytest.mark.cpu,
    pytest.mark.slow,
    pytest.mark.serial,
]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

_TAICHI_READY = False


def _init_once():
    global _TAICHI_READY
    if not _TAICHI_READY:
        from geotaichi import init

        init(arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
        _TAICHI_READY = True


def _build_two_body_case(
    assemble_type="MatrixFree",
    search="LinkedCell",
    with_wall=False,
    ccd_type="ccd",
    contact_damping_stiffness=0.0,
    contact_model="BarrierIPC",
    dhat=0.25,
    friction=0.2,
    joint=None,
    save_root=None,
):
    from geotaichi import DEM, polyhedron

    if save_root is None:
        save_path = tempfile.mkdtemp(prefix="geotaichi_affine_body_")
    else:
        case_name = (
            f"two_body_{assemble_type}_{search}_"
            f"{'wall' if with_wall else 'free'}_{ccd_type}_"
            f"cd{contact_damping_stiffness:g}"
        )
        save_path = os.path.join(os.fspath(save_root), case_name)
    mesh_path = os.path.join(ROOT, "tests/data/affine_octahedron.obj")

    dem = DEM(log=False)
    dem.set_configuration(
        domain=[4.0, 2.0, 2.0],
        scheme="AffineBody",
        search=search,
        gravity=[0.0, 0.0, 0.0],
        visualize=False,
        log=False,
    )
    dem.set_affine_body_parameters(
        contact_model=contact_model,
        assemble_type=assemble_type,
        young_modulus=2.0e4,
        # This is a strict nonlinear-convergence integration test.  Two
        # Newton iterations are insufficient once a wall barrier is active;
        # do not rely on the historical unconverged-step acceptance.
        max_newton_iteration=20,
        line_search_max_iteration=8,
        direct_hessian_dofs=96,
        max_step=0.05,
        ccd_type=ccd_type,
    )
    memory = {
        "max_material_number": 1,
        "max_affine_body_number": 2,
        "surface_node_number": 16,
        "body_coordination_number": 4,
        "wall_coordination_number": 4,
        "compaction_ratio": [1.0, 1.0],
    }
    if search == "HierarchicalLinkedCell":
        memory["hierarchical_size"] = [0.5]
    if with_wall:
        memory["max_plane_number"] = 1
    dem.memory_allocate(memory=memory, log=False)
    dem.set_solver(
        {
            "Timestep": 1.0e-3,
            "SimulationTime": 1.0e-3,
            "SaveInterval": 1.0e-3,
            "SavePath": save_path,
        },
        log=False,
    )
    dem.add_attribute(materialID=0, attribute={"Density": 1200.0})
    dem.add_template(template={"Name": "oct", "TemplateType": "AffineBody", "Object": polyhedron(file=mesh_path)})
    dem.create_body(
        body={
            "BodyType": "AffineBody",
            "Template": [
                {
                    "Name": "oct",
                    "GroupID": 0,
                    "MaterialID": 0,
                    # The octahedron has x support radius 0.2.  Keep a
                    # positive 0.02 gap while remaining inside Dhat=0.25;
                    # the former centers (1.86, 2.14) overlapped by 0.12 and
                    # made the logarithmic IPC barrier infinite at t=0.
                    "BodyPoint": [1.79, 1.0, 1.0],
                    "ScaleFactor": 0.2,
                    "InitialVelocity": [0.1, 0.0, 0.0],
                },
                {
                    "Name": "oct",
                    "GroupID": 0,
                    "MaterialID": 0,
                    "BodyPoint": [2.21, 1.0, 1.0],
                    "ScaleFactor": 0.2,
                    "InitialVelocity": [-0.1, 0.0, 0.0],
                },
            ],
        }
    )
    if joint is not None:
        dem.add_joint(joint)
    if with_wall:
        dem.add_wall(
            {
                "WallType": "Plane",
                "MaterialID": 0,
                # Keep the initial IPC state strictly feasible.  The old
                # z=0.8 plane passed exactly through the lowest octahedron
                # vertex, for which the logarithmic barrier is infinite.
                "WallCenter": np.array([0.0, 0.0, 0.79]),
                "OuterNormal": np.array([0.0, 0.0, 1.0]),
            }
        )
    dem.add_property(
        materialID1=0,
        materialID2=0,
        property={
            "Dhat": dhat,
            "BarrierStiffness": 2.0e4,
            "ContactDampingStiffness": contact_damping_stiffness,
            "Friction": friction,
        },
        dType="all",
    )
    dem.run()
    return dem


def _equilibrium_barrier_gap(normal_load, collision_weight, dhat, kappa):
    """Solve the IPC point-plane barrier force balance."""

    from src.physics_model.contact_model.ipc.IPC import (
        ipc_barrier_distance_terms_py,
    )

    def weighted_normal_force(gap):
        _, gradient, _ = ipc_barrier_distance_terms_py(
            gap,
            dhat,
            kappa=kappa,
        )
        return collision_weight * max(-gradient, 0.0)

    lower = max(1.0e-12 * dhat, np.finfo(float).tiny)
    upper = dhat * (1.0 - 1.0e-12)
    if not (weighted_normal_force(lower) > normal_load and weighted_normal_force(upper) < normal_load):
        raise RuntimeError("failed to bracket the IPC barrier equilibrium gap")

    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if weighted_normal_force(midpoint) > normal_load:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _build_incline_case(
    mu=0.2,
    ccd_type="accd",
    save_root=None,
    incline_degrees=30.0,
):
    from geotaichi import DEM, polyhedron

    theta = np.deg2rad(float(incline_degrees))
    normal = np.array([-np.sin(theta), 0.0, np.cos(theta)], dtype=np.float64)
    upslope = np.array([np.cos(theta), 0.0, np.sin(theta)], dtype=np.float64)
    downslope = -upslope
    scale = 0.25
    half_extent = 0.5 * scale
    support = half_extent
    density = 1000.0
    gravity = 9.81
    dhat = 0.03
    barrier_stiffness = 8.0e5
    nonlinear_velocity_tolerance = 1.0e-6
    # For a triangulated cube, the four vertices on the contacting face carry
    # a total barycentric surface area of 3 * scale**2.  Start at the gap where
    # the squared-distance IPC barrier balances the normal component
    # of gravity.  Otherwise the normal transient contaminates the tangential
    # sliding oracle through Coulomb friction and makes the result depend on
    # the sampled rebound phase.
    collision_weight = 3.0 * scale * scale
    mass = density * scale**3
    initial_gap = _equilibrium_barrier_gap(
        mass * gravity * np.cos(theta),
        collision_weight,
        dhat,
        barrier_stiffness,
    )
    initial_center = 0.55 * upslope + np.array([0.0, 0.5, 0.0]) + normal * (support + initial_gap)
    if save_root is None:
        save_path = tempfile.mkdtemp(prefix="geotaichi_affine_incline_")
    else:
        save_path = os.path.join(os.fspath(save_root), f"incline_mu{mu:g}_{ccd_type}")
    mesh_path = os.path.join(ROOT, "tests/data/affine_cube.obj")

    dem = DEM(log=False)
    dem.set_configuration(
        domain=[2.0, 1.0, 2.0],
        scheme="AffineBody",
        search="LinkedCell",
        gravity=[0.0, 0.0, -gravity],
        visualize=False,
        log=False,
    )
    dem.set_affine_body_parameters(
        assemble_type="MatrixFree",
        young_modulus=5.0e5,
        friction_epsv=1.0e-8,
        # The analytical Coulomb cancellation below uses the current normal
        # force.  Iterate the lagged normal force/tangent basis to its fixed
        # point instead of validating the intentionally one-pass reference
        # default (friction_iterations=1).
        friction_iterations=-1,
        friction_max_iterations=20,
        friction_tolerance=nonlinear_velocity_tolerance,
        # A four-iteration budget leaves the first low-friction step at an
        # O(1e-5) residual while the configured nonlinear tolerance is
        # O(1e-7).  Keep the strict no-unconverged-step contract and give the
        # wall/friction solve the same complete Newton budget as the other
        # affine-body integration cases above.
        # A 1e-6 m/s correction threshold means at most 1e-9 m per step and
        # 8e-8 m over this whole run, more than three orders below the low-mu
        # trajectory's 1% physical error budget. It remains four orders
        # stricter than the 1e-2 reference IPC default.
        newton_tolerance=nonlinear_velocity_tolerance,
        max_newton_iteration=100,
        line_search_max_iteration=50,
        max_step=0.02,
        ccd_type=ccd_type,
        ccd_eta=0.2,
        accd_tolerance=1.0e-6,
    )
    dem.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_affine_body_number": 1,
            "surface_node_number": 16,
            "max_plane_number": 1,
            "body_coordination_number": 4,
            "wall_coordination_number": 4,
            "compaction_ratio": [1.0, 1.0],
        },
        log=False,
    )
    time_step = 1.0e-3
    step_count = 80
    total_time = step_count * time_step
    dem.set_solver(
        {
            "Timestep": time_step,
            "SimulationTime": total_time,
            "SaveInterval": total_time,
            "SavePath": save_path,
        },
        log=False,
    )
    dem.add_attribute(
        materialID=0,
        attribute={"Density": density, "ForceLocalDamping": 0.0, "TorqueLocalDamping": 0.0},
    )
    dem.add_template(template={"Name": "cube", "TemplateType": "AffineBody", "Object": polyhedron(file=mesh_path)})
    dem.create_body(
        body={
            "BodyType": "AffineBody",
            "Template": {
                "Name": "cube",
                "GroupID": 0,
                "MaterialID": 0,
                "BodyPoint": initial_center.tolist(),
                "ScaleFactor": scale,
                "BodyOrientation": [0.0, -float(incline_degrees), 0.0],
                "InitialVelocity": [0.0, 0.0, 0.0],
                "Friction": mu,
            },
        }
    )
    dem.add_wall(
        {
            "WallType": "Plane",
            "MaterialID": 0,
            "WallCenter": np.array([0.0, 0.0, 0.0]),
            "OuterNormal": normal,
        }
    )
    dem.add_property(
        materialID1=0,
        materialID2=0,
        property={
            "Dhat": dhat,
            "BarrierStiffness": barrier_stiffness,
            "Friction": mu,
        },
        dType="all",
    )
    dem.run()
    vertices, _, _, _ = dem.enginer.state.surface_mesh()
    initial_vertices = np.asarray(dem.enginer.operator.rest_x_np, dtype=np.float64)
    final_center = np.mean(vertices, axis=0)
    center_increment = final_center - initial_center
    displacement = float(np.dot(center_increment, downslope))
    normal_displacement = float(np.dot(center_increment, normal))
    # While sliding, summing the tangential and mu-scaled normal equations
    # removes every normal-barrier transient:
    #
    #   s_ddot + mu * n_ddot = g (sin(theta) - mu cos(theta)).
    #
    # This remains true for the affine body's internal deformation forces,
    # whose mass-weighted resultant is zero.  It is a substantially sharper
    # oracle than pretending the finite-stiffness body has constant N(t).
    compensated_displacement = displacement + mu * normal_displacement
    acceleration = max(
        0.0,
        gravity * (np.sin(theta) - mu * np.cos(theta)),
    )
    # GeoTaichi advances velocity and then position (backward Euler).  For a
    # constant acceleration the exact discrete trajectory is
    # a * dt^2 * N * (N + 1) / 2, not the continuous 0.5*a*T^2 expression.
    expected = acceleration * time_step * time_step * step_count * (step_count + 1) / 2.0
    initial_gaps = initial_vertices @ normal
    contact_vertices = np.isclose(
        initial_gaps,
        np.min(initial_gaps),
        rtol=0.0,
        atol=1.0e-12,
    )
    interface_slip = float(
        np.max(np.abs((vertices[contact_vertices] - initial_vertices[contact_vertices]) @ downslope))
    )
    return (
        dem,
        displacement,
        normal_displacement,
        compensated_displacement,
        expected,
        interface_slip,
    )


def test_affine_body_backends(tmp_path):
    _init_once()
    for backend in ("MatrixFree", "COO"):
        dem = _build_two_body_case(assemble_type=backend, save_root=tmp_path)
        assert dem.enginer.last_linear_backend == backend
        assert np.isfinite(dem.enginer.state.y).all()
        assert dem.enginer.state.body_num == 2
        if backend == "MatrixFree":
            # Exercise the same Taichi kernels used by the CUDA-resident
            # nonlinear path on the CPU test backend.  Only the host/device
            # state-loading wrapper and gravity source differ, so energy,
            # residual and assembled Jacobian must agree with the CPU oracle.
            from src.dem.engines.AffineBodyOperator import MATRIX_COO

            engine = dem.enginer
            state = engine.state
            host_energy, host_gradient = engine.operator.assemble(
                state.y,
                state.tilde_y,
                state.hat_y,
                need_matrix=True,
                matrix_mode=MATRIX_COO,
            )
            host_matrix = engine.coo_matrix._to_numpy().copy()
            engine.operator.load_device_step_state(state.y, state.tilde_y, state.hat_y)
            device_energy = engine.operator.assemble_device(need_matrix=True, matrix_mode=MATRIX_COO)
            device_gradient = engine.operator.grad.to_numpy()[: engine.operator.control_num].reshape(-1)
            device_matrix = engine.coo_matrix._to_numpy().copy()

            assert device_energy == pytest.approx(host_energy, rel=1.0e-12, abs=1.0e-14)
            np.testing.assert_allclose(
                device_gradient,
                host_gradient,
                rtol=1.0e-12,
                atol=1.0e-13,
            )
            np.testing.assert_allclose(
                device_matrix,
                host_matrix,
                rtol=1.0e-12,
                atol=1.0e-13,
            )
    dem = _build_two_body_case(assemble_type="HashTriplet", save_root=tmp_path)
    assert dem.enginer.last_linear_backend == "HashTriplet"
    assert np.isfinite(dem.enginer.state.y).all()


def test_affine_body_wall_contact(tmp_path):
    _init_once()
    dem = _build_two_body_case(assemble_type="MatrixFree", with_wall=True, save_root=tmp_path)
    assert len(dem.enginer.state.bodies) == 2
    assert np.isfinite(dem.enginer.state.v_y).all()
    count = min(
        int(dem.enginer.operator.friction_contact_count[0]),
        dem.enginer.operator.friction_contact_capacity,
    )
    weights = dem.enginer.operator.friction_contact_weights.to_numpy()[:count]
    assert any(
        np.allclose(weight, [1.0, 0.0, 0.0, 0.0]) for weight in weights
    ), "planar wall friction must be present in the frozen lagged cache"
    assert os.path.exists(os.path.join(dem.sims.path, "walls", "DEMWall000001.npz"))


def test_affine_body_semi_ipc_device_contact(tmp_path):
    _init_once()
    dem = _build_two_body_case(
        assemble_type="HashTriplet",
        contact_model="SemiIPC",
        save_root=tmp_path,
    )
    diagnostics = dem.enginer.diagnostics_snapshot()["contact"]
    assert diagnostics["model"] == "SemiIPC"
    assert np.isfinite(dem.enginer.state.y).all()


def test_affine_revolute_joint_energy_motor_limits_damping_and_collision_filter(tmp_path):
    _init_once()
    from src.dem.engines.AffineBodyOperator import MATRIX_COO

    anchor = np.array([2.0, 1.0, 1.0], dtype=np.float64)
    dem = _build_two_body_case(
        assemble_type="COO",
        with_wall=True,
        joint={
            "JointType": "Revolute",
            "BodyID1": 0,
            "BodyID2": 1,
            "WorldAnchor": anchor,
            "WorldAxis": [0.0, 0.0, 1.0],
            "PositionStiffness": 1.0e5,
            "AxisStiffness": 1.0e5,
            "AngleLimit": [-20.0, 20.0],
            "LimitStiffness": 4.0e4,
        },
        save_root=tmp_path,
    )
    operator = dem.enginer.operator
    initial = np.asarray([body["y"] for body in dem.enginer.state.bodies], dtype=np.float64)

    def rotate_body(body_id, degrees, axis="z"):
        angle = np.deg2rad(degrees)
        c, s = np.cos(angle), np.sin(angle)
        if axis == "z":
            rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        else:
            rotation = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        controls = initial.copy()
        controls[body_id] = anchor + (controls[body_id] - anchor) @ rotation.T
        return controls

    def joint_terms(controls, history=None, need_matrix=False):
        flat = np.ascontiguousarray(controls.reshape((-1, 3)), dtype=np.float64)
        old = flat if history is None else np.ascontiguousarray(history.reshape((-1, 3)), dtype=np.float64)
        operator.y.from_numpy(flat)
        operator.hat_y.from_numpy(old)
        operator._clear_system(bool(need_matrix), MATRIX_COO)
        operator._assemble_joints(bool(need_matrix), MATRIX_COO)
        if need_matrix:
            operator.finalize_coo_assembly()
        gradient = operator.grad.to_numpy()[: operator.control_num].reshape(-1)
        matrix = operator.coo_matrix._to_numpy().copy() if need_matrix else None
        return float(operator.energy[None]), gradient, matrix

    collision_filter = operator.joint_collision_disabled.to_numpy()
    assert collision_filter[0, 1] == collision_filter[1, 0] == 1
    assert collision_filter[0, 0] == collision_filter[1, 1] == 0

    operator.y.from_numpy(np.ascontiguousarray(initial.reshape((-1, 3))))
    operator._reconstruct_vertices()
    candidate_count = operator.neighbor.update(
        operator.x,
        operator.dx,
        operator.faces,
        operator.edges,
        operator.node2body,
        operator.face2body,
        operator.edge2body,
        operator.dhat,
        swept=False,
    )
    assert candidate_count > 0
    operator._clear_system(False, MATRIX_COO)
    operator._assemble_particle_contacts(
        False,
        operator.neighbor.candidate_count,
        operator.neighbor.candidate_vertex,
        operator.neighbor.candidate_face,
        MATRIX_COO,
    )
    operator._assemble_edge_contacts(
        False,
        operator.neighbor.edge_candidate_count,
        operator.neighbor.candidate_edge0,
        operator.neighbor.candidate_edge1,
    )
    assert float(operator.energy[None]) == pytest.approx(0.0, abs=1.0e-15)
    operator._clear_system(True, MATRIX_COO)
    operator._assemble_body_pair_barrier_hessian(
        operator.neighbor.candidate_count,
        operator.neighbor.candidate_vertex,
        operator.neighbor.candidate_face,
        operator.neighbor.edge_candidate_count,
        operator.neighbor.candidate_edge0,
        operator.neighbor.candidate_edge1,
        operator.scale,
        MATRIX_COO,
    )
    operator.finalize_coo_assembly()
    assert np.all(operator.body_pair_segment_base.to_numpy() == -1)
    assert not np.any(operator.coo_matrix._to_numpy())
    operator._clear_system(False, MATRIX_COO)
    operator._assemble_wall_contacts(False, MATRIX_COO)
    assert float(operator.energy[None]) > 0.0

    operator.joint_limit_enabled[0] = 0
    operator.joint_motor_stiffness[0] = 0.0
    operator.joint_damping[0] = 0.0
    energy_initial = joint_terms(initial)[0]
    energy_hinge = joint_terms(rotate_body(1, 30.0))[0]
    translated = initial.copy()
    translated[1] += np.array([0.02, 0.0, 0.0])
    energy_translation = joint_terms(translated)[0]
    energy_wrong_axis = joint_terms(rotate_body(1, 10.0, axis="x"))[0]
    assert energy_initial == pytest.approx(0.0, abs=1.0e-14)
    assert energy_hinge == pytest.approx(0.0, abs=1.0e-13)
    assert energy_translation > 0.0
    assert energy_wrong_axis > 0.0

    operator.joint_motor_stiffness[0] = 3.0e4
    dem.set_joint_target_angle(0, 30.0)
    assert joint_terms(rotate_body(1, 30.0))[0] == pytest.approx(0.0, abs=1.0e-13)
    assert joint_terms(initial)[0] > 0.0

    operator.joint_motor_stiffness[0] = 0.0
    operator.joint_limit_enabled[0] = 1
    operator.joint_limit_lower[0] = np.deg2rad(-10.0)
    operator.joint_limit_upper[0] = np.deg2rad(10.0)
    assert joint_terms(rotate_body(1, 5.0))[0] == pytest.approx(0.0, abs=1.0e-13)
    assert joint_terms(rotate_body(1, 30.0))[0] > 0.0

    operator.joint_limit_enabled[0] = 0
    operator.joint_damping[0] = 20.0
    moved = rotate_body(1, 5.0)
    assert joint_terms(initial, history=initial)[0] == pytest.approx(0.0, abs=1.0e-14)
    assert joint_terms(moved, history=initial)[0] > 0.0

    operator.joint_damping[0] = 0.0
    _, gradient, hessian = joint_terms(translated, need_matrix=True)
    np.testing.assert_allclose(hessian, hessian.T, rtol=1.0e-12, atol=1.0e-13)
    assert np.linalg.eigvalsh(hessian)[0] >= -1.0e-12
    direction = np.random.default_rng(7).normal(size=gradient.size)
    direction /= np.linalg.norm(direction)
    epsilon = 1.0e-6
    energy_plus = joint_terms(translated.reshape(-1) + epsilon * direction)[0]
    energy_minus = joint_terms(translated.reshape(-1) - epsilon * direction)[0]
    derivative = (energy_plus - energy_minus) / (2.0 * epsilon)
    assert derivative == pytest.approx(float(gradient @ direction), rel=1.0e-7, abs=1.0e-10)


def test_affine_revolute_joint_barrieripc_adjoint(tmp_path):
    _init_once()
    dem = _build_two_body_case(
        assemble_type="COO",
        dhat=0.01,
        friction=0.0,
        joint={
            "JointType": "Revolute",
            "BodyID1": 0,
            "BodyID2": 1,
            "WorldAnchor": [2.0, 1.0, 1.0],
            "WorldAxis": [0.0, 0.0, 1.0],
            "PositionStiffness": 1.0e5,
            "AxisStiffness": 1.0e5,
            "MotorStiffness": 2.0e4,
            "TargetAngle": 5.0,
            "Damping": 20.0,
        },
        save_root=tmp_path,
    )
    seed = np.random.default_rng(11).normal(size=dem.enginer.state.y.shape)
    vjp = dem.differentiate_affine_step(seed)
    operator = dem.enginer.operator
    adjoint = vjp["adjoint"].reshape(-1)
    matrix = dem.enginer.adjoint_triplet.to_scipy(operator.control_num).tocsc()
    np.testing.assert_allclose(matrix.T @ adjoint, seed.reshape(-1), rtol=1.0e-8, atol=1.0e-10)
    translation = np.tile([0.3, -0.2, 0.1], (operator.control_num, 1))
    mass_translation = np.empty_like(translation)
    mass = operator.mass.to_numpy()[: operator.body_num]
    body_mass = operator.body_mass.to_numpy()[: operator.body_num]
    force_damp = operator.force_damp.to_numpy()[: operator.body_num]
    for body in range(operator.body_num):
        mass_translation[4 * body : 4 * body + 4] = (
            mass[body] @ translation[4 * body : 4 * body + 4]
            + 0.25 * float(dem.sims.delta) * body_mass[body] * force_damp[body] * translation[4 * body : 4 * body + 4]
        )
    np.testing.assert_allclose(
        matrix @ translation.reshape(-1),
        mass_translation.reshape(-1),
        rtol=2.0e-10,
        atol=2.0e-10,
    )

    def residual():
        return operator.assemble(
            dem.enginer.state.y,
            dem.enginer.state.tilde_y,
            dem.enginer.state.hat_y,
            need_matrix=False,
            project_spd=False,
        )[1]

    base_y = dem.enginer.state.y.copy()
    state_direction = np.random.default_rng(19).normal(size=base_y.shape)
    state_direction /= np.linalg.norm(state_direction)
    state_epsilon = 1.0e-6
    plus_state = operator.assemble(
        base_y + state_epsilon * state_direction,
        dem.enginer.state.tilde_y,
        dem.enginer.state.hat_y,
        need_matrix=False,
        project_spd=False,
    )[1]
    minus_state = operator.assemble(
        base_y - state_epsilon * state_direction,
        dem.enginer.state.tilde_y,
        dem.enginer.state.hat_y,
        need_matrix=False,
        project_spd=False,
    )[1]
    operator.assemble(
        base_y,
        dem.enginer.state.tilde_y,
        dem.enginer.state.hat_y,
        need_matrix=False,
        project_spd=False,
    )
    np.testing.assert_allclose(
        matrix @ state_direction.reshape(-1),
        (plus_state - minus_state) / (2.0 * state_epsilon),
        rtol=2.0e-6,
        atol=2.0e-8,
    )

    epsilon = 1.0e-6
    angle = float(operator.joint_target_angle[0])
    operator.joint_target_angle[0] = angle + epsilon
    plus = residual()
    operator.joint_target_angle[0] = angle - epsilon
    minus = residual()
    operator.joint_target_angle[0] = angle
    expected_target = -adjoint @ ((plus - minus) / (2.0 * epsilon))
    assert vjp["joint_target_angle_radians"][0] == pytest.approx(expected_target, rel=2.0e-7, abs=1.0e-10)

    gravity = dem.enginer.state.gravity.copy()
    for component in range(3):
        dem.enginer.state.gravity = gravity.copy()
        dem.enginer.state.gravity[component] += epsilon
        plus = residual()
        dem.enginer.state.gravity = gravity.copy()
        dem.enginer.state.gravity[component] -= epsilon
        minus = residual()
        expected = -adjoint @ ((plus - minus) / (2.0 * epsilon))
        assert vjp["gravity"][component] == pytest.approx(expected, rel=2.0e-7, abs=1.0e-10)
    dem.enginer.state.gravity = gravity

    for body in range(operator.body_num):
        young = float(operator.young[body])
        delta = max(1.0e-6, 1.0e-5 * young)
        operator.young[body] = young + delta
        plus = residual()
        operator.young[body] = young - delta
        minus = residual()
        operator.young[body] = young
        expected = -adjoint @ ((plus - minus) / (2.0 * delta))
        assert vjp["affine_young_modulus"][body] == pytest.approx(expected, rel=2.0e-7, abs=1.0e-10)

    damping = float(operator.joint_damping[0])
    operator.joint_damping[0] = damping + epsilon
    plus = residual()
    operator.joint_damping[0] = damping - epsilon
    minus = residual()
    operator.joint_damping[0] = damping
    expected_damping = -adjoint @ ((plus - minus) / (2.0 * epsilon))
    assert vjp["joint_damping"][0] == pytest.approx(expected_damping, rel=2.0e-7, abs=1.0e-10)

    trajectory = dem.differentiable_affine(steps=2)
    initial_y = operator.y.to_numpy()[: operator.control_num].copy()
    initial_v = operator.velocity_y.to_numpy()[: operator.control_num].copy()
    initial_time = float(dem.sims.current_time)
    initial_step = int(dem.sims.current_step)
    trajectory.step()
    trajectory.step()
    state_seed_y = 0.1 * np.random.default_rng(12).normal(size=initial_y.shape)
    state_seed_v = 1.0e-3 * np.random.default_rng(13).normal(size=initial_v.shape)
    state_vjp = trajectory.backward(state_seed_y, state_seed_v)
    direction_y = np.random.default_rng(14).normal(size=initial_y.shape)
    direction_v = np.random.default_rng(15).normal(size=initial_v.shape)

    def state_objective(y, velocity, joint_damping=None):
        operator.y.from_numpy(np.ascontiguousarray(y))
        operator.velocity_y.from_numpy(np.ascontiguousarray(velocity))
        if joint_damping is not None:
            operator.joint_damping[0] = joint_damping
        dem.sims.current_time = initial_time
        dem.sims.current_step = initial_step
        engine = dem.enginer
        for _ in range(2):
            engine.step(dem.sims, engine.scene)
            dem.sims.current_time += dem.sims.delta
            dem.sims.current_step += 1
        return float(
            np.sum(state_seed_y * operator.y.to_numpy()[: operator.control_num])
            + np.sum(state_seed_v * operator.velocity_y.to_numpy()[: operator.control_num])
        )

    state_epsilon = 2.0e-6
    state_difference = (
        state_objective(
            initial_y + state_epsilon * direction_y,
            initial_v + state_epsilon * direction_v,
        )
        - state_objective(
            initial_y - state_epsilon * direction_y,
            initial_v - state_epsilon * direction_v,
        )
    ) / (2.0 * state_epsilon)
    state_adjoint_direction = float(
        np.sum(state_vjp["initial_position"].reshape((-1, 3)) * direction_y)
        + np.sum(state_vjp["initial_velocity"].reshape((-1, 3)) * direction_v)
    )
    assert state_adjoint_direction == pytest.approx(state_difference, rel=5.0e-4, abs=5.0e-7)

    damping_epsilon = 2.0e-3
    trajectory_damping_difference = (
        state_objective(initial_y, initial_v, damping + damping_epsilon)
        - state_objective(initial_y, initial_v, damping - damping_epsilon)
    ) / (2.0 * damping_epsilon)
    assert state_vjp["joint_damping"][0] == pytest.approx(trajectory_damping_difference, rel=5.0e-3, abs=2.0e-7)
    operator.joint_damping[0] = damping


def test_differentiable_abd_two_step_lagged_friction(tmp_path):
    _init_once()
    dem = _build_two_body_case(assemble_type="COO", save_root=tmp_path)
    dem.sims.affine_newton_tolerance = 1.0e-10
    dem.sims.affine_linear_tolerance = 1.0e-11
    engine = dem.enginer
    operator = engine.operator
    velocity = operator.velocity_y.to_numpy()[: operator.control_num]
    velocity[:4] = [0.0, 0.2, 0.0]
    velocity[4:] = [0.0, -0.1, 0.0]
    operator.velocity_y.from_numpy(np.ascontiguousarray(velocity))

    trajectory = dem.differentiable_affine(steps=2)
    initial_y = operator.y.to_numpy()[: operator.control_num].copy()
    initial_v = velocity.copy()
    initial_time = float(dem.sims.current_time)
    initial_step = int(dem.sims.current_step)
    trajectory.step()
    trajectory.step()
    translation_weights = np.full(operator.control_num, 1.0 / operator.control_num)
    seed_y = translation_weights[:, None] * [0.1, -0.04, 0.02]
    seed_v = translation_weights[:, None] * [2.0e-3, -1.0e-3, 5.0e-4]
    vjp = trajectory.backward(seed_y, seed_v)
    assert int(operator.adjoint_friction_contact_count[None]) > 0

    translation_y = np.tile([0.3, -0.2, 0.1], (operator.control_num, 1))
    translation_v = np.tile([-0.1, 0.05, 0.2], (operator.control_num, 1))

    def objective(y, v):
        operator.y.from_numpy(np.ascontiguousarray(y))
        operator.velocity_y.from_numpy(np.ascontiguousarray(v))
        dem.sims.current_time = initial_time
        dem.sims.current_step = initial_step
        for _ in range(2):
            engine.step(dem.sims, engine.scene)
            dem.sims.current_time += dem.sims.delta
            dem.sims.current_step += 1
        terminal_y = operator.y.to_numpy()[: operator.control_num]
        terminal_v = operator.velocity_y.to_numpy()[: operator.control_num]
        return float(np.sum(seed_y * terminal_y) + np.sum(seed_v * terminal_v))

    epsilon = 2.0e-6
    finite_difference = (
        objective(
            initial_y + epsilon * translation_y,
            initial_v + epsilon * translation_v,
        )
        - objective(
            initial_y - epsilon * translation_y,
            initial_v - epsilon * translation_v,
        )
    ) / (2.0 * epsilon)
    adjoint_direction = float(
        np.sum(vjp["initial_position"].reshape((-1, 3)) * translation_y)
        + np.sum(vjp["initial_velocity"].reshape((-1, 3)) * translation_v)
    )
    assert adjoint_direction == pytest.approx(finite_difference, rel=2.0e-4, abs=2.0e-7), {
        "position": float(np.sum(vjp["initial_position"].reshape((-1, 3)) * translation_y)),
        "velocity": float(np.sum(vjp["initial_velocity"].reshape((-1, 3)) * translation_v)),
        "position_sum": np.sum(vjp["initial_position"].reshape((-1, 3)), axis=0),
        "velocity_sum": np.sum(vjp["initial_velocity"].reshape((-1, 3)), axis=0),
    }

    rng = np.random.default_rng(19)
    seed_y = 0.1 * rng.normal(size=initial_y.shape)
    seed_v = 1.0e-3 * rng.normal(size=initial_v.shape)
    friction_vjp = trajectory.backward(seed_y, seed_v)
    # Cross-step cache refresh remains stop-gradient in lagged mode.
    assert np.isfinite(friction_vjp["friction_scale"])
    assert abs(friction_vjp["friction_scale"]) > 1.0e-12


def test_affine_body_search_modes(tmp_path):
    _init_once()
    for search in ("BVH", "LinkedCell", "HierarchicalLinkedCell"):
        dem = _build_two_body_case(assemble_type="MatrixFree", search=search, save_root=tmp_path)
        assert dem.enginer.last_search_mode == search
        neighbor = dem.enginer.operator.neighbor
        assert neighbor.last_mode == search
        assert neighbor.last_candidate_overflow == 0
        assert neighbor.last_edge_candidate_overflow == 0
        assert neighbor.last_cell_overflow == 0
        assert 0 <= neighbor.last_vertex_face_candidate_pairs <= neighbor.vertex_num * neighbor.face_num
        assert 0 <= neighbor.last_edge_edge_candidate_pairs <= neighbor.edge_num * max(neighbor.edge_num - 1, 1) // 2
        assert (
            dem.enginer.last_candidate_pairs
            == neighbor.last_vertex_face_candidate_pairs + neighbor.last_edge_edge_candidate_pairs
        )
        if search == "BVH":
            assert hasattr(neighbor, "lbvh")
        elif search == "LinkedCell":
            assert hasattr(neighbor, "cell_count")
            assert hasattr(neighbor, "cell_face")
        else:
            assert hasattr(neighbor, "grid")
            assert hasattr(neighbor, "face_level")
        assert np.isfinite(dem.enginer.state.y).all()


def test_affine_body_ccd_modes(tmp_path):
    _init_once()
    for ccd_type in ("ccd", "accd"):
        dem = _build_two_body_case(
            assemble_type="MatrixFree",
            ccd_type=ccd_type,
            save_root=tmp_path,
        )
        assert dem.enginer.last_ccd_type == ccd_type
        assert 0.0 <= dem.enginer.last_ccd_step <= 1.0


@pytest.mark.parametrize("assemble_type", ["MatrixFree", "HashTriplet"])
def test_affine_body_contact_damping_energy_gradient_hessian(tmp_path, assemble_type):
    _init_once()
    dem = _build_two_body_case(
        assemble_type=assemble_type,
        contact_damping_stiffness=0.5,
        save_root=tmp_path,
    )
    operator = dem.enginer.operator
    dof = operator.dof
    count = int(operator.contact_damping_count[None])
    block_i = operator.contact_damping_block_i.to_numpy()[:count]
    block_j = operator.contact_damping_block_j.to_numpy()[:count]
    block_h = operator.contact_damping_block_h.to_numpy()[:count].reshape((-1, 3, 3))
    k_cd = np.zeros((dof, dof), dtype=np.float64)
    for i, j, block in zip(block_i, block_j, block_h):
        k_cd[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] += block
        if i != j:
            k_cd[3 * j : 3 * j + 3, 3 * i : 3 * i + 3] += block.T
    assert count <= operator.contact_damping_capacity
    assert np.linalg.norm(k_cd) > 0.0
    assert np.allclose(k_cd, k_cd.T, rtol=1.0e-7, atol=1.0e-9)
    eigenvalues = np.linalg.eigvalsh(k_cd)
    assert eigenvalues[0] >= -1.0e-10 * max(1.0, np.max(np.abs(eigenvalues)))
    assert not hasattr(operator, "K_contact_damping")
    assert os.path.exists(os.path.join(dem.sims.path, "particles", "AffineBody000001.npz"))
    assert os.path.exists(os.path.join(dem.sims.path, "particles", "AffineSurface000001.npz"))
    assert os.path.exists(os.path.join(dem.sims.path, "vtks", "GraphicAffineBody000001.vtu"))
    assert np.isfinite(dem.enginer.state.y).all()


def test_affine_cube_sliding_on_incline(tmp_path):
    _init_once()
    (
        dem_low_mu,
        displacement_low_mu,
        normal_displacement_low_mu,
        compensated_low_mu,
        expected_low_mu,
        _,
    ) = _build_incline_case(mu=0.2, ccd_type="accd", save_root=tmp_path)
    assert dem_low_mu.enginer.last_ccd_type == "accd"
    assert dem_low_mu.enginer.last_inner_converged
    assert dem_low_mu.enginer.last_friction_converged
    assert abs(compensated_low_mu - expected_low_mu) / expected_low_mu < 0.01, (
        f"sliding BE oracle mismatch: tangential={displacement_low_mu:.9e}, "
        f"normal={normal_displacement_low_mu:.9e}, "
        f"compensated={compensated_low_mu:.9e}, "
        f"expected={expected_low_mu:.9e}"
    )


if __name__ == "__main__":
    manual_root = tempfile.mkdtemp(prefix="geotaichi_affine_manual_")
    test_affine_body_backends(manual_root)
    test_affine_body_wall_contact(manual_root)
    test_affine_body_search_modes(manual_root)
    test_affine_body_ccd_modes(manual_root)
    for backend in ("MatrixFree", "HashTriplet"):
        test_affine_body_contact_damping_energy_gradient_hessian(manual_root, backend)
    test_affine_cube_sliding_on_incline(manual_root)

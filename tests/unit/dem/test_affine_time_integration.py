"""Device ABD time integration against constant-acceleration mechanics."""

from pathlib import Path
from tests.helpers.affine_contact_probe import AffineContactProbe

import numpy as np
import pytest

pytestmark = pytest.mark.isolated_dimension(3)


def build_body(
    tmp_path,
    integration,
    pair=False,
    damping=0.0,
    friction=0.0,
    friction_mode="lagged",
    viscous_friction=0.0,
    joint_damping=None,
):
    from geotaichi import DEM, polyhedron

    dem = DEM(log=False)
    dem.set_configuration(
        domain=[3.0, 3.0, 3.0],
        scheme="AffineBody",
        search="BVH",
        gravity=[0.0, 0.0, 0.0] if pair else [0.5, -1.0, -9.81],
        visualize=False,
        log=False,
    )
    dem.set_affine_body_parameters(
        newmark=integration,
        friction_mode=friction_mode,
        viscous_friction=viscous_friction,
        assemble_type="HashTriplet",
        young_modulus=500.0,
        dhat=0.01,
        barrier_stiffness=1e6,
        max_newton_iteration=8,
        newton_tolerance=1e-10,
        linear_tolerance=1e-14,
    )
    dem.memory_allocate(
        dict(
            max_material_number=1,
            max_affine_body_number=2 if pair else 1,
            surface_node_number=16,
            body_coordination_number=2,
            wall_coordination_number=1,
            max_point_triangle_pairs=128,
            max_edge_edge_pairs=512,
        ),
        log=False,
    )
    dem.set_solver(dict(Timestep=0.001, SimulationTime=0.004, SaveInterval=0.004, SavePath=str(tmp_path)), log=False)
    dem.add_attribute(0, dict(Density=1000.0, ForceLocalDamping=damping, TorqueLocalDamping=damping))
    mesh = Path(__file__).resolve().parents[2] / "data/affine_octahedron.obj"
    dem.add_template(dict(Name="body", TemplateType="AffineBody", Object=polyhedron(file=str(mesh))))
    for body in range(2 if pair else 1):
        dem.create_body(
            dict(
                BodyType="AffineBody",
                Template=dict(
                    Name="body",
                    GroupID=0,
                    MaterialID=0,
                    BodyPoint=[1.0 + 0.502 * body, 1.0, 1.0],
                    BoundingRadius=0.25,
                    InitialVelocity=[0.2, 0.1, 0.0],
                    Friction=0.0,
                ),
            )
        )
    dem.add_property(0, 0, dict(Dhat=0.01, BarrierStiffness=1e6, Friction=friction), dType="all")
    if joint_damping is not None:
        dem.add_joint(
            dict(
                JointType="Revolute",
                BodyID1=0,
                BodyID2=-1,
                WorldAnchor=[1.0, 1.0, 1.0],
                WorldAxis=[0.0, 1.0, 0.0],
                PositionStiffness=2e4,
                AxisStiffness=2e4,
                Damping=joint_damping,
            )
        )
    dem.add_essentials()
    dem.enginer.initialize(dem.sims, dem.scene)
    return dem


@pytest.mark.parametrize("integration", [(1.0, 0.5, 1.0), (0.5, 0.25, 0.5)])
@pytest.mark.parametrize("friction_mode", ["lagged", "fully_implicit"])
def test_accepted_contact_candidates_reproduce_fresh_assembly(
    taichi_runtime, tmp_path, monkeypatch, integration, friction_mode
):
    dem = build_body(tmp_path, integration, pair=True, friction=0.3, friction_mode=friction_mode)
    engine, op = dem.enginer, dem.enginer.operator
    op.initialize_contact_damping_device()
    engine._assemble_system_device(dem.sims, need_matrix=False)
    assert op.last_candidate_pairs > 0
    update, calls = op.neighbor.update, []

    def observed_update(*args, **kwargs):
        calls.append(1)
        return update(*args, **kwargs)

    monkeypatch.setattr(op.neighbor, "update", observed_update)
    direction = np.tile([-0.04, 0.006, 0.0], (op.control_num, 1))
    direction[4:] *= -1
    op.direction_y.from_numpy(direction)
    op.device_backup_line_search_base()
    for alpha in (0.5, 0.25, 0.0):
        op.device_restore_line_search_base()
        assert op.prepare_line_search_neighbors_device(0.5)
        prepared_count = op.last_candidate_pairs
        calls.clear()
        op.device_set_line_search_trial(alpha)
        engine._assemble_system_device(dem.sims, need_matrix=False, reuse_neighbor=True)
        energy = engine._assemble_system_device(dem.sims, need_matrix=True, reuse_neighbor=True)
        gradient = op.grad.to_numpy().copy()
        engine.hash_triplet.finalize_taichi_assembly()
        matrix = engine.hash_triplet.to_scipy(op.control_num).toarray()
        assert calls == []
        reference_energy = engine._assemble_system_device(dem.sims, need_matrix=True)
        engine.hash_triplet.finalize_taichi_assembly()
        reference_matrix = engine.hash_triplet.to_scipy(op.control_num).toarray()
        assert calls == [1] and prepared_count >= op.last_candidate_pairs
        if alpha == 0.5:
            assert prepared_count > op.last_candidate_pairs
        assert energy == pytest.approx(reference_energy, rel=1e-12, abs=1e-14)
        np.testing.assert_allclose(gradient, op.grad.to_numpy(), rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(matrix, reference_matrix, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("integration", [(1.0, 0.5, 1.0), (0.5, 0.25, 0.5)])
def test_abd_device_constant_acceleration_and_recorded_history(taichi_runtime, tmp_path, integration):
    dem = build_body(tmp_path, integration)
    op = dem.enginer.operator
    initial = op.y.to_numpy().copy()
    initial_velocity = op.velocity_y.to_numpy().copy()
    gravity = np.array([0.5, -1.0, -9.81])
    if op.use_newmark:
        np.testing.assert_allclose(op.acceleration_y.to_numpy(), np.tile(gravity, (4, 1)), atol=1e-9)
    for step in range(4):
        dem.enginer.step(dem.sims, dem.scene)
        dem.sims.current_time += dem.sims.delta
        dem.sims.current_step += 1
    dt, steps = 0.001, 4
    position_factor = 0.5 * (steps * dt) ** 2 if op.use_newmark else 0.5 * steps * (steps + 1) * dt**2
    expected = initial + steps * dt * initial_velocity + position_factor * gravity
    np.testing.assert_allclose(op.y.to_numpy(), expected, atol=1e-10, rtol=0)
    np.testing.assert_allclose(op.velocity_y.to_numpy(), initial_velocity + steps * dt * gravity, atol=2e-9)
    assert op.correction_velocity_factor == pytest.approx(integration[2] / (2 * integration[0] * integration[1] * dt))
    snapshot = dem.enginer.output_snapshot()
    if op.use_newmark:
        np.testing.assert_array_equal(snapshot["newmark"], integration)
        np.testing.assert_allclose(snapshot["a_y"], np.tile(gravity, (1, 4, 1)), atol=2e-6)


def test_newmark_failure_restores_position_velocity_and_acceleration(taichi_runtime, tmp_path, monkeypatch):
    dem = build_body(tmp_path, (0.5, 0.25, 0.5))
    op = dem.enginer.operator
    dem.enginer.step(dem.sims, dem.scene)
    before = [field.to_numpy().copy() for field in (op.y, op.velocity_y, op.acceleration_y)]

    def fail(_sims, _energy):
        op.y.fill(123.0)
        op.velocity_y.fill(456.0)
        op.acceleration_y.fill(789.0)
        raise RuntimeError("Affine IPC inner Newton solve did not converge")

    monkeypatch.setattr(dem.enginer, "_solve_lagged_inner_cuda", fail)
    with pytest.raises(RuntimeError, match="did not converge"):
        dem.enginer.step(dem.sims, dem.scene)
    for field, expected in zip((op.y, op.velocity_y, op.acceleration_y), before):
        np.testing.assert_array_equal(field.to_numpy(), expected)


def test_kappa_scales_contact_force_without_violating_action_reaction(taichi_runtime, tmp_path):
    dem = build_body(tmp_path, (1.0, 0.5, 1.0), pair=True)
    op = dem.enginer.operator
    reference = None
    for kappa in (1e6, 1e9, 2e14):
        op.pp_kappa.fill(kappa)
        op.assemble_device(need_matrix=False)
        force = -op.grad.to_numpy().reshape(2, 4, 3).sum(axis=1) / op.scale
        assert np.linalg.norm(force) > 0
        np.testing.assert_allclose(force.sum(axis=0) / np.linalg.norm(force), 0, atol=1e-12)
        if reference is None:
            reference = force / kappa
        else:
            np.testing.assert_allclose(force / kappa, reference, rtol=2e-12, atol=1e-15)


def test_newmark_mass_damping_matches_trapezoidal_solution(taichi_runtime, tmp_path):
    damping, dt = 0.4, 0.001
    dem = build_body(tmp_path, (0.5, 0.25, 0.5), damping=damping)
    op = dem.enginer.operator
    position, velocity = op.y.to_numpy(), op.velocity_y.to_numpy()
    gravity = np.array([0.5, -1.0, -9.81])
    np.testing.assert_allclose(op.acceleration_y.to_numpy(), gravity - damping * velocity, atol=1e-9)
    for _ in range(3):
        next_velocity = ((1 - 0.5 * dt * damping) * velocity + dt * gravity) / (1 + 0.5 * dt * damping)
        position += 0.5 * dt * (velocity + next_velocity)
        velocity = next_velocity
        dem.enginer.step(dem.sims, dem.scene)
    np.testing.assert_allclose(op.y.to_numpy(), position, atol=1e-10, rtol=0)
    np.testing.assert_allclose(op.velocity_y.to_numpy(), velocity, atol=3e-9, rtol=0)


def test_newmark_lagged_friction_is_a_dissipative_incremental_potential(taichi_runtime, tmp_path):
    from src.dem.engines.AffineBodyOperator import MATRIX_COO

    dem = build_body(tmp_path, (0.5, 0.25, 0.5), pair=True, friction=0.4)
    op = dem.enginer.operator
    op.device_begin_step(0.001)
    initial = op.y.to_numpy()
    trial = initial.copy()
    trial[:4, 1] += 1e-5
    op.y.from_numpy(trial)
    op.initialize_contact_damping_device()
    assert op.friction_contact_count[0] > 0

    def evaluate(values):
        op.y.from_numpy(values)
        op._reconstruct_vertices()
        op._clear_system(False, MATRIX_COO)
        op._assemble_lagged_friction(False, MATRIX_COO, False)
        return float(op.energy[None]), op.grad.to_numpy()

    energy, gradient = evaluate(trial)
    direction = np.random.default_rng(19).normal(size=trial.shape)
    step = 1e-8
    fd = (evaluate(trial + step * direction)[0] - evaluate(trial - step * direction)[0]) / (2 * step)
    assert np.sum(gradient * direction) == pytest.approx(fd, rel=2e-5, abs=1e-12)
    assert energy > 0 and np.sum(gradient * (trial - initial)) > 0
    np.testing.assert_allclose(gradient.sum(0), 0, atol=1e-12)


@pytest.mark.parametrize("kind", ["plane", "pt"])
def test_newmark_fully_implicit_friction_endpoint_velocity_and_jacobian(taichi_runtime, tmp_path, kind):
    dem = build_body(
        tmp_path, (0.5, 0.25, 0.5), friction_mode="fully_implicit", viscous_friction=2.0 if kind == "plane" else 0.0
    )
    op = dem.enginer.operator
    probe = AffineContactProbe(op)
    factor = op.correction_velocity_factor
    if kind == "plane":
        positions = np.array([0.0, 0.0, 0.005])
        origin = positions - np.array([0.02, -0.01, 0.0]) / factor

        def evaluate(x):
            return probe.evaluate_fully_implicit_plane_local(x, origin, np.zeros(3), [0.0, 0.0, 1.0], 0.01, 1e6, 0.0)

    else:
        positions = np.array([[0.1, 0.1, 0.005], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).ravel()
        origin = positions.copy()
        origin[:3] -= np.array([0.02, -0.01, 0.0]) / factor

        def evaluate(x):
            return probe.evaluate_fully_implicit_contact_local("pt", x, origin, 0.01, 1e6, 0.4)

    residual, jacobian = evaluate(positions)
    step = 1e-8
    identity = np.eye(len(positions))
    fd = np.column_stack(
        [(evaluate(positions + step * d)[0] - evaluate(positions - step * d)[0]) / (2 * step) for d in identity]
    )
    np.testing.assert_allclose(jacobian, fd, rtol=2e-5, atol=1e-8)
    assert residual[0] > 0 and residual[1] < 0
    if kind == "pt":
        np.testing.assert_allclose(residual.reshape(4, 3).sum(0), 0.0, atol=1e-12)
    else:
        np.testing.assert_allclose(residual / op.scale, 2.0 * np.array([0.02, -0.01, 0.0]), atol=1e-12)
        np.testing.assert_allclose(jacobian / op.scale, 2.0 * factor * np.diag([1.0, 1.0, 0.0]), atol=1e-9)


def test_newmark_joint_damping_gradient_and_nonnegative_power(taichi_runtime, tmp_path):
    from src.dem.engines.AffineBodyOperator import MATRIX_COO

    dem = build_body(tmp_path, (0.5, 0.25, 0.5), joint_damping=0.7)
    op = dem.enginer.operator
    op.device_begin_step(0.001)
    trial = op.y.to_numpy() + 1e-4 * np.random.default_rng(11).normal(size=(4, 3))

    def evaluate(values):
        op.y.from_numpy(values)
        op._clear_system(False, MATRIX_COO)
        op._assemble_joint_damping_compact(False, MATRIX_COO)
        return float(op.energy[None]), op.grad.to_numpy()

    energy, gradient = evaluate(trial)
    direction = np.random.default_rng(12).normal(size=(4, 3))
    delta = 1e-8
    fd = (evaluate(trial + delta * direction)[0] - evaluate(trial - delta * direction)[0]) / (2 * delta)
    assert np.sum(gradient * direction) == pytest.approx(fd, rel=2e-6, abs=1e-12)
    velocity = op.correction_velocity_factor * (trial - op.velocity_origin_y.to_numpy())
    assert energy > 0 and np.sum(gradient * velocity) > 0

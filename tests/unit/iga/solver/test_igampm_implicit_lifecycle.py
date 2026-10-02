"""Unit checks for the mocked implicit IGA-MPM lifecycle."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from src.igampm.engines import Engine
from src.utils.SolverRuntime import StepSchedule
from src.utils.StepRetry import StepRetryPolicy


class _Field:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64).copy()

    def fill(self, value):
        self.values.fill(value)

    def to_numpy(self):
        return self.values.copy()

    def from_numpy(self, values):
        self.values = np.asarray(values, dtype=np.float64).copy()

    def __getitem__(self, index):
        return self.values[index]


def test_explicit_host_linear_solver_boundary_uploads_solution(
    taichi_runtime,
):
    from scipy.sparse import eye

    engine = object.__new__(Engine)
    engine.monolithic_rhs = ti.field(ti.f64, shape=3)
    engine.monolithic_correction = ti.field(ti.f64, shape=3)
    engine.monolithic_rhs.from_numpy(np.array([1.0, -2.0, 3.0]))
    engine.monolithic_hash_matrix = SimpleNamespace(
        to_scipy=lambda active_nodes: eye(active_nodes, format="csr", dtype=np.float64)
    )

    def selected_solver(matrix, rhs):
        assert matrix.shape == (3, 3)
        np.testing.assert_array_equal(rhs, [1.0, -2.0, 3.0])
        return np.array([-1.0, 2.0, -3.0])

    result = engine._solve_monolithic_linear_system(
        {"active_nodes": 3, "active_dof": 3},
        linear_solve=selected_solver,
    )

    np.testing.assert_array_equal(engine.monolithic_correction.to_numpy(), [-1.0, 2.0, -3.0])
    assert result["backend"] == "explicit_host_linear_solve"
    assert result["solution_inf_norm"] == 3.0


def _mock_lifecycle_engine(total_lagrangian=False):
    events = []
    engine = object.__new__(Engine)
    engine.implicit_initialized = False
    engine.implicit_step_in_progress = False
    engine.implicit_step_index = 0
    engine._implicit_step_iga_displacement = None
    engine._implicit_step_mpm_displacement = None
    engine.activate_fric = False
    engine.curr_friction_contact_num = 0
    engine.compile_seconds = 0.0
    engine.timer = SimpleNamespace(section=lambda _name: nullcontext())
    engine.step_retry = StepRetryPolicy()
    engine.step_schedule = StepSchedule()
    engine.output_interval = 1
    engine.history = []
    engine.last_step_record = None
    engine.track_energy = False
    engine.add_implicit_energy_record = lambda _record, _result: None
    engine.time = 0.0
    engine.last_failure = None
    engine.last_contact_ccd_step = 1.0
    engine.last_contact_ccd_min_distance = np.inf
    engine.last_monolithic_iterations = 0
    engine.last_monolithic_residual = np.inf
    engine.last_monolithic_converged = False

    patch = SimpleNamespace(
        volume=_Field([0.0]),
        control_points=_Field([[1.0, 2.0]]),
        velocitys=_Field([[0.0, 0.0]]),
        accelerations=_Field([[0.0, 0.0]]),
    )
    iga = SimpleNamespace(
        patch=patch,
        grid_disp=_Field([7.0, 8.0]),
        grid_disp_temp=_Field([7.0, 8.0]),
        degree_of_freedom=2,
        total_step=3,
        dt=1.0,
    )

    def precompute():
        events.append("iga_precompute")
        patch.volume.from_numpy([2.0])

    def advance_iga():
        events.append("iga_advance")
        patch.control_points.from_numpy(patch.control_points.to_numpy() + iga.grid_disp.to_numpy().reshape((1, 2)))
        patch.velocitys.from_numpy([[3.0, 4.0]])

    iga.precompute = precompute
    iga.iterative_dynamic_advance = advance_iga
    engine.iga = iga

    grid = SimpleNamespace(
        m=_Field([1.0, 1.0] if total_lagrangian else [0.0, 0.0]),
        v=_Field([[0.0, 0.0], [0.0, 0.0]]),
        a=_Field([[0.0, 0.0], [0.0, 0.0]]),
    )
    particle = SimpleNamespace(
        x=_Field([[5.0, 6.0]]),
        v=_Field([[0.0, 0.0]]),
        a=_Field([[0.0, 0.0]]),
    )
    mpm = SimpleNamespace(
        particleNum=_Field([1.0]),
        F0=_Field(np.zeros((1, 2, 2))),
        particle=particle,
        grid=grid,
        grid_disp=_Field([9.0, 10.0, 11.0, 12.0]),
        grid_disp_temp=_Field([9.0, 10.0, 11.0, 12.0]),
        mass_vec=_Field(np.zeros(4)),
        node2dof=object(),
        active_dof=0,
        degree_of_freedom=4,
        integration=[1.0, 0.5, 1.0],
        coeffPIC=0.0,
        compute_traction=False,
        val_lim=1.0e-12,
        total_step=4,
        dt=1.0,
    )

    def init_f0():
        events.append("mpm_init_F0")
        mpm.F0.from_numpy([np.eye(2)])

    def compute_shape():
        events.append("mpm_compute_shape")

    def mass_p2g():
        events.append("mpm_mass_p2g")
        grid.m.from_numpy([1.0, 1.0])

    def find_active():
        events.append("mpm_find_active")

    def set_active():
        events.append("mpm_set_active")
        return 4

    def compute_mass_list(integration):
        del integration
        events.append("mpm_mass_list")
        mpm.mass_vec.from_numpy(np.ones(4))

    def nodal_vel_acc():
        events.append("mpm_nodal_vel_acc")

    def update_nodal_acc(integration):
        del integration
        events.append("mpm_update_nodal")
        grid.v.from_numpy(np.full((2, 2), 2.0))

    def advent_particles(coeff_pic):
        del coeff_pic
        events.append("mpm_advent")
        particle.x.from_numpy(particle.x.to_numpy() + [[0.25, 0.0]])

    mpm.init_F0 = init_f0
    mpm.compute_shapefn = compute_shape
    mpm.mass_p2g = mass_p2g
    mpm.find_active_node = find_active
    mpm.set_active_dof = set_active
    mpm.compute_mass_list = compute_mass_list
    mpm.compute_nodal_vel_acc = nodal_vel_acc
    mpm.update_nodal_acc = update_nodal_acc
    mpm.advent_particles = advent_particles
    mpm.prefix_sum_executor = SimpleNamespace(run=lambda field: events.append("mpm_prefix"))

    if total_lagrangian:
        mpm.grid_reset = lambda: events.append("mpm_tl_grid_reset")
        mpm.vel_acc_p2g = lambda: events.append("mpm_vel_acc_p2g")
    else:

        def grid_reset():
            events.append("mpm_ul_grid_reset")
            grid.m.fill(0.0)

        mpm.grid_reset = grid_reset
        mpm.mass_vel_acc_p2g = mass_p2g

    engine.mpm = mpm
    engine.total_lagrangian_mpm = bool(total_lagrangian)
    engine.prepare_mpm_step = (
        engine._prepare_total_lagrangian_mpm_step if total_lagrangian else engine._prepare_updated_lagrangian_mpm_step
    )
    engine.compute_mpm_traction = lambda: None
    engine.compute_dynamic_mass_list = lambda: mpm.compute_mass_list(mpm.integration)
    # The production lifecycle requires the device monolithic backend.  These
    # callbacks emulate only its state-inspection/snapshot protocol so this
    # unit test remains focused on orchestration rather than Taichi storage.
    engine.monolithic_hash_matrix = object()
    engine.implicit_state_status = _Field(np.zeros(6, dtype=np.int32))

    def inspect_reference_state():
        volume = patch.volume.to_numpy()
        deformation = mpm.F0.to_numpy()
        mass = grid.m.to_numpy()
        engine.implicit_state_status.from_numpy(
            np.asarray(
                [
                    int(not np.all(np.isfinite(volume))),
                    int(np.any(volume != 0.0)),
                    int(not np.all(np.isfinite(deformation))),
                    int(np.any(deformation != 0.0)),
                    int(not np.all(np.isfinite(mass))),
                    int(np.any(mass > mpm.val_lim)),
                ],
                dtype=np.int32,
            )
        )

    def save_entry_displacements():
        engine._mock_entry_iga = iga.grid_disp.to_numpy()
        engine._mock_entry_mpm = mpm.grid_disp.to_numpy()

    def restore_entry_displacements():
        iga.grid_disp.from_numpy(engine._mock_entry_iga)
        iga.grid_disp_temp.from_numpy(engine._mock_entry_iga)
        mpm.grid_disp.from_numpy(engine._mock_entry_mpm)
        mpm.grid_disp_temp.from_numpy(engine._mock_entry_mpm)

    def snapshot_physical_state():
        engine._mock_physical_state = (
            patch.control_points.to_numpy(),
            patch.velocitys.to_numpy(),
            patch.accelerations.to_numpy(),
            particle.x.to_numpy(),
            particle.v.to_numpy(),
            particle.a.to_numpy(),
            mpm.F0.to_numpy(),
            grid.v.to_numpy(),
            grid.a.to_numpy(),
        )

    def restore_physical_state():
        state = engine._mock_physical_state
        patch.control_points.from_numpy(state[0])
        patch.velocitys.from_numpy(state[1])
        patch.accelerations.from_numpy(state[2])
        particle.x.from_numpy(state[3])
        particle.v.from_numpy(state[4])
        particle.a.from_numpy(state[5])
        mpm.F0.from_numpy(state[6])
        grid.v.from_numpy(state[7])
        grid.a.from_numpy(state[8])

    engine._inspect_implicit_reference_state_device = inspect_reference_state
    engine._save_device_monolithic_entry_displacements = save_entry_displacements
    engine._restore_device_monolithic_entry_displacements = restore_entry_displacements
    engine._snapshot_implicit_physical_state_device = snapshot_physical_state
    engine._restore_implicit_physical_state_device = restore_physical_state
    engine.initialize_barrier = lambda *args: events.append("barrier")
    engine.minimum_contact_distance = lambda: 0.5
    engine.initialize_barrier = lambda *args: events.append("barrier")
    engine.minimum_contact_distance = lambda: 0.5
    engine.events = events
    return engine


def test_ul_lifecycle_rebuilds_dynamic_grid_every_step():
    engine = _mock_lifecycle_engine(total_lagrangian=False)

    begin = engine.begin_implicit_ipc_step()

    assert begin == {"step": 0, "active_mpm_dof": 4, "minimum_distance": 0.5}
    assert engine.events == [
        "iga_precompute",
        "mpm_init_F0",
        "mpm_ul_grid_reset",
        "mpm_compute_shape",
        "mpm_mass_p2g",
        "mpm_find_active",
        "mpm_prefix",
        "mpm_set_active",
        "mpm_nodal_vel_acc",
        "mpm_mass_list",
        "barrier",
    ]
    np.testing.assert_array_equal(engine.iga.grid_disp.to_numpy(), [0.0, 0.0])
    np.testing.assert_array_equal(engine.mpm.grid_disp.to_numpy(), np.zeros(4))
    np.testing.assert_array_equal(engine.mpm.mass_vec.to_numpy(), np.ones(4))

    engine.abort_implicit_ipc_step()
    engine.events.clear()
    engine.begin_implicit_ipc_step()
    assert "iga_precompute" not in engine.events
    assert "mpm_init_F0" not in engine.events
    assert engine.events.count("mpm_compute_shape") == 1


def test_tl_lifecycle_keeps_reference_mapping_and_refreshes_kinematics():
    engine = _mock_lifecycle_engine(total_lagrangian=True)

    engine.begin_implicit_ipc_step()

    assert engine.events == [
        "iga_precompute",
        "mpm_init_F0",
        "mpm_find_active",
        "mpm_prefix",
        "mpm_set_active",
        "mpm_mass_list",
        "mpm_tl_grid_reset",
        "mpm_vel_acc_p2g",
        "mpm_nodal_vel_acc",
        "barrier",
    ]
    assert "mpm_compute_shape" not in engine.events
    assert "mpm_mass_p2g" not in engine.events


def test_implicit_substep_advances_only_after_converged_newton():
    engine = _mock_lifecycle_engine(total_lagrangian=False)

    def converged_solve(**kwargs):
        del kwargs
        engine.events.append("solve")
        engine.iga.grid_disp.from_numpy([0.1, -0.2])
        engine.mpm.grid_disp.from_numpy([0.3, 0.0, 0.0, 0.0])
        return {"converged": True, "residual": 0.0}

    engine.solve_monolithic_newton = converged_solve
    result = engine.implicit_ipc_substep(include_friction=False, return_increments=True)

    assert result["accepted"] is True
    assert result["step"] == 0
    np.testing.assert_allclose(result["iga_increment"], [0.1, -0.2])
    np.testing.assert_allclose(result["mpm_increment"], [0.3, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(engine.iga.patch.control_points.to_numpy(), [[1.1, 1.8]])
    np.testing.assert_allclose(engine.mpm.particle.x.to_numpy(), [[5.25, 6.0]])
    np.testing.assert_array_equal(engine.iga.grid_disp.to_numpy(), [0.0, 0.0])
    np.testing.assert_array_equal(engine.mpm.grid_disp.to_numpy(), np.zeros(4))
    assert engine.events.index("solve") < engine.events.index("iga_advance")
    assert engine.events.index("iga_advance") < engine.events.index("mpm_advent")
    assert engine.implicit_step_in_progress is False


def test_accept_step_does_not_snapshot_increment_fields_by_default():
    class NoHostSnapshotField(_Field):
        def to_numpy(self):
            raise AssertionError("accepted substep downloaded a full increment")

    engine = object.__new__(Engine)
    engine.implicit_step_in_progress = True
    engine.implicit_step_index = 0
    engine.iga = SimpleNamespace(
        grid_disp=NoHostSnapshotField([0.1, -0.2]),
        grid_disp_temp=NoHostSnapshotField([0.1, -0.2]),
        iterative_dynamic_advance=lambda: None,
    )
    engine.mpm = SimpleNamespace(
        grid_disp=NoHostSnapshotField([0.3, 0.0]),
        grid_disp_temp=NoHostSnapshotField([0.3, 0.0]),
        integration=object(),
        coeffPIC=0.0,
        update_nodal_acc=lambda _integration: None,
        advent_particles=lambda _coeff_pic: None,
    )
    engine.initialize_barrier = lambda *_fields: None
    engine._snapshot_implicit_physical_state_device = lambda: None
    engine.minimum_contact_distance = lambda: 0.25

    result = engine.accept_implicit_ipc_step()

    assert result == {"step": 0, "minimum_distance": 0.25}


def test_failed_newton_aborts_without_physical_advance():
    engine = _mock_lifecycle_engine(total_lagrangian=False)
    control_points = engine.iga.patch.control_points.to_numpy()
    particles = engine.mpm.particle.x.to_numpy()

    def failed_solve(**kwargs):
        del kwargs
        engine.iga.grid_disp.from_numpy([0.4, 0.5])
        engine.mpm.grid_disp.from_numpy([0.6, 0.0, 0.0, 0.0])
        return {"converged": False, "residual": 1.0}

    engine.solve_monolithic_newton = failed_solve
    with pytest.raises(RuntimeError, match="did not converge"):
        engine.implicit_ipc_substep(include_friction=False)

    np.testing.assert_array_equal(engine.iga.patch.control_points.to_numpy(), control_points)
    np.testing.assert_array_equal(engine.mpm.particle.x.to_numpy(), particles)
    np.testing.assert_array_equal(engine.iga.grid_disp.to_numpy(), [0.0, 0.0])
    np.testing.assert_array_equal(engine.mpm.grid_disp.to_numpy(), np.zeros(4))
    assert "iga_advance" not in engine.events
    assert "mpm_advent" not in engine.events
    assert engine.implicit_step_in_progress is False


def test_accept_failure_restores_both_physical_backends():
    engine = _mock_lifecycle_engine(total_lagrangian=False)
    control_points = engine.iga.patch.control_points.to_numpy()
    iga_velocity = engine.iga.patch.velocitys.to_numpy()
    particles = engine.mpm.particle.x.to_numpy()
    grid_velocity = engine.mpm.grid.v.to_numpy()

    def converged_solve(**kwargs):
        del kwargs
        engine.iga.grid_disp.from_numpy([0.1, 0.2])
        engine.mpm.grid_disp.from_numpy([0.3, 0.0, 0.0, 0.0])
        return {"converged": True, "residual": 0.0}

    def failed_advent(coeff_pic):
        del coeff_pic
        engine.events.append("mpm_advent_failure")
        engine.mpm.particle.x.from_numpy([[99.0, 99.0]])
        raise RuntimeError("particle update failed")

    engine.solve_monolithic_newton = converged_solve
    engine.mpm.advent_particles = failed_advent
    with pytest.raises(RuntimeError, match="particle update failed"):
        engine.implicit_ipc_substep(include_friction=False)

    np.testing.assert_array_equal(engine.iga.patch.control_points.to_numpy(), control_points)
    np.testing.assert_array_equal(engine.iga.patch.velocitys.to_numpy(), iga_velocity)
    np.testing.assert_array_equal(engine.mpm.particle.x.to_numpy(), particles)
    np.testing.assert_array_equal(engine.mpm.grid.v.to_numpy(), grid_velocity)
    assert engine.implicit_step_index == 0
    assert engine.implicit_step_in_progress is False


def test_run_implicit_ipc_contact_honors_requested_step_count():
    engine = _mock_lifecycle_engine(total_lagrangian=False)

    def converged_solve(**kwargs):
        del kwargs
        engine.iga.grid_disp.from_numpy([0.05, 0.0])
        engine.mpm.grid_disp.from_numpy([0.0, 0.0, 0.0, 0.0])
        return {"converged": True, "residual": 0.0}

    engine.solve_monolithic_newton = converged_solve
    result = engine.run_implicit_ipc_contact(steps=2, include_friction=False)

    assert result["completed_steps"] == 2
    assert result["step"] == 1
    assert "iga_increment" not in result
    assert "mpm_increment" not in result
    assert engine.implicit_step_index == 2
    assert engine.events.count("iga_advance") == 2
    assert engine.events.count("mpm_advent") == 2
    assert engine.events.count("iga_precompute") == 1

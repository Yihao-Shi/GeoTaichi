"""Device replay adjoint for continuum Direct MPM--ABD trajectories."""

import math

import numpy as np
import taichi as ti

from src.mpm.soft_particle.DifferentiableMPM import DifferentiableMPM
from src.utils.errors import CapacityError, RecordingError, SimulationStateError


@ti.data_oriented
class DifferentiableMPMAffine:
    """Differentiate DP/VM/MCC and affine state at fixed contact topology.

    Each executed friction outer iteration is reversed through its frozen
    equilibrium and the analytic cache update. Contact switches are discrete.
    """

    def __init__(self, coupling, steps):
        self.coupling = coupling
        self.system = coupling.enginer
        self.affine = self.system.affine
        self.simulation = coupling.mpm.enginer
        self.mpm = self.system.mpm
        self.ipc = self.system.ipc
        self.mixed = self.system.mixed
        coupling.dem.enginer._validate_differentiable_configuration()
        if self.affine.contact_model != "BarrierIPC" or self.ipc.barrier.model != "BarrierIPC":
            raise ValueError("Direct MPM--ABD trajectories require BarrierIPC")
        if self.affine.wall_num:
            raise ValueError("Direct MPM--ABD trajectories do not differentiate DEM walls")
        steps = self.ipc._positive_integer(steps, "steps")
        self.mpm_diff = DifferentiableMPM(self.simulation, steps)
        self.capacity = steps
        self.record_count = 0
        self.dt = self.mpm_diff.dt
        self.controls = int(self.affine.control_num)
        self.position_tape = ti.Vector.field(3, ti.f64, shape=(steps + 1, self.controls))
        self.velocity_tape = ti.Vector.field(3, ti.f64, shape=(steps + 1, self.controls))
        self.equilibrium_rhs = ti.field(ti.f64, shape=self.system.dof_capacity)
        self.cache_rhs = ti.field(ti.f64, shape=self.system.dof_capacity)
        sims = coupling.dem.sims
        limit = sims.affine_friction_iterations
        self.outer_capacity = self.ipc._positive_integer(
            sims.affine_friction_max_iterations if limit <= 0 else limit, "friction iteration capacity"
        )
        self.outer_count = 0
        self.outer_position = ti.Vector.field(3, ti.f64, shape=(self.outer_capacity + 1, self.controls))
        self.outer_displacement = ti.field(ti.f64, shape=(self.outer_capacity + 1, self.mpm.degree_of_freedom))
        self.mixed_cache_position = ti.Vector.field(3, ti.f64, shape=self.mixed.combined_node_count)
        self.affine_cache_position = ti.Vector.field(3, ti.f64, shape=self.affine.vertex_num)
        self.affine_gravity_vjp = ti.Vector.field(3, ti.f64, shape=())
        self.affine_young_vjp = ti.field(ti.f64, shape=max(self.affine.body_num, 1))
        self.joint_target_vjp = ti.field(ti.f64, shape=max(self.affine.joint_num, 1))
        self.joint_damping_vjp = ti.field(ti.f64, shape=max(self.affine.joint_num, 1))
        self.friction_scale_vjp = ti.field(ti.f64, shape=())
        pair_shape = (self.mixed.mpm_body_count, self.mixed.fem_body_count)
        self.mixed_friction_vjp = ti.field(ti.f64, shape=pair_shape)
        self._record_affine_state(0)
        # shortcut: full O(steps*(particles+controls+contacts)) device tape; add checkpointing when storage limits trajectories.

    @ti.kernel
    def _record_affine_state(self, record: ti.i32):
        for control in range(self.controls):
            self.position_tape[record, control] = self.affine.y[control]
            self.velocity_tape[record, control] = self.affine.velocity_y[control]

    @ti.kernel
    def _load_affine_state(self, record: ti.i32):
        self.affine.accepted_translation_velocity[None] = ti.Vector.zero(ti.f64, 3)
        for control in range(self.controls):
            self.affine.y[control] = self.position_tape[record, control]
            self.affine.velocity_y[control] = self.velocity_tape[record, control]

    @ti.kernel
    def _restore_affine_state(self, record: ti.i32):
        previous_record = ti.max(record - 1, 0)
        for control in range(self.controls):
            previous = self.position_tape[previous_record, control]
            velocity = self.velocity_tape[previous_record, control]
            self.affine.y[control] = self.position_tape[record, control]
            self.affine.velocity_y[control] = self.velocity_tape[record, control]
            self.affine.previous_y[control] = previous
            self.affine.previous_velocity_y[control] = velocity
            self.affine.hat_y[control] = previous
            self.affine.tilde_y[control] = previous + self.dt * velocity

    @ti.kernel
    def _pack_equilibrium_rhs(self, active_dof: ti.i32):
        offset = ti.static(3 * self.controls)
        for dof in range(self.system.dof_capacity):
            value = 0.0
            if dof < offset:
                value = self.affine.linear_rhs[dof]
            elif dof - offset < active_dof:
                value = self.ipc.trajectory_grid_vjp[dof - offset]
            self.equilibrium_rhs[dof] = value

    @ti.kernel
    def _accumulate_equilibrium_parameters(self):
        self.mpm_diff.gravity_vjp[None] += self.ipc.gravity_vjp[None]
        self.mpm_diff.friction_vjp[None] += self.ipc.friction_parameter_vjp[None][0]
        self.affine_gravity_vjp[None] += self.affine.gravity_vjp[None]
        self.friction_scale_vjp[None] += self.affine.friction_scale_vjp[None]
        for body in range(self.affine.body_num):
            self.affine_young_vjp[body] += self.affine.young_vjp[body]
        for joint in range(self.affine.joint_num):
            self.joint_target_vjp[joint] += self.affine.joint_target_vjp[joint]
            self.joint_damping_vjp[joint] += self.affine.joint_damping_vjp[joint]

    @ti.kernel
    def _accumulate_material_parameters(self):
        self.mpm_diff.material_parameter_vjp[None] += self.ipc.material_parameter_vjp[None]

    def _record_friction_iterate(self, record):
        if record > self.outer_capacity:
            raise CapacityError("friction iteration limit changed beyond the allocated adjoint workspace")
        self._record_friction_iterate_kernel(record)
        self.outer_count = record

    @ti.kernel
    def _record_friction_iterate_kernel(self, record: ti.i32):
        for control in range(self.controls):
            self.outer_position[record, control] = self.affine.y[control]
        for dof in self.mpm.grid_disp:
            self.outer_displacement[record, dof] = self.mpm.grid_disp[dof]

    @ti.kernel
    def _load_friction_iterate(self, record: ti.i32):
        for control in range(self.controls):
            self.affine.y[control] = self.outer_position[record, control]
        for dof in self.mpm.grid_disp:
            self.mpm.grid_disp[dof] = self.outer_displacement[record, dof]

    @ti.kernel
    def _record_cache_geometry(self):
        for node in self.mixed_cache_position:
            self.mixed_cache_position[node] = self.mixed.positions[node]
        for vertex in self.affine_cache_position:
            self.affine_cache_position[vertex] = self.affine.x[vertex]

    @ti.kernel
    def _advance_outer_rhs(self):
        for dof in self.equilibrium_rhs:
            self.equilibrium_rhs[dof] = self.cache_rhs[dof]

    @ti.kernel
    def _accumulate_initial_cache_vjp(self):
        for control in range(self.controls):
            for component in ti.static(range(3)):
                self.affine.state_y_vjp[control][component] += self.cache_rhs[3 * control + component]

    def _sync_clock(self):
        for sims in (self.coupling.sims, self.coupling.mpm.sims, self.coupling.dem.sims):
            sims.current_time = float(self.mpm.time)
            sims.current_step = int(self.mpm.step_count)
            sims.CurrentTime[None] = float(self.mpm.time)

    def _advance(self, verbose, *, pullback=False, record_history=False):
        def substep(verbose):
            self.simulation.prepare_step_device()
            self.system.friction_recorder = self if pullback else None
            try:
                result = self.system.solve_lagged_equilibrium_device(self.coupling.dem.sims, record_adjoint=True)
            finally:
                self.system.friction_recorder = None
            if pullback:
                self._pullback_step()
            self.system.commit_step_device()
            return result

        result = self.mpm.run_substep(substep, verbose=verbose, record_history=record_history)
        self._sync_clock()
        return result

    def step(self, verbose=True):
        """Advance both children transactionally and record the accepted state."""
        if self.record_count >= self.capacity:
            raise SimulationStateError("Direct MPM--ABD trajectory tape is full")
        self._validate_clock()
        self.mpm_diff._record_terminal_grid()
        initial_time, initial_step = self.mpm.time, self.mpm.step_count
        self.ipc.trajectory_capture_active = True
        try:
            result = self._advance(verbose, record_history=True)
            self.mpm_diff._record_state(self.record_count + 1)
            self._record_affine_state(self.record_count + 1)
            self.ipc.record_trajectory_contact_state(self.record_count)
            self.record_count += 1
        except BaseException:
            self.mpm_diff._load_state(self.record_count)
            self._restore_affine_state(self.record_count)
            self.mpm_diff._restore_terminal_grid()
            self.mpm.time, self.mpm.step_count = initial_time, initial_step
            self._sync_clock()
            self.affine._reconstruct_vertices()
            raise
        finally:
            self.ipc.trajectory_capture_active = False
        return result

    def _validate_clock(self):
        if self.mpm.dt != self.dt or any(
            sims.delta != self.dt for sims in (self.coupling.sims, self.coupling.mpm.sims, self.coupling.dem.sims)
        ):
            raise ValueError("Direct MPM--ABD trajectories require an unchanged time step")
        if self.mpm.step_count != self.mpm_diff.initial_step + self.record_count or not math.isclose(
            self.mpm.time, self.mpm_diff.initial_time + self.record_count * self.dt, rel_tol=1e-12, abs_tol=1e-14
        ):
            raise SimulationStateError("advance the coupling only through its trajectory.step()")

    def _pullback_step(self):
        self.affine.device_prepare_step_state_adjoint(0)
        self.ipc.prepare_plastic_particle_step_adjoint_device()
        self._pack_equilibrium_rhs(int(self.mpm.active_dof))
        for record in reversed(range(1, self.outer_count + 1)):
            # Restore the exact input geometry of this frozen-friction solve.
            self._load_friction_iterate(record - 1)
            self.system.refresh_lagged_friction_device()
            self._record_cache_geometry()
            self.system.backup_lagged_friction_for_adjoint_device()
            self._load_friction_iterate(record)
            self.system.solve_adjoint_device(
                self.equilibrium_rhs, exact_plastic_tangent=True, use_trajectory_contacts=False
            )
            self.ipc.pullback_particle_equilibrium_device()
            self.affine._differentiate_gravity_parameter()
            self.affine._differentiate_young_parameter()
            self.affine._differentiate_joint_target_parameters()
            self.affine._differentiate_joint_damping_parameters()
            self.affine._differentiate_friction_scale_parameter()
            self.affine.device_propagate_step_state_adjoint()
            self.affine.device_propagate_lagged_friction_state_adjoint()
            count = int(self.mixed.culling.point_triangle_count[None])
            self.mixed.pullback_barrier_input_state(self.ipc, count, self.mixed.candidate_field())
            self.cache_rhs.fill(0.0)
            if self.mixed.activate_friction:
                self.mixed.pullback_friction_input_state(
                    self.ipc, int(self.mixed.friction_count), self.mixed_friction_vjp
                )
                self.mixed.pullback_friction_cache(
                    self.ipc,
                    int(self.mixed.friction_count),
                    self.mixed_cache_position,
                    self.outer_displacement,
                    record - 1,
                    self.cache_rhs,
                )
            if self.ipc.activate_fric:
                self.ipc.pullback_ground_friction_cache(
                    self.outer_displacement, record - 1, self.cache_rhs, 3 * self.controls
                )
            self.affine.device_propagate_mesh_friction_cache(self.affine_cache_position, self.cache_rhs)
            self._accumulate_equilibrium_parameters()
            if record > 1:
                self.ipc.accumulate_equilibrium_history_vjp()
                self._advance_outer_rhs()
        self._accumulate_initial_cache_vjp()
        self.ipc._differentiate_p2g_input_state()
        self.ipc._combine_plastic_step_vjp()
        self._load_friction_iterate(self.outer_count)
        self.affine._reconstruct_vertices()

    def backward(self, terminal_state_vjp, verbose=False):
        """Reverse recorded steps; restore the accepted terminal simulation."""
        if not self.record_count:
            raise RecordingError("record at least one Direct MPM--ABD step before backward()")
        self._validate_clock()
        if not isinstance(terminal_state_vjp, dict):
            raise TypeError("terminal_state_vjp must be a dictionary")
        self.ipc._load_particle_output_state_vjp(terminal_state_vjp)
        self.ipc._load_plastic_output_state_vjp(terminal_state_vjp)
        for name, field in (
            ("affine_position", self.affine.state_y_vjp),
            ("affine_velocity", self.affine.state_velocity_vjp),
        ):
            seed = np.asarray(terminal_state_vjp.get(name, np.zeros((self.controls, 3))), dtype=np.float64)
            if seed.size != 3 * self.controls or not np.isfinite(seed).all():
                raise ValueError(f"{name} seed must be finite with {self.controls} control vectors")
            field.from_numpy(np.ascontiguousarray(seed.reshape(self.controls, 3)))
        self.mpm_diff._clear_parameter_vjp()
        for field in (
            self.affine_gravity_vjp,
            self.affine_young_vjp,
            self.joint_target_vjp,
            self.joint_damping_vjp,
            self.friction_scale_vjp,
            self.mixed_friction_vjp,
        ):
            field.fill(0.0)
        terminal_time, terminal_step = self.mpm.time, self.mpm.step_count
        self.mpm_diff._record_terminal_grid()
        try:
            for record in reversed(range(self.record_count)):
                self.mpm_diff._load_state(record)
                self._load_affine_state(record)
                self.mpm.time = self.mpm_diff.initial_time + record * self.dt
                self.mpm.step_count = self.mpm_diff.initial_step + record
                self.ipc.begin_trajectory_adjoint(record)
                try:
                    self._advance(verbose, pullback=True)
                finally:
                    self.ipc.end_trajectory_adjoint()
                self._accumulate_material_parameters()
                self.ipc.promote_plastic_particle_input_vjp_device()
        finally:
            self.mpm_diff._load_state(self.record_count)
            self._restore_affine_state(self.record_count)
            self.mpm_diff._restore_terminal_grid()
            self.mpm.time, self.mpm.step_count = terminal_time, terminal_step
            self._sync_clock()
            self.affine._reconstruct_vertices()
        if self.mpm_diff.initial_step == 0:
            self.mpm_diff._accumulate_initial_acceleration_gravity_vjp()
        result = self.mpm_diff.parameter_and_state_vjps()
        result["mpm_gravity"] = result["gravity"]
        result["affine_gravity"] = np.asarray(self.affine_gravity_vjp[None], dtype=np.float64)
        result["gravity"] = result["mpm_gravity"] + result["affine_gravity"]
        result.update(
            affine_initial_position=self.affine.state_y_vjp.to_numpy()[: self.controls].copy(),
            affine_initial_velocity=self.affine.state_velocity_vjp.to_numpy()[: self.controls].copy(),
            affine_young_modulus=self.affine_young_vjp.to_numpy()[: self.affine.body_num].copy(),
            joint_target_angle_radians=self.joint_target_vjp.to_numpy()[: self.affine.joint_num].copy(),
            joint_damping=self.joint_damping_vjp.to_numpy()[: self.affine.joint_num].copy(),
            friction_scale=float(self.friction_scale_vjp[None]),
            mixed_friction_coefficient=self.mixed_friction_vjp.to_numpy(),
        )
        return result


__all__ = ["DifferentiableMPMAffine"]

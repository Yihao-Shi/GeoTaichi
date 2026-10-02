"""Device checkpoint/replay adjoint for Direct IPC ULMPM trajectories."""

import numpy as np
import taichi as ti

import src.mpm.config as config
from src.mpm.soft_particle.IPCULMPM import IPCULMPM


@ti.data_oriented
class DifferentiableMPM:
    """Differentiate a fixed-step DP/VM Direct-ULMPM trajectory."""

    def __init__(self, simulation, steps=None):
        if not isinstance(simulation, IPCULMPM):
            raise ValueError("DifferentiableMPM requires Direct IPCULMPM")
        mpm = simulation.mpm
        if type(mpm.material).__name__ not in {
            "FiniteStrainDruckerPragerModel",
            "FiniteStrainVonMisesModel",
        }:
            raise ValueError("DifferentiableMPM requires DP or von Mises")
        if not config.DYNAMIC or mpm.step_retry.enabled:
            raise ValueError("DifferentiableMPM requires dynamic fixed-step Newmark")
        if mpm.is_axisymmetric or mpm.velocity_proj:
            raise ValueError("DifferentiableMPM currently requires Cartesian transfer " "and velocity_projection=False")
        if mpm.shape_function_name == "gimp":
            raise ValueError("DifferentiableMPM supports linear or quadratic B-spline transfer")
        ipc = simulation.ipc
        if ipc.is_semi:
            raise ValueError("DifferentiableMPM currently requires BarrierIPC")
        ipc._validate_differentiable_friction()
        if ipc.ground.num and not np.allclose(np.asarray(ipc.ground.np_vel, dtype=np.float64), 0.0):
            raise ValueError("DifferentiableMPM requires stationary ground")
        remaining = max(
            int(mpm.total_step) * int(mpm.output_interval) - int(mpm.step_count),
            1,
        )
        self.capacity = remaining if steps is None else int(steps)
        if self.capacity <= 0:
            raise ValueError("steps must be positive")

        self.simulation = simulation
        self.ipc = ipc
        self.mpm = mpm
        self.dt = float(mpm.dt)
        self.initial_time = float(mpm.time)
        self.initial_step = int(mpm.step_count)
        self.record_count = 0
        self.particle_count = int(mpm.particleNum[0])
        history_size = int(mpm.material.history_state_size)
        tape_shape = (self.capacity + 1, self.particle_count)
        self.position_tape = ti.Vector.field(config.DIM, ti.f64, shape=tape_shape)
        self.velocity_tape = ti.Vector.field(config.DIM, ti.f64, shape=tape_shape)
        self.acceleration_tape = ti.Vector.field(config.DIM, ti.f64, shape=tape_shape)
        self.deformation_tape = ti.Matrix.field(3, 3, ti.f64, shape=tape_shape)
        self.history_tape = ti.Vector.field(history_size, ti.f64, shape=tape_shape)
        self.terminal_grid_velocity = ti.Vector.field(config.DIM, ti.f64, shape=mpm.total_background_grid_num)
        self.terminal_grid_acceleration = ti.Vector.field(config.DIM, ti.f64, shape=mpm.total_background_grid_num)
        self.terminal_grid_mass = ti.field(ti.f64, shape=mpm.total_background_grid_num)
        self.terminal_grid_displacement = ti.field(ti.f64, shape=mpm.degree_of_freedom)
        self.gravity_vjp = ti.Vector.field(config.DIM, ti.f64, shape=())
        self.material_parameter_vjp = ti.Vector.field(4, ti.f64, shape=())
        self.friction_vjp = ti.field(ti.f64, shape=())
        # ponytail: this is a full O(steps*particles) tape; add interval
        # checkpoint/recompute only when production trajectory memory demands it.
        ipc.allocate_trajectory_contact_tape(self.capacity)
        self._record_state(0)

    @ti.kernel
    def _record_state(self, record: ti.i32):
        for particle in range(self.particle_count):
            self.position_tape[record, particle] = self.mpm.particle[particle].x
            self.velocity_tape[record, particle] = self.mpm.particle[particle].v
            self.acceleration_tape[record, particle] = self.mpm.particle[particle].a
            self.deformation_tape[record, particle] = self.mpm.F0[particle]
            self.history_tape[record, particle] = self.mpm.material.get_history_state(particle)

    @ti.kernel
    def _load_state(self, record: ti.i32):
        for particle in range(self.particle_count):
            self.mpm.particle[particle].x = self.position_tape[record, particle]
            self.mpm.particle[particle].v = self.velocity_tape[record, particle]
            self.mpm.particle[particle].a = self.acceleration_tape[record, particle]
            self.mpm.F0[particle] = self.deformation_tape[record, particle]
            self.mpm.material.set_history_state(particle, self.history_tape[record, particle])

    @ti.kernel
    def _record_terminal_grid(self):
        for node in self.mpm.grid:
            self.terminal_grid_velocity[node] = self.mpm.grid[node].v
            self.terminal_grid_acceleration[node] = self.mpm.grid[node].a
            self.terminal_grid_mass[node] = self.mpm.grid[node].m
        for dof in self.mpm.grid_disp:
            self.terminal_grid_displacement[dof] = self.mpm.grid_disp[dof]

    @ti.kernel
    def _restore_terminal_grid(self):
        for node in self.mpm.grid:
            self.mpm.grid[node].v = self.terminal_grid_velocity[node]
            self.mpm.grid[node].a = self.terminal_grid_acceleration[node]
            self.mpm.grid[node].m = self.terminal_grid_mass[node]
        for dof in self.mpm.grid_disp:
            self.mpm.grid_disp[dof] = self.terminal_grid_displacement[dof]

    @ti.kernel
    def _clear_parameter_vjp(self):
        self.gravity_vjp[None] = ti.Vector.zero(ti.f64, config.DIM)
        self.material_parameter_vjp[None] = ti.Vector.zero(ti.f64, 4)
        self.friction_vjp[None] = 0.0

    @ti.kernel
    def _accumulate_parameter_vjp(self):
        self.gravity_vjp[None] += self.ipc.gravity_vjp[None]
        self.material_parameter_vjp[None] += self.ipc.material_parameter_vjp[None]
        self.friction_vjp[None] += self.ipc.friction_parameter_vjp[None][0]

    @ti.kernel
    def _accumulate_initial_acceleration_gravity_vjp(self):
        for particle in range(self.particle_count):
            for component in ti.static(range(config.DIM)):
                ti.atomic_add(
                    self.gravity_vjp[None][component],
                    self.ipc.particle_acceleration_vjp[particle][component],
                )

    def step(self, verbose=True):
        """Advance one step and append its accepted particle state to tape."""
        if self.record_count >= self.capacity:
            raise RuntimeError("differentiable MPM trajectory tape is full")
        if float(self.mpm.dt) != self.dt:
            raise ValueError("time step changed during differentiable trajectory")
        if int(self.mpm.step_count) != self.initial_step + self.record_count:
            raise ValueError("advance the solver only through DifferentiableMPM.step()")
        self.ipc.trajectory_capture_active = True
        try:
            result = self.simulation.step(verbose=verbose)
            self.record_count += 1
            self._record_state(self.record_count)
            self.ipc.record_trajectory_contact_state(self.record_count - 1)
        finally:
            self.ipc.trajectory_capture_active = False
        return result

    def backward(self, terminal_state_vjp, verbose=False):
        """Reverse all recorded steps and return initial-state/parameter VJPs."""
        if self.record_count == 0:
            raise RuntimeError("record at least one step before backward()")
        if not isinstance(terminal_state_vjp, dict):
            raise TypeError("terminal_state_vjp must be a dictionary")
        ipc = self.ipc
        ipc._load_particle_output_state_vjp(terminal_state_vjp)
        ipc._load_plastic_output_state_vjp(terminal_state_vjp)
        self._clear_parameter_vjp()
        terminal_time = float(self.mpm.time)
        terminal_step = int(self.mpm.step_count)
        terminal_history_length = len(self.mpm.history)
        self._record_terminal_grid()
        try:
            for record in reversed(range(self.record_count)):
                self._load_state(record)
                self.mpm.time = self.initial_time + record * self.dt
                self.mpm.step_count = self.initial_step + record

                ipc.begin_trajectory_adjoint(record)
                ipc.pending_adjoint_seed = ipc.trajectory_grid_vjp
                ipc.pending_adjoint_mode = "plastic_particle_device"
                ipc.last_elastic_differentiation = None
                try:
                    self.simulation.step(verbose=verbose, record_history=False)
                    if ipc.last_elastic_differentiation is not True:
                        raise RuntimeError("Direct IPC-MPM replay completed without its " "device adjoint")
                finally:
                    ipc.pending_adjoint_seed = None
                    ipc.pending_adjoint_mode = None
                    ipc.end_trajectory_adjoint()
                self._accumulate_parameter_vjp()
                ipc.promote_plastic_particle_input_vjp_device()
        finally:
            self._load_state(self.record_count)
            self._restore_terminal_grid()
            self.mpm.time = terminal_time
            self.mpm.step_count = terminal_step
            del self.mpm.history[terminal_history_length:]

        if self.initial_step == 0:
            # Direct MPM initializes every particle acceleration from gravity.
            self._accumulate_initial_acceleration_gravity_vjp()
        position_vjp = ipc.particle_position_vjp.to_numpy()[: self.particle_count].copy()
        velocity_vjp = ipc.particle_velocity_vjp.to_numpy()[: self.particle_count].copy()
        acceleration_vjp = ipc.particle_acceleration_vjp.to_numpy()[: self.particle_count].copy()
        gravity_vjp = np.asarray(self.gravity_vjp[None], dtype=np.float64).copy()
        material_parameters = np.asarray(self.material_parameter_vjp[None], dtype=np.float64).copy()
        if type(self.mpm.material).__name__ == "FiniteStrainDruckerPragerModel":
            material_parameter_names = {
                "cohesion": float(material_parameters[2]),
                "friction_angle_degrees": float(material_parameters[3]),
            }
        else:
            material_parameter_names = {
                "yield_stress": float(material_parameters[2]),
                "hardening_modulus": float(material_parameters[3]),
            }
        return {
            "initial_position": position_vjp,
            "initial_velocity": velocity_vjp,
            "initial_acceleration": acceleration_vjp,
            "initial_deformation_gradient": ipc.plastic_deformation_vjp.to_numpy()[: self.particle_count].copy(),
            "initial_plastic_deformation_inverse": ipc.plastic_inverse_vjp.to_numpy()[: self.particle_count].copy(),
            "initial_equivalent_plastic_strain": ipc.plastic_equivalent_strain_vjp.to_numpy()[
                : self.particle_count
            ].copy(),
            "initial_volumetric_plastic_strain": ipc.plastic_volumetric_strain_vjp.to_numpy()[
                : self.particle_count
            ].copy(),
            "gravity": gravity_vjp,
            "material_parameters": material_parameters,
            "young_modulus": float(material_parameters[0]),
            "poisson_ratio": float(material_parameters[1]),
            "dp_cohesion_or_vm_yield_stress": float(material_parameters[2]),
            "dp_friction_angle_or_vm_hardening": float(material_parameters[3]),
            **material_parameter_names,
            "friction_coefficient": float(self.friction_vjp[None]),
        }


__all__ = ["DifferentiableMPM"]

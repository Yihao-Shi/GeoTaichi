"""Device checkpoint/replay adjoint for AffineBody trajectories."""

import numpy as np
import taichi as ti

from .AffineBodyEngine import AffineBodyEngine


@ti.data_oriented
class DifferentiableABD:
    """Differentiate fixed-step BarrierIPC AffineBody trajectories."""

    def __init__(self, engine, steps):
        if not isinstance(engine, AffineBodyEngine) or engine.operator is None:
            raise ValueError("DifferentiableABD requires an initialized AffineBodyEngine")
        engine._validate_differentiable_configuration()
        if not engine.device_nonlinear_path or engine.step_retry.enabled:
            raise ValueError("DifferentiableABD requires the device fixed-step path without retries")
        self.capacity = int(steps)
        if self.capacity <= 0:
            raise ValueError("steps must be positive")

        self.engine = engine
        self.operator = engine.operator
        self.sims = engine.sims
        self.dt = float(self.sims.dt[None])
        self.initial_time = float(self.sims.current_time)
        self.initial_step = int(self.sims.current_step)
        self.record_count = 0
        self.control_num = int(self.operator.control_num)
        tape_shape = (self.capacity + 1, self.control_num)
        self.position_tape = ti.Vector.field(3, ti.f64, shape=tape_shape)
        self.velocity_tape = ti.Vector.field(3, ti.f64, shape=tape_shape)
        self.translation_tape = ti.Vector.field(3, ti.f64, shape=self.capacity + 1)
        self.gravity_vjp = ti.Vector.field(3, ti.f64, shape=())
        self.young_vjp = ti.field(ti.f64, shape=max(self.operator.body_num, 1))
        self.joint_target_vjp = ti.field(ti.f64, shape=max(self.operator.joint_num, 1))
        self.joint_damping_vjp = ti.field(ti.f64, shape=max(self.operator.joint_num, 1))
        self.friction_scale_vjp = ti.field(ti.f64, shape=())
        # ponytail: full O(steps*controls) tape; add interval checkpointing
        # only when long robot trajectories make this storage material.
        self._record_state(0)

    @ti.kernel
    def _record_state(self, record: ti.i32):
        for control in range(self.control_num):
            self.position_tape[record, control] = self.operator.y[control]
            self.velocity_tape[record, control] = self.operator.velocity_y[control]
        self.translation_tape[record] = self.operator.accepted_translation_velocity[None]

    @ti.kernel
    def _load_state(self, record: ti.i32):
        for control in range(self.control_num):
            self.operator.y[control] = self.position_tape[record, control]
            self.operator.velocity_y[control] = self.velocity_tape[record, control]

    @ti.kernel
    def _restore_accepted_state(self, record: ti.i32, dt: ti.f64):
        self.operator.accepted_translation_velocity[None] = self.translation_tape[record]
        for control in range(self.control_num):
            previous = self.position_tape[record - 1, control]
            previous_velocity = self.velocity_tape[record - 1, control]
            self.operator.y[control] = self.position_tape[record, control]
            self.operator.velocity_y[control] = self.velocity_tape[record, control]
            self.operator.previous_y[control] = previous
            self.operator.previous_velocity_y[control] = previous_velocity
            self.operator.hat_y[control] = previous
            self.operator.tilde_y[control] = previous + dt * previous_velocity

    @ti.kernel
    def _clear_parameter_vjp(self):
        self.gravity_vjp[None] = ti.Vector.zero(float, 3)
        self.friction_scale_vjp[None] = 0.0
        for body in range(self.operator.body_num):
            self.young_vjp[body] = 0.0
        for joint in range(self.operator.joint_num):
            self.joint_target_vjp[joint] = 0.0
            self.joint_damping_vjp[joint] = 0.0

    @ti.kernel
    def _accumulate_parameter_vjp(self):
        self.gravity_vjp[None] += self.operator.gravity_vjp[None]
        self.friction_scale_vjp[None] += self.operator.friction_scale_vjp[None]
        for body in range(self.operator.body_num):
            self.young_vjp[body] += self.operator.young_vjp[body]
        for joint in range(self.operator.joint_num):
            self.joint_target_vjp[joint] += self.operator.joint_target_vjp[joint]
            self.joint_damping_vjp[joint] += self.operator.joint_damping_vjp[joint]

    def step(self):
        """Advance once and append the accepted device state to the tape."""
        if self.record_count >= self.capacity:
            raise RuntimeError("differentiable ABD trajectory tape is full")
        if float(self.sims.dt[None]) != self.dt or float(self.sims.delta) != self.dt:
            raise ValueError("time step changed during differentiable trajectory")
        if int(self.sims.current_step) != self.initial_step + self.record_count:
            raise ValueError("advance the solver only through DifferentiableABD.step()")
        result = self.engine.step(self.sims, self.engine.scene)
        self.sims.current_time += self.dt
        self.sims.current_step += 1
        self.record_count += 1
        self._record_state(self.record_count)
        return result

    def backward(self, position_gradient, velocity_gradient=None):
        """Reverse all recorded steps and return state/parameter VJPs."""
        if self.record_count == 0:
            raise RuntimeError("record at least one step before backward()")
        shape = (self.control_num, 3)

        def seed(value):
            result = np.zeros(shape, dtype=np.float64) if value is None else np.asarray(value, dtype=np.float64)
            if result.size != 3 * self.control_num or not np.all(np.isfinite(result)):
                raise ValueError("objective gradients must be finite AffineBody control arrays")
            return np.ascontiguousarray(result.reshape(shape))

        self.operator.state_y_vjp.from_numpy(seed(position_gradient))
        self.operator.state_velocity_vjp.from_numpy(seed(velocity_gradient))
        self._clear_parameter_vjp()
        terminal_time = float(self.sims.current_time)
        terminal_step = int(self.sims.current_step)
        history_length = len(self.engine.history)
        completed = False
        try:
            for record in reversed(range(self.record_count)):
                self._load_state(record)
                self.sims.current_time = self.initial_time + record * self.dt
                self.sims.current_step = self.initial_step + record
                self.engine.step(self.sims, self.engine.scene)
                self.engine.pullback_step_device()
                self._accumulate_parameter_vjp()
            completed = True
        finally:
            try:
                if completed:
                    # Restore the final step's frozen lagged cache on device.
                    self._load_state(self.record_count - 1)
                    self.sims.current_time = terminal_time - self.dt
                    self.sims.current_step = terminal_step - 1
                    self.engine.step(self.sims, self.engine.scene)
            finally:
                self._restore_accepted_state(self.record_count, self.dt)
                self.sims.current_time = terminal_time
                self.sims.current_step = terminal_step
                del self.engine.history[history_length:]

        target_radians = self.joint_target_vjp.to_numpy()[: self.operator.joint_num].copy()
        return {
            "initial_position": self.operator.state_y_vjp.to_numpy()[: self.control_num].reshape(
                self.engine.state.y.shape
            ),
            "initial_velocity": self.operator.state_velocity_vjp.to_numpy()[: self.control_num].reshape(
                self.engine.state.v_y.shape
            ),
            "gravity": np.asarray(self.gravity_vjp[None], dtype=np.float64),
            "affine_young_modulus": self.young_vjp.to_numpy()[: self.operator.body_num].copy(),
            "joint_target_angle_radians": target_radians,
            "joint_target_angle_degrees": target_radians * (np.pi / 180.0),
            "joint_damping": self.joint_damping_vjp.to_numpy()[: self.operator.joint_num].copy(),
            "friction_scale": float(self.friction_scale_vjp[None]),
        }


__all__ = ["DifferentiableABD"]

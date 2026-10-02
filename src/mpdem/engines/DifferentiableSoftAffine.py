"""Legacy device tape/replay adjoint for Soft-Affine IPC trajectories."""

import numpy as np
import taichi as ti

from .SoftAffineIPCEngine import SoftAffineIPCEngine


@ti.data_oriented
class DifferentiableSoftAffine:
    """Differentiate fixed-step Soft-Affine trajectories.

    Plastic soft-particle trajectories are intentionally unavailable; DP/VM
    differentiation belongs to ordinary Direct MPM.
    """

    def __init__(self, engine, sims, scene, steps):
        if not isinstance(engine, SoftAffineIPCEngine) or engine.operator is None:
            raise ValueError("DifferentiableSoftAffine requires an initialized SoftAffineIPCEngine")
        operator = engine.operator
        model = getattr(operator.soft_material.matProps, "model", None)
        if getattr(model, "is_finite_strain_plastic", False):
            raise ValueError(
                "Soft-particle/LSMPM trajectories are hyperelastic-only; "
                "plastic differentiation is available only on ordinary Direct MPM."
            )
        if type(model).__name__ not in {
            "FiniteStrainDruckerPragerModel",
            "FiniteStrainVonMisesModel",
        }:
            raise ValueError(
                "DifferentiableSoftAffine multi-step replay has no supported "
                "hyperelastic state tape yet; use differentiate_elastic_step()"
            )
        if operator.is_semi or operator.affine.levelset_contact:
            raise ValueError("DifferentiableSoftAffine currently requires mesh BarrierIPC")
        if operator.fully_implicit:
            raise ValueError(
                "DifferentiableSoftAffine supports lagged friction only; " "fully implicit friction is not implemented"
            )
        if operator.affine.wall_num or operator.affine.contact_damping_stiffness > 0.0:
            raise ValueError("DifferentiableSoftAffine currently excludes walls and " "contact damping")
        if engine.step_retry.enabled:
            raise ValueError("DifferentiableSoftAffine requires fixed steps without retries")
        self.capacity = int(steps)
        if self.capacity <= 0:
            raise ValueError("steps must be positive")

        self.engine = engine
        self.operator = operator
        self.sims = sims
        self.scene = scene
        self.model = model
        self.dt = float(sims.dt[None])
        self.initial_time = float(sims.current_time)
        self.initial_step = int(sims.current_step)
        self.record_count = 0
        self.affine_count = int(operator.affine.control_num)
        self.soft_count = int(operator.soft_point_num)
        records = self.capacity + 1
        self.affine_position_tape = ti.Vector.field(3, ti.f64, shape=(records, self.affine_count))
        self.affine_velocity_tape = ti.Vector.field(3, ti.f64, shape=(records, self.affine_count))
        self.soft_position_tape = ti.Vector.field(3, ti.f64, shape=(records, self.soft_count))
        self.soft_velocity_tape = ti.Vector.field(3, ti.f64, shape=(records, self.soft_count))
        self.soft_deformation_tape = ti.Matrix.field(3, 3, ti.f64, shape=(records, self.soft_count))
        self.soft_history_tape = ti.Vector.field(
            int(model.history_state_size),
            ti.f64,
            shape=(records, self.soft_count),
        )
        self.gravity_vjp = ti.Vector.field(3, ti.f64, shape=())
        self.material_parameter_vjp = ti.Vector.field(4, ti.f64, shape=())
        self.affine_young_vjp = ti.field(ti.f64, shape=max(operator.affine.body_num, 1))
        self.joint_target_vjp = ti.field(ti.f64, shape=max(operator.affine.joint_num, 1))
        self.joint_damping_vjp = ti.field(ti.f64, shape=max(operator.affine.joint_num, 1))
        self.friction_scale_vjp = ti.field(ti.f64, shape=())
        # ponytail: full O(steps*(particles+controls)) tape; add interval
        # checkpointing only when production trajectories make it material.
        self._record_state(0)

    @ti.kernel
    def _record_state(self, record: ti.i32):
        for control in range(self.affine_count):
            self.affine_position_tape[record, control] = self.operator.affine.y[control]
            self.affine_velocity_tape[record, control] = self.operator.affine.velocity_y[control]
        for particle in range(self.soft_count):
            self.soft_position_tape[record, particle] = self.scene.soft_point[particle].x
            self.soft_velocity_tape[record, particle] = self.scene.soft_point[particle].v
            self.soft_deformation_tape[record, particle] = self.scene.soft_point[particle].F
            self.soft_history_tape[record, particle] = self.model.get_history_state(particle)

    @ti.kernel
    def _load_state(self, record: ti.i32):
        for control in range(self.affine_count):
            self.operator.affine.y[control] = self.affine_position_tape[record, control]
            self.operator.affine.velocity_y[control] = self.affine_velocity_tape[record, control]
        for particle in range(self.soft_count):
            self.scene.soft_point[particle].x = self.soft_position_tape[record, particle]
            self.scene.soft_point[particle].v = self.soft_velocity_tape[record, particle]
            self.scene.soft_point[particle].F = self.soft_deformation_tape[record, particle]
            self.model.set_history_state(particle, self.soft_history_tape[record, particle])

    @ti.kernel
    def _clear_parameter_vjp(self):
        self.gravity_vjp[None] = ti.Vector.zero(ti.f64, 3)
        self.material_parameter_vjp[None] = ti.Vector.zero(ti.f64, 4)
        self.friction_scale_vjp[None] = 0.0
        for body in range(self.operator.affine.body_num):
            self.affine_young_vjp[body] = 0.0
        for joint in range(self.operator.affine.joint_num):
            self.joint_target_vjp[joint] = 0.0
            self.joint_damping_vjp[joint] = 0.0

    @ti.kernel
    def _accumulate_parameter_vjp(self):
        self.gravity_vjp[None] += self.operator.gravity_vjp[None]
        self.material_parameter_vjp[None] += self.operator.material_parameter_vjp[None]
        self.friction_scale_vjp[None] += self.operator.affine.friction_scale_vjp[None]
        for body in range(self.operator.affine.body_num):
            self.affine_young_vjp[body] += self.operator.affine.young_vjp[body]
        for joint in range(self.operator.affine.joint_num):
            self.joint_target_vjp[joint] += self.operator.affine.joint_target_vjp[joint]
            self.joint_damping_vjp[joint] += self.operator.affine.joint_damping_vjp[joint]

    @staticmethod
    def _seed(value, shape, name):
        result = np.zeros(shape, dtype=np.float64) if value is None else np.asarray(value, dtype=np.float64)
        if result.shape != shape or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be finite with shape {shape}")
        return np.ascontiguousarray(result)

    def step(self):
        if self.record_count >= self.capacity:
            raise RuntimeError("differentiable Soft-Affine trajectory tape is full")
        if float(self.sims.dt[None]) != self.dt or float(self.sims.delta) != self.dt:
            raise ValueError("time step changed during differentiable trajectory")
        if int(self.sims.current_step) != self.initial_step + self.record_count:
            raise ValueError("advance the solver only through DifferentiableSoftAffine.step()")
        result = self.engine.step(self.sims, self.scene)
        self.sims.current_time += self.dt
        self.sims.current_step += 1
        self.record_count += 1
        self._record_state(self.record_count)
        return result

    def backward(self, terminal_state_vjp):
        if self.record_count == 0:
            raise RuntimeError("record at least one step before backward()")
        if not isinstance(terminal_state_vjp, dict):
            raise TypeError("terminal_state_vjp must be a dictionary")
        affine_shape = (self.affine_count, 3)
        soft_vector_shape = (self.soft_count, 3)
        self.operator.affine.state_y_vjp.from_numpy(
            self._seed(
                terminal_state_vjp.get("affine_position"),
                affine_shape,
                "affine_position",
            )
        )
        self.operator.affine.state_velocity_vjp.from_numpy(
            self._seed(
                terminal_state_vjp.get("affine_velocity"),
                affine_shape,
                "affine_velocity",
            )
        )
        self.operator.soft_output_position_vjp.from_numpy(
            self._seed(
                terminal_state_vjp.get("position"),
                soft_vector_shape,
                "position",
            )
        )
        self.operator.soft_output_velocity_vjp.from_numpy(
            self._seed(
                terminal_state_vjp.get("velocity"),
                soft_vector_shape,
                "velocity",
            )
        )
        self.operator._load_soft_plastic_output_state_vjp(terminal_state_vjp)
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
                self.engine.pending_elastic_adjoint_seed = self.operator.trajectory_rhs
                self.engine.pending_adjoint_mode = "trajectory"
                self.engine.last_elastic_differentiation = None
                try:
                    self.engine.step(self.sims, self.scene)
                    if self.engine.last_elastic_differentiation is not True:
                        raise RuntimeError("Soft-Affine replay completed without its adjoint")
                finally:
                    self.engine.pending_elastic_adjoint_seed = None
                    self.engine.pending_adjoint_mode = None
                self._accumulate_parameter_vjp()
            completed = True
        finally:
            try:
                if completed:
                    self._load_state(self.record_count - 1)
                    self.sims.current_time = terminal_time - self.dt
                    self.sims.current_step = terminal_step - 1
                    self.engine.step(self.sims, self.scene)
            finally:
                self._load_state(self.record_count)
                self.sims.current_time = terminal_time
                self.sims.current_step = terminal_step
                del self.engine.history[history_length:]

        target_radians = self.joint_target_vjp.to_numpy()[: self.operator.affine.joint_num].copy()
        material_parameters = np.asarray(self.material_parameter_vjp[None], dtype=np.float64).copy()
        if type(self.model).__name__ == "FiniteStrainDruckerPragerModel":
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
            "initial_affine_position": self.operator.affine.state_y_vjp.to_numpy()[: self.affine_count].copy(),
            "initial_affine_velocity": self.operator.affine.state_velocity_vjp.to_numpy()[: self.affine_count].copy(),
            "initial_position": self.operator.soft_output_position_vjp.to_numpy()[: self.soft_count].copy(),
            "initial_velocity": self.operator.soft_output_velocity_vjp.to_numpy()[: self.soft_count].copy(),
            "initial_deformation_gradient": self.operator.plastic_output_deformation_vjp.to_numpy()[
                : self.soft_count
            ].copy(),
            "initial_plastic_deformation_inverse": self.operator.plastic_output_inverse_vjp.to_numpy()[
                : self.soft_count
            ].copy(),
            "initial_equivalent_plastic_strain": self.operator.plastic_output_equivalent_vjp.to_numpy()[
                : self.soft_count
            ].copy(),
            "initial_volumetric_plastic_strain": self.operator.plastic_output_volumetric_vjp.to_numpy()[
                : self.soft_count
            ].copy(),
            "gravity": np.asarray(self.gravity_vjp[None], dtype=np.float64),
            "material_parameters": material_parameters,
            "young_modulus": float(material_parameters[0]),
            "poisson_ratio": float(material_parameters[1]),
            "dp_cohesion_or_vm_yield_stress": float(material_parameters[2]),
            "dp_friction_angle_or_vm_hardening": float(material_parameters[3]),
            **material_parameter_names,
            "affine_young_modulus": self.affine_young_vjp.to_numpy()[: self.operator.affine.body_num].copy(),
            "joint_target_angle_radians": target_radians,
            "joint_target_angle_degrees": target_radians * (np.pi / 180.0),
            "joint_damping": self.joint_damping_vjp.to_numpy()[: self.operator.affine.joint_num].copy(),
            "friction_scale": float(self.friction_scale_vjp[None]),
        }


__all__ = ["DifferentiableSoftAffine"]

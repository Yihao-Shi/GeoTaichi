"""Persistent Taichi fields and vector kernels shared by FEM integrators."""

import math

import numpy as np
import taichi as ti


@ti.data_oriented
class FEMState:
    """Device-resident nodal state.

    Python owns the nonlinear/time-integration control flow, while every
    operation over nodal degrees of freedom is performed by a Taichi kernel.
    Host arrays are accepted only at initialization and at user boundary-data
    interfaces.
    """

    def __init__(self, positions, velocity, mass):
        if ti.lang.impl.get_runtime().prog is None:
            raise RuntimeError(
                "FEM requires an initialized Taichi runtime; call " "geotaichi.init(...) or taichi.init(...) first"
            )
        positions = np.asarray(positions)
        velocity = np.asarray(velocity)
        mass = np.asarray(mass)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("FEM positions must have shape (number_of_nodes, 3)")
        if velocity.shape != positions.shape or mass.shape != (positions.shape[0],):
            raise ValueError("FEM velocity/mass shapes do not match positions")

        self.node_count = int(positions.shape[0])
        self.dof_count = 3 * self.node_count
        self.real_type = ti.lang.impl.current_cfg().default_fp
        self.numpy_type = np.float64 if self.real_type == ti.f64 else np.float32

        self.position = ti.Vector.field(3, dtype=self.real_type, shape=self.node_count)
        self.old_position = ti.Vector.field(3, dtype=self.real_type, shape=self.node_count)
        self.reference_position = ti.Vector.field(3, dtype=self.real_type, shape=self.node_count)
        self.velocity = ti.Vector.field(3, dtype=self.real_type, shape=self.node_count)
        self.old_velocity = ti.Vector.field(3, dtype=self.real_type, shape=self.node_count)
        self.acceleration = ti.Vector.field(3, dtype=self.real_type, shape=self.node_count)
        self.old_acceleration = ti.Vector.field(3, dtype=self.real_type, shape=self.node_count)
        self.reaction = ti.Vector.field(3, dtype=self.real_type, shape=self.node_count)
        self.boundary_force = ti.Vector.field(3, dtype=self.real_type, shape=self.node_count)
        self.boundary_force_scale = ti.field(dtype=self.real_type, shape=())
        self.external_force = ti.Vector.field(3, dtype=self.real_type, shape=self.node_count)
        self.residual = ti.Vector.field(3, dtype=self.real_type, shape=self.node_count)
        self.direction = ti.Vector.field(3, dtype=self.real_type, shape=self.node_count)
        self.trial_position = ti.Vector.field(3, dtype=self.real_type, shape=self.node_count)
        self.predicted_position = ti.Vector.field(3, dtype=self.real_type, shape=self.node_count)
        self.mass = ti.field(dtype=self.real_type, shape=self.node_count)
        self.gravity = ti.Vector.field(3, dtype=self.real_type, shape=())
        self.constrained = ti.field(dtype=ti.i32, shape=self.dof_count)
        self.prescribed_displacement = ti.field(dtype=self.real_type, shape=self.dof_count)

        self.squared_norm = ti.field(dtype=self.real_type, shape=())
        self.directional_derivative = ti.field(dtype=self.real_type, shape=())
        self.kinetic_energy = ti.field(dtype=self.real_type, shape=())
        # Positive cumulative work removed by the explicit mass-proportional
        # damping force.  Keeping the ledger on device avoids a nodal download
        # at every explicit step and makes it part of coupled NPZ checkpoints.
        self.damping_dissipation = ti.field(dtype=self.real_type, shape=())
        self.dynamic_potential = ti.field(dtype=self.real_type, shape=())
        self.minimum_mass = ti.field(dtype=self.real_type, shape=())

        initial_positions = np.ascontiguousarray(positions, dtype=self.numpy_type)
        self.position.from_numpy(initial_positions)
        self.old_position.from_numpy(initial_positions)
        self.reference_position.from_numpy(initial_positions)
        self.velocity.from_numpy(np.ascontiguousarray(velocity, dtype=self.numpy_type))
        self.mass.from_numpy(np.ascontiguousarray(mass, dtype=self.numpy_type))
        self.gravity.fill(0.0)
        self.acceleration.fill(0.0)
        self.old_velocity.fill(0.0)
        self.old_acceleration.fill(0.0)
        self.reaction.fill(0.0)
        self.boundary_force.fill(0.0)
        self.boundary_force_scale.fill(1.0)
        self.external_force.fill(0.0)
        self.residual.fill(0.0)
        self.direction.fill(0.0)
        self.trial_position.from_numpy(initial_positions)
        self.predicted_position.from_numpy(initial_positions)
        self.constrained.fill(0)
        self.prescribed_displacement.fill(0.0)
        self.damping_dissipation.fill(0.0)

    @ti.kernel
    def save_step_state(self):
        for node in range(self.node_count):
            self.old_position[node] = self.position[node]
            self.old_velocity[node] = self.velocity[node]
            self.old_acceleration[node] = self.acceleration[node]

    @ti.kernel
    def prepare_explicit_step(self, save_old_state: ti.i32):
        """Reset nodal loading and retain old state only when required."""
        for node in range(self.node_count):
            if save_old_state != 0:
                self.old_position[node] = self.position[node]
                self.old_velocity[node] = self.velocity[node]
                self.old_acceleration[node] = self.acceleration[node]
            for component in ti.static(range(3)):
                self.external_force[node][component] = (
                    self.mass[node] * self.gravity[None][component]
                    + self.boundary_force_scale[None] * self.boundary_force[node][component]
                )

    @ti.kernel
    def restore_step_state(self):
        for node in range(self.node_count):
            self.position[node] = self.old_position[node]
            self.velocity[node] = self.old_velocity[node]
            self.acceleration[node] = self.old_acceleration[node]
            self.trial_position[node] = self.old_position[node]
            self.predicted_position[node] = self.old_position[node]
            self.residual[node] = ti.Vector.zero(self.real_type, 3)
            self.direction[node] = ti.Vector.zero(self.real_type, 3)

    @ti.kernel
    def set_boundary_data(
        self,
        dofs: ti.types.ndarray(dtype=ti.i32, ndim=1),
        values: ti.types.ndarray(ndim=1),
    ):
        for dof in range(self.dof_count):
            self.constrained[dof] = 0
            self.prescribed_displacement[dof] = 0.0
        for index in range(dofs.shape[0]):
            dof = dofs[index]
            self.constrained[dof] = 1
            self.prescribed_displacement[dof] = values[index]

    @ti.kernel
    def apply_boundary(self, dt: ti.f64, update_velocity: ti.i32):
        for dof in range(self.dof_count):
            if self.constrained[dof] != 0:
                node = dof // 3
                component = dof - 3 * node
                value = self.reference_position[node][component] + self.prescribed_displacement[dof]
                self.position[node][component] = value
                if update_velocity != 0 and dt > 0.0:
                    self.velocity[node][component] = (value - self.old_position[node][component]) / dt
                else:
                    self.velocity[node][component] = 0.0
                self.acceleration[node][component] = 0.0

    def set_boundary_force(self, values):
        values = np.ascontiguousarray(values, dtype=self.numpy_type)
        if values.shape != (self.node_count, 3):
            raise ValueError("FEM boundary force must have shape (number_of_nodes, 3)")
        self.boundary_force.from_numpy(values)

    @ti.kernel
    def set_boundary_force_scale(self, value: ti.f64):
        """Set one device scalar multiplying the persistent Neumann field."""

        self.boundary_force_scale[None] = value

    @ti.kernel
    def set_gravity(self, gravity: ti.types.vector(3, ti.f64)):
        self.gravity[None] = gravity

    @ti.kernel
    def build_external_force(self):
        for node in range(self.node_count):
            for component in ti.static(range(3)):
                self.external_force[node][component] = (
                    self.mass[node] * self.gravity[None][component]
                    + self.boundary_force_scale[None] * self.boundary_force[node][component]
                )

    @ti.kernel
    def explicit_update(
        self,
        internal_force: ti.template(),
        damping: ti.f64,
        dt: ti.f64,
        track_energy: ti.template(),
    ):
        for node in range(self.node_count):
            inverse_mass = 1.0 / self.mass[node]
            old_velocity = self.velocity[node]
            if ti.static(track_energy) and damping > 0.0:
                ti.atomic_add(
                    self.damping_dissipation[None],
                    damping * self.mass[node] * old_velocity.norm_sqr() * dt,
                )
            for component in ti.static(range(3)):
                residual = (
                    self.external_force[node][component]
                    - internal_force[node][component]
                    - damping * self.mass[node] * self.velocity[node][component]
                )
                self.residual[node][component] = residual
                self.acceleration[node][component] = residual * inverse_mass
                self.velocity[node][component] += dt * self.acceleration[node][component]
                self.position[node][component] += dt * self.velocity[node][component]

    @ti.kernel
    def assemble_equilibrium(
        self,
        internal_force: ti.template(),
        damping: ti.f64,
        include_inertia: ti.i32,
    ):
        for node in range(self.node_count):
            for component in ti.static(range(3)):
                value = (
                    internal_force[node][component]
                    + damping * self.mass[node] * self.velocity[node][component]
                    - self.external_force[node][component]
                )
                if include_inertia != 0:
                    value += self.mass[node] * self.acceleration[node][component]
                self.residual[node][component] = value
                if self.constrained[3 * node + component] != 0:
                    self.reaction[node][component] = value
                else:
                    self.reaction[node][component] = 0.0

    @ti.kernel
    def build_newmark_prediction(self, dt: ti.f64, beta: ti.f64):
        for node in range(self.node_count):
            self.predicted_position[node] = (
                self.old_position[node]
                + dt * self.old_velocity[node]
                + dt * dt * (0.5 - beta) * self.old_acceleration[node]
            )

    def assemble_implicit_residual(
        self,
        internal_force: ti.template(),
        damping: ti.f64,
        dt: ti.f64,
        beta: ti.f64,
        gamma: ti.f64,
        quasi_static: ti.i32,
    ):
        self.assemble_implicit_residual_at(
            self.position,
            internal_force,
            damping,
            dt,
            beta,
            gamma,
            quasi_static,
        )

    @ti.kernel
    def assemble_implicit_residual_at(
        self,
        positions: ti.template(),
        internal_force: ti.template(),
        damping: ti.f64,
        dt: ti.f64,
        beta: ti.f64,
        gamma: ti.f64,
        quasi_static: ti.i32,
    ):
        acceleration_factor = 1.0 / (beta * dt * dt)
        for node in range(self.node_count):
            acceleration = acceleration_factor * (positions[node] - self.predicted_position[node])
            velocity = self.old_velocity[node] + dt * (
                (1.0 - gamma) * self.old_acceleration[node] + gamma * acceleration
            )
            for component in ti.static(range(3)):
                value = internal_force[node][component] - self.external_force[node][component]
                if quasi_static == 0:
                    value += self.mass[node] * (acceleration[component] + damping * velocity[component])
                self.residual[node][component] = value

    @ti.kernel
    def finalize_newmark(
        self,
        dt: ti.f64,
        beta: ti.f64,
        gamma: ti.f64,
        quasi_static: ti.i32,
    ):
        acceleration_factor = 1.0 / (beta * dt * dt)
        for node in range(self.node_count):
            if quasi_static != 0:
                self.velocity[node] = ti.Vector.zero(self.real_type, 3)
                self.acceleration[node] = ti.Vector.zero(self.real_type, 3)
            else:
                acceleration = acceleration_factor * (self.position[node] - self.predicted_position[node])
                velocity = self.old_velocity[node] + dt * (
                    (1.0 - gamma) * self.old_acceleration[node] + gamma * acceleration
                )
                for component in ti.static(range(3)):
                    if self.constrained[3 * node + component] != 0:
                        velocity[component] = (self.position[node][component] - self.old_position[node][component]) / dt
                        acceleration[component] = 0.0
                self.velocity[node] = velocity
                self.acceleration[node] = acceleration

    @ti.kernel
    def set_trial_position(self, step: ti.f64):
        for node in range(self.node_count):
            self.trial_position[node] = self.position[node] + step * self.direction[node]
            for component in ti.static(range(3)):
                dof = 3 * node + component
                if self.constrained[dof] != 0:
                    self.trial_position[node][component] = (
                        self.reference_position[node][component] + self.prescribed_displacement[dof]
                    )

    @ti.kernel
    def accept_trial_position(self):
        for node in range(self.node_count):
            self.position[node] = self.trial_position[node]

    @ti.kernel
    def copy_direction_from_flat(self, flat_direction: ti.template()):
        for node, component in ti.ndrange(self.node_count, 3):
            dof = 3 * node + component
            if self.constrained[dof] == 0:
                self.direction[node][component] = flat_direction[dof]
            else:
                self.direction[node][component] = 0.0

    @ti.kernel
    def copy_residual_to_flat(self, flat_residual: ti.template()):
        for node, component in ti.ndrange(self.node_count, 3):
            dof = 3 * node + component
            if self.constrained[dof] == 0:
                flat_residual[dof] = self.residual[node][component]
            else:
                flat_residual[dof] = 0.0

    @ti.kernel
    def reduce_residual_and_slope(self):
        self.squared_norm[None] = 0.0
        self.directional_derivative[None] = 0.0
        for node, component in ti.ndrange(self.node_count, 3):
            dof = 3 * node + component
            if self.constrained[dof] == 0:
                residual = self.residual[node][component]
                ti.atomic_add(self.squared_norm[None], residual * residual)
                ti.atomic_add(
                    self.directional_derivative[None],
                    residual * self.direction[node][component],
                )

    @ti.kernel
    def reduce_external_norm(self):
        self.squared_norm[None] = 0.0
        for node, component in ti.ndrange(self.node_count, 3):
            if self.constrained[3 * node + component] == 0:
                value = self.external_force[node][component]
                ti.atomic_add(self.squared_norm[None], value * value)

    @ti.kernel
    def reduce_direction_inf_norm(self):
        self.squared_norm[None] = 0.0
        for node, component in ti.ndrange(self.node_count, 3):
            if self.constrained[3 * node + component] == 0:
                ti.atomic_max(
                    self.squared_norm[None],
                    ti.abs(self.direction[node][component]),
                )

    @ti.kernel
    def reduce_kinetic_energy(self):
        self.kinetic_energy[None] = 0.0
        for node in range(self.node_count):
            ti.atomic_add(
                self.kinetic_energy[None],
                0.5 * self.mass[node] * self.velocity[node].dot(self.velocity[node]),
            )

    @ti.kernel
    def reduce_dynamic_potential(
        self,
        positions: ti.template(),
        damping: ti.f64,
        dt: ti.f64,
        beta: ti.f64,
        gamma: ti.f64,
        quasi_static: ti.i32,
    ):
        self.dynamic_potential[None] = 0.0
        acceleration_factor = 1.0 / (beta * dt * dt)
        velocity_factor = gamma / (beta * dt)
        for node in range(self.node_count):
            value = -self.external_force[node].dot(positions[node])
            if quasi_static == 0:
                displacement = positions[node] - self.predicted_position[node]
                value += 0.5 * acceleration_factor * self.mass[node] * displacement.dot(displacement)
                velocity_constant = (
                    self.old_velocity[node]
                    + dt * (1.0 - gamma) * self.old_acceleration[node]
                    - velocity_factor * self.predicted_position[node]
                )
                value += 0.5 * damping * velocity_factor * self.mass[node] * positions[node].dot(positions[node])
                value += damping * self.mass[node] * velocity_constant.dot(positions[node])
            ti.atomic_add(self.dynamic_potential[None], value)

    def residual_norm(self):
        self.reduce_residual_and_slope()
        return math.sqrt(max(float(self.squared_norm[None]), 0.0))

    def external_norm(self):
        self.reduce_external_norm()
        return math.sqrt(max(float(self.squared_norm[None]), 0.0))

    def direction_inf_norm(self):
        self.reduce_direction_inf_norm()
        return max(float(self.squared_norm[None]), 0.0)


__all__ = ["FEMState"]

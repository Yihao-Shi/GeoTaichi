"""Device-resident trajectory adjoints for elastic implicit FEM/BarrierIPC."""

import numpy as np
import taichi as ti

from src.fem.cloth.ClothAssembler import ClothAssembler
from src.fem.engines.ClassicalAssembler import ClassicalAssembler
from src.fem.engines.ClassicalFEM import ClassicalImplicitFEM
from src.fem.engines.ClothFEM import ClothImplicitFEM


@ti.data_oriented
class DifferentiableFEM:
    """Differentiate a fixed-step elastic FEM trajectory.

    The first production stage supports StVK/NeoHookean solids, equal-modulus
    TRI3 cloth with quadratic or dihedral bending, and frictionless
    BarrierIPC. Rest geometry, mass, prescribed boundary data, and discrete
    contact features are held fixed by the piecewise-smooth adjoint. All core
    work remains on device unless the user explicitly selects the SciPy linear
    solve; that option transfers only the assembled system and solution.
    """

    def __init__(self, solver, residual_tolerance=None, steps=None):
        is_classical = isinstance(solver, ClassicalImplicitFEM) and solver.mesh.is_volume
        is_cloth = isinstance(solver, ClothImplicitFEM) and solver.mesh.is_membrane
        if not (is_classical or is_cloth):
            raise ValueError("DifferentiableFEM requires classical implicit volume FEM or " "implicit TRI3 cloth FEM")
        if solver.quasi_static or solver.step_retry.enabled:
            raise ValueError("DifferentiableFEM requires dynamic fixed-step Newmark")
        if solver.state.numpy_type != np.float64:
            raise ValueError("DifferentiableFEM requires Taichi f64")
        contact = solver.contact_assembler
        if contact is not None:
            if (
                hasattr(contact, "assemblers")
                or not contact.is_ipc
                or contact.contact.self_contact
                or not contact.contact.planes
            ):
                raise ValueError("DifferentiableFEM currently supports fixed-plane BarrierIPC")
            if contact.activate_friction:
                raise ValueError("DifferentiableFEM first stage requires friction_coefficient=0")
        self.residual_tolerance = None if residual_tolerance is None else float(residual_tolerance)
        if self.residual_tolerance is not None and (
            not np.isfinite(self.residual_tolerance) or self.residual_tolerance <= 0.0
        ):
            raise ValueError("residual_tolerance must be finite and positive")

        self.solver = solver
        self.dt = float(solver.dt)
        self.initial_step = int(solver.step_count)
        self.capacity = max(int(solver.total_step) - self.initial_step, 1) if steps is None else int(steps)
        if self.capacity <= 0:
            raise ValueError("steps must be positive")
        self.record_count = 0
        self.node_count = int(solver.mesh.number_of_nodes)
        self.dof_count = 3 * self.node_count
        self.is_cloth = is_cloth
        self.material_parameter_name = "stretch_stiffness" if is_cloth else "young_modulus"
        if is_cloth:
            stretch = float(getattr(solver.material, "stretch_stiffness", 0.0))
            compression = float(getattr(solver.material, "compression_stiffness", stretch))
            if not np.isclose(stretch, compression):
                raise ValueError("DifferentiableFEM cloth currently requires equal stretch/compression stiffness")
            forward_assembler = solver.cloth_assembler
            if forward_assembler.stitch_count or forward_assembler.spring_count or forward_assembler.sdf_count:
                raise ValueError("DifferentiableFEM cloth does not yet support optional stitch/spring/SDF energies")
        real = solver.state.real_type
        self.position_tape = ti.Vector.field(3, dtype=real, shape=(self.capacity, self.node_count))
        self.constraint_tape = ti.field(ti.i32, shape=(self.capacity, self.dof_count))
        self.replay_position = ti.Vector.field(3, dtype=real, shape=self.node_count)
        self.replay_constrained = ti.field(ti.i32, shape=self.dof_count)
        self.mechanical_force = ti.Vector.field(3, dtype=real, shape=self.node_count)
        self.bx = ti.Vector.field(3, dtype=real, shape=self.node_count)
        self.bv = ti.Vector.field(3, dtype=real, shape=self.node_count)
        self.ba = ti.Vector.field(3, dtype=real, shape=self.node_count)
        self.adjoint_rhs = ti.Vector.field(3, dtype=real, shape=self.node_count)
        self.adjoint = ti.Vector.field(3, dtype=real, shape=self.node_count)
        self.gravity_vjp = ti.Vector.field(3, dtype=real, shape=())
        self.young_vjp = ti.field(dtype=real, shape=())
        self.bending_modulus_vjp = ti.field(dtype=real, shape=())
        self.kappa_vjp = ti.field(dtype=real, shape=())
        self.reverse_stiffness = None
        if is_cloth:
            # Newton may use PSD tangents for globalization, but the discrete
            # adjoint must use the exact Jacobian of the equilibrium residual.
            self.assembler = ClothAssembler(
                solver.mesh,
                solver.material,
                bending_model=solver.cloth_assembler.bending_model,
                project_pd=False,
                project_bending_pd=False,
                assemble_type=solver.assemble_type,
                linear_solver=solver.linear_solver,
                linear_solver_tolerance=solver.linear_solver_tolerance,
                linear_solver_relative_tolerance=solver.linear_solver_relative_tolerance,
                linear_solver_max_iters=solver.linear_solver_max_iters,
            )
        else:
            self.assembler = ClassicalAssembler(
                solver.mesh,
                solver.element,
                solver.material,
                project_pd=False,
                assemble_type=solver.assemble_type,
                linear_solver=solver.linear_solver,
                linear_solver_tolerance=solver.linear_solver_tolerance,
                linear_solver_relative_tolerance=solver.linear_solver_relative_tolerance,
                linear_solver_max_iters=solver.linear_solver_max_iters,
                spatial_dimension=solver.dimension,
            )

    @ti.kernel
    def _record(self, index: ti.i32, position: ti.template(), constrained: ti.template()):
        for node, component in ti.ndrange(self.node_count, 3):
            self.position_tape[index, node][component] = position[node][component]
            dof = 3 * node + component
            self.constraint_tape[index, dof] = constrained[dof]

    @ti.kernel
    def _load_record(self, index: ti.i32):
        for node, component in ti.ndrange(self.node_count, 3):
            self.replay_position[node][component] = self.position_tape[index, node][component]
            dof = 3 * node + component
            self.replay_constrained[dof] = self.constraint_tape[index, dof]

    @ti.kernel
    def _copy_mechanical_force(self, force: ti.template()):
        for node in range(self.node_count):
            self.mechanical_force[node] = force[node]

    @ti.kernel
    def _prepare_adjoint_rhs(self, acceleration_factor: ti.f64, gamma_dt: ti.f64):
        for node, component in ti.ndrange(self.node_count, 3):
            dof = 3 * node + component
            if self.replay_constrained[dof] == 0:
                combined = acceleration_factor * (self.ba[node][component] + gamma_dt * self.bv[node][component])
                self.adjoint_rhs[node][component] = -(self.bx[node][component] + combined)
            else:
                self.adjoint_rhs[node][component] = 0.0

    @ti.kernel
    def _clear_parameter_vjp(self):
        self.gravity_vjp[None] = ti.Vector.zero(float, 3)
        self.young_vjp[None] = 0.0
        self.bending_modulus_vjp[None] = 0.0
        self.kappa_vjp[None] = 0.0

    @ti.kernel
    def _accumulate_cloth_parameter_vjp(
        self,
        bending_force: ti.template(),
        stretch_stiffness: ti.f64,
        bending_modulus: ti.f64,
    ):
        for node, component in ti.ndrange(self.node_count, 3):
            dof = 3 * node + component
            if self.replay_constrained[dof] == 0:
                adjoint = self.adjoint[node][component]
                bend = bending_force[node][component]
                ti.atomic_add(
                    self.young_vjp[None],
                    -adjoint * (self.mechanical_force[node][component] - bend) / stretch_stiffness,
                )
                if bending_modulus > 0.0:
                    ti.atomic_add(self.bending_modulus_vjp[None], -adjoint * bend / bending_modulus)

    @ti.kernel
    def _propagate_step(
        self,
        total_force: ti.template(),
        mass: ti.template(),
        young: ti.f64,
        kappa: ti.f64,
        acceleration_factor: ti.f64,
        prediction_factor: ti.f64,
        dynamic: ti.f64,
        dt: ti.f64,
        gamma: ti.f64,
        damping: ti.f64,
    ):
        for node, component in ti.ndrange(self.node_count, 3):
            dof = 3 * node + component
            if self.replay_constrained[dof] == 0:
                adjoint = self.adjoint[node][component]
                weighted = mass[node] * adjoint
                old_bv = self.bv[node][component]
                combined = acceleration_factor * (self.ba[node][component] + dt * gamma * old_bv)
                ti.atomic_add(self.gravity_vjp[None][component], weighted)
                if ti.static(not self.is_cloth):
                    ti.atomic_add(
                        self.young_vjp[None],
                        -adjoint * self.mechanical_force[node][component] / young,
                    )
                if kappa > 0.0:
                    ti.atomic_add(
                        self.kappa_vjp[None],
                        -adjoint * (total_force[node][component] - self.mechanical_force[node][component]) / kappa,
                    )
                self.bx[node][component] = -combined + dynamic * weighted
                self.ba[node][component] = (
                    dt * (1.0 - gamma) * old_bv
                    - prediction_factor * combined
                    + (dynamic * prediction_factor - damping * dt * (1.0 - gamma)) * weighted
                )
                self.bv[node][component] = old_bv - dt * combined + (dynamic * dt - damping) * weighted
            else:
                self.bx[node][component] = 0.0
                self.bv[node][component] = 0.0
                self.ba[node][component] = 0.0

    def step(self):
        """Advance, record on device, and return the device position field."""
        solver = self.solver
        if solver.dt != self.dt:
            raise ValueError("time step changed during differentiable trajectory")
        if solver.step_count != self.initial_step + self.record_count:
            raise ValueError("advance the solver only through DifferentiableFEM.step()")
        if self.record_count >= self.capacity:
            raise RuntimeError("differentiable trajectory exceeded the configured FEM step count")
        converged = solver.substep()
        if not converged or not solver.last_friction_converged:
            raise RuntimeError("cannot differentiate an unconverged FEM solve")
        residual = solver.state.residual_norm()
        if not np.isfinite(residual):
            raise RuntimeError("forward FEM residual is non-finite")
        if self.residual_tolerance is not None and residual > self.residual_tolerance:
            raise RuntimeError(
                "forward residual is too large for the FEM adjoint: "
                f"residual={residual:.6e}, tolerance={self.residual_tolerance:.6e}"
            )
        self._record(self.record_count, solver.state.position, solver.state.constrained)
        self.record_count += 1
        return solver.state.position

    def backward(self, position_gradient, velocity_gradient=None, acceleration_gradient=None):
        """Return VJPs of a terminal-state objective through the recorded steps."""
        if self.record_count == 0:
            raise RuntimeError("record at least one step before backward()")
        shape = (self.node_count, 3)

        def seed(value):
            result = np.zeros(shape) if value is None else np.asarray(value, dtype=np.float64)
            if result.shape != shape or not np.all(np.isfinite(result)):
                raise ValueError("objective gradients must be finite arrays with the nodal position shape")
            return np.ascontiguousarray(result)

        self.bx.from_numpy(seed(position_gradient))
        self.bv.from_numpy(seed(velocity_gradient))
        self.ba.from_numpy(seed(acceleration_gradient))
        self._clear_parameter_vjp()
        solver = self.solver
        acceleration_factor = 1.0 / (solver.beta * self.dt**2)
        prediction_factor = self.dt**2 * (0.5 - solver.beta)
        dynamic = (1.0 + solver.damping * self.dt * solver.gamma) * acceleration_factor
        contact = solver.contact_assembler
        young = float(
            getattr(
                solver.material,
                "stretch_stiffness",
                getattr(solver.material, "young", 1.0),
            )
        )
        if not np.isfinite(young) or young <= 0.0:
            raise ValueError("differentiable FEM material stiffness must be finite and positive")
        kappa = 0.0 if contact is None else float(contact.contact.kappa)
        bending_modulus = 0.0 if not self.is_cloth else float(solver.material.quadratic_bending_modulus)

        for record in reversed(range(self.record_count)):
            self._load_record(record)
            if self.is_cloth:
                force, stiffness = self.assembler.assemble_device(
                    self.replay_position,
                    need_stiffness=True,
                )
            else:
                force, stiffness = self.assembler.assemble_device(
                    self.replay_position,
                    need_stiffness=True,
                    stiffness=self.reverse_stiffness,
                )
            self.reverse_stiffness = stiffness
            self._copy_mechanical_force(force)
            if contact is not None:
                contact.prepare_iteration_device(self.replay_position)
                force, stiffness = contact.assemble_device(
                    self.replay_position,
                    force,
                    stiffness,
                    need_stiffness=True,
                )
            stiffness.set_mass_diagonal(solver.state.mass, dynamic)
            self._prepare_adjoint_rhs(acceleration_factor, self.dt * solver.gamma)
            stiffness.solve_device(
                self.adjoint_rhs,
                self.adjoint,
                self.replay_constrained,
                fallback_to_bicgstab=self.assembler.linear_solver == "PCG",
            )
            self._propagate_step(
                force,
                solver.state.mass,
                young,
                kappa,
                acceleration_factor,
                prediction_factor,
                dynamic,
                self.dt,
                solver.gamma,
                solver.damping,
            )
            if self.is_cloth:
                bending_force = self.assembler.assemble_bending_force_device(self.replay_position)
                self._accumulate_cloth_parameter_vjp(bending_force, young, bending_modulus)

        result = {
            "young_modulus": float(self.young_vjp[None]),
            "friction_coefficient": 0.0,
            "kappa": float(self.kappa_vjp[None]),
            "gravity": np.asarray(self.gravity_vjp[None], dtype=np.float64),
            "initial_position": self.bx.to_numpy(),
            "initial_velocity": self.bv.to_numpy(),
            "initial_acceleration": self.ba.to_numpy(),
        }
        if self.is_cloth:
            result["stretch_stiffness"] = result.pop("young_modulus")
            bending_scale = solver.material.thickness**3 / (24.0 * (1.0 - solver.material.bending_poisson_ratio**2))
            result["bending_modulus"] = float(self.bending_modulus_vjp[None])
            result["bending_stiffness"] = result["bending_modulus"] * bending_scale
        return result


__all__ = ["DifferentiableFEM"]

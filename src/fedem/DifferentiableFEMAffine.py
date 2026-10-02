"""Device trajectory adjoint for elastic FEM--AffineBody BarrierIPC."""

import numpy as np
import taichi as ti

from src.fem.engines.DifferentiableFEM import DifferentiableFEM
from src.linear_solver.MatrixFreePBICGSTAB import MatrixFreePBICGSTAB
from src.physics_model.contact_model.ipc.ContactDistance import (
    edge_edge_distance2,
    edge_edge_distance_grad,
    point_triangle_distance2,
    point_triangle_distance_grad,
)
from src.physics_model.contact_model.ipc.ContactMollifier import (
    edge_edge_mollifier,
    edge_edge_mollifier_threshold,
)
from src.physics_model.contact_model.ipc.IPC import (
    ipc_friction_f1_over_speed,
    ipc_toolkit_barrier_distance2_offset_terms,
)

from .AffineIPCEngine import FEMAffineIPCEngine


@ti.data_oriented
class DifferentiableFEMAffine:
    """Differentiate fixed-step cloth/solid FEM--ABD trajectories."""

    def __init__(self, engine, steps):
        if not isinstance(engine, FEMAffineIPCEngine):
            raise ValueError("DifferentiableFEMAffine requires an initialized FEM--AffineBody IPC engine")
        if engine.contact.is_semi:
            raise ValueError("FEM--AffineBody differentiable simulation currently requires BarrierIPC")
        if engine.step_retry.enabled:
            raise ValueError("FEM--AffineBody differentiable simulation requires fixed steps without retry")
        if engine.fem_contact is not None:
            raise ValueError("FEM--AffineBody differentiable simulation does not yet support FEM self-contact")
        engine.dem_engine._validate_differentiable_configuration()
        friction_active, friction_iterations, automatic_friction, _ = engine._friction_controls()
        if friction_active and (automatic_friction or friction_iterations != 1):
            raise ValueError("differentiable FEM--AffineBody lagged friction requires friction_iterations=1")
        self.capacity = int(steps)
        if self.capacity <= 0:
            raise ValueError("steps must be positive")

        self.engine = engine
        self.fem = engine.fem
        self.affine = engine.affine
        self.fem_diff = DifferentiableFEM(self.fem, steps=self.capacity)
        self.dt = float(engine.dt)
        self.initial_time = float(engine.time)
        self.initial_step = int(engine.step_count)
        self.record_count = 0
        self.fem_nodes = int(engine.fem_nodes)
        self.affine_controls = int(engine.affine_controls)

        self.fem_position_tape = ti.Vector.field(3, ti.f64, shape=(self.capacity + 1, self.fem_nodes))
        self.fem_velocity_tape = ti.Vector.field(3, ti.f64, shape=(self.capacity + 1, self.fem_nodes))
        self.fem_acceleration_tape = ti.Vector.field(3, ti.f64, shape=(self.capacity + 1, self.fem_nodes))
        self.affine_position_tape = ti.Vector.field(3, ti.f64, shape=(self.capacity + 1, self.affine_controls))
        self.affine_velocity_tape = ti.Vector.field(3, ti.f64, shape=(self.capacity + 1, self.affine_controls))
        self.affine_gravity_vjp = ti.Vector.field(3, ti.f64, shape=())
        self.affine_young_vjp = ti.field(ti.f64, shape=max(self.affine.body_num, 1))
        self.joint_target_vjp = ti.field(ti.f64, shape=max(self.affine.joint_num, 1))
        self.joint_damping_vjp = ti.field(ti.f64, shape=max(self.affine.joint_num, 1))
        self.friction_scale_vjp = ti.field(ti.f64, shape=())
        self.mixed_friction_vjp = ti.field(
            ti.f64,
            shape=(engine.contact.affine_body_count, engine.contact.fem_body_count),
        )
        self.coo_adjoint_solver = None
        # ponytail: full state tape is simplest; add interval checkpointing
        # only when coupled trajectory storage becomes a measured bottleneck.
        self._record_state(0)

    @ti.kernel
    def _record_state(self, record: ti.i32):
        for node in range(self.fem_nodes):
            self.fem_position_tape[record, node] = self.fem.state.position[node]
            self.fem_velocity_tape[record, node] = self.fem.state.velocity[node]
            self.fem_acceleration_tape[record, node] = self.fem.state.acceleration[node]
        for control in range(self.affine_controls):
            self.affine_position_tape[record, control] = self.affine.y[control]
            self.affine_velocity_tape[record, control] = self.affine.velocity_y[control]

    @ti.kernel
    def _load_state(self, record: ti.i32):
        for node in range(self.fem_nodes):
            self.fem.state.position[node] = self.fem_position_tape[record, node]
            self.fem.state.velocity[node] = self.fem_velocity_tape[record, node]
            self.fem.state.acceleration[node] = self.fem_acceleration_tape[record, node]
        for control in range(self.affine_controls):
            self.affine.y[control] = self.affine_position_tape[record, control]
            self.affine.velocity_y[control] = self.affine_velocity_tape[record, control]

    @ti.kernel
    def _restore_accepted_state(self, record: ti.i32, dt: ti.f64):
        previous = ti.max(record - 1, 0)
        for node in range(self.fem_nodes):
            self.fem.state.position[node] = self.fem_position_tape[record, node]
            self.fem.state.velocity[node] = self.fem_velocity_tape[record, node]
            self.fem.state.acceleration[node] = self.fem_acceleration_tape[record, node]
            self.fem.state.old_position[node] = self.fem_position_tape[previous, node]
            self.fem.state.old_velocity[node] = self.fem_velocity_tape[previous, node]
            self.fem.state.old_acceleration[node] = self.fem_acceleration_tape[previous, node]
        self.affine.accepted_translation_velocity[None] = ti.Vector.zero(float, 3)
        for control in range(self.affine_controls):
            old_position = self.affine_position_tape[previous, control]
            old_velocity = self.affine_velocity_tape[previous, control]
            self.affine.y[control] = self.affine_position_tape[record, control]
            self.affine.velocity_y[control] = self.affine_velocity_tape[record, control]
            self.affine.previous_y[control] = old_position
            self.affine.previous_velocity_y[control] = old_velocity
            self.affine.hat_y[control] = old_position
            self.affine.tilde_y[control] = old_position + dt * old_velocity

    @ti.kernel
    def _prepare_coupled_rhs(self):
        offset = ti.static(3 * self.affine_controls)
        for dof in range(self.engine.dof_count):
            value = 0.0
            if dof < offset:
                value = self.affine.linear_rhs[dof]
            else:
                local = dof - offset
                node = local // 3
                component = local % 3
                # DifferentiableFEM routes this field through solve_device(),
                # whose pack kernel negates residual-like inputs.  This
                # monolithic solve consumes the algebraic RHS directly.
                value = -self.fem_diff.adjoint_rhs[node][component]
            self.engine.rhs[dof] = value
            self.engine.correction[dof] = 0.0

    @ti.kernel
    def _copy_fem_constraints(self):
        for dof in range(3 * self.fem_nodes):
            self.fem_diff.replay_constrained[dof] = self.fem.state.constrained[dof]

    @ti.kernel
    def _scatter_coupled_adjoint(self, inverse_dt2: ti.f64):
        offset = ti.static(3 * self.affine_controls)
        for dof in range(offset):
            # The monolithic residual contains dt^-2 times the AffineBody
            # energy gradient; its native pullback kernels consume the
            # unscaled energy adjoint.
            self.affine.linear_x[dof] = inverse_dt2 * self.engine.correction[dof]
        for node, component in ti.ndrange(self.fem_nodes, 3):
            self.fem_diff.adjoint[node][component] = self.engine.correction[offset + 3 * node + component]

    @ti.kernel
    def _clear_affine_parameter_vjp(self):
        self.affine_gravity_vjp[None] = ti.Vector.zero(float, 3)
        self.friction_scale_vjp[None] = 0.0
        for body in range(self.affine.body_num):
            self.affine_young_vjp[body] = 0.0
        for joint in range(self.affine.joint_num):
            self.joint_target_vjp[joint] = 0.0
            self.joint_damping_vjp[joint] = 0.0
        for affine_body, fem_body in self.mixed_friction_vjp:
            self.mixed_friction_vjp[affine_body, fem_body] = 0.0

    @ti.kernel
    def _accumulate_affine_parameter_vjp(self):
        self.affine_gravity_vjp[None] += self.affine.gravity_vjp[None]
        self.friction_scale_vjp[None] += self.affine.friction_scale_vjp[None]
        for body in range(self.affine.body_num):
            self.affine_young_vjp[body] += self.affine.young_vjp[body]
        for joint in range(self.affine.joint_num):
            self.joint_target_vjp[joint] += self.affine.joint_target_vjp[joint]
            self.joint_damping_vjp[joint] += self.affine.joint_damping_vjp[joint]

    @ti.func
    def _mixed_lagged_contraction(
        self,
        lagged,
        current,
        adjoint,
        stencil,
        contact_id,
        contact_kind: ti.template(),
        base_distance2,
        base_coefficient,
    ):
        distance2 = 0.0
        distance_gradient = ti.Vector.zero(float, 12)
        affine_body = 0
        fem_body = 0
        if ti.static(contact_kind == 0):
            distance2, distance_gradient, unused_type = point_triangle_distance_grad(
                ti.Vector([lagged[0, 0], lagged[0, 1], lagged[0, 2]]),
                ti.Vector([lagged[1, 0], lagged[1, 1], lagged[1, 2]]),
                ti.Vector([lagged[2, 0], lagged[2, 1], lagged[2, 2]]),
                ti.Vector([lagged[3, 0], lagged[3, 1], lagged[3, 2]]),
            )
            affine_body, fem_body = self.engine.contact._pair_indices(stencil)
        else:
            distance2, distance_gradient, unused_type = edge_edge_distance_grad(
                ti.Vector([lagged[0, 0], lagged[0, 1], lagged[0, 2]]),
                ti.Vector([lagged[1, 0], lagged[1, 1], lagged[1, 2]]),
                ti.Vector([lagged[2, 0], lagged[2, 1], lagged[2, 2]]),
                ti.Vector([lagged[3, 0], lagged[3, 1], lagged[3, 2]]),
            )
            affine_body, fem_body = self.engine.contact._edge_pair_indices(stencil)

        contraction = 0.0
        distance = ti.sqrt(ti.max(distance2, 0.0))
        relative = ti.Vector.zero(float, 3)
        for component in ti.static(range(3)):
            relative[component] = distance_gradient[component]
            if ti.static(contact_kind == 1):
                relative[component] += distance_gradient[3 + component]
        if distance > 1.0e-15 and relative.norm() > 1.0e-15:
            normal = relative.normalized()
            weights = ti.Vector.zero(float, 4)
            for site, component in ti.static(ti.ndrange(4, 3)):
                weights[site] += distance_gradient[3 * site + component] * normal[component] / (2.0 * distance)
            dmin = self.engine.contact.pair_dmin[affine_body, fem_body]
            dhat = self.engine.contact.pair_dhat[affine_body, fem_body]
            shifted = distance2 - dmin * dmin
            active_gap2 = (2.0 * dmin + dhat) * dhat
            mollifier = 1.0
            if ti.static(contact_kind == 1):
                p0 = ti.Vector([lagged[0, 0], lagged[0, 1], lagged[0, 2]])
                p1 = ti.Vector([lagged[1, 0], lagged[1, 1], lagged[1, 2]])
                q0 = ti.Vector([lagged[2, 0], lagged[2, 1], lagged[2, 2]])
                q1 = ti.Vector([lagged[3, 0], lagged[3, 1], lagged[3, 2]])
                threshold = edge_edge_mollifier_threshold(
                    self.engine.contact.culling.reference_position[stencil[0]],
                    self.engine.contact.culling.reference_position[stencil[1]],
                    self.engine.contact.culling.reference_position[stencil[2]],
                    self.engine.contact.culling.reference_position[stencil[3]],
                )
                mollifier = edge_edge_mollifier(p0, p1, q0, q1, threshold)
            if shifted > 0.0 and shifted < active_gap2 and mollifier >= 1.0 - 1.0e-12:
                normalized_kappa = self.engine.contact.pair_kappa[affine_body, fem_body] / (active_gap2 * active_gap2)
                unused_energy, first, unused_second = ipc_toolkit_barrier_distance2_offset_terms(
                    distance2,
                    dhat,
                    dmin,
                    normalized_kappa,
                    0,
                )
                base_shifted = base_distance2 - dmin * dmin
                base_force = 0.0
                if base_shifted > 0.0 and base_shifted < active_gap2:
                    unused_base_energy, base_first, unused_base_second = ipc_toolkit_barrier_distance2_offset_terms(
                        base_distance2,
                        dhat,
                        dmin,
                        normalized_kappa,
                        0,
                    )
                    base_force = -base_first * ti.sqrt(base_shifted)
                coefficient = 0.0
                if base_force > 1.0e-30:
                    coefficient = base_coefficient * (-first * ti.sqrt(shifted)) / base_force
                if coefficient > 0.0:
                    relative_increment = ti.Vector.zero(float, 3)
                    relative_adjoint = ti.Vector.zero(float, 3)
                    for site, component in ti.static(ti.ndrange(4, 3)):
                        relative_increment[component] += weights[site] * (
                            current[site, component] - lagged[site, component]
                        )
                        relative_adjoint[component] += weights[site] * adjoint[site, component]
                    velocity = (relative_increment - normal * normal.dot(relative_increment)) / self.dt
                    speed = velocity.norm()
                    gradient = (
                        coefficient
                        * ipc_friction_f1_over_speed(
                            speed,
                            self.engine.contact.pair_epsv[affine_body, fem_body],
                        )
                        * velocity
                    )
                    contraction = relative_adjoint.dot(gradient)
        return contraction

    @ti.kernel
    def _propagate_mixed_lagged_cache(self, count: ti.i32, contact_kind: ti.template()):
        # ponytail: a device-local 12-coordinate VJP is the smallest exact-map
        # implementation; replace it with closed forms only if profiling shows
        # mixed-contact adjoints are dominated by these evaluations.
        for contact_id, coordinate in ti.ndrange(count, 12):
            stencil = ti.Vector.zero(ti.i32, 4)
            coefficient = 0.0
            if ti.static(contact_kind == 0):
                stencil = self.engine.contact.friction_pt_candidate[contact_id]
                coefficient = self.engine.contact.friction_pt_coefficient[contact_id]
            else:
                stencil = self.engine.contact.friction_ee_candidate[contact_id]
                coefficient = self.engine.contact.friction_ee_coefficient[contact_id]
            if coefficient > 0.0:
                lagged = ti.Matrix.zero(float, 4, 3)
                current = ti.Matrix.zero(float, 4, 3)
                adjoint = ti.Matrix.zero(float, 4, 3)
                scale = 1.0
                for site, component in ti.static(ti.ndrange(4, 3)):
                    node = stencil[site]
                    lagged[site, component] = self.engine.contact.friction_hat[node][component]
                    current[site, component] = self.engine.contact.position[node][component]
                    scale = ti.max(scale, ti.abs(lagged[site, component]))
                    for support in range(self.engine.contact.support_count[node]):
                        block = self.engine.contact.support_block[node][support]
                        adjoint[site, component] += (
                            self.engine.contact.support_weight[node][support]
                            * self.engine.correction[3 * block + component]
                        )
                site = coordinate // 3
                component = coordinate % 3
                step = 1.0e-5 * scale
                base_distance2 = 0.0
                if ti.static(contact_kind == 0):
                    base_distance2 = point_triangle_distance2(
                        ti.Vector([lagged[0, 0], lagged[0, 1], lagged[0, 2]]),
                        ti.Vector([lagged[1, 0], lagged[1, 1], lagged[1, 2]]),
                        ti.Vector([lagged[2, 0], lagged[2, 1], lagged[2, 2]]),
                        ti.Vector([lagged[3, 0], lagged[3, 1], lagged[3, 2]]),
                    )
                else:
                    base_distance2 = edge_edge_distance2(
                        ti.Vector([lagged[0, 0], lagged[0, 1], lagged[0, 2]]),
                        ti.Vector([lagged[1, 0], lagged[1, 1], lagged[1, 2]]),
                        ti.Vector([lagged[2, 0], lagged[2, 1], lagged[2, 2]]),
                        ti.Vector([lagged[3, 0], lagged[3, 1], lagged[3, 2]]),
                    )
                lagged[site, component] += step
                plus = self._mixed_lagged_contraction(
                    lagged,
                    current,
                    adjoint,
                    stencil,
                    contact_id,
                    contact_kind,
                    base_distance2,
                    coefficient,
                )
                lagged[site, component] -= 2.0 * step
                minus = self._mixed_lagged_contraction(
                    lagged,
                    current,
                    adjoint,
                    stencil,
                    contact_id,
                    contact_kind,
                    base_distance2,
                    coefficient,
                )
                value = -(plus - minus) / (2.0 * step)
                node = stencil[site]
                for support in range(self.engine.contact.support_count[node]):
                    block = self.engine.contact.support_block[node][support]
                    contribution = self.engine.contact.support_weight[node][support] * value
                    if block < ti.static(self.affine_controls):
                        ti.atomic_add(self.affine.state_y_vjp[block][component], contribution)
                    else:
                        ti.atomic_add(
                            self.fem_diff.bx[block - ti.static(self.affine_controls)][component],
                            contribution,
                        )

    @ti.kernel
    def _propagate_mixed_lagged_friction(self, count: ti.i32, contact_kind: ti.template()):
        for contact_id in range(count):
            stencil = ti.Vector.zero(ti.i32, 4)
            weights = ti.Vector.zero(float, 4)
            coefficient = 0.0
            affine_body = 0
            fem_body = 0
            if ti.static(contact_kind == 0):
                stencil = self.engine.contact.friction_pt_candidate[contact_id]
                weights = self.engine.contact.friction_pt_weight[contact_id]
                coefficient = self.engine.contact.friction_pt_coefficient[contact_id]
                affine_body, fem_body = self.engine.contact._pair_indices(stencil)
            else:
                stencil = self.engine.contact.friction_ee_candidate[contact_id]
                weights = self.engine.contact.friction_ee_weight[contact_id]
                coefficient = self.engine.contact.friction_ee_coefficient[contact_id]
                affine_body, fem_body = self.engine.contact._edge_pair_indices(stencil)
            if coefficient > 0.0:
                relative_adjoint = ti.Vector.zero(float, 3)
                for site in ti.static(range(4)):
                    node = stencil[site]
                    for support in range(self.engine.contact.support_count[node]):
                        block = self.engine.contact.support_block[node][support]
                        factor = weights[site] * self.engine.contact.support_weight[node][support]
                        for component in ti.static(range(3)):
                            relative_adjoint[component] += factor * self.engine.correction[3 * block + component]
                gradient = ti.Vector.zero(float, 3)
                for component in ti.static(range(3)):
                    if ti.static(contact_kind == 0):
                        gradient[component] = self.engine.contact.friction_pt_gradient[contact_id, component]
                    else:
                        gradient[component] = self.engine.contact.friction_ee_gradient[contact_id, component]
                friction = self.engine.contact.pair_friction[affine_body, fem_body]
                if friction > 0.0:
                    ti.atomic_add(
                        self.mixed_friction_vjp[affine_body, fem_body],
                        -relative_adjoint.dot(gradient) / friction,
                    )

    def _set_clock(self, record):
        time = self.initial_time + record * self.dt
        step = self.initial_step + record
        self.engine.time = time
        self.engine.step_count = step
        self.engine.simulation.current_time = time
        self.engine.simulation.current_step = step
        self.fem.time = time
        self.fem.step_count = step
        self.engine.dem_wrapper.sims.current_time = time
        self.engine.dem_wrapper.sims.current_step = step

    def _solve_adjoint(self, system):
        matrix = system["matrix"]
        self.engine.correction.fill(0.0)
        if self.engine.linear_solver == "Scipy":
            self.engine._solve_linear_system(system)
            return
        if self.engine.assemble_type == "HashTriplet":
            forward_solver = matrix.solver
            matrix.solver = "PCG"
            try:
                result = matrix.solve_flat_system(
                    self.engine.rhs,
                    self.engine.correction,
                    active_nodes=self.engine.node_count,
                    tol=self.engine.linear_solver_tolerance,
                    rel_tol=self.engine.linear_solver_relative_tolerance,
                    maxiter=self.engine.linear_solver_max_iters,
                    return_solution=False,
                    fallback_to_bicgstab=True,
                )
            finally:
                matrix.solver = forward_solver
        else:
            if self.coo_adjoint_solver is None:
                self.coo_adjoint_solver = MatrixFreePBICGSTAB(self.engine.dof_count)
            solver = self.coo_adjoint_solver
            converged = solver.solve(
                matrix.linear_operator,
                self.engine.rhs,
                self.engine.correction,
                self.engine.coo_diagonal,
                self.engine.dof_count,
                tol=self.engine.linear_solver_tolerance,
                rel_tol=self.engine.linear_solver_relative_tolerance,
                maxiter=self.engine.linear_solver_max_iters,
            )
            result = {
                "converged": bool(converged),
                "iterations": int(solver.last_iterations),
                "residual": float(solver.last_residual),
            }
        if not result["converged"]:
            raise RuntimeError(
                "FEM--AffineBody coupled adjoint did not converge: "
                f"residual={result['residual']:.6e}, iterations={result['iterations']}"
            )

    def step(self):
        """Advance one fixed step and record the accepted state on device."""
        if self.record_count >= self.capacity:
            raise RuntimeError("differentiable FEM--AffineBody trajectory tape is full")
        if self.engine.dt != self.dt or self.engine.step_count != self.initial_step + self.record_count:
            raise ValueError("advance only fixed steps through DifferentiableFEMAffine.step()")
        converged = self.engine.step(verbose=False, record_history=False)
        automatic_friction = self.engine._friction_controls()[2]
        if not converged or (automatic_friction and not self.engine.last_friction_converged):
            raise RuntimeError("cannot differentiate an unconverged FEM--AffineBody step")
        self.record_count += 1
        self._record_state(self.record_count)
        return self.fem.state.position, self.affine.y

    def backward(
        self,
        fem_position_gradient,
        affine_position_gradient=None,
        *,
        fem_velocity_gradient=None,
        fem_acceleration_gradient=None,
        affine_velocity_gradient=None,
    ):
        """Reverse all recorded steps and return FEM/ABD state and parameter VJPs."""
        if self.record_count == 0:
            raise RuntimeError("record at least one step before backward()")

        def seed(value, shape, name):
            result = np.zeros(shape, dtype=np.float64) if value is None else np.asarray(value, dtype=np.float64)
            if result.size != int(np.prod(shape)) or not np.all(np.isfinite(result)):
                raise ValueError(f"{name} must contain {int(np.prod(shape))} finite values")
            return np.ascontiguousarray(result.reshape(shape))

        fem_shape = (self.fem_nodes, 3)
        affine_shape = (self.affine_controls, 3)
        self.fem_diff.bx.from_numpy(seed(fem_position_gradient, fem_shape, "fem_position_gradient"))
        self.fem_diff.bv.from_numpy(seed(fem_velocity_gradient, fem_shape, "fem_velocity_gradient"))
        self.fem_diff.ba.from_numpy(seed(fem_acceleration_gradient, fem_shape, "fem_acceleration_gradient"))
        self.affine.state_y_vjp.from_numpy(seed(affine_position_gradient, affine_shape, "affine_position_gradient"))
        self.affine.state_velocity_vjp.from_numpy(
            seed(affine_velocity_gradient, affine_shape, "affine_velocity_gradient")
        )
        self.fem_diff._clear_parameter_vjp()
        self._clear_affine_parameter_vjp()

        acceleration_factor = 1.0 / (self.fem.beta * self.dt**2)
        prediction_factor = self.dt**2 * (0.5 - self.fem.beta)
        dynamic = (1.0 + self.fem.damping * self.dt * self.fem.gamma) * acceleration_factor
        young = float(
            getattr(
                self.fem.material,
                "stretch_stiffness",
                getattr(self.fem.material, "young", 1.0),
            )
        )
        bending_modulus = float(self.fem.material.quadratic_bending_modulus) if self.fem_diff.is_cloth else 0.0
        terminal_time = float(self.engine.time)
        terminal_step = int(self.engine.step_count)
        completed = False
        try:
            for record in reversed(range(self.record_count)):
                self._load_state(record)
                self._set_clock(record)
                self.engine.step(verbose=False, record_history=False)
                self.affine.device_prepare_step_state_adjoint(0)
                self._copy_fem_constraints()
                self.fem_diff._prepare_adjoint_rhs(
                    acceleration_factor,
                    self.dt * self.fem.gamma,
                )
                self.affine.device_restore_lagged_friction_for_adjoint()
                self.engine.contact.restore_lagged_friction_for_adjoint_device()
                self.affine.device_enter_equilibrium_adjoint()
                try:
                    system = self.engine.assemble_system(
                        need_matrix=True,
                        project_spd=False,
                        solver_shift=False,
                        fem_assembler=self.fem_diff.assembler,
                    )
                    self.fem_diff._copy_mechanical_force(system["fem_internal_force"])
                    self._prepare_coupled_rhs()
                    self._solve_adjoint(system)
                    self._scatter_coupled_adjoint(1.0 / (self.dt * self.dt))
                    self.affine._differentiate_gravity_parameter()
                    self.affine._differentiate_young_parameter()
                    self.affine._differentiate_joint_target_parameters()
                    self.affine._differentiate_joint_damping_parameters()
                    self.affine._differentiate_friction_scale_parameter()
                    self.affine.device_propagate_step_state_adjoint()
                    self.affine.device_propagate_lagged_friction_state_adjoint()
                    self._accumulate_affine_parameter_vjp()
                    self.fem_diff._propagate_step(
                        system["fem_internal_force"],
                        self.fem.state.mass,
                        young,
                        0.0,
                        acceleration_factor,
                        prediction_factor,
                        dynamic,
                        self.dt,
                        self.fem.gamma,
                        self.fem.damping,
                    )
                    if self.fem_diff.is_cloth:
                        bending_force = self.fem_diff.assembler.assemble_bending_force_device(self.fem.state.position)
                        self.fem_diff._accumulate_cloth_parameter_vjp(
                            bending_force,
                            young,
                            bending_modulus,
                        )
                    if self.engine.contact.activate_friction:
                        self._propagate_mixed_lagged_friction(
                            int(self.engine.contact.friction_pt_count),
                            0,
                        )
                        self._propagate_mixed_lagged_cache(
                            int(self.engine.contact.friction_pt_count),
                            0,
                        )
                        self._propagate_mixed_lagged_friction(
                            int(self.engine.contact.friction_ee_count),
                            1,
                        )
                        self._propagate_mixed_lagged_cache(
                            int(self.engine.contact.friction_ee_count),
                            1,
                        )
                finally:
                    self.affine.device_leave_equilibrium_adjoint()
            completed = True
        finally:
            if completed:
                # Recreate the final step's device contact/friction caches so
                # callers may continue the accepted trajectory after backward.
                self._load_state(self.record_count - 1)
                self._set_clock(self.record_count - 1)
                self.engine.step(verbose=False, record_history=False)
            self._restore_accepted_state(self.record_count, self.dt)
            self._set_clock(self.record_count)
            self.affine.sync_output_state()
            if not completed:
                self.engine.time = terminal_time
                self.engine.step_count = terminal_step

        target_radians = self.joint_target_vjp.to_numpy()[: self.affine.joint_num].copy()
        result = {
            "fem_initial_position": self.fem_diff.bx.to_numpy(),
            "fem_initial_velocity": self.fem_diff.bv.to_numpy(),
            "fem_initial_acceleration": self.fem_diff.ba.to_numpy(),
            "affine_initial_position": self.affine.state_y_vjp.to_numpy()[: self.affine_controls],
            "affine_initial_velocity": self.affine.state_velocity_vjp.to_numpy()[: self.affine_controls],
            "gravity": np.asarray(self.fem_diff.gravity_vjp[None], dtype=np.float64)
            + np.asarray(self.affine_gravity_vjp[None], dtype=np.float64),
            "fem_gravity": np.asarray(self.fem_diff.gravity_vjp[None], dtype=np.float64),
            "affine_gravity": np.asarray(self.affine_gravity_vjp[None], dtype=np.float64),
            "affine_young_modulus": self.affine_young_vjp.to_numpy()[: self.affine.body_num].copy(),
            "joint_target_angle_radians": target_radians,
            "joint_target_angle_degrees": target_radians * (np.pi / 180.0),
            "joint_damping": self.joint_damping_vjp.to_numpy()[: self.affine.joint_num].copy(),
            "friction_scale": float(self.friction_scale_vjp[None]),
            "mixed_friction_coefficient": self.mixed_friction_vjp.to_numpy(),
        }
        if self.fem_diff.is_cloth:
            scale = self.fem.material.thickness**3 / (24.0 * (1.0 - self.fem.material.bending_poisson_ratio**2))
            result["stretch_stiffness"] = float(self.fem_diff.young_vjp[None])
            result["bending_modulus"] = float(self.fem_diff.bending_modulus_vjp[None])
            result["bending_stiffness"] = result["bending_modulus"] * scale
        else:
            result["young_modulus"] = float(self.fem_diff.young_vjp[None])
        return result


__all__ = ["DifferentiableFEMAffine"]

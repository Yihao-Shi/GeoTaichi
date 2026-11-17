from src.utils.errors import LinearSolveError, SimulationStateError

import math
import numpy as np
import taichi as ti
from src.dem.engines.AffineBodyOperator import TaichiAffineBodyOperator


@ti.data_oriented
class TaichiAffineDiffIPCProjector(object):
    """GPU translation-only DiffIPC projection for dense affine packings.

    Static template data is uploaded once through ``TaichiAffineBodyOperator``.
    All nonlinear, line-search, Hessian-vector, and Krylov iterations remain
    in Taichi fields; the host reads scalar convergence diagnostics only.
    """

    def __init__(
        self,
        operator,
        base_controls,
        translation_bounds,
        *,
        contact_representation="LevelSet",
    ):
        if not isinstance(operator, TaichiAffineBodyOperator):
            raise TypeError("operator must be a TaichiAffineBodyOperator")
        representation = str(contact_representation)
        if representation not in ("LevelSet", "TriangleMesh"):
            raise ValueError("DiffIPC contact_representation must be 'LevelSet' or " "'TriangleMesh'")
        if representation == "LevelSet" and not operator.levelset_contact:
            raise ValueError("LevelSet DiffIPC projector requires level-set contact")
        if representation == "TriangleMesh" and operator.levelset_contact:
            raise ValueError("TriangleMesh DiffIPC projector requires mesh contact")
        self.contact_representation = representation
        self.operator = operator
        self.body_num = operator.body_num
        self.control_num = operator.control_num
        self.base_y = ti.Vector.field(3, float, shape=max(self.control_num, 1))
        self.base_center = ti.Vector.field(3, float, shape=max(self.body_num, 1))
        self.body_radius = ti.field(float, shape=max(self.body_num, 1))
        self.translation = ti.Vector.field(3, float, shape=max(self.body_num, 1))
        self.trial_translation = ti.Vector.field(3, float, shape=max(self.body_num, 1))
        self.lower = ti.Vector.field(3, float, shape=max(self.body_num, 1))
        self.upper = ti.Vector.field(3, float, shape=max(self.body_num, 1))
        self.gradient = ti.Vector.field(3, float, shape=max(self.body_num, 1))
        self.direction = ti.Vector.field(3, float, shape=max(self.body_num, 1))
        self.residual = ti.Vector.field(3, float, shape=max(self.body_num, 1))
        self.cg_p = ti.Vector.field(3, float, shape=max(self.body_num, 1))
        self.cg_Ap = ti.Vector.field(3, float, shape=max(self.body_num, 1))
        self.adjoint_gradient = ti.Vector.field(3, float, shape=max(self.body_num, 1))
        self.free = ti.Vector.field(3, ti.i32, shape=max(self.body_num, 1))
        self.energy = ti.field(float, shape=())
        self.minimum_gap = ti.field(float, shape=())
        self.sampled_contacts = ti.field(ti.i32, shape=())
        self.active_contacts = ti.field(ti.i32, shape=())
        self.reduction = ti.field(float, shape=())
        base_controls_np = np.ascontiguousarray(
            np.asarray(base_controls, dtype=np.float64).reshape(self.control_num, 3)
        )
        self.base_y.from_numpy(base_controls_np)
        self.base_center.from_numpy(np.ascontiguousarray(base_controls_np.reshape(self.body_num, 4, 3).mean(axis=1)))
        self.body_radius.from_numpy(
            np.ascontiguousarray(
                np.asarray(
                    [float(body["scale"]) * float(body["template"].bounding_radius) for body in operator.state.bodies],
                    dtype=np.float64,
                )
            )
        )
        if translation_bounds is None:
            self.lower.fill(-1.0e30)
            self.upper.fill(1.0e30)
        else:
            bounds = np.asarray(translation_bounds, dtype=np.float64).reshape(self.body_num, 3, 2)
            self.lower.from_numpy(np.ascontiguousarray(bounds[:, :, 0]))
            self.upper.from_numpy(np.ascontiguousarray(bounds[:, :, 1]))
        self._initialize()
        self.last_kappa = 0.0
        self.last_delta = 0.0
        self.last_anchor_stiffness = 0.0
        self.last_hessian_shift = 0.0
        self.last_iterations = 0
        self.last_stages = 0
        self.last_success = False

    @ti.kernel
    def _initialize(self):
        for body, component in ti.ndrange(self.body_num, 3):
            # Random Distribute samples the full region, so a particle center
            # can start inside the boundary clearance implied by its bounding
            # radius.  Begin from the nearest feasible translation instead of
            # asking the first Armijo trial to make a discontinuous projection.
            value = ti.min(
                self.upper[body][component],
                ti.max(self.lower[body][component], 0.0),
            )
            self.translation[body][component] = value
            self.trial_translation[body][component] = value

    @ti.func
    def _continued_barrier(self, gap, dhat, kappa, delta):
        value = 0.0
        first = 0.0
        second = 0.0
        if gap >= delta:
            value, first, second = self.operator._ipc_barrier_gap(gap, dhat, kappa)
        else:
            value_delta, first_delta, second_delta = self.operator._ipc_barrier_gap(delta, dhat, kappa)
            increment = gap - delta
            value = value_delta + first_delta * increment + 0.5 * second_delta * increment * increment
            first = first_delta + second_delta * increment
            second = second_delta
        return value, first, second

    @ti.func
    def _gap_data(self, vertex_id, target_body, translations: ti.template()):
        source_body = self.operator.node2body[vertex_id]
        source_point = translations[source_body]
        for control in range(4):
            source_point += self.operator.basis[vertex_id, control] * self.base_y[source_body * 4 + control]
        target_y0 = self.base_y[target_body * 4] + translations[target_body]
        target_A = ti.Matrix.cols(
            [
                self.base_y[target_body * 4 + 1] - self.base_y[target_body * 4],
                self.base_y[target_body * 4 + 2] - self.base_y[target_body * 4],
                self.base_y[target_body * 4 + 3] - self.base_y[target_body * 4],
            ]
        )
        gap = self.operator.dhat
        normal = ti.Vector.zero(float, 3)
        gap_hessian = ti.Matrix.zero(float, 3, 3)
        inside = False
        valid = ti.abs(target_A.determinant()) > 1.0e-12
        if valid:
            inverse_A = target_A.inverse()
            material = inverse_A @ (source_point - target_y0)
            scale = self.operator.body_scale[target_body]
            coordinate = (material - ti.Vector([0.25, 0.25, 0.25])) / scale
            phi, phi_gradient, phi_hessian, inside = self.operator._sample_affine_levelset(target_body, coordinate)
            if inside:
                gap = scale * phi
                normal = inverse_A.transpose() @ phi_gradient
                gap_hessian = inverse_A.transpose() @ (phi_hessian / scale) @ inverse_A
        return source_body, gap, normal, gap_hessian, inside

    @ti.func
    def _surface_vertex(self, vertex_id, translations: ti.template()):
        body = self.operator.node2body[vertex_id]
        point = translations[body]
        for control in range(4):
            point += self.operator.basis[vertex_id, control] * self.base_y[body * 4 + control]
        return point

    @ti.func
    def _pair_may_contact(self, source_body, target_body, translations: ti.template()):
        center_delta = (
            self.base_center[source_body]
            + translations[source_body]
            - self.base_center[target_body]
            - translations[target_body]
        )
        margin = self.operator._pp_dhat(source_body, target_body)
        return center_delta.norm() <= (self.body_radius[source_body] + self.body_radius[target_body] + margin)

    @ti.kernel
    def _assemble_energy_gradient(
        self,
        translations: ti.template(),
        anchor_stiffness: float,
        kappa: float,
        continuation_distance: float,
    ):
        self.energy[None] = 0.0
        self.minimum_gap[None] = 1.0e30
        self.sampled_contacts[None] = 0
        self.active_contacts[None] = 0
        for body in range(self.body_num):
            value = translations[body]
            self.gradient[body] = anchor_stiffness * value
            ti.atomic_add(
                self.energy[None],
                0.5 * anchor_stiffness * value.dot(value),
            )
        for vertex_id, target_body in ti.ndrange(self.operator.vertex_num, self.body_num):
            source_body = self.operator.node2body[vertex_id]
            if self.operator._body_pair_allowed(source_body, target_body) and self._pair_may_contact(
                source_body, target_body, translations
            ):
                (
                    unused_source,
                    gap,
                    normal,
                    unused_gap_hessian,
                    inside,
                ) = self._gap_data(vertex_id, target_body, translations)
                if inside:
                    ti.atomic_add(self.sampled_contacts[None], 1)
                    ti.atomic_min(self.minimum_gap[None], gap)
                    dhat = self.operator._pp_dhat(source_body, target_body)
                    if gap < dhat:
                        ti.atomic_add(self.active_contacts[None], 1)
                        value, first, unused_second = self._continued_barrier(
                            gap,
                            dhat,
                            kappa,
                            continuation_distance,
                        )
                        coefficient = 0.5 * self.operator.node_area[vertex_id]
                        ti.atomic_add(self.energy[None], coefficient * value)
                        contact_gradient = coefficient * first * normal
                        for component in ti.static(range(3)):
                            ti.atomic_add(
                                self.gradient[source_body][component],
                                contact_gradient[component],
                            )
                            ti.atomic_add(
                                self.gradient[target_body][component],
                                -contact_gradient[component],
                            )

    @ti.kernel
    def _classify_free(self, tolerance: float):
        for body, component in ti.ndrange(self.body_num, 3):
            value = self.translation[body][component]
            gradient = self.gradient[body][component]
            at_lower = value <= self.lower[body][component] + tolerance
            at_upper = value >= self.upper[body][component] - tolerance
            active = (at_lower and gradient > 0.0) or (at_upper and gradient < 0.0)
            self.free[body][component] = 0 if active else 1

    @ti.kernel
    def _prepare_cg(self):
        self.reduction[None] = 0.0
        for body, component in ti.ndrange(self.body_num, 3):
            rhs = 0.0
            if self.free[body][component] == 1:
                rhs = -self.gradient[body][component]
            self.direction[body][component] = 0.0
            self.residual[body][component] = rhs
            self.cg_p[body][component] = rhs
            ti.atomic_add(self.reduction[None], rhs * rhs)

    @ti.kernel
    def _prepare_adjoint(self, loss_gradient: ti.template()):
        self.reduction[None] = 0.0
        for body, component in ti.ndrange(self.body_num, 3):
            rhs = 0.0
            if self.free[body][component] == 1:
                rhs = loss_gradient[body][component]
            self.direction[body][component] = 0.0
            self.residual[body][component] = rhs
            self.cg_p[body][component] = rhs
            ti.atomic_add(self.reduction[None], rhs * rhs)

    @ti.kernel
    def _apply_hessian(
        self,
        vector: ti.template(),
        output: ti.template(),
        anchor_stiffness: float,
        kappa: float,
        continuation_distance: float,
        hessian_shift: float,
    ):
        for body, component in ti.ndrange(self.body_num, 3):
            if self.free[body][component] == 1:
                output[body][component] = (anchor_stiffness + hessian_shift) * vector[body][component]
            else:
                output[body][component] = vector[body][component]
        for vertex_id, target_body in ti.ndrange(self.operator.vertex_num, self.body_num):
            source_body = self.operator.node2body[vertex_id]
            if self.operator._body_pair_allowed(source_body, target_body) and self._pair_may_contact(
                source_body, target_body, self.translation
            ):
                (
                    unused_source,
                    gap,
                    normal,
                    gap_hessian,
                    inside,
                ) = self._gap_data(vertex_id, target_body, self.translation)
                dhat = self.operator._pp_dhat(source_body, target_body)
                if inside and gap < dhat:
                    unused_value, first, second = self._continued_barrier(
                        gap,
                        dhat,
                        kappa,
                        continuation_distance,
                    )
                    coefficient = 0.5 * self.operator.node_area[vertex_id]
                    local_hessian = coefficient * (second * normal.outer_product(normal) + first * gap_hessian)
                    relative_vector = ti.Vector.zero(float, 3)
                    for component in ti.static(range(3)):
                        if self.free[source_body][component] == 1:
                            relative_vector[component] += vector[source_body][component]
                        if self.free[target_body][component] == 1:
                            relative_vector[component] -= vector[target_body][component]
                    product = local_hessian @ relative_vector
                    for component in ti.static(range(3)):
                        if self.free[source_body][component] == 1:
                            ti.atomic_add(
                                output[source_body][component],
                                product[component],
                            )
                        if self.free[target_body][component] == 1:
                            ti.atomic_add(
                                output[target_body][component],
                                -product[component],
                            )

    @ti.kernel
    def _dot(self, left: ti.template(), right: ti.template()):
        self.reduction[None] = 0.0
        for body, component in ti.ndrange(self.body_num, 3):
            ti.atomic_add(
                self.reduction[None],
                left[body][component] * right[body][component],
            )

    @ti.kernel
    def _cg_update(self, alpha: float):
        self.reduction[None] = 0.0
        for body, component in ti.ndrange(self.body_num, 3):
            if self.free[body][component] == 1:
                self.direction[body][component] += alpha * self.cg_p[body][component]
                self.residual[body][component] -= alpha * self.cg_Ap[body][component]
                ti.atomic_add(
                    self.reduction[None],
                    self.residual[body][component] ** 2,
                )

    @ti.kernel
    def _cg_next(self, beta: float):
        for body, component in ti.ndrange(self.body_num, 3):
            if self.free[body][component] == 1:
                self.cg_p[body][component] = self.residual[body][component] + beta * self.cg_p[body][component]
            else:
                self.cg_p[body][component] = 0.0

    @ti.kernel
    def _finish_adjoint(self, anchor_stiffness: float):
        for body, component in ti.ndrange(self.body_num, 3):
            self.adjoint_gradient[body][component] = (
                anchor_stiffness * self.direction[body][component] if self.free[body][component] == 1 else 0.0
            )

    @ti.kernel
    def _add_direct_adjoint(self, direct_gradient: ti.template()):
        for body in range(self.body_num):
            self.adjoint_gradient[body] += direct_gradient[body]

    @ti.kernel
    def _fallback_direction(self):
        for body, component in ti.ndrange(self.body_num, 3):
            self.direction[body][component] = (
                -self.gradient[body][component] if self.free[body][component] == 1 else 0.0
            )

    @ti.kernel
    def _make_trial(self, alpha: float):
        for body, component in ti.ndrange(self.body_num, 3):
            value = self.translation[body][component] + alpha * self.direction[body][component]
            self.trial_translation[body][component] = ti.min(
                self.upper[body][component],
                ti.max(self.lower[body][component], value),
            )

    @ti.kernel
    def _accept_trial(self):
        for body in range(self.body_num):
            self.translation[body] = self.trial_translation[body]

    @ti.kernel
    def _apply_to_operator_controls(self):
        for body, control in ti.ndrange(self.body_num, 4):
            value = self.base_y[body * 4 + control] + self.translation[body]
            self.operator.y[body * 4 + control] = value
            self.operator.hat_y[body * 4 + control] = value
            self.operator.tilde_y[body * 4 + control] = value
            self.operator.previous_y[body * 4 + control] = value

    @ti.kernel
    def apply_to_generated_positions(self, positions: ti.template()):
        for body in range(self.body_num):
            positions[body] += self.translation[body]

    def _linear_solve(
        self,
        anchor_stiffness,
        kappa,
        continuation_distance,
        hessian_shift,
        tolerance,
        maximum_iterations,
    ):
        self._prepare_cg()
        residual2 = float(self.reduction[None])
        initial2 = max(residual2, 1.0e-30)
        for _ in range(int(maximum_iterations)):
            if residual2 <= tolerance * tolerance * initial2:
                break
            self._apply_hessian(
                self.cg_p,
                self.cg_Ap,
                anchor_stiffness,
                kappa,
                continuation_distance,
                hessian_shift,
            )
            self._dot(self.cg_p, self.cg_Ap)
            denominator = float(self.reduction[None])
            if not math.isfinite(denominator) or denominator <= 1.0e-30:
                self._fallback_direction()
                return False
            alpha = residual2 / denominator
            self._cg_update(alpha)
            next_residual2 = float(self.reduction[None])
            beta = next_residual2 / max(residual2, 1.0e-30)
            self._cg_next(beta)
            residual2 = next_residual2
        return True

    def _maximum_feasible_step(self):
        """Return the line-search cap; mesh subclasses override with ACCD."""
        return 1.0

    def solve_adjoint(
        self,
        loss_final_center_gradient,
        *,
        direct_anchor_gradient=None,
        linear_tolerance=1.0e-7,
        maximum_iterations=500,
    ):
        """Backpropagate through the converged packing projection on device.

        If the random centers are ``p`` and the projected translations are
        ``t``, the inner equation is ``W t + grad E(p+t)=0``.  Therefore the
        final-center sensitivity is obtained from
        ``(W+E'').T lambda=dL/d(p+t)`` and ``dL/dp=W.T lambda``.  Bound-active
        coordinates use the reduced KKT system and return zero sensitivity.
        ``loss_final_center_gradient`` and the returned field both stay on
        the Taichi device.
        """
        if not self.last_success:
            raise SimulationStateError("solve() must reach feasibility before adjoint")
        self._assemble_energy_gradient(
            self.translation,
            self.last_anchor_stiffness,
            self.last_kappa,
            self.last_delta,
        )
        self._classify_free(1.0e-12)
        self._prepare_adjoint(loss_final_center_gradient)
        residual2 = float(self.reduction[None])
        initial2 = max(residual2, 1.0e-30)
        for _ in range(int(maximum_iterations)):
            if residual2 <= linear_tolerance * linear_tolerance * initial2:
                break
            self._apply_hessian(
                self.cg_p,
                self.cg_Ap,
                self.last_anchor_stiffness,
                self.last_kappa,
                self.last_delta,
                0.0,
            )
            self._dot(self.cg_p, self.cg_Ap)
            denominator = float(self.reduction[None])
            if not math.isfinite(denominator) or denominator <= 1.0e-30:
                raise SimulationStateError("DiffIPC adjoint Krylov solve encountered non-positive curvature")
            alpha = residual2 / denominator
            self._cg_update(alpha)
            next_residual2 = float(self.reduction[None])
            beta = next_residual2 / max(residual2, 1.0e-30)
            self._cg_next(beta)
            residual2 = next_residual2
        if residual2 > linear_tolerance * linear_tolerance * initial2:
            raise LinearSolveError("DiffIPC adjoint Krylov solve did not converge")
        self._finish_adjoint(self.last_anchor_stiffness)
        if direct_anchor_gradient is not None:
            self._add_direct_adjoint(direct_anchor_gradient)
        return self.adjoint_gradient

    def solve(
        self,
        *,
        dhat,
        kappa,
        anchor_stiffness=1.0,
        continuation_ratio=0.1,
        stiffness_growth=10.0,
        maximum_stages=10,
        maximum_iterations=100,
        gradient_tolerance=1.0e-8,
        gap_tolerance=1.0e-7,
        linear_tolerance=1.0e-6,
        linear_maximum_iterations=200,
        line_search_maximum_iterations=40,
        hessian_shift=1.0e-9,
    ):
        continuation_distance = continuation_ratio * float(dhat)
        total_iterations = 0
        initial_gap = None
        final_energy = np.inf
        for stage in range(int(maximum_stages)):
            stage_kappa = float(kappa) * float(stiffness_growth) ** stage
            for _ in range(int(maximum_iterations)):
                total_iterations += 1
                self._assemble_energy_gradient(
                    self.translation,
                    anchor_stiffness,
                    stage_kappa,
                    continuation_distance,
                )
                energy = float(self.energy[None])
                minimum_gap = float(self.minimum_gap[None])
                if initial_gap is None:
                    initial_gap = minimum_gap
                self._classify_free(1.0e-12)
                self._dot(self.gradient, self.gradient)
                gradient_norm = math.sqrt(max(float(self.reduction[None]), 0.0))
                if gradient_norm <= gradient_tolerance:
                    break
                self._linear_solve(
                    anchor_stiffness,
                    stage_kappa,
                    continuation_distance,
                    hessian_shift,
                    linear_tolerance,
                    linear_maximum_iterations,
                )
                self._dot(self.gradient, self.direction)
                slope = float(self.reduction[None])
                if not math.isfinite(slope) or slope >= 0.0:
                    self._fallback_direction()
                    self._dot(self.gradient, self.direction)
                    slope = float(self.reduction[None])
                alpha = min(1.0, float(self._maximum_feasible_step()))
                accepted = False
                for _ in range(int(line_search_maximum_iterations)):
                    self._make_trial(alpha)
                    self._assemble_energy_gradient(
                        self.trial_translation,
                        anchor_stiffness,
                        stage_kappa,
                        continuation_distance,
                    )
                    trial_energy = float(self.energy[None])
                    if math.isfinite(trial_energy) and trial_energy <= energy + 1.0e-4 * alpha * slope:
                        self._accept_trial()
                        final_energy = trial_energy
                        accepted = True
                        break
                    alpha *= 0.5
                if not accepted:
                    break
            self._assemble_energy_gradient(
                self.translation,
                anchor_stiffness,
                stage_kappa,
                continuation_distance,
            )
            final_energy = float(self.energy[None])
            minimum_gap = float(self.minimum_gap[None])
            self.last_stages = stage + 1
            if minimum_gap >= gap_tolerance:
                break
        self.last_kappa = stage_kappa
        self.last_delta = continuation_distance
        self.last_anchor_stiffness = float(anchor_stiffness)
        self.last_hessian_shift = float(hessian_shift)
        self.last_iterations = total_iterations
        self.last_success = bool(float(self.minimum_gap[None]) >= gap_tolerance)
        self._apply_to_operator_controls()
        return {
            "success": self.last_success,
            "initial_minimum_gap": float(initial_gap),
            "minimum_gap": float(self.minimum_gap[None]),
            "energy": final_energy,
            "iterations": total_iterations,
            "continuation_stages": self.last_stages,
            "active_contacts": int(self.active_contacts[None]),
        }

"""Differentiable IPC projectors for affine bodies."""

import math

import numpy as np
import taichi as ti

from src.contact_detection.continuous_contact_detection import (
    ccd_mode_parameters,
    edge_edge_accd,
    edge_edge_ccd,
    point_triangle_accd,
    point_triangle_ccd,
)
from src.physics_model.contact_model.ipc.ContactDistance import (
    edge_edge_distance_grad_hess,
    point_triangle_distance_grad,
    point_triangle_distance_grad_hess,
)
from src.physics_model.contact_model.ipc.ContactMollifier import (
    edge_edge_mollifier,
    edge_edge_mollifier_threshold,
)

from .AffineBodyOperator import TaichiAffineBodyOperator


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
            raise RuntimeError("solve() must reach feasibility before adjoint")
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
                raise RuntimeError("DiffIPC adjoint Krylov solve encountered non-positive curvature")
            alpha = residual2 / denominator
            self._cg_update(alpha)
            next_residual2 = float(self.reduction[None])
            beta = next_residual2 / max(residual2, 1.0e-30)
            self._cg_next(beta)
            residual2 = next_residual2
        if residual2 > linear_tolerance * linear_tolerance * initial2:
            raise RuntimeError("DiffIPC adjoint Krylov solve did not converge")
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


@ti.data_oriented
class TaichiAffineMeshDiffIPCProjector(TaichiAffineDiffIPCProjector):
    """Triangle-mesh overlap recovery followed by strict PT/EE DiffIPC.

    The recovery phase is the rigid-translation pullback of the forward flow
    in Minarcik et al., *Untangling Surfaces via Shape and Mesh Repulsion*.
    Cross-body Gaussian surface repulsion and the intersecting-triangle
    Minkowski penalty are evaluated on the surface vertices, then summed onto
    each affine body's translation.  ARAP is intentionally absent: it is
    invariant under the only initialization DOFs allowed here and every
    particle mesh therefore remains exactly rigid.

    Once all inter-body intersections and containments are gone, the second
    phase minimizes the ordinary mesh IPC PT/EE barrier from that feasible
    state.  ACCD caps every line-search step, and the converged exact Hessian
    is reused by :meth:`solve_adjoint`.
    """

    def __init__(self, operator, base_controls, translation_bounds):
        super().__init__(
            operator,
            base_controls,
            translation_bounds,
            contact_representation="TriangleMesh",
        )
        self.mesh_intersections = ti.field(ti.i32, shape=())
        self.mesh_inside_vertices = ti.field(ti.i32, shape=())
        self.maximum_gradient = ti.field(float, shape=())
        self._ccd_mode = "accd"
        self._ccd_eta = 0.1
        self._ccd_thickness = 1.0e-9
        self._ccd_maximum_iterations = 50
        self.last_untangle_result = None
        self.ccd_positions = ti.Vector.field(3, float, shape=max(1, operator.vertex_num))
        self.ccd_directions = ti.Vector.field(3, float, shape=max(1, operator.vertex_num))

    @ti.func
    def _mesh_pair_may_interact(
        self,
        body_i,
        body_j,
        translations: ti.template(),
        margin,
    ):
        center_delta = self.base_center[body_i] + translations[body_i] - self.base_center[body_j] - translations[body_j]
        return center_delta.norm() <= (self.body_radius[body_i] + self.body_radius[body_j] + margin)

    @ti.func
    def _triangle_minkowski(self, a0, a1, a2, b0, b1, b2):
        """Triangle Minkowski support penalty."""
        face_a = ti.Matrix.rows([a0, a1, a2])
        face_b = ti.Matrix.rows([b0, b1, b2])
        edge_a = ti.Matrix.rows([a1 - a0, a2 - a1, a0 - a2])
        edge_b = ti.Matrix.rows([b1 - b0, b2 - b1, b0 - b2])
        # The unique SAT set avoids duplicate axes in Taichi kernels.
        axes = ti.Matrix.zero(float, 11, 3)
        normal_a = (a1 - a0).cross(a2 - a0)
        normal_b = (b1 - b0).cross(b2 - b0)
        for component in ti.static(range(3)):
            axes[0, component] = normal_a[component]
            axes[1, component] = normal_b[component]
        for edge_i, edge_j in ti.ndrange(3, 3):
            axis = ti.Vector(
                [
                    edge_a[edge_i, 0],
                    edge_a[edge_i, 1],
                    edge_a[edge_i, 2],
                ]
            ).cross(
                ti.Vector(
                    [
                        edge_b[edge_j, 0],
                        edge_b[edge_j, 1],
                        edge_b[edge_j, 2],
                    ]
                )
            )
            axis_id = 2 + 3 * edge_i + edge_j
            for component in ti.static(range(3)):
                axes[axis_id, component] = axis[component]
        best_phi = -1.0e30
        best_normal = ti.Vector.zero(float, 3)
        for axis_id in range(11):
            raw_normal = ti.Vector(
                [
                    axes[axis_id, 0],
                    axes[axis_id, 1],
                    axes[axis_id, 2],
                ]
            )
            normal_length = raw_normal.norm()
            if normal_length > 1.0e-20:
                support_max = -1.0e30
                support_min = 1.0e30
                for site_a, site_b in ti.ndrange(3, 3):
                    candidate = raw_normal.dot(
                        ti.Vector(
                            [
                                face_a[site_a, 0] - face_b[site_b, 0],
                                face_a[site_a, 1] - face_b[site_b, 1],
                                face_a[site_a, 2] - face_b[site_b, 2],
                            ]
                        )
                    )
                    support_max = ti.max(support_max, candidate)
                    support_min = ti.min(support_min, candidate)
                phi = -support_max / normal_length
                phi_negative = support_min / normal_length
                if phi > best_phi:
                    best_phi = phi
                    best_normal = raw_normal / normal_length
                if phi_negative > best_phi:
                    best_phi = phi_negative
                    best_normal = -raw_normal / normal_length
        if best_phi < -0.5e30:
            best_phi = 0.0
            best_normal = ti.Vector.zero(float, 3)
        return best_phi, best_normal

    @ti.func
    def _segment_triangle_intersection(self, endpoint0, endpoint1, a, b, c, tolerance):
        direction = endpoint1 - endpoint0
        edge0 = b - a
        edge1 = c - a
        cross = direction.cross(edge1)
        determinant = edge0.dot(cross)
        hit = False
        if ti.abs(determinant) > tolerance:
            inverse_determinant = 1.0 / determinant
            offset = endpoint0 - a
            u = offset.dot(cross) * inverse_determinant
            if u >= -tolerance and u <= 1.0 + tolerance:
                q = offset.cross(edge0)
                v = direction.dot(q) * inverse_determinant
                if v >= -tolerance and u + v <= 1.0 + tolerance:
                    parameter = edge1.dot(q) * inverse_determinant
                    hit = parameter >= -tolerance and parameter <= 1.0 + tolerance
        return hit

    @ti.func
    def _orient2d(self, p, q, r, drop_axis):
        value = 0.0
        if drop_axis == 0:
            value = (q[1] - p[1]) * (r[2] - p[2]) - (q[2] - p[2]) * (r[1] - p[1])
        elif drop_axis == 1:
            value = (q[0] - p[0]) * (r[2] - p[2]) - (q[2] - p[2]) * (r[0] - p[0])
        else:
            value = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return value

    @ti.func
    def _coplanar_point_in_triangle(self, point, a, b, c, drop_axis, tolerance):
        s0 = self._orient2d(a, b, point, drop_axis)
        s1 = self._orient2d(b, c, point, drop_axis)
        s2 = self._orient2d(c, a, point, drop_axis)
        nonnegative = s0 >= -tolerance and s1 >= -tolerance and s2 >= -tolerance
        nonpositive = s0 <= tolerance and s1 <= tolerance and s2 <= tolerance
        return nonnegative or nonpositive

    @ti.func
    def _coplanar_segments_intersect(self, a0, a1, b0, b1, drop_axis, tolerance):
        a_b0 = self._orient2d(a0, a1, b0, drop_axis)
        a_b1 = self._orient2d(a0, a1, b1, drop_axis)
        b_a0 = self._orient2d(b0, b1, a0, drop_axis)
        b_a1 = self._orient2d(b0, b1, a1, drop_axis)
        hit = a_b0 * a_b1 <= tolerance * tolerance and b_a0 * b_a1 <= tolerance * tolerance
        # The orientation-product test alone marks every pair of collinear
        # segments as intersecting, including disjoint ones.  Require overlap
        # of their projected bounding intervals in that degenerate case.
        collinear = (
            ti.abs(a_b0) <= tolerance
            and ti.abs(a_b1) <= tolerance
            and ti.abs(b_a0) <= tolerance
            and ti.abs(b_a1) <= tolerance
        )
        if hit and collinear:
            for component in ti.static(range(3)):
                if component != drop_axis:
                    lower_a = ti.min(a0[component], a1[component])
                    upper_a = ti.max(a0[component], a1[component])
                    lower_b = ti.min(b0[component], b1[component])
                    upper_b = ti.max(b0[component], b1[component])
                    hit = hit and (ti.max(lower_a, lower_b) <= ti.min(upper_a, upper_b) + tolerance)
        return hit

    @ti.func
    def _triangles_intersect(self, a0, a1, a2, b0, b1, b2, tolerance):
        hit = (
            self._segment_triangle_intersection(a0, a1, b0, b1, b2, tolerance)
            or self._segment_triangle_intersection(a1, a2, b0, b1, b2, tolerance)
            or self._segment_triangle_intersection(a2, a0, b0, b1, b2, tolerance)
            or self._segment_triangle_intersection(b0, b1, a0, a1, a2, tolerance)
            or self._segment_triangle_intersection(b1, b2, a0, a1, a2, tolerance)
            or self._segment_triangle_intersection(b2, b0, a0, a1, a2, tolerance)
        )
        normal_a = (a1 - a0).cross(a2 - a0)
        normal_b = (b1 - b0).cross(b2 - b0)
        normal_scale = normal_a.norm() * normal_b.norm()
        coplanar = (
            normal_scale > 1.0e-30
            and normal_a.cross(normal_b).norm() <= tolerance * normal_scale
            and ti.abs((b0 - a0).dot(normal_a)) <= tolerance * ti.max(normal_a.norm(), 1.0)
        )
        if not hit and coplanar:
            absolute_normal = ti.abs(normal_a)
            drop_axis = 0
            if absolute_normal[1] > absolute_normal[drop_axis]:
                drop_axis = 1
            if absolute_normal[2] > absolute_normal[drop_axis]:
                drop_axis = 2
            hit = (
                self._coplanar_point_in_triangle(a0, b0, b1, b2, drop_axis, tolerance)
                or self._coplanar_point_in_triangle(b0, a0, a1, a2, drop_axis, tolerance)
                or self._coplanar_segments_intersect(a0, a1, b0, b1, drop_axis, tolerance)
                or self._coplanar_segments_intersect(a1, a2, b0, b1, drop_axis, tolerance)
                or self._coplanar_segments_intersect(a2, a0, b0, b1, drop_axis, tolerance)
                or self._coplanar_segments_intersect(a0, a1, b1, b2, drop_axis, tolerance)
                or self._coplanar_segments_intersect(a1, a2, b1, b2, drop_axis, tolerance)
                or self._coplanar_segments_intersect(a2, a0, b1, b2, drop_axis, tolerance)
                or self._coplanar_segments_intersect(a0, a1, b2, b0, drop_axis, tolerance)
                or self._coplanar_segments_intersect(a1, a2, b2, b0, drop_axis, tolerance)
                or self._coplanar_segments_intersect(a2, a0, b2, b0, drop_axis, tolerance)
            )
        return hit

    @ti.func
    def _point_inside_mesh_body(
        self,
        point,
        target_body,
        translations: ti.template(),
    ):
        # Generalized winding number through the signed solid angle.  The
        # absolute value makes the containment check independent of whether
        # the input uses inward or outward consistent winding.
        solid_angle = 0.0
        for face_id in range(self.operator.face_num):
            if self.operator.face2body[face_id] == target_body:
                face = self.operator.faces[face_id]
                a = self._surface_vertex(face[0], translations) - point
                b = self._surface_vertex(face[1], translations) - point
                c = self._surface_vertex(face[2], translations) - point
                la = ti.max(a.norm(), 1.0e-30)
                lb = ti.max(b.norm(), 1.0e-30)
                lc = ti.max(c.norm(), 1.0e-30)
                numerator = a.dot(b.cross(c))
                denominator = la * lb * lc + a.dot(b) * lc + b.dot(c) * la + c.dot(a) * lb
                solid_angle += 2.0 * ti.atan2(numerator, denominator)
        return ti.abs(solid_angle) > 2.0 * math.pi

    @ti.kernel
    def _assemble_untangle_energy_gradient(
        self,
        translations: ti.template(),
        anchor_stiffness: float,
        separation_weight: float,
        gaussian_weight: float,
        gaussian_epsilon: float,
        gaussian_cutoff: float,
        minkowski_weight: float,
        inclusion_weight: float,
        intersection_tolerance: float,
    ):
        self.energy[None] = 0.0
        self.mesh_intersections[None] = 0
        self.mesh_inside_vertices[None] = 0
        self.sampled_contacts[None] = 0
        self.active_contacts[None] = 0
        for body in range(self.body_num):
            value = translations[body]
            self.gradient[body] = anchor_stiffness * value
            ti.atomic_add(
                self.energy[None],
                0.5 * anchor_stiffness * value.dot(value),
            )

        # A translation-only restriction removes the local deformation modes
        # available to the original surface untangler.  This smooth bounding-
        # sphere term supplies one coherent separating direction per body
        # pair, preventing the many triangle-pair gradients from cancelling
        # in a multi-particle knot.  It is only a recovery merit term: the
        # exact mesh intersection/containment predicates decide termination.
        # ponytail: initialization-only O(B^2) sphere broad phase; replace it
        # with a body BVH only if dense-packing initialization profiles demand it.
        for body_i, body_j in ti.ndrange(self.body_num, self.body_num):
            if body_i < body_j and self.operator._body_pair_allowed(body_i, body_j):
                center_delta = (
                    self.base_center[body_i] + translations[body_i] - self.base_center[body_j] - translations[body_j]
                )
                center_distance = center_delta.norm()
                radius_sum = self.body_radius[body_i] + self.body_radius[body_j]
                overlap = ti.max(radius_sum - center_distance, 0.0)
                if overlap > 0.0:
                    direction = ti.Vector.zero(float, 3)
                    if center_distance > 1.0e-20:
                        direction = center_delta / center_distance
                    else:
                        # Deterministically break the exactly concentric
                        # translation symmetry without host-side randomness.
                        direction[body_i % 3] = 1.0
                    ti.atomic_add(
                        self.energy[None],
                        0.5 * separation_weight * overlap * overlap,
                    )
                    pair_gradient = -separation_weight * overlap * direction
                    for component in ti.static(range(3)):
                        ti.atomic_add(
                            self.gradient[body_i][component],
                            pair_gradient[component],
                        )
                        ti.atomic_add(
                            self.gradient[body_j][component],
                            -pair_gradient[component],
                        )

        epsilon2 = gaussian_epsilon * gaussian_epsilon
        inverse_epsilon2 = 1.0 / ti.max(epsilon2, 1.0e-30)
        inverse_epsilon4 = inverse_epsilon2 * inverse_epsilon2
        cutoff2 = gaussian_cutoff * gaussian_cutoff
        for vertex_i, vertex_j in ti.ndrange(self.operator.vertex_num, self.operator.vertex_num):
            if vertex_i < vertex_j:
                body_i = self.operator.node2body[vertex_i]
                body_j = self.operator.node2body[vertex_j]
                if self.operator._body_pair_allowed(body_i, body_j) and self._mesh_pair_may_interact(
                    body_i,
                    body_j,
                    translations,
                    gaussian_cutoff,
                ):
                    point_i = self._surface_vertex(vertex_i, translations)
                    point_j = self._surface_vertex(vertex_j, translations)
                    delta = point_i - point_j
                    distance2 = delta.dot(delta)
                    if gaussian_cutoff <= 0.0 or distance2 <= cutoff2:
                        area_product = self.operator.node_area[vertex_i] * self.operator.node_area[vertex_j]
                        exponential = ti.exp(-distance2 * inverse_epsilon2)
                        value = gaussian_weight * area_product * exponential * inverse_epsilon2
                        ti.atomic_add(self.energy[None], value)
                        coefficient = -2.0 * gaussian_weight * area_product * exponential * inverse_epsilon4
                        pair_gradient = coefficient * delta
                        for component in ti.static(range(3)):
                            ti.atomic_add(
                                self.gradient[body_i][component],
                                pair_gradient[component],
                            )
                            ti.atomic_add(
                                self.gradient[body_j][component],
                                -pair_gradient[component],
                            )

        for face_i, face_j in ti.ndrange(self.operator.face_num, self.operator.face_num):
            if face_i < face_j:
                body_i = self.operator.face2body[face_i]
                body_j = self.operator.face2body[face_j]
                if self.operator._body_pair_allowed(body_i, body_j) and self._mesh_pair_may_interact(
                    body_i, body_j, translations, 0.0
                ):
                    triangle_i = self.operator.faces[face_i]
                    triangle_j = self.operator.faces[face_j]
                    a0 = self._surface_vertex(triangle_i[0], translations)
                    a1 = self._surface_vertex(triangle_i[1], translations)
                    a2 = self._surface_vertex(triangle_i[2], translations)
                    b0 = self._surface_vertex(triangle_j[0], translations)
                    b1 = self._surface_vertex(triangle_j[1], translations)
                    b2 = self._surface_vertex(triangle_j[2], translations)
                    if self._triangles_intersect(
                        a0,
                        a1,
                        a2,
                        b0,
                        b1,
                        b2,
                        intersection_tolerance,
                    ):
                        ti.atomic_add(self.mesh_intersections[None], 1)
                        ti.atomic_add(self.active_contacts[None], 1)
                        phi, normal = self._triangle_minkowski(a0, a1, a2, b0, b1, b2)
                        if phi < 0.0:
                            ti.atomic_add(
                                self.energy[None],
                                minkowski_weight * (-phi),
                            )
                            pair_gradient = minkowski_weight * normal
                            for component in ti.static(range(3)):
                                ti.atomic_add(
                                    self.gradient[body_i][component],
                                    pair_gradient[component],
                                )
                                ti.atomic_add(
                                    self.gradient[body_j][component],
                                    -pair_gradient[component],
                                )

        # Face intersections do not detect strict containment.  The winding
        # test closes that gap and uses the exact nearest PT distance gradient
        # to select an escape direction without constructing an SDF.
        for vertex_id, target_body in ti.ndrange(self.operator.vertex_num, self.body_num):
            source_body = self.operator.node2body[vertex_id]
            if self.operator._body_pair_allowed(source_body, target_body) and self._mesh_pair_may_interact(
                source_body, target_body, translations, 0.0
            ):
                point = self._surface_vertex(vertex_id, translations)
                if self._point_inside_mesh_body(point, target_body, translations):
                    ti.atomic_add(self.mesh_inside_vertices[None], 1)
                    ti.atomic_add(self.active_contacts[None], 1)
                    minimum_distance2 = 1.0e30
                    minimum_face = -1
                    for face_id in range(self.operator.face_num):
                        if self.operator.face2body[face_id] == target_body:
                            face = self.operator.faces[face_id]
                            distance2, unused_gradient, unused_type = point_triangle_distance_grad(
                                point,
                                self._surface_vertex(face[0], translations),
                                self._surface_vertex(face[1], translations),
                                self._surface_vertex(face[2], translations),
                            )
                            if distance2 < minimum_distance2:
                                minimum_distance2 = distance2
                                minimum_face = face_id
                    if minimum_face >= 0:
                        face = self.operator.faces[minimum_face]
                        distance2, distance_gradient, unused_type = point_triangle_distance_grad(
                            point,
                            self._surface_vertex(face[0], translations),
                            self._surface_vertex(face[1], translations),
                            self._surface_vertex(face[2], translations),
                        )
                        distance = ti.sqrt(ti.max(distance2, 1.0e-30))
                        coefficient = inclusion_weight * self.operator.node_area[vertex_id]
                        ti.atomic_add(self.energy[None], coefficient * distance)
                        for component in ti.static(range(3)):
                            component_gradient = coefficient * distance_gradient[component] / (2.0 * distance)
                            ti.atomic_add(
                                self.gradient[source_body][component],
                                component_gradient,
                            )
                            ti.atomic_add(
                                self.gradient[target_body][component],
                                -component_gradient,
                            )

    @ti.kernel
    def _compute_maximum_gradient(self):
        self.maximum_gradient[None] = 0.0
        for body in range(self.body_num):
            free_gradient = ti.Vector.zero(float, 3)
            for component in ti.static(range(3)):
                if self.free[body][component] == 1:
                    free_gradient[component] = self.gradient[body][component]
            ti.atomic_max(self.maximum_gradient[None], free_gradient.norm())

    def untangle(
        self,
        *,
        gaussian_epsilon,
        separation_weight=10.0,
        gaussian_weight=1.0,
        gaussian_cutoff=None,
        minkowski_weight=1.0,
        inclusion_weight=1.0,
        anchor_stiffness=1.0,
        stiffness_growth=10.0,
        maximum_stages=6,
        maximum_iterations=250,
        gradient_tolerance=1.0e-9,
        line_search_maximum_iterations=40,
        step_fraction=0.25,
        intersection_tolerance=1.0e-10,
    ):
        """Remove inter-body mesh intersections without deforming a mesh."""
        gaussian_epsilon = float(gaussian_epsilon)
        if not math.isfinite(gaussian_epsilon) or gaussian_epsilon <= 0.0:
            raise ValueError("Mesh untangling GaussianEpsilon must be positive")
        if gaussian_cutoff is None:
            gaussian_cutoff = 4.0 * gaussian_epsilon
        gaussian_cutoff = float(gaussian_cutoff)
        if not math.isfinite(gaussian_cutoff) or gaussian_cutoff <= 0.0:
            raise ValueError("Mesh untangling GaussianCutoff must be positive")
        named_weights = {
            "SeparationWeight": separation_weight,
            "GaussianWeight": gaussian_weight,
            "MinkowskiWeight": minkowski_weight,
            "InclusionWeight": inclusion_weight,
            "AnchorStiffness": anchor_stiffness,
        }
        for name, weight in named_weights.items():
            if not math.isfinite(float(weight)) or float(weight) < 0.0:
                raise ValueError(f"Mesh untangling {name} must be finite and nonnegative")
        if not math.isfinite(float(stiffness_growth)) or float(stiffness_growth) <= 1.0:
            raise ValueError("Mesh untangling stiffness growth must be finite and greater " "than one")
        if int(maximum_stages) <= 0 or int(maximum_iterations) <= 0:
            raise ValueError("Mesh untangling stage and iteration limits must be positive")
        if int(line_search_maximum_iterations) <= 0:
            raise ValueError("Mesh untangling line-search iteration limit must be positive")
        if not math.isfinite(float(step_fraction)) or float(step_fraction) <= 0.0:
            raise ValueError("Mesh untangling step fraction must be positive")
        if not math.isfinite(float(intersection_tolerance)) or float(intersection_tolerance) <= 0.0:
            raise ValueError("Mesh untangling intersection tolerance must be positive")
        total_iterations = 0
        initial_overlap_count = None
        final_energy = math.inf
        success = False
        final_stage = 0
        for stage in range(int(maximum_stages)):
            final_stage = stage + 1
            stage_scale = float(stiffness_growth) ** stage
            # Keep the local shape terms fixed and continue only the coherent
            # rigid-pair term.  Growing every face penalty together preserves
            # cancellation in a multi-particle knot; growing the body-pair
            # term removes that translation-only null mode while retaining
            # the paper's Minkowski directions as local geometric guidance.
            stage_minkowski = float(minkowski_weight)
            stage_separation = float(separation_weight) * stage_scale
            for _ in range(int(maximum_iterations)):
                total_iterations += 1
                self._assemble_untangle_energy_gradient(
                    self.translation,
                    float(anchor_stiffness),
                    stage_separation,
                    float(gaussian_weight),
                    gaussian_epsilon,
                    gaussian_cutoff,
                    stage_minkowski,
                    float(inclusion_weight),
                    float(intersection_tolerance),
                )
                final_energy = float(self.energy[None])
                overlap_count = int(self.mesh_intersections[None]) + int(self.mesh_inside_vertices[None])
                if initial_overlap_count is None:
                    initial_overlap_count = overlap_count
                if overlap_count == 0:
                    success = True
                    break
                self._classify_free(1.0e-12)
                self._fallback_direction()
                self._dot(self.gradient, self.direction)
                slope = float(self.reduction[None])
                gradient_norm = math.sqrt(max(-slope, 0.0))
                if gradient_norm <= float(gradient_tolerance):
                    break
                self._compute_maximum_gradient()
                maximum_gradient = max(float(self.maximum_gradient[None]), 1.0e-30)
                alpha = min(
                    1.0,
                    float(step_fraction) * gaussian_epsilon / maximum_gradient,
                )
                accepted = False
                for _ in range(int(line_search_maximum_iterations)):
                    self._make_trial(alpha)
                    self._assemble_untangle_energy_gradient(
                        self.trial_translation,
                        float(anchor_stiffness),
                        stage_separation,
                        float(gaussian_weight),
                        gaussian_epsilon,
                        gaussian_cutoff,
                        stage_minkowski,
                        float(inclusion_weight),
                        float(intersection_tolerance),
                    )
                    trial_energy = float(self.energy[None])
                    if math.isfinite(trial_energy) and trial_energy <= final_energy + 1.0e-4 * alpha * slope:
                        self._accept_trial()
                        final_energy = trial_energy
                        accepted = True
                        break
                    alpha *= 0.5
                if not accepted:
                    break
            if success:
                break
        self._assemble_untangle_energy_gradient(
            self.translation,
            float(anchor_stiffness),
            float(separation_weight) * float(stiffness_growth) ** max(final_stage - 1, 0),
            float(gaussian_weight),
            gaussian_epsilon,
            gaussian_cutoff,
            float(minkowski_weight),
            float(inclusion_weight),
            float(intersection_tolerance),
        )
        final_overlap_count = int(self.mesh_intersections[None]) + int(self.mesh_inside_vertices[None])
        success = final_overlap_count == 0
        result = {
            "success": success,
            "initial_overlap_count": int(initial_overlap_count or 0),
            "overlap_count": final_overlap_count,
            "triangle_intersections": int(self.mesh_intersections[None]),
            "inside_vertices": int(self.mesh_inside_vertices[None]),
            "energy": float(self.energy[None]),
            "iterations": total_iterations,
            "continuation_stages": final_stage,
        }
        self.last_untangle_result = result
        return result

    @ti.kernel
    def _assemble_energy_gradient(
        self,
        translations: ti.template(),
        anchor_stiffness: float,
        kappa: float,
        continuation_distance: float,
    ):
        # ``continuation_distance`` is intentionally unused here.  This phase
        # starts only after untangling and therefore evaluates the strict IPC
        # logarithmic barrier on positive PT/EE distances.
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

        for vertex_id, face_id in ti.ndrange(self.operator.vertex_num, self.operator.face_num):
            body_i = self.operator.node2body[vertex_id]
            body_j = self.operator.face2body[face_id]
            if self.operator._body_pair_allowed(body_i, body_j) and self._pair_may_contact(
                body_i, body_j, translations
            ):
                face = self.operator.faces[face_id]
                distance2, distance_gradient, unused_hessian, unused_type = point_triangle_distance_grad_hess(
                    self._surface_vertex(vertex_id, translations),
                    self._surface_vertex(face[0], translations),
                    self._surface_vertex(face[1], translations),
                    self._surface_vertex(face[2], translations),
                )
                distance = ti.sqrt(ti.max(distance2, 1.0e-30))
                ti.atomic_add(self.sampled_contacts[None], 1)
                ti.atomic_min(self.minimum_gap[None], distance)
                dhat = self.operator._pp_dhat(body_i, body_j)
                if distance < dhat:
                    ti.atomic_add(self.active_contacts[None], 1)
                    value, first, unused_second = self.operator._ipc_barrier_gap(distance, dhat, kappa)
                    coefficient = 0.25 * self.operator.node_area[vertex_id]
                    ti.atomic_add(self.energy[None], coefficient * value)
                    for component in ti.static(range(3)):
                        relative_gradient = distance_gradient[component] / (2.0 * distance)
                        contact_gradient = coefficient * first * relative_gradient
                        ti.atomic_add(
                            self.gradient[body_i][component],
                            contact_gradient,
                        )
                        ti.atomic_add(
                            self.gradient[body_j][component],
                            -contact_gradient,
                        )

        for edge_i, edge_j in ti.ndrange(self.operator.edge_num, self.operator.edge_num):
            if edge_i < edge_j:
                body_i = self.operator.edge2body[edge_i]
                body_j = self.operator.edge2body[edge_j]
                if self.operator._body_pair_allowed(body_i, body_j) and self._pair_may_contact(
                    body_i, body_j, translations
                ):
                    edge0 = self.operator.edges[edge_i]
                    edge1 = self.operator.edges[edge_j]
                    p0 = self._surface_vertex(edge0[0], translations)
                    p1 = self._surface_vertex(edge0[1], translations)
                    q0 = self._surface_vertex(edge1[0], translations)
                    q1 = self._surface_vertex(edge1[1], translations)
                    distance2, distance_gradient, unused_hessian, unused_type = edge_edge_distance_grad_hess(
                        p0, p1, q0, q1
                    )
                    distance = ti.sqrt(ti.max(distance2, 1.0e-30))
                    ti.atomic_add(self.sampled_contacts[None], 1)
                    ti.atomic_min(self.minimum_gap[None], distance)
                    dhat = self.operator._pp_dhat(body_i, body_j)
                    if distance < dhat:
                        ti.atomic_add(self.active_contacts[None], 1)
                        value, first, unused_second = self.operator._ipc_barrier_gap(distance, dhat, kappa)
                        eps_x = edge_edge_mollifier_threshold(
                            self.operator.rest_x[edge0[0]],
                            self.operator.rest_x[edge0[1]],
                            self.operator.rest_x[edge1[0]],
                            self.operator.rest_x[edge1[1]],
                        )
                        mollifier = edge_edge_mollifier(p0, p1, q0, q1, eps_x)
                        coefficient = (
                            0.25 * (self.operator.edge_area[edge_i] + self.operator.edge_area[edge_j]) * mollifier
                        )
                        ti.atomic_add(self.energy[None], coefficient * value)
                        for component in ti.static(range(3)):
                            relative_gradient = (distance_gradient[component] + distance_gradient[3 + component]) / (
                                2.0 * distance
                            )
                            contact_gradient = coefficient * first * relative_gradient
                            ti.atomic_add(
                                self.gradient[body_i][component],
                                contact_gradient,
                            )
                            ti.atomic_add(
                                self.gradient[body_j][component],
                                -contact_gradient,
                            )

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

        for vertex_id, face_id in ti.ndrange(self.operator.vertex_num, self.operator.face_num):
            body_i = self.operator.node2body[vertex_id]
            body_j = self.operator.face2body[face_id]
            if self.operator._body_pair_allowed(body_i, body_j) and self._pair_may_contact(
                body_i, body_j, self.translation
            ):
                face = self.operator.faces[face_id]
                distance2, distance_gradient, distance_hessian, unused_type = point_triangle_distance_grad_hess(
                    self._surface_vertex(vertex_id, self.translation),
                    self._surface_vertex(face[0], self.translation),
                    self._surface_vertex(face[1], self.translation),
                    self._surface_vertex(face[2], self.translation),
                )
                distance = ti.sqrt(ti.max(distance2, 1.0e-30))
                dhat = self.operator._pp_dhat(body_i, body_j)
                if distance < dhat:
                    unused_value, first, second = self.operator._ipc_barrier_gap(distance, dhat, kappa)
                    gradient_gap = ti.Vector.zero(float, 3)
                    hessian_gap = ti.Matrix.zero(float, 3, 3)
                    for row in ti.static(range(3)):
                        gradient_gap[row] = distance_gradient[row] / (2.0 * distance)
                        for column in ti.static(range(3)):
                            hessian_gap[row, column] = distance_hessian[row, column] / (
                                2.0 * distance
                            ) - distance_gradient[row] * distance_gradient[column] / (
                                4.0 * distance * distance * distance
                            )
                    coefficient = 0.25 * self.operator.node_area[vertex_id]
                    local_hessian = coefficient * (
                        second * gradient_gap.outer_product(gradient_gap) + first * hessian_gap
                    )
                    relative_vector = ti.Vector.zero(float, 3)
                    for component in ti.static(range(3)):
                        if self.free[body_i][component] == 1:
                            relative_vector[component] += vector[body_i][component]
                        if self.free[body_j][component] == 1:
                            relative_vector[component] -= vector[body_j][component]
                    product = local_hessian @ relative_vector
                    for component in ti.static(range(3)):
                        if self.free[body_i][component] == 1:
                            ti.atomic_add(output[body_i][component], product[component])
                        if self.free[body_j][component] == 1:
                            ti.atomic_add(output[body_j][component], -product[component])

        for edge_i, edge_j in ti.ndrange(self.operator.edge_num, self.operator.edge_num):
            if edge_i < edge_j:
                body_i = self.operator.edge2body[edge_i]
                body_j = self.operator.edge2body[edge_j]
                if self.operator._body_pair_allowed(body_i, body_j) and self._pair_may_contact(
                    body_i, body_j, self.translation
                ):
                    edge0 = self.operator.edges[edge_i]
                    edge1 = self.operator.edges[edge_j]
                    p0 = self._surface_vertex(edge0[0], self.translation)
                    p1 = self._surface_vertex(edge0[1], self.translation)
                    q0 = self._surface_vertex(edge1[0], self.translation)
                    q1 = self._surface_vertex(edge1[1], self.translation)
                    distance2, distance_gradient, distance_hessian, unused_type = edge_edge_distance_grad_hess(
                        p0, p1, q0, q1
                    )
                    distance = ti.sqrt(ti.max(distance2, 1.0e-30))
                    dhat = self.operator._pp_dhat(body_i, body_j)
                    if distance < dhat:
                        unused_value, first, second = self.operator._ipc_barrier_gap(distance, dhat, kappa)
                        gradient2 = ti.Vector.zero(float, 3)
                        hessian2 = ti.Matrix.zero(float, 3, 3)
                        for row in ti.static(range(3)):
                            gradient2[row] = distance_gradient[row] + distance_gradient[3 + row]
                            for column in ti.static(range(3)):
                                value = 0.0
                                for site_i, site_j in ti.static(ti.ndrange(2, 2)):
                                    value += distance_hessian[
                                        3 * site_i + row,
                                        3 * site_j + column,
                                    ]
                                hessian2[row, column] = value
                        gradient_gap = gradient2 / (2.0 * distance)
                        hessian_gap = ti.Matrix.zero(float, 3, 3)
                        for row, column in ti.static(ti.ndrange(3, 3)):
                            hessian_gap[row, column] = hessian2[row, column] / (2.0 * distance) - gradient2[
                                row
                            ] * gradient2[column] / (4.0 * distance * distance * distance)
                        eps_x = edge_edge_mollifier_threshold(
                            self.operator.rest_x[edge0[0]],
                            self.operator.rest_x[edge0[1]],
                            self.operator.rest_x[edge1[0]],
                            self.operator.rest_x[edge1[1]],
                        )
                        mollifier = edge_edge_mollifier(p0, p1, q0, q1, eps_x)
                        coefficient = (
                            0.25 * (self.operator.edge_area[edge_i] + self.operator.edge_area[edge_j]) * mollifier
                        )
                        local_hessian = coefficient * (
                            second * gradient_gap.outer_product(gradient_gap) + first * hessian_gap
                        )
                        relative_vector = ti.Vector.zero(float, 3)
                        for component in ti.static(range(3)):
                            if self.free[body_i][component] == 1:
                                relative_vector[component] += vector[body_i][component]
                            if self.free[body_j][component] == 1:
                                relative_vector[component] -= vector[body_j][component]
                        product = local_hessian @ relative_vector
                        for component in ti.static(range(3)):
                            if self.free[body_i][component] == 1:
                                ti.atomic_add(
                                    output[body_i][component],
                                    product[component],
                                )
                            if self.free[body_j][component] == 1:
                                ti.atomic_add(
                                    output[body_j][component],
                                    -product[component],
                                )

    @ti.kernel
    def _prepare_mesh_sweep(self):
        for vertex in range(self.operator.vertex_num):
            self.ccd_positions[vertex] = self._surface_vertex(vertex, self.translation)
            self.ccd_directions[vertex] = self.direction[self.operator.node2body[vertex]]

    @ti.kernel
    def _compute_mesh_ccd_step(
        self,
        eta: float,
        thickness: float,
        maximum_iterations: ti.i32,
        accd: ti.template(),
        point_count: ti.template(),
        point_vertices: ti.template(),
        point_faces: ti.template(),
        edge_count: ti.template(),
        edge_first: ti.template(),
        edge_second: ti.template(),
    ):
        self.operator.ccd_alpha[None] = 1.0
        for body, component in ti.ndrange(self.body_num, 3):
            direction = self.direction[body][component]
            alpha = 1.0
            if direction > 0.0:
                alpha = (self.upper[body][component] - self.translation[body][component]) / direction
            elif direction < 0.0:
                alpha = (self.lower[body][component] - self.translation[body][component]) / direction
            ti.atomic_min(
                self.operator.ccd_alpha[None],
                ti.max(0.0, ti.min(1.0, alpha)),
            )

        for candidate in range(point_count[None]):
            vertex_id = point_vertices[candidate]
            face_id = point_faces[candidate]
            body_i = self.operator.node2body[vertex_id]
            body_j = self.operator.face2body[face_id]
            if self.operator._body_pair_allowed(body_i, body_j):
                face = self.operator.faces[face_id]
                point_displacement = self.direction[body_i]
                face_displacement = self.direction[body_j]
                alpha = 1.0
                if ti.static(accd):
                    alpha = point_triangle_accd(
                        self._surface_vertex(vertex_id, self.translation),
                        self._surface_vertex(face[0], self.translation),
                        self._surface_vertex(face[1], self.translation),
                        self._surface_vertex(face[2], self.translation),
                        point_displacement,
                        face_displacement,
                        face_displacement,
                        face_displacement,
                        eta,
                        thickness,
                        maximum_iterations,
                    )
                else:
                    alpha = point_triangle_ccd(
                        self._surface_vertex(vertex_id, self.translation),
                        self._surface_vertex(face[0], self.translation),
                        self._surface_vertex(face[1], self.translation),
                        self._surface_vertex(face[2], self.translation),
                        point_displacement,
                        face_displacement,
                        face_displacement,
                        face_displacement,
                        eta,
                        maximum_iterations,
                    )
                ti.atomic_min(
                    self.operator.ccd_alpha[None],
                    ti.max(0.0, ti.min(1.0, alpha)),
                )

        for candidate in range(edge_count[None]):
            edge_i = edge_first[candidate]
            edge_j = edge_second[candidate]
            if edge_i < edge_j:
                body_i = self.operator.edge2body[edge_i]
                body_j = self.operator.edge2body[edge_j]
                if self.operator._body_pair_allowed(body_i, body_j):
                    edge0 = self.operator.edges[edge_i]
                    edge1 = self.operator.edges[edge_j]
                    displacement_i = self.direction[body_i]
                    displacement_j = self.direction[body_j]
                    alpha = 1.0
                    if ti.static(accd):
                        alpha = edge_edge_accd(
                            self._surface_vertex(edge0[0], self.translation),
                            self._surface_vertex(edge0[1], self.translation),
                            self._surface_vertex(edge1[0], self.translation),
                            self._surface_vertex(edge1[1], self.translation),
                            displacement_i,
                            displacement_i,
                            displacement_j,
                            displacement_j,
                            eta,
                            thickness,
                            maximum_iterations,
                        )
                    else:
                        alpha = edge_edge_ccd(
                            self._surface_vertex(edge0[0], self.translation),
                            self._surface_vertex(edge0[1], self.translation),
                            self._surface_vertex(edge1[0], self.translation),
                            self._surface_vertex(edge1[1], self.translation),
                            displacement_i,
                            displacement_i,
                            displacement_j,
                            displacement_j,
                            eta,
                            maximum_iterations,
                        )
                    ti.atomic_min(
                        self.operator.ccd_alpha[None],
                        ti.max(0.0, ti.min(1.0, alpha)),
                    )

    def _maximum_feasible_step(self):
        if self._ccd_mode in ("none", "off"):
            return 1.0
        self._prepare_mesh_sweep()
        self.operator.neighbor.update(
            self.ccd_positions,
            self.ccd_directions,
            self.operator.faces,
            self.operator.edges,
            self.operator.node2body,
            self.operator.face2body,
            self.operator.edge2body,
            self._ccd_thickness,
            swept=True,
        )
        self._compute_mesh_ccd_step(
            float(self._ccd_eta),
            float(self._ccd_thickness),
            int(self._ccd_maximum_iterations),
            self._ccd_mode == "accd",
            self.operator.neighbor.candidate_count,
            self.operator.neighbor.candidate_vertex,
            self.operator.neighbor.candidate_face,
            self.operator.neighbor.edge_candidate_count,
            self.operator.neighbor.candidate_edge0,
            self.operator.neighbor.candidate_edge1,
        )
        return float(self.operator.ccd_alpha[None])

    def solve(
        self,
        *,
        ccd_type="accd",
        ccd_eta=0.1,
        accd_tolerance=1.0e-9,
        ccd_maximum_iterations=50,
        **kwargs,
    ):
        if self.last_untangle_result is None:
            raise RuntimeError("TriangleMesh DiffIPC requires untangle() before strict IPC")
        if not self.last_untangle_result["success"]:
            raise RuntimeError("TriangleMesh DiffIPC cannot start while mesh overlaps remain")
        mode, eta, thickness = ccd_mode_parameters(ccd_type, ccd_eta, accd_tolerance)
        if mode not in ("ccd", "accd", "none", "off"):
            raise ValueError("TriangleMesh DiffIPC CCDType must be ccd or accd")
        self._ccd_mode = mode
        self._ccd_eta = eta
        self._ccd_thickness = thickness
        self._ccd_maximum_iterations = int(ccd_maximum_iterations)
        result = super().solve(**kwargs)
        result["untangling"] = dict(self.last_untangle_result)
        result["ccd_type"] = mode
        result["ccd_step"] = float(self.operator.ccd_alpha[None])
        return result


__all__ = ["TaichiAffineDiffIPCProjector", "TaichiAffineMeshDiffIPCProjector"]

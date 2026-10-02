"""Frictionless DiffIPC reference operators for affine-body contact.

This module is intentionally host-side.  It supplies an exact second-order
oracle for triangle-mesh point/triangle contact and a representation-agnostic
implicit-equilibrium adjoint.  Production Taichi kernels are tested against
these formulas; differentiable optimization can reuse the same 12-DOF affine
pullback without recording Newton iterations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.physics_model.contact_model.ipc.IPC import (
    ipc_barrier_distance_terms_py,
)
from src.physics_model.contact_model.ipc.LevelSetAffine import (
    LevelSetIPCAssembly,
    _as_float_array,
    _as_positive_scalar,
    _project_psd,
    continued_ipc_barrier_distance_terms,
)


@dataclass
class AffinePrimitiveGap:
    distance: float
    gradient: np.ndarray
    hessian: np.ndarray
    feature: str


class _SecondOrder:
    """Scalar value carrying a dense gradient and Hessian."""

    __slots__ = ("value", "gradient", "hessian")

    def __init__(self, value, gradient, hessian):
        self.value = float(value)
        self.gradient = gradient
        self.hessian = hessian

    @classmethod
    def constant(cls, value, dof):
        return cls(
            value,
            np.zeros(dof, dtype=np.float64),
            np.zeros((dof, dof), dtype=np.float64),
        )

    @classmethod
    def variable(cls, value, index, dof):
        gradient = np.zeros(dof, dtype=np.float64)
        gradient[index] = 1.0
        return cls(value, gradient, np.zeros((dof, dof), dtype=np.float64))

    def _coerce(self, other):
        if isinstance(other, _SecondOrder):
            return other
        return _SecondOrder.constant(other, self.gradient.size)

    def __add__(self, other):
        other = self._coerce(other)
        return _SecondOrder(
            self.value + other.value,
            self.gradient + other.gradient,
            self.hessian + other.hessian,
        )

    __radd__ = __add__

    def __neg__(self):
        return _SecondOrder(-self.value, -self.gradient, -self.hessian)

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        other = self._coerce(other)
        return _SecondOrder(
            self.value * other.value,
            self.gradient * other.value + other.gradient * self.value,
            self.hessian * other.value
            + other.hessian * self.value
            + np.outer(self.gradient, other.gradient)
            + np.outer(other.gradient, self.gradient),
        )

    __rmul__ = __mul__

    def reciprocal(self):
        if abs(self.value) <= 1.0e-30:
            raise ZeroDivisionError("second-order reciprocal is singular")
        first = -1.0 / (self.value * self.value)
        second = 2.0 / (self.value**3)
        return _SecondOrder(
            1.0 / self.value,
            first * self.gradient,
            first * self.hessian + second * np.outer(self.gradient, self.gradient),
        )

    def __truediv__(self, other):
        return self * self._coerce(other).reciprocal()

    def __rtruediv__(self, other):
        return self._coerce(other) * self.reciprocal()

    def sqrt(self):
        if self.value <= 0.0:
            raise ValueError("distance derivative is singular at zero")
        root = np.sqrt(self.value)
        first = 0.5 / root
        second = -0.25 / (self.value * root)
        return _SecondOrder(
            root,
            first * self.gradient,
            first * self.hessian + second * np.outer(self.gradient, self.gradient),
        )


def _dual_dot(left, right):
    return sum((a * b for a, b in zip(left, right)), 0.0)


def _dual_cross(left, right):
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def _subtract(left, right):
    return [a - b for a, b in zip(left, right)]


def _closest_point_triangle_feature(point, vertex0, vertex1, vertex2):
    """Return Ericson's closest feature, selected from primal values only."""
    p = np.asarray(point, dtype=np.float64)
    a = np.asarray(vertex0, dtype=np.float64)
    b = np.asarray(vertex1, dtype=np.float64)
    c = np.asarray(vertex2, dtype=np.float64)
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(ab @ ap)
    d2 = float(ac @ ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return "vertex0"
    bp = p - b
    d3 = float(ab @ bp)
    d4 = float(ac @ bp)
    if d3 >= 0.0 and d4 <= d3:
        return "vertex1"
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return "edge01"
    cp = p - c
    d5 = float(ab @ cp)
    d6 = float(ac @ cp)
    if d6 >= 0.0 and d5 <= d6:
        return "vertex2"
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return "edge20"
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        return "edge12"
    return "face"


def _dual_point_line_distance2(point, vertex0, vertex1):
    edge = _subtract(vertex1, vertex0)
    delta = _subtract(point, vertex0)
    parameter = _dual_dot(delta, edge) / _dual_dot(edge, edge)
    closest = [vertex0[d] + parameter * edge[d] for d in range(3)]
    residual = _subtract(point, closest)
    return _dual_dot(residual, residual)


def _dual_line_line_distance2(vertex00, vertex01, vertex10, vertex11):
    direction0 = _subtract(vertex01, vertex00)
    direction1 = _subtract(vertex11, vertex10)
    offset = _subtract(vertex00, vertex10)
    a = _dual_dot(direction0, direction0)
    b = _dual_dot(direction0, direction1)
    c = _dual_dot(direction0, offset)
    e = _dual_dot(direction1, direction1)
    f = _dual_dot(direction1, offset)
    denominator = a * e - b * b
    parameter0 = (b * f - c * e) / denominator
    parameter1 = (a * f - b * c) / denominator
    closest0 = [vertex00[d] + parameter0 * direction0[d] for d in range(3)]
    closest1 = [vertex10[d] + parameter1 * direction1[d] for d in range(3)]
    residual = _subtract(closest0, closest1)
    return _dual_dot(residual, residual)


def _closest_segment_parameters(vertex00, vertex01, vertex10, vertex11):
    """Return closest parameters on two primal line segments."""
    p0 = np.asarray(vertex00, dtype=np.float64)
    p1 = np.asarray(vertex01, dtype=np.float64)
    q0 = np.asarray(vertex10, dtype=np.float64)
    q1 = np.asarray(vertex11, dtype=np.float64)
    direction0 = p1 - p0
    direction1 = q1 - q0
    offset = p0 - q0
    a = float(direction0 @ direction0)
    e = float(direction1 @ direction1)
    if a <= 1.0e-30 or e <= 1.0e-30:
        raise ValueError("edge-edge derivative requires nondegenerate edges")
    b = float(direction0 @ direction1)
    c = float(direction0 @ offset)
    f = float(direction1 @ offset)
    denominator = a * e - b * b
    if denominator > 1.0e-14 * a * e:
        parameter0 = np.clip((b * f - c * e) / denominator, 0.0, 1.0)
    else:
        parameter0 = 0.0
    parameter1 = (b * parameter0 + f) / e
    if parameter1 < 0.0:
        parameter1 = 0.0
        parameter0 = np.clip(-c / a, 0.0, 1.0)
    elif parameter1 > 1.0:
        parameter1 = 1.0
        parameter0 = np.clip((b - c) / a, 0.0, 1.0)
    return float(parameter0), float(parameter1), float(denominator), a * e


def affine_edge_edge_gap(
    controls0,
    controls1,
    edge0_basis,
    edge1_basis,
):
    """Exact fixed-feature EE distance derivatives in 24 affine DOFs.

    The primal closest-point parameters select endpoint/edge/interior-edge
    features.  That discrete feature is held fixed during differentiation,
    matching the piecewise-smooth primitive calculus used by mesh IPC.  The
    parallel-edge degeneracy remains the responsibility of IPC's edge-edge
    mollifier and is deliberately not hidden by this oracle.
    """
    controls0 = _as_float_array(controls0, (4, 3), "controls0")
    controls1 = _as_float_array(controls1, (4, 3), "controls1")
    edge0_basis = _as_float_array(edge0_basis, (2, 4), "edge0_basis")
    edge1_basis = _as_float_array(edge1_basis, (2, 4), "edge1_basis")
    packed = np.concatenate((controls0.reshape(-1), controls1.reshape(-1)))
    variables = [_SecondOrder.variable(value, index, 24) for index, value in enumerate(packed)]

    def endpoint(body_offset, basis):
        return [
            sum(basis[control] * variables[body_offset + 3 * control + component] for control in range(4))
            for component in range(3)
        ]

    edge0 = [endpoint(0, edge0_basis[vertex]) for vertex in range(2)]
    edge1 = [endpoint(12, edge1_basis[vertex]) for vertex in range(2)]
    edge0_value = np.asarray([[entry.value for entry in vertex] for vertex in edge0])
    edge1_value = np.asarray([[entry.value for entry in vertex] for vertex in edge1])
    parameter0, parameter1, denominator, scale = _closest_segment_parameters(
        edge0_value[0], edge0_value[1], edge1_value[0], edge1_value[1]
    )
    tolerance = 1.0e-10
    interior0 = tolerance < parameter0 < 1.0 - tolerance
    interior1 = tolerance < parameter1 < 1.0 - tolerance
    endpoint0 = 0 if parameter0 <= 0.5 else 1
    endpoint1 = 0 if parameter1 <= 0.5 else 1
    if interior0 and interior1:
        if denominator <= 1.0e-14 * scale:
            raise ValueError("parallel interior edges require IPC's edge-edge mollifier")
        squared = _dual_line_line_distance2(edge0[0], edge0[1], edge1[0], edge1[1])
        feature = "edge_edge"
    elif interior0:
        squared = _dual_point_line_distance2(edge1[endpoint1], edge0[0], edge0[1])
        feature = f"edge0_vertex{endpoint1}"
    elif interior1:
        squared = _dual_point_line_distance2(edge0[endpoint0], edge1[0], edge1[1])
        feature = f"vertex{endpoint0}_edge1"
    else:
        residual = _subtract(edge0[endpoint0], edge1[endpoint1])
        squared = _dual_dot(residual, residual)
        feature = f"vertex{endpoint0}_vertex{endpoint1}"
    distance = squared.sqrt()
    return AffinePrimitiveGap(
        distance=distance.value,
        gradient=np.ascontiguousarray(distance.gradient),
        hessian=np.ascontiguousarray(0.5 * (distance.hessian + distance.hessian.T)),
        feature=feature,
    )


def affine_point_triangle_gap(
    source_controls,
    target_controls,
    source_basis,
    target_triangle_basis,
):
    """Exact fixed-feature PT distance derivatives in 24 affine DOFs.

    The closest feature is selected from the primal geometry and held fixed
    while differentiating, matching IPC's piecewise-smooth primitive
    distance.  Calls at feature transitions or zero distance are deliberately
    rejected because neither standard IPC nor DiffIPC is differentiable
    there.
    """
    source_controls = _as_float_array(source_controls, (4, 3), "source_controls")
    target_controls = _as_float_array(target_controls, (4, 3), "target_controls")
    source_basis = _as_float_array(source_basis, (4,), "source_basis")
    target_triangle_basis = _as_float_array(target_triangle_basis, (3, 4), "target_triangle_basis")
    packed = np.concatenate((source_controls.reshape(-1), target_controls.reshape(-1)))
    variables = [_SecondOrder.variable(value, index, 24) for index, value in enumerate(packed)]
    source = [
        sum(source_basis[control] * variables[3 * control + component] for control in range(4))
        for component in range(3)
    ]
    triangle = [
        [
            sum(
                target_triangle_basis[vertex, control] * variables[12 + 3 * control + component] for control in range(4)
            )
            for component in range(3)
        ]
        for vertex in range(3)
    ]
    source_value = np.array([entry.value for entry in source])
    triangle_value = np.array([[entry.value for entry in vertex] for vertex in triangle])
    feature = _closest_point_triangle_feature(source_value, triangle_value[0], triangle_value[1], triangle_value[2])
    if feature == "vertex0":
        residual = _subtract(source, triangle[0])
        squared = _dual_dot(residual, residual)
    elif feature == "vertex1":
        residual = _subtract(source, triangle[1])
        squared = _dual_dot(residual, residual)
    elif feature == "vertex2":
        residual = _subtract(source, triangle[2])
        squared = _dual_dot(residual, residual)
    elif feature == "edge01":
        squared = _dual_point_line_distance2(source, triangle[0], triangle[1])
    elif feature == "edge12":
        squared = _dual_point_line_distance2(source, triangle[1], triangle[2])
    elif feature == "edge20":
        squared = _dual_point_line_distance2(source, triangle[2], triangle[0])
    else:
        edge0 = _subtract(triangle[1], triangle[0])
        edge1 = _subtract(triangle[2], triangle[0])
        normal = _dual_cross(edge0, edge1)
        delta = _subtract(source, triangle[0])
        signed_numerator = _dual_dot(delta, normal)
        squared = signed_numerator * signed_numerator / _dual_dot(normal, normal)
    distance = squared.sqrt()
    return AffinePrimitiveGap(
        distance=distance.value,
        gradient=np.ascontiguousarray(distance.gradient),
        hessian=np.ascontiguousarray(0.5 * (distance.hessian + distance.hessian.T)),
        feature=feature,
    )


@dataclass
class AffineTriangleBody:
    controls: np.ndarray
    surface_basis: np.ndarray
    surface_weight: np.ndarray
    faces: np.ndarray

    def __post_init__(self):
        self.controls = _as_float_array(self.controls, (4, 3), "controls")
        basis = np.asarray(self.surface_basis, dtype=np.float64)
        if basis.ndim != 2 or basis.shape[1] != 4:
            raise ValueError("surface_basis must have shape (vertex_count, 4)")
        weights = np.asarray(self.surface_weight, dtype=np.float64).reshape(-1)
        if weights.size != basis.shape[0] or np.any(weights <= 0.0):
            raise ValueError("surface_weight must be positive per vertex")
        faces = np.asarray(self.faces, dtype=np.int32)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (face_count, 3)")
        if np.any(faces < 0) or np.any(faces >= basis.shape[0]):
            raise ValueError("faces contain an invalid surface vertex")
        self.surface_basis = np.ascontiguousarray(basis)
        self.surface_weight = np.ascontiguousarray(weights)
        self.faces = np.ascontiguousarray(faces)


class AffineTriangleIPC:
    """Host DiffIPC PT contact oracle pulled to AffineBody controls."""

    def __init__(
        self,
        bodies,
        *,
        dhat,
        kappa,
        time_scale=1.0,
        hessian_mode="exact",
    ):
        self.bodies = list(bodies)
        if not self.bodies:
            raise ValueError("at least one triangle affine body is required")
        if not all(isinstance(body, AffineTriangleBody) for body in self.bodies):
            raise TypeError("bodies must contain AffineTriangleBody values")
        self.dhat = _as_positive_scalar(dhat, "dhat")
        self.kappa = _as_positive_scalar(kappa, "kappa")
        self.time_scale = _as_positive_scalar(time_scale, "time_scale")
        mode = str(hessian_mode).strip().replace("-", "_").lower()
        aliases = {
            "exact": "exact",
            "projected": "projected",
            "psd": "projected",
            "gauss_newton": "gauss_newton",
        }
        if mode not in aliases:
            raise ValueError("invalid hessian_mode")
        self.hessian_mode = aliases[mode]
        self.body_num = len(self.bodies)
        self.dof = 12 * self.body_num

    def pack_controls(self):
        return np.concatenate([body.controls.reshape(-1) for body in self.bodies])

    def set_controls(self, packed):
        packed = _as_float_array(packed, (self.dof,), "packed controls")
        for body_id, body in enumerate(self.bodies):
            body.controls = packed[12 * body_id : 12 * (body_id + 1)].reshape(4, 3).copy()

    def assemble(
        self,
        controls=None,
        *,
        need_hessian=True,
        continued=False,
        continuation_distance=None,
        kappa=None,
    ):
        if controls is not None:
            self.set_controls(controls)
        barrier_kappa = self.kappa if kappa is None else _as_positive_scalar(kappa, "kappa")
        energy = 0.0
        gradient = np.zeros(self.dof, dtype=np.float64)
        hessian = np.zeros((self.dof, self.dof), dtype=np.float64)
        minimum_gap = np.inf
        active_contacts = 0
        sampled_contacts = 0
        for source_id, source in enumerate(self.bodies):
            for target_id, target in enumerate(self.bodies):
                if source_id == target_id:
                    continue
                indices = np.concatenate(
                    (
                        np.arange(12 * source_id, 12 * (source_id + 1)),
                        np.arange(12 * target_id, 12 * (target_id + 1)),
                    )
                )
                for basis, weight in zip(source.surface_basis, source.surface_weight):
                    for face in target.faces:
                        gap = affine_point_triangle_gap(
                            source.controls,
                            target.controls,
                            basis,
                            target.surface_basis[face],
                        )
                        sampled_contacts += 1
                        minimum_gap = min(minimum_gap, gap.distance)
                        if continued:
                            value, first, second = continued_ipc_barrier_distance_terms(
                                gap.distance,
                                self.dhat,
                                kappa=barrier_kappa,
                                continuation_distance=continuation_distance,
                            )
                        else:
                            value, first, second = ipc_barrier_distance_terms_py(
                                gap.distance, self.dhat, kappa=barrier_kappa
                            )
                        if not np.isfinite(value):
                            return LevelSetIPCAssembly(
                                np.inf,
                                gradient,
                                hessian,
                                minimum_gap,
                                active_contacts,
                                sampled_contacts,
                                False,
                            )
                        if value == 0.0 and first == 0.0 and second == 0.0:
                            continue
                        active_contacts += 1
                        coefficient = 0.5 * self.time_scale * float(weight)
                        exact = second * np.outer(gap.gradient, gap.gradient) + first * gap.hessian
                        if self.hessian_mode == "projected":
                            local_hessian = _project_psd(exact)
                        elif self.hessian_mode == "gauss_newton":
                            local_hessian = second * np.outer(gap.gradient, gap.gradient)
                        else:
                            local_hessian = exact
                        energy += coefficient * value
                        gradient[indices] += coefficient * first * gap.gradient
                        if need_hessian:
                            hessian[np.ix_(indices, indices)] += coefficient * local_hessian
        return LevelSetIPCAssembly(
            float(energy),
            gradient,
            0.5 * (hessian + hessian.T),
            float(minimum_gap),
            active_contacts,
            sampled_contacts,
            bool(minimum_gap > 0.0),
        )


@dataclass
class AffineEquilibriumResult:
    controls: np.ndarray
    objective: float
    residual_norm: float
    iterations: int
    success: bool
    message: str


class DiffIPCAffineEquilibrium:
    """Implicit-function adjoint for a frictionless affine IPC equilibrium.

    For either triangle-mesh or level-set contact, the inner equation is

    ``R(y,a)=W(y-a)+grad(E_contact(y))=0``.

    DiffIPC then solves ``R_y.T lambda = dL/dy`` and returns
    ``dL/da = partial_a L + W.T lambda``.  No differentiation through Newton
    iterations is performed.
    """

    def __init__(self, contact, *, anchor_stiffness=1.0):
        required = ("dof", "pack_controls", "set_controls", "assemble")
        if not all(hasattr(contact, name) for name in required):
            raise TypeError("contact does not implement the affine IPC interface")
        self.contact = contact
        self.dof = int(contact.dof)
        stiffness = np.asarray(anchor_stiffness, dtype=np.float64)
        if stiffness.ndim == 0:
            value = _as_positive_scalar(stiffness, "anchor_stiffness")
            self.anchor_matrix = value * np.eye(self.dof)
        else:
            self.anchor_matrix = _as_float_array(stiffness, (self.dof, self.dof), "anchor_stiffness")
            eigenvalues = np.linalg.eigvalsh(0.5 * (self.anchor_matrix + self.anchor_matrix.T))
            if eigenvalues.min() <= 0.0:
                raise ValueError("anchor_stiffness must be positive definite")
        self.initial_controls = contact.pack_controls().copy()
        self.anchor = self.initial_controls.copy()
        self.residual_jacobian = None
        self.last_result = None

    def solve(self, *, anchor=None, maximum_iterations=300):
        # Explicit host-side optimization/adjoint oracle; never imported by
        # the production Taichi solve path.
        from scipy.optimize import minimize

        if anchor is not None:
            self.anchor = _as_float_array(anchor, (self.dof,), "anchor")
        initial = self.initial_controls.copy()

        def objective(controls):
            assembly = self.contact.assemble(controls, need_hessian=False)
            displacement = controls - self.anchor
            value = 0.5 * float(displacement @ self.anchor_matrix @ displacement) + assembly.energy
            gradient = self.anchor_matrix @ displacement + assembly.gradient
            return value, gradient

        optimum = minimize(
            objective,
            initial,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": int(maximum_iterations),
                "ftol": 1.0e-14,
                "gtol": 1.0e-11,
                "maxls": 80,
            },
        )
        controls = np.asarray(optimum.x, dtype=np.float64)
        # A projected Hessian is useful only as a nonlinear search direction.
        # The implicit-function Jacobian must differentiate the exact residual,
        # regardless of the contact object's forward globalization mode.
        hessian_mode = getattr(self.contact, "hessian_mode", None)
        if hessian_mode is not None:
            self.contact.hessian_mode = "exact"
        try:
            assembly = self.contact.assemble(controls, need_hessian=True)
        finally:
            if hessian_mode is not None:
                self.contact.hessian_mode = hessian_mode
        residual = self.anchor_matrix @ (controls - self.anchor) + assembly.gradient
        self.residual_jacobian = self.anchor_matrix + assembly.hessian
        self.contact.set_controls(controls)
        residual_norm = float(np.linalg.norm(residual))
        success = bool(np.isfinite(optimum.fun) and assembly.feasible)
        self.last_result = AffineEquilibriumResult(
            controls=controls.reshape(-1, 4, 3),
            objective=float(optimum.fun),
            residual_norm=residual_norm,
            iterations=int(getattr(optimum, "nit", 0)),
            success=success,
            message=str(getattr(optimum, "message", "")),
        )
        return self.last_result

    def adjoint_anchor_gradient(self, loss_control_gradient, *, direct_anchor_gradient=None):
        if self.residual_jacobian is None or self.last_result is None:
            raise RuntimeError("solve() must be called before the adjoint")
        loss_gradient = _as_float_array(loss_control_gradient, (self.dof,), "loss_control_gradient")
        direct = (
            np.zeros(self.dof, dtype=np.float64)
            if direct_anchor_gradient is None
            else _as_float_array(
                direct_anchor_gradient,
                (self.dof,),
                "direct_anchor_gradient",
            )
        )
        adjoint = np.linalg.solve(self.residual_jacobian.T, loss_gradient)
        return direct + self.anchor_matrix.T @ adjoint


__all__ = [
    "AffineEquilibriumResult",
    "AffinePrimitiveGap",
    "AffineTriangleBody",
    "AffineTriangleIPC",
    "DiffIPCAffineEquilibrium",
    "affine_edge_edge_gap",
    "affine_point_triangle_gap",
]

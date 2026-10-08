"""Geometry utilities shared by IPC NURBS contact implementations.

This module owns the geometry-only pieces historically implemented under
``src.igampm.contact``: CPU curve closest-point queries, Taichi curve-distance
queries, and point--NURBS distance derivatives.
It deliberately does not own contact search, quadrature measures, constitutive
barrier/friction laws, or solver-specific matrix scattering.

The Taichi functions infer their spatial dimension from vector or matrix
arguments wherever possible, so the same implementation can be used by IGA,
MPM, and future FEM contact code in both two and three dimensions.
The moving-query variants evaluate linear virtual control-point trajectories
without writing shared geometry, which lets each ACCD contact pair own its
complete device-side iteration.
"""

import numpy as np
import taichi as ti

from .ContactAssembly import psd_project_nd, symmetric_eigendecomposition_nd
from src.utils.MatrixFunction import inverse_matrix_2x2
from src.utils.ScalarFunction import clamp
from src.utils.TypeDefination import mat2x2
from src.utils.constants import DBL_EPSILON

CURVE_DISTANCE_TOL = 100 * DBL_EPSILON
CLOSEST_POINT_PARAMETER_TOL = 1.0e-12
CLOSEST_POINT_STATIONARITY_TOL = 1.0e-10
CLOSEST_POINT_MAX_ITERATIONS = 64
CLOSEST_POINT_MAX_BACKTRACKS = 12


def nurbs_basis_py(degree, knots, parameter):
    """Evaluate the non-rational B-spline basis active at ``parameter``."""
    knots = np.asarray(knots, dtype=np.float64)
    degree = int(degree)
    num_ctrlpts = len(knots) - degree - 1
    if parameter >= knots[num_ctrlpts]:
        span = num_ctrlpts - 1
    elif parameter <= knots[degree]:
        span = degree
    else:
        low = degree
        high = num_ctrlpts
        span = (low + high) // 2
        while parameter < knots[span] or parameter >= knots[span + 1]:
            if parameter < knots[span]:
                high = span
            else:
                low = span
            span = (low + high) // 2

    basis = np.zeros(degree + 1, dtype=np.float64)
    left = np.zeros(degree + 1, dtype=np.float64)
    right = np.zeros(degree + 1, dtype=np.float64)
    basis[0] = 1.0
    for j in range(1, degree + 1):
        left[j] = parameter - knots[span + 1 - j]
        right[j] = knots[span + j] - parameter
        saved = 0.0
        for r in range(j):
            denominator = right[r + 1] + left[j - r]
            temp = 0.0 if denominator == 0.0 else basis[r] / denominator
            basis[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        basis[j] = saved
    return span, basis


def eval_curve_py(degree, knots, ctrlpts, weights, parameter):
    """Evaluate a rational NURBS curve on the CPU."""
    ctrlpts = np.asarray(ctrlpts, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    span, basis = nurbs_basis_py(degree, knots, parameter)
    numerator = np.zeros(ctrlpts.shape[1], dtype=np.float64)
    denominator = 0.0
    first = span - int(degree)
    for local, shape in enumerate(basis):
        ctrlpt_id = first + local
        weighted_shape = shape * weights[ctrlpt_id]
        numerator += weighted_shape * ctrlpts[ctrlpt_id]
        denominator += weighted_shape
    return numerator / denominator


def closest_curve_point_py(degree, knots, ctrlpts, weights, point):
    """Return the globally best sampled curve point on the CPU.

    Every non-empty knot span is sampled and every sampled local minimum is
    refined with a bounded golden-section search.  The two domain endpoints
    are compared explicitly; this is important because an open NURBS endpoint
    is a valid closest feature and an interior-only golden search merely
    approaches it.
    """
    knots = np.asarray(knots, dtype=np.float64)
    ctrlpts = np.asarray(ctrlpts, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    degree = int(degree)
    num_ctrlpts = len(knots) - degree - 1
    if degree < 1 or num_ctrlpts <= degree:
        raise ValueError("a contact curve requires degree >= 1 and enough control points")
    if knots.ndim != 1 or ctrlpts.ndim != 2 or weights.ndim != 1:
        raise ValueError("invalid NURBS curve array dimensions")
    if ctrlpts.shape[0] != num_ctrlpts or weights.size != num_ctrlpts:
        raise ValueError("NURBS knots, control points, and weights are inconsistent")
    if point.shape != (ctrlpts.shape[1],):
        raise ValueError("query point dimension does not match the NURBS curve")
    if (
        not np.all(np.isfinite(knots))
        or not np.all(np.isfinite(ctrlpts))
        or not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(point))
    ):
        raise ValueError("NURBS closest-point input must be finite")
    if np.any(weights <= 0.0):
        raise ValueError("NURBS contact weights must be positive")
    if np.any(np.diff(knots) < 0.0):
        raise ValueError("NURBS knot vector must be nondecreasing")
    lower = float(knots[degree])
    upper = float(knots[num_ctrlpts])
    if not upper > lower:
        raise ValueError("NURBS contact parameter domain must have positive length")

    def objective(parameter):
        residual = eval_curve_py(degree, knots, ctrlpts, weights, float(parameter)) - point
        value = float(np.dot(residual, residual))
        if not np.isfinite(value):
            raise RuntimeError("NURBS curve evaluation produced a non-finite distance")
        return value

    # Build a global sample set span-by-span so narrow knot spans are not lost
    # in a uniform parameter grid.
    samples = []
    subdivisions = max(8, 4 * (degree + 1))
    for span in range(degree, num_ctrlpts):
        left = float(knots[span])
        right = float(knots[span + 1])
        if right > left:
            samples.extend(np.linspace(left, right, subdivisions + 1, endpoint=True).tolist())
    samples = np.unique(np.asarray(samples, dtype=np.float64))
    if samples.size < 2:
        raise RuntimeError("NURBS curve has no non-empty knot span")
    values = np.asarray([objective(parameter) for parameter in samples])

    # Endpoints are inserted first and ties never replace an earlier candidate,
    # so an endpoint minimum is returned as the exact endpoint.
    candidates = [
        (lower, objective(lower)),
        (upper, objective(upper)),
    ]
    inverse_golden_ratio = 0.6180339887498948
    for index, value in enumerate(values):
        left_value = values[index - 1] if index > 0 else np.inf
        right_value = values[index + 1] if index + 1 < values.size else np.inf
        if value > left_value or value > right_value:
            continue
        left = float(samples[max(0, index - 1)])
        right = float(samples[min(samples.size - 1, index + 1)])
        if not right > left:
            candidates.append((float(samples[index]), float(value)))
            continue
        x1 = right - inverse_golden_ratio * (right - left)
        x2 = left + inverse_golden_ratio * (right - left)
        f1 = objective(x1)
        f2 = objective(x2)
        for _ in range(32):
            if f1 > f2:
                left = x1
                x1 = x2
                f1 = f2
                x2 = left + inverse_golden_ratio * (right - left)
                f2 = objective(x2)
            else:
                right = x2
                x2 = x1
                f2 = f1
                x1 = right - inverse_golden_ratio * (right - left)
                f1 = objective(x1)
        parameter = 0.5 * (left + right)
        candidates.append((parameter, objective(parameter)))

    parameter, distance2 = candidates[0]
    for candidate_parameter, candidate_distance2 in candidates[1:]:
        if candidate_distance2 < distance2:
            parameter = candidate_parameter
            distance2 = candidate_distance2
    return float(parameter), float(np.sqrt(max(distance2, 0.0)))


@ti.func
def squared_norm_nd(vector):
    """Dimension-generic squared Euclidean norm."""
    value = 0.0
    for d in ti.static(range(vector.n)):
        value += vector[d] * vector[d]
    return value


@ti.func
def _isfinite_scalar(value):
    return (value == value) and ti.abs(value) < 1.0e300


@ti.func
def _isfinite_vector(vector):
    result = 1
    for d in ti.static(range(vector.n)):
        if not _isfinite_scalar(vector[d]):
            result = 0
    return result


@ti.func
def _curve_closest_state(
    start_knot,
    start_ctrlpt,
    num_knot,
    knot_vector_u,
    ctrlpts,
    control_directions,
    alpha,
    weight,
    point,
    parameter,
    basis,
):
    position, tangent, curvature = basis.NurbsBasisMovingInterpolations2ndDers1d(
        start_knot,
        start_ctrlpt,
        num_knot,
        parameter,
        knot_vector_u,
        ctrlpts,
        control_directions,
        alpha,
        weight,
    )
    residual = position - point
    distance2 = squared_norm_nd(residual)
    valid = (
        _isfinite_vector(position)
        and _isfinite_vector(tangent)
        and _isfinite_vector(curvature)
        and _isfinite_vector(residual)
        and _isfinite_scalar(distance2)
    )
    return residual, tangent, curvature, distance2, valid


@ti.func
def _projected_gradient_1d(parameter, lower, upper, gradient):
    tolerance = CLOSEST_POINT_PARAMETER_TOL * (1.0 + ti.abs(lower) + ti.abs(upper))
    projected = gradient
    if parameter <= lower + tolerance:
        projected = ti.min(gradient, 0.0)
    elif parameter >= upper - tolerance:
        projected = ti.max(gradient, 0.0)
    return projected


@ti.func
def _curve_projected_newton_candidate(
    start_knot,
    start_ctrlpt,
    num_knot,
    knot_vector_u,
    ctrlpts,
    control_directions,
    alpha,
    weight,
    point,
    seed,
    lower,
    upper,
    basis,
):
    # As in the surface search, accept a trial and reuse its derivatives in
    # the same loop, with one geometric evaluation site for Newton/backtracking.
    parameter = clamp(lower, upper, seed)
    trial_parameter = parameter
    residual = ti.Vector.zero(ti.f64, point.n)
    distance2 = 1.0e300
    success = 0
    iteration = 0
    active = 1
    in_trial = 0
    backtrack = 0
    step = 0.0
    while active != 0:
        trial_residual, tangent, curvature, trial_distance2, valid = _curve_closest_state(
            start_knot,
            start_ctrlpt,
            num_knot,
            knot_vector_u,
            ctrlpts,
            control_directions,
            alpha,
            weight,
            point,
            trial_parameter,
            basis,
        )
        decrease_tolerance = 1.0e-14 * (1.0 + distance2)
        if valid != 0 and (in_trial == 0 or trial_distance2 <= distance2 + decrease_tolerance):
            parameter = trial_parameter
            residual, distance2 = trial_residual, trial_distance2
            iteration += in_trial
            gradient = residual.dot(tangent)
            projected_gradient = _projected_gradient_1d(parameter, lower, upper, gradient)
            tangent_norm = ti.sqrt(ti.max(squared_norm_nd(tangent), 0.0))
            stationarity_tolerance = CLOSEST_POINT_STATIONARITY_TOL * (
                1.0 + ti.sqrt(ti.max(distance2, 0.0)) * tangent_norm
            )
            if (
                distance2 <= CURVE_DISTANCE_TOL * CURVE_DISTANCE_TOL
                or ti.abs(projected_gradient) <= stationarity_tolerance
            ):
                success = 1
                active = 0
            elif iteration >= CLOSEST_POINT_MAX_ITERATIONS:
                active = 0
            else:
                hessian = squared_norm_nd(tangent) + residual.dot(curvature)
                width = upper - lower
                if _isfinite_scalar(hessian) and hessian > 1.0e-14:
                    step = -gradient / hessian
                else:
                    step = -0.25 * width * ti.math.sign(projected_gradient)
                if not _isfinite_scalar(step) or projected_gradient * step >= 0.0:
                    step = -0.25 * width * ti.math.sign(projected_gradient)
                backtrack = 0
                in_trial = 1
        elif in_trial == 0:
            residual, distance2 = trial_residual, trial_distance2
            active = 0
        else:
            step *= 0.5
            backtrack += 1
            if backtrack >= CLOSEST_POINT_MAX_BACKTRACKS:
                active = 0
        if active != 0 and in_trial != 0:
            trial_parameter = clamp(lower, upper, parameter + step)
    return parameter, distance2, residual, success


@ti.func
def get_distance_to_curve_moving_fixed_dim(
    start_knot,
    start_ctrlpt,
    num_knot,
    knot_vector_u,
    ctrlpts,
    control_directions,
    alpha,
    weight,
    point,
    basis,
):
    """Closest point on a NURBS curve using safeguarded span multistart."""
    degree = ti.static(basis.basis_u.degree)
    num_ctrlpts = num_knot - 1 - degree
    lower = knot_vector_u[start_knot + degree]
    upper = knot_vector_u[start_knot + num_ctrlpts]
    assert upper > lower, "NURBS curve has no non-empty parameter domain"

    best_parameter = lower
    best_distance2 = 1.0e300
    best_residual = ti.Vector.zero(ti.f64, point.n)
    found = 0

    # Preserve span-midpoint then Greville seed order, but share one solver
    # call site rather than inlining the full Newton search twice. The serial
    # loop also preserves the pair-local argmin when called at kernel scope.
    span_count = num_ctrlpts - degree
    candidate = 0
    while candidate < span_count + num_ctrlpts:
        candidate_lower, candidate_upper = lower, upper
        seed = 0.0
        if candidate < span_count:
            span = degree + candidate
            candidate_lower = knot_vector_u[start_knot + span]
            candidate_upper = knot_vector_u[start_knot + span + 1]
            seed = 0.5 * (candidate_lower + candidate_upper)
        else:
            control_point = candidate - span_count
            if ti.static(degree <= 3):
                for offset in ti.static(range(1, degree + 1)):
                    seed += knot_vector_u[start_knot + control_point + offset]
            else:
                offset = 1
                while offset <= degree:
                    seed += knot_vector_u[start_knot + control_point + offset]
                    offset += 1
            seed /= degree
        if candidate_upper > candidate_lower:
            parameter, distance2, residual, success = _curve_projected_newton_candidate(
                start_knot,
                start_ctrlpt,
                num_knot,
                knot_vector_u,
                ctrlpts,
                control_directions,
                alpha,
                weight,
                point,
                seed,
                candidate_lower,
                candidate_upper,
                basis,
            )
            if success != 0 and (found == 0 or distance2 < best_distance2):
                best_parameter = parameter
                best_distance2 = distance2
                best_residual = residual
                found = 1
        candidate += 1

    assert found != 0, "NURBS curve closest-point solve failed"
    return (
        clamp(lower, upper, best_parameter),
        ti.sqrt(ti.max(best_distance2, 0.0)),
        best_residual,
    )


@ti.func
def get_distance_to_curve_fixed_dim(
    start_knot,
    start_ctrlpt,
    num_knot,
    knot_vector_u,
    ctrlpts,
    weight,
    point,
    basis,
):
    """Closest point on a stationary NURBS curve."""
    return get_distance_to_curve_moving_fixed_dim(
        start_knot,
        start_ctrlpt,
        num_knot,
        knot_vector_u,
        ctrlpts,
        ctrlpts,
        0.0,
        weight,
        point,
        basis,
    )


@ti.func
def evaluate_distance_to_curve_fixed_dim(
    start_knot,
    start_ctrlpt,
    num_knot,
    knot_vector_u,
    ctrlpts,
    weight,
    point,
    parameter,
    basis,
):
    residual, _, _, distance2, valid = _curve_closest_state(
        start_knot,
        start_ctrlpt,
        num_knot,
        knot_vector_u,
        ctrlpts,
        ctrlpts,
        0.0,
        weight,
        point,
        parameter,
        basis,
    )
    assert valid != 0, "NURBS curve evaluation failed"
    return ti.sqrt(ti.max(distance2, 0.0)), residual


@ti.func
def _surface_closest_state(
    start_knot_u,
    start_knot_v,
    start_ctrlpt,
    num_knot_u,
    num_knot_v,
    knot_vector_u,
    knot_vector_v,
    ctrlpts,
    control_directions,
    alpha,
    weight,
    point,
    parameter_u,
    parameter_v,
    basis,
):
    (
        position,
        tangent_u,
        tangent_v,
        curvature_uu,
        curvature_vv,
        curvature_uv,
    ) = basis.NurbsBasisMovingInterpolations2ndDers2d(
        start_knot_u,
        start_knot_v,
        start_ctrlpt,
        num_knot_u,
        num_knot_v,
        parameter_u,
        parameter_v,
        knot_vector_u,
        knot_vector_v,
        ctrlpts,
        control_directions,
        alpha,
        weight,
    )
    residual = position - point
    distance2 = squared_norm_nd(residual)
    valid = (
        _isfinite_vector(position)
        and _isfinite_vector(tangent_u)
        and _isfinite_vector(tangent_v)
        and _isfinite_vector(curvature_uu)
        and _isfinite_vector(curvature_vv)
        and _isfinite_vector(curvature_uv)
        and _isfinite_vector(residual)
        and _isfinite_scalar(distance2)
    )
    return (
        residual,
        tangent_u,
        tangent_v,
        curvature_uu,
        curvature_vv,
        curvature_uv,
        distance2,
        valid,
    )


@ti.func
def _surface_projected_gradients(
    parameter_u,
    parameter_v,
    lower_u,
    upper_u,
    lower_v,
    upper_v,
    gradient_u,
    gradient_v,
):
    projected_u = gradient_u
    projected_v = gradient_v
    tolerance_u = CLOSEST_POINT_PARAMETER_TOL * (1.0 + ti.abs(lower_u) + ti.abs(upper_u))
    tolerance_v = CLOSEST_POINT_PARAMETER_TOL * (1.0 + ti.abs(lower_v) + ti.abs(upper_v))
    if parameter_u <= lower_u + tolerance_u:
        projected_u = ti.min(gradient_u, 0.0)
    elif parameter_u >= upper_u - tolerance_u:
        projected_u = ti.max(gradient_u, 0.0)
    if parameter_v <= lower_v + tolerance_v:
        projected_v = ti.min(gradient_v, 0.0)
    elif parameter_v >= upper_v - tolerance_v:
        projected_v = ti.max(gradient_v, 0.0)
    return projected_u, projected_v


@ti.func
def _surface_projected_newton_candidate(
    start_knot_u,
    start_knot_v,
    start_ctrlpt,
    num_knot_u,
    num_knot_v,
    knot_vector_u,
    knot_vector_v,
    ctrlpts,
    control_directions,
    alpha,
    weight,
    point,
    seed_u,
    seed_v,
    lower_u,
    upper_u,
    lower_v,
    upper_v,
    basis,
):
    # Evaluate each candidate once. Accepted trial derivatives become the next
    # Newton state; rejected trials only shrink the same direction. Keeping one
    # state-evaluation call site avoids duplicating large basis IR in nested
    # Newton/backtracking loops (pathological Taichi 1.7 CFG compilation).
    parameter_u = clamp(lower_u, upper_u, seed_u)
    parameter_v = clamp(lower_v, upper_v, seed_v)
    trial_u, trial_v = parameter_u, parameter_v
    residual = ti.Vector.zero(ti.f64, point.n)
    distance2 = 1.0e300
    success = 0
    iteration = 0
    active = 1
    in_trial = 0
    backtrack = 0
    delta_u, delta_v = 0.0, 0.0
    gradient_delta_u, gradient_delta_v = 0.0, 0.0
    using_projected_gradient = 0
    while active != 0:
        (
            trial_residual,
            tangent_u,
            tangent_v,
            curvature_uu,
            curvature_vv,
            curvature_uv,
            trial_distance2,
            valid,
        ) = _surface_closest_state(
            start_knot_u,
            start_knot_v,
            start_ctrlpt,
            num_knot_u,
            num_knot_v,
            knot_vector_u,
            knot_vector_v,
            ctrlpts,
            control_directions,
            alpha,
            weight,
            point,
            trial_u,
            trial_v,
            basis,
        )
        decrease_tolerance = 1.0e-14 * (1.0 + distance2)
        if valid != 0 and (in_trial == 0 or trial_distance2 <= distance2 + decrease_tolerance):
            parameter_u, parameter_v = trial_u, trial_v
            residual, distance2 = trial_residual, trial_distance2
            iteration += in_trial
            gradient_u = residual.dot(tangent_u)
            gradient_v = residual.dot(tangent_v)
            projected_u, projected_v = _surface_projected_gradients(
                parameter_u,
                parameter_v,
                lower_u,
                upper_u,
                lower_v,
                upper_v,
                gradient_u,
                gradient_v,
            )
            tangent_scale = ti.sqrt(ti.max(squared_norm_nd(tangent_u) + squared_norm_nd(tangent_v), 0.0))
            stationarity_tolerance = CLOSEST_POINT_STATIONARITY_TOL * (
                1.0 + ti.sqrt(ti.max(distance2, 0.0)) * tangent_scale
            )
            if (
                distance2 <= CURVE_DISTANCE_TOL * CURVE_DISTANCE_TOL
                or ti.sqrt(projected_u * projected_u + projected_v * projected_v) <= stationarity_tolerance
            ):
                success = 1
                active = 0
            elif iteration >= CLOSEST_POINT_MAX_ITERATIONS:
                active = 0
            else:
                tangent_u_norm2 = squared_norm_nd(tangent_u)
                tangent_v_norm2 = squared_norm_nd(tangent_v)
                hessian_uu = tangent_u_norm2 + residual.dot(curvature_uu)
                hessian_uv = tangent_u.dot(tangent_v) + residual.dot(curvature_uv)
                hessian_vv = tangent_v_norm2 + residual.dot(curvature_vv)
                determinant = hessian_uu * hessian_vv - hessian_uv * hessian_uv
                delta_u = 0.0
                delta_v = 0.0
                gradient_delta_u = -projected_u / ti.max(tangent_u_norm2, 1.0e-30)
                gradient_delta_v = -projected_v / ti.max(tangent_v_norm2, 1.0e-30)
                using_projected_gradient = 0
                newton_valid = (
                    _isfinite_scalar(hessian_uu)
                    and _isfinite_scalar(hessian_uv)
                    and _isfinite_scalar(hessian_vv)
                    and determinant > 1.0e-20
                    and hessian_uu > 1.0e-14
                )
                if newton_valid:
                    delta_u = (-hessian_vv * gradient_u + hessian_uv * gradient_v) / determinant
                    delta_v = (hessian_uv * gradient_u - hessian_uu * gradient_v) / determinant

                if (
                    not newton_valid
                    or not _isfinite_scalar(delta_u)
                    or not _isfinite_scalar(delta_v)
                    or gradient_u * delta_u + gradient_v * delta_v >= 0.0
                ):
                    delta_u = gradient_delta_u
                    delta_v = gradient_delta_v
                    using_projected_gradient = 1

                # A descent Newton step can become a zero or ascent step after box
                # projection (notably at surface edges/corners).  Do not accept that
                # unchanged point repeatedly; use a feasible projected-gradient step.
                projected_step_u = clamp(lower_u, upper_u, parameter_u + delta_u) - parameter_u
                projected_step_v = clamp(lower_v, upper_v, parameter_v + delta_v) - parameter_v
                if gradient_u * projected_step_u + gradient_v * projected_step_v >= 0.0:
                    delta_u = gradient_delta_u
                    delta_v = gradient_delta_v
                    using_projected_gradient = 1

                backtrack = 0
                in_trial = 1
        elif in_trial == 0:
            residual, distance2 = trial_residual, trial_distance2
            active = 0
        else:
            delta_u *= 0.5
            delta_v *= 0.5
            backtrack += 1
            if backtrack >= 2 * CLOSEST_POINT_MAX_BACKTRACKS:
                active = 0
        if active != 0 and in_trial != 0:
            if backtrack == CLOSEST_POINT_MAX_BACKTRACKS and using_projected_gradient == 0:
                delta_u = gradient_delta_u
                delta_v = gradient_delta_v
                using_projected_gradient = 1
            trial_u = clamp(lower_u, upper_u, parameter_u + delta_u)
            trial_v = clamp(lower_v, upper_v, parameter_v + delta_v)
    return parameter_u, parameter_v, distance2, residual, success


@ti.func
def get_distance_to_surface_moving_fixed_dim(
    start_knot_u,
    start_knot_v,
    start_ctrlpt,
    num_knot_u,
    num_knot_v,
    knot_vector_u,
    knot_vector_v,
    ctrlpts,
    control_directions,
    alpha,
    weight,
    point,
    basis,
    span_cache: ti.template() = None,
    surface_id=0,
    initial_parameter: ti.template() = None,
):
    """Knot-span multistart; optional stationary hulls and an extra seed."""
    if ti.static(not isinstance(span_cache, type(None))):
        assert alpha == 0.0, "cached NURBS span bounds require stationary geometry"
    degree_u = ti.static(basis.basis_u.degree)
    degree_v = ti.static(basis.basis_v.degree)
    num_ctrlpts_u = num_knot_u - degree_u - 1
    num_ctrlpts_v = num_knot_v - degree_v - 1
    lower_u = knot_vector_u[start_knot_u + degree_u]
    upper_u = knot_vector_u[start_knot_u + num_ctrlpts_u]
    lower_v = knot_vector_v[start_knot_v + degree_v]
    upper_v = knot_vector_v[start_knot_v + num_ctrlpts_v]
    assert upper_u > lower_u and upper_v > lower_v, "NURBS surface has no non-empty parameter domain"

    best_u = lower_u
    best_v = lower_v
    best_distance2 = 1.0e300
    best_residual = ti.Vector.zero(ti.f64, point.n)
    found = 0

    # Use the Greville abscissa of the geometrically nearest control point as
    # an additional global seed.  Keeping this seed in the same runtime loop
    # as the span seeds avoids a second Taichi call site (and a large duplicate
    # inlining cost) while still adding a basin independent of span midpoints.
    control_seed_u = lower_u
    control_seed_v = lower_v
    nearest_control_distance2 = 1.0e300
    control_seed_found = 0
    if ti.static(not isinstance(span_cache, type(None))):
        control_id = span_cache.nearest_control_point(surface_id, point)
        control_seed_u = span_cache.control_greville[control_id][0]
        control_seed_v = span_cache.control_greville[control_id][1]
        control_seed_found = 1
    else:
        # This argmin belongs to one point, including direct kernel-scope calls.
        control_v = 0
        while control_v < num_ctrlpts_v:
            greville_v = 0.0
            if ti.static(degree_v <= 3):
                for offset_v in ti.static(range(1, degree_v + 1)):
                    greville_v += knot_vector_v[start_knot_v + control_v + offset_v]
            else:
                offset_v = 1
                while offset_v <= degree_v:
                    greville_v += knot_vector_v[start_knot_v + control_v + offset_v]
                    offset_v += 1
            greville_v /= degree_v
            for control_u in range(num_ctrlpts_u):
                control_id = start_ctrlpt + control_v * num_ctrlpts_u + control_u
                assert weight[control_id] > 0.0, "NURBS contact weights must be positive"
                control_position = ctrlpts[control_id]
                if alpha != 0.0:
                    control_position += alpha * control_directions[control_id]
                control_residual = control_position - point
                control_distance2 = squared_norm_nd(control_residual)
                if (
                    _isfinite_vector(control_residual)
                    and _isfinite_scalar(control_distance2)
                    and (control_seed_found == 0 or control_distance2 < nearest_control_distance2)
                ):
                    greville_u = 0.0
                    if ti.static(degree_u <= 3):
                        for offset_u in ti.static(range(1, degree_u + 1)):
                            greville_u += knot_vector_u[start_knot_u + control_u + offset_u]
                    else:
                        offset_u = 1
                        while offset_u <= degree_u:
                            greville_u += knot_vector_u[start_knot_u + control_u + offset_u]
                            offset_u += 1
                    control_seed_u = greville_u / degree_u
                    control_seed_v = greville_v
                    nearest_control_distance2 = control_distance2
                    control_seed_found = 1
            control_v += 1
    assert control_seed_found != 0, "NURBS surface control points are non-finite"
    used_control_seed = 0
    used_initial_seed = 0

    # Constrain one solve to every non-empty tensor-product knot span.  A
    # minimum on a span edge/corner is therefore considered explicitly.  The
    # first valid span also dispatches the global nearest-control-point seed;
    # ``seed_kind`` is a runtime loop so the expensive Newton body has only one
    # syntactic call site.
    span_cursor = 0
    span_end = (num_ctrlpts_u - degree_u) * (num_ctrlpts_v - degree_v)
    if ti.static(not isinstance(span_cache, type(None))):
        span_cursor = span_cache.span_tree_prefix[surface_id]
        span_end = span_cache.span_tree_prefix[surface_id + 1]
    while span_cursor < span_end:
        span_u, span_v = degree_u, degree_v
        if ti.static(not isinstance(span_cache, type(None))):
            left, _, escape, span = span_cache.span_tree_nodes[span_cursor]
            node_lower_bound2 = span_cache.span_node_distance_squared(span_cursor, point)
            if found != 0 and node_lower_bound2 > best_distance2 + 1.0e-14 * (1.0 + best_distance2):
                span_cursor = escape
                continue
            if span < 0:
                span_cursor = left
                continue
            local_span = span - span_cache.prefix_num_spans_field[surface_id]
            span_u += local_span // (num_ctrlpts_v - degree_v)
            span_v += local_span % (num_ctrlpts_v - degree_v)
            span_cursor = escape
        else:
            span_u += span_cursor // (num_ctrlpts_v - degree_v)
            span_v += span_cursor % (num_ctrlpts_v - degree_v)
            span_cursor += 1
        span_lower_u = knot_vector_u[start_knot_u + span_u]
        span_upper_u = knot_vector_u[start_knot_u + span_u + 1]
        span_lower_v = knot_vector_v[start_knot_v + span_v]
        span_upper_v = knot_vector_v[start_knot_v + span_v + 1]
        if span_upper_u > span_lower_u and span_upper_v > span_lower_v:
            # A positive-weight NURBS span lies inside the convex hull
            # of its active controls, so this AABB distance is a
            # conservative lower bound for the global closest point.
            span_lower = ti.Vector.zero(ti.f64, point.n)
            span_upper = ti.Vector.zero(ti.f64, point.n)
            if ti.static(not isinstance(span_cache, type(None))):
                span_id = (
                    span_cache.prefix_num_spans_field[surface_id]
                    + (span_u - degree_u) * (num_ctrlpts_v - degree_v)
                    + span_v
                    - degree_v
                )
                span_lower = span_cache.span_lower[span_id]
                span_upper = span_cache.span_upper[span_id]
            else:
                for direction in ti.static(range(point.n)):
                    span_lower[direction] = 1.0e300
                    span_upper[direction] = -1.0e300
                for control_v in range(span_v - degree_v, span_v + 1):
                    for control_u in range(span_u - degree_u, span_u + 1):
                        control_id = start_ctrlpt + control_v * num_ctrlpts_u + control_u
                        control_position = ctrlpts[control_id] + alpha * control_directions[control_id]
                        for direction in ti.static(range(point.n)):
                            span_lower[direction] = ti.min(span_lower[direction], control_position[direction])
                            span_upper[direction] = ti.max(span_upper[direction], control_position[direction])
            lower_bound2 = 0.0
            for direction in ti.static(range(point.n)):
                offset = ti.max(
                    span_lower[direction] - point[direction],
                    point[direction] - span_upper[direction],
                    0.0,
                )
                lower_bound2 += offset * offset
            # Keep one Newton call site to avoid duplicating its large
            # Taichi IR. The extra seed only tightens the upper bound;
            # every span that can improve it still runs its old seed.
            for seed_index in range(ti.static(3 if not isinstance(initial_parameter, type(None)) else 2)):
                seed_kind = seed_index - ti.static(1 if not isinstance(initial_parameter, type(None)) else 0)
                span_can_improve = found == 0 or lower_bound2 <= best_distance2 + 1.0e-14 * (1.0 + best_distance2)
                run_seed = (seed_kind == 0 and span_can_improve) or (seed_kind == 1 and used_control_seed == 0)
                if ti.static(not isinstance(initial_parameter, type(None))):
                    run_seed = run_seed or (
                        seed_kind == -1 and used_initial_seed == 0 and _isfinite_vector(initial_parameter)
                    )
                if run_seed:
                    seed_u = 0.5 * (span_lower_u + span_upper_u)
                    seed_v = 0.5 * (span_lower_v + span_upper_v)
                    seed_lower_u = span_lower_u
                    seed_upper_u = span_upper_u
                    seed_lower_v = span_lower_v
                    seed_upper_v = span_upper_v
                    if seed_kind == 1:
                        seed_u = control_seed_u
                        seed_v = control_seed_v
                        seed_lower_u = lower_u
                        seed_upper_u = upper_u
                        seed_lower_v = lower_v
                        seed_upper_v = upper_v
                        used_control_seed = 1
                    if ti.static(not isinstance(initial_parameter, type(None))):
                        if seed_kind == -1:
                            seed_u = initial_parameter[0]
                            seed_v = initial_parameter[1]
                            seed_lower_u = lower_u
                            seed_upper_u = upper_u
                            seed_lower_v = lower_v
                            seed_upper_v = upper_v
                            used_initial_seed = 1
                    u, v, distance2, residual, success = _surface_projected_newton_candidate(
                        start_knot_u,
                        start_knot_v,
                        start_ctrlpt,
                        num_knot_u,
                        num_knot_v,
                        knot_vector_u,
                        knot_vector_v,
                        ctrlpts,
                        control_directions,
                        alpha,
                        weight,
                        point,
                        seed_u,
                        seed_v,
                        seed_lower_u,
                        seed_upper_u,
                        seed_lower_v,
                        seed_upper_v,
                        basis,
                    )
                    if success != 0 and (found == 0 or distance2 < best_distance2):
                        best_u = u
                        best_v = v
                        best_distance2 = distance2
                        best_residual = residual
                        found = 1

    assert found != 0, "NURBS surface closest-point solve failed"
    return (
        clamp(lower_u, upper_u, best_u),
        clamp(lower_v, upper_v, best_v),
        ti.sqrt(ti.max(best_distance2, 0.0)),
        best_residual,
    )


@ti.func
def get_distance_to_surface_fixed_dim(
    start_knot_u,
    start_knot_v,
    start_ctrlpt,
    num_knot_u,
    num_knot_v,
    knot_vector_u,
    knot_vector_v,
    ctrlpts,
    weight,
    point,
    basis,
    span_cache: ti.template() = None,
    surface_id=0,
    initial_parameter: ti.template() = None,
):
    """Closest point on a stationary NURBS surface."""
    return get_distance_to_surface_moving_fixed_dim(
        start_knot_u,
        start_knot_v,
        start_ctrlpt,
        num_knot_u,
        num_knot_v,
        knot_vector_u,
        knot_vector_v,
        ctrlpts,
        ctrlpts,
        0.0,
        weight,
        point,
        basis,
        span_cache,
        surface_id,
        initial_parameter,
    )


@ti.func
def evaluate_distance_to_surface_fixed_dim(
    start_knot_u,
    start_knot_v,
    start_ctrlpt,
    num_knot_u,
    num_knot_v,
    knot_vector_u,
    knot_vector_v,
    ctrlpts,
    weight,
    point,
    parameter_u,
    parameter_v,
    basis,
):
    residual, _, _, _, _, _, distance2, valid = _surface_closest_state(
        start_knot_u,
        start_knot_v,
        start_ctrlpt,
        num_knot_u,
        num_knot_v,
        knot_vector_u,
        knot_vector_v,
        ctrlpts,
        ctrlpts,
        0.0,
        weight,
        point,
        parameter_u,
        parameter_v,
        basis,
    )
    assert valid != 0, "NURBS surface evaluation failed"
    return ti.sqrt(ti.max(distance2, 0.0)), residual


@ti.func
def outer_product_nd(vector1, vector2):
    """Dimension-generic outer product used by the NURBS derivatives."""
    matrix = ti.Matrix.zero(ti.f64, vector1.n, vector2.n)
    if ti.static(vector1.n * vector2.n <= 9):
        for i in ti.static(range(vector1.n)):
            for j in ti.static(range(vector2.n)):
                matrix[i, j] = vector1[i] * vector2[j]
    else:
        flat = 0
        while flat < vector1.n * vector2.n:
            i = flat // vector2.n
            j = flat - i * vector2.n
            matrix[i, j] = vector1[i] * vector2[j]
            flat += 1
    return matrix


@ti.func
def _euclidean_psd_pullback_metric(gram, reduced_hessian, required_rank):
    """Return ``P`` such that ``R.T @ P @ R`` is ``PSD(R.T K R)``.

    ``gram`` is ``R @ R.T`` and ``reduced_hessian`` is ``K``.  Whitening
    ``R`` turns its row space into an orthonormal basis, so clamping the small
    matrix ``sqrt(G) K sqrt(G)`` is exactly the Euclidean spectral projection
    of the full local Hessian, rather than a blockwise or frozen-coordinate
    approximation.  Point--NURBS distance Hessians have rank at most
    ``spatial_dimension + surface_parameter_dimension`` (five in 3D), making
    this the GPU equivalent of IPC's local ``makePD`` without a large dense
    eigensolve over every control-point DOF.
    """
    # Normalize the reduced columns before the rank decision.  J columns are
    # dimensionless while closest-parameter columns carry a geometry scale;
    # applying one absolute threshold to the unscaled Gram matrix can
    # incorrectly delete a valid IFT mode merely because the model is small.
    scale = ti.Vector.zero(ti.f64, gram.n)
    inverse_scale = ti.Vector.zero(ti.f64, gram.n)
    normalized_gram = ti.Matrix.zero(ti.f64, gram.n, gram.n)
    scaled_hessian = ti.Matrix.zero(ti.f64, gram.n, gram.n)
    if ti.static(gram.n <= 3):
        for mode in ti.static(range(gram.n)):
            scale[mode] = ti.sqrt(ti.max(gram[mode, mode], 0.0))
            if scale[mode] > 1.0e-30:
                inverse_scale[mode] = 1.0 / scale[mode]
    else:
        mode = 0
        while mode < gram.n:
            scale[mode] = ti.sqrt(ti.max(gram[mode, mode], 0.0))
            if scale[mode] > 1.0e-30:
                inverse_scale[mode] = 1.0 / scale[mode]
            mode += 1
    for row in range(gram.n):
        for column in range(gram.n):
            normalized_gram[row, column] = inverse_scale[row] * gram[row, column] * inverse_scale[column]
            scaled_hessian[row, column] = scale[row] * reduced_hessian[row, column] * scale[column]

    eigenvalues, eigenvectors = symmetric_eigendecomposition_nd(normalized_gram)
    rank_tolerance = 128.0 * 2.220446049250313e-16 * ti.static(gram.n)
    square_root = ti.Matrix.zero(ti.f64, gram.n, gram.n)
    inverse_square_root = ti.Matrix.zero(ti.f64, gram.n, gram.n)
    numerical_rank = 0
    mode = 0
    while mode < gram.n:
        value = ti.max(eigenvalues[mode], 0.0)
        root = ti.sqrt(value)
        inverse_root = 0.0
        if value > rank_tolerance:
            inverse_root = 1.0 / root
            numerical_rank += 1
        row = 0
        while row < gram.n:
            column = 0
            while column < gram.n:
                projector_entry = eigenvectors[row, mode] * eigenvectors[column, mode]
                square_root[row, column] += root * projector_entry
                inverse_square_root[row, column] += inverse_root * projector_entry
                column += 1
            row += 1
        mode += 1

    whitened_hessian = square_root @ scaled_hessian @ square_root
    projected_whitened = psd_project_nd(whitened_hessian)
    normalized_metric = inverse_square_root @ projected_whitened @ inverse_square_root
    metric = ti.Matrix.zero(ti.f64, gram.n, gram.n)
    for row in range(gram.n):
        for column in range(gram.n):
            metric[row, column] = inverse_scale[row] * normalized_metric[row, column] * inverse_scale[column]
    valid = numerical_rank >= required_rank
    return 0.5 * (metric + metric.transpose()), valid


@ti.func
def curve_control_reduced_jacobian(
    pointer,
    tangent,
    shape,
    shape_derivative,
    free_parameter,
):
    dimension = ti.static(pointer.n)
    jacobian = ti.Matrix.zero(ti.f64, dimension + 1, dimension)
    for component in ti.static(range(dimension)):
        jacobian[component, component] = shape
        if free_parameter != 0:
            jacobian[dimension, component] = shape * tangent[component] + shape_derivative * pointer[component]
    return jacobian


@ti.func
def curve_point_reduced_jacobian(pointer, tangent, free_parameter):
    dimension = ti.static(pointer.n)
    jacobian = ti.Matrix.zero(ti.f64, dimension + 1, dimension)
    for component in ti.static(range(dimension)):
        jacobian[component, component] = -1.0
        if free_parameter != 0:
            jacobian[dimension, component] = -tangent[component]
    return jacobian


@ti.func
def curve_barrier_projected_metric(
    pointer,
    distance,
    tangent,
    curvature,
    shape_values,
    shape_derivatives,
    free_parameter,
    barrier_gradient,
    barrier_hessian,
    measure,
):
    """Exact IPC spectral clamp for a point--NURBS-curve local Hessian."""
    dimension = ti.static(pointer.n)
    reduced_dimension = ti.static(dimension + 1)
    gram = ti.Matrix.zero(ti.f64, reduced_dimension, reduced_dimension)
    if ti.static(shape_values.n <= 3):
        for support in ti.static(range(shape_values.n)):
            jacobian = curve_control_reduced_jacobian(
                pointer,
                tangent,
                shape_values[support],
                shape_derivatives[support],
                free_parameter,
            )
            gram += jacobian @ jacobian.transpose()
    else:
        support = 0
        while support < shape_values.n:
            jacobian = curve_control_reduced_jacobian(
                pointer,
                tangent,
                shape_values[support],
                shape_derivatives[support],
                free_parameter,
            )
            gram += jacobian @ jacobian.transpose()
            support += 1
    point_jacobian = curve_point_reduced_jacobian(pointer, tangent, free_parameter)
    gram += point_jacobian @ point_jacobian.transpose()

    inverse_distance = 1.0 / distance
    alpha = 0.5 * barrier_gradient * inverse_distance
    beta = 0.25 * (
        barrier_hessian * inverse_distance * inverse_distance
        - barrier_gradient * inverse_distance * inverse_distance * inverse_distance
    )
    reduced_hessian = ti.Matrix.zero(ti.f64, reduced_dimension, reduced_dimension)
    for row in ti.static(range(dimension)):
        for column in ti.static(range(dimension)):
            reduced_hessian[row, column] = (
                2.0 * alpha * (1.0 if row == column else 0.0) + 4.0 * beta * pointer[row] * pointer[column]
            ) * measure
    coefficient = tangent.dot(tangent) + curvature.dot(pointer)
    valid_ift = 1
    if free_parameter != 0:
        if ti.abs(coefficient) > 1.0e-30:
            reduced_hessian[dimension, dimension] = -2.0 * alpha / coefficient * measure
        else:
            valid_ift = 0
    required_rank = dimension
    if free_parameter != 0:
        required_rank += 1
    metric, valid_rank = _euclidean_psd_pullback_metric(gram, reduced_hessian, required_rank)
    return metric, valid_ift != 0 and valid_rank


@ti.func
def surface_control_reduced_jacobian(
    pointer,
    tangent_u,
    tangent_v,
    shape,
    derivative_u,
    derivative_v,
    free_u,
    free_v,
):
    dimension = ti.static(pointer.n)
    jacobian = ti.Matrix.zero(ti.f64, dimension + 2, dimension)
    for component in ti.static(range(dimension)):
        jacobian[component, component] = shape
        if free_u != 0:
            jacobian[dimension, component] = shape * tangent_u[component] + derivative_u * pointer[component]
        if free_v != 0:
            jacobian[dimension + 1, component] = shape * tangent_v[component] + derivative_v * pointer[component]
    return jacobian


@ti.func
def surface_point_reduced_jacobian(pointer, tangent_u, tangent_v, free_u, free_v):
    dimension = ti.static(pointer.n)
    jacobian = ti.Matrix.zero(ti.f64, dimension + 2, dimension)
    for component in ti.static(range(dimension)):
        jacobian[component, component] = -1.0
        if free_u != 0:
            jacobian[dimension, component] = -tangent_u[component]
        if free_v != 0:
            jacobian[dimension + 1, component] = -tangent_v[component]
    return jacobian


@ti.func
def surface_barrier_projected_metric(
    pointer,
    distance,
    tangent_u,
    tangent_v,
    curvature_uu,
    curvature_vv,
    curvature_uv,
    shape_values,
    derivative_u,
    derivative_v,
    free_u,
    free_v,
    barrier_gradient,
    barrier_hessian,
    measure,
):
    """Exact IPC spectral clamp for a point--NURBS-surface local Hessian."""
    dimension = ti.static(pointer.n)
    reduced_dimension = ti.static(dimension + 2)
    gram = ti.Matrix.zero(ti.f64, reduced_dimension, reduced_dimension)
    support = 0
    while support < shape_values.n:
        jacobian = surface_control_reduced_jacobian(
            pointer,
            tangent_u,
            tangent_v,
            shape_values[support],
            derivative_u[support],
            derivative_v[support],
            free_u,
            free_v,
        )
        gram += jacobian @ jacobian.transpose()
        support += 1
    point_jacobian = surface_point_reduced_jacobian(pointer, tangent_u, tangent_v, free_u, free_v)
    gram += point_jacobian @ point_jacobian.transpose()

    inverse_distance = 1.0 / distance
    alpha = 0.5 * barrier_gradient * inverse_distance
    beta = 0.25 * (
        barrier_hessian * inverse_distance * inverse_distance
        - barrier_gradient * inverse_distance * inverse_distance * inverse_distance
    )
    reduced_hessian = ti.Matrix.zero(ti.f64, reduced_dimension, reduced_dimension)
    for row in ti.static(range(dimension)):
        for column in ti.static(range(dimension)):
            reduced_hessian[row, column] = (
                2.0 * alpha * (1.0 if row == column else 0.0) + 4.0 * beta * pointer[row] * pointer[column]
            ) * measure

    coefficient_uu = tangent_u.dot(tangent_u) + curvature_uu.dot(pointer)
    coefficient_uv = tangent_u.dot(tangent_v) + curvature_uv.dot(pointer)
    coefficient_vv = tangent_v.dot(tangent_v) + curvature_vv.dot(pointer)
    inverse_uu = 0.0
    inverse_uv = 0.0
    inverse_vv = 0.0
    valid_ift = 1
    if free_u != 0 and free_v != 0:
        determinant = coefficient_uu * coefficient_vv - coefficient_uv * coefficient_uv
        if ti.abs(determinant) > 1.0e-30:
            inverse_uu = coefficient_vv / determinant
            inverse_uv = -coefficient_uv / determinant
            inverse_vv = coefficient_uu / determinant
        else:
            valid_ift = 0
    elif free_u != 0:
        if ti.abs(coefficient_uu) > 1.0e-30:
            inverse_uu = 1.0 / coefficient_uu
        else:
            valid_ift = 0
    elif free_v != 0:
        if ti.abs(coefficient_vv) > 1.0e-30:
            inverse_vv = 1.0 / coefficient_vv
        else:
            valid_ift = 0

    reduced_hessian[dimension, dimension] = -2.0 * alpha * inverse_uu * measure
    reduced_hessian[dimension, dimension + 1] = -2.0 * alpha * inverse_uv * measure
    reduced_hessian[dimension + 1, dimension] = -2.0 * alpha * inverse_uv * measure
    reduced_hessian[dimension + 1, dimension + 1] = -2.0 * alpha * inverse_vv * measure
    required_rank = dimension
    if free_u != 0:
        required_rank += 1
    if free_v != 0:
        required_rank += 1
    metric, valid_rank = _euclidean_psd_pullback_metric(gram, reduced_hessian, required_rank)
    return metric, valid_ift != 0 and valid_rank


@ti.data_oriented
class PointNurbsDerivative:
    """First and second derivatives of squared point--NURBS distance.

    ``dimension`` is only needed by the two legacy velocity-Jacobian methods,
    whose scalar arguments do not carry a dimension.  All distance methods
    infer it from their vector inputs.  Omitting it preserves historical IGA
    behavior by reading the configured IGA--MPM dimension at construction.
    """

    def __init__(self, dimension=None):
        if dimension is None:
            import src.igampm.config as igampm_config

            dimension = igampm_config.DIM
        dimension = int(dimension)
        if dimension not in (2, 3):
            raise ValueError("PointNurbsDerivative dimension must be 2 or 3")
        self.dimension = dimension

    @ti.func
    def Ddistance2_div_Dpoint(self, pointer):
        return -2.0 * pointer

    @ti.func
    def Ddistance2_div_Dctrlpt(self, pointer, nshape):
        return 2.0 * nshape * pointer

    @ti.func
    def Dknot_div_Dpoint_Curve(self, pointer, tangent, curvature):
        coefficient = tangent.dot(tangent) + curvature.dot(pointer)
        result = ti.Vector.zero(ti.f64, pointer.n)
        if ti.abs(coefficient) > 1.0e-30:
            result = tangent / coefficient
        return result

    @ti.func
    def Dknot_div_Dpoint_Surface(
        self,
        pointer,
        tangent_u,
        tangent_v,
        curvature_uu,
        curvature_vv,
        curvature_uv,
    ):
        coefficient_matrix = mat2x2(
            [0.0, 0.0],
            [0.0, 0.0],
        )
        coefficient_uu = tangent_u.dot(tangent_u) + curvature_uu.dot(pointer)
        coefficient_uv = tangent_u.dot(tangent_v) + curvature_uv.dot(pointer)
        coefficient_vv = tangent_v.dot(tangent_v) + curvature_vv.dot(pointer)
        determinant = coefficient_uu * coefficient_vv - coefficient_uv * coefficient_uv
        if ti.abs(determinant) > 1.0e-30:
            coefficient_matrix = mat2x2(
                [coefficient_vv / determinant, -coefficient_uv / determinant],
                [-coefficient_uv / determinant, coefficient_uu / determinant],
            )
        return (
            coefficient_matrix[0, 0] * tangent_u + coefficient_matrix[0, 1] * tangent_v,
            coefficient_matrix[1, 0] * tangent_u + coefficient_matrix[1, 1] * tangent_v,
        )

    @ti.func
    def D2distance2_div_D2point_Curve(self, pointer, tangent, curvature):
        identity = ti.Matrix.identity(ti.f64, pointer.n)
        parameter_gradient = self.Dknot_div_Dpoint_Curve(pointer, tangent, curvature)
        return -2.0 * (outer_product_nd(tangent, parameter_gradient) - identity)

    @ti.func
    def D2distance2_div_D2point_CurveActive(self, pointer, tangent, curvature, free_parameter):
        result = 2.0 * ti.Matrix.identity(ti.f64, pointer.n)
        if free_parameter != 0:
            result = self.D2distance2_div_D2point_Curve(pointer, tangent, curvature)
        return result

    @ti.func
    def D2distance2_div_D2point_Surface(
        self,
        pointer,
        tangent_u,
        tangent_v,
        curvature_uu,
        curvature_vv,
        curvature_uv,
    ):
        identity = ti.Matrix.identity(ti.f64, pointer.n)
        du_dpoint, dv_dpoint = self.Dknot_div_Dpoint_Surface(
            pointer,
            tangent_u,
            tangent_v,
            curvature_uu,
            curvature_vv,
            curvature_uv,
        )
        return -2.0 * (outer_product_nd(tangent_u, du_dpoint) + outer_product_nd(tangent_v, dv_dpoint) - identity)

    @ti.func
    def D2distance2_div_D2point_SurfaceActive(
        self,
        pointer,
        tangent_u,
        tangent_v,
        curvature_uu,
        curvature_vv,
        curvature_uv,
        free_u,
        free_v,
    ):
        result = 2.0 * ti.Matrix.identity(ti.f64, pointer.n)
        if free_u != 0 and free_v != 0:
            result = self.D2distance2_div_D2point_Surface(
                pointer,
                tangent_u,
                tangent_v,
                curvature_uu,
                curvature_vv,
                curvature_uv,
            )
        elif free_u != 0:
            result = self.D2distance2_div_D2point_Curve(pointer, tangent_u, curvature_uu)
        elif free_v != 0:
            result = self.D2distance2_div_D2point_Curve(pointer, tangent_v, curvature_vv)
        return result

    @ti.func
    def D2distance2_div_D2ctrlpt_Curve(
        self,
        pointer,
        tangent,
        curvature,
        nshape1,
        derivative1,
        nshape2,
        derivative2,
    ):
        identity = ti.Matrix.identity(ti.f64, pointer.n)
        coefficient = tangent.dot(tangent) + curvature.dot(pointer)
        mixed1 = derivative1 * pointer + nshape1 * tangent
        mixed2 = derivative2 * pointer + nshape2 * tangent
        result = 2.0 * nshape1 * nshape2 * identity
        if ti.abs(coefficient) > 1.0e-30:
            result -= 2.0 * outer_product_nd(mixed1, mixed2) / coefficient
        return result

    @ti.func
    def D2distance2_div_D2ctrlpt_CurveActive(
        self,
        pointer,
        tangent,
        curvature,
        nshape1,
        derivative1,
        nshape2,
        derivative2,
        free_parameter,
    ):
        result = 2.0 * nshape1 * nshape2 * ti.Matrix.identity(ti.f64, pointer.n)
        if free_parameter != 0:
            result = self.D2distance2_div_D2ctrlpt_Curve(
                pointer,
                tangent,
                curvature,
                nshape1,
                derivative1,
                nshape2,
                derivative2,
            )
        return result

    @ti.func
    def D2distance2_div_D2ctrlpt_Surface(
        self,
        pointer,
        tangent_u,
        tangent_v,
        curvature_uu,
        curvature_vv,
        curvature_uv,
        nshape1,
        derivative_u1,
        derivative_v1,
        nshape2,
        derivative_u2,
        derivative_v2,
    ):
        identity = ti.Matrix.identity(ti.f64, pointer.n)
        coefficient_matrix = mat2x2([0.0, 0.0], [0.0, 0.0])
        coefficient_uu = tangent_u.dot(tangent_u) + curvature_uu.dot(pointer)
        coefficient_uv = tangent_u.dot(tangent_v) + curvature_uv.dot(pointer)
        coefficient_vv = tangent_v.dot(tangent_v) + curvature_vv.dot(pointer)
        determinant = coefficient_uu * coefficient_vv - coefficient_uv * coefficient_uv
        if ti.abs(determinant) > 1.0e-30:
            coefficient_matrix = mat2x2(
                [coefficient_vv / determinant, -coefficient_uv / determinant],
                [-coefficient_uv / determinant, coefficient_uu / determinant],
            )
        mixed_u1 = derivative_u1 * pointer + nshape1 * tangent_u
        mixed_v1 = derivative_v1 * pointer + nshape1 * tangent_v
        mixed_u2 = derivative_u2 * pointer + nshape2 * tangent_u
        mixed_v2 = derivative_v2 * pointer + nshape2 * tangent_v
        mixed1 = ti.Matrix.rows([mixed_u1, mixed_v1])
        mixed2 = ti.Matrix.rows([mixed_u2, mixed_v2])
        return 2.0 * (nshape1 * nshape2 * identity - mixed1.transpose() @ coefficient_matrix @ mixed2)

    @ti.func
    def D2distance2_div_D2ctrlpt_SurfaceActive(
        self,
        pointer,
        tangent_u,
        tangent_v,
        curvature_uu,
        curvature_vv,
        curvature_uv,
        nshape1,
        derivative_u1,
        derivative_v1,
        nshape2,
        derivative_u2,
        derivative_v2,
        free_u,
        free_v,
    ):
        result = 2.0 * nshape1 * nshape2 * ti.Matrix.identity(ti.f64, pointer.n)
        if free_u != 0 and free_v != 0:
            result = self.D2distance2_div_D2ctrlpt_Surface(
                pointer,
                tangent_u,
                tangent_v,
                curvature_uu,
                curvature_vv,
                curvature_uv,
                nshape1,
                derivative_u1,
                derivative_v1,
                nshape2,
                derivative_u2,
                derivative_v2,
            )
        elif free_u != 0:
            result = self.D2distance2_div_D2ctrlpt_Curve(
                pointer,
                tangent_u,
                curvature_uu,
                nshape1,
                derivative_u1,
                nshape2,
                derivative_u2,
            )
        elif free_v != 0:
            result = self.D2distance2_div_D2ctrlpt_Curve(
                pointer,
                tangent_v,
                curvature_vv,
                nshape1,
                derivative_v1,
                nshape2,
                derivative_v2,
            )
        return result

    @ti.func
    def D2distance2_div_DctrlptDpoint_Curve(self, pointer, tangent, curvature, nshape, derivative):
        identity = ti.Matrix.identity(ti.f64, pointer.n)
        parameter_gradient = self.Dknot_div_Dpoint_Curve(pointer, tangent, curvature)
        return 2.0 * (
            (outer_product_nd(tangent, parameter_gradient) - identity) * nshape
            + outer_product_nd(pointer * derivative, parameter_gradient)
        )

    @ti.func
    def D2distance2_div_DctrlptDpoint_CurveActive(
        self, pointer, tangent, curvature, nshape, derivative, free_parameter
    ):
        result = -2.0 * nshape * ti.Matrix.identity(ti.f64, pointer.n)
        if free_parameter != 0:
            result = self.D2distance2_div_DctrlptDpoint_Curve(pointer, tangent, curvature, nshape, derivative)
        return result

    @ti.func
    def D2distance2_div_DctrlptDpoint_Surface(
        self,
        pointer,
        tangent_u,
        tangent_v,
        curvature_uu,
        curvature_vv,
        curvature_uv,
        nshape,
        derivative_u,
        derivative_v,
    ):
        identity = ti.Matrix.identity(ti.f64, pointer.n)
        du_dpoint, dv_dpoint = self.Dknot_div_Dpoint_Surface(
            pointer,
            tangent_u,
            tangent_v,
            curvature_uu,
            curvature_vv,
            curvature_uv,
        )
        return 2.0 * (
            (outer_product_nd(tangent_u, du_dpoint) + outer_product_nd(tangent_v, dv_dpoint) - identity) * nshape
            + outer_product_nd(pointer * derivative_u, du_dpoint)
            + outer_product_nd(pointer * derivative_v, dv_dpoint)
        )

    @ti.func
    def D2distance2_div_DctrlptDpoint_SurfaceActive(
        self,
        pointer,
        tangent_u,
        tangent_v,
        curvature_uu,
        curvature_vv,
        curvature_uv,
        nshape,
        derivative_u,
        derivative_v,
        free_u,
        free_v,
    ):
        result = -2.0 * nshape * ti.Matrix.identity(ti.f64, pointer.n)
        if free_u != 0 and free_v != 0:
            result = self.D2distance2_div_DctrlptDpoint_Surface(
                pointer,
                tangent_u,
                tangent_v,
                curvature_uu,
                curvature_vv,
                curvature_uv,
                nshape,
                derivative_u,
                derivative_v,
            )
        elif free_u != 0:
            result = self.D2distance2_div_DctrlptDpoint_Curve(
                pointer,
                tangent_u,
                curvature_uu,
                nshape,
                derivative_u,
            )
        elif free_v != 0:
            result = self.D2distance2_div_DctrlptDpoint_Curve(
                pointer,
                tangent_v,
                curvature_vv,
                nshape,
                derivative_v,
            )
        return result

    @ti.func
    def Dvelocity_div_Dpoint(self):
        return ti.Matrix.identity(ti.f64, self.dimension)

    @ti.func
    def Dvelocity_div_Dctrlpt(self, nshape):
        return -nshape * ti.Matrix.identity(ti.f64, self.dimension)


__all__ = [
    "CURVE_DISTANCE_TOL",
    "PointNurbsDerivative",
    "closest_curve_point_py",
    "curve_barrier_projected_metric",
    "curve_control_reduced_jacobian",
    "curve_point_reduced_jacobian",
    "eval_curve_py",
    "evaluate_distance_to_curve_fixed_dim",
    "evaluate_distance_to_surface_fixed_dim",
    "get_distance_to_curve_fixed_dim",
    "get_distance_to_surface_fixed_dim",
    "nurbs_basis_py",
    "outer_product_nd",
    "squared_norm_nd",
    "surface_barrier_projected_metric",
    "surface_control_reduced_jacobian",
    "surface_point_reduced_jacobian",
]

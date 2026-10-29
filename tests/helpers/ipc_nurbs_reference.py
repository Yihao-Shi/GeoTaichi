"""Host NURBS contact oracle for tests only."""

import numpy as np


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

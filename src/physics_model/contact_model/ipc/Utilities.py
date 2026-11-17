from src.utils.MatrixFunction import psd_project_nd
from src.utils.MatrixFunction import symmetric_eigendecomposition_nd
from src.utils.ScalarFunction import _isfinite_scalar
from src.utils.VectorFunction import squared_norm_nd, _isfinite_vector
from src.utils.MatrixFunction import outer_product_nd
from src.utils.constants import DBL_EPSILON
import taichi as ti


import numpy as np
from src.utils.linalg import make_list
from src.utils.ScalarFunction import clamp

CURVE_DISTANCE_TOL = 100 * DBL_EPSILON
CLOSEST_POINT_PARAMETER_TOL = 1.0e-12
CLOSEST_POINT_STATIONARITY_TOL = 1.0e-10
CLOSEST_POINT_MAX_ITERATIONS = 64
CLOSEST_POINT_MAX_BACKTRACKS = 12


def _single_finite_parameter(name, value, *, positive=False, nonnegative=False):
    """Validate the scalar public parameters stored in one-entry Taichi fields."""
    array = np.asarray(make_list(value), dtype=np.float64).reshape(-1)
    if array.size != 1:
        raise ValueError(f"{name} must be a scalar")
    scalar = float(array[0])
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


@ti.func
def ipc_barrier_distance_terms(distance, active_distance, kappa):
    """Squared-distance barrier and derivatives w.r.t. distance."""
    distance2 = distance * distance
    active_distance2 = active_distance * active_distance
    energy, gradient_distance2, hessian_distance2 = ipc_barrier_distance2_terms(distance2, active_distance2, kappa)
    gradient = 2.0 * distance * gradient_distance2
    hessian = 2.0 * gradient_distance2 + 4.0 * distance2 * hessian_distance2
    return energy, gradient, hessian


@ti.func
def ipc_barrier_force_magnitude(
    distance2,
    dhat,
    dmin,
    kappa,
    use_physical_barrier,
):
    """Normal-force magnitude used to freeze lagged Coulomb friction."""
    _, gradient, _ = ipc_barrier_distance2_offset_terms(distance2, dhat, dmin, kappa, use_physical_barrier)
    return -2.0 * ti.sqrt(ti.max(distance2, 0.0)) * gradient


@ti.func
def ipc_barrier_force_magnitude_gradient(
    distance2,
    distance2_gradient,
    dhat,
    dmin,
    kappa,
    use_physical_barrier,
):
    """Gradient of normal-force magnitude with respect to stencil DOFs."""
    _, gradient, hessian = ipc_barrier_distance2_offset_terms(distance2, dhat, dmin, kappa, use_physical_barrier)
    distance = ti.sqrt(ti.max(distance2, 1.0e-300))
    return -(2.0 * distance * hessian + gradient / distance) * distance2_gradient


@ti.func
def ipc_lagged_friction_cache_vjp(
    positions,
    references,
    adjoint,
    distance2,
    distance_gradient,
    distance_hessian,
    coefficient,
    coefficient_gradient,
    timestep,
    epsv,
    edge_contact: ti.template(),
):
    """VJP of frozen mesh-friction data with respect to its source geometry.

    Current positions, step references and the adjoint stay fixed here. The
    squared-distance Hessian supplies exact closest-weight and normal changes.
    """
    distance = ti.sqrt(ti.max(distance2, 1e-300))
    normal = ti.Vector.zero(float, 3)
    for component in ti.static(range(3)):
        numerator = distance_gradient[component]
        if ti.static(edge_contact):
            numerator += distance_gradient[3 + component]
        normal[component] = numerator / (2.0 * distance)
    weights = ti.Vector.zero(float, 4)
    relative = ti.Vector.zero(float, 3)
    relative_adjoint = ti.Vector.zero(float, 3)
    for site in range(4):
        for component in ti.static(range(3)):
            weights[site] += distance_gradient[3 * site + component] * normal[component] / (2.0 * distance)
        for component in ti.static(range(3)):
            relative[component] += (
                weights[site] * (positions[3 * site + component] - references[3 * site + component]) / timestep
            )
            relative_adjoint[component] += weights[site] * adjoint[3 * site + component]
    tangent = ti.Matrix.identity(float, 3) - normal.outer_product(normal)
    velocity = tangent @ relative
    speed = velocity.norm()
    profile = ipc_friction_f1_over_speed(speed, epsv)
    radial = profile * velocity
    velocity_vjp = coefficient * profile * relative_adjoint
    if speed > 0.0:
        velocity_vjp += (
            coefficient * ipc_friction_hessian_term(speed, epsv) / speed * velocity * velocity.dot(relative_adjoint)
        )
    normal_vjp = -velocity_vjp * normal.dot(relative) - relative * velocity_vjp.dot(normal)
    weight_vjp = ti.Vector.zero(float, 4)
    for site in range(4):
        for component in ti.static(range(3)):
            weight_vjp[site] += coefficient * adjoint[3 * site + component] * radial[component]
            weight_vjp[site] += (
                (tangent @ velocity_vjp)[component]
                * (positions[3 * site + component] - references[3 * site + component])
                / timestep
            )
    result = relative_adjoint.dot(radial) * coefficient_gradient
    for column in range(12):
        dd = distance_gradient[column] / (2.0 * distance)
        dn = ti.Vector.zero(float, 3)
        for component in ti.static(range(3)):
            numerator = distance_hessian[component, column]
            if ti.static(edge_contact):
                numerator += distance_hessian[3 + component, column]
            dn[component] = numerator / (2.0 * distance) - normal[component] * dd / distance
        result[column] += normal_vjp.dot(dn)
        for site in range(4):
            dw = -weights[site] * dd / distance
            for component in ti.static(range(3)):
                dw += (
                    distance_hessian[3 * site + component, column] * normal[component]
                    + distance_gradient[3 * site + component] * dn[component]
                ) / (2.0 * distance)
            result[column] += weight_vjp[site] * dw
    return result


@ti.func
def ipc_friction_f1(speed, epsv):
    """IPC's C1 stick-to-slip mollifier."""
    value = 1.0
    if speed < epsv:
        value = speed * (-speed + 2.0 * epsv) / (epsv * epsv)
    return value


@ti.func
def ipc_friction_f1_derivative(speed, epsv):
    value = 0.0
    if speed < epsv:
        value = 2.0 * (epsv - speed) / (epsv * epsv)
    return value


@ti.func
def ipc_fully_implicit_profile_over_speed(speed, epsv, profile_id):
    """Evaluate ``p(v; eps) = s(v; eps) / v`` without a zero division."""
    value = 0.0
    if profile_id == 0:
        # ``FrictionProfile::Quadratic`` and Eq. (5) in the paper.
        if speed < epsv:
            value = (2.0 - speed / epsv) / epsv
        else:
            value = 1.0 / speed
    else:
        # Stabilized C-infinity profile.
        value = 1.0 / (speed + 0.1 * epsv)
    return value


@ti.func
def ipc_fully_implicit_profile_over_speed_derivative(speed, epsv, profile_id):
    """Exact derivative of :func:`ipc_fully_implicit_profile_over_speed`."""
    value = 0.0
    if profile_id == 0:
        if speed < epsv:
            value = -1.0 / (epsv * epsv)
        else:
            value = -1.0 / (speed * speed)
    else:
        denominator = speed + 0.1 * epsv
        value = -1.0 / (denominator * denominator)
    return value


@ti.func
def ipc_stribeck_falloff(speed, stribeck_velocity):
    """Compact cubic ``g`` from Eq. (6) of the 2024 paper."""
    value = 0.0
    if speed <= stribeck_velocity:
        x = speed / stribeck_velocity
        value = (2.0 * x + 1.0) * (x - 1.0) * (x - 1.0)
    return value


@ti.func
def ipc_stribeck_falloff_derivative(speed, stribeck_velocity):
    """Derivative of the compact cubic falloff with respect to speed."""
    value = 0.0
    if speed <= stribeck_velocity:
        value = 6.0 * speed * (speed - stribeck_velocity) / (stribeck_velocity * stribeck_velocity * stribeck_velocity)
    return value


@ti.func
def ipc_fully_implicit_scalar_law(
    speed,
    normal_force,
    mu_dynamic,
    mu_static,
    mu_viscous,
    stribeck_velocity,
    epsv,
    profile_id,
):
    """Device form of the published radial law and exact derivatives.

    The three returned scalars are the resistance factor, its derivative with
    respect to tangential speed, and its derivative with respect to the
    current normal force.  Curved-contact kernels combine these values with
    their exact geometry Jacobians; unlike the point-plane convenience
    routine below, no normal or closest-coordinate derivative is omitted.
    """
    profile = ipc_fully_implicit_profile_over_speed(speed, epsv, profile_id)
    profile_derivative = ipc_fully_implicit_profile_over_speed_derivative(speed, epsv, profile_id)
    friction_difference = mu_static - mu_dynamic
    falloff = 0.0
    falloff_derivative = 0.0
    if friction_difference != 0.0:
        falloff = ipc_stribeck_falloff(speed, stribeck_velocity)
        falloff_derivative = ipc_stribeck_falloff_derivative(speed, stribeck_velocity)
    effective_mu = mu_dynamic + friction_difference * falloff
    factor_per_normal_force = effective_mu * profile
    radial_factor = normal_force * factor_per_normal_force + mu_viscous
    radial_speed_derivative = normal_force * (
        friction_difference * falloff_derivative * profile + effective_mu * profile_derivative
    )
    return (
        radial_factor,
        radial_speed_derivative,
        factor_per_normal_force,
    )


@ti.func
def ipc_fully_implicit_point_plane_stribeck_force(
    relative_velocity,
    normal,
    normal_force,
    mu_dynamic,
    mu_static,
    mu_viscous,
    stribeck_velocity,
    epsv,
    profile_id,
):
    """Evaluate the published point-plane force without its Jacobian.

    This is the residual-only counterpart of
    :func:`ipc_fully_implicit_point_plane_stribeck_friction`.  Keeping it
    separate lets GPU Armijo probes avoid all normal-force/profile derivative
    and dense Jacobian work while evaluating exactly the same force law.
    """
    dimension = ti.static(normal.n)
    tangent = ti.Matrix.identity(ti.f64, dimension) - normal.outer_product(normal)
    tangential_velocity = tangent @ relative_velocity
    speed = tangential_velocity.norm()
    profile = ipc_fully_implicit_profile_over_speed(speed, epsv, profile_id)
    friction_difference = mu_static - mu_dynamic
    falloff = 0.0
    if friction_difference != 0.0:
        falloff = ipc_stribeck_falloff(speed, stribeck_velocity)
    effective_mu = mu_dynamic + friction_difference * falloff
    radial_factor = normal_force * effective_mu * profile + mu_viscous
    return radial_factor * tangential_velocity


@ti.func
def ipc_fully_implicit_point_plane_stribeck_friction(
    relative_velocity,
    normal,
    normal_force,
    normal_force_gradient,
    mu_dynamic,
    mu_static,
    mu_viscous,
    stribeck_velocity,
    epsv,
    profile_id,
    velocity_displacement_scale,
):
    """Friction force and exact point-plane force Jacobian.

    The returned force is the *positive resistance vector* that callers
    subtract from the momentum residual. For a fixed planar contact map this
    evaluates the paper's friction map and its exact Jacobian, then adds the
    configuration derivative of the current normal force. The latter outer
    product is why the resulting Jacobian is generally non-symmetric; it is
    not a Hessian of an incremental potential. The derivatives are taken
    directly from the published equations.

    Derivatives of the sliding basis/contact Jacobian are exactly zero for the
    stationary point-plane configuration accepted by the production solver.
    General curved or deforming contact must add those terms before using this
    kernel.
    """
    dimension = ti.static(normal.n)
    tangent = ti.Matrix.identity(ti.f64, dimension) - normal.outer_product(normal)
    tangential_velocity = tangent @ relative_velocity
    speed = tangential_velocity.norm()

    profile = ipc_fully_implicit_profile_over_speed(speed, epsv, profile_id)
    profile_derivative = ipc_fully_implicit_profile_over_speed_derivative(speed, epsv, profile_id)
    friction_difference = mu_static - mu_dynamic
    falloff = 0.0
    falloff_derivative = 0.0
    # v_s is immaterial in the ordinary Coulomb case mu_s == mu_d.  This
    # branch also accepts explicit v_s=0 Coulomb configurations
    # without ever evaluating the otherwise undefined ratio v/v_s.
    if friction_difference != 0.0:
        falloff = ipc_stribeck_falloff(speed, stribeck_velocity)
        falloff_derivative = ipc_stribeck_falloff_derivative(speed, stribeck_velocity)
    effective_mu = mu_dynamic + friction_difference * falloff

    # eta(z, lambda) = z [lambda mu(|z|) p(|z|) + mu_v].
    radial_factor = normal_force * effective_mu * profile + mu_viscous
    radial_factor_derivative = normal_force * (
        mu_dynamic * profile_derivative
        + friction_difference * (falloff_derivative * profile + falloff * profile_derivative)
    )
    force = radial_factor * tangential_velocity

    velocity_jacobian = radial_factor * ti.Matrix.identity(ti.f64, dimension)
    if speed > 0.0:
        velocity_jacobian += (radial_factor_derivative / speed) * tangential_velocity.outer_product(tangential_velocity)

    # Only the Coulomb/Stribeck part is proportional to lambda; viscous
    # friction is independent of the normal-force multiplier in Eq. (6).
    force_per_normal_force = effective_mu * profile * tangential_velocity
    jacobian = (
        velocity_displacement_scale * tangent @ velocity_jacobian @ tangent
        + force_per_normal_force.outer_product(normal_force_gradient)
    )
    return force, jacobian


@ti.func
def ipc_fully_implicit_point_plane_friction(
    relative_velocity,
    normal,
    mu_lambda,
    mu_lambda_gradient,
    epsv,
    velocity_displacement_scale,
):
    """Current friction force and exact point-plane displacement Jacobian.

    This is the fully implicit isotropic C1 law for a contact whose normal,
    tangent projector, and point-to-DOF map are constant.  ``mu_lambda`` and
    its spatial gradient are evaluated at the current point.  Curved/sliding
    contact stencils require additional derivatives and deliberately do not
    use this restricted kernel.
    """
    # Backward-compatible special case: mu_static = mu_dynamic = 1 and the
    # caller has already folded mu into ``mu_lambda``.
    return ipc_fully_implicit_point_plane_stribeck_friction(
        relative_velocity,
        normal,
        mu_lambda,
        mu_lambda_gradient,
        1.0,
        1.0,
        0.0,
        epsv,
        epsv,
        0,
        velocity_displacement_scale,
    )


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
    position, tangent, curvature = basis.nurbs_basis_moving_interpolations2nd_ders1d(
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
    ) = basis.nurbs_basis_moving_interpolations2nd_ders2d(
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


@ti.func
def ipc_friction_f0(speed, epsv, timestep):
    """Antiderivative of the IPC smoothed Coulomb force in velocity form."""
    value = speed * timestep
    if speed < epsv:
        displacement = speed * timestep
        threshold = epsv * timestep
        threshold2 = threshold * threshold
        value = displacement * displacement * (-displacement / 3.0 + threshold) / threshold2 + threshold / 3.0
    return value


@ti.func
def ipc_friction_f1_over_speed(speed, epsv):
    """Return ``f1(speed) / speed`` using its finite zero-speed limit."""
    value = 0.0
    if speed < epsv:
        value = (-speed + 2.0 * epsv) / (epsv * epsv)
    else:
        value = 1.0 / speed
    return value


@ti.func
def ipc_friction_hessian_term(speed, epsv):
    """Radial Hessian coefficient used by the lagged friction potential."""
    value = 0.0
    if speed < epsv:
        value = -1.0 / (epsv * epsv)
    else:
        value = -1.0 / (speed * speed)
    return value


@ti.func
def ipc_barrier_distance2_terms(distance2, active_distance2, kappa):
    """Clamped-log IPC barrier acting on squared distance.

    This is ``BarrierPotential``'s default ``ClampedLogBarrier`` with
    ``dmin=0``: ``b(d², d_hat²)``. It deliberately does not divide by
    ``d_hat⁴``.
    """
    energy = 0.0
    gradient = 0.0
    hessian = 0.0
    if distance2 <= 0.0:
        # Exact ClampedLogBarrier domain semantics.  IPC's CCD/line search
        # must keep production iterates strictly positive; an invalid state is
        # therefore exposed as +inf instead of being hidden by a gap floor.
        energy = ti.math.inf
    elif distance2 < active_distance2:
        diff = distance2 - active_distance2
        log_term = ti.log(distance2 / active_distance2)
        energy = -kappa * diff * diff * log_term
        gradient = -kappa * (2.0 * diff * log_term + diff * diff / distance2)
        hessian = -kappa * (2.0 * log_term + 4.0 * diff / distance2 - diff * diff / (distance2 * distance2))
    return energy, gradient, hessian


@ti.func
def ipc_barrier_distance2_offset_terms(
    distance2,
    dhat,
    dmin,
    kappa,
    use_physical_barrier,
):
    """Barrier-potential terms with a minimum separation.

    The returned derivatives are with respect to the *unshifted* squared
    distance.  ``dhat`` is the activation gap above ``dmin`` (the active
    distance is therefore ``dmin + dhat``).
    """
    shifted_distance2 = distance2 - dmin * dmin
    shifted_dhat2 = (2.0 * dmin + dhat) * dhat
    energy, gradient, hessian = ipc_barrier_distance2_terms(shifted_distance2, shifted_dhat2, kappa)
    if use_physical_barrier != 0:
        # ClampedLogBarrier::units(x) == x^2.
        units = shifted_dhat2 * shifted_dhat2
        physical_scale = dhat / ti.max(units, 1.0e-300)
        energy *= physical_scale
        gradient *= physical_scale
        hessian *= physical_scale
    return energy, gradient, hessian


@ti.func
def ipc_barrier_distance_offset_terms(
    distance,
    dhat,
    dmin,
    kappa,
    use_physical_barrier,
):
    """Offset barrier and derivatives with respect to distance."""
    distance2 = distance * distance
    energy, gradient_distance2, hessian_distance2 = ipc_barrier_distance2_offset_terms(
        distance2, dhat, dmin, kappa, use_physical_barrier
    )
    gradient = 2.0 * distance * gradient_distance2
    hessian = 2.0 * gradient_distance2 + 4.0 * distance2 * hessian_distance2
    return energy, gradient, hessian

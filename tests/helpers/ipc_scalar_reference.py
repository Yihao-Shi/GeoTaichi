"""Independent host IPC scalar laws for numerical verification only."""

import numpy as np
from src.physics_model.contact_model.ipc.IPC import _normalize_friction_profile, _single_finite_parameter


def ipc_fully_implicit_scalar_law_py(
    speed,
    normal_force,
    mu_dynamic,
    mu_static,
    mu_viscous,
    stribeck_velocity,
    epsv,
    profile="quadratic",
):
    """Evaluate the radial friction law and its exact scalar derivatives.

    This host-side counterpart of the Taichi kernels is intended for contact
    stencils whose geometry Jacobian is assembled with algorithmic
    differentiation.  For tangential velocity ``z`` the resistance is

    ``eta(z, lambda) = z * radial_factor``.

    The return value is ``(radial_factor, d_factor_d_speed,
    d_factor_d_normal_force)``.  Keeping this constitutive scalar law here
    prevents the PT/EE, affine, and IGA assemblers from carrying independent
    copies of the paper's Stribeck/profile formulas.
    """
    speed = _single_finite_parameter("speed", speed, nonnegative=True)
    normal_force = _single_finite_parameter("normal_force", normal_force, nonnegative=True)
    mu_dynamic = _single_finite_parameter("dynamic_friction", mu_dynamic, nonnegative=True)
    mu_static = _single_finite_parameter("static_friction", mu_static, nonnegative=True)
    mu_viscous = _single_finite_parameter("viscous_friction", mu_viscous, nonnegative=True)
    epsv = _single_finite_parameter("epsv", epsv, positive=True)
    stribeck_velocity = _single_finite_parameter("stribeck_velocity", stribeck_velocity, nonnegative=True)
    profile_id = (
        int(profile)
        if isinstance(profile, (int, np.integer)) and int(profile) in (0, 1)
        else _normalize_friction_profile(profile)
    )
    friction_difference = mu_static - mu_dynamic
    if friction_difference != 0.0 and stribeck_velocity <= 0.0:
        raise ValueError("stribeck_velocity must be positive when static_friction " "differs from dynamic_friction")

    if profile_id == 0:
        if speed < epsv:
            profile_value = (2.0 - speed / epsv) / epsv
            profile_derivative = -1.0 / (epsv * epsv)
        else:
            profile_value = 1.0 / speed
            profile_derivative = -1.0 / (speed * speed)
    else:
        denominator = speed + 0.1 * epsv
        profile_value = 1.0 / denominator
        profile_derivative = -1.0 / (denominator * denominator)

    falloff = 0.0
    falloff_derivative = 0.0
    if friction_difference != 0.0 and speed <= stribeck_velocity:
        x = speed / stribeck_velocity
        falloff = (2.0 * x + 1.0) * (x - 1.0) * (x - 1.0)
        falloff_derivative = (
            6.0 * speed * (speed - stribeck_velocity) / (stribeck_velocity * stribeck_velocity * stribeck_velocity)
        )

    effective_mu = mu_dynamic + friction_difference * falloff
    factor_per_normal_force = effective_mu * profile_value
    radial_factor = normal_force * factor_per_normal_force + mu_viscous
    radial_factor_derivative = normal_force * (
        friction_difference * falloff_derivative * profile_value + effective_mu * profile_derivative
    )
    return (
        radial_factor,
        radial_factor_derivative,
        factor_per_normal_force,
    )


def ipc_barrier_distance_terms_py(
    distance,
    dhat,
    dmin=0.0,
    kappa=1.0,
    use_physical_barrier=False,
):
    """Evaluate offset IPC barrier distance terms on the host.

    Returns energy, first derivative, and second derivative with respect to
    the unshifted Euclidean ``distance``. The algebra and physical-unit
    scaling match :func:`ipc_toolkit_barrier_distance_offset_terms`.
    Invalid states (``distance <= dmin``) return ``(+inf, 0, 0)`` so callers
    can reject them through CCD/line search without silently clamping a gap.
    """
    distance = _single_finite_parameter("distance", distance, nonnegative=True)
    dhat = _single_finite_parameter("dhat", dhat, positive=True)
    dmin = _single_finite_parameter("dmin", dmin, nonnegative=True)
    kappa = _single_finite_parameter("kappa", kappa, positive=True)

    distance2 = distance * distance
    shifted_distance2 = distance2 - dmin * dmin
    shifted_dhat2 = (2.0 * dmin + dhat) * dhat
    energy = 0.0
    gradient_distance2 = 0.0
    hessian_distance2 = 0.0
    if shifted_distance2 <= 0.0:
        return np.inf, 0.0, 0.0
    if shifted_distance2 < shifted_dhat2:
        diff = shifted_distance2 - shifted_dhat2
        log_term = np.log(shifted_distance2 / shifted_dhat2)
        energy = -kappa * diff * diff * log_term
        gradient_distance2 = -kappa * (2.0 * diff * log_term + diff * diff / shifted_distance2)
        hessian_distance2 = -kappa * (
            2.0 * log_term + 4.0 * diff / shifted_distance2 - diff * diff / (shifted_distance2 * shifted_distance2)
        )
        if bool(use_physical_barrier):
            physical_scale = dhat / (shifted_dhat2 * shifted_dhat2)
            energy *= physical_scale
            gradient_distance2 *= physical_scale
            hessian_distance2 *= physical_scale

    gradient = 2.0 * distance * gradient_distance2
    hessian = 2.0 * gradient_distance2 + 4.0 * distance2 * hessian_distance2
    return energy, gradient, hessian


def clamped_log_barrier_second_derivative(distance, active_distance):
    """Evaluate the clamped-log second derivative on the host."""
    distance = float(distance)
    active_distance = float(active_distance)
    if distance <= 0.0 or distance >= active_distance:
        return 0.0
    ratio = active_distance / distance
    return (ratio + 2.0) * ratio - 2.0 * np.log(distance / active_distance) - 3.0


def initial_barrier_stiffness(
    bbox_diagonal,
    dhat,
    average_mass,
    grad_energy,
    grad_barrier,
    min_barrier_stiffness_scale=1.0e11,
    dmin=0.0,
):
    """Compute the initial adaptive barrier stiffness.

    Returns ``(stiffness, maximum_stiffness)``.  ``grad_barrier`` must be the
    gradient of the unit-stiffness barrier potential.
    """
    bbox_diagonal = _single_finite_parameter("bbox_diagonal", bbox_diagonal, positive=True)
    dhat = _single_finite_parameter("dhat", dhat, positive=True)
    average_mass = _single_finite_parameter("average_mass", average_mass, positive=True)
    min_scale = _single_finite_parameter(
        "min_barrier_stiffness_scale",
        min_barrier_stiffness_scale,
        positive=True,
    )
    dmin = _single_finite_parameter("dmin", dmin, nonnegative=True)
    grad_energy = np.asarray(grad_energy, dtype=np.float64).reshape(-1)
    grad_barrier = np.asarray(grad_barrier, dtype=np.float64).reshape(-1)
    if grad_energy.shape != grad_barrier.shape:
        raise ValueError("grad_energy and grad_barrier must have the same shape")
    if not np.isfinite(grad_energy).all() or not np.isfinite(grad_barrier).all():
        raise ValueError("barrier stiffness gradients must be finite")

    dhat2 = dhat * dhat
    dmin2 = dmin * dmin
    distance0 = (1.0e-8 * bbox_diagonal + dmin) ** 2
    active_shifted_distance2 = 2.0 * dmin * dhat + dhat2
    if distance0 - dmin2 >= active_shifted_distance2:
        distance0 = dmin * dhat + 0.5 * dhat2
    denominator = 4.0 * distance0 * clamped_log_barrier_second_derivative(distance0 - dmin2, active_shifted_distance2)
    minimum_stiffness = min_scale * average_mass / denominator
    maximum_stiffness = 100.0 * minimum_stiffness

    stiffness = 1.0
    barrier_norm2 = float(np.dot(grad_barrier, grad_barrier))
    if barrier_norm2 > 0.0:
        stiffness = -float(np.dot(grad_barrier, grad_energy)) / barrier_norm2
    stiffness = float(np.clip(stiffness, minimum_stiffness, maximum_stiffness))
    return stiffness, maximum_stiffness


def update_barrier_stiffness(
    previous_min_distance_squared,
    min_distance_squared,
    max_barrier_stiffness,
    barrier_stiffness,
    bbox_diagonal,
    dhat_epsilon_scale=1.0e-9,
    dmin=0.0,
):
    """Update the adaptive barrier stiffness."""
    previous = _single_finite_parameter(
        "previous_min_distance_squared",
        previous_min_distance_squared,
        nonnegative=True,
    )
    current = _single_finite_parameter("min_distance_squared", min_distance_squared, nonnegative=True)
    maximum = _single_finite_parameter("max_barrier_stiffness", max_barrier_stiffness, positive=True)
    stiffness = _single_finite_parameter("barrier_stiffness", barrier_stiffness, positive=True)
    bbox_diagonal = _single_finite_parameter("bbox_diagonal", bbox_diagonal, positive=True)
    epsilon_scale = _single_finite_parameter("dhat_epsilon_scale", dhat_epsilon_scale, positive=True)
    dmin = _single_finite_parameter("dmin", dmin, nonnegative=True)
    epsilon2 = (epsilon_scale * (bbox_diagonal + dmin)) ** 2
    if previous < epsilon2 and current < epsilon2 and current < previous:
        return min(maximum, 2.0 * stiffness)
    return stiffness

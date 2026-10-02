"""Shared Incremental Potential Contact (IPC) constitutive kernels.

This module deliberately contains only geometry-independent contact laws.  The
contact stencil (point/edge/triangle/NURBS), quadrature weights, closest-point
coordinates, and matrix scattering remain the responsibility of each engine.

The scalar functions implement the IPC barrier and the smoothed
lagged-friction mollifier. They accept all material/time-step parameters as
arguments so that engines with per-material contact properties can use the
same implementation as engines storing a single ``Barrier``/``Friction``
object.
"""

import numpy as np
import taichi as ti

from src.utils.linalg import make_list


def normalize_ipc_model(value):
    """Return the public IPC normal-contact formulation name."""
    key = str(value).strip().replace("_", "").replace("-", "").replace(" ", "").lower()
    aliases = {
        "ipc": "BarrierIPC",
        "barrier": "BarrierIPC",
        "barrieripc": "BarrierIPC",
        "incrementalpotentialcontact": "BarrierIPC",
        "semi": "SemiIPC",
        "semiipc": "SemiIPC",
        "al": "SemiIPC",
        "augmentlagrangian": "SemiIPC",
        "augmentedlagrangian": "SemiIPC",
    }
    if key not in aliases:
        raise ValueError("IPC contact model must be 'BarrierIPC' or 'SemiIPC'")
    return aliases[key]


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


def _normalize_friction_profile(value):
    """Return the friction-profile id.

    ``quadratic`` is the compact C1 profile in Eq. (5) of Larionov et al. and
    is also the profile used by IPC. ``stabilized`` is a C-infinity
    alternative. Keeping this choice explicit avoids silently mixing the two
    regularizations when comparing results.
    """
    key = str(value).strip().replace("-", "_").replace(" ", "_").lower()
    aliases = {
        "quadratic": 0,
        "c1": 0,
        "ipc": 0,
        "stabilized": 1,
        "stabilised": 1,
        "cinfinity": 1,
        "c_infinity": 1,
    }
    if key not in aliases:
        raise ValueError("friction_profile must be 'quadratic' (paper/IPC C1) or " "'stabilized' (C-infinity)")
    return aliases[key]


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


def semi_ipc_terms_py(gap, multiplier, penalty):
    """Projected augmented-Lagrangian contact terms for semi-IPC.

    ``gap >= 0`` is the unilateral constraint, ``multiplier`` is its
    persistent non-negative multiplier, and ``penalty`` is ``mu``.
    Derivatives are with respect to ``gap``.
    """
    gap = _single_finite_parameter("gap", gap)
    multiplier = _single_finite_parameter("multiplier", multiplier, nonnegative=True)
    penalty = _single_finite_parameter("penalty", penalty, positive=True)
    shifted = multiplier - penalty * gap
    projected = max(shifted, 0.0)
    energy = (projected * projected - multiplier * multiplier) / (2.0 * penalty)
    return energy, -projected, penalty if shifted >= 0.0 else 0.0


def semi_ipc_update_multiplier_py(gap, multiplier, penalty):
    """Host update ``lambda <- max(0, lambda - mu * gap)``."""
    return -semi_ipc_terms_py(gap, multiplier, penalty)[1]


@ti.func
def semi_ipc_update_multiplier(gap, multiplier, penalty):
    """Device multiplier update used by semi-IPC contact assemblers."""
    return ti.max(0.0, multiplier - penalty * gap)


@ti.func
def semi_ipc_terms(gap, multiplier, penalty):
    """Device form of :func:`semi_ipc_terms_py`."""
    shifted = multiplier - penalty * gap
    projected = semi_ipc_update_multiplier(gap, multiplier, penalty)
    energy = (projected * projected - multiplier * multiplier) / (2.0 * penalty)
    gradient = -projected
    hessian = 0.0
    if shifted >= 0.0:
        hessian = penalty
    return energy, gradient, hessian


@ti.func
def semi_ipc_hash_index(key, capacity):
    value = (
        ti.cast(key[0], ti.u32) * ti.u32(73856093)
        ^ ti.cast(key[1], ti.u32) * ti.u32(19349663)
        ^ ti.cast(key[2], ti.u32) * ti.u32(83492791)
        ^ ti.cast(key[3], ti.u32) * ti.u32(2654435761)
    )
    return ti.cast(value % ti.cast(capacity, ti.u32), ti.i32)


@ti.func
def semi_ipc_same_key(first, second):
    return first[0] == second[0] and first[1] == second[1] and first[2] == second[2] and first[3] == second[3]


@ti.func
def semi_ipc_find_or_insert(states, keys, multipliers, count, key, capacity):
    """Find/reserve a persistent four-integer SemiIPC contact slot."""
    slot = -1
    start = semi_ipc_hash_index(key, capacity)
    attempt = 0
    while attempt < capacity and slot < 0:
        index = (start + attempt) % capacity
        state = states[index]
        if state == 0:
            if semi_ipc_same_key(keys[index], key):
                slot = index
            else:
                attempt += 1
        elif state == 2:
            previous = ti.atomic_min(states[index], 1)
            if previous == 2:
                keys[index] = key
                multipliers[index] = 0.0
                ti.atomic_add(count[None], 1)
                states[index] = 0
                slot = index
            elif previous == 0:
                states[index] = 0
            else:
                # Do not spin on a slot owned by another thread. Threads in
                # the same warp can otherwise wait for one another forever.
                attempt += 1
        else:
            attempt += 1
    return slot


@ti.func
def semi_ipc_find(states, keys, key, capacity):
    slot = -1
    start = semi_ipc_hash_index(key, capacity)
    attempt = 0
    while attempt < capacity and slot < 0:
        index = (start + attempt) % capacity
        state = states[index]
        if state == 0:
            if semi_ipc_same_key(keys[index], key):
                slot = index
            else:
                attempt += 1
        elif state == 2:
            attempt = capacity
        else:
            # A publisher owns this slot. Continue probing instead of
            # deadlocking its warp while waiting for state to become ready.
            attempt += 1
    return slot


@ti.func
def ipc_toolkit_barrier_distance2_terms(distance2, active_distance2, kappa):
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
def ipc_toolkit_barrier_distance_terms(distance, active_distance, kappa):
    """Squared-distance barrier and derivatives w.r.t. distance."""
    distance2 = distance * distance
    active_distance2 = active_distance * active_distance
    energy, gradient_distance2, hessian_distance2 = ipc_toolkit_barrier_distance2_terms(
        distance2, active_distance2, kappa
    )
    gradient = 2.0 * distance * gradient_distance2
    hessian = 2.0 * gradient_distance2 + 4.0 * distance2 * hessian_distance2
    return energy, gradient, hessian


@ti.func
def ipc_toolkit_barrier_distance2_offset_terms(
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
    energy, gradient, hessian = ipc_toolkit_barrier_distance2_terms(shifted_distance2, shifted_dhat2, kappa)
    if use_physical_barrier != 0:
        # ClampedLogBarrier::units(x) == x^2.
        units = shifted_dhat2 * shifted_dhat2
        physical_scale = dhat / ti.max(units, 1.0e-300)
        energy *= physical_scale
        gradient *= physical_scale
        hessian *= physical_scale
    return energy, gradient, hessian


@ti.func
def ipc_toolkit_barrier_distance_offset_terms(
    distance,
    dhat,
    dmin,
    kappa,
    use_physical_barrier,
):
    """Offset barrier and derivatives with respect to distance."""
    distance2 = distance * distance
    energy, gradient_distance2, hessian_distance2 = ipc_toolkit_barrier_distance2_offset_terms(
        distance2, dhat, dmin, kappa, use_physical_barrier
    )
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
    _, gradient, _ = ipc_toolkit_barrier_distance2_offset_terms(distance2, dhat, dmin, kappa, use_physical_barrier)
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
    _, gradient, hessian = ipc_toolkit_barrier_distance2_offset_terms(
        distance2, dhat, dmin, kappa, use_physical_barrier
    )
    distance = ti.sqrt(ti.max(distance2, 1.0e-300))
    return -(2.0 * distance * hessian + gradient / distance) * distance2_gradient


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
def ipc_friction_hessian_term(speed, epsv):
    """Radial Hessian coefficient used by the lagged friction potential."""
    value = 0.0
    if speed < epsv:
        value = -1.0 / (epsv * epsv)
    else:
        value = -1.0 / (speed * speed)
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


@ti.data_oriented
class Barrier:
    """Field-backed BarrierIPC/SemiIPC parameters used by coupled engines."""

    def __init__(self, **kwargs):
        requested_model = kwargs.get(
            "ipc_model",
            kwargs.get("normal_contact_model", kwargs.get("contact_form", "BarrierIPC")),
        )
        self.model = normalize_ipc_model(requested_model)
        self.is_semi = self.model == "SemiIPC"
        kappa = _single_finite_parameter("kappa", kwargs.get("kappa", 1.0e5), positive=True)
        penalty = _single_finite_parameter("penalty", kwargs.get("penalty", kappa), positive=True)
        dhat = _single_finite_parameter("dhat", kwargs.get("dhat", 1.0e-3), positive=True)
        dmin = _single_finite_parameter("dmin", kwargs.get("dmin", 0.0), nonnegative=True)
        self.kappa = ti.field(ti.f64, shape=1)
        self.dhat = ti.field(ti.f64, shape=1)
        self.dmin = ti.field(ti.f64, shape=1)
        self.penalty = ti.field(ti.f64, shape=1)
        self.kappa[0] = kappa
        self.dhat[0] = dhat
        self.dmin[0] = dmin
        self.penalty[0] = penalty
        self.max_penalty = _single_finite_parameter("max_penalty", kwargs.get("max_penalty", 1.0e12), positive=True)
        if self.max_penalty < penalty:
            raise ValueError("max_penalty must be at least penalty")
        self.penalty_growth = _single_finite_parameter(
            "penalty_growth", kwargs.get("penalty_growth", 4.0), positive=True
        )
        if self.penalty_growth <= 1.0:
            raise ValueError("penalty_growth must be greater than one")
        self.penalty_update_interval = int(kwargs.get("penalty_update_interval", 3))
        if self.penalty_update_interval <= 0:
            raise ValueError("penalty_update_interval must be positive")
        self.sufficient_reduction = _single_finite_parameter(
            "sufficient_reduction",
            kwargs.get("sufficient_reduction", 0.75),
            positive=True,
        )
        if self.sufficient_reduction >= 1.0:
            raise ValueError("sufficient_reduction must be less than one")
        self.constraint_tolerance = _single_finite_parameter(
            "constraint_tolerance",
            kwargs.get("constraint_tolerance", 1.0e-6),
            nonnegative=True,
        )
        self.use_physical_barrier = bool(kwargs.get("use_physical_barrier", kwargs.get("physical_barrier", False)))
        self._use_physical_barrier_id = int(self.use_physical_barrier)
        if "barrier_form" in kwargs:
            raise ValueError(
                "barrier_form is no longer configurable; GeoTaichi contact "
                "always uses the finite-clearance BarrierIPC law"
            )

    @ti.func
    def _terms(self, distance):
        return ipc_toolkit_barrier_distance_offset_terms(
            distance,
            self.dhat[0],
            self.dmin[0],
            self.kappa[0],
            ti.static(self._use_physical_barrier_id),
        )

    @property
    def stiffness(self):
        return self.kappa[0]

    @stiffness.setter
    def stiffness(self, stiffness):
        self.kappa[0] = _single_finite_parameter("stiffness", stiffness, positive=True)

    @property
    def threshold(self):
        return self.dmin[0] + self.dhat[0]

    @threshold.setter
    def threshold(self, threshold):
        threshold = _single_finite_parameter("threshold", threshold, positive=True)
        if threshold <= self.dmin[0]:
            raise ValueError("threshold must be greater than dmin")
        self.dhat[0] = threshold - self.dmin[0]

    @property
    def activation_gap(self):
        return self.dhat[0]

    @activation_gap.setter
    def activation_gap(self, dhat):
        self.dhat[0] = _single_finite_parameter("dhat", dhat, positive=True)

    @property
    def minimum_distance(self):
        return self.dmin[0]

    @minimum_distance.setter
    def minimum_distance(self, dmin):
        dmin = _single_finite_parameter("dmin", dmin, nonnegative=True)
        self.dmin[0] = dmin

    @ti.func
    def activation_distance_term(self):
        return self.dmin[0] + self.dhat[0]

    @ti.func
    def energy_term(self, distance):
        energy, _, _ = self._terms(distance)
        return energy

    @ti.func
    def grad_term(self, distance):
        _, gradient, _ = self._terms(distance)
        return gradient

    @ti.func
    def hess_term(self, distance):
        _, _, hessian = self._terms(distance)
        return hessian

    @ti.func
    def energy(self, distance, area=1.0):
        return self.energy_term(distance) * area

    @ti.func
    def gradient(self, distance, ddistance2_dvertex, area=1.0):
        return 0.5 * self.grad_term(distance) / distance * ddistance2_dvertex * area

    @ti.func
    def hessian(
        self,
        distance,
        ddistance2_dvertex_1,
        ddistance2_dvertex_2,
        d2distance2_d2vertex,
        area=1.0,
    ):
        inv_distance = 1.0 / distance
        inv_distance2 = inv_distance * inv_distance
        inv_distance3 = inv_distance2 * inv_distance
        barrier_gradient = self.grad_term(distance)
        barrier_hessian = self.hess_term(distance)
        hessian = (
            0.25
            * (barrier_hessian * inv_distance2 - barrier_gradient * inv_distance3)
            * ddistance2_dvertex_1.outer_product(ddistance2_dvertex_2)
        )
        hessian += 0.5 * barrier_gradient * inv_distance * d2distance2_d2vertex
        return hessian * area


@ti.data_oriented
class Friction:
    """Field-backed parameters for smooth IPC friction laws.

    The legacy ``mu`` interface remains the equal-static/dynamic Coulomb case.
    Fully implicit contact additionally accepts:
    ``dynamic_friction``, ``static_friction``, ``viscous_friction``,
    ``stribeck_velocity`` and ``friction_profile``.
    """

    def __init__(self, **kwargs):
        legacy_mu = _single_finite_parameter(
            "mu",
            kwargs.get("mu", kwargs.get("dynamic_friction", 0.0)),
            nonnegative=True,
        )
        epsv = _single_finite_parameter("epsv", kwargs.get("epsv", 1.0e-3), positive=True)
        mu_dynamic = _single_finite_parameter(
            "dynamic_friction",
            kwargs.get("dynamic_friction", kwargs.get("mu_dynamic", legacy_mu)),
            nonnegative=True,
        )
        mu_static = _single_finite_parameter(
            "static_friction",
            kwargs.get("static_friction", kwargs.get("mu_static", mu_dynamic)),
            nonnegative=True,
        )
        mu_viscous = _single_finite_parameter(
            "viscous_friction",
            kwargs.get("viscous_friction", kwargs.get("mu_viscous", 0.0)),
            nonnegative=True,
        )
        stribeck_velocity = _single_finite_parameter(
            "stribeck_velocity",
            kwargs.get("stribeck_velocity", 10.0 * epsv),
            nonnegative=True,
        )
        if mu_static != mu_dynamic and stribeck_velocity <= 0.0:
            raise ValueError("stribeck_velocity must be positive when static_friction " "differs from dynamic_friction")
        self.mu = ti.field(ti.f64, shape=1)
        self.epsv = ti.field(ti.f64, shape=1)
        self.mu_dynamic = ti.field(ti.f64, shape=1)
        self.mu_static = ti.field(ti.f64, shape=1)
        self.mu_viscous = ti.field(ti.f64, shape=1)
        self.stribeck_velocity = ti.field(ti.f64, shape=1)
        # Lagged IPC has one Coulomb coefficient. If the caller uses only the
        # Use dynamic friction for the backward-compatible scalar field.
        self.mu[0] = mu_dynamic if "mu" not in kwargs else legacy_mu
        self.epsv[0] = epsv
        self.mu_dynamic[0] = mu_dynamic
        self.mu_static[0] = mu_static
        self.mu_viscous[0] = mu_viscous
        self.stribeck_velocity[0] = stribeck_velocity
        self.profile_id = _normalize_friction_profile(kwargs.get("friction_profile", "quadratic"))

    @property
    def has_friction(self):
        return (
            float(self.mu[0]) > 0.0
            or float(self.mu_dynamic[0]) > 0.0
            or float(self.mu_static[0]) > 0.0
            or float(self.mu_viscous[0]) > 0.0
        )

    @property
    def friction(self):
        return self.mu[0]

    @friction.setter
    def friction(self, friction):
        value = _single_finite_parameter("friction", friction, nonnegative=True)
        self.mu[0] = value
        self.mu_dynamic[0] = value
        self.mu_static[0] = value

    @property
    def dynamic_friction(self):
        return self.mu_dynamic[0]

    @dynamic_friction.setter
    def dynamic_friction(self, friction):
        value = _single_finite_parameter("dynamic_friction", friction, nonnegative=True)
        self.mu_dynamic[0] = value
        self.mu[0] = value

    @property
    def static_friction(self):
        return self.mu_static[0]

    @static_friction.setter
    def static_friction(self, friction):
        self.mu_static[0] = _single_finite_parameter("static_friction", friction, nonnegative=True)

    @property
    def threshold(self):
        return self.epsv[0]

    @threshold.setter
    def threshold(self, threshold):
        self.epsv[0] = _single_finite_parameter("epsv", threshold, positive=True)

    @ti.func
    def energy_term(self, speed, timestep):
        return ipc_friction_f0(speed, self.epsv[0], timestep)

    @ti.func
    def grad_term(self, speed):
        return ipc_friction_f1_over_speed(speed, self.epsv[0])

    @ti.func
    def hess_term(self, speed):
        return ipc_friction_hessian_term(speed, self.epsv[0])

    @ti.func
    def energy(self, relative_displacement, normal, mu_lambda, timestep, area=1.0):
        tangent = ti.Matrix.identity(ti.f64, normal.n) - normal.outer_product(normal)
        # FrictionPotential is a potential in relative *velocity*.  GeoTaichi's
        # nonlinear unknown is displacement, hence v = T^T du / h and the
        # incremental potential is h D(v).  ``energy_term`` contains that h.
        tangential_velocity = tangent.transpose() @ relative_displacement / timestep
        return mu_lambda * self.energy_term(tangential_velocity.norm(), timestep) * area

    @ti.func
    def gradient(
        self,
        relative_displacement,
        normal,
        mu_lambda,
        timestep,
        dvelocity_dvertex,
        area=1.0,
    ):
        tangent = ti.Matrix.identity(ti.f64, normal.n) - normal.outer_product(normal)
        tangential_velocity = tangent.transpose() @ relative_displacement / timestep
        speed = tangential_velocity.norm()
        # Gradient of h D(v) with respect to v.  The caller-provided Jacobian
        # maps its vertex/generalized variable to relative velocity.
        gradient_velocity = mu_lambda * timestep * self.grad_term(speed) * tangent @ tangential_velocity
        return gradient_velocity @ dvelocity_dvertex * area

    @ti.func
    def hessian(
        self,
        relative_displacement,
        normal,
        mu_lambda,
        timestep,
        dvelocity_dvertex_1,
        dvelocity_dvertex_2,
        area=1.0,
    ):
        tangent = ti.Matrix.identity(ti.f64, normal.n) - normal.outer_product(normal)
        tangential_velocity = tangent.transpose() @ relative_displacement / timestep
        speed = tangential_velocity.norm()
        f1_over_speed = self.grad_term(speed)
        inner = f1_over_speed * ti.Matrix.identity(ti.f64, normal.n)
        if speed > 0.0:
            inner += (self.hess_term(speed) / speed) * tangential_velocity.outer_product(tangential_velocity)
        hessian_velocity = mu_lambda * timestep * tangent @ inner @ tangent.transpose()
        return dvelocity_dvertex_1.transpose() @ hessian_velocity @ dvelocity_dvertex_2 * area


__all__ = [
    "normalize_ipc_model",
    "Barrier",
    "Friction",
    "ipc_toolkit_barrier_distance2_terms",
    "ipc_toolkit_barrier_distance_terms",
    "ipc_toolkit_barrier_distance2_offset_terms",
    "ipc_toolkit_barrier_distance_offset_terms",
    "ipc_barrier_distance_terms_py",
    "semi_ipc_terms_py",
    "semi_ipc_update_multiplier_py",
    "semi_ipc_terms",
    "semi_ipc_update_multiplier",
    "semi_ipc_hash_index",
    "semi_ipc_same_key",
    "semi_ipc_find_or_insert",
    "semi_ipc_find",
    "ipc_barrier_force_magnitude",
    "ipc_barrier_force_magnitude_gradient",
    "clamped_log_barrier_second_derivative",
    "initial_barrier_stiffness",
    "update_barrier_stiffness",
    "ipc_friction_f0",
    "ipc_friction_f1",
    "ipc_friction_f1_derivative",
    "ipc_friction_f1_over_speed",
    "ipc_friction_hessian_term",
    "ipc_fully_implicit_profile_over_speed",
    "ipc_fully_implicit_profile_over_speed_derivative",
    "ipc_fully_implicit_scalar_law",
    "ipc_stribeck_falloff",
    "ipc_stribeck_falloff_derivative",
    "ipc_fully_implicit_point_plane_friction",
    "ipc_fully_implicit_point_plane_stribeck_force",
    "ipc_fully_implicit_point_plane_stribeck_friction",
    "ipc_fully_implicit_scalar_law_py",
]

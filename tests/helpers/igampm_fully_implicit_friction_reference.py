"""NumPy oracle for Taichi point--NURBS friction verification tests.

This file is not a production backend.  The nonlinear unknown is the displacement of
the IGA control points and the active MPM grid nodes.  This module performs a
local, analytic first-order linearization of the paper's friction residual.
In particular, the closest NURBS parameter is differentiated with the
implicit-function theorem; it is not frozen and the production Jacobian is
not obtained by finite differences.
"""

from __future__ import annotations

import numpy as np

from src.physics_model.contact_model.ipc import (
    ipc_barrier_distance_terms_py,
    ipc_fully_implicit_scalar_law_py,
)


def newmark_endpoint_velocity_coefficients(integration, timestep):
    """Return ``(dq, v_n, a_n)`` coefficients for endpoint velocity."""
    integration = np.asarray(integration, dtype=np.float64).reshape(-1)
    timestep = float(timestep)
    if (
        integration.size < 3
        or not np.all(np.isfinite(integration[:3]))
        or integration[0] <= 0.0
        or integration[1] <= 0.0
        or not np.isfinite(timestep)
        or timestep <= 0.0
    ):
        raise ValueError(
            "fully implicit friction requires positive finite Newmark "
            "[alpha, beta, gamma] and timestep"
        )
    alpha, beta, gamma = integration[:3]
    displacement = 0.5 * gamma / (alpha * beta * timestep)
    previous_velocity = -(0.5 * gamma / (alpha * beta) - 1.0)
    previous_acceleration = -0.5 * timestep * (gamma / beta - 2.0)
    return float(displacement), float(previous_velocity), float(previous_acceleration)


def _parameter_derivative(
    residual,
    tangents,
    curvatures,
    direct_residual_jacobian,
    direct_tangent_jacobians,
    free_parameters,
):
    """Differentiate the constrained closest-point stationarity equations."""
    parameter_count = tangents.shape[1]
    dof_count = direct_residual_jacobian.shape[1]
    result = np.zeros((parameter_count, dof_count), dtype=np.float64)
    free = np.flatnonzero(np.asarray(free_parameters, dtype=bool))
    if free.size == 0:
        return result

    hessian = tangents.T @ tangents
    for first in range(parameter_count):
        for second in range(parameter_count):
            hessian[first, second] += float(
                np.dot(residual, curvatures[first, second])
            )

    right_hand_side = np.empty((parameter_count, dof_count), dtype=np.float64)
    for parameter in range(parameter_count):
        right_hand_side[parameter] = (
            tangents[:, parameter] @ direct_residual_jacobian
            + residual @ direct_tangent_jacobians[parameter]
        )

    reduced = hessian[np.ix_(free, free)]
    if not np.all(np.isfinite(reduced)):
        raise RuntimeError("NURBS closest-point Hessian is non-finite")
    scale = max(float(np.linalg.norm(reduced, ord=2)), 1.0)
    singular_values = np.linalg.svd(reduced, compute_uv=False)
    if singular_values.size == 0 or singular_values[-1] <= 1.0e-12 * scale:
        raise RuntimeError(
            "NURBS closest-point stationarity Hessian is singular; the "
            "fully implicit geometry derivative is not uniquely defined"
        )
    result[free] = np.linalg.solve(
        reduced, -right_hand_side[free]
    )
    return result


def point_nurbs_friction_residual_jacobian(
    *,
    control_positions,
    point_position,
    shape_values,
    shape_first_derivatives,
    shape_second_derivatives,
    control_dofs,
    point_dofs,
    point_weights,
    endpoint_velocity,
    endpoint_velocity_displacement_scale,
    free_parameters,
    area,
    dhat,
    dmin,
    kappa,
    use_physical_barrier,
    mu_dynamic,
    mu_static,
    mu_viscous,
    stribeck_velocity,
    epsv,
    friction_profile,
    need_jacobian=True,
):
    """Return local force residual and ``-d(residual)/dq``.

    ``control_dofs[i, d]`` and ``point_dofs[j, d]`` index a compact local
    displacement/velocity vector.  Repeated entries are supported, which is
    useful for coincident NURBS boundary control points.
    """
    control_positions = np.asarray(control_positions, dtype=np.float64)
    point_position = np.asarray(point_position, dtype=np.float64).reshape(-1)
    shape_values = np.asarray(shape_values, dtype=np.float64).reshape(-1)
    shape_first_derivatives = np.asarray(
        shape_first_derivatives, dtype=np.float64
    )
    shape_second_derivatives = np.asarray(
        shape_second_derivatives, dtype=np.float64
    )
    control_dofs = np.asarray(control_dofs, dtype=np.int64)
    point_dofs = np.asarray(point_dofs, dtype=np.int64)
    point_weights = np.asarray(point_weights, dtype=np.float64).reshape(-1)
    endpoint_velocity = np.asarray(endpoint_velocity, dtype=np.float64).reshape(-1)
    velocity_scale = np.asarray(
        endpoint_velocity_displacement_scale, dtype=np.float64
    ).reshape(-1)

    dimension = point_position.size
    support_size = shape_values.size
    parameter_count = shape_first_derivatives.shape[1]
    dof_count = endpoint_velocity.size
    if control_positions.shape != (support_size, dimension):
        raise ValueError("control_positions and NURBS shape values are inconsistent")
    if shape_first_derivatives.shape != (support_size, parameter_count):
        raise ValueError("invalid first NURBS shape derivative dimensions")
    if shape_second_derivatives.shape != (
        support_size,
        parameter_count,
        parameter_count,
    ):
        raise ValueError("invalid second NURBS shape derivative dimensions")
    if control_dofs.shape != (support_size, dimension):
        raise ValueError("control_dofs must have one vector block per support")
    if point_dofs.shape != (point_weights.size, dimension):
        raise ValueError("point_dofs and point_weights are inconsistent")
    if velocity_scale.size != dof_count:
        raise ValueError("endpoint velocity coefficients have the wrong size")
    if not (
        np.all(np.isfinite(control_positions))
        and np.all(np.isfinite(point_position))
        and np.all(np.isfinite(shape_values))
        and np.all(np.isfinite(shape_first_derivatives))
        and np.all(np.isfinite(shape_second_derivatives))
        and np.all(np.isfinite(point_weights))
        and np.all(np.isfinite(endpoint_velocity))
        and np.all(np.isfinite(velocity_scale))
    ):
        raise ValueError("fully implicit NURBS contact input must be finite")

    surface_position = shape_values @ control_positions
    residual = surface_position - point_position
    distance = float(np.linalg.norm(residual))
    if not np.isfinite(distance) or distance <= float(dmin):
        raise RuntimeError(
            "fully implicit NURBS friction requires distance strictly above dmin"
        )
    normal = residual / distance

    # B maps generalized endpoint velocities to point-minus-surface velocity.
    relative_map = np.zeros((dimension, dof_count), dtype=np.float64)
    for support in range(support_size):
        for component in range(dimension):
            relative_map[component, control_dofs[support, component]] -= (
                shape_values[support]
            )
    for point_node, weight in enumerate(point_weights):
        for component in range(dimension):
            relative_map[component, point_dofs[point_node, component]] += weight

    if not need_jacobian:
        # Residual-only Armijo probes do not need closest-parameter, normal,
        # tangent-projector, or normal-force derivatives.  Evaluate the same
        # constitutive force and return before constructing any dense local
        # Jacobian data.
        tangent_projector = np.eye(dimension) - np.outer(normal, normal)
        relative_velocity = relative_map @ endpoint_velocity
        tangential_velocity = tangent_projector @ relative_velocity
        speed = float(np.linalg.norm(tangential_velocity))
        _, barrier_gradient, _ = ipc_barrier_distance_terms_py(
            distance,
            dhat,
            dmin,
            kappa,
            use_physical_barrier,
        )
        normal_force = -float(area) * barrier_gradient
        if normal_force <= 0.0:
            return (
                np.zeros(dof_count, dtype=np.float64),
                None,
                {"distance": distance, "normal_force": 0.0, "speed": speed},
            )
        radial_factor, _, _ = ipc_fully_implicit_scalar_law_py(
            speed,
            normal_force,
            mu_dynamic,
            mu_static,
            mu_viscous,
            stribeck_velocity,
            epsv,
            friction_profile,
        )
        resistance = radial_factor * tangential_velocity
        force_residual = -np.einsum("ji,j->i", relative_map, resistance)
        if not np.all(np.isfinite(force_residual)):
            raise RuntimeError(
                "fully implicit NURBS friction produced non-finite values"
            )
        return force_residual, None, {
            "distance": distance,
            "normal_force": normal_force,
            "speed": speed,
        }

    direct_residual_jacobian = -relative_map
    tangents = control_positions.T @ shape_first_derivatives
    curvatures = np.empty(
        (parameter_count, parameter_count, dimension), dtype=np.float64
    )
    for first in range(parameter_count):
        for second in range(parameter_count):
            curvatures[first, second] = (
                shape_second_derivatives[:, first, second] @ control_positions
            )

    direct_tangent_jacobians = np.zeros(
        (parameter_count, dimension, dof_count), dtype=np.float64
    )
    for parameter in range(parameter_count):
        for support in range(support_size):
            derivative = shape_first_derivatives[support, parameter]
            for component in range(dimension):
                direct_tangent_jacobians[
                    parameter, component, control_dofs[support, component]
                ] += derivative

    parameter_jacobian = _parameter_derivative(
        residual,
        tangents,
        curvatures,
        direct_residual_jacobian,
        direct_tangent_jacobians,
        free_parameters,
    )
    residual_jacobian = (
        direct_residual_jacobian + tangents @ parameter_jacobian
    )
    distance_jacobian = normal @ residual_jacobian
    tangent_projector = np.eye(dimension) - np.outer(normal, normal)
    normal_jacobian = tangent_projector @ residual_jacobian / distance

    shape_jacobian = shape_first_derivatives @ parameter_jacobian
    relative_map_jacobian = np.zeros(
        (dimension, dof_count, dof_count), dtype=np.float64
    )
    for support in range(support_size):
        for component in range(dimension):
            relative_map_jacobian[
                component, control_dofs[support, component], :
            ] -= shape_jacobian[support]

    relative_velocity = relative_map @ endpoint_velocity
    relative_velocity_jacobian = (
        relative_map * velocity_scale[np.newaxis, :]
        + np.einsum(
            "ijk,j->ik", relative_map_jacobian, endpoint_velocity
        )
    )

    projector_jacobian = np.empty(
        (dimension, dimension, dof_count), dtype=np.float64
    )
    for column in range(dof_count):
        dn = normal_jacobian[:, column]
        projector_jacobian[:, :, column] = -(
            np.outer(dn, normal) + np.outer(normal, dn)
        )
    tangential_velocity = tangent_projector @ relative_velocity
    tangential_velocity_jacobian = tangent_projector @ relative_velocity_jacobian
    tangential_velocity_jacobian += np.einsum(
        "ijk,j->ik", projector_jacobian, relative_velocity
    )
    speed = float(np.linalg.norm(tangential_velocity))

    _, barrier_gradient, barrier_hessian = ipc_barrier_distance_terms_py(
        distance,
        dhat,
        dmin,
        kappa,
        use_physical_barrier,
    )
    normal_force = -float(area) * barrier_gradient
    if normal_force <= 0.0:
        return (
            np.zeros(dof_count, dtype=np.float64),
            np.zeros((dof_count, dof_count), dtype=np.float64),
            {"distance": distance, "normal_force": 0.0, "speed": speed},
        )
    normal_force_jacobian = (
        -float(area) * barrier_hessian * distance_jacobian
    )

    radial_factor, radial_speed_derivative, factor_per_normal_force = (
        ipc_fully_implicit_scalar_law_py(
            speed,
            normal_force,
            mu_dynamic,
            mu_static,
            mu_viscous,
            stribeck_velocity,
            epsv,
            friction_profile,
        )
    )
    velocity_jacobian = radial_factor * np.eye(dimension)
    if speed > 0.0:
        velocity_jacobian += (
            radial_speed_derivative / speed
        ) * np.outer(tangential_velocity, tangential_velocity)
    resistance = radial_factor * tangential_velocity
    resistance_jacobian = velocity_jacobian @ tangential_velocity_jacobian
    resistance_jacobian += np.outer(
        factor_per_normal_force * tangential_velocity,
        normal_force_jacobian,
    )

    force_residual = -np.einsum("ji,j->i", relative_map, resistance)
    negative_force_jacobian = np.einsum(
        "ji,jk->ik", relative_map, resistance_jacobian
    )
    for column in range(dof_count):
        negative_force_jacobian[:, column] += np.einsum(
            "ji,j->i",
            relative_map_jacobian[:, :, column],
            resistance,
        )

    if not (
        np.all(np.isfinite(force_residual))
        and np.all(np.isfinite(negative_force_jacobian))
    ):
        raise RuntimeError("fully implicit NURBS friction produced non-finite values")
    return force_residual, negative_force_jacobian, {
        "distance": distance,
        "normal_force": normal_force,
        "speed": speed,
        "parameter_jacobian": parameter_jacobian,
        "normal_jacobian": normal_jacobian,
    }


__all__ = [
    "newmark_endpoint_velocity_coefficients",
    "point_nurbs_friction_residual_jacobian",
]

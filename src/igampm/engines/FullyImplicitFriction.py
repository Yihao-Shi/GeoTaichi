"""Exact Taichi derivatives for fully implicit point--NURBS friction.

They keep the closest-coordinate, contact normal, tangent projector, normal
force, Stribeck law, and endpoint-velocity derivatives.  The returned matrix
blocks are therefore the complete generally nonsymmetric ``-dR/dq`` blocks;
no PSD projection, symmetrization, or frozen-geometry approximation is used.
"""

import taichi as ti

from src.physics_model.contact_model.ipc.IPC import (
    ipc_fully_implicit_scalar_law,
)


@ti.func
def curve_parameter_jacobian(
    residual,
    tangent,
    curvature,
    direct_weight,
    shape_derivative,
    is_control,
    parameter_is_free,
):
    """Return ``du/dq_block`` from the closest-curve stationarity IFT."""
    parameter_jacobian = ti.Vector.zero(ti.f64, residual.n)
    valid = 1
    if parameter_is_free != 0:
        stationarity_hessian = tangent.dot(tangent) + residual.dot(curvature)
        direct_stationarity = direct_weight * tangent
        if is_control != 0:
            direct_stationarity += shape_derivative * residual
        finite = 1
        if not (-ti.math.inf < stationarity_hessian and stationarity_hessian < ti.math.inf):
            finite = 0
        for component in ti.static(range(residual.n)):
            value = direct_stationarity[component]
            if not (-ti.math.inf < value and value < ti.math.inf):
                finite = 0
        scale = ti.max(ti.abs(stationarity_hessian), 1.0)
        if finite == 0 or ti.abs(stationarity_hessian) <= 1.0e-12 * scale:
            valid = 0
        else:
            parameter_jacobian = -direct_stationarity / stationarity_hessian
            for component in ti.static(range(residual.n)):
                value = parameter_jacobian[component]
                if not (-ti.math.inf < value and value < ti.math.inf):
                    valid = 0
    return parameter_jacobian, valid


@ti.func
def surface_parameter_jacobian(
    residual,
    tangent_u,
    tangent_v,
    curvature_uu,
    curvature_vv,
    curvature_uv,
    direct_weight,
    shape_derivative_u,
    shape_derivative_v,
    is_control,
    parameter_u_is_free,
    parameter_v_is_free,
):
    """Return the exact ``d(u,v)/dq_block`` constrained IFT derivative."""
    dimension = ti.static(residual.n)
    result = ti.Matrix.zero(ti.f64, 2, dimension)
    valid = 1

    a00 = tangent_u.dot(tangent_u) + residual.dot(curvature_uu)
    a01 = tangent_u.dot(tangent_v) + residual.dot(curvature_uv)
    a11 = tangent_v.dot(tangent_v) + residual.dot(curvature_vv)
    c0 = direct_weight * tangent_u
    c1 = direct_weight * tangent_v
    if is_control != 0:
        c0 += shape_derivative_u * residual
        c1 += shape_derivative_v * residual

    finite = 1
    if not (-ti.math.inf < a00 and a00 < ti.math.inf):
        finite = 0
    if not (-ti.math.inf < a01 and a01 < ti.math.inf):
        finite = 0
    if not (-ti.math.inf < a11 and a11 < ti.math.inf):
        finite = 0
    for component in ti.static(range(dimension)):
        value0 = c0[component]
        value1 = c1[component]
        if not (-ti.math.inf < value0 and value0 < ti.math.inf):
            finite = 0
        if not (-ti.math.inf < value1 and value1 < ti.math.inf):
            finite = 0

    if parameter_u_is_free != 0 and parameter_v_is_free != 0:
        matrix_scale = ti.max(ti.max(ti.abs(a00), ti.abs(a01)), ti.abs(a11))
        if finite == 0 or matrix_scale <= 0.5e-12:
            valid = 0
        else:
            scaled_a00 = a00 / matrix_scale
            scaled_a01 = a01 / matrix_scale
            scaled_a11 = a11 / matrix_scale
            trace = scaled_a00 + scaled_a11
            root = ti.sqrt(
                ti.max(
                    (scaled_a00 - scaled_a11) * (scaled_a00 - scaled_a11) + 4.0 * scaled_a01 * scaled_a01,
                    0.0,
                )
            )
            signed_root = root
            if trace < 0.0:
                signed_root = -root
            eigenvalue_big = 0.5 * (trace + signed_root)
            determinant = scaled_a00 * scaled_a11 - scaled_a01 * scaled_a01
            eigenvalue_small = 0.0
            if eigenvalue_big == 0.0:
                valid = 0
            else:
                eigenvalue_small = determinant / eigenvalue_big
                singular_max = ti.max(ti.abs(eigenvalue_big), ti.abs(eigenvalue_small))
                singular_min = ti.min(ti.abs(eigenvalue_big), ti.abs(eigenvalue_small))
                scaled_threshold = 1.0e-12 * ti.max(singular_max, 1.0 / matrix_scale)
                if (
                    not (
                        -ti.math.inf < eigenvalue_small
                        and eigenvalue_small < ti.math.inf
                        and -ti.math.inf < determinant
                        and determinant < ti.math.inf
                    )
                    or singular_min <= scaled_threshold
                ):
                    valid = 0
            if valid != 0:
                for component in ti.static(range(dimension)):
                    rhs0 = c0[component] / matrix_scale
                    rhs1 = c1[component] / matrix_scale
                    result[0, component] = -(scaled_a11 * rhs0 - scaled_a01 * rhs1) / determinant
                    result[1, component] = -(-scaled_a01 * rhs0 + scaled_a00 * rhs1) / determinant
    elif parameter_u_is_free != 0:
        scale = ti.max(ti.abs(a00), 1.0)
        if finite == 0 or ti.abs(a00) <= 1.0e-12 * scale:
            valid = 0
        else:
            for component in ti.static(range(dimension)):
                result[0, component] = -c0[component] / a00
    elif parameter_v_is_free != 0:
        scale = ti.max(ti.abs(a11), 1.0)
        if finite == 0 or ti.abs(a11) <= 1.0e-12 * scale:
            valid = 0
        else:
            for component in ti.static(range(dimension)):
                result[1, component] = -c1[component] / a11
    for row in ti.static(range(2)):
        for column in ti.static(range(dimension)):
            value = result[row, column]
            if not (-ti.math.inf < value and value < ti.math.inf):
                valid = 0
    return result, valid


@ti.func
def curve_geometry_jacobian(direct_weight, tangent, parameter_jacobian):
    dimension = ti.static(tangent.n)
    result = direct_weight * ti.Matrix.identity(ti.f64, dimension)
    result += tangent.outer_product(parameter_jacobian)
    return result


@ti.func
def surface_geometry_jacobian(direct_weight, tangent_u, tangent_v, parameter_jacobian):
    dimension = ti.static(tangent_u.n)
    result = direct_weight * ti.Matrix.identity(ti.f64, dimension)
    for row in ti.static(range(dimension)):
        for column in ti.static(range(dimension)):
            result[row, column] += (
                tangent_u[row] * parameter_jacobian[0, column] + tangent_v[row] * parameter_jacobian[1, column]
            )
    return result


@ti.func
def curve_relative_velocity_jacobian(
    relative_weight,
    endpoint_velocity_scale,
    surface_velocity_derivative,
    parameter_jacobian,
):
    dimension = ti.static(surface_velocity_derivative.n)
    result = relative_weight * endpoint_velocity_scale * ti.Matrix.identity(ti.f64, dimension)
    result -= surface_velocity_derivative.outer_product(parameter_jacobian)
    return result


@ti.func
def surface_relative_velocity_jacobian(
    relative_weight,
    endpoint_velocity_scale,
    surface_velocity_derivative_u,
    surface_velocity_derivative_v,
    parameter_jacobian,
):
    dimension = ti.static(surface_velocity_derivative_u.n)
    result = relative_weight * endpoint_velocity_scale * ti.Matrix.identity(ti.f64, dimension)
    for row in ti.static(range(dimension)):
        for column in ti.static(range(dimension)):
            result[row, column] -= (
                surface_velocity_derivative_u[row] * parameter_jacobian[0, column]
                + surface_velocity_derivative_v[row] * parameter_jacobian[1, column]
            )
    return result


@ti.func
def fully_implicit_resistance_state(
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
    """Return the exact radial resistance state needed by every block."""
    dimension = ti.static(normal.n)
    projector = ti.Matrix.identity(ti.f64, dimension) - normal.outer_product(normal)
    tangential_velocity = projector @ relative_velocity
    speed = tangential_velocity.norm()
    radial_factor, radial_speed_derivative, factor_per_normal_force = ipc_fully_implicit_scalar_law(
        speed,
        normal_force,
        mu_dynamic,
        mu_static,
        mu_viscous,
        stribeck_velocity,
        epsv,
        profile_id,
    )
    resistance = radial_factor * tangential_velocity
    velocity_jacobian = radial_factor * ti.Matrix.identity(ti.f64, dimension)
    if speed > 0.0:
        velocity_jacobian += (radial_speed_derivative / speed) * tangential_velocity.outer_product(tangential_velocity)
    return (
        projector,
        tangential_velocity,
        resistance,
        velocity_jacobian,
        factor_per_normal_force,
    )


@ti.func
def resistance_input_jacobian(
    residual_geometry_jacobian,
    relative_velocity_jacobian,
    relative_velocity,
    normal,
    distance,
    projector,
    tangential_velocity,
    velocity_jacobian,
    factor_per_normal_force,
    area,
    barrier_hessian,
):
    """Differentiate resistance with respect to one generalized block."""
    normal_jacobian = projector @ residual_geometry_jacobian / distance
    distance_jacobian = normal @ residual_geometry_jacobian
    normal_force_jacobian = -area * barrier_hessian * distance_jacobian
    tangential_velocity_jacobian = projector @ relative_velocity_jacobian
    tangential_velocity_jacobian -= normal.dot(relative_velocity) * normal_jacobian
    tangential_velocity_jacobian -= normal.outer_product(normal_jacobian.transpose() @ relative_velocity)
    result = velocity_jacobian @ tangential_velocity_jacobian
    result += (factor_per_normal_force * tangential_velocity).outer_product(normal_force_jacobian)
    return result


@ti.func
def curve_negative_force_jacobian_block(
    relative_output_weight,
    output_is_control,
    output_shape_derivative,
    input_parameter_jacobian,
    resistance,
    resistance_jacobian,
):
    """Return one complete curve-contact block of ``-dR/dq``."""
    result = relative_output_weight * resistance_jacobian
    if output_is_control != 0:
        shape_jacobian = output_shape_derivative * input_parameter_jacobian
        result -= resistance.outer_product(shape_jacobian)
    return result


@ti.func
def surface_negative_force_jacobian_block(
    relative_output_weight,
    output_is_control,
    output_shape_derivative_u,
    output_shape_derivative_v,
    input_parameter_jacobian,
    resistance,
    resistance_jacobian,
):
    """Return one complete surface-contact block of ``-dR/dq``."""
    result = relative_output_weight * resistance_jacobian
    if output_is_control != 0:
        dimension = ti.static(resistance.n)
        shape_jacobian = ti.Vector.zero(ti.f64, dimension)
        for component in ti.static(range(dimension)):
            shape_jacobian[component] = (
                output_shape_derivative_u * input_parameter_jacobian[0, component]
                + output_shape_derivative_v * input_parameter_jacobian[1, component]
            )
        result -= resistance.outer_product(shape_jacobian)
    return result


__all__ = [
    "curve_parameter_jacobian",
    "surface_parameter_jacobian",
    "curve_geometry_jacobian",
    "surface_geometry_jacobian",
    "curve_relative_velocity_jacobian",
    "surface_relative_velocity_jacobian",
    "fully_implicit_resistance_state",
    "resistance_input_jacobian",
    "curve_negative_force_jacobian_block",
    "surface_negative_force_jacobian_block",
]

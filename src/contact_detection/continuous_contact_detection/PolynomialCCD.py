"""Polynomial continuous-contact queries shared by implicit solvers."""

import taichi as ti

from src.utils.Root import (
    getSmallestPositiveRealQuadRoot,
)


@ti.func
def _signed_cuberoot(value):
    result = 0.0
    if value > 0.0:
        result = ti.pow(value, 1.0 / 3.0)
    elif value < 0.0:
        result = -ti.pow(-value, 1.0 / 3.0)
    return result


@ti.func
def _sort_three_roots(roots):
    if roots[1] < roots[0]:
        roots[0], roots[1] = roots[1], roots[0]
    if roots[2] < roots[1]:
        roots[1], roots[2] = roots[2], roots[1]
    if roots[1] < roots[0]:
        roots[0], roots[1] = roots[1], roots[0]
    return roots


@ti.func
def real_cubic_roots(cubic, quadratic, linear, constant):
    """Return every real root of a cubic in a sorted fixed-size vector.

    Missing entries use ``+inf``.  The implementation covers the one-, two-,
    and three-real-root branches and also degrades to a quadratic or linear
    equation.  This is used by analytic geometric CCD, which must inspect all
    coplanarity roots rather than only the smallest algebraic root.
    """

    roots = ti.Vector([ti.math.inf, ti.math.inf, ti.math.inf])
    coefficient_scale = ti.max(
        1.0,
        ti.max(
            ti.abs(cubic),
            ti.max(
                ti.abs(quadratic),
                ti.max(ti.abs(linear), ti.abs(constant)),
            ),
        ),
    )
    tolerance = 1.0e-14 * coefficient_scale
    if ti.abs(cubic) <= tolerance:
        if ti.abs(quadratic) <= tolerance:
            if ti.abs(linear) > tolerance:
                roots[0] = -constant / linear
        else:
            discriminant = linear * linear - 4.0 * quadratic * constant
            discriminant_tolerance = 1.0e-14 * ti.max(
                1.0,
                ti.abs(linear * linear) + ti.abs(4.0 * quadratic * constant),
            )
            if discriminant >= -discriminant_tolerance:
                square_root = ti.sqrt(ti.max(discriminant, 0.0))
                signed_square_root = square_root
                if linear < 0.0:
                    signed_square_root = -square_root
                q = -0.5 * (linear + signed_square_root)
                if ti.abs(q) > tolerance:
                    roots[0] = q / quadratic
                    roots[1] = constant / q
                else:
                    roots[0] = -linear / (2.0 * quadratic)
    else:
        normalized_quadratic = quadratic / cubic
        normalized_linear = linear / cubic
        normalized_constant = constant / cubic
        depressed_linear = normalized_linear - (normalized_quadratic * normalized_quadratic / 3.0)
        depressed_constant = (
            2.0 * normalized_quadratic**3 / 27.0 - normalized_quadratic * normalized_linear / 3.0 + normalized_constant
        )
        half_q = 0.5 * depressed_constant
        third_p = depressed_linear / 3.0
        discriminant = half_q * half_q + third_p**3
        discriminant_tolerance = 1.0e-14 * ti.max(1.0, ti.abs(half_q * half_q) + ti.abs(third_p**3))
        shift = normalized_quadratic / 3.0
        if discriminant > discriminant_tolerance:
            square_root = ti.sqrt(discriminant)
            roots[0] = _signed_cuberoot(-half_q + square_root) + _signed_cuberoot(-half_q - square_root) - shift
        elif discriminant >= -discriminant_tolerance:
            repeated_root = _signed_cuberoot(-half_q)
            roots[0] = 2.0 * repeated_root - shift
            roots[1] = -repeated_root - shift
        else:
            radius = 2.0 * ti.sqrt(ti.max(-third_p, 0.0))
            cosine_argument = -half_q / ti.sqrt(ti.max(-(third_p**3), 1.0e-300))
            cosine_argument = ti.max(-1.0, ti.min(1.0, cosine_argument))
            angle = ti.acos(cosine_argument) / 3.0
            roots[0] = radius * ti.cos(angle) - shift
            roots[1] = radius * ti.cos(angle - 2.0943951023931953) - shift
            roots[2] = radius * ti.cos(angle - 4.1887902047863905) - shift
    return _sort_three_roots(roots)


@ti.func
def point_point_quadratic_ccd(
    point0,
    point1,
    displacement0,
    displacement1,
    minimum_distance,
    slackness,
):
    """Return the first point-point distance TOI scaled by ``slackness``."""

    relative_position = point0 - point1
    relative_displacement = displacement0 - displacement1
    quadratic_a = relative_displacement.dot(relative_displacement)
    quadratic_b = 2.0 * relative_position.dot(relative_displacement)
    quadratic_c = relative_position.dot(relative_position) - minimum_distance * minimum_distance
    solution = getSmallestPositiveRealQuadRoot(quadratic_a, quadratic_b, quadratic_c)
    if solution < 0.0:
        solution = 1.0e20
    else:
        solution *= slackness
    return solution


@ti.func
def _matrix_column(matrix, column):
    return ti.Vector([matrix[row, column] for row in ti.static(range(matrix.m))])


@ti.func
def deformation_gradient_ccd(
    deformation_gradient,
    deformation_gradient_increment,
    slackness,
):
    """Return the first step that reaches the safeguarded determinant.

    The returned root solves
    ``det(F + alpha * dF) = (1 - slackness) * det(F)`` for 2D or 3D square
    deformation gradients.  A full step is returned when the determinant is
    provably safe over ``[0, 1]``; otherwise the first positive root is used.
    """

    cubic = 0.0
    quadratic = 0.0
    linear = 0.0
    constant = slackness * deformation_gradient.determinant()
    if ti.static(deformation_gradient.n == 3):
        current0 = _matrix_column(deformation_gradient, 0)
        current1 = _matrix_column(deformation_gradient, 1)
        current2 = _matrix_column(deformation_gradient, 2)
        increment0 = _matrix_column(deformation_gradient_increment, 0)
        increment1 = _matrix_column(deformation_gradient_increment, 1)
        increment2 = _matrix_column(deformation_gradient_increment, 2)
        linear = (
            ti.Matrix.cols([increment0, current1, current2]).determinant()
            + ti.Matrix.cols([current0, increment1, current2]).determinant()
            + ti.Matrix.cols([current0, current1, increment2]).determinant()
        )
        quadratic = (
            ti.Matrix.cols([increment0, increment1, current2]).determinant()
            + ti.Matrix.cols([increment0, current1, increment2]).determinant()
            + ti.Matrix.cols([current0, increment1, increment2]).determinant()
        )
        cubic = deformation_gradient_increment.determinant()
    else:
        current0 = _matrix_column(deformation_gradient, 0)
        current1 = _matrix_column(deformation_gradient, 1)
        increment0 = _matrix_column(deformation_gradient_increment, 0)
        increment1 = _matrix_column(deformation_gradient_increment, 1)
        linear = (
            increment0[0] * current1[1]
            - increment0[1] * current1[0]
            + current0[0] * increment1[1]
            - current0[1] * increment1[0]
        )
        quadratic = increment0[0] * increment1[1] - increment0[1] * increment1[0]

    coefficient_scale = ti.max(
        1.0,
        ti.abs(cubic) + ti.abs(quadratic) + ti.abs(linear) + ti.abs(constant),
    )
    value_tolerance = 1.0e-14 * coefficient_scale
    needs_root = constant <= value_tolerance
    variation_bound = ti.abs(cubic) + ti.abs(quadratic) + ti.abs(linear)
    if not needs_root and constant <= variation_bound + value_tolerance:
        end_value = cubic + quadratic + linear + constant
        if end_value <= value_tolerance:
            needs_root = True

        # With p(0) > 0, a root can lie in (0, 1) only when an endpoint or an
        # interior stationary point reaches zero.
        derivative_a = 3.0 * cubic
        derivative_b = 2.0 * quadratic
        derivative_c = linear
        derivative_scale = ti.max(
            1.0,
            ti.abs(derivative_a) + ti.abs(derivative_b) + ti.abs(derivative_c),
        )
        derivative_tolerance = 1.0e-14 * derivative_scale
        stationary = ti.Vector([ti.math.inf, ti.math.inf])
        if ti.abs(derivative_a) <= derivative_tolerance:
            if ti.abs(derivative_b) > derivative_tolerance:
                stationary[0] = -derivative_c / derivative_b
        else:
            discriminant = derivative_b * derivative_b - 4.0 * derivative_a * derivative_c
            discriminant_tolerance = 1.0e-14 * ti.max(
                1.0,
                ti.abs(derivative_b * derivative_b) + ti.abs(4.0 * derivative_a * derivative_c),
            )
            if discriminant >= -discriminant_tolerance:
                square_root = ti.sqrt(ti.max(discriminant, 0.0))
                signed_square_root = square_root
                if derivative_b < 0.0:
                    signed_square_root = -square_root
                q = -0.5 * (derivative_b + signed_square_root)
                if ti.abs(q) > derivative_tolerance:
                    stationary[0] = q / derivative_a
                    stationary[1] = derivative_c / q
                else:
                    stationary[0] = -derivative_b / (2.0 * derivative_a)
        for stationary_index in ti.static(range(2)):
            alpha = stationary[stationary_index]
            if alpha > 0.0 and alpha < 1.0:
                value = ((cubic * alpha + quadratic) * alpha + linear) * alpha + constant
                if value <= value_tolerance:
                    needs_root = True

    solution = 1.0
    if needs_root:
        roots = real_cubic_roots(cubic, quadratic, linear, constant)
        solution = 1.0e20
        for root_index in ti.static(range(3)):
            root = roots[root_index]
            if root > 0.0:
                solution = ti.min(solution, root)
    return solution

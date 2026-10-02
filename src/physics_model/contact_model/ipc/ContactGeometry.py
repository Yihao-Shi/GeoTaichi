"""Closest-point geometry and signed contact stencils shared by IPC engines.

The convention throughout this module is ``delta = source - target``.  Signed
weights therefore sum to zero and can be used directly for relative
displacements, friction gradients, and FEM/MPM/IGA matrix pullbacks.
"""

import taichi as ti


# Stable GeoTaichi feature ids used by ContactDistance's generated kernels.
# Callers should use these names instead of raw integers when sharing stencils
# with FEM.
PT_VERTEX0 = 0
PT_VERTEX1 = 1
PT_VERTEX2 = 2
PT_EDGE01 = 3
PT_EDGE12 = 4
PT_EDGE02 = 5
PT_FACE = 6

EE_A0_B0 = 0
EE_A0_B1 = 1
EE_A0_B = 2
EE_A1_B0 = 3
EE_A1_B1 = 4
EE_A1_B = 5
EE_A_B0 = 6
EE_A_B1 = 7
EE_A_B = 8

# Alternate IPC feature ordering. The explicit prefix prevents code from
# accidentally mixing these ids with the legacy GeoTaichi ordering above.
IPC_EE_A0_B0 = 0
IPC_EE_A0_B1 = 1
IPC_EE_A1_B0 = 2
IPC_EE_A1_B1 = 3
IPC_EE_A_B0 = 4
IPC_EE_A_B1 = 5
IPC_EE_A0_B = 6
IPC_EE_A1_B = 7
IPC_EE_A_B = 8


@ti.func
def legacy_ee_type_to_ipc(feature_type):
    official_type = feature_type
    if feature_type == EE_A0_B:
        official_type = IPC_EE_A0_B
    elif feature_type == EE_A1_B0:
        official_type = IPC_EE_A1_B0
    elif feature_type == EE_A1_B1:
        official_type = IPC_EE_A1_B1
    elif feature_type == EE_A1_B:
        official_type = IPC_EE_A1_B
    elif feature_type == EE_A_B0:
        official_type = IPC_EE_A_B0
    elif feature_type == EE_A_B1:
        official_type = IPC_EE_A_B1
    return official_type


@ti.func
def ipc_ee_type_to_legacy(feature_type):
    legacy_type = feature_type
    if feature_type == IPC_EE_A1_B0:
        legacy_type = EE_A1_B0
    elif feature_type == IPC_EE_A1_B1:
        legacy_type = EE_A1_B1
    elif feature_type == IPC_EE_A_B0:
        legacy_type = EE_A_B0
    elif feature_type == IPC_EE_A_B1:
        legacy_type = EE_A_B1
    elif feature_type == IPC_EE_A0_B:
        legacy_type = EE_A0_B
    elif feature_type == IPC_EE_A1_B:
        legacy_type = EE_A1_B
    return legacy_type


def legacy_ee_type_to_ipc_py(feature_type):
    """Host equivalent of :func:`legacy_ee_type_to_ipc`."""
    mapping = (0, 1, 6, 2, 3, 7, 4, 5, 8)
    feature_type = int(feature_type)
    if feature_type < 0 or feature_type >= len(mapping):
        raise ValueError("legacy EE feature type must be in [0, 8]")
    return mapping[feature_type]


def ipc_ee_type_to_legacy_py(feature_type):
    """Host equivalent of :func:`ipc_ee_type_to_legacy`."""
    mapping = (0, 1, 3, 4, 6, 7, 2, 5, 8)
    feature_type = int(feature_type)
    if feature_type < 0 or feature_type >= len(mapping):
        raise ValueError("IPC EE feature type must be in [0, 8]")
    return mapping[feature_type]


@ti.func
def edge_edge_parallel_distance_type_ipc(a0, a1, b0, b1):
    """Classify non-degenerate parallel edges in IPC feature ordering."""
    edge_a = a1 - a0
    inverse_length2 = 1.0 / edge_a.dot(edge_a)
    alpha = (b0 - a0).dot(edge_a) * inverse_length2
    beta = (b1 - a0).dot(edge_a) * inverse_length2
    feature_a = 2
    feature_b = 0
    if alpha < 0.0:
        feature_a = 2 if 0.0 <= beta and beta <= 1.0 else 0
        feature_b = 0 if beta <= alpha else (1 if beta <= 1.0 else 2)
    elif alpha > 1.0:
        feature_a = 2 if 0.0 <= beta and beta <= 1.0 else 1
        feature_b = 0 if beta >= alpha else (1 if 0.0 <= beta else 2)
    else:
        feature_a = 2
        feature_b = 0

    feature_type = IPC_EE_A_B0
    if feature_b < 2:
        feature_type = 2 * feature_a + feature_b
    else:
        feature_type = 6 + feature_a
    return feature_type


@ti.func
def edge_edge_distance_type_ipc(a0, a1, b0, b1):
    """Robust EE feature classifier in IPC feature ordering."""
    edge_a = a1 - a0
    edge_b = b1 - b0
    offset = a0 - b0
    aa = edge_a.dot(edge_a)
    ab = edge_a.dot(edge_b)
    bb = edge_b.dot(edge_b)
    ao = edge_a.dot(offset)
    bo = edge_b.dot(offset)
    denominator = aa * bb - ab * ab
    feature_type = IPC_EE_A_B

    if aa == 0.0 and bb == 0.0:
        feature_type = IPC_EE_A0_B0
    elif aa == 0.0:
        feature_type = IPC_EE_A0_B
    elif bb == 0.0:
        feature_type = IPC_EE_A_B0
    else:
        parallel_tolerance = 2.5e-16 * aa * bb
        if edge_a.cross(edge_b).norm_sqr() < parallel_tolerance:
            feature_type = edge_edge_parallel_distance_type_ipc(a0, a1, b0, b1)
        else:
            parameter_a_numerator = ab * bo - bb * ao
            parameter_b_numerator = 0.0
            parameter_b_denominator = denominator
            if parameter_a_numerator <= 0.0:
                parameter_b_numerator = bo
                parameter_b_denominator = bb
                feature_type = IPC_EE_A0_B
            elif parameter_a_numerator >= denominator:
                parameter_b_numerator = bo + ab
                parameter_b_denominator = bb
                feature_type = IPC_EE_A1_B
            else:
                parameter_b_numerator = aa * bo - ab * ao
                parameter_b_denominator = denominator
                feature_type = IPC_EE_A_B

            if parameter_b_numerator <= 0.0:
                if -ao <= 0.0:
                    feature_type = IPC_EE_A0_B0
                elif -ao >= aa:
                    feature_type = IPC_EE_A1_B0
                else:
                    feature_type = IPC_EE_A_B0
            elif parameter_b_numerator >= parameter_b_denominator:
                if -ao + ab <= 0.0:
                    feature_type = IPC_EE_A0_B1
                elif -ao + ab >= aa:
                    feature_type = IPC_EE_A1_B1
                else:
                    feature_type = IPC_EE_A_B1
    return feature_type


@ti.func
def edge_edge_distance_type_legacy(a0, a1, b0, b1):
    """Official classifier translated to ContactDistance's stable legacy ids."""
    return ipc_ee_type_to_legacy(edge_edge_distance_type_ipc(a0, a1, b0, b1))


@ti.func
def _clamp_unit(value):
    return ti.min(ti.max(value, 0.0), 1.0)


@ti.func
def _solve_symmetric_2x2(a00, a01, a11, b0, b1):
    determinant = a00 * a11 - a01 * a01
    x0 = 0.0
    x1 = 0.0
    if ti.abs(determinant) > 1.0e-30:
        x0 = (a11 * b0 - a01 * b1) / determinant
        x1 = (a00 * b1 - a01 * b0) / determinant
    return x0, x1


@ti.func
def closest_point_segment(point, endpoint0, endpoint1):
    edge = endpoint1 - endpoint0
    parameter = _clamp_unit(edge.dot(point - endpoint0) / ti.max(edge.dot(edge), 1.0e-30))
    return endpoint0 + parameter * edge, parameter


@ti.func
def closest_point_triangle(point, vertex0, vertex1, vertex2):
    """Ericson region-test closest point with barycentric coordinates."""
    edge01 = vertex1 - vertex0
    edge02 = vertex2 - vertex0
    point0 = point - vertex0
    d1 = edge01.dot(point0)
    d2 = edge02.dot(point0)
    closest = vertex0
    barycentric = ti.Vector([1.0, 0.0, 0.0])
    done = False
    if d1 <= 0.0 and d2 <= 0.0:
        done = True
    if not done:
        point1 = point - vertex1
        d3 = edge01.dot(point1)
        d4 = edge02.dot(point1)
        if d3 >= 0.0 and d4 <= d3:
            closest = vertex1
            barycentric = ti.Vector([0.0, 1.0, 0.0])
            done = True
        if not done:
            vc = d1 * d4 - d3 * d2
            if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                parameter = d1 / ti.max(d1 - d3, 1.0e-30)
                closest = vertex0 + parameter * edge01
                barycentric = ti.Vector([1.0 - parameter, parameter, 0.0])
                done = True
            if not done:
                point2 = point - vertex2
                d5 = edge01.dot(point2)
                d6 = edge02.dot(point2)
                if d6 >= 0.0 and d5 <= d6:
                    closest = vertex2
                    barycentric = ti.Vector([0.0, 0.0, 1.0])
                    done = True
                if not done:
                    vb = d5 * d2 - d1 * d6
                    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                        parameter = d2 / ti.max(d2 - d6, 1.0e-30)
                        closest = vertex0 + parameter * edge02
                        barycentric = ti.Vector([1.0 - parameter, 0.0, parameter])
                        done = True
                    if not done:
                        va = d3 * d6 - d5 * d4
                        if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
                            parameter = (d4 - d3) / ti.max(d4 - d3 + d5 - d6, 1.0e-30)
                            closest = vertex1 + parameter * (vertex2 - vertex1)
                            barycentric = ti.Vector([0.0, 1.0 - parameter, parameter])
                            done = True
                        if not done:
                            denominator = ti.max(va + vb + vc, 1.0e-30)
                            bary1 = vb / denominator
                            bary2 = vc / denominator
                            closest = vertex0 + bary1 * edge01 + bary2 * edge02
                            barycentric = ti.Vector([1.0 - bary1 - bary2, bary1, bary2])
    return closest, barycentric


@ti.func
def closest_points_segments(endpoint_a0, endpoint_a1, endpoint_b0, endpoint_b1):
    """Robust closest points on two finite segments, including degeneracy."""
    direction_a = endpoint_a1 - endpoint_a0
    direction_b = endpoint_b1 - endpoint_b0
    offset = endpoint_a0 - endpoint_b0
    aa = direction_a.dot(direction_a)
    ab = direction_a.dot(direction_b)
    bb = direction_b.dot(direction_b)
    ao = direction_a.dot(offset)
    bo = direction_b.dot(offset)
    denominator = aa * bb - ab * ab
    parameter_a = 0.0
    parameter_b = 0.0
    epsilon = 1.0e-30

    if aa <= epsilon and bb <= epsilon:
        parameter_a = 0.0
        parameter_b = 0.0
    elif aa <= epsilon:
        parameter_b = _clamp_unit(bo / ti.max(bb, epsilon))
    elif bb <= epsilon:
        parameter_a = _clamp_unit(-ao / ti.max(aa, epsilon))
    else:
        if denominator > epsilon:
            parameter_a = _clamp_unit((ab * bo - bb * ao) / denominator)
        projected_b = ab * parameter_a + bo
        if projected_b < 0.0:
            parameter_b = 0.0
            parameter_a = _clamp_unit(-ao / ti.max(aa, epsilon))
        elif projected_b > bb:
            parameter_b = 1.0
            parameter_a = _clamp_unit((ab - ao) / ti.max(aa, epsilon))
        else:
            parameter_b = projected_b / ti.max(bb, epsilon)

    closest_a = endpoint_a0 + parameter_a * direction_a
    closest_b = endpoint_b0 + parameter_b * direction_b
    return closest_a, closest_b, parameter_a, parameter_b


@ti.func
def point_triangle_coordinates_from_type(point, vertex0, vertex1, vertex2, feature_type):
    """Coordinates matching ContactDistance's PT feature numbering 0..6."""
    bary1 = 0.0
    bary2 = 0.0
    if feature_type == 1:
        bary1 = 1.0
    elif feature_type == 2:
        bary2 = 1.0
    elif feature_type == 3:
        edge = vertex1 - vertex0
        bary1 = _clamp_unit((point - vertex0).dot(edge) / ti.max(edge.dot(edge), 1.0e-30))
    elif feature_type == 4:
        edge = vertex2 - vertex1
        bary2 = _clamp_unit((point - vertex1).dot(edge) / ti.max(edge.dot(edge), 1.0e-30))
        bary1 = 1.0 - bary2
    elif feature_type == 5:
        edge = vertex2 - vertex0
        bary2 = _clamp_unit((point - vertex0).dot(edge) / ti.max(edge.dot(edge), 1.0e-30))
    elif feature_type == 6:
        edge1 = vertex1 - vertex0
        edge2 = vertex2 - vertex0
        offset = point - vertex0
        bary1, bary2 = _solve_symmetric_2x2(
            edge1.dot(edge1),
            edge1.dot(edge2),
            edge2.dot(edge2),
            edge1.dot(offset),
            edge2.dot(offset),
        )
    return bary1, bary2


@ti.func
def edge_edge_coordinates_from_type(endpoint_a0, endpoint_a1, endpoint_b0, endpoint_b1, feature_type):
    """Coordinates matching ContactDistance's EE feature numbering 0..8."""
    parameter_a = 0.0
    parameter_b = 0.0
    if feature_type == 1:
        parameter_b = 1.0
    elif feature_type == 2:
        edge = endpoint_b1 - endpoint_b0
        parameter_b = _clamp_unit((endpoint_a0 - endpoint_b0).dot(edge) / ti.max(edge.dot(edge), 1.0e-30))
    elif feature_type == 3:
        parameter_a = 1.0
    elif feature_type == 4:
        parameter_a = 1.0
        parameter_b = 1.0
    elif feature_type == 5:
        edge = endpoint_b1 - endpoint_b0
        parameter_a = 1.0
        parameter_b = _clamp_unit((endpoint_a1 - endpoint_b0).dot(edge) / ti.max(edge.dot(edge), 1.0e-30))
    elif feature_type == 6:
        edge = endpoint_a1 - endpoint_a0
        parameter_a = _clamp_unit((endpoint_b0 - endpoint_a0).dot(edge) / ti.max(edge.dot(edge), 1.0e-30))
    elif feature_type == 7:
        edge = endpoint_a1 - endpoint_a0
        parameter_a = _clamp_unit((endpoint_b1 - endpoint_a0).dot(edge) / ti.max(edge.dot(edge), 1.0e-30))
        parameter_b = 1.0
    elif feature_type == 8:
        offset = endpoint_a0 - endpoint_b0
        edge_a = endpoint_a1 - endpoint_a0
        edge_b = endpoint_b1 - endpoint_b0
        parameter_a, parameter_b = _solve_symmetric_2x2(
            edge_a.dot(edge_a),
            -edge_a.dot(edge_b),
            edge_b.dot(edge_b),
            -offset.dot(edge_a),
            offset.dot(edge_b),
        )
    return parameter_a, parameter_b


@ti.func
def normalized_contact_direction(delta, fallback):
    normal = ti.Vector.zero(float, delta.n)
    normal[0] = 1.0
    norm = delta.norm()
    if norm > 1.0e-12:
        normal = delta / norm
    else:
        fallback_norm = fallback.norm()
        if fallback_norm > 1.0e-12:
            normal = fallback / fallback_norm
    return normal


@ti.func
def point_triangle_contact_frame(point, vertex0, vertex1, vertex2, feature_type):
    bary1, bary2 = point_triangle_coordinates_from_type(point, vertex0, vertex1, vertex2, feature_type)
    barycentric = ti.Vector([1.0 - bary1 - bary2, bary1, bary2])
    closest = barycentric[0] * vertex0 + barycentric[1] * vertex1 + barycentric[2] * vertex2
    fallback = (vertex1 - vertex0).cross(vertex2 - vertex0)
    normal = normalized_contact_direction(point - closest, fallback)
    return closest, barycentric, normal


@ti.func
def edge_edge_contact_frame(endpoint_a0, endpoint_a1, endpoint_b0, endpoint_b1, feature_type):
    parameter_a, parameter_b = edge_edge_coordinates_from_type(
        endpoint_a0, endpoint_a1, endpoint_b0, endpoint_b1, feature_type
    )
    closest_a = (1.0 - parameter_a) * endpoint_a0 + parameter_a * endpoint_a1
    closest_b = (1.0 - parameter_b) * endpoint_b0 + parameter_b * endpoint_b1
    fallback = (endpoint_a1 - endpoint_a0).cross(endpoint_b1 - endpoint_b0)
    normal = normalized_contact_direction(closest_a - closest_b, fallback)
    return closest_a, closest_b, parameter_a, parameter_b, normal


@ti.func
def point_point_stencil_weights():
    return ti.Vector([1.0, -1.0, 0.0, 0.0])


@ti.func
def point_edge_stencil_weights(edge_parameter):
    return ti.Vector([1.0, -(1.0 - edge_parameter), -edge_parameter, 0.0])


@ti.func
def point_triangle_stencil_weights(barycentric):
    return ti.Vector([1.0, -barycentric[0], -barycentric[1], -barycentric[2]])


@ti.func
def edge_edge_stencil_weights(parameter_a, parameter_b):
    return ti.Vector([1.0 - parameter_a, parameter_a, -(1.0 - parameter_b), -parameter_b])


# -----------------------------------------------------------------------------
# Unconstrained closest-point coordinates used by IPC tangent/friction terms.
#
# These functions deliberately differ from the finite/clamped closest-point
# routines above:
# PE returns the coordinate on the supporting line, PT returns coordinates on
# the supporting plane, and EE returns coordinates on the two supporting
# lines.  The derivatives are ordered by endpoint coordinates, e.g.
# ``[p, t0, t1, t2]`` for PT and ``[ea0, ea1, eb0, eb1]`` for EE.
# Degenerate primitives are outside the domain of these rational functions.


@ti.func
def _affine_dot_terms3(value0, value1, value2, coefficients_u, coefficients_v):
    """Value, gradient, and Hessian of a dot product of affine vectors."""
    dimension = ti.static(value0.n)
    # Form the affine vectors directly. Taichi 1.7 can read a local matrix
    # through runtime row/column indices, but a subsequent statically lowered
    # ``dot`` can observe the pre-loop values of that matrix on CPU/GPU. This
    # made the scalar value ignore non-x components while its hand-written
    # derivative used all components. Explicit site expressions are both the
    # exact IPC definition and independent of that local-tensor SSA hazard.
    vector_u = coefficients_u[0] * value0 + coefficients_u[1] * value1 + coefficients_u[2] * value2
    vector_v = coefficients_v[0] * value0 + coefficients_v[1] * value1 + coefficients_v[2] * value2

    value = vector_u.dot(vector_v)
    gradient = ti.Vector.zero(float, 3 * dimension)
    hessian = ti.Matrix.zero(float, 3 * dimension, 3 * dimension)
    for site_i in range(3):
        for component in range(dimension):
            row = site_i * dimension + component
            gradient[row] = coefficients_u[site_i] * vector_v[component] + coefficients_v[site_i] * vector_u[component]
            for site_j in range(3):
                column = site_j * dimension + component
                hessian[row, column] = (
                    coefficients_u[site_i] * coefficients_v[site_j] + coefficients_v[site_i] * coefficients_u[site_j]
                )
    return value, gradient, hessian


@ti.func
def _affine_dot_terms4(value0, value1, value2, value3, coefficients_u, coefficients_v):
    """3D four-site specialization of :func:`_affine_dot_terms3`."""
    vector_u = (
        coefficients_u[0] * value0
        + coefficients_u[1] * value1
        + coefficients_u[2] * value2
        + coefficients_u[3] * value3
    )
    vector_v = (
        coefficients_v[0] * value0
        + coefficients_v[1] * value1
        + coefficients_v[2] * value2
        + coefficients_v[3] * value3
    )

    value = vector_u.dot(vector_v)
    gradient = ti.Vector.zero(float, 12)
    hessian = ti.Matrix.zero(float, 12, 12)
    for site_i in range(4):
        for component in range(3):
            row = 3 * site_i + component
            gradient[row] = coefficients_u[site_i] * vector_v[component] + coefficients_v[site_i] * vector_u[component]
            for site_j in range(4):
                column = 3 * site_j + component
                hessian[row, column] = (
                    coefficients_u[site_i] * coefficients_v[site_j] + coefficients_v[site_i] * coefficients_u[site_j]
                )
    return value, gradient, hessian


@ti.func
def _quotient_terms(
    numerator,
    numerator_gradient,
    numerator_hessian,
    denominator,
    denominator_gradient,
    denominator_hessian,
):
    """Second-order quotient rule for scalar functions."""
    inv_denominator = 1.0 / denominator
    inv_denominator2 = inv_denominator * inv_denominator
    inv_denominator3 = inv_denominator2 * inv_denominator
    value = numerator * inv_denominator
    gradient = inv_denominator * numerator_gradient - numerator * inv_denominator2 * denominator_gradient
    hessian = inv_denominator * numerator_hessian
    hessian -= inv_denominator2 * (
        numerator_gradient.outer_product(denominator_gradient)
        + denominator_gradient.outer_product(numerator_gradient)
        + numerator * denominator_hessian
    )
    hessian += 2.0 * numerator * inv_denominator3 * denominator_gradient.outer_product(denominator_gradient)
    return value, gradient, hessian


@ti.func
def point_edge_closest_point_terms(point, endpoint0, endpoint1):
    """Unconstrained PE coordinate and its first two derivatives.

    The coordinate ``alpha`` satisfies ``q = e0 + alpha * (e1 - e0)`` and is
    not clamped to ``[0, 1]``.  The derivative ordering is ``[p, e0, e1]``.
    This supports both 2D and 3D vectors.
    """
    offset_coefficients = ti.Vector([1.0, -1.0, 0.0])
    edge_coefficients = ti.Vector([0.0, -1.0, 1.0])
    numerator, numerator_gradient, numerator_hessian = _affine_dot_terms3(
        point,
        endpoint0,
        endpoint1,
        offset_coefficients,
        edge_coefficients,
    )
    denominator, denominator_gradient, denominator_hessian = _affine_dot_terms3(
        point,
        endpoint0,
        endpoint1,
        edge_coefficients,
        edge_coefficients,
    )
    return _quotient_terms(
        numerator,
        numerator_gradient,
        numerator_hessian,
        denominator,
        denominator_gradient,
        denominator_hessian,
    )


@ti.func
def point_edge_closest_point(point, endpoint0, endpoint1):
    coordinate, _, _ = point_edge_closest_point_terms(point, endpoint0, endpoint1)
    return coordinate


@ti.func
def point_edge_closest_point_jacobian(point, endpoint0, endpoint1):
    _, jacobian, _ = point_edge_closest_point_terms(point, endpoint0, endpoint1)
    return jacobian


@ti.func
def point_edge_closest_point_hessian(point, endpoint0, endpoint1):
    _, _, hessian = point_edge_closest_point_terms(point, endpoint0, endpoint1)
    return hessian


@ti.func
def _symmetric_2x2_solution_terms(
    matrix00,
    matrix00_gradient,
    matrix00_hessian,
    matrix01,
    matrix01_gradient,
    matrix01_hessian,
    matrix11,
    matrix11_gradient,
    matrix11_hessian,
    rhs0,
    rhs0_gradient,
    rhs0_hessian,
    rhs1,
    rhs1_gradient,
    rhs1_hessian,
):
    """Differentiate ``A(x) c(x) = b(x)`` through second order."""
    determinant = matrix00 * matrix11 - matrix01 * matrix01
    inverse = ti.Matrix([[matrix11, -matrix01], [-matrix01, matrix00]]) / determinant
    coordinates = inverse @ ti.Vector([rhs0, rhs1])
    jacobian = ti.Matrix.zero(float, 2, 12)
    hessian0 = ti.Matrix.zero(float, 12, 12)
    hessian1 = ti.Matrix.zero(float, 12, 12)

    for variable in range(12):
        matrix_derivative = ti.Matrix(
            [
                [matrix00_gradient[variable], matrix01_gradient[variable]],
                [matrix01_gradient[variable], matrix11_gradient[variable]],
            ]
        )
        rhs_derivative = ti.Vector([rhs0_gradient[variable], rhs1_gradient[variable]])
        coordinate_derivative = inverse @ (rhs_derivative - matrix_derivative @ coordinates)
        jacobian[0, variable] = coordinate_derivative[0]
        jacobian[1, variable] = coordinate_derivative[1]

    for variable_i in range(12):
        matrix_derivative_i = ti.Matrix(
            [
                [matrix00_gradient[variable_i], matrix01_gradient[variable_i]],
                [matrix01_gradient[variable_i], matrix11_gradient[variable_i]],
            ]
        )
        derivative_i = ti.Vector([jacobian[0, variable_i], jacobian[1, variable_i]])
        for variable_j in range(12):
            matrix_derivative_j = ti.Matrix(
                [
                    [
                        matrix00_gradient[variable_j],
                        matrix01_gradient[variable_j],
                    ],
                    [
                        matrix01_gradient[variable_j],
                        matrix11_gradient[variable_j],
                    ],
                ]
            )
            matrix_second_derivative = ti.Matrix(
                [
                    [
                        matrix00_hessian[variable_i, variable_j],
                        matrix01_hessian[variable_i, variable_j],
                    ],
                    [
                        matrix01_hessian[variable_i, variable_j],
                        matrix11_hessian[variable_i, variable_j],
                    ],
                ]
            )
            rhs_second_derivative = ti.Vector(
                [
                    rhs0_hessian[variable_i, variable_j],
                    rhs1_hessian[variable_i, variable_j],
                ]
            )
            derivative_j = ti.Vector([jacobian[0, variable_j], jacobian[1, variable_j]])
            coordinate_second_derivative = inverse @ (
                rhs_second_derivative
                - matrix_second_derivative @ coordinates
                - matrix_derivative_i @ derivative_j
                - matrix_derivative_j @ derivative_i
            )
            hessian0[variable_i, variable_j] = coordinate_second_derivative[0]
            hessian1[variable_i, variable_j] = coordinate_second_derivative[1]
    return coordinates, jacobian, hessian0, hessian1


@ti.func
def point_triangle_closest_point_terms(point, vertex0, vertex1, vertex2):
    """Unconstrained PT plane coordinates and their derivatives.

    Returns ``(u, v)`` such that the closest point on the supporting plane is
    ``t0 + u * (t1 - t0) + v * (t2 - t0)``.  Derivatives are ordered as
    ``[p, t0, t1, t2]``.
    """
    edge01 = ti.Vector([0.0, -1.0, 1.0, 0.0])
    edge02 = ti.Vector([0.0, -1.0, 0.0, 1.0])
    offset = ti.Vector([1.0, -1.0, 0.0, 0.0])
    matrix00, matrix00_gradient, matrix00_hessian = _affine_dot_terms4(point, vertex0, vertex1, vertex2, edge01, edge01)
    matrix01, matrix01_gradient, matrix01_hessian = _affine_dot_terms4(point, vertex0, vertex1, vertex2, edge01, edge02)
    matrix11, matrix11_gradient, matrix11_hessian = _affine_dot_terms4(point, vertex0, vertex1, vertex2, edge02, edge02)
    rhs0, rhs0_gradient, rhs0_hessian = _affine_dot_terms4(point, vertex0, vertex1, vertex2, edge01, offset)
    rhs1, rhs1_gradient, rhs1_hessian = _affine_dot_terms4(point, vertex0, vertex1, vertex2, edge02, offset)
    return _symmetric_2x2_solution_terms(
        matrix00,
        matrix00_gradient,
        matrix00_hessian,
        matrix01,
        matrix01_gradient,
        matrix01_hessian,
        matrix11,
        matrix11_gradient,
        matrix11_hessian,
        rhs0,
        rhs0_gradient,
        rhs0_hessian,
        rhs1,
        rhs1_gradient,
        rhs1_hessian,
    )


@ti.func
def point_triangle_closest_point(point, vertex0, vertex1, vertex2):
    coordinates, _, _, _ = point_triangle_closest_point_terms(point, vertex0, vertex1, vertex2)
    return coordinates


@ti.func
def point_triangle_closest_point_jacobian(point, vertex0, vertex1, vertex2):
    _, jacobian, _, _ = point_triangle_closest_point_terms(point, vertex0, vertex1, vertex2)
    return jacobian


@ti.func
def point_triangle_closest_point_hessian(point, vertex0, vertex1, vertex2):
    _, _, hessian0, hessian1 = point_triangle_closest_point_terms(point, vertex0, vertex1, vertex2)
    return hessian0, hessian1


@ti.func
def edge_edge_closest_point_terms(endpoint_a0, endpoint_a1, endpoint_b0, endpoint_b1):
    """Unconstrained EE line coordinates and their first two derivatives.

    The returned ``(s, t)`` minimizes the distance between
    ``ea0 + s * (ea1-ea0)`` and ``eb0 + t * (eb1-eb0)``.  Derivatives are
    ordered as ``[ea0, ea1, eb0, eb1]``.
    """
    edge_a = ti.Vector([-1.0, 1.0, 0.0, 0.0])
    edge_b = ti.Vector([0.0, 0.0, -1.0, 1.0])
    offset = ti.Vector([1.0, 0.0, -1.0, 0.0])
    matrix00, matrix00_gradient, matrix00_hessian = _affine_dot_terms4(
        endpoint_a0, endpoint_a1, endpoint_b0, endpoint_b1, edge_a, edge_a
    )
    negative_edge_b = -edge_b
    matrix01, matrix01_gradient, matrix01_hessian = _affine_dot_terms4(
        endpoint_a0,
        endpoint_a1,
        endpoint_b0,
        endpoint_b1,
        edge_a,
        negative_edge_b,
    )
    matrix11, matrix11_gradient, matrix11_hessian = _affine_dot_terms4(
        endpoint_a0, endpoint_a1, endpoint_b0, endpoint_b1, edge_b, edge_b
    )
    negative_offset = -offset
    rhs0, rhs0_gradient, rhs0_hessian = _affine_dot_terms4(
        endpoint_a0,
        endpoint_a1,
        endpoint_b0,
        endpoint_b1,
        negative_offset,
        edge_a,
    )
    rhs1, rhs1_gradient, rhs1_hessian = _affine_dot_terms4(
        endpoint_a0,
        endpoint_a1,
        endpoint_b0,
        endpoint_b1,
        offset,
        edge_b,
    )
    return _symmetric_2x2_solution_terms(
        matrix00,
        matrix00_gradient,
        matrix00_hessian,
        matrix01,
        matrix01_gradient,
        matrix01_hessian,
        matrix11,
        matrix11_gradient,
        matrix11_hessian,
        rhs0,
        rhs0_gradient,
        rhs0_hessian,
        rhs1,
        rhs1_gradient,
        rhs1_hessian,
    )


@ti.func
def edge_edge_closest_point(endpoint_a0, endpoint_a1, endpoint_b0, endpoint_b1):
    coordinates, _, _, _ = edge_edge_closest_point_terms(endpoint_a0, endpoint_a1, endpoint_b0, endpoint_b1)
    return coordinates


@ti.func
def edge_edge_closest_point_jacobian(endpoint_a0, endpoint_a1, endpoint_b0, endpoint_b1):
    _, jacobian, _, _ = edge_edge_closest_point_terms(endpoint_a0, endpoint_a1, endpoint_b0, endpoint_b1)
    return jacobian


@ti.func
def edge_edge_closest_point_hessian(endpoint_a0, endpoint_a1, endpoint_b0, endpoint_b1):
    _, _, hessian0, hessian1 = edge_edge_closest_point_terms(endpoint_a0, endpoint_a1, endpoint_b0, endpoint_b1)
    return hessian0, hessian1

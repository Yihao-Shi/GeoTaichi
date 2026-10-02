"""Analytic zero-thickness continuous collision detection (CCD).

Point--point CCD solves the squared-distance quadratic.  Point--triangle and
edge--edge CCD solve the cubic coplanarity polynomial and then validate the
primitive coordinates at every real root in ``[0, 1]``.  Configurations whose
coplanarity polynomial vanishes identically are handled by the shared ACCD
distance fallback because an in-plane impact is not isolated by that cubic.
"""

import taichi as ti

from .AdditiveCCD import edge_edge_accd, point_edge_accd, point_triangle_accd
from .PolynomialCCD import real_cubic_roots
from src.physics_model.contact_model.ipc.ContactGeometry import (
    closest_point_segment,
    closest_point_triangle,
    closest_points_segments,
)


@ti.func
def point_edge_ccd(
    point0,
    endpoint00,
    endpoint10,
    point_displacement,
    endpoint0_displacement,
    endpoint1_displacement,
    gap_fraction,
    max_iteration,
):
    """Analytic zero-thickness CCD for a moving 2D point and segment.

    Collinearity is quadratic in time.  Each real root is checked against the
    finite segment; an identically collinear trajectory falls back to the
    distance-based conservative loop.
    """
    relative0 = point0 - endpoint00
    edge0 = endpoint10 - endpoint00
    relative_displacement = point_displacement - endpoint0_displacement
    edge_displacement = endpoint1_displacement - endpoint0_displacement
    constant = relative0[0] * edge0[1] - relative0[1] * edge0[0]
    linear = (
        relative_displacement[0] * edge0[1]
        - relative_displacement[1] * edge0[0]
        + relative0[0] * edge_displacement[1]
        - relative0[1] * edge_displacement[0]
    )
    quadratic = relative_displacement[0] * edge_displacement[1] - relative_displacement[1] * edge_displacement[0]
    scale = ti.max(1.0, ti.max(relative0.norm(), edge0.norm()))
    coefficient_norm = ti.max(ti.abs(quadratic), ti.max(ti.abs(linear), ti.abs(constant)))
    closest0, _ = closest_point_segment(point0, endpoint00, endpoint10)
    initial_distance2 = (point0 - closest0).dot(point0 - closest0)
    geometric_tolerance = 1.0e-10 * scale
    toi = 1.0
    if initial_distance2 <= geometric_tolerance * geometric_tolerance:
        toi = 0.0
    elif coefficient_norm <= 1.0e-13 * scale * scale:
        toi = point_edge_accd(
            point0,
            endpoint00,
            endpoint10,
            point_displacement,
            endpoint0_displacement,
            endpoint1_displacement,
            gap_fraction,
            0.0,
            max_iteration,
        )
    else:
        roots = real_cubic_roots(0.0, quadratic, linear, constant)
        first_root = ti.math.inf
        for root_id in ti.static(range(3)):
            root = roots[root_id]
            if root >= -1.0e-10 and root <= 1.0 + 1.0e-10:
                root = ti.max(0.0, ti.min(1.0, root))
                point = point0 + root * point_displacement
                endpoint0 = endpoint00 + root * endpoint0_displacement
                endpoint1 = endpoint10 + root * endpoint1_displacement
                closest, parameter = closest_point_segment(point, endpoint0, endpoint1)
                distance2 = (point - closest).dot(point - closest)
                if (
                    parameter >= -1.0e-10
                    and parameter <= 1.0 + 1.0e-10
                    and distance2 <= geometric_tolerance * geometric_tolerance
                    and root < first_root
                ):
                    first_root = root
        if first_root < ti.math.inf:
            toi = (1.0 - gap_fraction) * first_root
    return ti.max(0.0, ti.min(1.0, toi))


def ccd_mode_parameters(ccd_type, eta, accd_tolerance):
    """Normalize the requested CCD mode and finite ACCD clearance.

    ``eta`` is the retained fraction of the initial excess distance. ACCD
    defaults to at most ten percent.
    """

    mode = str(ccd_type).strip().lower()
    eta = float(eta)
    thickness = 0.0
    if mode == "accd":
        eta = min(eta, 0.1)
        thickness = float(accd_tolerance)
    return mode, eta, thickness


@ti.func
def linear_gap_ccd(gap, relative_displacement, slackness):
    """Analytic CCD for a linearly changing signed gap."""

    toi = 1.0
    if gap <= 0.0:
        toi = 0.0
    elif relative_displacement < 0.0:
        toi = slackness * gap / (-relative_displacement)
    return ti.max(0.0, ti.min(1.0, toi))


@ti.func
def _point_triangle_distance_squared(point, vertex0, vertex1, vertex2):
    closest, _ = closest_point_triangle(point, vertex0, vertex1, vertex2)
    delta = point - closest
    return ti.max(delta.dot(delta), 0.0)


@ti.func
def _edge_edge_distance_squared(endpoint_a0, endpoint_a1, endpoint_b0, endpoint_b1):
    closest_a, closest_b, _, _ = closest_points_segments(endpoint_a0, endpoint_a1, endpoint_b0, endpoint_b1)
    delta = closest_a - closest_b
    return ti.max(delta.dot(delta), 0.0)


@ti.func
def _length_scale4(point0, point1, point2, point3):
    scale = 1.0
    scale = ti.max(scale, (point0 - point1).norm())
    scale = ti.max(scale, (point0 - point2).norm())
    scale = ti.max(scale, (point0 - point3).norm())
    scale = ti.max(scale, (point1 - point2).norm())
    scale = ti.max(scale, (point1 - point3).norm())
    scale = ti.max(scale, (point2 - point3).norm())
    return scale


@ti.func
def _coplanarity_coefficients(origin0, axis10, axis20, dorigin, daxis1, daxis2):
    cross0 = axis10.cross(axis20)
    cross1 = axis10.cross(daxis2) + daxis1.cross(axis20)
    cross2 = daxis1.cross(daxis2)
    constant = origin0.dot(cross0)
    linear = origin0.dot(cross1) + dorigin.dot(cross0)
    quadratic = origin0.dot(cross2) + dorigin.dot(cross1)
    cubic = dorigin.dot(cross2)
    return cubic, quadratic, linear, constant


@ti.func
def point_point_ccd(
    point0,
    point1,
    displacement0,
    displacement1,
    gap_fraction,
    max_iteration,
):
    """Analytic zero-thickness point--point CCD.

    ``max_iteration`` is accepted for a uniform primitive interface and is not
    used by this quadratic query.
    """

    relative_position = point0 - point1
    relative_displacement = displacement0 - displacement1
    initial_distance_squared = relative_position.dot(relative_position)
    scale = ti.max(1.0, relative_position.norm())
    geometric_tolerance = 1.0e-10 * scale
    toi = 1.0
    if initial_distance_squared <= geometric_tolerance * geometric_tolerance:
        toi = 0.0
    else:
        quadratic = relative_displacement.dot(relative_displacement)
        linear = 2.0 * relative_position.dot(relative_displacement)
        constant = initial_distance_squared
        roots = real_cubic_roots(0.0, quadratic, linear, constant)
        first_root = ti.math.inf
        for root_id in ti.static(range(3)):
            root = roots[root_id]
            if root >= 0.0 and root <= 1.0 and root < first_root:
                first_root = root
        if first_root < ti.math.inf:
            toi = (1.0 - gap_fraction) * first_root
    return ti.max(0.0, ti.min(1.0, toi))


@ti.func
def point_triangle_ccd(
    point0,
    triangle00,
    triangle10,
    triangle20,
    point_displacement,
    triangle0_displacement,
    triangle1_displacement,
    triangle2_displacement,
    gap_fraction,
    max_iteration,
):
    """Analytic zero-thickness point--triangle CCD."""

    length_scale = _length_scale4(point0, triangle00, triangle10, triangle20)
    geometric_tolerance = 1.0e-9 * length_scale
    initial_distance_squared = _point_triangle_distance_squared(point0, triangle00, triangle10, triangle20)
    toi = 1.0
    if initial_distance_squared <= geometric_tolerance * geometric_tolerance:
        toi = 0.0
    else:
        origin0 = point0 - triangle00
        axis10 = triangle10 - triangle00
        axis20 = triangle20 - triangle00
        dorigin = point_displacement - triangle0_displacement
        daxis1 = triangle1_displacement - triangle0_displacement
        daxis2 = triangle2_displacement - triangle0_displacement
        cubic, quadratic, linear, constant = _coplanarity_coefficients(origin0, axis10, axis20, dorigin, daxis1, daxis2)
        coefficient_norm = ti.max(
            ti.abs(cubic),
            ti.max(
                ti.abs(quadratic),
                ti.max(ti.abs(linear), ti.abs(constant)),
            ),
        )
        if coefficient_norm <= 1.0e-13 * length_scale**3:
            # The entire trajectory is coplanar; the cubic contains no isolated
            # event, so use distance-based conservative advancement.
            toi = point_triangle_accd(
                point0,
                triangle00,
                triangle10,
                triangle20,
                point_displacement,
                triangle0_displacement,
                triangle1_displacement,
                triangle2_displacement,
                gap_fraction,
                0.0,
                max_iteration,
            )
        else:
            roots = real_cubic_roots(cubic, quadratic, linear, constant)
            first_root = ti.math.inf
            for root_id in ti.static(range(3)):
                root = roots[root_id]
                if root >= -1.0e-10 and root <= 1.0 + 1.0e-10:
                    root = ti.max(0.0, ti.min(1.0, root))
                    point = point0 + root * point_displacement
                    triangle0 = triangle00 + root * triangle0_displacement
                    triangle1 = triangle10 + root * triangle1_displacement
                    triangle2 = triangle20 + root * triangle2_displacement
                    distance_squared = _point_triangle_distance_squared(point, triangle0, triangle1, triangle2)
                    if distance_squared <= geometric_tolerance * geometric_tolerance and root < first_root:
                        first_root = root
            if first_root < ti.math.inf:
                toi = (1.0 - gap_fraction) * first_root
    return ti.max(0.0, ti.min(1.0, toi))


@ti.func
def edge_edge_ccd(
    edge_a00,
    edge_a10,
    edge_b00,
    edge_b10,
    edge_a0_displacement,
    edge_a1_displacement,
    edge_b0_displacement,
    edge_b1_displacement,
    gap_fraction,
    max_iteration,
):
    """Analytic zero-thickness edge--edge CCD."""

    length_scale = _length_scale4(edge_a00, edge_a10, edge_b00, edge_b10)
    geometric_tolerance = 1.0e-9 * length_scale
    initial_distance_squared = _edge_edge_distance_squared(edge_a00, edge_a10, edge_b00, edge_b10)
    toi = 1.0
    if initial_distance_squared <= geometric_tolerance * geometric_tolerance:
        toi = 0.0
    else:
        origin0 = edge_b00 - edge_a00
        axis10 = edge_a10 - edge_a00
        axis20 = edge_b10 - edge_b00
        dorigin = edge_b0_displacement - edge_a0_displacement
        daxis1 = edge_a1_displacement - edge_a0_displacement
        daxis2 = edge_b1_displacement - edge_b0_displacement
        cubic, quadratic, linear, constant = _coplanarity_coefficients(origin0, axis10, axis20, dorigin, daxis1, daxis2)
        coefficient_norm = ti.max(
            ti.abs(cubic),
            ti.max(
                ti.abs(quadratic),
                ti.max(ti.abs(linear), ti.abs(constant)),
            ),
        )
        if coefficient_norm <= 1.0e-13 * length_scale**3:
            toi = edge_edge_accd(
                edge_a00,
                edge_a10,
                edge_b00,
                edge_b10,
                edge_a0_displacement,
                edge_a1_displacement,
                edge_b0_displacement,
                edge_b1_displacement,
                gap_fraction,
                0.0,
                max_iteration,
            )
        else:
            roots = real_cubic_roots(cubic, quadratic, linear, constant)
            first_root = ti.math.inf
            for root_id in ti.static(range(3)):
                root = roots[root_id]
                if root >= -1.0e-10 and root <= 1.0 + 1.0e-10:
                    root = ti.max(0.0, ti.min(1.0, root))
                    edge_a0 = edge_a00 + root * edge_a0_displacement
                    edge_a1 = edge_a10 + root * edge_a1_displacement
                    edge_b0 = edge_b00 + root * edge_b0_displacement
                    edge_b1 = edge_b10 + root * edge_b1_displacement
                    distance_squared = _edge_edge_distance_squared(edge_a0, edge_a1, edge_b0, edge_b1)
                    if distance_squared <= geometric_tolerance * geometric_tolerance and root < first_root:
                        first_root = root
            if first_root < ti.math.inf:
                toi = (1.0 - gap_fraction) * first_root
    return ti.max(0.0, ti.min(1.0, toi))


__all__ = [
    "ccd_mode_parameters",
    "edge_edge_ccd",
    "linear_gap_ccd",
    "point_point_ccd",
    "point_triangle_ccd",
]

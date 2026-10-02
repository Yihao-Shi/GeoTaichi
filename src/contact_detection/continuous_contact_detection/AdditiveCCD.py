"""Iterative additive continuous collision detection (ACCD).

The public functions implement the conservative-advancement construction of
Li et al.  ``minimum_distance`` is the requested finite clearance and
``gap_fraction`` is the fraction of the initial excess gap left unused.

All geometry updates and distance queries execute in Taichi device functions.
Broad-phase candidate construction and global minimum reductions remain the
responsibility of the caller.
"""

import taichi as ti

from src.physics_model.contact_model.ipc.ContactGeometry import (
    closest_point_segment,
    closest_point_triangle,
    closest_points_segments,
)


@ti.func
def _excess_distance(distance_squared, minimum_distance):
    """Return ``d - d_min`` without subtractive cancellation."""

    distance = ti.sqrt(ti.max(distance_squared, 0.0))
    excess = (distance_squared - minimum_distance * minimum_distance) / ti.max(distance + minimum_distance, 1.0e-30)
    return distance, excess


@ti.func
def _point_triangle_distance_squared(point, vertex0, vertex1, vertex2):
    closest, _ = closest_point_triangle(point, vertex0, vertex1, vertex2)
    delta = point - closest
    return delta.dot(delta)


@ti.func
def _point_edge_distance_squared(point, endpoint0, endpoint1):
    closest, _ = closest_point_segment(point, endpoint0, endpoint1)
    delta = point - closest
    return ti.max(delta.dot(delta), 0.0)


@ti.func
def point_edge_accd(
    point0,
    endpoint00,
    endpoint10,
    point_displacement0,
    endpoint0_displacement0,
    endpoint1_displacement0,
    gap_fraction,
    minimum_distance,
    max_iteration,
):
    """Iterative finite-clearance ACCD for a moving 2D point and segment."""
    point = point0
    endpoint0 = endpoint00
    endpoint1 = endpoint10
    point_displacement = point_displacement0
    endpoint0_displacement = endpoint0_displacement0
    endpoint1_displacement = endpoint1_displacement0
    for component in ti.static(range(point.n)):
        mean_displacement = (
            point_displacement[component] + endpoint0_displacement[component] + endpoint1_displacement[component]
        ) / 3.0
        point_displacement[component] -= mean_displacement
        endpoint0_displacement[component] -= mean_displacement
        endpoint1_displacement[component] -= mean_displacement
    motion_bound = point_displacement.norm() + ti.sqrt(
        ti.max(
            endpoint0_displacement.dot(endpoint0_displacement),
            endpoint1_displacement.dot(endpoint1_displacement),
        )
    )
    minimum_distance = ti.max(minimum_distance, 0.0)
    gap_fraction = ti.max(0.0, ti.min(gap_fraction, 1.0))
    _, excess = _excess_distance(
        _point_edge_distance_squared(point, endpoint0, endpoint1),
        minimum_distance,
    )
    toi = 1.0
    if excess <= 1.0e-12:
        toi = 0.0
    elif motion_bound > 0.0:
        target_excess = gap_fraction * excess
        accepted_toi = 0.0
        active = True
        iteration = 0
        while active and iteration < max_iteration:
            lower_bound = (1.0 - gap_fraction) * excess / ti.max(motion_bound, 1.0e-30)
            if lower_bound <= 1.0e-12:
                active = False
            elif accepted_toi + lower_bound >= 1.0:
                accepted_toi = 1.0
                active = False
            else:
                trial_point = point + lower_bound * point_displacement
                trial_endpoint0 = endpoint0 + lower_bound * endpoint0_displacement
                trial_endpoint1 = endpoint1 + lower_bound * endpoint1_displacement
                _, trial_excess = _excess_distance(
                    _point_edge_distance_squared(trial_point, trial_endpoint0, trial_endpoint1),
                    minimum_distance,
                )
                if accepted_toi > 0.0 and trial_excess < target_excess:
                    active = False
                else:
                    accepted_toi += lower_bound
                    point = trial_point
                    endpoint0 = trial_endpoint0
                    endpoint1 = trial_endpoint1
                    excess = trial_excess
            iteration += 1
        toi = ti.max(0.0, ti.min(1.0, accepted_toi))
    return toi


@ti.func
def _edge_edge_distance_squared(endpoint_a0, endpoint_a1, endpoint_b0, endpoint_b1):
    closest_a, closest_b, _, _ = closest_points_segments(endpoint_a0, endpoint_a1, endpoint_b0, endpoint_b1)
    delta = closest_a - closest_b
    return ti.max(delta.dot(delta), 0.0)


@ti.func
def _remove_point_triangle_mean_motion(
    point_displacement,
    triangle0_displacement,
    triangle1_displacement,
    triangle2_displacement,
):
    for component in ti.static(range(3)):
        mean_displacement = (
            point_displacement[component]
            + triangle0_displacement[component]
            + triangle1_displacement[component]
            + triangle2_displacement[component]
        ) * 0.25
        point_displacement[component] -= mean_displacement
        triangle0_displacement[component] -= mean_displacement
        triangle1_displacement[component] -= mean_displacement
        triangle2_displacement[component] -= mean_displacement
    return (
        point_displacement,
        triangle0_displacement,
        triangle1_displacement,
        triangle2_displacement,
    )


@ti.func
def _remove_edge_edge_mean_motion(
    edge_a0_displacement,
    edge_a1_displacement,
    edge_b0_displacement,
    edge_b1_displacement,
):
    for component in ti.static(range(3)):
        mean_displacement = (
            edge_a0_displacement[component]
            + edge_a1_displacement[component]
            + edge_b0_displacement[component]
            + edge_b1_displacement[component]
        ) * 0.25
        edge_a0_displacement[component] -= mean_displacement
        edge_a1_displacement[component] -= mean_displacement
        edge_b0_displacement[component] -= mean_displacement
        edge_b1_displacement[component] -= mean_displacement
    return (
        edge_a0_displacement,
        edge_a1_displacement,
        edge_b0_displacement,
        edge_b1_displacement,
    )


@ti.func
def point_point_accd(
    point0,
    point1,
    displacement0,
    displacement1,
    gap_fraction,
    minimum_distance,
    max_iteration,
):
    """Iterative point--point ACCD with a finite minimum distance."""

    minimum_distance = ti.max(minimum_distance, 0.0)
    gap_fraction = ti.max(0.0, ti.min(gap_fraction, 1.0))
    relative_position = point0 - point1
    relative_displacement = displacement0 - displacement1
    motion_bound = relative_displacement.norm()
    _, excess = _excess_distance(relative_position.dot(relative_position), minimum_distance)
    toi = 1.0
    if excess <= 1.0e-12:
        toi = 0.0
    elif motion_bound > 0.0:
        target_excess = gap_fraction * excess
        accepted_toi = 0.0
        active = True
        iteration = 0
        while active and iteration < max_iteration:
            lower_bound = (1.0 - gap_fraction) * excess / ti.max(motion_bound, 1.0e-30)
            if lower_bound <= 1.0e-12:
                active = False
            elif accepted_toi + lower_bound >= 1.0:
                accepted_toi = 1.0
                active = False
            else:
                trial_toi = accepted_toi + lower_bound
                trial_relative = point0 + trial_toi * displacement0 - point1 - trial_toi * displacement1
                _, trial_excess = _excess_distance(trial_relative.dot(trial_relative), minimum_distance)
                if accepted_toi > 0.0 and trial_excess < target_excess:
                    active = False
                else:
                    accepted_toi = trial_toi
                    excess = trial_excess
            iteration += 1
        toi = ti.max(0.0, ti.min(1.0, accepted_toi))
    return toi


@ti.func
def point_triangle_accd(
    point0,
    triangle00,
    triangle10,
    triangle20,
    point_displacement0,
    triangle0_displacement0,
    triangle1_displacement0,
    triangle2_displacement0,
    gap_fraction,
    minimum_distance,
    max_iteration,
):
    """Iterative point--triangle ACCD with a finite minimum distance."""

    point = point0
    triangle0 = triangle00
    triangle1 = triangle10
    triangle2 = triangle20
    (
        point_displacement,
        triangle0_displacement,
        triangle1_displacement,
        triangle2_displacement,
    ) = _remove_point_triangle_mean_motion(
        point_displacement0,
        triangle0_displacement0,
        triangle1_displacement0,
        triangle2_displacement0,
    )
    motion_bound = point_displacement.norm() + ti.sqrt(
        ti.max(
            triangle0_displacement.dot(triangle0_displacement),
            ti.max(
                triangle1_displacement.dot(triangle1_displacement),
                triangle2_displacement.dot(triangle2_displacement),
            ),
        )
    )
    minimum_distance = ti.max(minimum_distance, 0.0)
    gap_fraction = ti.max(0.0, ti.min(gap_fraction, 1.0))
    distance_squared = _point_triangle_distance_squared(point, triangle0, triangle1, triangle2)
    _, excess = _excess_distance(distance_squared, minimum_distance)
    toi = 1.0
    if excess <= 1.0e-12:
        toi = 0.0
    elif motion_bound > 0.0:
        target_excess = gap_fraction * excess
        accepted_toi = 0.0
        active = True
        iteration = 0
        while active and iteration < max_iteration:
            lower_bound = (1.0 - gap_fraction) * excess / ti.max(motion_bound, 1.0e-30)
            if lower_bound <= 1.0e-12:
                active = False
            elif accepted_toi + lower_bound >= 1.0:
                accepted_toi = 1.0
                active = False
            else:
                trial_point = point + lower_bound * point_displacement
                trial_triangle0 = triangle0 + lower_bound * triangle0_displacement
                trial_triangle1 = triangle1 + lower_bound * triangle1_displacement
                trial_triangle2 = triangle2 + lower_bound * triangle2_displacement
                trial_distance_squared = _point_triangle_distance_squared(
                    trial_point,
                    trial_triangle0,
                    trial_triangle1,
                    trial_triangle2,
                )
                _, trial_excess = _excess_distance(trial_distance_squared, minimum_distance)
                if accepted_toi > 0.0 and trial_excess < target_excess:
                    active = False
                else:
                    accepted_toi += lower_bound
                    point = trial_point
                    triangle0 = trial_triangle0
                    triangle1 = trial_triangle1
                    triangle2 = trial_triangle2
                    excess = trial_excess
            iteration += 1
        toi = ti.max(0.0, ti.min(1.0, accepted_toi))
    return toi


@ti.func
def edge_edge_accd(
    edge_a00,
    edge_a10,
    edge_b00,
    edge_b10,
    edge_a0_displacement0,
    edge_a1_displacement0,
    edge_b0_displacement0,
    edge_b1_displacement0,
    gap_fraction,
    minimum_distance,
    max_iteration,
):
    """Iterative edge--edge ACCD with a finite minimum distance."""

    edge_a0 = edge_a00
    edge_a1 = edge_a10
    edge_b0 = edge_b00
    edge_b1 = edge_b10
    (
        edge_a0_displacement,
        edge_a1_displacement,
        edge_b0_displacement,
        edge_b1_displacement,
    ) = _remove_edge_edge_mean_motion(
        edge_a0_displacement0,
        edge_a1_displacement0,
        edge_b0_displacement0,
        edge_b1_displacement0,
    )
    motion_bound = ti.sqrt(
        ti.max(
            edge_a0_displacement.dot(edge_a0_displacement),
            edge_a1_displacement.dot(edge_a1_displacement),
        )
    ) + ti.sqrt(
        ti.max(
            edge_b0_displacement.dot(edge_b0_displacement),
            edge_b1_displacement.dot(edge_b1_displacement),
        )
    )
    minimum_distance = ti.max(minimum_distance, 0.0)
    gap_fraction = ti.max(0.0, ti.min(gap_fraction, 1.0))
    distance_squared = _edge_edge_distance_squared(edge_a0, edge_a1, edge_b0, edge_b1)
    _, excess = _excess_distance(distance_squared, minimum_distance)
    toi = 1.0
    if excess <= 1.0e-12:
        toi = 0.0
    elif motion_bound > 0.0:
        target_excess = gap_fraction * excess
        accepted_toi = 0.0
        active = True
        iteration = 0
        while active and iteration < max_iteration:
            lower_bound = (1.0 - gap_fraction) * excess / ti.max(motion_bound, 1.0e-30)
            if lower_bound <= 1.0e-12:
                active = False
            elif accepted_toi + lower_bound >= 1.0:
                accepted_toi = 1.0
                active = False
            else:
                trial_edge_a0 = edge_a0 + lower_bound * edge_a0_displacement
                trial_edge_a1 = edge_a1 + lower_bound * edge_a1_displacement
                trial_edge_b0 = edge_b0 + lower_bound * edge_b0_displacement
                trial_edge_b1 = edge_b1 + lower_bound * edge_b1_displacement
                trial_distance_squared = _edge_edge_distance_squared(
                    trial_edge_a0,
                    trial_edge_a1,
                    trial_edge_b0,
                    trial_edge_b1,
                )
                _, trial_excess = _excess_distance(trial_distance_squared, minimum_distance)
                if accepted_toi > 0.0 and trial_excess < target_excess:
                    active = False
                else:
                    accepted_toi += lower_bound
                    edge_a0 = trial_edge_a0
                    edge_a1 = trial_edge_a1
                    edge_b0 = trial_edge_b0
                    edge_b1 = trial_edge_b1
                    excess = trial_excess
            iteration += 1
        toi = ti.max(0.0, ti.min(1.0, accepted_toi))
    return toi


@ti.func
def linear_gap_accd(
    gap,
    relative_displacement,
    slackness,
    minimum_distance,
):
    """Analytic signed-gap query with a finite additive clearance."""

    effective_gap = gap - ti.max(minimum_distance, 0.0)
    toi = 0.0
    if effective_gap > 0.0:
        toi = 1.0
        if relative_displacement < 0.0:
            toi = slackness * effective_gap / (-relative_displacement)
    return ti.max(0.0, ti.min(1.0, toi))


@ti.func
def point_nurbs_accd_increment(
    distance,
    maximum_relative_control_displacement,
    conservative_rescaling,
    minimum_distance,
):
    """One rigorous point--NURBS conservative-advancement increment.

    For fixed positive weights, every NURBS point displacement is a convex
    combination of control-point displacements.  Consequently
    ``max_i ||dp-dP_i||`` bounds the relative point--surface motion and is
    invariant under common rigid translation, even when the closest NURBS
    parameter changes.  The caller must accumulate this increment and
    recompute the closest distance after moving both sides.
    """

    relative_motion_bound = ti.max(maximum_relative_control_displacement, 0.0)
    increment = 1.0
    excess = distance - ti.max(minimum_distance, 0.0)
    if excess <= 0.0:
        increment = 0.0
    elif relative_motion_bound > 0.0:
        increment = conservative_rescaling * excess / relative_motion_bound
    return ti.max(0.0, increment)


__all__ = [
    "edge_edge_accd",
    "linear_gap_accd",
    "point_nurbs_accd_increment",
    "point_point_accd",
    "point_triangle_accd",
]

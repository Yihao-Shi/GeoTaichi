"""Near-parallel segment classification must not use an infinite-line gap."""

import itertools

import numpy as np
import pytest
import taichi as ti

from src.physics_model.contact_model.ipc.ContactDistance import (
    edge_edge_distance2_from_type,
    edge_edge_distance_type,
)

from src.physics_model.contact_model.ipc.ContactGeometry import closest_points_segments, edge_edge_coordinates_from_type
from tests.helpers.fem_contact_reference import _closest_points_segments
from src.fem.soft_particle.ContactKernel import _closest_segment_parameters

pytestmark = [pytest.mark.unit, pytest.mark.ipc, pytest.mark.contact]


@ti.kernel
def _distance(x: ti.types.ndarray(dtype=ti.f64, ndim=2)) -> ti.f64:
    a0 = ti.Vector([x[0, 0], x[0, 1], x[0, 2]])
    a1 = ti.Vector([x[1, 0], x[1, 1], x[1, 2]])
    b0 = ti.Vector([x[2, 0], x[2, 1], x[2, 2]])
    b1 = ti.Vector([x[3, 0], x[3, 1], x[3, 2]])
    kind = edge_edge_distance_type(a0, a1, b0, b1)
    return edge_edge_distance2_from_type(a0, a1, b0, b1, kind)


def test_separated_near_parallel_edges_keep_endpoint_gap(taichi_runtime):
    # Captured CUDA trial: the old dot-product differences classified this as EE.
    x = np.array(
        [
            [0.9524999997193113, 0.5834999997193104, 0.05384608174985583],
            [0.9525000012201649, 0.5835000012201640, 0.09881724700927602],
            [0.9525000011881976, 0.5835000011881969, 0.10275846091050644],
            [0.9525000035983687, 0.5835000035983681, 0.14773367183937802],
        ]
    )
    expected = float(np.sum((x[1] - x[2]) ** 2))
    for reverse_a, reverse_b, swap in itertools.product((False, True), repeat=3):
        a, b = x[:2], x[2:]
        if reverse_a:
            a = a[::-1]
        if reverse_b:
            b = b[::-1]
        points = np.concatenate((b, a) if swap else (a, b))
        for scale in (0.01, 1.0, 100.0):
            assert _distance(points * scale) == pytest.approx(expected * scale**2, rel=2e-12)


def test_near_parallel_interior_closest_points_remain_interior(taichi_runtime):
    # Both closest parameters are 1/2; treating all near-parallel edges as
    # endpoint contacts would add the in-plane endpoint separation to the gap.
    x = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, -2e-7, 1e-8], [1.0, 2e-7, 1e-8]])
    assert _distance(x) == pytest.approx(1e-16, rel=1e-12, abs=1e-28)


@ti.kernel
def _closest_geometry(x: ti.types.ndarray(dtype=ti.f64, ndim=2)) -> ti.types.vector(5, ti.f64):
    a0 = ti.Vector([x[0, 0], x[0, 1], x[0, 2]])
    a1 = ti.Vector([x[1, 0], x[1, 1], x[1, 2]])
    b0 = ti.Vector([x[2, 0], x[2, 1], x[2, 2]])
    b1 = ti.Vector([x[3, 0], x[3, 1], x[3, 2]])
    p, q, s, t = closest_points_segments(a0, a1, b0, b1)
    kind = edge_edge_distance_type(a0, a1, b0, b1)
    fs, ft = edge_edge_coordinates_from_type(a0, a1, b0, b1, kind)
    return ti.Vector([(p - q).norm_sqr(), s, t, fs, ft])


@ti.kernel
def _soft_parameters(x: ti.types.ndarray(dtype=ti.f64, ndim=2)) -> ti.types.vector(2, ti.f64):
    a0 = ti.Vector([x[0, 0], x[0, 1], x[0, 2]])
    a1 = ti.Vector([x[1, 0], x[1, 1], x[1, 2]])
    b0 = ti.Vector([x[2, 0], x[2, 1], x[2, 2]])
    b1 = ti.Vector([x[3, 0], x[3, 1], x[3, 2]])
    s, t = _closest_segment_parameters(a0, a1, b0, b1)
    return ti.Vector([s, t])


def test_ccd_and_friction_coordinates_match_near_parallel_interior(taichi_runtime):
    # Analytic closest points at s=t=0.5, despite the ill-conditioned Gram matrix.
    x = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, -2e-7, 1e-8], [1.0, 2e-7, 1e-8]])
    result = np.asarray(_closest_geometry(x))
    assert result[0] == pytest.approx(1e-16, rel=1e-12, abs=1e-28)
    np.testing.assert_allclose(result[1:], 0.5, rtol=0, atol=2e-15)
    p, q, s, t = _closest_points_segments(*x)
    np.testing.assert_allclose([s, t], [0.5, 0.5], rtol=0, atol=2e-15)
    assert np.sum((p - q) ** 2) == pytest.approx(1e-16, rel=1e-12, abs=1e-28)
    # Scale above the explicit soft-particle path's existing degeneracy threshold.
    np.testing.assert_allclose(np.asarray(_soft_parameters(x * 100)), [0.5, 0.5], rtol=0, atol=2e-15)

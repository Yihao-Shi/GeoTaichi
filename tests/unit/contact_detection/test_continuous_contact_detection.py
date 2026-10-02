"""Contracts for the shared continuous contact-detection primitives."""

import numpy as np
import pytest
import taichi as ti

from src.contact_detection.continuous_contact_detection import (
    ccd_mode_parameters,
    deformation_gradient_ccd,
    edge_edge_accd,
    edge_edge_ccd,
    linear_gap_accd,
    linear_gap_ccd,
    point_edge_accd,
    point_edge_ccd,
    point_nurbs_accd_increment,
    point_point_accd,
    point_point_ccd,
    point_point_quadratic_ccd,
    point_triangle_ccd,
    point_triangle_accd,
    real_cubic_roots,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.contact,
    pytest.mark.geometry,
    pytest.mark.cpu,
]


@ti.kernel
def _evaluate_geometric_ccd() -> ti.types.vector(6, ti.f64):
    zero = ti.Vector.zero(ti.f64, 3)
    point_point_toi = point_point_ccd(
        ti.Vector([0.0, 0.0, 0.0]),
        ti.Vector([2.0, 0.0, 0.0]),
        ti.Vector([2.0, 0.0, 0.0]),
        zero,
        0.2,
        100,
    )
    point_triangle_toi = point_triangle_ccd(
        ti.Vector([0.0, 0.0, 1.0]),
        ti.Vector([-1.0, -1.0, 0.0]),
        ti.Vector([1.0, -1.0, 0.0]),
        ti.Vector([0.0, 1.0, 0.0]),
        ti.Vector([0.0, 0.0, -2.0]),
        zero,
        zero,
        zero,
        0.2,
        100,
    )
    edge_edge_toi = edge_edge_ccd(
        ti.Vector([-1.0, 0.0, 1.0]),
        ti.Vector([1.0, 0.0, 1.0]),
        ti.Vector([0.0, -1.0, 0.0]),
        ti.Vector([0.0, 1.0, 0.0]),
        ti.Vector([0.0, 0.0, -2.0]),
        ti.Vector([0.0, 0.0, -2.0]),
        zero,
        zero,
        0.2,
        100,
    )
    quadratic_toi = point_point_quadratic_ccd(
        ti.Vector([0.0, 0.0, 0.0]),
        ti.Vector([2.0, 0.0, 0.0]),
        ti.Vector([2.0, 0.0, 0.0]),
        zero,
        0.2,
        0.9,
    )
    return ti.Vector(
        [
            point_point_toi,
            point_triangle_toi,
            edge_edge_toi,
            quadratic_toi,
            linear_gap_ccd(1.0, -2.0, 0.9),
            point_edge_ccd(
                ti.Vector([0.0, 1.0]),
                ti.Vector([-1.0, 0.0]),
                ti.Vector([1.0, 0.0]),
                ti.Vector([0.0, -2.0]),
                ti.Vector.zero(ti.f64, 2),
                ti.Vector.zero(ti.f64, 2),
                0.2,
                100,
            ),
        ]
    )


@ti.kernel
def _evaluate_material_ccd() -> ti.types.vector(6, ti.f64):
    current2 = ti.Matrix.identity(ti.f64, 2)
    increment2 = ti.Matrix.zero(ti.f64, 2, 2)
    increment2[0, 0] = -2.0
    current3 = ti.Matrix.identity(ti.f64, 3)
    increment3 = ti.Matrix.zero(ti.f64, 3, 3)
    increment3[0, 0] = -2.0
    compression3 = ti.Matrix.diag(3, -1.0)
    compression3[1, 1] = -2.0
    compression3[2, 2] = -3.0
    crossing_with_safe_endpoint = ti.Matrix.zero(ti.f64, 3, 3)
    crossing_with_safe_endpoint[0, 0] = -3.0
    crossing_with_safe_endpoint[1, 1] = -2.0
    return ti.Vector(
        [
            deformation_gradient_ccd(current2, increment2, 0.8),
            deformation_gradient_ccd(current3, increment3, 0.8),
            deformation_gradient_ccd(current3, compression3, 0.8),
            deformation_gradient_ccd(current3, compression3, 1.0),
            deformation_gradient_ccd(current3, crossing_with_safe_endpoint, 0.8),
            deformation_gradient_ccd(current3, 0.1 * current3, 0.8),
        ]
    )


@ti.kernel
def _evaluate_additive_ccd() -> ti.types.vector(6, ti.f64):
    zero = ti.Vector.zero(ti.f64, 3)
    point_displacement = ti.Vector([0.0, 0.0, -2.0])
    return ti.Vector(
        [
            point_point_accd(
                ti.Vector([0.0, 0.0, 0.0]),
                ti.Vector([2.0, 0.0, 0.0]),
                ti.Vector([2.0, 0.0, 0.0]),
                zero,
                0.1,
                0.2,
                100,
            ),
            point_triangle_accd(
                ti.Vector([0.0, 0.0, 1.0]),
                ti.Vector([-1.0, -1.0, 0.0]),
                ti.Vector([1.0, -1.0, 0.0]),
                ti.Vector([0.0, 1.0, 0.0]),
                point_displacement,
                zero,
                zero,
                zero,
                0.1,
                0.2,
                100,
            ),
            edge_edge_accd(
                ti.Vector([-1.0, 0.0, 1.0]),
                ti.Vector([1.0, 0.0, 1.0]),
                ti.Vector([0.0, -1.0, 0.0]),
                ti.Vector([0.0, 1.0, 0.0]),
                point_displacement,
                point_displacement,
                zero,
                zero,
                0.1,
                0.2,
                100,
            ),
            linear_gap_accd(1.0, -2.0, 0.9, 0.2),
            point_nurbs_accd_increment(
                1.0,
                point_displacement.norm(),
                0.9,
                0.2,
            ),
            point_edge_accd(
                ti.Vector([0.0, 1.0]),
                ti.Vector([-1.0, 0.0]),
                ti.Vector([1.0, 0.0]),
                ti.Vector([0.0, -2.0]),
                ti.Vector.zero(ti.f64, 2),
                ti.Vector.zero(ti.f64, 2),
                0.1,
                0.2,
                100,
            ),
        ]
    )


@ti.kernel
def _evaluate_ccd_theory_contracts() -> ti.types.vector(9, ti.f64):
    zero = ti.Vector.zero(ti.f64, 3)
    common = ti.Vector([0.4, -0.3, 0.2])
    cubic_roots = real_cubic_roots(1.0, -1.5, 0.66, -0.08)
    return ti.Vector(
        [
            cubic_roots[0],
            cubic_roots[1],
            cubic_roots[2],
            point_triangle_ccd(
                ti.Vector([0.0, -1.0, 0.0]),
                ti.Vector([-1.0, 0.0, 0.0]),
                ti.Vector([1.0, 0.0, 0.0]),
                ti.Vector([0.0, 1.0, 0.0]),
                ti.Vector([0.0, 2.0, 0.0]),
                zero,
                zero,
                zero,
                0.2,
                100,
            ),
            edge_edge_ccd(
                ti.Vector([-1.0, -1.0, 0.0]),
                ti.Vector([1.0, -1.0, 0.0]),
                ti.Vector([0.0, -0.5, 0.0]),
                ti.Vector([0.0, 0.5, 0.0]),
                ti.Vector([0.0, 2.0, 0.0]),
                ti.Vector([0.0, 2.0, 0.0]),
                zero,
                zero,
                0.2,
                100,
            ),
            edge_edge_ccd(
                ti.Vector([-1.0, 0.0, 0.0]),
                ti.Vector([1.0, 0.0, 0.0]),
                ti.Vector([0.0, -1.0, 0.0]),
                ti.Vector([0.0, 1.0, 0.0]),
                zero,
                zero,
                zero,
                zero,
                0.2,
                100,
            ),
            point_triangle_ccd(
                ti.Vector([0.0, 0.0, 1.0]),
                ti.Vector([-1.0, -1.0, 0.0]),
                ti.Vector([1.0, -1.0, 0.0]),
                ti.Vector([0.0, 1.0, 0.0]),
                common,
                common,
                common,
                common,
                0.2,
                100,
            ),
            point_nurbs_accd_increment(1.0, 0.0, 0.9, 0.2),
            point_triangle_accd(
                ti.Vector([0.0, 0.0, 0.1]),
                ti.Vector([-1.0, -1.0, 0.0]),
                ti.Vector([1.0, -1.0, 0.0]),
                ti.Vector([0.0, 1.0, 0.0]),
                zero,
                zero,
                zero,
                zero,
                0.1,
                0.2,
                100,
            ),
        ]
    )


def test_accd_mode_applies_tolerance_and_eta_cap():
    assert ccd_mode_parameters("CCD", 0.2, 1.0e-6) == (
        "ccd",
        0.2,
        0.0,
    )
    assert ccd_mode_parameters("ACCD", 0.2, 1.0e-6) == (
        "accd",
        0.1,
        1.0e-6,
    )
    assert ccd_mode_parameters("off", 0.2, 1.0e-6) == (
        "off",
        0.2,
        0.0,
    )


def test_geometric_ccd_primitives_return_safeguarded_toi(taichi_runtime):
    values = np.asarray(_evaluate_geometric_ccd(), dtype=np.float64)

    np.testing.assert_allclose(
        values,
        np.array([0.8, 0.4, 0.4, 0.81, 0.45, 0.4]),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_material_ccd_is_dimension_independent(taichi_runtime):
    values = np.asarray(_evaluate_material_ccd(), dtype=np.float64)
    compression_root = min(
        root.real for root in np.roots([-6.0, 11.0, -6.0, 0.8]) if root.real > 0.0 and abs(root.imag) < 1.0e-12
    )
    interior_crossing_root = min(
        root.real for root in np.roots([6.0, -5.0, 0.8]) if root.real > 0.0 and abs(root.imag) < 1.0e-12
    )

    np.testing.assert_allclose(
        values,
        np.array([0.4, 0.4, compression_root, 1.0 / 3.0, interior_crossing_root, 1.0]),
        rtol=0.0,
        atol=1.0e-10,
    )


def test_additive_ccd_preserves_finite_clearance(taichi_runtime):
    values = np.asarray(_evaluate_additive_ccd(), dtype=np.float64)

    np.testing.assert_allclose(
        values,
        np.array([0.81, 0.36, 0.36, 0.36, 0.36, 0.36]),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_ccd_theory_contracts_cover_roots_degeneracy_and_relative_motion(
    taichi_runtime,
):
    values = np.asarray(_evaluate_ccd_theory_contracts(), dtype=np.float64)

    np.testing.assert_allclose(
        values,
        np.array([0.2, 0.5, 0.8, 0.4, 0.2, 0.0, 1.0, 1.0, 0.0]),
        rtol=0.0,
        atol=1.0e-10,
    )

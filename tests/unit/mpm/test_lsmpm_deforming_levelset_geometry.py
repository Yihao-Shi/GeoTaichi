from __future__ import annotations


import numpy as np


from research.LSMPM.scripts.analyze_v1_deforming_levelset import (
    affine_rounded_cube_signed_distance,
)
from research.LSMPM.scripts.run_v1_square_rotation import (
    cube_signed_distance,
    rounded_cube_volume,
    rounded_square_contour,
    sample_rounded_cube_material_points,
)


def test_rounded_cube_material_points_match_reference_volume() -> None:
    points, point_volume = sample_rounded_cube_material_points(
        nominal_spacing=0.05,
        points_per_cell=1,
        corner_radius=0.15,
    )

    phi = cube_signed_distance(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        corner_radius=0.15,
    )
    assert points.shape == (7752, 3)
    assert np.max(phi) <= 1.0e-12
    assert np.isclose(
        points.shape[0] * point_volume,
        rounded_cube_volume(corner_radius=0.15),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_rounded_square_contour_is_closed_and_on_exact_zero_surface() -> None:
    contour = rounded_square_contour(corner_radius=0.15)
    phi = cube_signed_distance(
        contour[:, 0],
        contour[:, 1],
        np.zeros(contour.shape[0]),
        corner_radius=0.15,
    )

    assert np.array_equal(contour[0], contour[-1])
    assert np.max(np.abs(phi)) < 1.0e-13


def test_affine_support_distance_recovers_identity_rounded_cube() -> None:
    coordinates = np.linspace(-0.65, 0.65, 11)
    zz, yy, xx = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    expected = cube_signed_distance(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        corner_radius=0.15,
    )
    measured = affine_rounded_cube_signed_distance(
        points,
        np.eye(3),
        spacing=0.05,
        corner_radius=0.15,
        direction_count=768,
    )
    narrow_band = np.abs(expected) <= 0.15

    assert np.max(np.abs(measured[narrow_band] - expected[narrow_band])) < 5.0e-3

import numpy as np

from src.mpm.generator.BodyGenerator import soft_template_shape_bounds


def test_soft_template_shape_bounds_include_surface_and_point_domains():
    surface = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -0.5, 0.0],
            [0.0, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    points = np.asarray(
        [[-0.8, -0.3, -0.2], [0.7, 0.4, 0.25]], dtype=np.float64
    )
    volumes = np.asarray([0.008, 0.027], dtype=np.float64)
    scale = 2.5

    shape_min, shape_max, shape_radius = soft_template_shape_bounds(
        surface, points, volumes, scale
    )

    scaled_surface = scale * surface
    scaled_points = scale * points
    padding = 0.5 * np.cbrt(volumes * scale ** 3)
    expected_min = np.minimum(
        scaled_surface.min(axis=0),
        (scaled_points - padding[:, None]).min(axis=0),
    )
    expected_max = np.maximum(
        scaled_surface.max(axis=0),
        (scaled_points + padding[:, None]).max(axis=0),
    )
    expected_radius = max(
        np.linalg.norm(scaled_surface, axis=1).max(),
        (
            np.linalg.norm(scaled_points, axis=1)
            + np.sqrt(3.0) * padding
        ).max(),
    )

    np.testing.assert_allclose(shape_min, expected_min)
    np.testing.assert_allclose(shape_max, expected_max)
    assert np.isclose(shape_radius, expected_radius)

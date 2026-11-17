import numpy as np
import pytest

from src.mpm.generator.Body import Body


def test_box_sampling_does_not_lose_integral_cells_to_roundoff():
    cube = Body()
    cube.add_cube([0.35, 0.35, 0.04], [0.65, 0.65, 0.24], 0.05)
    assert cube.bodies["body_0"]["points"].shape == (144, 3)

    rectangle = Body()
    rectangle.add_rectangle([0.1, 0.1], [0.3, 0.3], 0.05)
    assert rectangle.bodies["body_0"]["points"].shape == (16, 2)


def test_cube_ppc_boundary_contains_only_outer_subparticle_layer():
    cube = Body()
    cube.add_cube([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], 1.0, ppc=2)
    body = cube.bodies["body_0"]
    points = body["points"]
    boundary = body["boundary_ids"]

    assert points.shape == (64, 3)
    assert boundary.shape == (56,)
    boundary_points = points[boundary]
    assert np.all(np.any((boundary_points == 0.25) | (boundary_points == 1.75), axis=1))
    assert np.unique(boundary).shape == boundary.shape
    np.testing.assert_array_equal(
        cube.get_surface_ids("body_0", all_particles=True),
        np.arange(64, dtype=np.int32),
    )

    with pytest.raises(ValueError, match="positive integer"):
        Body().add_cube([0.0] * 3, [1.0] * 3, 0.5, ppc=0)

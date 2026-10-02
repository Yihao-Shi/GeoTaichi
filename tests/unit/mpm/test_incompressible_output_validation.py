import numpy as np

from examples.mpm.IncompressibleFluid.validate_dam_break_visible_sdf_2d import box_penetration
from examples.mpm.IncompressibleFluid.validate_large_tank_incompressible_3d import (
    enclosed_air_count,
    saved_grid_spacing,
)
from examples.mpm.IncompressibleFluid.wavemaker_tank_3d import deep_air_particle_count


def test_square_sdf_penetration_is_zero_outside_and_depth_inside():
    position = np.array([[0.30, 0.10], [0.35, 0.10], [0.40, 0.10]])

    np.testing.assert_allclose(box_penetration(position), [0.0, 0.0, 0.04])


def test_wavemaker_deep_air_count_excludes_interface_particles():
    cells = np.zeros((5, 5, 5), dtype=np.int32)
    cells[:, :, :2] = 1
    particle_cells = np.array([[2, 2, 2], [2, 2, 4]])

    assert deep_air_particle_count(cells, particle_cells) == 1


def test_saved_grid_spacing_handles_non_divisible_domain_extent():
    dims = np.array([4, 31, 5], dtype=np.int32)
    axes = [(-0.05 + 0.05 * np.arange(count)) for count in dims]
    coords = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    spacing = saved_grid_spacing({"dims": dims, "coords": coords})

    assert np.allclose(spacing, 0.05)
    assert np.floor(1.39 / spacing[1]) == 27


def test_enclosed_air_count_detects_multicell_holes_only():
    cells = np.ones((5, 5, 5), dtype=np.int32)
    cells[2, 2, 2] = 0
    cells[2, 2, 3] = 0
    assert enclosed_air_count(cells) == 2

    cells[2, 2, 4] = 0
    assert enclosed_air_count(cells) == 0

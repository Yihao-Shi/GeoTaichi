import numpy as np
import pytest

from examples.mpm.IncompressibleFluid.dam_break_visible_sdf_2d.draw.validate_dam_break_visible_sdf_2d import (
    polygon_penetration,
)
from examples.mpm.IncompressibleFluid.large_tank_incompressible_3d.draw.validate_large_tank_incompressible_3d import (
    enclosed_air_count,
    saved_grid_spacing,
)
from examples.mpm.IncompressibleFluid.wavemaker_tank_3d.draw.evaluate_wavemaker_tank_3d import (
    deep_air_particle_count,
    deep_air_particle_mask,
    nonzero_spray_pressure_count,
    sdf_mean_surface,
    sdf_surface_profile_metrics,
)


def test_star_sdf_penetration_respects_tips_and_concave_notches():
    from examples.mpm.IncompressibleFluid.dam_break_visible_sdf_2d.obstacle import VERTICES

    center = VERTICES.mean(axis=0)
    position = np.vstack(([0.30, 0.10], VERTICES[0], center, center + [0, 0.06], center + 1.3 * (VERTICES[1] - center)))

    penetration = polygon_penetration(position)
    np.testing.assert_allclose(penetration[:2], 0.0)
    assert penetration[2] > 0.03
    assert penetration[3] > 0.0
    assert penetration[4] == 0.0


def test_wavemaker_deep_air_count_excludes_interface_particles():
    cells = np.zeros((5, 5, 5), dtype=np.int32)
    cells[:, :, :2] = 1
    particle_cells = np.array([[2, 2, 2], [2, 2, 4]])

    assert deep_air_particle_count(cells, particle_cells) == 1


def test_detached_spray_must_remain_pressure_free():
    cells = np.zeros((5, 5, 5), dtype=np.int32)
    cells[:, :, :2] = 1
    particle_cells = np.array([[2, 2, 2], [2, 2, 4]])
    spray = deep_air_particle_mask(cells, particle_cells)
    assert spray.tolist() == [False, True]
    assert nonzero_spray_pressure_count(np.array([100.0, 0.0]), spray) == 0
    assert nonzero_spray_pressure_count(np.array([100.0, 1.0]), spray) == 1


def test_surface_statistics_exclude_columns_intersecting_the_piston():
    dx = 0.02
    cells = np.zeros((14, 8, 10), dtype=np.int32)
    cells[:, :, :6] = 1
    cells[2:4, :, 1:6] = 0
    phi = np.where(cells == 1, -0.01, 0.01)
    assert sdf_surface_profile_metrics(cells, phi, dx)[1] > 0.007
    assert sdf_surface_profile_metrics(cells, phi, dx, piston_position=0.075) == pytest.approx((0, 0, 0))
    assert sdf_mean_surface(cells, phi, dx, piston_position=0.075) == pytest.approx(0.12)


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


def test_closed_tank_side_wall_is_not_an_ambient_air_outlet():
    cells = np.ones((5, 5, 5), dtype=np.int32)
    cells[:, :, 4] = 0
    cells[0:3, 2, 1] = 0
    phi = np.where(cells == 0, 0.1, -0.1)
    assert enclosed_air_count(cells, phi) == 0
    assert enclosed_air_count(cells, phi, exterior_faces=[(2, -1)]) == 3
    cells[0, 2, 1:4] = 0
    phi = np.where(cells == 0, 0.1, -0.1)
    assert enclosed_air_count(cells, phi, exterior_faces=[(2, -1)]) == 0


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("air_phi,expected", [(0.2, 1), (1.0, 1), (2.0, 0)])
def test_sdf_air_connectivity_uses_bilinear_saddle(reverse, air_phi, expected):
    cells = np.ones((5, 5), dtype=np.int32)
    cells[2, 2] = cells[3, 3] = cells[4, 3] = 0
    phi = np.where(cells == 0, air_phi, -1.0)
    if reverse:
        cells, phi = cells[:, ::-1], phi[:, ::-1]
    assert enclosed_air_count(cells) == 1
    assert enclosed_air_count(cells, phi) == expected


def test_sdf_air_connectivity_does_not_cross_solid():
    cells = np.ones((5, 5), dtype=np.int32)
    cells[2, 2] = cells[3, 3] = cells[4, 3] = 0
    phi = np.where(cells == 0, 2.0, -1.0)
    cells[2, 3] = 2
    assert enclosed_air_count(cells, phi) == 1


@pytest.mark.parametrize("axes", [(0, 1), (0, 2), (1, 2)])
@pytest.mark.parametrize("air_phi,expected", [(0.2, 1), (2.0, 0)])
def test_three_dimensional_sdf_face_connections(axes, air_phi, expected):
    cells = np.ones((3, 3, 3), dtype=np.int32)
    cells[1, 1, 1] = 0
    boundary = [1, 1, 1]
    boundary[axes[0]] = boundary[axes[1]] = 0
    cells[tuple(boundary)] = 0
    phi = np.where(cells == 0, air_phi, -1.0)
    assert enclosed_air_count(cells) == 1
    assert enclosed_air_count(cells, phi) == expected


def test_three_dimensional_sdf_connections_preserve_solid_barriers():
    cells = np.ones((3, 3, 3), dtype=np.int32)
    cells[0, 0, 1] = cells[1, 1, 1] = 0
    phi = np.where(cells == 0, 2.0, -1.0)
    cells[0, 1, 1] = 2
    assert enclosed_air_count(cells, phi) == 1


def test_sdf_air_connectivity_rejects_invalid_fields():
    cells = np.zeros((3, 3), dtype=np.int32)
    with pytest.raises(ValueError):
        enclosed_air_count(cells, np.full((3, 3), np.nan))
    with pytest.raises(ValueError):
        enclosed_air_count(cells, np.zeros((3, 2)))


def test_case_policy_accepts_entrained_air_but_rejects_unsupported_pressure(tmp_path):
    from examples.mpm.IncompressibleFluid.large_tank_incompressible_3d.draw.validate_large_tank_incompressible_3d import (
        validate,
    )

    (tmp_path / "particles").mkdir()
    (tmp_path / "grids").mkdir()
    coordinates = np.stack(np.meshgrid(*([np.arange(-1, 7) * 0.1] * 3), indexing="ij"), axis=-1)
    active_types = np.zeros((5, 5, 5), dtype=np.int32)
    active_types[:, :, :2] = 1
    active_types[2, 2, 0] = 0  # A wall-touching air pocket does not vent through a closed floor.
    types = np.pad(active_types, 1, constant_values=2)
    positions = np.array([[0.15, 0.15, 0.15], [0.25, 0.25, 0.25], [0.25, 0.25, 0.45]])
    for index, time in enumerate([0, 0.1, 0.2]):
        np.savez(
            tmp_path / f"particles/MPMParticle{index:06d}.npz",
            t_current=time,
            active=np.ones(3),
            mass=np.ones(3),
            volume=np.full(3, 0.001),
            position=positions,
            velocity=np.zeros((3, 3)),
            pressure=[100.0, 10.0, 0.01],
        )
        np.savez(
            tmp_path / f"grids/MPMGrid{index:06d}.npz",
            t_current=time,
            dims=[8, 8, 8],
            coords=coordinates.reshape(-1, 3),
            cell_type=types[..., None],
            cell_pressure=np.zeros((*types.shape, 1)),
            cell_fluid_sdf=np.where(types == 1, -0.05, 0.05)[..., None],
        )
    assert not validate(tmp_path, 0.2, domain=[0.5] * 3)["passed"]
    assert not validate(tmp_path, 0.2, domain=[0.5] * 3, allow_enclosed_air=True)["passed"]
    result = validate(
        tmp_path, 0.2, domain=[0.5] * 3, allow_enclosed_air=True, allow_detached_spray=True, atmospheric_pressure=0.01
    )
    assert result["passed"]
    assert result["snapshots"][-1]["sdf_enclosed_air_cells"] == 1
    assert result["snapshots"][-1]["detached_spray_particles"] == 1
    np.savez(
        tmp_path / "particles/MPMParticle000002.npz",
        t_current=0.2,
        active=np.ones(3),
        mass=np.ones(3),
        volume=np.full(3, 0.001),
        position=positions,
        velocity=np.zeros((3, 3)),
        pressure=[100.0, 10.0, 1.0],
    )
    result = validate(
        tmp_path, 0.2, domain=[0.5] * 3, allow_enclosed_air=True, allow_detached_spray=True, atmospheric_pressure=0.01
    )
    assert not result["passed"]
    assert result["snapshots"][-1]["non_atmospheric_detached_spray_pressure_particles"] == 1


def test_wave_pressure_contract_keeps_zero_as_its_default():
    from examples.mpm.IncompressibleFluid.large_tank_incompressible_3d.draw.validate_large_tank_incompressible_3d import (
        validate,
    )

    spray = np.array([False, True])
    assert nonzero_spray_pressure_count(np.array([100.0, 0.01]), spray) == 1
    assert nonzero_spray_pressure_count(np.array([100.0, 0.01]), spray, 0.01) == 0
    assert nonzero_spray_pressure_count(np.array([100.0, 0.0]), spray, 0.01) == 0
    assert nonzero_spray_pressure_count(np.array([100.0, 1.0]), spray, 0.01) == 1

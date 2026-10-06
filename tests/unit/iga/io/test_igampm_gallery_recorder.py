"""Unit checks for aligned IGA-MPM gallery output."""

import numpy as np
import pyvista as pv
import pytest

from src.igampm.GalleryRecorder import (
    write_iga_frame,
    write_mpm_particle_frame,
)
from src.nurbs.BasicVolume import Cube


def _unit_cube():
    cube = Cube()
    cube.set_parameters(start_point=[0.0, 0.0, 0.0], size=[1.0, 0.5, 0.2])
    cube.generate_knot_u(degree=2, num_ctrlpts=3)
    cube.generate_knot_v(degree=2, num_ctrlpts=3)
    cube.generate_knot_w(degree=2, num_ctrlpts=3)
    cube.generate_ctrlpts()
    cube.generate_weights()
    return cube


def test_barrier_block_mass_and_surface_measure_do_not_drift_with_refinement():
    from examples.igampm.iga_mpm_barrier_contact.iga_mpm_barrier_contact import _soft_block_points

    expected_volume = 0.28 * 0.24 * 0.14
    expected_surface_area = 2.0 * (0.28 * 0.24 + 0.28 * 0.14 + 0.24 * 0.14)
    for refinement in (1, 2, 4):
        points, boundary, volume, surface_measure = _soft_block_points(refinement)
        assert volume.shape == (points.shape[0],)
        assert surface_measure.shape == (boundary.shape[0],)
        assert np.sum(volume) == pytest.approx(expected_volume)
        assert np.sum(surface_measure) == pytest.approx(expected_surface_area)

    size = (0.24, 0.24, 0.36)
    points, boundary, volume, surface_measure = _soft_block_points(2, size=size)
    assert points.shape[0] == 7 * 7 * 10
    assert np.sum(volume) == pytest.approx(np.prod(size))
    assert np.sum(surface_measure) == pytest.approx(2.0 * (size[0] * size[1] + size[0] * size[2] + size[1] * size[2]))


def test_host_gallery_writers_produce_aligned_vtu_frames(tmp_path):
    cube = _unit_cube()
    initial = cube.control_points.copy()
    current = initial.copy()
    current[:, 2] += 0.04 * current[:, 0] * current[:, 1]
    stress = np.linspace(0.0, 1.0, cube.num_ctrlpts)

    iga_path = write_iga_frame(
        tmp_path,
        7,
        [(cube, current, initial, stress)],
        resolution=(4, 3, 2),
    )
    positions = np.asarray(
        [[0.2, 0.2, 0.4], [0.4, 0.2, 0.35], [0.6, 0.2, 0.3]],
        dtype=np.float64,
    )
    velocities = np.asarray(
        [[0.1, 0.0, -0.2], [0.2, 0.0, -0.1], [0.3, 0.0, 0.0]],
        dtype=np.float64,
    )
    mpm_path = write_mpm_particle_frame(
        tmp_path,
        7,
        positions,
        velocities,
        body_ids=[0, 0, 1],
        contact_samples=[1, 0, 1],
        state_data={
            "stress": [0.0, 2.0, 4.0],
            "equivalent_plastic_strain": [0.0, 0.1, 0.2],
        },
    )

    assert iga_path.name == "GraphicIGA000007.vtu"
    assert mpm_path.name == "GraphicMPMParticle000007.vtu"
    iga = pv.read(iga_path)
    mpm = pv.read(mpm_path)
    assert iga.n_points == 4 * 3 * 2
    assert iga.n_cells == 3 * 2 * 1
    assert set(("displacement", "stress", "patch_id")) <= set(iga.point_data.keys())
    assert mpm.n_points == positions.shape[0]
    assert np.allclose(mpm.points, positions)
    assert set(
        (
            "velocity",
            "speed",
            "bodyID",
            "contact_sample",
            "stress",
            "equivalent_plastic_strain",
        )
    ) <= set(mpm.point_data.keys())
    assert np.array_equal(mpm.point_data["contact_sample"], [1, 0, 1])
    assert np.allclose(mpm.point_data["equivalent_plastic_strain"], [0.0, 0.1, 0.2])

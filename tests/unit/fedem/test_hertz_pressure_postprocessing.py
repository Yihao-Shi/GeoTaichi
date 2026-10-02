import math

import numpy as np

from examples.fedem.HertzContact.hertz_contact import (
    RADIUS,
    TARGET_INDENTATION,
    _pressure_profile,
    _projected_nodal_areas,
    _write_pressure_sample_archive,
)


def test_projected_pressure_profile_is_conservative_and_archived(tmp_path):
    positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.2], [1.0, 1.0, 0.3], [0.0, 1.0, 0.1]])
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    projected_area = _projected_nodal_areas(positions, faces)
    np.testing.assert_allclose(projected_area, [1.0 / 3.0, 1.0 / 6.0, 1.0 / 3.0, 1.0 / 6.0])

    contact_radius = math.sqrt(RADIUS * TARGET_INDENTATION)
    angles = np.deg2rad([0.0, 120.0, 240.0])
    candidate_position = np.column_stack(
        (
            0.8 * contact_radius * np.cos(angles),
            0.8 * contact_radius * np.sin(angles),
            np.zeros(3),
        )
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)
    candidate_area = _projected_nodal_areas(candidate_position, faces)
    pressure = 125.0
    force = pressure * candidate_area
    radial = np.linalg.norm(candidate_position[:, :2], axis=1)
    normal_force = np.column_stack((np.zeros(3), np.zeros(3), force))
    snapshot = {
        "time": 1.0,
        "center_xy": np.zeros(2),
        "radial_distance": radial,
        "normal_force_magnitude": force,
        "projected_node_area": candidate_area,
        "candidate_node_ids": np.arange(radial.size, dtype=np.int32),
        "candidate_normal_force": normal_force,
        "candidate_position": candidate_position,
        "candidate_radial_distance": radial,
        "candidate_normal_force_magnitude": force,
        "candidate_projected_node_area": candidate_area,
        "candidate_reference_node_area": 1.1 * candidate_area,
        "equivalent_contact_radius": contact_radius,
        "boundary_radius_cv": 0.0,
    }

    profile, _, summary = _pressure_profile([snapshot], faces, 0.5, 1.5)
    assert profile[0]["numerical_pressure"] > 0.0
    assert profile[0]["quadrature_time_samples"] > 0
    assert summary["annular_force_closure_relative"] < 1.0e-14
    assert math.isclose(summary["annular_force_integral"], float(np.sum(force)))
    assert summary["error_norm"] == "current-projected-area-weighted surface L2"

    archive = _write_pressure_sample_archive(tmp_path, [snapshot], faces, 0.5, 1.5)
    assert archive == {
        "file": "pressure_equilibrium_samples.npz",
        "sample_count": 1,
        "node_time_sample_count": 3,
    }
    with np.load(tmp_path / archive["file"]) as saved:
        np.testing.assert_array_equal(saved["offsets"], [0, 3])
        np.testing.assert_array_equal(saved["surface_faces"], faces)
        np.testing.assert_allclose(saved["projected_nodal_area"], candidate_area)
        np.testing.assert_allclose(saved["normal_force"], normal_force)

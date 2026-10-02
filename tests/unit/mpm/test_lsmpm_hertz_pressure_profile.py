import importlib.util
from pathlib import Path
import sys

import numpy as np

from src.mpm.generator.Body import sample_solid_hemisphere_halton

SCRIPT_DIR = Path(__file__).resolve().parents[3] / "research" / "LSMPM" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "run_v2_sphere_compression",
    SCRIPT_DIR / "run_v2_sphere_compression.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "analyze_v2_hertz_fem_reference",
    SCRIPT_DIR / "analyze_v2_hertz_fem_reference.py",
)
ANALYZER = importlib.util.module_from_spec(ANALYSIS_SPEC)
ANALYSIS_SPEC.loader.exec_module(ANALYZER)


def test_lower_hemisphere_halton_points_use_mass_center_coordinates():
    radius = 2.0
    points = sample_solid_hemisphere_halton(
        [0.0, 0.0, 0.0],
        radius,
        16384,
        side="lower",
        center_is_mass_center=True,
        seed=20260728,
    )
    geometric_center = np.asarray([0.0, 0.0, 3.0 * radius / 8.0])
    relative = points - geometric_center

    assert np.max(relative[:, 2]) <= 0.0
    assert np.max(np.linalg.norm(relative, axis=1)) <= radius
    np.testing.assert_allclose(np.mean(points, axis=0), np.zeros(3), atol=2.0e-3 * radius)


def test_lower_hemisphere_signed_distance_has_flat_top_and_curved_bottom():
    radius = 2.0
    points = np.asarray(
        [
            [0.0, 0.0, -5.0 * radius / 8.0],
            [0.0, 0.0, 3.0 * radius / 8.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0 * radius / 8.0 + 0.2],
        ]
    )
    distance = MODULE.lower_hemisphere_signed_distance(points, radius)

    np.testing.assert_allclose(
        distance,
        [0.0, 0.0, -3.0 * radius / 8.0, 0.2],
        atol=1.0e-12,
    )


def _analytical_profile(tmp_path, pressure_relative_offset=0.0, diameter_samples=40):
    radius = 8.0
    poisson = 0.3
    effective_modulus = 200.0 / (1.0 - poisson**2)
    reference_force = 0.625 * np.pi * radius**2
    contact_radius = MODULE.sphere_plane_hertz_contact_radius(reference_force, radius, effective_modulus)
    peak_pressure = MODULE.sphere_plane_hertz_peak_pressure(reference_force, contact_radius)
    edges = np.linspace(0.0, contact_radius, diameter_samples // 2 + 1)
    sample_radius = 0.5 * (edges[:-1] + edges[1:])
    projected_area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    pressure = np.asarray(
        [
            MODULE.annular_hertz_pressure(lower, upper, contact_radius, peak_pressure)
            for lower, upper in zip(edges[:-1], edges[1:])
        ]
    )
    pressure[min(7, pressure.size - 1)] *= 1.0 + pressure_relative_offset
    metrics = MODULE.write_pressure_profile(
        case_dir=tmp_path,
        positions_xy=np.column_stack((sample_radius, np.zeros_like(sample_radius))),
        radii=sample_radius,
        heights=np.zeros_like(sample_radius),
        nodal_force=pressure * projected_area,
        surface_area=projected_area,
        projected_area=projected_area,
        penetration=np.zeros_like(sample_radius),
        center_z=1.0,
        wall_reaction=reference_force,
        reference_force=reference_force,
        effective_modulus=effective_modulus,
        sphere_radius=radius,
        requested_bins=8,
        requested_diameter_samples=diameter_samples,
    )
    profile = np.atleast_1d(
        np.genfromtxt(
            tmp_path / "pressure_profile_diameter.csv",
            delimiter=",",
            names=True,
        )
    )
    return metrics, profile


def test_hertz_diameter_profile_has_40_ordered_mirrored_samples(tmp_path):
    metrics, profile = _analytical_profile(tmp_path)

    assert profile.shape == (40,)
    assert np.all(np.diff(profile["x_over_a"]) > 0.0)
    assert np.isclose(profile["x_over_a"][0], -0.975)
    assert np.isclose(profile["x_over_a"][-1], 0.975)
    np.testing.assert_allclose(profile["x_over_a"], -profile["x_over_a"][::-1])
    assert metrics["pressure_profile_diameter_sample_count"] == 40
    assert metrics["pressure_profile_diameter_valid_sample_count"] == 40
    assert metrics["pressure_profile_diameter_max_relative_error"] < 1.0e-12
    assert metrics["pressure_profile_diameter_five_percent_passed"] is True


def test_hertz_diameter_profile_rejects_a_sample_above_five_percent(tmp_path):
    metrics, _ = _analytical_profile(tmp_path, pressure_relative_offset=0.06)

    assert np.isclose(
        metrics["pressure_profile_diameter_max_relative_error"],
        0.06,
    )
    assert metrics["pressure_profile_diameter_samples_within_five_percent"] == 38
    assert metrics["pressure_profile_diameter_five_percent_passed"] is False


def test_fem_reference_profile_has_ten_independent_annuli_without_smoothing(
    tmp_path,
):
    metrics, profile = _analytical_profile(tmp_path, diameter_samples=20)

    assert profile.shape == (20,)
    assert metrics["pressure_profile_independent_annulus_count"] == 10
    assert metrics["pressure_profile_valid_independent_annulus_count"] == 10
    assert metrics["pressure_profile_diameter_annular_relative_l2_error"] < 1.0e-12
    assert metrics["pressure_profile_interpolated_or_smoothed"] is False
    assert metrics["pressure_profile_mirrored_for_plot_only"] is True


def test_fem_reference_surface_keeps_authored_lower_orientation(tmp_path):
    import trimesh

    path = tmp_path / "fem_fine_hemisphere.stl"
    node_count = MODULE.write_locally_refined_lower_hemisphere_mesh(path, 0.5)
    mesh = trimesh.load_mesh(path, process=True)
    centered_vertices = np.asarray(mesh.vertices) - np.asarray(mesh.center_mass)

    assert mesh.is_watertight
    assert mesh.volume > 0.0
    assert 1000 < node_count < 5000
    assert np.max(np.abs(MODULE.lower_hemisphere_signed_distance(centered_vertices, 0.5))) < 5.0e-4
    flipped_vertices = centered_vertices.copy()
    flipped_vertices[:, 2] *= -1.0
    assert np.max(np.abs(MODULE.lower_hemisphere_signed_distance(flipped_vertices, 0.5))) > 0.1
    assert "polyhedron(file=str(shape_mesh)).reset(False)" in (SCRIPT_DIR / "run_v2_sphere_compression.py").read_text(
        encoding="utf-8"
    )


def test_fem_reference_full_sphere_surface_matches_analytic_sdf(tmp_path):
    import trimesh

    path = tmp_path / "fem_fine_sphere.stl"
    node_count = MODULE.write_locally_refined_sphere_mesh(
        path,
        0.5,
        contact_spacing_over_radius=(MODULE.HERTZ_SURFACE_SPACING_OVER_RADIUS["contact_cap"]),
    )
    mesh = trimesh.load_mesh(path, process=True)

    assert mesh.is_watertight
    assert mesh.volume > 0.0
    assert 1500 < node_count < 6000
    assert np.max(np.abs(MODULE.sphere_signed_distance(mesh.vertices, 0.5))) < 1.0e-6
    centered_vertices = mesh.vertices - mesh.center_mass[None, :]
    assert np.max(np.abs(MODULE.sphere_signed_distance(centered_vertices, 0.5, -mesh.center_mass))) < 1.0e-6


def test_retry21_uses_the_fem_full_sphere_and_upper_cap_depth():
    source = (SCRIPT_DIR / "run_gpu_server_float64_queue_retry21.sh").read_text(encoding="utf-8")

    assert "--benchmark-geometry full_sphere" in source
    assert "--remote-load-layer-depth-ratio 0.40" in source


def test_retry21_rectilinear_tetrahedra_pass_fem_quality_floor():
    quality = MODULE.kuhn_tetra_mean_ratio_bounds(0.035, 0.14)

    assert np.isclose(quality["coarse_to_fine_spacing_ratio"], 4.0)
    assert quality["minimum_tetra_mean_ratio"] > 0.05
    assert quality["minimum_tetra_mean_ratio"] < quality["maximum_tetra_mean_ratio"]


def test_current_triangle_area_uses_deformed_geometry_and_wall_projection():
    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)

    nodal_area, nodal_projected_area, triangle_area = MODULE.current_triangle_nodal_areas(positions, faces)

    np.testing.assert_allclose(triangle_area, [np.sqrt(2.0)])
    np.testing.assert_allclose(nodal_area, np.sqrt(2.0) / 3.0)
    np.testing.assert_allclose(nodal_projected_area, 1.0 / 3.0)


def test_retry21_equilibrium_window_uses_reaction_indentation_and_energy():
    history = np.zeros(
        51,
        dtype=[
            ("time", np.float64),
            ("wall_reaction_force", np.float64),
            ("indentation", np.float64),
            ("kinetic_energy", np.float64),
            ("external_work", np.float64),
        ],
    )
    history["time"] = np.linspace(0.55, 0.60, history.size)
    history["wall_reaction_force"] = -ANALYZER.TARGET_FORCE
    history["indentation"] = ANALYZER.TARGET_INDENTATION
    history["kinetic_energy"] = 1.0e-3
    history["external_work"] = 1.0

    window = ANALYZER.equilibrium_window_metrics(history, 0.55, 0.60)

    assert window["history_sample_count"] == 51
    assert window["reaction_relative_error"] < 1.0e-14
    assert window["reaction_coefficient_variation"] < 1.0e-14
    assert window["indentation_relative_error"] < 1.0e-14
    assert np.isclose(window["mean_kinetic_to_external_work"], 1.0e-3)

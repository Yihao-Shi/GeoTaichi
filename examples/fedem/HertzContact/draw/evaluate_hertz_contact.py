"""Evaluation and postprocessing for hertz_contact."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import math
from pathlib import Path
import numpy as np

from examples.fedem.HertzContact.hertz_contact_parameters import (
    POISSON,
    PRESSURE_ANNULUS_COUNT,
    PRESSURE_TRIANGLE_SUBDIVISIONS,
    RADIUS,
    TARGET_INDENTATION,
    TARGET_LOAD,
    YOUNG,
)


def _hertz_contact_radius(force: float) -> float:
    effective_modulus = YOUNG / (1.0 - POISSON * POISSON)
    return float((3.0 * max(force, 0.0) * RADIUS / (4.0 * effective_modulus)) ** (1.0 / 3.0))


def _equilibrium_window(records, end_time: float, duration: float):
    selected = [record for record in records if end_time - duration - 1.0e-12 <= record["time"] <= end_time + 1.0e-12]
    reactions = np.asarray([record["reaction_force"] for record in selected])
    indentations = np.asarray([record["indentation"] for record in selected])
    contact_radii = np.asarray([record["equivalent_contact_radius"] for record in selected])
    kinetic = np.asarray([record["kinetic_energy"] for record in selected])
    strain = np.asarray([record["strain_energy"] for record in selected])
    mean_reaction = float(np.mean(reactions))
    mean_indentation = float(np.mean(indentations))
    mean_contact_radius = float(np.mean(contact_radii))
    analytical_contact_radius = _hertz_contact_radius(mean_reaction)
    return {
        "window_start": end_time - duration,
        "window_end": end_time,
        "sample_count": len(selected),
        "mean_reaction_force": mean_reaction,
        "reaction_relative_error": abs(mean_reaction - TARGET_LOAD) / TARGET_LOAD,
        "reaction_coefficient_variation": float(np.std(reactions) / max(abs(mean_reaction), 1.0e-30)),
        "mean_indentation": mean_indentation,
        "indentation_relative_error": abs(mean_indentation - TARGET_INDENTATION) / TARGET_INDENTATION,
        "mean_contact_radius": mean_contact_radius,
        "analytical_contact_radius": analytical_contact_radius,
        "contact_radius_relative_error": abs(mean_contact_radius - analytical_contact_radius)
        / max(analytical_contact_radius, 1.0e-30),
        "mean_kinetic_strain_ratio": float(np.mean(kinetic) / max(np.mean(strain), 1.0e-30)),
    }


def _annular_pressure_forces(
    snapshot,
    surface_faces: np.ndarray,
    edges: np.ndarray,
    subdivisions: int = PRESSURE_TRIANGLE_SUBDIVISIONS,
):
    """Conservatively remap piecewise-linear nodal pressure to radial annuli."""
    node_ids = np.asarray(snapshot["candidate_node_ids"], dtype=np.int64)
    forces = np.asarray(snapshot["candidate_normal_force_magnitude"], dtype=np.float64)
    areas = np.asarray(snapshot["candidate_projected_node_area"], dtype=np.float64)
    positions = np.asarray(snapshot["candidate_position"], dtype=np.float64)
    bin_force = np.zeros(len(edges) - 1, dtype=np.float64)
    bin_samples = np.zeros(len(edges) - 1, dtype=np.int64)
    if node_ids.size == 0:
        return bin_force, 0.0, 0.0, bin_samples

    local_index = np.full(
        max(int(np.max(surface_faces)), int(np.max(node_ids))) + 1,
        -1,
        dtype=np.int64,
    )
    local_index[node_ids] = np.arange(node_ids.size)
    local_faces = local_index[surface_faces]
    complete = np.all(local_faces >= 0, axis=1)
    if np.any(~complete):
        known = local_faces[~complete] >= 0
        known_local = np.maximum(local_faces[~complete], 0)
        if np.any(np.any(known & (forces[known_local] > 0.0), axis=1)):
            raise RuntimeError("candidate archive omits a surface face incident to an active contact node")
    local_faces = local_faces[complete]

    pressure = forces / np.maximum(areas, 1.0e-30)
    triangle_xy = positions[local_faces, :2] - np.asarray(snapshot["center_xy"])[None, None, :]
    triangle_pressure = pressure[local_faces]
    edge_a = triangle_xy[:, 1] - triangle_xy[:, 0]
    edge_b = triangle_xy[:, 2] - triangle_xy[:, 0]
    triangle_area = 0.5 * np.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0])
    selected = (triangle_area > 0.0) & np.any(triangle_pressure > 0.0, axis=1)
    triangle_xy = triangle_xy[selected]
    triangle_pressure = triangle_pressure[selected]
    triangle_area = triangle_area[selected]

    barycentric = []
    for first in range(subdivisions):
        for second in range(subdivisions - first):
            u = (first + 1.0 / 3.0) / subdivisions
            v = (second + 1.0 / 3.0) / subdivisions
            barycentric.append((1.0 - u - v, u, v))
    for first in range(subdivisions - 1):
        for second in range(subdivisions - first - 1):
            u = (first + 2.0 / 3.0) / subdivisions
            v = (second + 2.0 / 3.0) / subdivisions
            barycentric.append((1.0 - u - v, u, v))
    barycentric = np.asarray(barycentric, dtype=np.float64)

    outside_force = 0.0
    remapped_force = 0.0
    for start in range(0, len(triangle_area), 256):
        stop = start + 256
        sample_xy = np.einsum("qv,fvd->fqd", barycentric, triangle_xy[start:stop])
        sample_pressure = np.einsum("qv,fv->fq", barycentric, triangle_pressure[start:stop])
        sample_force = sample_pressure * (triangle_area[start:stop, None] / subdivisions**2)
        sample_radius = np.linalg.norm(sample_xy, axis=2)
        inside = sample_radius <= edges[-1]
        indices = np.searchsorted(edges, sample_radius[inside], side="right") - 1
        indices = np.minimum(indices, len(bin_force) - 1)
        np.add.at(bin_force, indices, sample_force[inside])
        np.add.at(bin_samples, indices, 1)
        outside_force += float(np.sum(sample_force[~inside]))
        remapped_force += float(np.sum(sample_force))

    expected_force = float(np.sum(forces))
    closure = abs(remapped_force - expected_force) / max(abs(expected_force), 1.0e-30)
    if closure > 1.0e-10:
        raise RuntimeError(f"piecewise-linear pressure remap does not conserve force: {closure:.3e}")
    return bin_force, outside_force, remapped_force, bin_samples


def _pressure_profile(
    snapshots,
    surface_faces: np.ndarray,
    start_time: float,
    end_time: float,
    bin_count: int = PRESSURE_ANNULUS_COUNT,
):
    contact_radius = math.sqrt(RADIUS * TARGET_INDENTATION)
    peak_pressure = 3.0 * TARGET_LOAD / (2.0 * math.pi * contact_radius**2)
    edges = np.linspace(0.0, contact_radius, bin_count + 1)
    annulus_areas = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    per_bin = [[] for _ in range(bin_count)]
    quadrature_samples = np.zeros(bin_count, dtype=np.int64)
    snapshot_count = 0
    total_normal_force = 0.0
    force_inside_analytical_radius = 0.0
    outside_analytical_radius_force = 0.0
    maximum_active_radius = 0.0
    active_radius_force_samples = []
    surface_error_numerator = 0.0
    surface_error_denominator = 0.0
    equivalent_contact_radii = []
    boundary_radius_cvs = []
    for snapshot in snapshots:
        if not start_time - 1.0e-12 <= snapshot["time"] <= end_time + 1.0e-12:
            continue
        snapshot_count += 1
        if snapshot["equivalent_contact_radius"] > 0.0:
            equivalent_contact_radii.append(snapshot["equivalent_contact_radius"])
            boundary_radius_cvs.append(snapshot["boundary_radius_cv"])
        radial = snapshot["radial_distance"]
        force = snapshot["normal_force_magnitude"]
        total_normal_force += float(np.sum(force))
        if radial.size:
            maximum_active_radius = max(maximum_active_radius, float(np.max(radial)))
            active_radius_force_samples.extend(zip(radial.tolist(), force.tolist()))
        candidate_radial = snapshot["candidate_radial_distance"]
        candidate_force = snapshot["candidate_normal_force_magnitude"]
        candidate_area = snapshot["candidate_projected_node_area"]
        annular_force, outside_force, _, sample_counts = _annular_pressure_forces(
            snapshot,
            surface_faces,
            edges,
        )
        force_inside_analytical_radius += float(np.sum(annular_force))
        outside_analytical_radius_force += outside_force
        for index in range(bin_count):
            per_bin[index].append(float(annular_force[index] / annulus_areas[index]))
            quadrature_samples[index] += int(sample_counts[index])
        selected = candidate_radial <= contact_radius
        if np.any(selected):
            numerical_pressure = candidate_force[selected] / np.maximum(candidate_area[selected], 1.0e-30)
            analytical_pressure = peak_pressure * np.sqrt(
                np.maximum(
                    1.0 - (candidate_radial[selected] / contact_radius) ** 2,
                    0.0,
                )
            )
            surface_error_numerator += float(
                np.sum(candidate_area[selected] * (numerical_pressure - analytical_pressure) ** 2)
            )
            surface_error_denominator += float(np.sum(candidate_area[selected] * analytical_pressure**2))
    profile = []
    for index in range(bin_count):
        center = 0.5 * (edges[index] + edges[index + 1])
        lower = edges[index]
        upper = edges[index + 1]
        # Compare the conservative numerical annular average with the exact
        # Hertz annular average, not the pressure at the annulus midpoint.
        analytical = (
            2.0
            * peak_pressure
            * contact_radius**2
            / (3.0 * (upper**2 - lower**2))
            * (
                max(1.0 - (lower / contact_radius) ** 2, 0.0) ** 1.5
                - max(1.0 - (upper / contact_radius) ** 2, 0.0) ** 1.5
            )
        )
        analytical_point = peak_pressure * math.sqrt(max(1.0 - (center / contact_radius) ** 2, 0.0))
        numerical = float(np.mean(per_bin[index])) if per_bin[index] else math.nan
        profile.append(
            {
                "radius": center,
                "radius_over_contact_radius": center / contact_radius,
                "numerical_pressure": numerical,
                "analytical_pressure": analytical,
                "analytical_point_pressure": analytical_point,
                "annulus_area": float(annulus_areas[index]),
                "numerical_force": numerical * annulus_areas[index],
                "quadrature_time_samples": int(quadrature_samples[index]),
            }
        )
    valid = [row for row in profile if math.isfinite(row["numerical_pressure"]) and row["quadrature_time_samples"] > 0]
    annular_relative_l2 = (
        math.sqrt(
            sum(row["annulus_area"] * (row["numerical_pressure"] - row["analytical_pressure"]) ** 2 for row in valid)
            / max(
                sum(row["annulus_area"] * row["analytical_pressure"] ** 2 for row in valid),
                1.0e-30,
            )
        )
        if valid
        else math.inf
    )
    surface_relative_l2 = (
        math.sqrt(surface_error_numerator / surface_error_denominator) if surface_error_denominator > 0.0 else math.inf
    )
    # The ten annular averages are independent observations. Mirror them only
    # for the conventional diameter plot; do not interpolate or smooth them.
    diameter_profile = []
    for sign, rows in ((-1.0, reversed(valid)), (1.0, valid)):
        for row in rows:
            diameter_profile.append(
                {
                    "x_over_contact_radius": sign * row["radius_over_contact_radius"],
                    "numerical_pressure": row["numerical_pressure"],
                    "analytical_pressure": row["analytical_pressure"],
                    "quadrature_time_samples": row["quadrature_time_samples"],
                    "source_radius_over_contact_radius": row["radius_over_contact_radius"],
                }
            )
    force_weighted_radius_99 = 0.0
    if active_radius_force_samples and total_normal_force > 0.0:
        ordered = np.asarray(
            sorted(active_radius_force_samples, key=lambda item: item[0]),
            dtype=np.float64,
        )
        cumulative = np.cumsum(ordered[:, 1])
        threshold = 0.99 * cumulative[-1]
        force_weighted_radius_99 = float(ordered[min(int(np.searchsorted(cumulative, threshold)), len(ordered) - 1), 0])
    annular_force = float(sum(row["numerical_force"] for row in profile))
    mean_force_inside = force_inside_analytical_radius / max(snapshot_count, 1)
    mean_total_force = total_normal_force / max(snapshot_count, 1)
    return (
        profile,
        diameter_profile,
        {
            "contact_radius": contact_radius,
            "peak_pressure": peak_pressure,
            "snapshot_count": snapshot_count,
            "requested_annulus_count": bin_count,
            "valid_bin_count": len(valid),
            "independent_annulus_count": len(valid),
            "diameter_point_count": len(diameter_profile),
            "mirrored_for_plot_only": True,
            "interpolated_or_smoothed": False,
            "annular_remap": "piecewise-linear pressure on current projected surface triangles",
            "annular_triangle_subdivisions": PRESSURE_TRIANGLE_SUBDIVISIONS,
            "relative_l2": surface_relative_l2,
            "surface_quadrature_relative_l2": surface_relative_l2,
            "annular_profile_relative_l2": annular_relative_l2,
            "error_norm": "current-projected-area-weighted surface L2",
            "analytical_reference": "nodal Hertz pressure for L2; annular average for plotted profile",
            "annular_force_integral": annular_force,
            "mean_force_inside_analytical_radius": mean_force_inside,
            "mean_total_normal_force": mean_total_force,
            "annular_force_closure_relative": (
                abs(annular_force + outside_analytical_radius_force / max(snapshot_count, 1) - mean_total_force)
                / max(abs(mean_total_force), 1.0e-30)
            ),
            "annular_force_over_mean_total_normal_force": (
                annular_force / mean_total_force if mean_total_force > 0.0 else 0.0
            ),
            "force_outside_analytical_radius_fraction": (
                outside_analytical_radius_force / total_normal_force if total_normal_force > 0.0 else 0.0
            ),
            "maximum_active_radius_over_analytical_radius": (maximum_active_radius / contact_radius),
            "force_weighted_radius_99_over_analytical_radius": (force_weighted_radius_99 / contact_radius),
            "mean_equivalent_contact_radius": (
                float(np.mean(equivalent_contact_radii)) if equivalent_contact_radii else 0.0
            ),
            "equivalent_contact_radius_relative_error": (
                abs(float(np.mean(equivalent_contact_radii)) - contact_radius) / contact_radius
                if equivalent_contact_radii
                else math.inf
            ),
            "mean_boundary_radius_cv": (float(np.mean(boundary_radius_cvs)) if boundary_radius_cvs else math.inf),
        },
    )


def _write_pressure_sample_archive(
    output: Path,
    snapshots,
    surface_faces: np.ndarray,
    start_time: float,
    end_time: float,
) -> dict[str, int | str]:
    selected = [snapshot for snapshot in snapshots if start_time - 1.0e-12 <= snapshot["time"] <= end_time + 1.0e-12]
    counts = np.asarray(
        [snapshot["candidate_node_ids"].size for snapshot in selected],
        dtype=np.int64,
    )
    offsets = np.concatenate(([0], np.cumsum(counts)))

    def concatenate(name, shape, dtype):
        values = [np.asarray(snapshot[name]) for snapshot in selected]
        return np.concatenate(values, axis=0) if values else np.empty(shape, dtype=dtype)

    path = output / "pressure_equilibrium_samples.npz"
    np.savez_compressed(
        path,
        sample_time=np.asarray([snapshot["time"] for snapshot in selected]),
        sample_center_xy=np.asarray(
            [snapshot["center_xy"] for snapshot in selected],
            dtype=np.float64,
        ).reshape(-1, 2),
        offsets=offsets,
        surface_faces=np.asarray(surface_faces, dtype=np.int32),
        node_ids=concatenate("candidate_node_ids", (0,), np.int32),
        normal_force=concatenate("candidate_normal_force", (0, 3), np.float64),
        current_position=concatenate("candidate_position", (0, 3), np.float64),
        projected_nodal_area=concatenate("candidate_projected_node_area", (0,), np.float64),
        reference_nodal_area=concatenate("candidate_reference_node_area", (0,), np.float64),
    )
    return {
        "file": path.name,
        "sample_count": int(len(selected)),
        "node_time_sample_count": int(offsets[-1]),
    }

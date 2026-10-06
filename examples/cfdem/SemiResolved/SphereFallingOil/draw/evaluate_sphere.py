"""Evaluation and postprocessing for sphere."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import math
import numpy as np


def terminal_speed(times, velocities):
    start = max(0, int(0.8 * len(times)))
    samples = -velocities[start:, 0, 2]
    return (float(np.mean(samples)), float(np.std(samples)))


def load_velocity_experiment(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2 or not np.isfinite(data).all():
        raise ValueError(f"invalid Ten Cate velocity data in {path}")
    return data


def evaluate(config, times, centers, velocities, solid_volume_error, ibm_l2_error, experiment=None):
    metrics = {
        "case": "semi-sphere",
        "reference": config["reference"],
        "samples": len(times),
        "final_time": float(times[-1]),
        "solid_volume_relative_error": solid_volume_error,
        "ibm_velocity_l2_error": ibm_l2_error,
    }
    speed, deviation = terminal_speed(times, velocities)
    duration_complete = float(times[-1]) >= 0.95 * config["time"]
    metrics.update(terminal_speed=speed, terminal_speed_std=deviation, validation_duration_complete=duration_complete)
    comparison_speed = float(np.max(-velocities[:, 0, 2]))
    metrics["peak_speed"] = comparison_speed
    relative_error = abs(comparison_speed - config["reference_speed"]) / config["reference_speed"]
    metrics.update(reference_speed=config["reference_speed"], terminal_speed_relative_error=relative_error)
    metrics["peak_speed_relative_error"] = metrics.pop("terminal_speed_relative_error")
    metrics["unbounded_reference_speed"] = config["unbounded_reference_speed"]
    tolerance = 0.2
    passed = duration_complete and relative_error <= tolerance
    expected_acceleration = (
        9.81
        * (config["particle_density"] - config["fluid_density"])
        / (config["particle_density"] + config["added_mass_coefficient"] * config["fluid_density"])
    )
    initial_acceleration = float(-(velocities[1, 0, 2] - velocities[0, 0, 2]) / (times[1] - times[0]))
    acceleration_error = abs(initial_acceleration - expected_acceleration) / expected_acceleration
    metrics.update(
        initial_acceleration=initial_acceleration,
        reference_initial_acceleration=expected_acceleration,
        initial_acceleration_relative_error=acceleration_error,
    )
    passed = passed and acceleration_error <= 0.2
    clearance = float(np.min(np.minimum(centers, np.asarray(config["domain"]) - centers)) - 0.5 * config["diameter"])
    particle_mass = config["particle_density"] * math.pi * config["diameter"] ** 3 / 6.0
    elastic_overlap_tolerance = config["unbounded_reference_speed"] * math.sqrt(
        particle_mass / config["contact_stiffness"]
    )
    metrics["minimum_wall_clearance"] = clearance
    metrics["elastic_contact_overlap_tolerance"] = elastic_overlap_tolerance
    metrics["sphere_stays_inside_tank"] = clearance >= -elastic_overlap_tolerance
    passed = passed and metrics["sphere_stays_inside_tank"]
    if experiment is None:
        experiment = load_velocity_experiment(ROOT / config["velocity_experiment"])
    covered = (experiment[:, 0] >= times[0]) & (experiment[:, 0] <= times[-1])
    peak = float(np.max(np.abs(experiment[:, 1])))
    samples = experiment[covered]
    error = np.interp(samples[:, 0], times, velocities[:, 0, 2]) - samples[:, 1]
    relative_rmse = float(np.sqrt(np.mean(error**2)) / peak) if len(samples) else None
    pre_wall = experiment[:, 0] <= config["experiment_pre_wall_cutoff_s"]
    pre_wall_covered = covered & pre_wall
    pre_wall_samples = experiment[pre_wall_covered]
    pre_wall_peak = float(np.max(np.abs(experiment[pre_wall, 1])))
    pre_wall_error = np.interp(pre_wall_samples[:, 0], times, velocities[:, 0, 2]) - pre_wall_samples[:, 1]
    pre_wall_relative_rmse = (
        float(np.sqrt(np.mean(pre_wall_error**2)) / pre_wall_peak) if len(pre_wall_samples) else None
    )
    metrics.update(
        experiment_time_window_complete=bool(covered.all()),
        experiment_velocity_rmse_over_peak=relative_rmse,
        experiment_pre_wall_cutoff_s=config["experiment_pre_wall_cutoff_s"],
        experiment_pre_wall_time_window_complete=bool(np.all(pre_wall_covered[pre_wall])),
        experiment_pre_wall_velocity_rmse_over_peak=pre_wall_relative_rmse,
        experiment_rmse_tolerance=config["experiment_rmse_tolerance"],
        validation_scope=(
            "unscaled pre-near-wall experimental velocity history, full simulation duration, "
            "and tank clearance; full-history RMSE retained as a diagnostic"
        ),
    )
    passed = (
        passed
        and bool(np.all(pre_wall_covered[pre_wall]))
        and pre_wall_relative_rmse is not None
        and pre_wall_relative_rmse <= config["experiment_rmse_tolerance"]
    )
    metrics["passed"] = bool(passed)
    return metrics

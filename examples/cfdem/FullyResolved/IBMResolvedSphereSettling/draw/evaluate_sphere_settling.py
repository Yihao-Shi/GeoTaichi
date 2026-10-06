"""Evaluation and postprocessing for sphere_settling."""

import math
import numpy as np


def evaluate(config, times, centers, velocities, solid_volume_error, ibm_l2_error, experiment=None):
    del experiment
    times = np.asarray(times, dtype=float)
    centers = np.asarray(centers, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    if times.ndim != 1 or len(times) < 2 or centers.shape != (len(times), 1, 3):
        raise ValueError("sphere validation requires at least two (time, 1 body, 3 coordinates) samples")
    if velocities.shape != centers.shape or not all(
        np.all(np.isfinite(values)) for values in (times, centers, velocities)
    ):
        raise ValueError("sphere trajectory and velocity samples must be finite and shape-compatible")
    sample_intervals = np.diff(times)
    if np.any(sample_intervals <= 0.0):
        raise ValueError("sphere sample times must be strictly increasing")
    metrics = {
        "case": "full-sphere",
        "reference": config["reference"],
        "samples": len(times),
        "final_time": float(times[-1]),
        "solid_volume_relative_error": solid_volume_error,
        "ibm_velocity_l2_error": ibm_l2_error,
    }
    settling_speed = -velocities[:, 0, 2]
    peak_index = int(np.argmax(settling_speed))
    maximum_speed = float(settling_speed[peak_index])
    late_start = max(0, len(times) - max(3, math.ceil(0.2 * len(times))))
    late_speed = settling_speed[late_start:]
    speed, deviation = float(np.mean(late_speed)), float(np.std(late_speed))
    late_relative_range = float(np.ptp(late_speed) / max(maximum_speed, 1.0e-12))
    late_level_relative_error = abs(speed - maximum_speed) / max(maximum_speed, 1.0e-12)
    max_sample_interval = float(np.max(sample_intervals))
    duration_complete = abs(float(times[-1]) - config["time"]) <= max_sample_interval + 1.0e-12
    sampling_complete = max_sample_interval <= config["maximum_sample_interval"] + 1.0e-12
    plateau_sampled = (
        max(late_relative_range, late_level_relative_error) <= config["late_speed_relative_range_tolerance"]
    )
    relative_error = abs(maximum_speed - config["reference_speed"]) / config["reference_speed"]
    ibm_relative_error = ibm_l2_error / max(maximum_speed, 1.0e-12)
    metrics.update(
        terminal_speed=speed,
        terminal_speed_std=deviation,
        maximum_speed=maximum_speed,
        maximum_speed_time=float(times[peak_index]),
        maximum_speed_relative_error=relative_error,
        late_speed_relative_range=late_relative_range,
        late_speed_level_relative_error=late_level_relative_error,
        velocity_plateau_sampled=plateau_sampled,
        validation_duration_complete=duration_complete,
        maximum_sample_interval=max_sample_interval,
        validation_sampling_complete=sampling_complete,
        ibm_velocity_relative_error=ibm_relative_error,
        reference_speed=config["reference_speed"],
        terminal_speed_relative_error=relative_error,
    )
    metrics["unbounded_reference_speed"] = config["unbounded_reference_speed"]
    passed = duration_complete and sampling_complete and plateau_sampled
    passed = passed and relative_error <= config["maximum_speed_relative_tolerance"]
    passed = passed and solid_volume_error <= config["solid_volume_relative_tolerance"]
    passed = passed and ibm_relative_error <= config["ibm_velocity_relative_tolerance"]
    clearance = float(np.min(np.minimum(centers, np.asarray(config["domain"]) - centers)) - 0.5 * config["diameter"])
    metrics["minimum_wall_clearance"] = clearance
    metrics["sphere_stays_inside_tank"] = clearance >= -1e-10 * config["diameter"]
    passed = passed and metrics["sphere_stays_inside_tank"]
    metrics["passed"] = bool(passed)
    return metrics

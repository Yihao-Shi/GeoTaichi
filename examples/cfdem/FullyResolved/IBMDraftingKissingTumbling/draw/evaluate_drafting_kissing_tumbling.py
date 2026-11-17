"""Evaluation and postprocessing for drafting_kissing_tumbling."""

import math
import numpy as np


def evaluate(config, times, centers, velocities, solid_volume_error, ibm_l2_error, experiment=None):
    del experiment
    times = np.asarray(times, dtype=float)
    centers = np.asarray(centers, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    if times.ndim != 1 or len(times) < 2 or centers.shape != (len(times), 2, 3):
        raise ValueError("DKT validation requires at least two (time, 2 bodies, 3 coordinates) samples")
    if velocities.shape != centers.shape or not all(
        np.all(np.isfinite(values)) for values in (times, centers, velocities)
    ):
        raise ValueError("DKT trajectory and velocity samples must be finite and shape-compatible")
    sample_intervals = np.diff(times)
    if np.any(sample_intervals <= 0.0):
        raise ValueError("DKT sample times must be strictly increasing")
    metrics = {
        "case": "dkt",
        "reference": config["reference"],
        "samples": len(times),
        "final_time": float(times[-1]),
        "solid_volume_relative_error": solid_volume_error,
        "ibm_velocity_l2_error": ibm_l2_error,
    }
    diameter = config["diameter"]
    distance = np.linalg.norm(centers[:, 1] - centers[:, 0], axis=1)
    trailing_settling_speed = -velocities[:, 1, 2]
    collision_index = int(np.argmax(trailing_settling_speed))
    contact_time = float(times[collision_index])
    collision_speed = float(trailing_settling_speed[collision_index])
    collision_peak_resolved = 0 < collision_index < len(times) - 1
    post_collision_deceleration = bool(
        collision_peak_resolved
        and np.min(trailing_settling_speed[collision_index + 1 :])
        <= collision_speed - 0.02 * config["reference_collision_speed"]
    )
    initial_lateral = float(np.linalg.norm((centers[0, 1] - centers[0, 0])[:2]))
    lateral = float(np.linalg.norm((centers[-1, 1] - centers[-1, 0])[:2]))
    roles_switched = bool(centers[-1, 1, 2] < centers[-1, 0, 2])
    domain = np.asarray(config["domain"])
    boundary_clearance = float(np.min(np.minimum(centers, domain - centers)) - 0.5 * diameter)
    max_overlap_ratio = float(np.max(np.maximum(0.0, diameter - distance)) / diameter)
    max_sample_interval = float(np.max(sample_intervals))
    duration_error = abs(float(times[-1]) - config["time"])
    duration_complete = duration_error <= max_sample_interval + 1.0e-12
    sampling_complete = max_sample_interval <= config["maximum_sample_interval"] + 1.0e-12
    drafting_window = (times >= 0.15) & (times < min(0.30, contact_time))
    drafting_detected = bool(
        np.any(drafting_window)
        and np.mean(-velocities[drafting_window, 1, 2]) > np.mean(-velocities[drafting_window, 0, 2])
    )
    kissing_detected = bool(np.min(distance) <= 1.01 * diameter)
    separated_after_kissing = bool(
        distance[-1] - np.min(distance) >= config["minimum_post_kissing_separation_ratio"] * diameter
    )
    contact_time_relative_error = (
        abs(contact_time - config["reference_contact_time"]) / config["reference_contact_time"]
    )
    collision_speed_relative_error = (
        abs(collision_speed - config["reference_collision_speed"]) / config["reference_collision_speed"]
    )
    ibm_velocity_relative_error = ibm_l2_error / config["reference_collision_speed"]
    metrics.update(
        contact_time=contact_time if math.isfinite(contact_time) else None,
        contact_time_definition="time of maximum trailing-particle settling speed",
        reference_contact_time=config["reference_contact_time"],
        collision_speed=collision_speed if math.isfinite(collision_speed) else None,
        reference_collision_speed=config["reference_collision_speed"],
        final_center_distance_ratio=float(distance[-1] / diameter),
        minimum_center_distance_ratio=float(np.min(distance) / diameter),
        final_lateral_separation=lateral,
        final_lateral_separation_ratio=lateral / diameter,
        lateral_separation_growth_ratio=(lateral - initial_lateral) / diameter,
        leading_trailing_roles_switched=roles_switched,
        minimum_boundary_clearance=boundary_clearance,
        max_overlap_ratio=max_overlap_ratio,
        validation_duration_complete=duration_complete,
        maximum_sample_interval=max_sample_interval,
        validation_sampling_complete=sampling_complete,
        drafting_detected=drafting_detected,
        kissing_detected=kissing_detected,
        separated_after_kissing=separated_after_kissing,
        contact_time_relative_error=contact_time_relative_error,
        collision_speed_relative_error=collision_speed_relative_error,
        ibm_velocity_relative_error=ibm_velocity_relative_error,
        collision_peak_resolved=collision_peak_resolved,
        post_collision_deceleration_detected=post_collision_deceleration,
    )
    passed = (
        duration_complete
        and sampling_complete
        and collision_peak_resolved
        and post_collision_deceleration
        and drafting_detected
        and kissing_detected
        and separated_after_kissing
        and (lateral - initial_lateral >= config["minimum_lateral_growth_ratio"] * diameter)
        and (distance[-1] >= config["minimum_final_distance_ratio"] * diameter)
        and roles_switched
        and (boundary_clearance >= 0.0)
        and (max_overlap_ratio <= config["maximum_overlap_ratio"])
        and (solid_volume_error <= config["solid_volume_relative_tolerance"])
        and (ibm_velocity_relative_error <= config["ibm_velocity_relative_tolerance"])
    )
    passed = (
        passed
        and contact_time_relative_error <= config["contact_time_relative_tolerance"]
        and collision_speed_relative_error <= config["collision_speed_relative_tolerance"]
    )
    metrics["passed"] = bool(passed)
    return metrics

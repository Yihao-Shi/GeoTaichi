"""Evaluation and postprocessing for ellipsoid_settling."""

import json
from pathlib import Path
import numpy as np


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def aggregate_ellipsoid(output: Path, strict: bool):
    rows = []
    for flatness in (0.2, 0.4, 0.6, 0.8, 1.0):
        path = output / f"fi_{flatness:.1f}" / "metrics.json"
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    speeds = [row["terminal_speed"] for row in rows]
    monotone = all((right > left for left, right in zip(speeds, speeds[1:])))
    payload = {
        "case": "ellipsoid-sweep",
        "flatness": [row["flatness"] for row in rows],
        "terminal_speed": speeds,
        "monotone_with_flatness": monotone,
        "passed": monotone and all((row["passed"] for row in rows)),
        "reference": "Fan et al. (2022); Lai et al. (2023)",
    }
    write_json(output / "metrics.json", payload)
    print(json.dumps(payload, sort_keys=True))
    if strict and (not payload["passed"]):
        raise SystemExit("ellipsoid sweep did not reproduce the published monotone trend")


def terminal_speed(times, velocities):
    start = max(0, int(0.8 * len(times)))
    samples = -velocities[start:, 0, 2]
    return (float(np.mean(samples)), float(np.std(samples)))


def evaluate(config, times, centers, velocities, solid_volume_error, ibm_l2_error, experiment=None):
    metrics = {
        "case": "ellipsoid",
        "reference": config["reference"],
        "samples": len(times),
        "final_time": float(times[-1]),
        "solid_volume_relative_error": solid_volume_error,
        "ibm_velocity_l2_error": ibm_l2_error,
    }
    speed, deviation = terminal_speed(times, velocities)
    duration_complete = float(times[-1]) >= 0.95 * config["time"]
    metrics.update(terminal_speed=speed, terminal_speed_std=deviation, validation_duration_complete=duration_complete)
    metrics["flatness"] = config["flatness"]
    passed = speed > 0.0 and deviation <= max(0.2 * speed, 1e-05)
    passed = passed and solid_volume_error <= 0.15
    passed = passed and ibm_l2_error <= max(0.3 * speed, 0.0001)
    metrics["passed"] = bool(passed)
    return metrics

"""A steady but truncated trajectory is not a completed settling benchmark."""

import numpy as np
import pytest

from examples.cfdem.FullyResolved.IBMResolvedEllipsoidSettling.ellipsoid_settling import case_parameters
from examples.cfdem.FullyResolved.IBMResolvedEllipsoidSettling.draw.evaluate_ellipsoid_settling import evaluate


@pytest.mark.parametrize("end_fraction", [0.5, 0.95, 0.999, 1.0])
def test_ellipsoid_requires_final_time_even_when_speed_is_steady(end_fraction):
    config = case_parameters()
    times = np.linspace(0.0, config["time"] * end_fraction, 101)
    centers = np.tile(config["centers"], (len(times), 1, 1))
    velocities = np.zeros_like(centers)
    velocities[1:, 0, 2] = -0.03
    metrics = evaluate(config, times, centers, velocities, 0.0, 0.0)
    assert metrics["validation_duration_complete"] == (end_fraction == 1.0)
    assert metrics["passed"] == (end_fraction == 1.0)

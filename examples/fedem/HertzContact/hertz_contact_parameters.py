"""Shared model parameters for hertz_contact."""

from __future__ import annotations
import math

RADIUS = 0.05

YOUNG = 2.0e5

POISSON = 0.30

TARGET_INDENTATION = 0.03 * RADIUS

TARGET_LOAD = (4.0 / 3.0) * (YOUNG / (1.0 - POISSON * POISSON)) * math.sqrt(RADIUS) * TARGET_INDENTATION**1.5

PRESSURE_ANNULUS_COUNT = 10

PRESSURE_TRIANGLE_SUBDIVISIONS = 32

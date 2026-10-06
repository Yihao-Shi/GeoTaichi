"""Shared model parameters for taylor_green_vortex_2d."""

import math
import numpy as np

DOMAIN = np.array([math.pi, math.pi])

STRICT_TOLERANCES = {
    "velocity_relative_l2": 0.03,
    "velocity_relative_linf": 0.03,
    "kinetic_energy_relative_error": 0.05,
    "pressure_relative_l2": 0.10,
}

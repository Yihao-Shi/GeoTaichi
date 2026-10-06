"""Shared model parameters for granular_water_leakage_3d."""

import numpy as np

DOMAIN = (0.36, 0.16, 0.36)

UPPER_ORIGIN = np.array((0.03, 0.05, 0.125))

CRACK_LEFT = UPPER_ORIGIN[0] + 0.147

CRACK_RIGHT = CRACK_LEFT + 0.006

RECEIVER_ORIGIN = np.array((0.03, 0.02, 0.005))

RECEIVER_SIZE = np.array((0.30, 0.12, 0.09))

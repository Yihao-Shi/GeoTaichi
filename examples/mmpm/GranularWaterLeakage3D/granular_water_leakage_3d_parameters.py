"""Shared model parameters for granular_water_leakage_3d."""

import numpy as np

DOMAIN = (0.36, 0.16, 0.36)

UPPER_ORIGIN = np.array((0.03, 0.05, 0.125))
UPPER_SIZE = np.array((0.30, 0.06, 0.20))

CRACK_LEFT = UPPER_ORIGIN[0] + 0.147

CRACK_RIGHT = CRACK_LEFT + 0.006

RECEIVER_ORIGIN = np.array((0.03, 0.02, 0.005))

RECEIVER_SIZE = np.array((0.30, 0.12, 0.09))


def represented_crack_bounds(dx, crack_width=None):
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("grid spacing must be finite and positive")
    center = 0.5 * (CRACK_LEFT + CRACK_RIGHT)
    if crack_width is None:
        left = round(center / dx) * dx
        return left, left + dx
    if not np.isfinite(crack_width) or crack_width <= 0:
        raise ValueError("crack width must be finite and positive")
    cells = round(crack_width / dx)
    if cells < 2 or not np.isclose(cells * dx, crack_width, atol=1e-12, rtol=0):
        raise ValueError("custom crack width must span at least two whole grid cells")
    left = round((center - crack_width / 2) / dx) * dx
    right = left + cells * dx
    if left <= UPPER_ORIGIN[0] or right >= UPPER_ORIGIN[0] + UPPER_SIZE[0]:
        raise ValueError("crack must leave a bottom wall on each side")
    return left, right

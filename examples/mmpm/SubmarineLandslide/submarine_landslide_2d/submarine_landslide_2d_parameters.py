"""Shared model parameters for submarine_landslide_2d."""

import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAVE_PATH = os.environ.get(
    "GEOTAICHI_RZADKIEWICZ_LANDSLIDE_SAVE_PATH",
    os.path.join(SCRIPT_DIR, "RzadkiewiczSubmarineLandslide"),
)

TANK_LENGTH = 4.0

WATER_DEPTH = 1.6

SLOPE_START_X = TANK_LENGTH - WATER_DEPTH

SLOPE_POINT = np.array([SLOPE_START_X, 0.0], dtype=np.float64)

SLOPE_NORMAL = np.array([1.0, -1.0], dtype=np.float64) / np.sqrt(2.0)

ELEMENT_SIZE = float(os.environ.get("GEOTAICHI_RZADKIEWICZ_LANDSLIDE_ELEMENT_SIZE", "0.01"))

SAVE_INTERVAL = float(os.environ.get("GEOTAICHI_RZADKIEWICZ_LANDSLIDE_SAVE_INTERVAL", "0.04"))

SAND_TOP = WATER_DEPTH - 0.1

SAND_LEFT_X = 3.0

SAND_RIGHT_X = 3.9

SAND_POLYGON = [
    [SAND_LEFT_X, SAND_LEFT_X - SLOPE_START_X],
    [SAND_RIGHT_X, SAND_TOP],
    [SAND_LEFT_X, SAND_TOP],
]

WATER_POLYGON = [
    [0.0, 0.0],
    [SLOPE_START_X, 0.0],
    [TANK_LENGTH, WATER_DEPTH],
    [0.0, WATER_DEPTH],
]

FREE_SURFACE_BIN_COUNT = int(
    os.environ.get(
        "GEOTAICHI_RZADKIEWICZ_LANDSLIDE_SURFACE_BINS",
        str(int(round(TANK_LENGTH / (2.0 * ELEMENT_SIZE)))),
    )
)

REFERENCE_DOI = "10.1016/j.cma.2024.117064"
MAXIMUM_POROSITY = 0.50

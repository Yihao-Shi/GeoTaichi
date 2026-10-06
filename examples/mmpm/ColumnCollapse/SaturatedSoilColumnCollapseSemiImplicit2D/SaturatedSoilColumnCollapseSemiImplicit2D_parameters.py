"""Shared model parameters for SaturatedSoilColumnCollapseSemiImplicit2D."""

import os

DOMAIN = (0.70, 0.12)

COLUMN = (0.04, 0.06)

SIMULATION_TIME = float(os.environ.get("GT_SATURATED_COLUMN_TIME", "0.5"))

SAVE_PATH = os.environ.get(
    "GT_SATURATED_COLUMN_SAVE_PATH",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SaturatedSoilColumnCollapseSemiImplicit2D"
    ),
)

SOLVER_TYPE = os.environ.get("GT_SATURATED_COLUMN_SOLVER_TYPE", "SemiImplicit")

"""Shared model parameters for u_tube_flow_2d."""

import math
import os

PREFIX = "GT_UTUBE_2D_"


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(PREFIX + name, default))


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(PREFIX + name, default))


DX = env_float("DX", 0.05)

DT = env_float("DT", 5.0e-3)

SIMULATION_TIME = env_float("TIME", 100.0)

PPC = env_int("PPC", 2)

HYDRAULIC_CONDUCTIVITY = env_float("HYDRAULIC_CONDUCTIVITY", 0.05)

WIDTH = 3.0

PHYSICAL_HEIGHT = 3.35

DOMAIN = [WIDTH, 3.4]  # one cell of headroom makes both MGPCG cell counts even

POROUS_ORIGIN = [1.0, 0.0]

POROUS_SIZE = [1.0, 1.0]

LEFT_WATER_DEPTH = 3.0

RIGHT_WATER_DEPTH = 2.0

INITIAL_HEAD_DIFFERENCE = LEFT_WATER_DEPTH - RIGHT_WATER_DEPTH

POROSITY = 0.4


def aligned_count(length: float, spacing: float) -> int:
    count = round(length / spacing)
    if not math.isclose(count * spacing, length, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"geometry length {length:g} is not aligned with spacing={spacing:g}")
    return count


DOMAIN_CELLS = [aligned_count(length, DX) for length in DOMAIN]

FLUID_AREA = LEFT_WATER_DEPTH + POROUS_SIZE[0] * POROUS_SIZE[1] + RIGHT_WATER_DEPTH

EXPECTED_FLUID_PARTICLES = math.ceil(FLUID_AREA * PPC**2 / DX**2)

EXPECTED_SOLID_PARTICLES = math.ceil(math.prod(POROUS_SIZE) * PPC**2 / DX**2)

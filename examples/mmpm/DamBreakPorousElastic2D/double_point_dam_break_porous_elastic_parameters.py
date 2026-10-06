"""Shared model parameters for double_point_dam_break_porous_elastic."""

import math
import os

PREFIX = "GT_DOUBLE_POINT_POROUS_DAM_"


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(PREFIX + name, default))


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(PREFIX + name, default))


DX = env_float("DX", 0.01)

DT = env_float("DT", 1.0e-3)

ALPHA_PIC = env_float("ALPHA_PIC", 1.0)

VELOCITY_PROJECTION = os.environ.get(PREFIX + "VELOCITY_PROJECTION", "PIC")

BACKGROUND_DAMPING = env_float("BACKGROUND_DAMPING", 0.5)

PARTICLE_SHIFTING = os.environ.get(PREFIX + "PARTICLE_SHIFTING", "1") == "1"

PARTICLE_SHIFTING_END_TIME = env_float("PARTICLE_SHIFTING_END_TIME", 15.0)

PARTICLE_SHIFTING_SETTLING_SCALE = env_float("PARTICLE_SHIFTING_SETTLING_SCALE", 1.0)

FLUID_PPC = env_int("FLUID_PPC", 4)

SOLID_PPC = env_int("SOLID_PPC", 4)

POROSITY = 0.39

GRAIN_DIAMETER = 3.0e-3

SOLID_FRACTION = 1.0 - POROSITY

PERMEABILITY = (
    GRAIN_DIAMETER**2
    / 18.0
    / (10.0 * SOLID_FRACTION**2 / POROSITY**3 + SOLID_FRACTION * POROSITY * (1.0 + 1.5 * math.sqrt(SOLID_FRACTION)))
)

EXPERIMENTAL_DOMAIN = [0.892, 0.37]

DOMAIN = [0.90, 0.38]

BASE_WATER_SIZE = [EXPERIMENTAL_DOMAIN[0], 0.025]

UPPER_WATER_SIZE = [0.28, 0.14 - BASE_WATER_SIZE[1]]

POROUS_ORIGIN = [0.30, 0.0]

POROUS_SIZE = [0.29, 0.37]


def aligned_count(length: float, spacing: float) -> int:
    count = round(length / spacing)
    if not math.isclose(count * spacing, length, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"geometry length {length:g} is not aligned with spacing={spacing:g}")
    return count


DOMAIN_CELLS = [aligned_count(length, DX) for length in DOMAIN]

POROUS_RIGHT = POROUS_ORIGIN[0] + POROUS_SIZE[0]

FLUID_PARTICLE_VOLUME = DX**2 / FLUID_PPC**2

SOLID_PARTICLE_VOLUME = DX**2 / SOLID_PPC**2

EXPECTED_FLUID_PARTICLES = math.ceil(math.prod(BASE_WATER_SIZE) / FLUID_PARTICLE_VOLUME) + math.ceil(
    math.prod(UPPER_WATER_SIZE) / FLUID_PARTICLE_VOLUME
)

EXPECTED_SOLID_PARTICLES = math.ceil(math.prod(POROUS_SIZE) / SOLID_PARTICLE_VOLUME)

MAX_PARTICLES = EXPECTED_FLUID_PARTICLES + EXPECTED_SOLID_PARTICLES

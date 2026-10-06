"""Shared physical parameters for the axisymmetric CPT examples.

The values are copied from ``Pile2DAxisy_dem.py`` and are the single source of
truth for the MPM, FEM--MPM, and IGA--MPM CPT drivers.  A coupled driver may
adapt the dimensional representation, an unavailable constitutive model, its
contact controls, and the numerical material needed to model the otherwise-
rigid penetrator, but every adaptation must remain explicit.  Solver-specific
contact parameters are kept separate because DEM penalties and IPC barrier
coefficients do not have the same meaning.
"""

import math

DOMAIN = (0.6, 2.508)
SOIL_ORIGIN = (0.0, 0.0)
SOIL_SIZE = (0.6, 1.5)
GRID_SIZE = 0.006
PARTICLES_PER_CELL = 2
GRAVITY = (0.0, -9.8)
BACKGROUND_DAMPING = 0.05
ALPHA_PIC = 0.05
MAPPING = "USF"
SHAPE_FUNCTION = "GIMP"
STABILIZATION = "B-Bar Method"

SURFACE_ORIGIN = (0.0, SOIL_SIZE[1] - 0.5 * GRID_SIZE)
SURFACE_SIZE = (SOIL_SIZE[0], 0.5 * GRID_SIZE)

MAX_MATERIAL_NUMBER = 1
MAX_PARTICLE_NUMBER = 800_000
MAX_VELOCITY_CONSTRAINT = 134_474
MAX_PARTICLE_TRACTION_CONSTRAINT = 134_474

TIMESTEP = 1.0e-5
SIMULATION_TIME = 10.0
SAVE_INTERVAL = 0.2
PILE_SPEED = 0.1

SURFACE_PRESSURE = 150.0e3
INITIAL_STRESS = (-75.0e3, -150.0e3, -75.0e3, 0.0, 0.0, 0.0)

PILE_PROFILE = (
    (0.001, 1.5000),
    (0.018, 1.5312),
    (0.018, 2.5000),
    (0.001, 2.5000),
)

SOIL_MATERIAL = {
    "MaterialID": 1,
    "Density": 1600.0,
    "YoungModulus": 60.0e6,
    "PoissonRatio": 0.30,
    "e0": 0.62,
    "e_Tao": 0.90,
    "lambda_c": 0.119,
    "ksi": 0.23,
    "nd": 1.70,
    "nf": 2.68,
    "fai_c": 30.0,
    "Cohesion": 3000.0,
}

# Standard rigid-polygon MPM contact from Pile2DAxisy_dem.py.
MPM_DEM_CONTACT = {
    "stiffness": (1.0e5, 1.0e5),
    "friction": 0.0,
}

# Coupling-only values.  These are numerical contact controls, not additional
# soil calibration parameters.  They must be reported independently when a
# coupled CPT result is compared with the standard MPM case.
FEMPM_EXPLICIT_CONTACT = {
    "NormalStiffness": 1.0e7,
    "TangentialStiffness": 1.0e7,
    "Friction": 0.0,
    "NormalViscousDamping": 0.05,
    "TangentialViscousDamping": 0.05,
}

IGAMPM_EXPLICIT_CONTACT = {
    "NormalStiffness": 1.0e7,
    "TangentialStiffness": 1.0e7,
    "StaticFriction": 0.0,
    "DynamicFriction": 0.0,
    "NormalViscousDamping": 0.05,
    "TangentialViscousDamping": 0.05,
}

IPC_CONTACT = {
    # IPC's activation distance is dmin + dhat. Keep it within half a
    # particle spacing so the initially separated pile does not preload the
    # soil through the barrier before prescribed penetration starts.
    "dhat": (0.5 / PARTICLES_PER_CELL - 0.1) * GRID_SIZE,
    "dmin": 0.1 * GRID_SIZE,
    "kappa": 60.0e6,
    "friction_coefficient": 0.0,
    "epsv": 1.0e-4,
}

PENETRATOR_MATERIAL = {
    "density": 7850.0,
    "young_modulus": 200.0e9,
    "poisson_ratio": 0.30,
}

# The coupled explicit implementations are currently Cartesian 3-D.  The
# standard meridian section is extruded through eight standard grid cells;
# this is a numerical representation choice, not a second CPT calibration.
COUPLED_SLICE_THICKNESS = 8.0 * GRID_SIZE
COUPLED_DOMAIN = (DOMAIN[0], COUPLED_SLICE_THICKNESS, DOMAIN[1])
COUPLED_SOIL_SIZE = (SOIL_SIZE[0], COUPLED_SLICE_THICKNESS, SOIL_SIZE[1])
COUPLED_GRAVITY = (GRAVITY[0], 0.0, GRAVITY[1])
COUPLED_INITIAL_STRESS = (
    INITIAL_STRESS[0],
    INITIAL_STRESS[2],
    INITIAL_STRESS[1],
    0.0,
    0.0,
    0.0,
)

# DP is the available constitutive common denominator of native explicit MPM
# and Direct implicit MPM.  Density, elasticity, friction and cohesion come
# from the standard SDMC CPT.  Associated flow is used because it is the
# finite-strain Direct DP implementation shared by both IPC routes.  This is a
# documented model substitution suitable for route validation, not curve
# calibration against the SDMC reference.
COUPLED_DP_MATERIAL = {
    "density": SOIL_MATERIAL["Density"],
    "young_modulus": SOIL_MATERIAL["YoungModulus"],
    "poisson_ratio": SOIL_MATERIAL["PoissonRatio"],
    "friction_angle": SOIL_MATERIAL["fai_c"],
    "dilation_angle": SOIL_MATERIAL["fai_c"],
    "cohesion": SOIL_MATERIAL["Cohesion"],
}


def environment_float(environment, name, default):
    """Read one finite positive runtime override from an environment map."""
    value = float(environment.get(name, default))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value

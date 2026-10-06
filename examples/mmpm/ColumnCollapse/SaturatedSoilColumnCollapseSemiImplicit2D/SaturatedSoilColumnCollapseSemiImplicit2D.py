"""2D single-point adaptation of Ceccato et al.'s saturated column collapse.

The reference uses two independent sets of solid/fluid points.  This case keeps
the measured geometry and material scales but deliberately exercises the
single-layer formulation: the solid update is explicit and only pressure is
solved implicitly (incompressible u-v-p projection, or u-p with Darcy flow and
pressure storage). Phase separation and air entry are outside its scope; the
comparison is qualitative.
"""

import math
import os
import sys

import numpy as np
import taichi as ti

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from examples.mmpm.ColumnCollapse.SaturatedSoilColumnCollapseSemiImplicit2D.SaturatedSoilColumnCollapseSemiImplicit2D_parameters import (
    COLUMN,
    DOMAIN,
    SAVE_PATH,
    SIMULATION_TIME,
    SOLVER_TYPE,
)
from examples.mmpm.ColumnCollapse.SaturatedSoilColumnCollapseSemiImplicit2D.draw.evaluate_SaturatedSoilColumnCollapseSemiImplicit2D import (
    postprocess,
)


from geotaichi import MPM, init

REFERENCE = "Ceccato et al. (2020), Soils and Foundations 60, 683-696"
REFERENCE_DOI = "10.1016/j.sandf.2020.04.004"
# Ceccato et al.'s flume is 0.70 m long.  Keeping the full length prevents the
# released column from striking an artificial right wall during validation.
GRAVITY = 9.8
POROSITY = 0.4
SOLID_DENSITY = 2600.0
FLUID_DENSITY = 1000.0
MAXIMUM_POROSITY = 0.50
INTRINSIC_PERMEABILITY = 1.021e-10
FLUID_VISCOSITY = 1.002e-3
HYDRAULIC_CONDUCTIVITY = INTRINSIC_PERMEABILITY * FLUID_DENSITY * GRAVITY / FLUID_VISCOSITY
# This single-point saturated validation must retain negative gauge pressure.
# At room temperature, water vapor pressure minus atmospheric pressure is about
# -99 kPa (NIST water saturation data); -98 kPa is a conservative lower bound.
# Air entry/desaturation is not modelled. Set 0 explicitly only to reproduce the
# reference double-point model's zero-suction approximation, not as a solver fix.
CAVITATION_PRESSURE = float(os.environ.get("GT_SATURATED_COLUMN_CAVITATION_PRESSURE", "-98000.0"))
FRICTION_ANGLE = float(os.environ.get("GT_SATURATED_COLUMN_FRICTION", "35.0"))
K0 = 1.0 - math.sin(math.radians(FRICTION_ANGLE))

CELL_SIZE = float(os.environ.get("GT_SATURATED_COLUMN_CELL_SIZE", "0.004"))
# The generator accepts a tensor-product count. 4x4 is the smallest layout
# meeting the reference recommendation of 12 material points per active cell.
PARTICLES_PER_CELL = int(os.environ.get("GT_SATURATED_COLUMN_PPC", "4"))
DT = float(os.environ.get("GT_SATURATED_COLUMN_DT", "1.0e-5"))
SAVE_INTERVAL = float(os.environ.get("GT_SATURATED_COLUMN_SAVE_INTERVAL", "0.01"))
ARCH = os.environ.get("GT_SATURATED_COLUMN_ARCH", "gpu")
BACKGROUND_DAMPING = float(os.environ.get("GT_SATURATED_COLUMN_DAMPING", "0.02"))
ALPHA_PIC = float(os.environ.get("GT_SATURATED_COLUMN_ALPHA_PIC", "0.005"))
SHAPE_FUNCTION = os.environ.get("GT_SATURATED_COLUMN_SHAPE_FUNCTION", "GIMP")
if SOLVER_TYPE not in ("SemiImplicit", "SemiImplicit_u_p"):
    raise ValueError("GT_SATURATED_COLUMN_SOLVER_TYPE must be SemiImplicit or SemiImplicit_u_p")
PRESSURE_SOLVER = os.environ.get(
    "GT_SATURATED_COLUMN_PRESSURE_SOLVER",
    "MGPCG" if SOLVER_TYPE == "SemiImplicit" else "PCG",
)
PRESSURE_STABILIZE = os.environ.get("GT_SATURATED_COLUMN_PRESSURE_STABILIZE") or None
PRESSURE_BETA = float(
    os.environ.get("GT_SATURATED_COLUMN_PRESSURE_BETA", "0.0" if PRESSURE_SOLVER == "MGPCG" else "1.0")
)
MULTIGRID_LEVELS = 2
SKIP_POSTPROCESS = os.environ.get("GEOTAICHI_SKIP_POSTPROCESS", "0") == "1"

EXPECTED_PARTICLES = math.ceil(COLUMN[0] * COLUMN[1] * PARTICLES_PER_CELL**2 / CELL_SIZE**2)
MAX_PARTICLES = int(os.environ.get("GT_SATURATED_COLUMN_MAX_PARTICLES", str(EXPECTED_PARTICLES)))
CELL_COUNTS = np.floor((1.0 + 1.0e-6) * np.asarray(DOMAIN) / CELL_SIZE).astype(int)
if PRESSURE_SOLVER == "MGPCG":
    MG_MULTIPLE = 2 ** (MULTIGRID_LEVELS - 1)
    CELL_COUNTS = MG_MULTIPLE * np.ceil(CELL_COUNTS / MG_MULTIPLE).astype(int)
NX, NY = CELL_COUNTS
BOUNDARY_CONSTRAINTS = 3 * (NX + 1) + 2 * NY
MAX_VELOCITY_CONSTRAINTS = int(
    os.environ.get("GT_SATURATED_COLUMN_MAX_VELOCITY_CONSTRAINTS", str(BOUNDARY_CONSTRAINTS))
)


@ti.kernel
def initialize_hydrostatic_column(
    particle_count: int,
    column_height: float,
    gravity: float,
    porosity: float,
    solid_density: float,
    fluid_density: float,
    k0: float,
    particle: ti.template(),
):
    for p in range(particle_count):
        depth = ti.max(column_height - particle[p].x[1], 0.0)
        effective_vertical = (1.0 - porosity) * (solid_density - fluid_density) * gravity * depth
        particle[p].stress = ti.Vector(
            [-k0 * effective_vertical, -effective_vertical, -k0 * effective_vertical, 0.0, 0.0, 0.0]
        )
        particle[p].pressure = fluid_density * gravity * depth


def build_and_run():
    print(
        f"{REFERENCE}: dx={CELL_SIZE:g} m, ppc={PARTICLES_PER_CELL} per axis, "
        f"expected_particles={EXPECTED_PARTICLES}, shape={SHAPE_FUNCTION}, "
        f"solver={SOLVER_TYPE}, pressure_stabilize={PRESSURE_STABILIZE}, "
        f"hydraulic_conductivity={HYDRAULIC_CONDUCTIVITY:.6e} m/s"
    )
    init(dim=2, arch=ARCH, cpu_max_num_threads=4)
    mpm = MPM()
    mpm.set_configuration(
        domain=list(DOMAIN),
        background_damping=BACKGROUND_DAMPING,
        gravity=[0.0, -GRAVITY],
        alphaPIC=ALPHA_PIC,
        mapping="USF" if SOLVER_TYPE == "SemiImplicit_u_p" else "USL",
        shape_function=SHAPE_FUNCTION,
        material_type="TwoPhaseSingleLayer",
        solver_type=SOLVER_TYPE,
        free_surface_detection=True,
    )
    mpm.set_semi_implicit_solver_parameters(
        {
            "assemble_type": "MatrixFree",
            "pressure_solver": PRESSURE_SOLVER,
            "linear_solver": PRESSURE_SOLVER,
            "pressure_beta": PRESSURE_BETA,
            "pressure_stabilize": PRESSURE_STABILIZE,
            "max_iteration_number": 200,
            "residual_tolerance": 1.0e-6,
            "multilevel": MULTIGRID_LEVELS,
            "pre_and_post_smoothing": 2,
            "bottom_smoothing": 8,
        }
    )
    mpm.set_solver(
        solver={
            "Timestep": DT,
            "SimulationTime": SIMULATION_TIME,
            "SaveInterval": SAVE_INTERVAL,
            "SavePath": SAVE_PATH,
        }
    )
    mpm.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_particle_number": MAX_PARTICLES,
            "max_constraint_number": {"max_velocity_constraint": MAX_VELOCITY_CONSTRAINTS},
        }
    )
    mpm.add_material(
        model="MohrCoulomb",
        material={
            "MaterialID": 1,
            "SolidDensity": SOLID_DENSITY,
            "FluidDensity": FLUID_DENSITY,
            "Porosity": POROSITY,
            "MaximumPorosity": MAXIMUM_POROSITY,
            "FluidBulkModulus": 2.0e7,
            "CavitationPressure": CAVITATION_PRESSURE,
            "Permeability": HYDRAULIC_CONDUCTIVITY,
            "FluidViscosity": FLUID_VISCOSITY,
            "DragModel": "Darcy",
            "YoungModulus": 10.0e6,
            "PoissonRatio": 0.3,
            "Cohesion": 0.0,
            "Friction": FRICTION_ANGLE,
            "Dilation": 0.0,
        },
    )
    mpm.add_element(element={"ElementType": "Q4N2D", "ElementSize": [CELL_SIZE, CELL_SIZE]})
    mpm.add_region(
        region={
            "Name": "saturated_column",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": [0.0, 0.0],
            "BoundingBoxSize": list(COLUMN),
        }
    )
    mpm.add_body(
        body={
            "Template": {
                "RegionName": "saturated_column",
                "nParticlesPerCell": PARTICLES_PER_CELL,
                "BodyID": 0,
                "MaterialID": 1,
                "InitialVelocity": [0.0, 0.0],
                "FixVelocity": ["Free", "Free"],
            }
        }
    )
    initialize_hydrostatic_column(
        int(mpm.scene.particleNum[0]),
        COLUMN[1],
        GRAVITY,
        POROSITY,
        SOLID_DENSITY,
        FLUID_DENSITY,
        K0,
        mpm.scene.particle,
    )
    mpm.add_boundary_condition(
        boundary=[
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, 0.0],
                "StartPoint": [0.0, 0.0],
                "EndPoint": [DOMAIN[0], 0.0],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, None],
                "StartPoint": [0.0, 0.0],
                "EndPoint": [0.0, DOMAIN[1]],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, None],
                "StartPoint": [DOMAIN[0], 0.0],
                "EndPoint": [DOMAIN[0], DOMAIN[1]],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [None, 0.0],
                "StartPoint": [0.0, DOMAIN[1]],
                "EndPoint": [DOMAIN[0], DOMAIN[1]],
            },
        ]
    )
    mpm.select_save_data(particle=True, grid=True, object=False)
    mpm.run()
    if not SKIP_POSTPROCESS:
        postprocess()
        mpm.postprocessing(read_path=SAVE_PATH, write_background_grid=True)


if __name__ == "__main__":
    build_and_run()

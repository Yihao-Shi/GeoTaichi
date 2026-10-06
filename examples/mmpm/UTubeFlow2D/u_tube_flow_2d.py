"""Two-dimensional U-tube seepage (Zhang et al., 2027, Section 4.2)."""

import math
import os
import sys
from pathlib import Path

import taichi as ti

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.mmpm.UTubeFlow2D.u_tube_flow_2d_parameters import (
    DOMAIN,
    DOMAIN_CELLS,
    DT,
    DX,
    EXPECTED_FLUID_PARTICLES,
    EXPECTED_SOLID_PARTICLES,
    HYDRAULIC_CONDUCTIVITY,
    INITIAL_HEAD_DIFFERENCE,
    LEFT_WATER_DEPTH,
    PHYSICAL_HEIGHT,
    POROSITY,
    POROUS_ORIGIN,
    POROUS_SIZE,
    PPC,
    PREFIX,
    RIGHT_WATER_DEPTH,
    SIMULATION_TIME,
    WIDTH,
    aligned_count,
    env_float,
)
from examples.mmpm.UTubeFlow2D.draw.evaluate_u_tube_flow_2d import (
    postprocess,
)


from geotaichi import MPM, init  # noqa: E402

ARCH = os.environ.get(PREFIX + "ARCH", "gpu")
OUTPUT = (
    Path(
        os.environ.get(
            PREFIX + "OUTPUT",
            Path(__file__).resolve().parent / "OutputData" / "paper_section_4_2_2d",
        )
    )
    .expanduser()
    .resolve()
)
SAVE_INTERVAL = env_float("SAVE_INTERVAL", 2.0)
SKIP_POSTPROCESS = os.environ.get(PREFIX + "SKIP_POSTPROCESS", "0") == "1"


MAX_PARTICLES = EXPECTED_FLUID_PARTICLES + EXPECTED_SOLID_PARTICLES
MAX_VELOCITY_CONSTRAINTS = 2 * (aligned_count(POROUS_SIZE[0], DX) + 1) * (aligned_count(POROUS_SIZE[1], DX) + 1)

if min(DX, DT, SIMULATION_TIME, SAVE_INTERVAL, HYDRAULIC_CONDUCTIVITY) <= 0.0 or PPC <= 0:
    raise ValueError("spacing, time, conductivity, and particles per cell must be positive")
if any(count % 2 for count in DOMAIN_CELLS):
    raise ValueError("the two-level MGPCG grid must have even cell counts")


@ti.kernel
def initialize_pressure(particle_count: int, particle: ti.template()):
    for p in range(particle_count):
        x = particle[p].x[0]
        head = LEFT_WATER_DEPTH
        if x > POROUS_ORIGIN[0]:
            head = LEFT_WATER_DEPTH - (x - POROUS_ORIGIN[0]) * INITIAL_HEAD_DIFFERENCE / POROUS_SIZE[0]
        if x >= POROUS_ORIGIN[0] + POROUS_SIZE[0]:
            head = RIGHT_WATER_DEPTH
        particle[p].pressure = 1000.0 * 9.81 * ti.max(head - particle[p].x[1], 0.0)


if os.environ.get(PREFIX + "POSTPROCESS_ONLY", "0") == "1":
    postprocess(OUTPUT)
    raise SystemExit


print("# Zhang et al. (2027), Section 4.2: two-dimensional U-tube seepage")
print(
    f"# domain={DOMAIN}, cells={DOMAIN_CELLS}, dx={DX:g}, ppc={PPC}, "
    f"particles={MAX_PARTICLES}, steps={math.ceil(SIMULATION_TIME / DT)}"
)

init(dim=2, arch=ARCH, default_fp="float64", device_memory_GB=env_float("DEVICE_MEMORY_GB", 2.0))
mpm = MPM()
mpm.set_configuration(
    domain=DOMAIN,
    background_damping=0.01,
    gravity=[0.0, -9.81],
    alphaPIC=1.0,
    mapping="USL",
    shape_function="QuadBSpline",
    material_type="TwoPhaseDoubleLayer",
    solver_type="SemiImplicit",
    velocity_projection="Affine",
    delayed_fluid_advection=True,
    particle_shifting=True,
    visualize=True,
)
mpm.set_solver(
    {
        "Timestep": DT,
        "SimulationTime": SIMULATION_TIME,
        "SaveInterval": SAVE_INTERVAL,
        "SavePath": str(OUTPUT),
    }
)
mpm.set_semi_implicit_solver_parameters(
    {
        "assemble_type": "MatrixFree",
        "pressure_solver": "MGPCG",
        "linear_solver": "MGPCG",
        "max_iteration_number": 200,
        "residual_tolerance": 1.0e-7,
        "multilevel": 2,
        "pre_and_post_smoothing": 2,
        "bottom_smoothing": 8,
    }
)
mpm.memory_allocate(
    {
        "max_material_number": 2,
        "max_particle_number": MAX_PARTICLES,
        "max_constraint_number": {"max_velocity_constraint": MAX_VELOCITY_CONSTRAINTS},
    }
)
mpm.add_material(
    model="LinearElastic",
    material={
        "MaterialID": 1,
        "SolidDensity": 2650.0,
        "FluidDensity": 1000.0,
        "Porosity": POROSITY,
        "FluidBulkModulus": 2.2e8,
        "Permeability": HYDRAULIC_CONDUCTIVITY,
        "FluidViscosity": 1.0e-3,
        "GrainDiameter": 3.0e-3,
        "DragModel": "Darcy",
        "YoungModulus": 4.0e7,
        "PoissonRatio": 0.25,
    },
)
mpm.add_material(
    model="LinearElastic",
    material={
        "MaterialID": 2,
        "SolidDensity": 2650.0,
        "FluidDensity": 1000.0,
        # The constitutive input requires an open interval; 0.9999 is the
        # solver's own clear-fluid cap and is numerically equivalent to one.
        "Porosity": 0.9999,
        "FluidBulkModulus": 2.2e8,
        "Permeability": HYDRAULIC_CONDUCTIVITY,
        "FluidViscosity": 1.0e-3,
        "GrainDiameter": 3.0e-3,
        "DragModel": "Darcy",
        "YoungModulus": 4.0e7,
        "PoissonRatio": 0.25,
    },
)
mpm.add_element(element={"ElementType": "Q4N2D", "ElementSize": [DX, DX]})
mpm.add_region(
    region=[
        {
            "Name": "left_water",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": [0.0, 0.0],
            "BoundingBoxSize": [1.0, LEFT_WATER_DEPTH],
        },
        {
            "Name": "porous_water",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": POROUS_ORIGIN,
            "BoundingBoxSize": POROUS_SIZE,
        },
        {
            "Name": "right_water",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": [2.0, 0.0],
            "BoundingBoxSize": [1.0, RIGHT_WATER_DEPTH],
        },
        {
            "Name": "fixed_porous_skeleton",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": POROUS_ORIGIN,
            "BoundingBoxSize": POROUS_SIZE,
        },
    ]
)
mpm.add_body(
    body={
        "Template": [
            {
                "RegionName": region,
                "nParticlesPerCell": PPC,
                "BodyID": 0,
                "MaterialID": 2,
                "Phase": "Fluid",
                "InitialVelocity": [0.0, 0.0],
                "FixVelocity": ["Free", "Free"],
            }
            for region in ("left_water", "right_water")
        ]
        + [
            {
                "RegionName": "porous_water",
                "nParticlesPerCell": PPC,
                "BodyID": 0,
                "MaterialID": 1,
                "Phase": "Fluid",
                "InitialVelocity": [0.0, 0.0],
                "FixVelocity": ["Free", "Free"],
            },
            {
                "RegionName": "fixed_porous_skeleton",
                "nParticlesPerCell": PPC,
                "BodyID": 0,
                "MaterialID": 1,
                "Phase": "Solid",
                "InitialVelocity": [0.0, 0.0],
                "FixVelocity": ["Fix", "Fix"],
            },
        ]
    }
)
initialize_pressure(int(mpm.scene.particleNum[0]), mpm.scene.particle)

walls = [
    ([0.0, 0.0], [0.0, PHYSICAL_HEIGHT], [-1.0, 0.0]),
    ([WIDTH, 0.0], [WIDTH, PHYSICAL_HEIGHT], [1.0, 0.0]),
    ([0.0, 0.0], [WIDTH, 0.0], [0.0, -1.0]),
    ([1.0, 1.0], [1.0, PHYSICAL_HEIGHT], [1.0, 0.0]),
    ([2.0, 1.0], [2.0, PHYSICAL_HEIGHT], [-1.0, 0.0]),
    ([1.0, 1.0], [2.0, 1.0], [0.0, 1.0]),
]
mpm.add_boundary_condition(
    boundary=[
        {
            "BoundaryType": "SolidCell",
            "StartPoint": start,
            "EndPoint": end,
            "Norm": normal,
            "CellThickness": 1,
        }
        for start, end, normal in walls
    ]
    + [
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, 0.0],
            "StartPoint": POROUS_ORIGIN,
            "EndPoint": [2.0, 1.0],
        }
    ]
)
mpm.select_save_data(particle=True, grid=False, object=False)
mpm.run()

if not SKIP_POSTPROCESS:
    mpm.postprocessing()
postprocess(OUTPUT)

"""Juel et al. Section 5.5 case 1: dam break through a rigid porous column."""

import os
import sys
from pathlib import Path

import taichi as ti

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.mmpm.DamBreakPorousElastic2D.double_point_dam_break_porous_elastic_parameters import (
    ALPHA_PIC,
    BACKGROUND_DAMPING,
    BASE_WATER_SIZE,
    DOMAIN,
    DOMAIN_CELLS,
    DT,
    DX,
    EXPECTED_FLUID_PARTICLES,
    EXPECTED_SOLID_PARTICLES,
    EXPERIMENTAL_DOMAIN,
    FLUID_PPC,
    GRAIN_DIAMETER,
    MAX_PARTICLES,
    PARTICLE_SHIFTING,
    PARTICLE_SHIFTING_END_TIME,
    PARTICLE_SHIFTING_SETTLING_SCALE,
    PERMEABILITY,
    POROSITY,
    POROUS_ORIGIN,
    POROUS_RIGHT,
    POROUS_SIZE,
    PREFIX,
    SOLID_PPC,
    UPPER_WATER_SIZE,
    VELOCITY_PROJECTION,
    aligned_count,
    env_float,
)
from examples.mmpm.DamBreakPorousElastic2D.draw.evaluate_double_point_dam_break_porous_elastic import (
    postprocess,
)


from geotaichi import MPM, init  # noqa: E402

ARCH = os.environ.get(PREFIX + "ARCH", "gpu")
OUTPUT = Path(os.environ.get(PREFIX + "OUTPUT", Path(__file__).resolve().parent / "OutputData")).expanduser().resolve()
SIMULATION_TIME = env_float("TIME", 70.0)
SAVE_INTERVAL = env_float("SAVE_INTERVAL", 0.2)
# Eq. (22), the zero-relative-velocity limit of the Beetstra model.
YOUNG_MODULUS = env_float("YOUNG_MODULUS", 4.0e7)
SKIP_POSTPROCESS = os.environ.get(PREFIX + "SKIP_POSTPROCESS", "0") == "1"

# The physical flume is embedded in a cell-aligned computational box.  Walls
# remain at the experimental dimensions.
BASE_WATER_ORIGIN = [0.0, 0.0]
UPPER_WATER_ORIGIN = [0.0, BASE_WATER_SIZE[1]]


MAX_VELOCITY_CONSTRAINTS = 2 * (aligned_count(POROUS_SIZE[0], DX) + 1) * (aligned_count(POROUS_SIZE[1], DX) + 1)

if min(DX, DT, SIMULATION_TIME, SAVE_INTERVAL, PERMEABILITY, YOUNG_MODULUS) <= 0.0 or min(FLUID_PPC, SOLID_PPC) <= 0:
    raise ValueError("spacing, time, permeability, modulus, and particles per cell must be positive")
if not 0.0 <= PARTICLE_SHIFTING_SETTLING_SCALE <= 1.0:
    raise ValueError("particle shifting settling scale must be in [0, 1]")
if any(count % 2 for count in DOMAIN_CELLS):
    raise ValueError("the two-level MGPCG grid must have even cell counts")


@ti.kernel
def initialize_hydrostatic_pressure(particle_count: int, particle: ti.template()):
    for p in range(particle_count):
        if int(particle[p].phase) == 2:
            water_top = BASE_WATER_SIZE[1]
            if particle[p].x[0] <= UPPER_WATER_SIZE[0]:
                water_top = UPPER_WATER_ORIGIN[1] + UPPER_WATER_SIZE[1]
            particle[p].pressure = 1000.0 * 9.81 * ti.max(water_top - particle[p].x[1], 0.0)


if os.environ.get(PREFIX + "POSTPROCESS_ONLY", "0") == "1":
    postprocess(OUTPUT)
    raise SystemExit


print("# Juel et al. Section 5.5 case 1: dam break through a rigid porous column")
print(f"# domain={DOMAIN}, cells={DOMAIN_CELLS}, dx={DX:g}, " f"fluid_ppc={FLUID_PPC}, solid_ppc={SOLID_PPC}")
print(
    f"# exact capacity={MAX_PARTICLES} ({EXPECTED_FLUID_PARTICLES} fluid + "
    f"{EXPECTED_SOLID_PARTICLES} solid), Beetstra rest permeability={PERMEABILITY:g} m^2"
)

init(dim=2, arch=ARCH, default_fp="float64", device_memory_GB=env_float("DEVICE_MEMORY_GB", 4.0))
mpm = MPM()
mpm.set_configuration(
    domain=DOMAIN,
    background_damping=(
        0.0 if PARTICLE_SHIFTING and 0.0 < PARTICLE_SHIFTING_END_TIME < SIMULATION_TIME else BACKGROUND_DAMPING
    ),
    gravity=[0.0, -9.81],
    alphaPIC=ALPHA_PIC,
    mapping="USL",
    shape_function="QuadBSpline",
    material_type="TwoPhaseDoubleLayer",
    solver_type="SemiImplicit",
    velocity_projection=VELOCITY_PROJECTION,
    delayed_fluid_advection=True,
    particle_shifting=PARTICLE_SHIFTING,
    particle_shifting_scale=1.0,
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
        "max_material_number": 1,
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
        "Permeability": PERMEABILITY,
        "FluidViscosity": 1.0e-3,
        "GrainDiameter": GRAIN_DIAMETER,
        "DragModel": "Beetstra",
        "YoungModulus": YOUNG_MODULUS,
        "PoissonRatio": 0.25,
    },
)
mpm.add_element(element={"ElementType": "Q4N2D", "ElementSize": [DX, DX]})
mpm.add_region(
    region=[
        {
            "Name": "base_water",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": BASE_WATER_ORIGIN,
            "BoundingBoxSize": BASE_WATER_SIZE,
        },
        {
            "Name": "upper_water_column",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": UPPER_WATER_ORIGIN,
            "BoundingBoxSize": UPPER_WATER_SIZE,
        },
        {
            "Name": "rigid_porous_block",
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
                "RegionName": "base_water",
                "nParticlesPerCell": FLUID_PPC,
                "BodyID": 0,
                "MaterialID": 1,
                "Phase": "Fluid",
                "InitialVelocity": [0.0, 0.0],
                "FixVelocity": ["Free", "Free"],
            },
            {
                "RegionName": "upper_water_column",
                "nParticlesPerCell": FLUID_PPC,
                "BodyID": 0,
                "MaterialID": 1,
                "Phase": "Fluid",
                "InitialVelocity": [0.0, 0.0],
                "FixVelocity": ["Free", "Free"],
            },
            {
                "RegionName": "rigid_porous_block",
                "nParticlesPerCell": SOLID_PPC,
                "BodyID": 0,
                "MaterialID": 1,
                "Phase": "Solid",
                "InitialVelocity": [0.0, 0.0],
                "FixVelocity": ["Fix", "Fix"],
            },
        ]
    }
)
initialize_hydrostatic_pressure(int(mpm.scene.particleNum[0]), mpm.scene.particle)

walls = [
    ([0.0, 0.0], [0.0, EXPERIMENTAL_DOMAIN[1]], [-1.0, 0.0]),
    ([EXPERIMENTAL_DOMAIN[0], 0.0], EXPERIMENTAL_DOMAIN, [1.0, 0.0]),
    ([0.0, 0.0], [EXPERIMENTAL_DOMAIN[0], 0.0], [0.0, -1.0]),
    ([0.0, EXPERIMENTAL_DOMAIN[1]], EXPERIMENTAL_DOMAIN, [0.0, 1.0]),
]
mpm.add_boundary_condition(
    boundary=[
        {
            "BoundaryType": "SolidCell",
            "StartPoint": start,
            "EndPoint": end,
            "Norm": normal,
            "CellThickness": 2,
        }
        for start, end, normal in walls
    ]
    + [
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, 0.0],
            "StartPoint": POROUS_ORIGIN,
            "EndPoint": [POROUS_RIGHT, POROUS_ORIGIN[1] + POROUS_SIZE[1]],
        }
    ]
)
mpm.select_save_data(particle=True, grid=False, object=False)
if PARTICLE_SHIFTING and 0.0 < PARTICLE_SHIFTING_END_TIME < SIMULATION_TIME:
    mpm.modify_parameters(SimulationTime=PARTICLE_SHIFTING_END_TIME)
    mpm.run()
    mpm.sims.set_particle_shifting_scale(PARTICLE_SHIFTING_SETTLING_SCALE)
    mpm.modify_parameters(SimulationTime=SIMULATION_TIME, background_damping=BACKGROUND_DAMPING)
mpm.run()

if not SKIP_POSTPROCESS:
    mpm.postprocessing()
postprocess(OUTPUT)

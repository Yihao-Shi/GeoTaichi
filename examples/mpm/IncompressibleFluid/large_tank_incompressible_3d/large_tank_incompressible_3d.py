import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

DT = float(os.environ.get("GEOTAICHI_LARGE_TANK_DT", "1.0e-4"))
SIMULATION_TIME = float(os.environ.get("GEOTAICHI_LARGE_TANK_TIME", "5.0"))
SAVE_INTERVAL = float(os.environ.get("GEOTAICHI_LARGE_TANK_SAVE_INTERVAL", "0.1"))
SAVE_PATH = os.environ.get(
    "GEOTAICHI_LARGE_TANK_SAVE_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "OutputData", "large_tank_incompressible_3d"),
)
LINEAR_SOLVER = os.environ.get("GEOTAICHI_LARGE_TANK_LINEAR_SOLVER", "MGPCG")
PARTICLE_SHIFTING = os.environ.get("GEOTAICHI_LARGE_TANK_PARTICLE_SHIFTING", "1") != "0"
DENSITY_PROJECTION = os.environ.get("GEOTAICHI_LARGE_TANK_DENSITY_PROJECTION", "1") != "0"
RESTART_FROM = int(os.environ.get("GEOTAICHI_LARGE_TANK_RESTART_FROM", "0"))
RESTART_PATH = os.environ.get("GEOTAICHI_LARGE_TANK_RESTART_PATH", SAVE_PATH)
N_PARTICLES_PER_CELL = int(os.environ.get("GEOTAICHI_LARGE_TANK_PARTICLES_PER_CELL", "4"))
MAX_PARTICLE_NUMBER = int(os.environ.get("GEOTAICHI_LARGE_TANK_MAX_PARTICLES", "2700000"))
ARCH = os.environ.get("GEOTAICHI_ARCH", "cpu")
DEFAULT_DEVICE_MEMORY_GB = "5.5" if ARCH.lower() in ("gpu", "cuda") else "2"

init(
    dim=3,
    arch=ARCH,
    device_memory_GB=float(os.environ.get("GEOTAICHI_DEVICE_MEMORY_GB", DEFAULT_DEVICE_MEMORY_GB)),
    debug=False,
    kernel_profiler=os.environ.get("GEOTAICHI_KERNEL_PROFILER", "1") != "0",
)

mpm = MPM()

mpm.set_configuration(
    domain=[6.0, 1.42, 6.0],
    background_damping=0.00,
    alphaPIC=0.5,
    mapping="USL",
    shape_function="QuadBSpline",
    gravity=[0.0, 0.0, -9.8],
    material_type="Fluid",
    velocity_projection="Affine",
    solver_type="Implicit",
    discretization="FDM",
    fluid_level_set=True,
    fluid_domain_volume_fraction=0.2,
    particle_shifting=PARTICLE_SHIFTING,
    density_projection=DENSITY_PROJECTION,
)

mpm.set_implicit_solver_parameters(
    linear_solver=LINEAR_SOLVER,
    multilevel=4,
    pre_and_post_smoothing=2,
    bottom_smoothing=20,
    max_iteration_number=200,
    residual_tolerance=1e-8,
)

mpm.set_solver(
    {"Timestep": DT, "SimulationTime": SIMULATION_TIME, "SaveInterval": SAVE_INTERVAL, "SavePath": SAVE_PATH}
)

mpm.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_particle_number": MAX_PARTICLE_NUMBER,
        "verlet_distance_multiplier": 1.0,
        "max_constraint_number": {},
    }
)

mpm.add_material(
    model="Newtonian",
    material={
        "MaterialID": 1,
        "Density": 1000.0,
        "Modulus": 2e6,
        "Viscosity": 1e-3,
        "ElementLength": 0.01,
        "cL": 1.5,
        "cQ": 2,
        "atmospheric_pressure": 0.01,
    },
)

mpm.add_element(element={"ElementType": "Staggered", "ElementSize": [0.05, 0.05, 0.05], "GhostCell": 1})


mpm.add_region(
    region=[
        {
            "Name": "region1",
            "Type": "Rectangle",
            "BoundingBoxPoint": [0.0, 0.0, 0.0],
            "BoundingBoxSize": [2.24, 1.42, 1.42],
        }
    ]
)

if RESTART_FROM > 0:
    mpm.read_restart(RESTART_FROM, RESTART_PATH, is_continue=True)
else:
    mpm.add_body(
        body={
            "Template": [
                {
                    "RegionName": "region1",
                    "nParticlesPerCell": N_PARTICLES_PER_CELL,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "InitialVelocity": [0, 0, 0],
                    "FixVelocity": ["Free", "Free", "Free"],
                }
            ]
        }
    )


mpm.add_boundary_condition(
    boundary=[
        {
            "BoundaryType": "SolidCell",
            "Norm": [-1.0, 0.0, 0.0],
            "StartPoint": [0.0, 0.0, 0.0],
            "EndPoint": [0.0, 1.42, 6.0],
            "CellThickness": 1,
        },
        {
            "BoundaryType": "SolidCell",
            "Norm": [1.0, 0.0, 0.0],
            "StartPoint": [6.0, 0.0, 0.0],
            "EndPoint": [6.0, 1.42, 6.0],
            "CellThickness": 1,
        },
        {
            "BoundaryType": "SolidCell",
            "Norm": [0.0, -1.0, 0.0],
            "StartPoint": [0.0, 0.0, 0.0],
            "EndPoint": [6.0, 0.0, 6.0],
            "CellThickness": 1,
        },
        {
            "BoundaryType": "SolidCell",
            "Norm": [0.0, 1.0, 0.0],
            "StartPoint": [0.0, 1.42, 0.0],
            "EndPoint": [6.0, 1.42, 6.0],
            "CellThickness": 1,
        },
        {
            "BoundaryType": "SolidCell",
            "Norm": [0.0, 0.0, -1.0],
            "StartPoint": [0.0, 0.0, 0.0],
            "EndPoint": [6.0, 1.42, 0.0],
            "CellThickness": 1,
        },
        {
            "BoundaryType": "SolidCell",
            "Norm": [0.0, 0.0, 1.0],
            "StartPoint": [0.0, 0.0, 6.0],
            "EndPoint": [6.0, 1.42, 6.0],
            "CellThickness": 1,
        },
    ]
)


mpm.select_save_data(grid=True)

mpm.run(gravity_field=True)

if os.environ.get("GEOTAICHI_SKIP_POSTPROCESS", "0") != "1":
    mpm.postprocessing()

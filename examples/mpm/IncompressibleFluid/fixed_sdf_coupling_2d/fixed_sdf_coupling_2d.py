import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *


ARCH = os.environ.get("GEOTAICHI_ARCH", "cpu")
DT = float(os.environ.get("GEOTAICHI_INCOMP_SDF_2D_DT", "2.0e-4"))
SIMULATION_TIME = float(os.environ.get("GEOTAICHI_INCOMP_SDF_2D_TIME", "2.0e-2"))
SAVE_INTERVAL = float(os.environ.get("GEOTAICHI_INCOMP_SDF_2D_SAVE_INTERVAL", "2.0e-3"))
SAVE_PATH = os.environ.get("GEOTAICHI_INCOMP_SDF_2D_SAVE_PATH", "fixed_sdf_coupling_2d")
DX = float(os.environ.get("GEOTAICHI_INCOMP_SDF_2D_DX", "0.01"))
PPC = int(os.environ.get("GEOTAICHI_INCOMP_SDF_2D_PPC", "3"))
MAX_PARTICLES = int(os.environ.get("GEOTAICHI_INCOMP_SDF_2D_MAX_PARTICLES", "50000"))
DOMAIN = [0.50, 0.30]
WATER_LOWER = [0.03, 0.02]
WATER_SIZE = [0.22, 0.20]
OBSTACLE_CENTER = [
    float(os.environ.get("GEOTAICHI_INCOMP_SDF_2D_OBSTACLE_X", "0.19")),
    float(os.environ.get("GEOTAICHI_INCOMP_SDF_2D_OBSTACLE_Y", "0.12")),
]
OBSTACLE_RADIUS = float(os.environ.get("GEOTAICHI_INCOMP_SDF_2D_OBSTACLE_RADIUS", "0.035"))


@ti.kernel
def kernel_add_circular_solid_sdf(
    center: ti.types.vector(2, float), radius: float, grid_size: ti.types.vector(2, float), solid_sdf: ti.template()
):
    for I in ti.grouped(solid_sdf):
        position = (I.cast(float) + 0.5) * grid_size
        phi = (position - center).norm() - radius
        ti.atomic_min(solid_sdf[I], phi)


def add_circular_obstacle_sdf(sims, scene):
    kernel_add_circular_solid_sdf(
        ti.Vector(OBSTACLE_CENTER), OBSTACLE_RADIUS, scene.element.grid_size, scene.element.cell.solid_sdf
    )


init(dim=2, arch=ARCH, cpu_max_num_threads=4, device_memory_GB=2, debug=False, kernel_profiler=True)

mpm = MPM()

print("# 2D incompressible MPM: dam-break water column around a fixed circular SDF obstacle")
print(f"# solid_sdf: phi(x) = ||x - {OBSTACLE_CENTER}|| - {OBSTACLE_RADIUS}")
print("# grid output stores cell_solid_sdf, cell_fluid_sdf, and cell_pressure")

mpm.set_configuration(
    domain=DOMAIN,
    background_damping=0.0,
    alphaPIC=0.5,
    mapping="USL",
    shape_function="QuadBSpline",
    gravity=[0.0, -9.8],
    material_type="Fluid",
    velocity_projection="Affine",
    solver_type="Implicit",
    discretization="FDM",
    fluid_level_set=True,
    fluid_domain_volume_fraction=0.15,
    solid_sdf_cut_cell=True,
    solid_cut_cell_min_fraction=0.05,
    particle_shifting=True,
    density_projection=True,
    density_projection_interior_only=False,
    visualize=False,
)

mpm.set_implicit_solver_parameters(
    linear_solver="MGPCG",
    multilevel=3,
    pre_and_post_smoothing=2,
    bottom_smoothing=20,
    max_iteration_number=120,
    residual_tolerance=1.0e-8,
)

mpm.set_solver(
    {"Timestep": DT, "SimulationTime": SIMULATION_TIME, "SaveInterval": SAVE_INTERVAL, "SavePath": SAVE_PATH}
)

mpm.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_particle_number": MAX_PARTICLES,
        "verlet_distance_multiplier": 1.0,
        "max_constraint_number": {},
    }
)

mpm.add_material(
    model="Newtonian",
    material={
        "MaterialID": 1,
        "Density": 1000.0,
        "Modulus": 2.0e6,
        "Viscosity": 1.0e-3,
        "ElementLength": DX,
        "cL": 1.5,
        "cQ": 2.0,
        "atmospheric_pressure": 0.0,
        "SurfaceTension": 0.0,
    },
)

mpm.add_element(element={"ElementType": "Staggered", "ElementSize": [DX, DX], "GhostCell": 1})

mpm.add_region(
    region=[
        {
            "Name": "water_column",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": WATER_LOWER,
            "BoundingBoxSize": WATER_SIZE,
            "rotate2D": 0.0,
        }
    ]
)

mpm.add_body(
    body={
        "Template": [
            {
                "RegionName": "water_column",
                "nParticlesPerCell": PPC,
                "BodyID": 0,
                "MaterialID": 1,
                "InitialVelocity": [0.0, 0.0],
                "FixVelocity": ["Free", "Free"],
            }
        ]
    }
)

mpm.add_boundary_condition(
    boundary=[
        {
            "BoundaryType": "SolidCell",
            "Norm": [0.0, -1.0],
            "StartPoint": [0.0, 0.0],
            "EndPoint": [DOMAIN[0], 0.0],
            "CellThickness": 1,
        },
        {
            "BoundaryType": "SolidCell",
            "Norm": [-1.0, 0.0],
            "StartPoint": [0.0, 0.0],
            "EndPoint": [0.0, DOMAIN[1]],
            "CellThickness": 1,
        },
        {
            "BoundaryType": "SolidCell",
            "Norm": [1.0, 0.0],
            "StartPoint": [DOMAIN[0], 0.0],
            "EndPoint": [DOMAIN[0], DOMAIN[1]],
            "CellThickness": 1,
        },
        {
            "BoundaryType": "SolidCell",
            "Norm": [0.0, 1.0],
            "StartPoint": [0.0, DOMAIN[1]],
            "EndPoint": [DOMAIN[0], DOMAIN[1]],
            "CellThickness": 1,
        },
    ]
)

mpm.select_save_data(particle=True, grid=True)

mpm.run(gravity_field=True, cut_cell_function=add_circular_obstacle_sdf)

if os.environ.get("GEOTAICHI_SKIP_POSTPROCESS", "0") != "1":
    mpm.postprocessing()

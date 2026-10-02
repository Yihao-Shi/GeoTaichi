"""Visually explicit 2-D dam break around a fixed square SDF obstacle.

The tall blue water column collapses under gravity, splits around the square
obstacle, and rejoins downstream inside a closed tank.  The obstacle and tank
are written as actual static VTU geometry so a gallery renderer does not have
to infer an otherwise invisible SDF from grid cell data.

The run writes the following files below ``<output>/vtks``:

* ``GraphicMPMParticle*.vtu`` -- water particles (about 51 frames by default),
* ``GraphicMPMGrid*.vtu`` -- pressure grid, including fluid/solid SDF fields,
* ``GalleryObstacle000000.vtu`` -- fixed square obstacle, and
* ``GalleryTank000000.vtu`` -- static tank linework.

Suggested L40S command::

    GEOTAICHI_GALLERY_SDF_2D_ARCH=gpu \
    GEOTAICHI_GALLERY_SDF_2D_OUTPUT=OutputData/gallery/dam_break_visible_sdf_2d \
    GEOTAICHI_SKIP_POSTPROCESS=1 \
    python examples/mpm/IncompressibleFluid/dam_break_visible_sdf_2d.py

Set ``GEOTAICHI_GALLERY_SDF_2D_ARCH=cpu`` for a CPU smoke run.  Timestep,
duration, grid spacing, output path, and save interval are configurable through
the similarly named ``..._DT``, ``..._TIME``, ``..._DX``, ``..._OUTPUT``, and
``..._SAVE_INTERVAL`` environment variables.
"""

import os
import sys

import meshio
import numpy as np
import taichi as ti

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import MPM, init

ENV_PREFIX = "GEOTAICHI_GALLERY_SDF_2D_"
ARCH = os.environ.get(f"{ENV_PREFIX}ARCH", os.environ.get("GEOTAICHI_ARCH", "gpu"))
DT = float(os.environ.get(f"{ENV_PREFIX}DT", "2.0e-4"))
SIMULATION_TIME = float(os.environ.get(f"{ENV_PREFIX}TIME", "4.0"))
SAVE_INTERVAL = float(os.environ.get(f"{ENV_PREFIX}SAVE_INTERVAL", str(SIMULATION_TIME / 50.0)))
SAVE_PATH = os.environ.get(
    f"{ENV_PREFIX}OUTPUT",
    "OutputData/gallery/dam_break_visible_sdf_2d",
)
DX = float(os.environ.get(f"{ENV_PREFIX}DX", "0.01"))
PPC = int(os.environ.get(f"{ENV_PREFIX}PPC", "3"))
MAX_PARTICLES = int(os.environ.get(f"{ENV_PREFIX}MAX_PARTICLES", "120000"))

DOMAIN = [0.90, 0.45]
WATER_LOWER = [0.04, 0.025]
WATER_SIZE = [0.23, 0.34]
OBSTACLE_LOWER = [0.35, 0.06]
OBSTACLE_SIZE = [0.11, 0.11]


def add_box_obstacle_sdf(sims, scene):
    """Insert the fixed square into the pressure grid's solid distance field."""
    del sims
    from src.mpm.engines.EngineKernel import kernel_update_solid_sdf_from_box

    lower = np.asarray(OBSTACLE_LOWER)
    kernel_update_solid_sdf_from_box(
        scene.element.grid_size,
        ti.Vector(lower),
        ti.Vector(lower + np.asarray(OBSTACLE_SIZE)),
        scene.element.cell.solid_sdf,
    )


def write_gallery_geometry(save_path):
    """Write visible static geometry matching the analytic SDF and tank walls."""
    vtk_path = os.path.join(save_path, "vtks")
    os.makedirs(vtk_path, exist_ok=True)

    x0, y0 = OBSTACLE_LOWER
    x1, y1 = np.asarray(OBSTACLE_LOWER) + np.asarray(OBSTACLE_SIZE)
    obstacle_points = np.asarray(
        ((x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)),
        dtype=np.float64,
    )
    obstacle = meshio.Mesh(
        points=obstacle_points,
        cells=[("quad", np.asarray(((0, 1, 2, 3),), dtype=np.int32))],
        cell_data={"gallery_feature": [np.ones(1, dtype=np.int32)]},
    )
    meshio.write(
        os.path.join(vtk_path, "GalleryObstacle000000.vtu"),
        obstacle,
        file_format="vtu",
    )

    tank_points = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (DOMAIN[0], 0.0, 0.0),
            (DOMAIN[0], DOMAIN[1], 0.0),
            (0.0, DOMAIN[1], 0.0),
        ],
        dtype=np.float64,
    )
    tank_lines = np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.int32)
    tank = meshio.Mesh(
        points=tank_points,
        cells=[("line", tank_lines)],
        cell_data={"gallery_feature": [np.full(4, 2, dtype=np.int32)]},
    )
    meshio.write(
        os.path.join(vtk_path, "GalleryTank000000.vtu"),
        tank,
        file_format="vtu",
    )


init(
    dim=2,
    arch=ARCH,
    cpu_max_num_threads=4,
    device_memory_GB=4,
    debug=False,
    kernel_profiler=False,
)

mpm = MPM()

print("# Gallery: 2-D incompressible dam break around a visible square SDF obstacle")
print(f"# obstacle: lower={OBSTACLE_LOWER}, size={OBSTACLE_SIZE}")
print(f"# output: {SAVE_PATH}/vtks (about {round(SIMULATION_TIME / SAVE_INTERVAL) + 1} frames)")

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
    density_projection_tolerance=0.008,
    density_projection_error_clamp=0.02,
    density_projection_max_shift_ratio=0.03,
    density_projection_interior_only=False,
    visualize=True,
)

mpm.set_implicit_solver_parameters(
    linear_solver="MGPCG",
    multilevel=3,
    pre_and_post_smoothing=2,
    bottom_smoothing=24,
    max_iteration_number=160,
    residual_tolerance=1.0e-8,
)

mpm.set_solver(
    {
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

mpm.add_element(
    element={
        "ElementType": "Staggered",
        "ElementSize": [DX, DX],
        "GhostCell": 1,
    }
)

mpm.add_region(
    region=[
        {
            "Name": "tall_water_column",
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
                "RegionName": "tall_water_column",
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
write_gallery_geometry(SAVE_PATH)

mpm.run(gravity_field=True, cut_cell_function=add_box_obstacle_sdf)

if os.environ.get("GEOTAICHI_SKIP_POSTPROCESS", "0") != "1":
    mpm.postprocessing()

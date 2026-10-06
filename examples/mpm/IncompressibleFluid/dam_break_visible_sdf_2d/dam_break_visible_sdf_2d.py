"""Visually explicit 2-D dam break around a fixed five-point-star SDF obstacle.

The tall blue water column collapses under gravity, splits around the polygonal
obstacle, and rejoins downstream inside a closed tank.  The obstacle and tank
are written as actual static VTU geometry so a gallery renderer does not have
to infer an otherwise invisible SDF from grid cell data.

The run writes the following files below ``<output>/vtks``:

* ``GraphicMPMParticle*.vtu`` -- water particles (about 51 frames by default),
* ``GraphicMPMGrid*.vtu`` -- pressure grid, including fluid/solid SDF fields,
* ``GalleryObstacle000000.vtu`` -- fixed five-point star, and
* ``GalleryTank000000.vtu`` -- static tank linework.

Local CPU command::

    GEOTAICHI_GALLERY_SDF_2D_ARCH=cpu \
    GEOTAICHI_GALLERY_SDF_2D_OUTPUT=examples/mpm/IncompressibleFluid/dam_break_visible_sdf_2d/OutputData/dam_break_star_sdf_2d_dx005_ppc2_v2 \
    GEOTAICHI_SKIP_POSTPROCESS=1 \
    python examples/mpm/IncompressibleFluid/dam_break_visible_sdf_2d/dam_break_visible_sdf_2d.py

Set ``GEOTAICHI_GALLERY_SDF_2D_ARCH=cpu`` for a CPU smoke run.  Timestep,
duration, grid spacing, particles per cell, output path, and save interval are
configurable through the similarly named ``..._DT``, ``..._TIME``, ``..._DX``,
``..._PPC``, ``..._OUTPUT``, and ``..._SAVE_INTERVAL`` environment variables.
"""

import os
import sys

import meshio
import numpy as np
import taichi as ti

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import MPM, init
from examples.mpm.IncompressibleFluid.dam_break_visible_sdf_2d.obstacle import VERTICES as OBSTACLE_VERTICES

ENV_PREFIX = "GEOTAICHI_GALLERY_SDF_2D_"
ARCH = os.environ.get(f"{ENV_PREFIX}ARCH", os.environ.get("GEOTAICHI_ARCH", "gpu"))
DT = float(os.environ.get(f"{ENV_PREFIX}DT", "2.0e-4"))
SIMULATION_TIME = float(os.environ.get(f"{ENV_PREFIX}TIME", "4.0"))
SAVE_INTERVAL = float(os.environ.get(f"{ENV_PREFIX}SAVE_INTERVAL", str(SIMULATION_TIME / 50.0)))
SAVE_PATH = os.environ.get(
    f"{ENV_PREFIX}OUTPUT",
    "examples/mpm/IncompressibleFluid/dam_break_visible_sdf_2d/OutputData/dam_break_star_sdf_2d_dx005_ppc2_v2",
)
DX = float(os.environ.get(f"{ENV_PREFIX}DX", "0.005"))
PPC = int(os.environ.get(f"{ENV_PREFIX}PPC", "2"))
MAX_PARTICLES = int(os.environ.get(f"{ENV_PREFIX}MAX_PARTICLES", "120000"))

DOMAIN = [0.90, 0.45]
WATER_LOWER = [0.04, 0.025]
WATER_SIZE = [0.23, 0.34]


@ti.kernel
def kernel_update_solid_sdf_from_polygon(
    vertices: ti.types.ndarray(dtype=ti.f64, ndim=2),
    grid_size: ti.types.vector(2, float),
    solid_sdf: ti.template(),
):
    for I in ti.grouped(solid_sdf):
        point = (I.cast(float) + 0.5) * grid_size
        distance = 1.0e6
        inside = 0
        for edge in range(vertices.shape[0]):
            next_edge = (edge + 1) % vertices.shape[0]
            start = ti.Vector([vertices[edge, 0], vertices[edge, 1]])
            end = ti.Vector([vertices[next_edge, 0], vertices[next_edge, 1]])
            segment = end - start
            fraction = ti.math.clamp((point - start).dot(segment) / segment.dot(segment), 0.0, 1.0)
            distance = ti.min(distance, (point - start - fraction * segment).norm())
            if (start[1] > point[1]) != (end[1] > point[1]):
                intersection_x = start[0] + (point[1] - start[1]) * segment[0] / segment[1]
                if point[0] < intersection_x:
                    inside = 1 - inside
        ti.atomic_min(solid_sdf[I], ti.select(inside == 1, -distance, distance))


def add_irregular_obstacle_sdf(sims, scene):
    """Insert the fixed polygon into the pressure grid's solid distance field."""
    del sims
    kernel_update_solid_sdf_from_polygon(
        OBSTACLE_VERTICES,
        scene.element.grid_size,
        scene.element.cell.solid_sdf,
    )


def write_gallery_geometry(save_path):
    """Write visible static geometry matching the analytic SDF and tank walls."""
    vtk_path = os.path.join(save_path, "vtks")
    os.makedirs(vtk_path, exist_ok=True)

    center = OBSTACLE_VERTICES.mean(axis=0)
    obstacle_points = np.column_stack((np.vstack((OBSTACLE_VERTICES, center)), np.zeros(len(OBSTACLE_VERTICES) + 1)))
    center_id = len(OBSTACLE_VERTICES)
    triangles = np.asarray(
        [(center_id, edge, (edge + 1) % center_id) for edge in range(center_id)],
        dtype=np.int32,
    )
    obstacle = meshio.Mesh(
        points=obstacle_points,
        cells=[("triangle", triangles)],
        cell_data={"gallery_feature": [np.ones(len(triangles), dtype=np.int32)]},
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

print("# Gallery: 2-D incompressible dam break around a visible five-point-star SDF obstacle")
print(f"# obstacle vertices: {OBSTACLE_VERTICES.tolist()}")
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

mpm.run(gravity_field=True, cut_cell_function=add_irregular_obstacle_sdf)

if os.environ.get("GEOTAICHI_SKIP_POSTPROCESS", "0") != "1":
    mpm.postprocessing()

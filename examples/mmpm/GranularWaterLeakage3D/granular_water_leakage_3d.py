"""3D incompressible two-phase MPM leakage benchmark without the elastic plate.

Geometry and material data follow Zhang et al. (2027), CMAME 463:119401,
Section 5.1 and Fig. 30.  The upper free surface is the zero-pressure boundary;
the paper's particle-replenishing inlet is not available in the MPM solver.
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import taichi as ti

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.mmpm.GranularWaterLeakage3D.granular_water_leakage_3d_parameters import (
    CRACK_LEFT,
    CRACK_RIGHT,
    DOMAIN,
    RECEIVER_ORIGIN,
    RECEIVER_SIZE,
    UPPER_ORIGIN,
)
from examples.mmpm.GranularWaterLeakage3D.draw.evaluate_granular_water_leakage_3d import (
    write_metrics,
)


from geotaichi import MPM, init

DX_DEFAULT = 0.005
UPPER_SIZE = np.array((0.30, 0.06, 0.20))
BED_HEIGHT = 0.06
WATER_HEIGHT = 0.20
POROSITY = 0.40
K0 = 1.0 - math.sin(math.radians(26.0))


def parse_args():
    parser = argparse.ArgumentParser(description="3D two-phase incompressible MPM granular-water leakage")
    parser.add_argument("--arch", default=os.environ.get("GEOTAICHI_ARCH", "gpu"))
    parser.add_argument("--dx", type=float, default=DX_DEFAULT)
    parser.add_argument("--dt", type=float, default=2.0e-5)
    parser.add_argument("--time", type=float, default=6.0)
    parser.add_argument("--save-interval", type=float, default=0.1)
    parser.add_argument("--ppc", type=int, default=2)
    parser.add_argument("--alpha-pic", type=float, default=1.0)
    parser.add_argument("--pressure-iterations", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "OutputData" / "paper_section_5_1_no_plate",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--no-post", action="store_true")
    return parser.parse_args()


def validate(args):
    if min(args.dx, args.dt, args.time, args.save_interval) <= 0.0 or min(args.ppc, args.pressure_iterations) <= 0:
        raise ValueError("dx, dt, time, save interval, ppc, and pressure iterations must be positive")
    if not 0.0 <= args.alpha_pic <= 1.0:
        raise ValueError("alpha-pic must lie in [0, 1]")
    cells = np.asarray(DOMAIN) / args.dx
    if not np.allclose(cells, np.round(cells), atol=1.0e-12, rtol=0.0):
        raise ValueError(f"dx={args.dx:g} must divide domain {DOMAIN}")
    if np.any(np.round(cells).astype(int) % 2):
        raise ValueError("two-level MGPCG requires an even cell count along every axis")


@ti.kernel
def initialize_stress(particle_count: int, particle: ti.template()):
    for p in range(particle_count):
        water_head = ti.max(UPPER_ORIGIN[2] + WATER_HEIGHT - particle[p].x[2], 0.0)
        particle[p].pressure = 1000.0 * 9.81 * water_head
        if int(particle[p].phase) == 1:
            soil_head = ti.max(UPPER_ORIGIN[2] + BED_HEIGHT - particle[p].x[2], 0.0)
            vertical = (1.0 - POROSITY) * (7850.0 - 1000.0) * 9.81 * soil_head
            particle[p].stress = ti.Vector([-K0 * vertical, -K0 * vertical, -vertical, 0.0, 0.0, 0.0])


def solid_cell(start, end, normal):
    return {
        "BoundaryType": "SolidCell",
        "StartPoint": list(start),
        "EndPoint": list(end),
        "Norm": list(normal),
        "CellThickness": 1,
    }


def solid_plane_cell(start, end, normal):
    boundary = solid_cell(start, end, normal)
    boundary["BoundaryType"] = "SolidPlaneCell"
    return boundary


def boundaries(dx):
    ux0, uy0, uz0 = UPPER_ORIGIN
    ux1, uy1, physical_top = UPPER_ORIGIN + UPPER_SIZE
    # The paper's upper boundary is an inlet/free-surface plane enclosed by
    # the tank sides.  Extend the four side walls through the numerical air
    # padding so splashed particles cannot bypass a wall whose top was
    # previously flush with the initial water level.
    numerical_top = DOMAIN[2]
    rx0, ry0, rz0 = RECEIVER_ORIGIN
    rx1, ry1, rz1 = RECEIVER_ORIGIN + RECEIVER_SIZE
    # A 6 mm slit cannot be represented exactly on the 5 mm pressure grid.
    # Leave one centered cell open (5 mm), rather than rounding it up to 10 mm.
    crack_left = round(0.5 * (CRACK_LEFT + CRACK_RIGHT) / dx) * dx
    crack_right = crack_left + dx
    return [
        # Upper tank: no-slip side walls and a split bottom leaving the 6 mm crack open.
        solid_cell((ux0, uy0, uz0), (ux0, uy1, physical_top), (-1.0, 0.0, 0.0)),
        solid_cell((ux1, uy0, uz0), (ux1, uy1, physical_top), (1.0, 0.0, 0.0)),
        solid_cell((ux0, uy0, uz0), (ux1, uy0, physical_top), (0.0, -1.0, 0.0)),
        solid_cell((ux0, uy1, uz0), (ux1, uy1, physical_top), (0.0, 1.0, 0.0)),
        # Plane-only caps prevent splashed particles from bypassing the tank
        # without adding empty solid cells above the initial free surface to
        # the multigrid pressure topology.
        solid_plane_cell((ux0, uy0, physical_top), (ux0, uy1, numerical_top), (-1.0, 0.0, 0.0)),
        solid_plane_cell((ux1, uy0, physical_top), (ux1, uy1, numerical_top), (1.0, 0.0, 0.0)),
        solid_plane_cell((ux0, uy0, physical_top), (ux1, uy0, numerical_top), (0.0, -1.0, 0.0)),
        solid_plane_cell((ux0, uy1, physical_top), (ux1, uy1, numerical_top), (0.0, 1.0, 0.0)),
        solid_cell((ux0, uy0, uz0), (crack_left, uy1, uz0), (0.0, 0.0, -1.0)),
        solid_cell((crack_right, uy0, uz0), (ux1, uy1, uz0), (0.0, 0.0, -1.0)),
        # Open-top receiving container.
        solid_cell((rx0, ry0, rz0), (rx1, ry1, rz0), (0.0, 0.0, -1.0)),
        solid_cell((rx0, ry0, rz0), (rx0, ry1, rz1), (-1.0, 0.0, 0.0)),
        solid_cell((rx1, ry0, rz0), (rx1, ry1, rz1), (1.0, 0.0, 0.0)),
        solid_cell((rx0, ry0, rz0), (rx1, ry0, rz1), (0.0, -1.0, 0.0)),
        solid_cell((rx0, ry1, rz0), (rx1, ry1, rz1), (0.0, 1.0, 0.0)),
    ]


def run(args):
    validate(args)
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    point_spacing = args.dx / args.ppc
    expected = math.ceil((np.prod(UPPER_SIZE) + UPPER_SIZE[0] * UPPER_SIZE[1] * BED_HEIGHT) / point_spacing**3)
    print(
        f"# Section 5.1 leakage: spacing={point_spacing:g} m, physical crack=0.006 m, "
        f"grid crack={args.dx:g} m, "
        f"fluid+solid capacity~{expected}, elastic plate omitted"
    )

    init(dim=3, arch=args.arch, default_fp="float64", device_memory_GB=6.0, offline_cache=True, debug=False)
    mpm = MPM()
    mpm.set_configuration(
        domain=list(DOMAIN),
        background_damping=0.02,
        gravity=[0.0, 0.0, -9.81],
        alphaPIC=args.alpha_pic,
        mapping="USL",
        shape_function="QuadBSpline",
        material_type="TwoPhaseDoubleLayer",
        solver_type="SemiImplicit",
        velocity_projection="Affine",
        delayed_fluid_advection=True,
        particle_shifting=True,
        free_surface_detection=False,
        visualize=True,
    )
    mpm.set_solver(
        {
            "Timestep": args.dt,
            "SimulationTime": args.time,
            "SaveInterval": args.save_interval,
            "SavePath": str(args.output),
        }
    )
    mpm.set_semi_implicit_solver_parameters(
        {
            "assemble_type": "MatrixFree",
            "pressure_solver": "MGPCG",
            "linear_solver": "MGPCG",
            "max_iteration_number": args.pressure_iterations,
            "residual_tolerance": 1.0e-7,
            "multilevel": 2,
            "pre_and_post_smoothing": 2,
            "bottom_smoothing": 8,
        }
    )
    mpm.memory_allocate(
        {
            "max_material_number": 1,
            "max_particle_number": int(1.1 * expected),
            "max_constraint_number": {"max_velocity_constraint": 1024},
        }
    )
    mpm.add_material(
        model="DruckerPrager",
        material={
            "MaterialID": 1,
            "SolidDensity": 7850.0,
            "FluidDensity": 1000.0,
            "Porosity": POROSITY,
            "MaximumPorosity": 0.64,
            "FluidBulkModulus": 2.2e9,
            "Permeability": 1.0e-8,
            "FluidViscosity": 1.0e-3,
            "GrainDiameter": 3.0e-3,
            "DragModel": "Ergun",
            "YoungModulus": 1.0e8,
            "PoissonRatio": 0.25,
            "Cohesion": 0.0,
            "Friction": 26.0,
            "Dilation": 0.0,
            "dpType": "MiddleCircumscribed",
        },
    )
    mpm.add_element({"ElementType": "R8N3D", "ElementSize": [args.dx] * 3})
    mpm.add_region(
        [
            {
                "Name": "upper_water",
                "Type": "Rectangle",
                "BoundingBoxPoint": UPPER_ORIGIN.tolist(),
                "BoundingBoxSize": [UPPER_SIZE[0], UPPER_SIZE[1], WATER_HEIGHT],
            },
            {
                "Name": "granular_bed",
                "Type": "Rectangle",
                "BoundingBoxPoint": UPPER_ORIGIN.tolist(),
                "BoundingBoxSize": [UPPER_SIZE[0], UPPER_SIZE[1], BED_HEIGHT],
            },
        ]
    )
    mpm.add_body(
        {
            "Template": [
                {
                    "RegionName": "upper_water",
                    "nParticlesPerCell": args.ppc,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "Phase": "Fluid",
                    "InitialVelocity": [0.0, 0.0, 0.0],
                    "FixVelocity": ["Free", "Free", "Free"],
                },
                {
                    "RegionName": "granular_bed",
                    "nParticlesPerCell": args.ppc,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "Phase": "Solid",
                    "InitialVelocity": [0.0, 0.0, 0.0],
                    "FixVelocity": ["Free", "Free", "Free"],
                },
            ]
        }
    )
    particle_count = int(mpm.scene.particleNum[0])
    if particle_count > int(1.1 * expected):
        raise RuntimeError("particle capacity estimate is too small")
    initialize_stress(particle_count, mpm.scene.particle)
    initial_phase = mpm.scene.particle.phase.to_numpy()[:particle_count]
    expected_fluid = int(np.count_nonzero(initial_phase == 2))
    expected_solid = int(np.count_nonzero(initial_phase == 1))
    mpm.add_boundary_condition(boundaries(args.dx))
    mpm.select_save_data(particle=True, grid=True, object=False)

    mpm.run()
    write_metrics(args.output, expected_fluid, expected_solid, args)
    if not args.no_post:
        mpm.postprocessing()


if __name__ == "__main__":
    run(parse_args())

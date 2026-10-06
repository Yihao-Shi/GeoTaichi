"""Three-dimensional two-point saturated slope with a device-side piston wavemaker.

The tank is filled with fluid points and the right-hand trapezoidal bed is a
second (solid) point set.  Thus water occupies the pore volume as well as the
open water above the slope.  The left-wall velocity is updated by a Taichi
callback after every accepted step; the time integration and boundary update
therefore remain on device.
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

from examples.mmpm.TwoPhaseWavemaker3D.two_layer_two_phase_wavemaker_3d_parameters import (
    PISTON_MEAN_X,
    SLOPE_HEIGHT,
    SLOPE_RUN,
    SLOPE_TOE,
    TANK,
    WALL_CELLS,
    WATER_DEPTH,
)
from examples.mmpm.TwoPhaseWavemaker3D.draw.evaluate_two_layer_two_phase_wavemaker_3d import (
    write_metrics,
)


from geotaichi import MPM, init
from examples.mpm.IncompressibleFluid.wavemaker_tank_3d.draw.evaluate_wavemaker_tank_3d import piston_displacement
from examples.mpm.IncompressibleFluid.wavemaker_tank_3d.wavemaker_tank_3d import (
    enforce_moving_piston_particles,
    piston_velocity,
)
from src.utils.SolverRuntime import python_callback

DX_DEFAULT = 0.02
# Keep seven cells of initial headspace: run-up compressed the former
# three-cell band to less than half a cell and made the pressure topology fail.
WAVE_FREQUENCY = 0.9
WAVE_VELOCITY = 0.24
WAVE_RAMP = 1.2
SOIL_YOUNG_MODULUS = 12.0e6
SOIL_COHESION = 300.0
SOIL_FRICTION = 26.0


def parse_args():
    parser = argparse.ArgumentParser(description="3D two-layer, two-phase incompressible MPM wavemaker tank")
    parser.add_argument("--arch", default=os.environ.get("GEOTAICHI_ARCH", "gpu"))
    parser.add_argument("--dx", type=float, default=float(os.environ.get("GT_WAVEMAKER_DX", DX_DEFAULT)))
    parser.add_argument("--dt", type=float, default=float(os.environ.get("GT_WAVEMAKER_DT", "1.0e-4")))
    parser.add_argument("--time", type=float, default=float(os.environ.get("GT_WAVEMAKER_TIME", "5.0")))
    parser.add_argument(
        "--save-interval", type=float, default=float(os.environ.get("GT_WAVEMAKER_SAVE_INTERVAL", "0.05"))
    )
    parser.add_argument("--fluid-ppc", type=int, default=int(os.environ.get("GT_WAVEMAKER_FLUID_PPC", "2")))
    parser.add_argument("--solid-ppc", type=int, default=int(os.environ.get("GT_WAVEMAKER_SOLID_PPC", "2")))
    parser.add_argument("--device-memory", type=float, default=4.0)
    parser.add_argument(
        "--pressure-iterations", type=int, default=int(os.environ.get("GT_WAVEMAKER_PRESSURE_ITERATIONS", "1000"))
    )
    parser.add_argument("--alpha-pic", type=float, default=0.30)
    parser.add_argument("--wave-velocity", type=float, default=WAVE_VELOCITY)
    parser.add_argument("--frequency", type=float, default=WAVE_FREQUENCY)
    parser.add_argument("--ramp-time", type=float, default=WAVE_RAMP)
    parser.add_argument("--maximum-porosity", type=float, default=0.56)
    parser.add_argument("--soil-young-modulus", type=float, default=SOIL_YOUNG_MODULUS)
    parser.add_argument("--soil-cohesion", type=float, default=SOIL_COHESION)
    parser.add_argument("--soil-friction", type=float, default=SOIL_FRICTION)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            os.environ.get(
                "GT_WAVEMAKER_OUTPUT",
                ROOT / "examples" / "mmpm" / "TwoPhaseWavemaker3D" / "OutputData" / "two_layer_two_phase_wavemaker_3d",
            )
        ),
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--no-post", action="store_true")
    return parser.parse_args()


def validate_geometry(args):
    if (
        min(args.dx, args.dt, args.time, args.save_interval, args.frequency, args.ramp_time, args.device_memory) <= 0.0
        or min(args.fluid_ppc, args.solid_ppc, args.pressure_iterations) <= 0
    ):
        raise ValueError("dx, dt, time, save interval, device memory, and particles per cell must be positive")
    if not 0.0 <= args.alpha_pic <= 1.0 or not 0.40 <= args.maximum_porosity <= 1.0 or args.wave_velocity < 0.0:
        raise ValueError(
            "alpha PIC must lie in [0, 1], maximum porosity in [0.40, 1], and wave velocity must be nonnegative"
        )
    if args.soil_young_modulus <= 0.0 or args.soil_cohesion < 0.0 or not 0.0 <= args.soil_friction < 90.0:
        raise ValueError("soil Young's modulus must be positive, cohesion nonnegative, and friction in [0, 90)")
    cells = np.asarray(TANK, dtype=float) / args.dx
    if not np.allclose(cells, np.round(cells), rtol=0.0, atol=1.0e-12):
        raise ValueError(f"dx={args.dx:g} must divide the tank dimensions {TANK}")
    if np.any(np.round(cells).astype(int) % 2):
        raise ValueError("two-level MGPCG requires an even number of cells along every tank axis")
    if not math.isclose(WATER_DEPTH / args.dx, round(WATER_DEPTH / args.dx), abs_tol=1.0e-12):
        raise ValueError("dx must also divide WATER_DEPTH")
    if PISTON_MEAN_X - args.wave_velocity / (2.0 * math.pi * args.frequency) <= args.dx:
        raise ValueError("piston stroke leaves insufficient clearance from x=0")


def make_slope_region(wall):
    slope_end = TANK[0] - wall

    def region_volume():
        plateau = (slope_end - (SLOPE_TOE + SLOPE_RUN)) * SLOPE_HEIGHT
        return TANK[1] * (plateau + 0.5 * SLOPE_RUN * SLOPE_HEIGHT)

    def region_function(position, particle_radius=0.0):
        local_x = position[0] - SLOPE_TOE
        top = ti.min(SLOPE_HEIGHT, SLOPE_HEIGHT * local_x / SLOPE_RUN)
        return (
            SLOPE_TOE <= position[0] <= slope_end
            and wall <= position[1] <= TANK[1] - wall
            and wall + 0.0 * particle_radius <= position[2] <= top
        )

    return {
        "Name": "right_trapezoidal_slope",
        "Type": "UserDefined",
        "BoundingBoxPoint": [SLOPE_TOE, wall, wall],
        "BoundingBoxSize": [slope_end - SLOPE_TOE, TANK[1] - 2.0 * wall, SLOPE_HEIGHT - wall],
        "RegionVolume": region_volume,
        "RegionFunction": region_function,
    }


@ti.kernel
def set_moving_piston_mac_boundary(
    velocity: float,
    wall_position: float,
    tank_width: float,
    tank_height: float,
    grid_size: ti.types.vector(3, float),
    cell_type: ti.template(),
    solid_velocity_x: ti.template(),
):
    for I in ti.grouped(cell_type):
        if (I[0] + 0.5) * grid_size[0] <= wall_position:
            cell_type[I] = 2
    for I in ti.grouped(solid_velocity_x):
        x = I[0] * grid_size[0]
        y = (I[1] + 0.5) * grid_size[1]
        z = (I[2] + 0.5) * grid_size[2]
        if x <= wall_position + grid_size[0] and 0.0 < y < tank_width and 0.0 < z < tank_height:
            solid_velocity_x[I] = velocity


@ti.kernel
def initialize_hydrostatic_state(
    particle_count: int, water_depth: float, friction_angle: float, particle: ti.template()
):
    k0 = 1.0 - ti.sin(friction_angle * math.pi / 180.0)
    for p in range(particle_count):
        water_head = ti.max(water_depth - particle[p].x[2], 0.0)
        particle[p].pressure = 1000.0 * 9.81 * water_head
        if int(particle[p].phase) == 1:
            slope_surface = ti.min(
                SLOPE_HEIGHT,
                ti.max(0.0, SLOPE_HEIGHT * (particle[p].x[0] - SLOPE_TOE) / SLOPE_RUN),
            )
            soil_head = ti.max(slope_surface - particle[p].x[2], 0.0)
            effective_vertical = (1.0 - particle[p].porosity) * (2650.0 - 1000.0) * 9.81 * soil_head
            particle[p].stress = ti.Vector(
                [-k0 * effective_vertical, -k0 * effective_vertical, -effective_vertical, 0.0, 0.0, 0.0]
            )


def run(args):
    validate_geometry(args)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    wall = WALL_CELLS * args.dx
    tank_cells = np.round(np.asarray(TANK) / args.dx).astype(int)
    boundary_layers = WALL_CELLS + 1
    static_velocity_constraints = boundary_layers * (
        (tank_cells[0] + 1) * (tank_cells[1] + 1)
        + 2 * (tank_cells[0] + 1) * (tank_cells[2] + 1)
        + (tank_cells[1] + 1) * (tank_cells[2] + 1)
    )
    expected_fluid = math.ceil(
        (TANK[0] - PISTON_MEAN_X - wall)
        * (TANK[1] - 2.0 * wall)
        * (WATER_DEPTH - wall)
        * args.fluid_ppc**3
        / args.dx**3
    )
    expected_solid = math.ceil(
        (TANK[0] - SLOPE_TOE - wall) * (TANK[1] - 2.0 * wall) * (SLOPE_HEIGHT - wall) * args.solid_ppc**3 / args.dx**3
    )
    max_particles = expected_fluid + expected_solid
    print(
        f"# 3D wavemaker tank: dx={args.dx:g}, water~{expected_fluid}, soil capacity~{expected_solid}, "
        f"slope height={SLOPE_HEIGHT:g} m ({SLOPE_HEIGHT / WATER_DEPTH:.2f} water depth), "
        f"plateau={TANK[0] - wall - SLOPE_TOE - SLOPE_RUN:g} m"
    )

    init(
        dim=3,
        arch=args.arch,
        default_fp="float64",
        device_memory_GB=args.device_memory,
        offline_cache=True,
        debug=False,
    )
    mpm = MPM()
    mpm.set_configuration(
        domain=list(TANK),
        background_damping=0.03,
        gravity=[0.0, 0.0, -9.81],
        alphaPIC=args.alpha_pic,
        mapping="USL",
        shape_function="QuadBSpline",
        material_type="TwoPhaseDoubleLayer",
        solver_type="SemiImplicit",
        velocity_projection="Affine",
        delayed_fluid_advection=True,
        particle_shifting=True,
        # DoubleLayer builds its own fluid SDF.  The generic density detector
        # assumes a single-phase velocity_gradient field.
        free_surface_detection=False,
        visualize=True,
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
    mpm.set_solver(
        {"Timestep": args.dt, "SimulationTime": args.time, "SaveInterval": args.save_interval, "SavePath": str(output)}
    )
    mpm.memory_allocate(
        {
            "max_material_number": 1,
            "max_particle_number": max_particles,
            "max_constraint_number": {"max_velocity_constraint": static_velocity_constraints},
        }
    )
    mpm.add_material(
        model="MohrCoulomb",
        material={
            "MaterialID": 1,
            "SolidDensity": 2650.0,
            "FluidDensity": 1000.0,
            "Porosity": 0.40,
            "MaximumPorosity": args.maximum_porosity,
            "FluidBulkModulus": 2.2e8,
            "Permeability": 2.0e-4,
            "FluidViscosity": 1.0e-3,
            "GrainDiameter": 3.0e-3,
            "DragModel": "Ergun",
            "YoungModulus": args.soil_young_modulus,
            "PoissonRatio": 0.30,
            "Cohesion": args.soil_cohesion,
            "Friction": args.soil_friction,
            "Dilation": 0.0,
        },
    )
    mpm.add_element({"ElementType": "R8N3D", "ElementSize": [args.dx, args.dx, args.dx]})
    mpm.add_region(
        [
            {
                "Name": "water",
                "Type": "Rectangle",
                "BoundingBoxPoint": [PISTON_MEAN_X, wall, wall],
                "BoundingBoxSize": [
                    TANK[0] - PISTON_MEAN_X - wall,
                    TANK[1] - 2.0 * wall,
                    WATER_DEPTH - wall,
                ],
            },
            make_slope_region(wall),
        ]
    )
    mpm.add_body(
        {
            "Template": [
                {
                    "RegionName": "water",
                    "nParticlesPerCell": args.fluid_ppc,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "Phase": "Fluid",
                    "InitialVelocity": [0.0, 0.0, 0.0],
                    "FixVelocity": ["Free", "Free", "Free"],
                },
                {
                    "RegionName": "right_trapezoidal_slope",
                    "nParticlesPerCell": args.solid_ppc,
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
    initial_phase = mpm.scene.particle.phase.to_numpy()[:particle_count]
    expected_fluid = int(np.count_nonzero(initial_phase == 2))
    expected_solid = int(np.count_nonzero(initial_phase == 1))
    initialize_hydrostatic_state(particle_count, WATER_DEPTH, args.soil_friction, mpm.scene.particle)

    walls = [
        ([wall, wall, wall], [TANK[0] - wall, wall, TANK[2]], [0.0, -1.0, 0.0]),
        ([wall, TANK[1] - wall, wall], [TANK[0] - wall, TANK[1] - wall, TANK[2]], [0.0, 1.0, 0.0]),
        ([wall, wall, wall], [TANK[0] - wall, TANK[1] - wall, wall], [0.0, 0.0, -1.0]),
        ([TANK[0] - wall, wall, wall], [TANK[0] - wall, TANK[1] - wall, TANK[2]], [1.0, 0.0, 0.0]),
    ]
    mpm.add_boundary_condition(
        [
            {
                "BoundaryType": "SolidCell",
                "StartPoint": start,
                "EndPoint": end,
                "Norm": normal,
                "CellThickness": WALL_CELLS,
            }
            for start, end, normal in walls
        ]
        + [
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [None, None, 0.0],
                "StartPoint": [0.0, 0.0, 0.0],
                "EndPoint": [TANK[0], TANK[1], wall],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [None, 0.0, None],
                "StartPoint": [0.0, 0.0, 0.0],
                "EndPoint": [TANK[0], wall, TANK[2]],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [None, 0.0, None],
                "StartPoint": [0.0, TANK[1] - wall, 0.0],
                "EndPoint": [TANK[0], TANK[1], TANK[2]],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, None, None],
                "StartPoint": [TANK[0] - wall, 0.0, 0.0],
                "EndPoint": [TANK[0], TANK[1], TANK[2]],
            },
        ]
    )

    def update_wavemaker(sims, scene):
        start_time = float(sims.current_time)
        end_time = min(start_time + float(sims.delta), args.time)
        velocity = piston_velocity(0.5 * (start_time + end_time), args.frequency, args.wave_velocity, args.ramp_time)
        position = PISTON_MEAN_X + piston_displacement(end_time, args.frequency, args.wave_velocity, args.ramp_time)
        set_moving_piston_mac_boundary(
            velocity,
            position,
            TANK[1],
            TANK[2],
            scene.element.grid_size,
            mpm.enginer.cell_type,
            mpm.enginer.solid_velocity_x,
        )

    @python_callback
    def enforce_wavemaker_particles():
        start_time = float(mpm.sims.current_time)
        end_time = min(start_time + float(mpm.sims.delta), args.time)
        velocity = piston_velocity(0.5 * (start_time + end_time), args.frequency, args.wave_velocity, args.ramp_time)
        position = PISTON_MEAN_X + piston_displacement(end_time, args.frequency, args.wave_velocity, args.ramp_time)
        enforce_moving_piston_particles(int(mpm.scene.particleNum[0]), position, velocity, args.dx, mpm.scene.particle)

    mpm.select_save_data(particle=True, grid=True, object=False)
    mpm.run(mac_boundary_function=update_wavemaker, function=enforce_wavemaker_particles)
    if not args.no_post:
        mpm.postprocessing()
    write_metrics(output, expected_fluid, expected_solid, args)


if __name__ == "__main__":
    run(parse_args())

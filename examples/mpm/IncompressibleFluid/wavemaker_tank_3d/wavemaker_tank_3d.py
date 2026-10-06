"""3-D incompressible MPM tank with a left-wall piston wavemaker."""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import taichi as ti

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.mpm.IncompressibleFluid.wavemaker_tank_3d.wavemaker_tank_3d_parameters import (
    PISTON_MEAN_X,
    TANK,
    WATER_DEPTH,
)
from examples.mpm.IncompressibleFluid.wavemaker_tank_3d.draw.evaluate_wavemaker_tank_3d import (
    piston_displacement,
    write_metrics,
)

os.environ.setdefault("GEOTAICHI_REAL_DTYPE", "float64")

from geotaichi import MPM, init  # noqa: E402
from src.mpm.engines.EngineKernel import kernel_update_solid_sdf_from_box  # noqa: E402
from src.utils.SolverRuntime import python_callback  # noqa: E402

WATER_ORIGIN = np.array([PISTON_MEAN_X, 0.0, 0.0])
WATER_SIZE = np.array([TANK[0] - PISTON_MEAN_X, TANK[1], WATER_DEPTH])
WAVE_FREQUENCY = 0.9
WAVE_VELOCITY = 0.24
WAVE_RAMP = 1.2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default=os.environ.get("GEOTAICHI_ARCH", "gpu"))
    parser.add_argument("--dx", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=2.5e-4)
    parser.add_argument("--time", type=float, default=5.5)
    parser.add_argument("--save-interval", type=float, default=0.05)
    parser.add_argument("--ppc", type=int, default=3)
    parser.add_argument("--velocity-amplitude", type=float, default=WAVE_VELOCITY)
    parser.add_argument("--alpha-pic", type=float, default=0.15)
    parser.add_argument("--frequency", type=float, default=WAVE_FREQUENCY)
    parser.add_argument("--ramp-time", type=float, default=WAVE_RAMP)
    parser.add_argument("--device-memory", type=float, default=7.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "OutputData" / "wavemaker_tank_3d",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--no-post", action="store_true")
    args = parser.parse_args()
    if min(args.dx, args.dt, args.time, args.save_interval, args.frequency, args.ramp_time) <= 0 or args.ppc <= 0:
        parser.error("spacing, time controls, frequency, ramp time, and ppc must be positive")
    if args.velocity_amplitude < 0:
        parser.error("--velocity-amplitude must be nonnegative")
    if PISTON_MEAN_X - args.velocity_amplitude / (2.0 * math.pi * args.frequency) <= 0.5 * args.dx:
        parser.error("piston stroke leaves insufficient clearance from x=0")
    if not 0.0 <= args.alpha_pic <= 1.0:
        parser.error("--alpha-pic must be in [0, 1]")
    counts = TANK / args.dx
    if not np.allclose(counts, np.round(counts), atol=1.0e-12, rtol=0.0):
        parser.error("--dx must divide all tank dimensions")
    if not math.isclose(round(WATER_DEPTH / args.dx) * args.dx, WATER_DEPTH, abs_tol=1.0e-12):
        parser.error("--dx must divide the water depth")
    if np.any(np.round(counts).astype(int) % 2):
        parser.error("the two-level MGPCG grid needs even cell counts")
    return args


@ti.kernel
def set_moving_piston_velocity(
    velocity: float,
    wall_position: float,
    tank_width: float,
    tank_height: float,
    grid_size: ti.types.vector(3, float),
    solid_velocity_x: ti.template(),
):
    for I in ti.grouped(solid_velocity_x):
        x = I[0] * grid_size[0]
        y = (I[1] + 0.5) * grid_size[1]
        z = (I[2] + 0.5) * grid_size[2]
        if x <= wall_position + grid_size[0] and 0.0 < y < tank_width and 0.0 < z < tank_height:
            solid_velocity_x[I] = velocity


@ti.kernel
def enforce_moving_piston_particles(
    particle_num: int,
    wall_position: float,
    wall_velocity: float,
    dx: float,
    particle: ti.template(),
):
    clearance = 1.0e-4 * dx
    for p in range(particle_num):
        if int(particle[p].active) == 1 and particle[p].x[0] < wall_position + clearance:
            particle[p].x[0] = wall_position + clearance
            if particle[p].v[0] < wall_velocity:
                particle[p].v[0] = wall_velocity


def piston_velocity(time, frequency, velocity_amplitude, ramp_time):
    omega = 2.0 * math.pi * frequency
    if time >= ramp_time:
        return velocity_amplitude * math.cos(omega * time)
    phase = math.pi * time / ramp_time
    ramp = 0.5 * (1.0 - math.cos(phase))
    ramp_rate = 0.5 * math.pi / ramp_time * math.sin(phase)
    displacement_amplitude = velocity_amplitude / omega
    return displacement_amplitude * (ramp_rate * math.sin(omega * time) + ramp * omega * math.cos(omega * time))


def run(args):
    output = args.output.expanduser().resolve()
    fluid_cells = np.round(WATER_SIZE / args.dx).astype(int)
    expected_particles = int(np.prod(fluid_cells) * args.ppc**3)
    print(
        f"# 3-D wavemaker tank: cells={np.round(TANK / args.dx).astype(int).tolist()}, particles={expected_particles}"
    )

    init(
        dim=3,
        arch=args.arch,
        default_fp="float64",
        device_memory_GB=args.device_memory,
        offline_cache=True,
        debug=False,
        kernel_profiler=False,
        log=False,
    )
    mpm = MPM()
    mpm.set_configuration(
        domain=TANK.tolist(),
        background_damping=0.0,
        alphaPIC=args.alpha_pic,
        mapping="USL",
        shape_function="QuadBSpline",
        gravity=[0.0, 0.0, -9.81],
        material_type="Fluid",
        velocity_projection="Affine",
        solver_type="Implicit",
        discretization="FDM",
        fluid_level_set=True,
        fluid_domain_volume_fraction=0.15,
        solid_sdf_cut_cell=True,
        fluid_wall_no_slip=False,
        particle_shifting=True,
        density_projection=True,
        density_projection_tolerance=0.008,
        density_projection_error_clamp=0.02,
        density_projection_max_shift_ratio=0.03,
        density_projection_interior_only=True,
        visualize=True,
    )
    mpm.set_implicit_solver_parameters(
        linear_solver="MGPCG",
        multilevel=2,
        pre_and_post_smoothing=2,
        bottom_smoothing=16,
        max_iteration_number=160,
        residual_tolerance=1.0e-8,
    )
    mpm.set_solver(
        {
            "Timestep": args.dt,
            "SimulationTime": args.time,
            "SaveInterval": args.save_interval,
            "SavePath": str(output),
        }
    )
    mpm.memory_allocate(
        {
            "max_material_number": 1,
            "max_particle_number": expected_particles,
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
            "ElementLength": args.dx,
            "cL": 1.5,
            "cQ": 2.0,
            "atmospheric_pressure": 0.0,
        },
    )
    mpm.add_element({"ElementType": "Staggered", "ElementSize": [args.dx] * 3, "GhostCell": 1})
    mpm.add_region(
        {
            "Name": "water",
            "Type": "Rectangle",
            "BoundingBoxPoint": WATER_ORIGIN.tolist(),
            "BoundingBoxSize": WATER_SIZE.tolist(),
        }
    )
    mpm.add_body(
        {
            "Template": [
                {
                    "RegionName": "water",
                    "nParticlesPerCell": args.ppc,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "InitialVelocity": [0.0, 0.0, 0.0],
                    "FixVelocity": ["Free", "Free", "Free"],
                }
            ]
        }
    )
    walls = [
        ([1.0, 0.0, 0.0], [TANK[0], 0.0, 0.0], TANK.tolist()),
        ([0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [TANK[0], 0.0, TANK[2]]),
        ([0.0, 1.0, 0.0], [0.0, TANK[1], 0.0], TANK.tolist()),
        ([0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [TANK[0], TANK[1], 0.0]),
    ]
    mpm.add_boundary_condition(
        [
            {"BoundaryType": "SolidCell", "Norm": normal, "StartPoint": start, "EndPoint": end, "CellThickness": 1}
            for normal, start, end in walls
        ]
    )

    requested_velocities = []
    requested_positions = []

    def update_wavemaker(sims, scene):
        start_time = float(sims.current_time)
        end_time = min(start_time + float(sims.delta), args.time)
        velocity = piston_velocity(
            0.5 * (start_time + end_time), args.frequency, args.velocity_amplitude, args.ramp_time
        )
        position = PISTON_MEAN_X + piston_displacement(
            end_time, args.frequency, args.velocity_amplitude, args.ramp_time
        )
        requested_velocities.append(velocity)
        requested_positions.append(position)
        mpm.enginer.ensure_solid_cut_cell_fields(sims, scene)
        mpm.enginer.apply_solid_cell_boundaries(sims, scene)
        kernel_update_solid_sdf_from_box(
            scene.element.grid_size,
            ti.Vector([0.0, 0.0, 0.0]),
            ti.Vector([position, TANK[1], TANK[2]]),
            scene.element.cell.solid_sdf,
        )
        set_moving_piston_velocity(
            velocity,
            position,
            TANK[1],
            TANK[2],
            scene.element.grid_size,
            mpm.enginer.solid_face_velocity[0],
        )

    @python_callback
    def enforce_wavemaker_particles():
        start_time = float(mpm.sims.current_time)
        end_time = min(start_time + float(mpm.sims.delta), args.time)
        velocity = piston_velocity(
            0.5 * (start_time + end_time), args.frequency, args.velocity_amplitude, args.ramp_time
        )
        position = PISTON_MEAN_X + piston_displacement(
            end_time, args.frequency, args.velocity_amplitude, args.ramp_time
        )
        enforce_moving_piston_particles(int(mpm.scene.particleNum[0]), position, velocity, args.dx, mpm.scene.particle)

    mpm.select_save_data(particle=True, grid=True)
    mpm.run(gravity_field=True, cut_cell_function=update_wavemaker, function=enforce_wavemaker_particles)
    if not args.no_post:
        mpm.postprocessing()
    write_metrics(output, expected_particles, requested_velocities, requested_positions, args)


if __name__ == "__main__":
    run(parse_args())

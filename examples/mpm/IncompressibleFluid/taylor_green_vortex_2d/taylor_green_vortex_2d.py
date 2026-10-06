"""Taylor--Green vortex verification for 2-D incompressible MPM."""

import argparse
import math
import os
import sys
from pathlib import Path

import taichi as ti

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.mpm.IncompressibleFluid.taylor_green_vortex_2d.taylor_green_vortex_2d_parameters import DOMAIN
from examples.mpm.IncompressibleFluid.taylor_green_vortex_2d.draw.evaluate_taylor_green_vortex_2d import (
    comparison_row,
    decay_time_to_fraction,
    write_metrics,
)

os.environ.setdefault("GEOTAICHI_REAL_DTYPE", "float64")

from geotaichi import MPM, init  # noqa: E402
from src.utils.SolverRuntime import python_callback  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default=os.environ.get("GEOTAICHI_ARCH", "gpu"))
    parser.add_argument("--cells", type=int, default=64)
    parser.add_argument("--ppc", type=int, default=4)
    parser.add_argument("--dt", type=float, default=5.0e-3)
    parser.add_argument("--time", type=float)
    parser.add_argument("--save-interval", type=float, default=1.0)
    parser.add_argument("--viscosity", type=float, default=0.05)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--steady-fraction", type=float, default=0.01)
    parser.add_argument("--verification-time", type=float, default=1.0)
    parser.add_argument("--alpha-pic", type=float, default=0.01)
    parser.add_argument("--device-memory", type=float, default=2.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "OutputData" / "taylor_green_vortex_2d",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--no-post", action="store_true")
    args = parser.parse_args()
    if args.cells <= 0 or args.cells % 4 or args.ppc <= 0:
        parser.error("--cells must be positive and divisible by four; --ppc must be positive")
    if min(args.dt, args.save_interval, args.viscosity, args.amplitude, args.verification_time) <= 0.0:
        parser.error("time controls, viscosity, and amplitude must be positive")
    if not 0.0 < args.steady_fraction < 1.0:
        parser.error("--steady-fraction must lie in (0, 1)")
    if args.time is None:
        args.time = decay_time_to_fraction(args.viscosity, args.steady_fraction)
    if args.time <= 0.0:
        parser.error("--time must be positive")
    if not 0.0 <= args.alpha_pic <= 1.0:
        parser.error("--alpha-pic must be in [0, 1]")
    return args


@ti.kernel
def initialize_velocity(particle_num: int, amplitude: float, particle: ti.template()):
    for p in range(particle_num):
        x, y = particle[p].x
        particle[p].v = amplitude * ti.Vector([ti.sin(x) * ti.cos(y), -ti.cos(x) * ti.sin(y)])


def run(args):
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    dx = math.pi / args.cells
    expected_particles = args.cells**2 * args.ppc**2
    sample_steps = max(1, round(args.save_interval / args.dt))

    init(
        dim=2,
        arch=args.arch,
        default_fp="float64",
        default_ip="int32",
        device_memory_GB=args.device_memory,
        offline_cache=True,
        debug=False,
        kernel_profiler=False,
        log=False,
    )
    mpm = MPM()
    mpm.set_configuration(
        domain=DOMAIN.tolist(),
        background_damping=0.0,
        alphaPIC=args.alpha_pic,
        mapping="USL",
        shape_function="QuadBSpline",
        gravity=[0.0, 0.0],
        material_type="Fluid",
        velocity_projection="PIC/FLIP",
        solver_type="Implicit",
        discretization="FDM",
        fluid_level_set=True,
        fluid_domain_volume_fraction=0.1,
        solid_sdf_cut_cell=True,
        fluid_wall_no_slip=False,
        particle_shifting=True,
        density_projection=False,
        visualize=True,
    )
    mpm.set_implicit_solver_parameters(
        linear_solver="MGPCG",
        multilevel=3,
        pre_and_post_smoothing=2,
        bottom_smoothing=20,
        max_iteration_number=160,
        residual_tolerance=1.0e-10,
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
            "Density": 1.0,
            "Modulus": 2.0e3,
            "Viscosity": args.viscosity,
            "ElementLength": dx,
            "cL": 1.5,
            "cQ": 2.0,
            "atmospheric_pressure": 0.0,
            "SurfaceTension": 0.0,
        },
    )
    mpm.add_element({"ElementType": "Staggered", "ElementSize": [dx, dx], "GhostCell": 1})
    mpm.add_region(
        {
            "Name": "fluid",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": [0.0, 0.0],
            "BoundingBoxSize": DOMAIN.tolist(),
            "rotate2D": 0.0,
        }
    )
    mpm.add_body(
        {
            "Template": [
                {
                    "RegionName": "fluid",
                    "nParticlesPerCell": args.ppc,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "InitialVelocity": [0.0, 0.0],
                    "FixVelocity": ["Free", "Free"],
                }
            ]
        }
    )
    initialize_velocity(int(mpm.scene.particleNum[0]), args.amplitude, mpm.scene.particle)
    walls = [
        ([-1.0, 0.0], [0.0, 0.0], [0.0, DOMAIN[1]]),
        ([1.0, 0.0], [DOMAIN[0], 0.0], DOMAIN.tolist()),
        ([0.0, -1.0], [0.0, 0.0], [DOMAIN[0], 0.0]),
        ([0.0, 1.0], [0.0, DOMAIN[1]], DOMAIN.tolist()),
    ]
    mpm.add_boundary_condition(
        [
            {"BoundaryType": "SolidCell", "Norm": normal, "StartPoint": start, "EndPoint": end, "CellThickness": 1}
            for normal, start, end in walls
        ]
    )

    diagnostics = []

    @python_callback
    def compare_with_exact_solution():
        step = int(mpm.sims.current_step) + 1
        time = float(mpm.sims.current_time + mpm.sims.delta)
        if step % sample_steps and time < args.time - 0.1 * args.dt:
            return
        diagnostics.append(comparison_row(mpm, args, time, dx))

    mpm.select_save_data(particle=True, grid=True)
    mpm.run(gravity_field=False, function=compare_with_exact_solution)
    if not args.no_post:
        mpm.postprocessing()

    write_metrics(output, diagnostics, expected_particles, args)


if __name__ == "__main__":
    run(parse_args())

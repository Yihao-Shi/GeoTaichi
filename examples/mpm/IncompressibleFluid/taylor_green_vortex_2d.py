"""Taylor--Green vortex verification for 2-D incompressible MPM."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import taichi as ti

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("GEOTAICHI_REAL_DTYPE", "float64")

from geotaichi import MPM, init  # noqa: E402
from src.utils.SolverRuntime import python_callback  # noqa: E402

DOMAIN = np.array([math.pi, math.pi])
STRICT_TOLERANCES = {
    "velocity_relative_l2": 0.03,
    "velocity_relative_linf": 0.03,
    "kinetic_energy_relative_error": 0.05,
    "pressure_relative_l2": 0.10,
}


def analytical_velocity(position, time, viscosity, amplitude=1.0):
    decay = amplitude * math.exp(-2.0 * viscosity * time)
    x, y = position.T
    return decay * np.column_stack((np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)))


def analytical_pressure(position, time, viscosity, density=1.0, amplitude=1.0):
    decay = math.exp(-4.0 * viscosity * time)
    x, y = position.T
    return 0.25 * density * amplitude**2 * decay * (np.cos(2.0 * x) + np.cos(2.0 * y))


def decay_time_to_fraction(viscosity, fraction):
    return -math.log(fraction) / (2.0 * viscosity)


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
        default=Path(__file__).resolve().parent / "OutputData" / "taylor_green_vortex_2d",
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
        particle_count = int(mpm.scene.particleNum[0])
        active = mpm.scene.particle.active.to_numpy()[:particle_count] > 0
        position = mpm.scene.particle.x.to_numpy()[:particle_count][active]
        velocity = mpm.scene.particle.v.to_numpy()[:particle_count][active]
        exact_velocity = analytical_velocity(position, time, args.viscosity, args.amplitude)
        difference = velocity - exact_velocity
        exact_norm = np.linalg.norm(exact_velocity)
        velocity_l2 = float(np.linalg.norm(difference) / max(exact_norm, np.finfo(float).eps))
        velocity_linf = float(
            np.linalg.norm(difference, axis=1).max() / (args.amplitude * math.exp(-2.0 * args.viscosity * time))
        )
        kinetic_energy = float(0.5 * np.mean(np.sum(velocity * velocity, axis=1)))
        exact_energy = 0.25 * args.amplitude**2 * math.exp(-4.0 * args.viscosity * time)

        ghost = int(mpm.scene.element.ghost_cell)
        active_slice = (slice(ghost, ghost + args.cells),) * 2
        cell_type = np.squeeze(mpm.scene.element.cell.type.to_numpy())[active_slice]
        numerical_pressure = np.squeeze(mpm.scene.element.cell.pressure.to_numpy())[active_slice]
        coordinates = (np.arange(args.cells) + 0.5) * dx
        xx, yy = np.meshgrid(coordinates, coordinates, indexing="ij")
        centers = np.column_stack((xx.ravel(), yy.ravel()))
        exact_pressure = analytical_pressure(
            centers, time, args.viscosity, density=1.0, amplitude=args.amplitude
        ).reshape(args.cells, args.cells)
        fluid = cell_type == 1
        numerical = numerical_pressure[fluid] - np.mean(numerical_pressure[fluid])
        exact = exact_pressure[fluid] - np.mean(exact_pressure[fluid])
        pressure_l2 = float(np.linalg.norm(numerical - exact) / max(np.linalg.norm(exact), np.finfo(float).eps))
        diagnostics.append(
            [
                time,
                int(active.sum()),
                velocity_l2,
                velocity_linf,
                kinetic_energy,
                exact_energy,
                abs(kinetic_energy - exact_energy) / exact_energy,
                pressure_l2,
                int(np.count_nonzero(cell_type == 0)),
            ]
        )

    mpm.select_save_data(particle=True, grid=True)
    mpm.run(gravity_field=False, function=compare_with_exact_solution)
    if not args.no_post:
        mpm.postprocessing()

    rows = np.asarray(diagnostics)
    verification_index = int(np.argmin(np.abs(rows[:, 0] - min(args.verification_time, args.time))))
    exact_final_fraction = math.exp(-2.0 * args.viscosity * float(rows[-1, 0]))
    numerical_final_fraction = math.sqrt(max(float(rows[-1, 4]), 0.0) / (0.25 * args.amplitude**2))
    metrics = {
        "case": "2-D incompressible MPM Taylor-Green vortex",
        "domain_m": DOMAIN.tolist(),
        "cells": args.cells,
        "ppc": args.ppc,
        "expected_particles": expected_particles,
        "minimum_particles": int(rows[:, 1].min()),
        "dt_s": args.dt,
        "final_time_s": float(rows[-1, 0]),
        "viscosity_m2_s": args.viscosity,
        "reynolds_number": args.amplitude / args.viscosity,
        "samples": len(rows),
        "verification_time_s": float(rows[verification_index, 0]),
        "verification_velocity_relative_l2": float(rows[verification_index, 2]),
        "verification_velocity_relative_linf": float(rows[verification_index, 3]),
        "verification_kinetic_energy_relative_error": float(rows[verification_index, 6]),
        "verification_pressure_relative_l2": float(rows[verification_index, 7]),
        "final_velocity_relative_l2": float(rows[-1, 2]),
        "maximum_velocity_relative_l2": float(rows[:, 2].max()),
        "final_velocity_relative_linf": float(rows[-1, 3]),
        "final_kinetic_energy_relative_error": float(rows[-1, 6]),
        "final_pressure_relative_l2": float(rows[-1, 7]),
        "maximum_interior_air_cells": int(rows[:, 8].max()),
        "steady_fraction_tolerance": args.steady_fraction,
        "analytical_final_velocity_fraction": exact_final_fraction,
        "numerical_final_velocity_fraction": numerical_final_fraction,
        "reached_practical_steady_state": bool(
            exact_final_fraction <= args.steady_fraction * (1.0 + 1.0e-10)
            and numerical_final_fraction <= 2.0 * args.steady_fraction
        ),
        "finite": bool(np.isfinite(rows).all()),
        "duration_complete": math.isclose(float(rows[-1, 0]), args.time, rel_tol=0.0, abs_tol=0.1 * args.dt),
        "strict_tolerances": STRICT_TOLERANCES,
    }
    metrics["passed"] = bool(
        metrics["finite"]
        and metrics["duration_complete"]
        and metrics["minimum_particles"] == expected_particles
        and metrics["maximum_interior_air_cells"] == 0
        and metrics["reached_practical_steady_state"]
        and metrics["verification_velocity_relative_l2"] <= STRICT_TOLERANCES["velocity_relative_l2"]
        and metrics["verification_velocity_relative_linf"] <= STRICT_TOLERANCES["velocity_relative_linf"]
        and metrics["verification_kinetic_energy_relative_error"] <= STRICT_TOLERANCES["kinetic_energy_relative_error"]
        and metrics["verification_pressure_relative_l2"] <= STRICT_TOLERANCES["pressure_relative_l2"]
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    np.savetxt(
        output / "taylor_green_comparison.csv",
        rows,
        delimiter=",",
        header=(
            "time_s,particles,velocity_relative_l2,velocity_relative_linf,kinetic_energy,"
            "exact_kinetic_energy,kinetic_energy_relative_error,pressure_relative_l2,interior_air_cells"
        ),
        comments="",
    )
    print(json.dumps(metrics, sort_keys=True))
    if args.strict and not metrics["passed"]:
        raise RuntimeError("Taylor-Green verification failed; inspect metrics.json")


if __name__ == "__main__":
    run(parse_args())

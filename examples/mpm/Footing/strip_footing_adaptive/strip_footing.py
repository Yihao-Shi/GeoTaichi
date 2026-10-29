"""Plane-strain StripFootingMC comparison with uniform, L1 and L2 adaptive grids."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"

import numpy as np
import taichi as ti

from geotaichi import MPM, init


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--arch", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--duration", type=float, default=17.5)
    parser.add_argument("--save-interval", type=float, default=1.0)
    parser.add_argument(
        "--refine-threshold",
        type=float,
        default=0.01,
        help="Accumulated equivalent plastic deviatoric strain; dimensionless",
    )
    parser.add_argument("--device-memory-gb", type=float, default=2.0)
    parser.add_argument("--smoke", action="store_true", help="Use a 2 by 2 soil block; not a production result")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not all(math.isfinite(x) and x > 0 for x in [args.duration, args.save_interval, args.refine_threshold]):
        parser.error("duration, save interval and refinement threshold must be finite and positive")
    if args.smoke and args.output is None:
        parser.error("--smoke requires a separate --output")
    if args.output is None:
        args.output = (
            Path(__file__).resolve().parent / "OutputData" / ["uniform", "adaptive_l1", "adaptive_l2"][args.level]
        )
    return args


def main():
    args = parse_args()
    if args.arch == "cpu":
        os.environ["GEOTAICHI_FORCE_CPU"] = "1"
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "run.json").exists() or any((args.output / "particles").glob("*.npz")):
        raise FileExistsError(f"Refusing to overwrite an existing run: {args.output}")
    started = time.perf_counter()
    init(
        dim=2,
        arch=args.arch,
        default_fp="float64",
        device_memory_GB=args.device_memory_gb,
        cpu_max_num_threads=4,
        log=False,
    )
    dt = 1e-5
    fine_h = 0.1
    coarse_h = fine_h * 2**args.level
    soil_size = 2.0 if args.smoke else 10.0
    domain = [soil_size, soil_size + 2.0]
    mpm = MPM(log=False)
    mpm.set_configuration(
        domain=domain,
        background_damping=0.0,
        gravity=[0.0, 0.0],
        alphaPIC=0.005,
        mapping="USL",
        shape_function="GIMP",
        stabilize="B-Bar Method",
        gauss_number=0,
    )
    mpm.set_solver(
        {
            "Timestep": dt,
            "SimulationTime": args.duration,
            "SaveInterval": args.save_interval,
            "SavePath": str(args.output),
        }
    )
    mpm.memory_allocate(
        {
            "max_material_number": 1,
            "max_particle_number": 200000 if not args.smoke else 10000,
            "max_constraint_number": {"max_velocity_constraint": 10000},
        }
    )
    mpm.add_contact(contact_type="GeoContact", friction=0.1, cut_off=0.8, penalty=[0.78, 2.5])
    mpm.add_material(
        model="MohrCoulomb",
        material={
            "MaterialID": 1,
            "Density": 1000.0,
            "YoungModulus": 1e5,
            "PoissonRatio": 0.49,
            "Cohesion": 1000.0,
            "Friction": 0.0,
            "Dilation": 0.0,
        },
    )
    element = {"ElementType": "Q4N2D", "ElementSize": [coarse_h, coarse_h]}
    if args.level:
        element["AdaptiveGrid"] = {
            "MaxLevel": args.level,
            "RefineInterval": 100,
            "RefineCriterion": "PlasticDevStrain",
            "RefineThreshold": args.refine_threshold,
            "RefineRatio": 1.0,
            "BufferCells": round(0.4 / coarse_h),
            "RefineParticles": True,
            "HangingConstraintMode": "ShapeFunction",
            "BridgingDomain": False,
        }
    mpm.add_element(element)
    regions = [
        ("soil", [0.0, 0.0], [soil_size, soil_size], 4),
        ("footing", [0.0, soil_size], [1.0, 2.0], 4),
    ]
    mpm.add_region(
        region=[
            {"Name": name, "Type": "Rectangle2D", "BoundingBoxPoint": point, "BoundingBoxSize": size}
            for name, point, size, _ in regions
        ]
    )
    for name, _, _, ppc in regions[:-1]:
        mpm.add_body(
            body={
                "Template": {
                    "RegionName": name,
                    "nParticlesPerCell": ppc,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "InitialVelocity": [0.0, 0.0],
                    "FixVelocity": ["Free", "Free"],
                }
            }
        )
    soil_count = int(mpm.scene.particleNum[0])
    mpm.add_body(
        body={
            "Template": {
                "RegionName": "footing",
                "nParticlesPerCell": 4 * 2**args.level,
                "BodyID": 1,
                "RigidBody": True,
                "Density": 1500.0,
                "InitialVelocity": [0.0, -0.02],
                "FixVelocity": ["Fix", "Fix"],
            }
        }
    )
    count = int(mpm.scene.particleNum[0])
    for start, end, velocity in [
        ([0.0, 0.0], [soil_size, 0.0], [0.0, 0.0]),
        ([0.0, 0.0], [0.0, soil_size + 2.0], [0.0, None]),
        ([soil_size, 0.0], [soil_size, soil_size + 2.0], [0.0, None]),
    ]:
        mpm.add_boundary_condition(
            boundary={"BoundaryType": "VelocityConstraint", "Velocity": velocity, "StartPoint": start, "EndPoint": end}
        )
    critical_dt = float(mpm.sims.CFL * mpm.scene.get_critical_timestep())
    if dt > critical_dt:
        raise ValueError(f"Prescribed dt={dt} exceeds native CFL bound {critical_dt}")
    steps = math.ceil(args.duration / dt)
    stride = 100
    history = ti.field(ti.f64, shape=(math.ceil(steps / stride), 11))
    step_index = ti.field(ti.i32, shape=())
    particle = mpm.scene.particle
    node = mpm.scene.node
    ramp_steps = round(5.0 / dt)
    initial_y = soil_size + 0.0125
    adaptive = args.level > 0
    refined_cell = getattr(mpm.scene.element, "refined_cell", None)
    split_particle_count = getattr(mpm.scene.element, "new_particle_num", None)

    def record_and_drive():
        step = step_index[None]
        if (step + 1) % stride == 0 or step + 1 == steps:
            row = step // stride
            history[row, 0] = ti.min((step + 1) * dt, args.duration)
            for ng in range(node.shape[0]):
                ti.atomic_add(history[row, 1], node[ng, 1].contact_force[0])
                ti.atomic_add(history[row, 2], node[ng, 1].contact_force[1])
                ti.atomic_add(history[row, 3], node[ng, 0].contact_force[1])
                if node[ng, 0].m > mpm.scene.mass_cut_off:
                    ti.atomic_add(history[row, 8], 1.0)
            history[row, 4] = initial_y - particle[soil_count].x[1]
            current_count = count
            if ti.static(adaptive):
                current_count = ti.max(count, split_particle_count[None])
                for cell in range(refined_cell.shape[0]):
                    level = int(refined_cell[cell])
                    if level > 0:
                        ti.atomic_add(history[row, 5], 1.0)
                    if level == 1:
                        ti.atomic_add(history[row, 6], 1.0)
                    elif level == 2:
                        ti.atomic_add(history[row, 7], 1.0)
            history[row, 9] = current_count
            history[row, 10] = current_count - (count - soil_count)
        # Match the original post-step ramp and the velocity held after its first run.
        velocity = -0.1 * ti.min(step, ramp_steps - 1) * dt / 5.0
        # Soil children are appended; rigid particles retain these original indices.
        for i in range(soil_count, count):
            particle[i].v = ti.Vector([0.0, velocity])
        step_index[None] += 1

    metadata = {
        "status": "running",
        "level": args.level,
        "coarse_h": coarse_h,
        "fine_h": fine_h,
        "particle_count": count,
        "soil_count": soil_count,
        "initial_spacing": coarse_h / 4,
        "soil_particles_per_cell_axis": 4,
        "rigid_spacing": fine_h / 4,
        "refine_particles": adaptive,
        "element": element,
        "dt": dt,
        "duration": args.duration,
        "save_interval": args.save_interval,
        "gauss_number": 0,
        "contact": {
            "law": mpm.scene.contact._name,
            "friction": mpm.scene.contact.friction,
            "cut_off": mpm.scene.contact.cut_off,
            "penalty": [mpm.scene.contact.alpha, mpm.scene.contact.beta],
        },
        "stabilize": "B-Bar Method",
        "backend": str(ti.lang.impl.current_cfg().arch),
        "precision": "float64",
        "smoke": args.smoke,
        "command": sys.argv,
        "history_columns": [
            "time_s",
            "footing_fx_N_per_m",
            "footing_fy_N_per_m",
            "soil_fy_N_per_m",
            "settlement_m",
            "refined_coarse_cells",
            "l1_coarse_cells",
            "l2_coarse_cells",
            "active_soil_nodes",
            "particle_count",
            "soil_particle_count",
        ],
        "historical_reference": "examples/mpm/Footing/FootingLargeStrain",
        "historical_comparison": "pending_evaluation",
    }
    contract = Path(__file__).with_name("model-contract.json")
    shutil.copyfile(contract, args.output / "executed-model-contract.json")
    shutil.copyfile(__file__, args.output / "executed-model.py.source")
    manifest = args.output / "run.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    mpm.select_save_data(particle=True, grid=False)
    try:
        mpm.run(function=record_and_drive)
        ti.sync()
        np.savetxt(
            args.output / "contact_history.csv",
            history.to_numpy(),
            delimiter=",",
            header=",".join(metadata["history_columns"]),
            comments="",
        )
        metadata.update(
            status="completed",
            final_time=mpm.sims.current_time,
            final_step=mpm.sims.current_step,
            effective_dt=float(mpm.sims.dt[None]),
            final_particle_count=int(mpm.scene.particleNum[0]),
            stage_timings={
                name: {"seconds": record.total, "calls": record.num} for name, record in mpm.sims.timer.records.items()
            },
            synchronized_stage_timers=mpm.sims.timer.synchronize,
            wall_seconds=time.perf_counter() - started,
        )
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Uniform 2D DP/MC comparison using section 4.3 parameters of Zhang et al. (2021), 103856."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"

import taichi as ti

from geotaichi import MPM, init


@ti.kernel
def maximum_speed(count: int, particle: ti.template()) -> float:
    speed = 0.0
    for i in range(count):
        ti.atomic_max(speed, particle[i].v.norm())
    return speed


@ti.kernel
def transfer_relaxed_state(count: int, source: ti.template(), target: ti.template()):
    # No facade transfers between independent elastic/plastic scenes; keep this stage transition on device.
    for i in range(count):
        target[i] = source[i]
        target[i].v = ti.Vector.zero(float, 2)
        target[i].velocity_gradient = ti.Matrix.zero(float, 2, 2)


def inside(position, radius=0.0):
    return position[0] < 20.0 or position[0] + position[1] < 25.0


def build_model(mpm, args, elastic=False):
    mpm.set_configuration(
        domain=[70.0, 12.0],
        is_2DAxisy=False,
        background_damping=0.7 if elastic else 0.0,
        gravity=[0.0, -9.81],
        alphaPIC=1.0 if elastic else 0.0,
        mapping="USL",
        shape_function="GIMP",
        random_field=False,
        stress_integration="SubStepping",
        solver_type="Explicit",
    )
    mpm.set_solver(
        {
            "Timestep": args.dt,
            "SimulationTime": 1.0 if elastic else args.duration,
            "SaveInterval": 1.0 if elastic else args.save_interval,
            "SavePath": str(args.output / "initialization" if elastic else args.output),
        }
    )
    mpm.memory_allocate(
        {
            "max_material_number": 1,
            "max_particle_number": 20000,
            "max_constraint_number": {"max_velocity_constraint": 4000, "max_friction_constraint": 2000},
        }
    )
    material = {"MaterialID": 1, "Density": 2000.0, "YoungModulus": 1e6, "PoissonRatio": 0.33}
    if not elastic:
        material.update(
            SoftType="Linear",
            Cohesion=20000.0,
            ResidualCohesion=4000.0,
            Friction=0.0,
            ResidualFriction=0.0,
            Dilation=0.0,
            ResidualDilation=0.0,
            PlasticDevStrain=0.0,
            # Native MC tensor-history invariant is the paper invariant divided by sqrt(2).
            ResidualPlasticDevStrain=(20000.0 - 4000.0)
            / (50000.0 * (math.sqrt(2.0) if args.material == "mc" else 1.0)),
        )
        if args.material == "dp":
            material["dpType"] = "Inscribed"
    model = "DruckerPrager" if args.material == "dp" else "MohrCoulomb"
    mpm.add_material(model="LinearElastic" if elastic else model, material=material)
    element = {"ElementType": "Q4N2D", "ElementSize": [args.cell_size, args.cell_size]}
    mpm.add_element(element)
    mpm.add_region(
        {
            "Name": "clay",
            "Type": "UserDefined",
            "BoundingBoxPoint": [0.0, 0.0],
            "BoundingBoxSize": [25.0, 5.0],
            "RegionVolume": lambda: 112.5,
            "RegionFunction": inside,
        }
    )
    mpm.add_body(
        {
            "Template": {
                "RegionName": "clay",
                "nParticlesPerCell": 2,
                "BodyID": 0,
                "MaterialID": 1,
                "InitialVelocity": [0.0, 0.0],
                "FixVelocity": ["Free", "Free"],
            }
        }
    )
    mpm.add_boundary_condition(
        [
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, 0.0],
                "StartPoint": [0.0, 0.0],
                "EndPoint": [25.0, 0.0],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, None],
                "StartPoint": [0.0, 0.0],
                "EndPoint": [0.0, 12.0],
            },
            {
                "BoundaryType": "FrictionConstraint",
                "Friction": args.friction,
                "Norm": [0.0, -1.0],
                "StartPoint": [25.0 + args.cell_size, 0.0],
                "EndPoint": [70.0, 0.0],
            },
        ]
    )
    mpm.select_save_data(grid=False)
    return mpm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", choices=("dp", "mc"), default="dp")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--save-interval", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=2e-4)
    parser.add_argument("--cell-size", type=float, default=0.2)
    parser.add_argument("--friction", type=float, default=0.3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = (
            Path(__file__).resolve().parent / "OutputData" / f"paper_4_3_{args.material}_uniform_h02_consistent_mu03"
        )
    if min(args.duration, args.save_interval, args.dt, args.cell_size) <= 0 or args.friction < 0:
        parser.error("times and cell size must be positive; friction must be nonnegative")
    if args.cell_size < 0.2:
        parser.error("cell sizes below 0.2 require a new particle capacity estimate")
    if not math.isclose(25 / args.cell_size, round(25 / args.cell_size)):
        parser.error("cell size must align the original toe x=25 with a grid node")
    args.output = args.output.resolve()
    if any(args.output.rglob("MPMParticle*.npz")):
        parser.error("output contains particle snapshots; choose a new directory")
    args.output.mkdir(parents=True, exist_ok=True)
    init(dim=2, arch="cpu", default_fp="float64", cpu_max_num_threads=4, log=False)
    metadata = {
        "command": sys.argv,
        "duration": args.duration,
        "save_interval": args.save_interval,
        "dt": args.dt,
        "cell_size": args.cell_size,
        "friction": args.friction,
        "adaptive": False,
        "material_model": "DruckerPrager" if args.material == "dp" else "MohrCoulomb",
        "dp_type": "Inscribed" if args.material == "dp" else None,
        "plastic_history": (
            "accumulated_full_tensor_norm" if args.material == "dp" else "native_deviatoric_tensor_invariant"
        ),
        "backend": str(ti.lang.impl.current_cfg().arch),
        "precision": "float64",
        "particle_capacity": 20000,
        "status": "initializing",
        "paper": "Zhang et al. (2021), Computers and Geotechnics 129, 103856, section 4.3",
    }
    manifest = args.output / "run.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    start = time.perf_counter()
    elastic = MPM()
    build_model(elastic, args, elastic=True)
    relaxation = []
    for elapsed in range(1, 11):
        if elapsed > 1:
            elastic.modify_parameters(SimulationTime=float(elapsed))
        elastic.run()
        speed = maximum_speed(int(elastic.scene.particleNum[0]), elastic.scene.particle)
        relaxation.append({"time": float(elapsed), "maximum_speed": speed})
        if speed < 0.002:
            break
    metadata["relaxation"] = relaxation
    if relaxation[-1]["maximum_speed"] >= 0.002:
        metadata["status"] = "initialization_failed"
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")
        raise RuntimeError("Elastic dynamic relaxation has not reached max speed < 0.002 m/s within 10 s")
    mpm = MPM()
    build_model(mpm, args)
    count = int(mpm.scene.particleNum[0])
    if count != int(elastic.scene.particleNum[0]):
        raise RuntimeError("Elastic and plastic scenes have different particle counts")
    transfer_relaxed_state(count, elastic.scene.particle, mpm.scene.particle)
    metadata.update(status="running", initialization_wall_seconds=time.perf_counter() - start)
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    mpm.run()
    metadata.update(
        status="completed",
        wall_seconds=time.perf_counter() - start,
        final_time=mpm.sims.current_time,
        final_step=mpm.sims.current_step,
        effective_dt=float(mpm.sims.dt[None]),
    )
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Triaxial shear of mixed MPM-soft and level-set DEM particles."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
CASE_DIR = Path(__file__).resolve().parent
SPHERE_MESH = ROOT / "assets/mesh/LSDEM/sphere.stl"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--rigid-count", type=int, default=12)
    parser.add_argument("--soft-count", type=int, default=12)
    parser.add_argument("--particles-per-cell", type=int, default=1)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--dt", type=float, default=2.0e-5)
    parser.add_argument("--save-every", type=int, default=400)
    parser.add_argument("--axial-speed", type=float, default=0.05)
    parser.add_argument("--confining-pressure", type=float, default=500.0)
    parser.add_argument("--smoke", action="store_true", help="Compile gate without deformation acceptance")
    parser.add_argument("--output-dir", default=str(CASE_DIR / "OutputData/soft_particle_lsdem_triaxial"))
    args = parser.parse_args()
    values = (
        args.rigid_count,
        args.soft_count,
        args.particles_per_cell,
        args.steps,
        args.dt,
        args.save_every,
        args.axial_speed,
        args.confining_pressure,
    )
    if not all(math.isfinite(float(value)) and value > 0 for value in values):
        parser.error("counts, time and loading controls must be finite and positive")
    return args


def quad(wall_id, vertices, normal, *, velocity=(0.0, 0.0, 0.0), pressure=None):
    specification = {
        "WallID": wall_id,
        "WallType": "Facet",
        "WallShape": "Polygon",
        "MaterialID": 1,
        "WallVertice": {f"vertice{index + 1}": np.asarray(point) for index, point in enumerate(vertices)},
        "OuterNormal": np.asarray(normal),
        "InitialVelocity": np.asarray(velocity),
    }
    if pressure is not None:
        specification.update(ControlType="Force", TargetStress=pressure, Alpha=0.75, LimitVelocity=0.025)
    return specification


def main():
    args = arguments()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import geotaichi as gt
    import taichi as ti
    from src.utils.SolverRuntime import python_callback

    gt.init(arch=args.arch, default_fp="float64", log=True, offline_cache=False)
    coupling = gt.MPDEM(log=True)
    coupling.set_configuration(
        domain=[1.0, 1.0, 1.0],
        coupling_scheme="MPDEM",
        particle_interaction=True,
        wall_interaction=False,
        gravity=[0.0, 0.0, 0.0],
        search="LinkedCell",
        visualize=True,
        log=True,
    )
    dem = coupling.dem
    dem.set_configuration(
        domain=[1.0, 1.0, 1.0],
        boundary=["Destroy", "Destroy", "Destroy"],
        gravity=[0.0, 0.0, 0.0],
        search="LinkedCell",
        scheme="LSMPM",
        shape_function="QuadBSpline",
        soft_grid_type="Hexahedron",
        visualize=True,
        log=True,
    )
    dem.memory_allocate(
        {
            "max_material_number": 2,
            "max_rigid_body_number": args.rigid_count,
            "max_soft_body_number": args.soft_count,
            "max_material_point_number": max(20000, 4096 * args.soft_count),
            "max_rigid_template_number": 1,
            "levelset_grid_number": max(30000, 2048 * (args.rigid_count + args.soft_count)),
            "surface_node_number": max(6000, 512 * (args.rigid_count + args.soft_count)),
            "max_facet_number": 12,
            "max_servo_wall_number": 4,
            "body_coordination_number": 32,
            "wall_coordination_number": 12,
            "wall_per_cell": 16,
            "verlet_distance_multiplier": [0.15, 0.15],
            "point_coordination_number": [20, 12],
            "compaction_ratio": [1.0, 1.0],
        },
        log=True,
    )
    coupling.set_solver(
        {
            "Timestep": args.dt,
            "SimulationTime": args.steps * args.dt,
            "SaveInterval": args.save_every * args.dt,
            "SavePath": args.output_dir,
        },
        log=True,
    )
    dem.add_attribute(
        0,
        {
            "Density": 1250.0,
            "ConstitutiveModel": "NeoHookean",
            "YoungModulus": 1.5e5,
            "PoissonRatio": 0.30,
            "ForceLocalDamping": 0.05,
            "TorqueLocalDamping": 0.05,
        },
    )
    dem.add_attribute(1, {"Density": 2500.0, "ForceLocalDamping": 0.0, "TorqueLocalDamping": 0.0})
    sphere = gt.polyhedron(file=str(SPHERE_MESH)).grids(space=0.20, extent=5)
    dem.add_template({"Name": "sphere", "Object": sphere})
    dem.add_region(
        {
            "Name": "specimen",
            "Type": "Rectangle",
            "BoundingBoxPoint": [0.25, 0.25, 0.25],
            "BoundingBoxSize": [0.50, 0.50, 0.50],
        }
    )
    common = {
        "Name": "sphere",
        "GroupID": 0,
        "MaterialID": 0,
        "MinBoundingRadius": 0.050,
        "MaxBoundingRadius": 0.058,
        "BodyOrientation": "uniform",
        "InitialVelocity": [0.0, 0.0, 0.0],
        "InitialAngularVelocity": [0.0, 0.0, 0.0],
        "FixMotion": ["Free", "Free", "Free"],
    }
    dem.add_body(
        {
            "GenerateType": "Generate",
            "RegionName": "specimen",
            "BodyType": "RigidBody",
            "PoissonSampling": False,
            "TryNumber": 30000,
            "Template": {**common, "BodyNumber": args.rigid_count},
        }
    )
    dem.add_body(
        {
            "GenerateType": "Generate",
            "RegionName": "specimen",
            "BodyType": "SoftBody",
            "PoissonSampling": False,
            "TryNumber": 30000,
            "Template": {
                **common,
                "BodyNumber": args.soft_count,
                "MaterialPointsPerCell": args.particles_per_cell,
            },
        }
    )

    lo, hi = 0.25, 0.75
    wall_specs = (
        quad(0, ((lo, lo, lo), (hi, lo, lo), (hi, hi, lo), (lo, hi, lo)), (0, 0, 1)),
        quad(
            1, ((lo, lo, hi), (lo, hi, hi), (hi, hi, hi), (hi, lo, hi)), (0, 0, -1), velocity=(0, 0, -args.axial_speed)
        ),
        quad(2, ((lo, lo, lo), (lo, hi, lo), (lo, hi, hi), (lo, lo, hi)), (1, 0, 0), pressure=args.confining_pressure),
        quad(3, ((hi, lo, lo), (hi, lo, hi), (hi, hi, hi), (hi, hi, lo)), (-1, 0, 0), pressure=args.confining_pressure),
        quad(4, ((lo, lo, lo), (lo, lo, hi), (hi, lo, hi), (hi, lo, lo)), (0, 1, 0), pressure=args.confining_pressure),
        quad(5, ((lo, hi, lo), (hi, hi, lo), (hi, hi, hi), (lo, hi, hi)), (0, -1, 0), pressure=args.confining_pressure),
    )
    for specification in wall_specs:
        dem.add_wall(specification)
    dem.choose_contact_model("Linear Model", "Linear Model")
    dem.add_property(
        0,
        0,
        {
            "NormalStiffness": 4.0e6,
            "TangentialStiffness": 2.0e6,
            "Friction": 0.45,
            "NormalViscousDamping": 0.15,
            "TangentialViscousDamping": 0.05,
        },
        dType="all",
    )
    dem.add_property(
        0,
        1,
        {
            "NormalStiffness": 8.0e6,
            "TangentialStiffness": 4.0e6,
            "Friction": 0.20,
            "NormalViscousDamping": 0.10,
            "TangentialViscousDamping": 0.03,
        },
        dType="all",
    )
    position = ti.Vector.field(3, float, shape=4)
    force = ti.Vector.field(3, float, shape=4)

    @ti.kernel
    def update_servo_measurements_kernel(wall: ti.template(), servo: ti.template()):
        for servo_id in range(4):
            position[servo_id] = servo[servo_id].get_geometry_center(wall)
            force[servo_id] = servo[servo_id].get_geometry_force(wall)
        width = position[1][0] - position[0][0]
        depth = position[3][1] - position[2][1]
        height = 0.5
        servo[0].update_area(height * depth)
        servo[1].update_area(height * depth)
        servo[2].update_area(width * height)
        servo[3].update_area(width * height)
        servo[0].update_current_force(-force[0][0])
        servo[1].update_current_force(force[1][0])
        servo[2].update_current_force(-force[2][1])
        servo[3].update_current_force(force[3][1])

    @python_callback
    def update_servo_measurements():
        update_servo_measurements_kernel(dem.scene.wall, dem.scene.servo)

    dem.select_save_data(particle=True, surface=True, wall=True)
    dem.servo_switch("On")
    coupling.run(callback=update_servo_measurements)

    point_count = int(dem.scene.softPointNum[0])
    soft = dem.scene.soft_point.to_numpy()
    servo = dem.scene.servo.to_numpy()
    measured_pressure = servo["current_force"][:4] / np.maximum(servo["area"][:4], 1.0e-30)
    walls = dem.scene.wall.to_numpy()
    wall_count = int(dem.scene.wallNum[0])
    top_indices = np.flatnonzero(walls["wallID"][:wall_count] == 1)
    final_top_z = float(
        np.mean(
            np.concatenate(
                (
                    walls["vertice1"][top_indices, 2],
                    walls["vertice2"][top_indices, 2],
                    walls["vertice3"][top_indices, 2],
                )
            )
        )
    )
    summary = {
        "case": "mpm_soft_particle_levelset_dem_triaxial_shear",
        "rigid_body_count": int(dem.scene.rigidNum[0] - dem.scene.softNum[0]),
        "soft_body_count": int(dem.scene.softNum[0]),
        "soft_material_point_count": point_count,
        "completed_time": float(dem.sims.current_time),
        "axial_displacement": float(hi - final_top_z),
        "target_confining_pressure": args.confining_pressure,
        "mean_final_confining_pressure": float(np.mean(measured_pressure)),
        "finite": bool(np.isfinite(soft["x"][:point_count]).all() and np.isfinite(measured_pressure).all()),
    }
    summary["passed"] = bool(
        summary["finite"]
        and summary["rigid_body_count"] == args.rigid_count
        and summary["soft_body_count"] == args.soft_count
        and (args.smoke or summary["axial_displacement"] > 0.001)
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not summary["passed"]:
        raise RuntimeError(f"soft-particle/LSDEM triaxial validation failed: {summary}")


if __name__ == "__main__":
    main()

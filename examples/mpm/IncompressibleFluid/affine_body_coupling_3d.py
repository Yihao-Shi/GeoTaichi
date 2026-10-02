#!/usr/bin/env python3
"""A submerged LevelSet affine body transferring momentum to incompressible MPM."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MESH = ROOT / "assets/mesh/AffineBody/lowpoly_sphere.obj"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "gpu"), default=os.environ.get("GEOTAICHI_ARCH", "gpu"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=2.0e-4)
    parser.add_argument("--device-memory-gb", type=float, default=2.0)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).with_name("OutputData") / "incompressible_affine_body_coupling_3d"),
    )
    args = parser.parse_args()
    if args.steps <= 0 or args.dt <= 0.0 or args.device_memory_gb <= 0.0:
        parser.error("steps, dt and device memory must be positive")

    import geotaichi as gt

    gt.init(
        arch=args.arch,
        default_fp="float64",
        device_memory_GB=args.device_memory_gb,
        offline_cache=False,
        log=True,
    )
    coupling = gt.DEMPM()
    domain = [0.32, 0.20, 0.28]
    spacing = 0.01
    coupling.set_configuration(
        domain=domain,
        coupling_scheme="MPDEM",
        particle_interaction=False,
        wall_interaction=False,
        gravity=[0.0, 0.0, 0.0],
        visualize=True,
    )

    mpm = coupling.mpm
    mpm.set_configuration(
        domain=domain,
        background_damping=0.0,
        alphaPIC=0.5,
        mapping="USL",
        shape_function="QuadBSpline",
        gravity=[0.0, 0.0, 0.0],
        material_type="Fluid",
        velocity_projection="Affine",
        solver_type="Implicit",
        discretization="FDM",
        fluid_level_set=True,
    )
    mpm.set_implicit_solver_parameters(
        linear_solver="MGPCG",
        multilevel=3,
        pre_and_post_smoothing=2,
        bottom_smoothing=12,
        max_iteration_number=200,
        residual_tolerance=1.0e-9,
    )
    mpm.memory_allocate(
        {
            "max_material_number": 1,
            "max_particle_number": 20000,
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
            "ElementLength": spacing,
            "cL": 1.5,
            "cQ": 2.0,
            "atmospheric_pressure": 0.0,
        },
    )
    mpm.add_element({"ElementType": "Staggered", "ElementSize": [spacing] * 3, "GhostCell": 1})
    mpm.add_region(
        {
            "Name": "water",
            "Type": "Rectangle",
            "BoundingBoxPoint": [0.02, 0.02, 0.02],
            "BoundingBoxSize": [0.28, 0.16, 0.22],
        }
    )
    mpm.add_body(
        {
            "Template": {
                "RegionName": "water",
                "nParticlesPerCell": 1,
                "BodyID": 0,
                "MaterialID": 1,
                "InitialVelocity": [0.0, 0.0, 0.0],
                "FixVelocity": ["Free", "Free", "Free"],
            }
        }
    )
    walls = []
    for axis in range(3):
        for side in (0, 1):
            point = [0.0, 0.0, 0.0]
            point[axis] = domain[axis] if side else 0.0
            normal = [0.0, 0.0, 0.0]
            normal[axis] = 1.0 if side else -1.0
            walls.append(
                {
                    "BoundaryType": "SolidCell",
                    "Norm": normal,
                    "StartPoint": point,
                    "EndPoint": [domain[d] if d != axis else point[d] for d in range(3)],
                    "CellThickness": 1,
                }
            )
    mpm.add_boundary_condition(walls)
    mpm.select_save_data(particle=True, grid=True)

    dem = coupling.dem
    dem.set_configuration(
        domain=domain,
        scheme="AffineBody",
        search="LinkedCell",
        gravity=[0.0, 0.0, 0.0],
        visualize=True,
    )
    dem.set_affine_body_parameters(
        assemble_type="HashTriplet",
        young_modulus=2.0e6,
        max_newton_iteration=40,
        newton_tolerance=1.0e-7,
        linear_tolerance=1.0e-9,
        linear_max_iteration=2000,
        max_step=0.02,
        ccd=False,
    )
    dem.memory_allocate(
        {
            "max_material_number": 1,
            "max_affine_body_number": 1,
            "max_rigid_template_number": 1,
            "levelset_grid_number": 100000,
            "surface_node_number": 12,
            "body_coordination_number": 1,
            "wall_coordination_number": 0,
            "compaction_ratio": [1.0, 1.0],
        }
    )
    dem.add_attribute(0, {"Density": 1200.0, "ForceLocalDamping": 0.0, "TorqueLocalDamping": 0.0})
    sphere = gt.polyhedron(file=str(MESH)).grids(space=0.2, extent=3)
    sphere.generate()
    dem.add_template(
        {
            "Name": "submerged_sphere",
            "TemplateType": "AffineBody",
            "Object": sphere,
            "ContactRepresentation": "LevelSet",
        }
    )
    dem.create_body(
        {
            "BodyType": "AffineBody",
            "Template": {
                "Name": "submerged_sphere",
                "GroupID": 0,
                "MaterialID": 0,
                "BodyPoint": [0.16, 0.10, 0.14],
                "BoundingRadius": 0.04,
                "InitialVelocity": [0.5, 0.0, 0.0],
            },
        }
    )
    dem.select_save_data(surface=True)

    coupling.set_solver(
        {
            "Timestep": args.dt,
            "SimulationTime": args.steps * args.dt,
            "SaveInterval": args.steps * args.dt,
            "SavePath": args.output_dir,
        }
    )
    coupling.memory_allocate({"body_coordination_number": 1, "wall_coordination_number": 0})
    result = coupling.run(mpm_gravity_field=False)

    particle_num = int(mpm.scene.particleNum[0])
    particle_position = mpm.scene.particle.x.to_numpy()[:particle_num]
    particle_velocity = mpm.scene.particle.v.to_numpy()[:particle_num]
    particle_pressure = mpm.recorder.sample_incompressible_cell_pressure(mpm.sims, mpm.scene, particle_position)
    cell_type = np.squeeze(mpm.scene.element.cell.type.to_numpy())
    cell_index = np.floor(particle_position / spacing).astype(np.int64) + mpm.scene.element.ghost_cell
    inside = np.all((cell_index >= 0) & (cell_index < np.asarray(cell_type.shape)), axis=1)
    occupied_type = cell_type[tuple(cell_index[inside].T)]
    operator = dem.enginer.operator
    affine_velocity = operator.velocity_y.to_numpy()[:4]
    hydrodynamic_force = np.sum(operator.external_generalized_force.to_numpy()[:4], axis=0)
    summary = {
        "case": "incompressible_mpm_affine_body_levelset_ibm",
        "steps": int(result["step"]),
        "completed_time": float(result["time"]),
        "particles": particle_num,
        "particles_outside_domain": int(np.count_nonzero(~inside)),
        "particles_in_air_cells": int(np.count_nonzero(occupied_type == 0)),
        "particles_in_solid_cells": int(np.count_nonzero(occupied_type == 2)),
        "fluid_speed_max": float(np.linalg.norm(particle_velocity, axis=1).max()),
        "pressure_range": [float(particle_pressure.min()), float(particle_pressure.max())],
        "body_mean_velocity": np.mean(affine_velocity, axis=0).tolist(),
        "hydrodynamic_force": hydrodynamic_force.tolist(),
        "finite": bool(
            np.isfinite(particle_velocity).all()
            and np.isfinite(particle_pressure).all()
            and np.isfinite(affine_velocity).all()
            and np.isfinite(hydrodynamic_force).all()
        ),
    }
    output = Path(args.output_dir)
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if (
        not summary["finite"]
        or summary["particles_outside_domain"] != 0
        or summary["particles_in_air_cells"] != 0
        or summary["particles_in_solid_cells"] != 0
        or not np.isclose(summary["completed_time"], args.steps * args.dt)
        or summary["fluid_speed_max"] <= 1.0e-10
        or np.linalg.norm(hydrodynamic_force) <= 1.0e-10
        or np.linalg.norm(np.mean(affine_velocity, axis=0) - np.array([0.5, 0.0, 0.0])) <= 1.0e-10
    ):
        raise RuntimeError(summary)


if __name__ == "__main__":
    main()

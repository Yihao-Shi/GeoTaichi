"""Generate soil first, then deposit and servo-compact a 9D stone column.

Run from the repository root. See README.md for the two-stage workflow.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

CASE = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE.parents[3]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--output-dir", type=Path, default=CASE / "OutputData/preparation")
    parser.add_argument("--smoke", action="store_true", help="Coarse grid, one batch, a few steps; not a specimen")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    if args.arch == "cpu":
        os.environ["GEOTAICHI_FORCE_CPU"] = "1"
    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    import geotaichi as gt
    import taichi as ti

    gt.init(dim=3, arch=args.arch, default_fp="float64", device_memory_GB=3, random_seed=0)
    model = gt.DEMPM()
    # DEM-only integration evaluates contact against the stationary MPM soil.
    model.set_configuration(
        domain=[0.8, 0.8, 1.85], coupling_scheme="DEM", particle_interaction=True, wall_interaction=False
    )
    model.dem.set_configuration(
        boundary=["Destroy"] * 3, gravity=[0, 0, -9.8], engine="VelocityVerlet", search="LinkedCell"
    )
    model.mpm.set_configuration(
        background_damping=0.2,
        alphaPIC=0.01,
        mapping="USF",
        shape_function="QuadBSpline",
        velocity_projection="Taylor",
        gravity=[0, 0, -9.8],
    )
    model.set_solver({"Timestep": 2e-5, "SimulationTime": 1.5, "SaveInterval": 3.0, "SavePath": str(output)})
    model.mpm.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_particle_number": 12000 if args.smoke else 764800,
            "verlet_distance_multiplier": 0.2,
            "max_constraint_number": {"max_velocity_constraint": 392043, "max_particle_traction_constraint": 203903},
        }
    )
    model.dem.memory_allocate(
        memory={
            "max_material_number": 2,
            "max_particle_number": 20000,
            "max_sphere_number": 20000,
            "max_facet_number": 4,
            "max_servo_wall_number": 1,
            "compaction_ratio": [0.4, 0.3],
            "verlet_distance_multiplier": 0.2,
            "body_coordination_number": 64,
            "wall_coordination_number": 2,
        }
    )
    model.memory_allocate(
        memory={"body_coordination_number": 32, "wall_coordination_number": 0, "compaction_ratio": [0.2, 0.0]}
    )
    for material, density, damping in [(0, 2580, 0.2), (1, 25800, 0.0)]:
        model.dem.add_attribute(
            materialID=material,
            attribute={"Density": density, "ForceLocalDamping": damping, "TorqueLocalDamping": damping},
        )
    model.dem.choose_contact_model(
        particle_particle_contact_model="Linear Rolling Model", particle_wall_contact_model="Linear Model"
    )
    model.dem.add_property(
        materialID1=0,
        materialID2=0,
        property={
            "EffectiveModulus": 1e7,
            "NormalToShearRatio": 1.5,
            "Friction": 0.81,
            "RollingFriction": 0.5,
            "TwistingFriction": 0.3,
            "NormalViscousDamping": 0.2,
            "TangentialViscousDamping": 0.2,
        },
    )
    model.dem.add_property(
        materialID1=0,
        materialID2=1,
        property={
            "EffectiveModulus": 1e7,
            "NormalToShearRatio": 1.5,
            "Friction": 0.5,
            "NormalViscousDamping": 0.1,
            "TangentialViscousDamping": 0.1,
        },
    )
    for wall_id, height, normal in [(0, 0.0, 1.0), (1, 1.85, -1.0)]:
        wall = {
            "WallID": wall_id,
            "WallType": "Facet",
            "WallShape": "Polygon",
            "MaterialID": 1,
            "WallVertice": {
                f"vertice{i+1}": [x, y, height]
                for i, (x, y) in enumerate([(0.35, 0.35), (0.45, 0.35), (0.45, 0.45), (0.35, 0.45)])
            },
            "OuterNormal": [0.0, 0.0, normal],
        }
        if wall_id == 1:
            wall.update(ControlType="Force", TargetStress=1e3, Alpha=0.4, LimitVelocity=0.01)
        model.dem.add_wall(body=wall)

    model.mpm.add_material(
        model="MohrCoulomb",
        material={
            "MaterialID": 1,
            "Density": 897,
            "YoungModulus": 4e6,
            "PoissonRatio": 0.4,
            "Friction": 0.0,
            "Dilation": 0.0,
            "Cohesion": 20000.0,
            "Tensile": 0.0,
        },
    )
    cell = 0.04 if args.smoke else 0.016
    model.mpm.add_element(element={"ElementType": "R8N3D", "ElementSize": [cell] * 3})

    def soil_region(position, radius=0.0):
        return (position[0] - 0.4) ** 2 + (position[1] - 0.4) ** 2 > (0.03 + radius) ** 2 and radius <= position[
            2
        ] <= 0.54 - radius

    model.mpm.add_region(
        region={
            "Name": "soil",
            "Type": "UserDefined",
            "BoundingBoxPoint": [0.0, 0.0, 0.0],
            "BoundingBoxSize": [0.8, 0.8, 0.54],
            "RegionFunction": soil_region,
            "RegionVolume": lambda: (0.8**2 - math.pi * 0.03**2) * 0.54,
        }
    )
    model.mpm.add_body(
        body={
            "Template": {
                "RegionName": "soil",
                "nParticlesPerCell": 1 if args.smoke else 2,
                "BodyID": 0,
                "MaterialID": 1,
                "InitialVelocity": [0.0, 0.0, 0.0],
                "FixVelocity": ["Fix"] * 3,
                "ParticleStress": {"InternalStress": [-1000.0, -1000.0, -1000.0, 0.0, 0.0, 0.0]},
            }
        }
    )

    def interface(position):
        r2 = (position[0] - 0.4) ** 2 + (position[1] - 0.4) ** 2
        return 0.03**2 < r2 < 0.08**2 and 0.0 <= position[2] <= 0.54

    model.mpm.choose_coupling_region(function=interface)
    initial_soil = model.mpm.scene.particle.x.to_numpy()[: model.mpm.scene.particleNum[0]].copy()
    model.mpm.select_save_data(grid=False)
    model.dem.select_save_data(sphere=True, wall=True, particle_particle_contact=True, particle_wall_contact=True)
    model.choose_contact_model(particle_particle_contact_model="Linear Model", particle_wall_contact_model=None)
    model.add_property(
        DEMmaterial=0,
        MPMmaterial=1,
        property={
            "EffectiveModulus": 1.2e6,
            "NormalToShearRatio": 1.5,
            "Friction": 0.3,
            "NormalViscousDamping": 0.05,
            "TangentialViscousDamping": 0.05,
        },
    )
    model.select_save_data(particle_particle_contact=True)

    def servo_force():
        force = model.dem.scene.servo[0].get_geometry_force(model.dem.scene.wall)
        model.dem.scene.servo[0].update_area(math.pi * 0.03**2)
        model.dem.scene.servo[0].update_current_force(ti.abs(force[2]))

    def move_top(height):
        current = model.dem.scene.wall[2].vertice1[2]
        model.dem.update_wall_status(
            wallID=1, property_name="Position", value=[0.0, 0.0, height - current], override=False
        )

    def trim(position, radius):
        return int(position[2] + radius >= 0.539 or (position[0] - 0.4) ** 2 + (position[1] - 0.4) ** 2 >= 0.0375**2)

    top = 0.0
    for batch in range(1 if args.smoke else 6):
        move_top(1.85)
        model.dem.servo_switch("Off")
        model.dem.add_region(
            region={
                "Name": "feed",
                "Type": "Cylinder",
                "BoundingBoxPoint": [0.38, 0.38, 1.1 * top],
                "BoundingBoxSize": [0.04, 0.04, 0.71],
            }
        )
        model.dem.add_body(
            body={
                "GenerateType": "Generate",
                "RegionName": "feed",
                "BodyType": "Sphere",
                "PoissonSampling": True,
                "TryNumber": 100000,
                "Template": {
                    "GroupID": 0,
                    "MaterialID": 0,
                    "MinRadius": 0.002,
                    "MaxRadius": 0.005,
                    "BodyNumber": 32 if args.smoke else 1200,
                },
            }
        )
        stage_start = model.sims.current_time
        model.modify_parameters(SimulationTime=stage_start + (4e-5 if args.smoke else 1.5))
        model.run(dem_callback=servo_force)
        if not args.smoke:
            model.dem.delete_particles(function=trim)
        top = float(model.dem.scene.find_right_top_scene(model.dem.sims)[2])
        move_top(1.12 * top)
        model.dem.servo_switch("On")
        # The original layer-wise servo method, with a 0.01 m/s wall speed cap.
        model.modify_parameters(SimulationTime=model.sims.current_time + (4e-5 if args.smoke else 10.0))
        model.run()
        model.dem.servo_switch("Off")
        top = float(model.dem.scene.find_right_top_scene(model.dem.sims)[2])
        if top >= 0.535:
            break
    if not args.smoke and top < 0.535:
        raise RuntimeError(f"Column remains underfilled after six layers: top={top:.6f} m")
    if not args.smoke:
        model.modify_parameters(SimulationTime=model.sims.current_time + 6.0)
        model.dem.servo_switch("On")
        model.run()
    n = int(model.mpm.scene.particleNum[0])
    np.testing.assert_array_equal(model.mpm.scene.particle.x.to_numpy()[:n], initial_soil)
    nd = int(model.dem.scene.particleNum[0])
    assert nd > 0 and np.isfinite(model.dem.scene.particle.x.to_numpy()[:nd]).all()
    assert np.isfinite(model.dem.scene.particle.v.to_numpy()[:nd]).all()
    assert (model.dem.scene.particle.rad.to_numpy()[:nd] > 0).all()
    metadata = {
        "frame": int(model.sims.current_print - 1),
        "smoke": args.smoke,
        "complete": not args.smoke,
        "column_top_m": top,
        "cell_size_m": cell,
        "soil_particles": n,
        "dem_particles": nd,
        "soil_fixed": True,
        "arch": str(ti.lang.impl.current_cfg().arch),
        "precision": "float64",
    }
    (output / "preparation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

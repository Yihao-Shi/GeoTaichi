"""Restore the prepared 9D column, equilibrate, and apply axial loading."""

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
    parser.add_argument("--preparation", type=Path, default=CASE / "OutputData/preparation")
    parser.add_argument("--output-dir", type=Path, default=CASE / "OutputData/loading")
    parser.add_argument("--smoke", action="store_true", help="A few steps using a smoke preparation")
    args = parser.parse_args()
    preparation, output = args.preparation.resolve(), args.output_dir.resolve()
    metadata = json.loads((preparation / "preparation.json").read_text())
    if metadata["smoke"] != args.smoke or (not args.smoke and not metadata["complete"]):
        raise ValueError("Loading requires a completed preparation with matching --smoke setting")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    if args.arch == "cpu":
        os.environ["GEOTAICHI_FORCE_CPU"] = "1"
    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    import geotaichi as gt
    import taichi as ti

    gt.init(dim=3, arch=args.arch, default_fp="float64", device_memory_GB=3)
    model = gt.DEMPM()
    model.set_configuration(
        domain=[0.8, 0.8, 1.85], coupling_scheme="DEM", particle_interaction=True, wall_interaction=False
    )
    model.dem.set_configuration(
        boundary=["Destroy"] * 3, gravity=[0.0, 0.0, -9.8], engine="VelocityVerlet", search="LinkedCell"
    )
    model.mpm.set_configuration(
        background_damping=0.05, alphaPIC=0.001, mapping="USF", shape_function="QuadBSpline", gravity=[0.0, 0.0, -9.8]
    )
    model.set_solver(
        {"Timestep": 2e-5, "SimulationTime": 4e-5 if args.smoke else 0.5, "SaveInterval": 0.1, "SavePath": str(output)}
    )
    model.mpm.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_particle_number": 12000 if args.smoke else 764800,
            "verlet_distance_multiplier": 0.6,
            "max_constraint_number": {"max_velocity_constraint": 603006, "max_particle_traction_constraint": 203903},
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
            "verlet_distance_multiplier": 0.1,
            "body_coordination_number": 64,
            "wall_coordination_number": 2,
        }
    )
    model.memory_allocate(
        memory={"body_coordination_number": 32, "wall_coordination_number": 0, "compaction_ratio": [0.2, 0.0]}
    )
    for material, density, damping in [(0, 2580, 0.7), (1, 25800, 0.0)]:
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
            "Friction": 0.81,
            "NormalViscousDamping": 0.1,
            "TangentialViscousDamping": 0.1,
        },
    )
    frame = metadata["frame"]
    model.dem.read_restart(file_number=frame, file_path=str(preparation), servo=True, is_continue=False)
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
    model.mpm.add_element(element={"ElementType": "R8N3D", "ElementSize": [metadata["cell_size_m"]] * 3})
    model.mpm.read_restart(file_number=frame, file_path=str(preparation), is_continue=False)
    model.mpm.update_particle_properties(property_name="fix_velocity", value=["Free"] * 3, bodyID=0)
    model.mpm.add_region(
        region={
            "Name": "surcharge",
            "Type": "Rectangle",
            "BoundingBoxPoint": [0.0, 0.0, 0.535],
            "BoundingBoxSize": [0.8, 0.8, 0.005],
        }
    )
    model.mpm.add_particle_traction(traction={"Pressure": [0.0, 0.0, -1000.0], "RegionName": "surcharge"})
    boundaries = [
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, 0.0, 0.0],
            "StartPoint": [0.0, 0.0, 0.0],
            "EndPoint": [0.8, 0.8, 0.0],
        }
    ]
    for axis in (0, 1):
        for side in (0.0, 0.8):
            start, end, velocity = [0.0, 0.0, 0.0], [0.8, 0.8, 0.55], [None] * 3
            start[axis] = end[axis] = side
            velocity[axis] = 0.0
            boundaries.append(
                {"BoundaryType": "VelocityConstraint", "Velocity": velocity, "StartPoint": start, "EndPoint": end}
            )
    model.mpm.add_boundary_condition(boundary=boundaries)

    # Keep the preparation's coupling prefix and contact-history indexing.
    model.mpm.select_save_data(grid=False)
    model.dem.select_save_data(sphere=True, wall=True, particle_particle_contact=True, particle_wall_contact=True)
    model.choose_contact_model(particle_particle_contact_model="Linear Model", particle_wall_contact_model=None)
    model.add_property(
        DEMmaterial=0,
        MPMmaterial=1,
        property={
            "EffectiveModulus": 1.2e6,
            "NormalToShearRatio": 1.0,
            "Friction": 0.5,
            "NormalViscousDamping": 0.05,
            "TangentialViscousDamping": 0.05,
        },
    )
    model.select_save_data(particle_particle_contact=True)
    model.read_restart(file_number=frame, file_path=str(preparation), ppcontact=True, pwcontact=False)

    def servo_force():
        force = model.dem.scene.servo[0].get_geometry_force(model.dem.scene.wall)
        model.dem.scene.servo[0].update_area(math.pi * 0.03**2)
        model.dem.scene.servo[0].update_current_force(ti.abs(force[2]))

    model.dem.servo_switch("On")
    model.run(dem_callback=servo_force)
    # Restart histories are consumed once; later stages retain the evolved history.
    model.read_restart(file_number=frame, file_path=str(preparation), ppcontact=False, pwcontact=False)
    model.dem.sims.history_contact_path.update(ppcontact=None, pwcontact=None)
    model.modify_parameters(SimulationTime=8e-5 if args.smoke else 0.6, coupling_scheme="MPDEM", SaveInterval=0.1)
    model.run(mpm_gravity_field=lambda points: 0.54 - points[:, 2])
    model.dem.servo_switch("Off")
    initial_top = float(model.dem.scene.wall[2].vertice1[2])
    model.dem.update_wall_status(wallID=1, property_name="Velocity", value=[0.0, 0.0, -0.001])
    model.modify_parameters(SimulationTime=1.2e-4 if args.smoke else 50.6, SaveInterval=1.0)
    model.run()
    settlement = initial_top - float(model.dem.scene.wall[2].vertice1[2])
    expected = 4e-8 if args.smoke else 0.05
    assert np.isclose(settlement, expected, rtol=1e-5, atol=1e-10), (settlement, expected)
    for scene in (model.mpm.scene, model.dem.scene):
        count = int(scene.particleNum[0])
        assert count > 0 and np.isfinite(scene.particle.x.to_numpy()[:count]).all()
        assert np.isfinite(scene.particle.v.to_numpy()[:count]).all()
    count = int(model.mpm.scene.particleNum[0])
    assert np.isfinite(model.mpm.scene.particle.stress.to_numpy()[:count]).all()
    volume = model.mpm.scene.particle.vol.to_numpy()[:count]
    assert np.isfinite(volume).all() and (volume > 0).all()
    summary = {
        "smoke": args.smoke,
        "time_s": float(model.sims.current_time),
        "settlement_m": settlement,
        "finite_state": True,
        "arch": str(ti.lang.impl.current_cfg().arch),
        "precision": "float64",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

import os
import sys
import json

from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import MPDEM, init, polyhedron


init(
    arch=os.environ.get("GT_ARCH", "cpu"),
    default_fp=os.environ.get("GT_DEFAULT_FP", "float64"),
    log=True,
    debug=False,
    offline_cache=False,
)

mesh_path = Path(ROOT) / "assets/mesh/LSDEM/sphere.stl"
sdf_extent = int(os.environ.get("GT_SOFT_LEVELSET_EXTENT", "5"))
save_path = os.environ.get(
    "GT_MPDEM_SAVE_PATH",
    os.environ.get("GT_DEM_SAVE_PATH", "OutputData/soft_rigid_generated_explicit"),
)
rigid_body_number = int(os.environ.get("GT_RIGID_BODY_NUMBER", "5"))
soft_body_number = int(os.environ.get("GT_SOFT_BODY_NUMBER", "4"))
body_coordination = int(os.environ.get("GT_DEM_BODY_COORDINATION", "24"))
wall_coordination = int(os.environ.get("GT_DEM_WALL_COORDINATION", "12"))
radius_min = float(os.environ.get("GT_BODY_MIN_RADIUS", "0.085"))
radius_max = float(os.environ.get("GT_BODY_MAX_RADIUS", "0.105"))
soft_grid_type = os.environ.get("GT_SOFT_GRID_TYPE", "Hexahedron")
total_body_number = rigid_body_number + soft_body_number

mpdem = MPDEM(log=True)
mpdem.set_configuration(
    domain=[1.05, 1.05, 1.45],
    coupling_scheme="MPDEM",
    particle_interaction=True,
    wall_interaction=False,
    gravity=[0.0, 0.0, -9.81],
    search=os.environ.get("GT_DEM_SEARCH", "LinkedCell"),
    visualize=True,
    log=True,
)

dem = mpdem.dem
dem.set_configuration(
    domain=[1.05, 1.05, 1.45],
    boundary=["Destroy", "Destroy", "Destroy"],
    gravity=[0.0, 0.0, -9.81],
    search=os.environ.get("GT_DEM_SEARCH", "LinkedCell"),
    scheme="LSMPM",
    shape_function=os.environ.get("GT_SOFT_SHAPE", "QuadBSpline"),
    soft_grid_type=soft_grid_type,
    visualize=True,
    log=True,
)

dem.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_rigid_body_number": rigid_body_number,
        "max_soft_body_number": soft_body_number,
        "max_material_point_number": max(
            20000,
            soft_body_number * 10000,
        ),
        "max_rigid_template_number": 1,
        "levelset_grid_number": max(20000, total_body_number * 2048),
        "surface_node_number": max(3000, total_body_number * 512),
        "max_plane_number": 6,
        "body_coordination_number": body_coordination,
        "wall_coordination_number": wall_coordination,
        "verlet_distance_multiplier": [0.15, 0.15],
        "point_coordination_number": [16, 8],
        "compaction_ratio": [1.0, 1.0],
        "wall_per_cell": 6,
    },
    log=True,
)

mpdem.set_solver(
    {
        "Timestep": float(os.environ.get("GT_DEM_DT", "1.0e-4")),
        "SimulationTime": float(os.environ.get("GT_DEM_SIM_TIME", "2.0")),
        "SaveInterval": float(os.environ.get("GT_DEM_SAVE_INTERVAL", "0.04")),
        "SavePath": save_path,
    },
    log=True,
)

soft_constitutive_model = os.environ.get("GT_SOFT_CONSTITUTIVE", "NeoHookean")
material_attribute = {
    "Density": 1200.0,
    "ConstitutiveModel": soft_constitutive_model,
    "YoungModulus": float(os.environ.get("GT_SOFT_YOUNG", "2.0e5")),
    "PoissonRatio": 0.3,
    "ForceLocalDamping": float(os.environ.get("GT_FORCE_LOCAL_DAMPING", "0.0")),
    "TorqueLocalDamping": float(os.environ.get("GT_TORQUE_LOCAL_DAMPING", "0.0")),
}
if soft_constitutive_model == "MooneyRivlin":
    material_attribute["Coefficient"] = [[0.0, 0.0], [2.0e4, 0.0]]
elif soft_constitutive_model == "Gent":
    material_attribute["Tensile1"] = 100.0
    material_attribute["Tensile2"] = 0.0
elif soft_constitutive_model == "Hydrogel":
    material_attribute["Tensile"] = 100.0

dem.add_attribute(materialID=0, attribute=material_attribute)

dem.add_template(
    template={
        "Name": "SphereTemplate",
        "Object": polyhedron(file=str(mesh_path)).grids(space=0.2, extent=sdf_extent),
    }
)

initial_layout = os.environ.get("GT_INITIAL_LAYOUT", "separated")
if initial_layout == "separated":
    rigid_positions = [
        [0.32, 0.32, 0.66],
        [0.32, 0.68, 0.66],
        [0.32, 0.32, 1.05],
        [0.32, 0.68, 1.05],
        [0.52, 0.50, 1.20],
    ]
    soft_positions = [
        [0.62, 0.32, 0.66],
        [0.62, 0.68, 0.66],
        [0.62, 0.32, 1.05],
        [0.62, 0.68, 1.05],
    ]
    if rigid_body_number > len(rigid_positions) or soft_body_number > len(soft_positions):
        raise ValueError("The separated example layout supports at most 5 rigid and 4 soft bodies")
    radius = 0.5 * (radius_min + radius_max)
    for position in rigid_positions[:rigid_body_number]:
        dem.create_body(
            body={
                "BodyType": "RigidBody",
                "Template": {
                    "Name": "SphereTemplate",
                    "GroupID": 0,
                    "MaterialID": 0,
                    "BodyPoint": position,
                    "BoundingRadius": radius,
                    "InitialVelocity": [0.25, 0.0, -0.35],
                    "InitialAngularVelocity": [0.0, 0.0, 0.0],
                    "BodyOrientation": "constant",
                    "FixMotion": ["Free", "Free", "Free"],
                },
            }
        )
    for position in soft_positions[:soft_body_number]:
        dem.create_body(
            body={
                "BodyType": "SoftBody",
                "Template": {
                    "Name": "SphereTemplate",
                    "GroupID": 0,
                    "MaterialID": 0,
                    "BodyPoint": position,
                    "BoundingRadius": radius,
                    "InitialVelocity": [-0.25, 0.0, -0.35],
                    "InitialAngularVelocity": [0.0, 0.0, 0.0],
                    "BodyOrientation": "constant",
                    "FixMotion": ["Free", "Free", "Free"],
                    "MaterialPointsPerCell": int(os.environ.get("GT_SOFT_PARTICLES_PER_CELL", "2")),
                },
            }
        )
elif initial_layout == "generated":
    dem.add_region(
        region={
            "Name": "drop_region",
            "Type": "Rectangle",
            "BoundingBoxPoint": [0.16, 0.16, 0.18],
            "BoundingBoxSize": [0.73, 0.73, 1.10],
        }
    )
    if rigid_body_number > 0:
        dem.add_body(
            body={
                "GenerateType": "Generate",
                "RegionName": "drop_region",
                "BodyType": "RigidBody",
                "PoissonSampling": False,
                "TryNumber": int(os.environ.get("GT_RIGID_TRY_NUMBER", "10000")),
                "Template": {
                    "Name": "SphereTemplate",
                    "GroupID": 0,
                    "MaterialID": 0,
                    "BodyNumber": rigid_body_number,
                    "MinBoundingRadius": radius_min,
                    "MaxBoundingRadius": radius_max,
                    "BodyOrientation": "uniform",
                    "InitialVelocity": [0.25, 0.0, -0.35],
                    "InitialAngularVelocity": [0.0, 0.0, 0.0],
                    "FixMotion": ["Free", "Free", "Free"],
                },
            }
        )
    if soft_body_number > 0:
        dem.add_body(
            body={
                "GenerateType": "Generate",
                "RegionName": "drop_region",
                "BodyType": "SoftBody",
                "PoissonSampling": False,
                "TryNumber": int(os.environ.get("GT_SOFT_TRY_NUMBER", "10000")),
                "Template": {
                    "Name": "SphereTemplate",
                    "GroupID": 0,
                    "MaterialID": 0,
                    "BodyNumber": soft_body_number,
                    "MinBoundingRadius": radius_min,
                    "MaxBoundingRadius": radius_max,
                    "BodyOrientation": "uniform",
                    "InitialVelocity": [-0.25, 0.0, -0.35],
                    "InitialAngularVelocity": [0.0, 0.0, 0.0],
                    "FixMotion": ["Free", "Free", "Free"],
                    "MaterialPointsPerCell": int(os.environ.get("GT_SOFT_PARTICLES_PER_CELL", "2")),
                },
            }
        )
else:
    raise ValueError("GT_INITIAL_LAYOUT must be 'separated' or 'generated'")

walls = [
    ([0.0, 0.0, 0.05], [0.0, 0.0, 1.0]),
    ([0.0, 0.0, 1.40], [0.0, 0.0, -1.0]),
    ([0.05, 0.0, 0.0], [1.0, 0.0, 0.0]),
    ([1.00, 0.0, 0.0], [-1.0, 0.0, 0.0]),
    ([0.0, 0.05, 0.0], [0.0, 1.0, 0.0]),
    ([0.0, 1.00, 0.0], [0.0, -1.0, 0.0]),
]
for wall_id, (center, normal) in enumerate(walls):
    dem.add_wall(
        body={
            "WallType": "Plane",
            "WallID": wall_id,
            "MaterialID": 0,
            "WallCenter": np.array(center),
            "OuterNormal": np.array(normal),
        }
    )

dem.choose_contact_model(
    particle_particle_contact_model="Linear Model",
    particle_wall_contact_model="Linear Model",
)

dem.add_property(
    materialID1=0,
    materialID2=0,
    property={
        "NormalStiffness": float(os.environ.get("GT_CONTACT_NORMAL_STIFFNESS", "5.0e7")),
        "TangentialStiffness": float(os.environ.get("GT_CONTACT_TANGENTIAL_STIFFNESS", "2.5e7")),
        "Friction": float(os.environ.get("GT_CONTACT_FRICTION", "0.45")),
        "NormalViscousDamping": float(os.environ.get("GT_CONTACT_NORMAL_DAMPING", "0.2")),
        "TangentialViscousDamping": float(os.environ.get("GT_CONTACT_TANGENTIAL_DAMPING", "0.05")),
    },
    dType="all",
)

dem.select_save_data(particle=True, surface=True, bounding=True, wall=True)
mpdem.run()

point_num = int(dem.scene.softPointNum[0])
soft_np = dem.scene.soft_point.to_numpy()
contact_force = soft_np["contact_force"][:point_num]
contact_norm = np.linalg.norm(contact_force, axis=1)

print(
    "RigidSoftSphereDropBox finished: "
    f"rigid={int(dem.scene.rigidNum[0] - dem.scene.softNum[0])}, "
    f"soft={int(dem.scene.softNum[0])}, "
    f"soft_points={int(dem.scene.softPointNum[0])}, "
    f"active_soft_contact_points={int(np.count_nonzero(contact_norm > 0.0))}, "
    f"max_soft_contact_force={float(contact_norm.max()) if point_num > 0 else 0.0}, "
    f"output={save_path}"
)

summary = {
    "case": "mpm_soft_particle_levelset_dem_pile",
    "rigid_body_count": int(dem.scene.rigidNum[0] - dem.scene.softNum[0]),
    "soft_body_count": int(dem.scene.softNum[0]),
    "soft_material_point_count": point_num,
    "active_soft_contact_points": int(np.count_nonzero(contact_norm > 0.0)),
    "maximum_soft_contact_force": float(contact_norm.max()) if point_num > 0 else 0.0,
    "finite": bool(np.isfinite(soft_np["x"][:point_num]).all() and np.isfinite(contact_force).all()),
}
structural_pass = bool(
    summary["finite"]
    and summary["rigid_body_count"] == rigid_body_number
    and summary["soft_body_count"] == soft_body_number
)
summary["passed"] = structural_pass and summary["active_soft_contact_points"] > 0
output = Path(save_path)
output.mkdir(parents=True, exist_ok=True)
(output / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
if not structural_pass or (not summary["passed"] and os.environ.get("GT_ALLOW_SMOKE", "0") != "1"):
    raise RuntimeError(f"soft-particle/LSDEM pile validation failed: {summary}")

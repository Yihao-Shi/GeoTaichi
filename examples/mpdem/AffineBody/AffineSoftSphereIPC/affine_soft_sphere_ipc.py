import os
import sys
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

default_fp = os.environ.get("GT_DEFAULT_FP", "float64")
os.environ["GEOTAICHI_REAL_DTYPE"] = default_fp

from geotaichi import MPDEM, init, polyhedron

init(
    arch=os.environ.get("GT_ARCH", "gpu"),
    default_fp=default_fp,
    log=True,
    debug=False,
    offline_cache=bool(int(os.environ.get("GT_OFFLINE_CACHE", "1"))),
)

soft_mesh = Path(ROOT) / "assets/mesh/LSDEM/sphere.stl"
affine_mesh = Path(
    os.environ.get(
        "GT_AFFINE_MESH",
        str(Path(ROOT) / "assets/mesh/AffineBody/lowpoly_sphere.obj"),
    )
)
save_path = os.environ.get("GT_MPDEM_SAVE_PATH", "soft_affine_ipc")
soft_body_number = int(os.environ.get("GT_SOFT_BODY_NUMBER", "2"))
affine_body_number = int(os.environ.get("GT_AFFINE_BODY_NUMBER", "2"))
total_body_number = soft_body_number + affine_body_number
collision_speed = float(os.environ.get("GT_IPC_COLLISION_SPEED", "0.18"))
direct_pair = bool(int(os.environ.get("GT_DIRECT_PAIR", "0")))
if direct_pair and (soft_body_number != 1 or affine_body_number != 1):
    raise ValueError("GT_DIRECT_PAIR requires one soft and one affine body")

mpdem = MPDEM(log=True)
mpdem.set_configuration(
    domain=[1.2, 1.0, 1.0],
    coupling_scheme="MPDEM",
    particle_interaction=True,
    wall_interaction=False,
    gravity=[0.0, 0.0, 0.0],
    search=os.environ.get("GT_DEM_SEARCH", "LinkedCell"),
    visualize=False,
    log=True,
)

dem = mpdem.dem
dem.set_configuration(
    domain=[1.2, 1.0, 1.0],
    boundary=["Destroy", "Destroy", "Destroy"],
    gravity=[0.0, 0.0, 0.0],
    search=os.environ.get("GT_DEM_SEARCH", "LinkedCell"),
    scheme="LSMPM",
    soft_rigid_contact="IPC",
    shape_function=os.environ.get("GT_SOFT_SHAPE", "QuadBSpline"),
    visualize=False,
    log=True,
)

dem.set_affine_body_parameters(
    assemble_type=os.environ.get("GT_AFFINE_ASSEMBLE_TYPE", "HashTriplet"),
    dhat=float(os.environ.get("GT_IPC_DHAT", "0.025")),
    barrier_stiffness=float(os.environ.get("GT_IPC_BARRIER_STIFFNESS", "5.0e5")),
    contact_damping_stiffness=float(os.environ.get("GT_IPC_CONTACT_DAMPING", "0.0")),
    friction_epsv=float(os.environ.get("GT_IPC_FRICTION_EPSV", "1.0e-4")),
    soft_background_damping=float(os.environ.get("GT_SOFT_BACKGROUND_DAMPING", "0.0")),
    max_newton_iteration=int(os.environ.get("GT_IPC_MAX_NEWTON", "4")),
    linear_tolerance=float(os.environ.get("GT_IPC_LINEAR_TOLERANCE", "1.0e-5")),
    linear_max_iteration=int(os.environ.get("GT_IPC_LINEAR_MAX_ITERATION", "1000")),
    line_search_max_iteration=int(os.environ.get("GT_IPC_LINE_SEARCH", "12")),
    max_step=float(os.environ.get("GT_IPC_MAX_STEP", "0.02")),
    ccd=True,
    ccd_type=os.environ.get("GT_IPC_CCD_TYPE", "ccd"),
    ccd_eta=float(os.environ.get("GT_IPC_CCD_ETA", "0.2")),
    ccd_max_iteration=int(os.environ.get("GT_IPC_CCD_MAX_ITERATION", "10000")),
)

dem.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_rigid_body_number": 0,
        "max_soft_body_number": soft_body_number,
        "max_material_point_number": max(12000, soft_body_number * 4096),
        "max_rigid_template_number": 1,
        "levelset_grid_number": max(16000, max(soft_body_number, 1) * 4096),
        "surface_node_number": max(3000, max(total_body_number, 1) * 512),
        "body_coordination_number": int(os.environ.get("GT_DEM_BODY_COORDINATION", "24")),
        "wall_coordination_number": 0,
        "verlet_distance_multiplier": [0.15, 0.15],
        "point_coordination_number": [16, 8],
        "compaction_ratio": [1.0, 1.0],
    },
    log=True,
)

time_step = float(os.environ.get("GT_DEM_DT", "1.0e-4"))
mpdem.set_solver(
    {
        "Timestep": time_step,
        "SimulationTime": float(os.environ.get("GT_DEM_SIM_TIME", "2.0")),
        "SaveInterval": float(os.environ.get("GT_DEM_SAVE_INTERVAL", "0.04")),
        "SavePath": save_path,
        "enable_step_retry": True,
        "step_retry_max_retries": 3,
        "step_retry_reduction": 0.5,
        "step_retry_minimum_timestep": time_step / 8.0,
    },
    log=True,
)

dem.add_attribute(
    materialID=0,
    attribute={
        "Density": 1200.0,
        "ConstitutiveModel": os.environ.get("GT_SOFT_CONSTITUTIVE", "NeoHookean"),
        "YoungModulus": float(os.environ.get("GT_SOFT_YOUNG", "2.0e5")),
        "PoissonRatio": 0.3,
        "ForceLocalDamping": 0.0,
        "TorqueLocalDamping": 0.0,
    },
)

dem.add_template(
    template={
        "Name": "SoftSphereTemplate",
        "Object": polyhedron(file=str(soft_mesh)).grids(
            space=float(os.environ.get("GT_SOFT_LEVELSET_SPACE", "0.2")),
            extent=int(os.environ.get("GT_SOFT_LEVELSET_EXTENT", "1")),
        ),
    }
)
dem.add_template(
    template={
        "Name": "AffineSphereTemplate",
        "TemplateType": "AffineBody",
        "Object": polyhedron(file=str(affine_mesh)),
    }
)

dem.add_region(
    region={
        "Name": "soft_region",
        "Type": "Rectangle",
        "BoundingBoxPoint": [0.22, 0.32, 0.42],
        "BoundingBoxSize": [0.20, 0.36, 0.18],
    }
)
dem.add_region(
    region={
        "Name": "affine_region",
        "Type": "Rectangle",
        "BoundingBoxPoint": [0.68, 0.32, 0.42],
        "BoundingBoxSize": [0.20, 0.36, 0.18],
    }
)

if direct_pair:
    radius = float(os.environ.get("GT_PAIR_RADIUS", "0.08"))
    dem.create_body(
        {
            "BodyType": "SoftBody",
            "Template": {
                "Name": "SoftSphereTemplate",
                "GroupID": 0,
                "MaterialID": 0,
                "BodyPoint": [0.38, 0.5, 0.5],
                "BoundingRadius": radius,
                "MaterialPointsPerCell": int(os.environ.get("GT_SOFT_PARTICLES_PER_CELL", "2")),
                "InitialVelocity": [collision_speed, 0.0, 0.0],
                "InitialAngularVelocity": [0.0, 0.0, 0.0],
                "BodyOrientation": [0.0, 0.0, 0.0],
                "FixMotion": ["Free", "Free", "Free"],
            },
        }
    )
    dem.create_body(
        {
            "BodyType": "AffineBody",
            "Template": {
                "Name": "AffineSphereTemplate",
                "GroupID": 0,
                "MaterialID": 0,
                "BodyPoint": [0.62, 0.5, 0.5],
                "BoundingRadius": radius,
                "InitialVelocity": [-collision_speed, 0.0, 0.0],
                "InitialAngularVelocity": [0.0, 0.0, 0.0],
                "BodyOrientation": [0.0, 0.0, 0.0],
                "Friction": float(os.environ.get("GT_IPC_FRICTION", "0.25")),
            },
        }
    )
elif soft_body_number > 0:
    dem.add_body(
        body={
            "GenerateType": "Generate",
            "RegionName": "soft_region",
            "BodyType": "SoftBody",
            "PoissonSampling": False,
            "TryNumber": int(os.environ.get("GT_SOFT_TRY_NUMBER", "5000")),
            "Template": {
                "Name": "SoftSphereTemplate",
                "GroupID": 0,
                "MaterialID": 0,
                "BodyNumber": soft_body_number,
                "MinBoundingRadius": float(os.environ.get("GT_SOFT_MIN_RADIUS", "0.075")),
                "MaxBoundingRadius": float(os.environ.get("GT_SOFT_MAX_RADIUS", "0.085")),
                "BodyOrientation": "uniform",
                "InitialVelocity": [collision_speed, 0.0, 0.0],
                "InitialAngularVelocity": [0.0, 0.0, 0.0],
                "FixMotion": ["Free", "Free", "Free"],
                "MaterialPointsPerCell": int(os.environ.get("GT_SOFT_PARTICLES_PER_CELL", "2")),
            },
        }
    )

if not direct_pair and affine_body_number > 0:
    dem.add_body(
        body={
            "GenerateType": "Generate",
            "RegionName": "affine_region",
            "BodyType": "AffineBody",
            "PoissonSampling": False,
            "TryNumber": int(os.environ.get("GT_AFFINE_TRY_NUMBER", "5000")),
            "Template": {
                "Name": "AffineSphereTemplate",
                "GroupID": 0,
                "MaterialID": 0,
                "BodyNumber": affine_body_number,
                "MinBoundingRadius": float(os.environ.get("GT_AFFINE_MIN_RADIUS", "0.075")),
                "MaxBoundingRadius": float(os.environ.get("GT_AFFINE_MAX_RADIUS", "0.085")),
                "BodyOrientation": "uniform",
                "InitialVelocity": [-collision_speed, 0.0, 0.0],
                "InitialAngularVelocity": [0.0, 0.0, 0.0],
                "Friction": float(os.environ.get("GT_IPC_FRICTION", "0.25")),
            },
        }
    )

dem.add_property(
    materialID1=0,
    materialID2=0,
    property={
        "Dhat": float(os.environ.get("GT_IPC_DHAT", "0.025")),
        "BarrierStiffness": float(os.environ.get("GT_IPC_BARRIER_STIFFNESS", "5.0e5")),
        "ContactDampingStiffness": float(os.environ.get("GT_IPC_CONTACT_DAMPING", "0.0")),
        "Friction": float(os.environ.get("GT_IPC_FRICTION", "0.25")),
    },
    dType="all",
)

dem.select_save_data(particle=True, surface=True, bounding=True, wall=False)
mpdem.run()

print(
    "AffineSoftSphereIPC finished: "
    f"soft={int(dem.scene.softNum[0])}, "
    f"affine={len(dem.scene.affine_bodies)}, "
    f"soft_points={int(dem.scene.softPointNum[0])}, "
    f"output={save_path}"
)

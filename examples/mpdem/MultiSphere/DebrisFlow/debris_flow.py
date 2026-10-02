import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

TERRAIN_FILE = os.path.join(ROOT, "assets", "data", "DebrisFlow", "dc7_dem_1411slide.txt")

import numpy as np

from geotaichi import *

save_path = os.environ.get("GT_SAVE_PATH", "OutputData")
restart_path = os.environ.get("GT_RESTART_PATH", save_path)
start_file = int(os.environ.get("GT_START_FILE", "-1"))
restart = start_file >= 0
simulation_time = float(os.environ.get("GT_SIM_TIME", "15.2"))
save_interval = float(os.environ.get("GT_SAVE_INTERVAL", "0.4"))
max_mpm_particles = int(os.environ.get("GT_MAX_MPM_PARTICLES", "1559361"))
device_memory_gb = float(os.environ.get("GT_DEVICE_MEMORY_GB", "4.0"))
wall_contact_ratio = float(os.environ.get("GT_WALL_CONTACT_RATIO", "0.15"))
max_wall_contact_pairs = int(os.environ.get("GT_MAX_WALL_CONTACT_PAIRS", "0"))
postprocess = os.environ.get("GT_POSTPROCESS", "1").lower() not in ("0", "false", "no", "off")
sparse_grid = os.environ.get("GT_SPARSE_GRID", "1").lower() not in ("0", "false", "no", "off")

if restart and "GT_MAX_MPM_PARTICLES" not in os.environ:
    restart_file = os.path.join(restart_path, "particles", f"MPMParticle{start_file:06d}.npz")
    restart_info = np.load(restart_file, allow_pickle=True)
    restart_particle_num = int(restart_info["body_num"])
    max_mpm_particles = max(restart_particle_num + 1024, int(1.1 * restart_particle_num))
    restart_info.close()

init(device_memory_GB=device_memory_gb, kernel_profiler=True, arch=os.environ.get("GT_ARCH", "gpu"))

dempm = DEMPM()

dempm.set_configuration(
    domain=ti.Vector([230.0, 260.0, 116.6]),
    coupling_scheme="MPM",
    particle_interaction=False,
    wall_interaction=True,
    digital_elevation_contact="heightfield",
)

dempm.mpm.set_configuration(
    background_damping=0.005,
    mode="Normal",
    alphaPIC=0.001,
    mapping="USF",
    shape_function="QuadBSpline",
    gravity=ti.Vector([0.0, 0.0, -9.8]),
    sparse_grid=sparse_grid,
)

dempm.dem.set_configuration(
    domain=ti.Vector([230.0, 260.0, 116.6]),
    gravity=ti.Vector([0.0, 0.0, -9.8]),
    engine="SymplecticEuler",
    search="LinkedCell",
)

dempm.set_solver(
    {"Timestep": 1e-4, "SimulationTime": simulation_time, "SaveInterval": save_interval, "SavePath": save_path}
)

dempm.dem.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_particle_number": 0,
        "max_sphere_number": 0,
        "max_digital_elevation_facet_number": 1,
        "verlet_distance_multiplier": 0.4,
        "body_coordination_number": 0,
        "wall_coordination_number": 18,
        "compaction_ratio": [0.25, 0.1],
    },
    log=True,
)

dempm.mpm.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_particle_number": max_mpm_particles,
        "verlet_distance_multiplier": 0.0,
        "max_constraint_number": {
            "max_reflection_constraint": 121914,
            "max_friction_constraint": 0,
            "max_velocity_constraint": 0,
        },
    }
)

dempm.memory_allocate(
    memory={
        "body_coordination_number": 0,
        "wall_coordination_number": 18,
        "compaction_ratio": [0.2, wall_contact_ratio],
        "max_wall_contact_pairs": max_wall_contact_pairs,
    }
)


dempm.dem.add_attribute(
    materialID=0, attribute={"Density": 26500, "ForceLocalDamping": 0.15, "TorqueLocalDamping": 0.05}
)

dempm.dem.choose_contact_model(particle_particle_contact_model=None, particle_wall_contact_model=None)

dempm.dem.add_wall(
    body={
        "WallType": "DigitalElevation",
        "WallID": 1,
        "MaterialID": 0,
        "DigitalElevation": np.flip(np.loadtxt(TERRAIN_FILE, skiprows=6) - 1597.4, 0),
        "CellSize": 1.0,
        "NoData": -11596.4,
        "Visualize": True,
    }
)

dempm.dem.select_save_data(particle=False)

dempm.mpm.add_material(
    model="DruckerPrager",
    material={
        "MaterialID": 1,
        "Density": 2650,
        "YoungModulus": 1e5,
        "PoissionRatio": 0.3,
        "Friction": 22,
        "Dilation": 0.0,
        "Cohesion": 0.0,
        "Tensile": 0.0,
    },
)

dempm.mpm.add_element(element={"ElementType": "R8N3D", "ElementSize": ti.Vector([0.5, 0.5, 0.5])})

if restart:
    dempm.mpm.read_restart(file_number=start_file, file_path=restart_path, is_continue=True)
    dempm.read_restart(file_number=start_file, file_path=restart_path)
else:
    dempm.mpm.add_body_from_file(
        body={"FileType": "TXT", "Template": {"BodyID": 0, "MaterialID": 1, "ParticleFile": "SpherePacking.txt"}}
    )

dempm.mpm.add_boundary_condition(
    boundary=[
        {
            "BoundaryType": "ReflectionConstraint",
            "Norm": [0.0, -1.0, 0.0],
            "StartPoint": [0.0, 0.0, 0.0],
            "EndPoint": [230.0, 0.0, 116.6],
        }
    ]
)

dempm.mpm.select_save_data()

dempm.choose_contact_model(particle_particle_contact_model=None, particle_wall_contact_model="Linear Model")

dempm.add_property(
    DEMmaterial=0,
    MPMmaterial=1,
    property={
        "NormalStiffness": 1e6,
        "TangentialStiffness": 1e6,
        "Friction": 0.5,
        "NormalViscousDamping": 0.5,
        "TangentialViscousDamping": 0.5,
    },
    dType="particle-wall",
)

dempm.run()

if postprocess:
    post_start_file = int(os.environ.get("GT_POST_START_FILE", "0"))
    post_start_path = os.path.join(save_path, "particles", f"MPMParticle{post_start_file:06d}.npz")
    if os.path.exists(post_start_path):
        dempm.mpm.postprocessing(start_file=post_start_file)
    else:
        print(f"Skip MPM postprocessing: {post_start_path} does not exist")

ti.profiler.print_kernel_profiler_info()

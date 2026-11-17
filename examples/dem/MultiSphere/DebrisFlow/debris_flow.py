import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import numpy as np

from geotaichi import *

simulation_time = float(os.environ.get("GT_SIM_TIME", "100"))
save_interval = float(os.environ.get("GT_SAVE_INTERVAL", "1"))
device_memory_gb = float(os.environ.get("GT_DEVICE_MEMORY_GB", "4.0"))
dem_search = os.environ.get("GT_DEM_SEARCH", "LinkedCell")

HERE = os.path.dirname(__file__)
TERRAIN_FILE = os.path.join(ROOT, "assets", "data", "DebrisFlow", "dc7_dem_1411slide.txt")


def data_file(filename):
    local_path = os.path.join(HERE, filename)
    if os.path.exists(local_path):
        return local_path
    raise FileNotFoundError(f"{filename} is not found in {HERE}")


init(
    arch=os.environ.get("GT_ARCH", "gpu"),
    log=False,
    debug=False,
    device_memory_GB=device_memory_gb,
    kernel_profiler=True,
)

dem = DEM()

dem.set_configuration(
    domain=[230.0, 260.0, 116.6],
    gravity=[0.0, 0.0, -9.8],
    boundary=["Destroy", "Destroy", "Destroy"],
    engine="SymplecticEuler",
    search=dem_search,
    digital_elevation_contact="heightfield",
)

dem.set_solver({"Timestep": 1e-3, "SimulationTime": simulation_time, "SaveInterval": save_interval})

dem.memory_allocate(
    memory={
        "max_material_number": 2,
        "max_particle_number": 1559361,
        "max_sphere_number": 1559361,
        "max_digital_elevation_facet_number": 1,
        "verlet_distance_multiplier": 0.4,
        "body_coordination_number": 16,
        "wall_coordination_number": 8,
        "compaction_ratio": [0.25, 0.1],
    },
    log=True,
)

dem.add_attribute(materialID=0, attribute={"Density": 2500, "ForceLocalDamping": 0.2, "TorqueLocalDamping": 0.2})

dem.add_attribute(materialID=1, attribute={"Density": 26500, "ForceLocalDamping": 0.1, "TorqueLocalDamping": 0.1})

dem.add_body_from_file(
    body={
        "WriteFile": True,
        "FileType": "TXT",
        "Template": {
            "BodyType": "Sphere",
            "GroupID": 0,
            "MaterialID": 0,
            "File": data_file("SpherePacking.txt"),
            "InitialVelocity": [0.0, 0.0, 0.0],
            "InitialAngularVelocity": [0.0, 0.0, 0.0],
            "FixVelocity": ["Free", "Free", "Free"],
            "FixAngularVelocity": ["Free", "Free", "Free"],
            "ParticleNumber": 1559360,
        },
    }
)

dem.add_wall(
    body={
        "WallType": "DigitalElevation",
        "WallID": 1,
        "MaterialID": 1,
        "DigitalElevation": np.flip(np.loadtxt(TERRAIN_FILE, skiprows=6) - 1597.4, 0),
        "CellSize": 1.0,
        "NoData": -11596.4,
        "Visualize": True,
    }
)

dem.set_static_wall()

dem.choose_contact_model(particle_particle_contact_model="Linear Model", particle_wall_contact_model="Linear Model")

dem.add_property(
    materialID1=0,
    materialID2=0,
    property={
        "NormalStiffness": 1e5,
        "TangentialStiffness": 1e5,
        "Friction": 0.5,
        "NormalViscousDamping": 0.5,
        "TangentialViscousDamping": 0.5,
    },
)

dem.add_property(
    materialID1=0,
    materialID2=1,
    property={
        "NormalStiffness": 1e6,
        "TangentialStiffness": 1e6,
        "Friction": 0.5,
        "NormalViscousDamping": 0.5,
        "TangentialViscousDamping": 0.5,
    },
)

dem.select_save_data(particle=True, wall=False)

dem.run()

ti.profiler.print_kernel_profiler_info()

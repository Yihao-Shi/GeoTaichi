import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

TERRAIN_FILE = os.path.join(ROOT, "assets", "data", "DebrisFlow", "dc7_dem_1411slide.txt")


from geotaichi import *

simulation_time = float(os.environ.get("GT_SIM_TIME", "16"))
save_interval = float(os.environ.get("GT_SAVE_INTERVAL", "0.4"))
device_memory_gb = float(os.environ.get("GT_DEVICE_MEMORY_GB", "7.6"))

init(
    arch=os.environ.get("GT_ARCH", "gpu"),
    log=True,
    debug=False,
    device_memory_GB=device_memory_gb,
    kernel_profiler=True,
)

lsdem = DEM()

lsdem.set_configuration(
    domain=ti.Vector([235.0, 265.0, 120.0]),
    scheme="LSDEM",
    gravity=ti.Vector([0.0, 0.0, -9.8]),
    visualize=False,
    digital_elevation_contact="heightfield",
)

lsdem.memory_allocate(
    memory={
        "max_material_number": 2,
        "max_rigid_body_number": 194920,
        "levelset_grid_number": 29600,
        "surface_node_number": 127,
        "max_digital_elevation_facet_number": 1,
        "body_coordination_number": 40,
        "wall_coordination_number": 8,
        "verlet_distance_multiplier": [0.1, 0.2],
        "point_coordination_number": [5, 1],
        "compaction_ratio": [0.32, 0.25, 0.15, 0.1],
    }
)

lsdem.set_solver({"Timestep": 1e-3, "SimulationTime": simulation_time, "SaveInterval": save_interval})

lsdem.add_attribute(materialID=0, attribute={"Density": 2650, "ForceLocalDamping": 0.05, "TorqueLocalDamping": 0.05})

lsdem.add_template(
    template={
        "Name": "Template1",
        "Object": polyhedron(file=f"{ROOT}/assets/mesh/LSDEM/sand.stl").grids(space=5, extent=3),
        "WriteFile": True,
    }
)

lsdem.add_body_from_file(
    body={
        "FileType": "TXT",
        "Template": [
            {
                "Name": "Template1",
                "BodyType": "RigidBody",
                "File": "BoundingSphere.txt",
                "GroupID": 0,
                "MaterialID": 0,
                "InitialVelocity": ti.Vector([0.0, 0.0, 0.0]),
                "InitialAngularVelocity": ti.Vector([0.0, 0.0, 0.0]),
                "ParticleNumber": 194920,
            }
        ],
    }
)

lsdem.add_wall(
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

lsdem.set_static_wall()

lsdem.choose_contact_model(particle_particle_contact_model="Linear Model", particle_wall_contact_model="Linear Model")


lsdem.add_property(
    materialID1=0,
    materialID2=0,
    property={
        "NormalStiffness": 1e6,
        "TangentialStiffness": 1e6,
        "Friction": 0.5,
        "NormalViscousDamping": 0.15,
        "TangentialViscousDamping": 0.15,
    },
)

lsdem.add_property(
    materialID1=0,
    materialID2=1,
    property={
        "NormalStiffness": 5e6,
        "TangentialStiffness": 5e6,
        "Friction": 0.3,
        "NormalViscousDamping": 0.15,
        "TangentialViscousDamping": 0.15,
    },
)

lsdem.select_save_data(particle=False, surface=False, wall=False)

lsdem.run()

ti.profiler.print_kernel_profiler_info()

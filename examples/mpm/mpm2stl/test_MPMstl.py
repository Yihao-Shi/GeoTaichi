import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from geotaichi import *

init(device_memory_GB=5.5, debug=True)

dempm = DEMPM()

dempm.set_configuration(
    domain=ti.Vector([5.0, 5.0, 5.0]), coupling_scheme="MPM", particle_interaction=False, wall_interaction=True
)

dempm.mpm.set_configuration(
    background_damping=0.005,
    #   mode="Lightweight",
    alphaPIC=0.001,
    mapping="USF",
    shape_function="QuadBSpline",
    gravity=ti.Vector([0.0, 0.0, -9.8]),
)

dempm.dem.set_configuration(
    domain=ti.Vector([5.0, 5.0, 5.0]),
    gravity=ti.Vector([0.0, 0.0, -9.8]),
    engine="SymplecticEuler",
    search="LinkedCell",
)

dempm.set_solver({"Timestep": 1e-4, "SimulationTime": 1.0, "SaveInterval": 0.01})

dempm.dem.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_particle_number": 0,
        "max_sphere_number": 0,
        "max_patch_number": 5632,
        "verlet_distance_multiplier": 0.01,
        "body_coordination_number": 24,
        "wall_coordination_number": 24,
        "compaction_ratio": [1.0, 1.0],
    },
    log=True,
)

dempm.mpm.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_particle_number": 80000,
        "verlet_distance_multiplier": 0.01,
        "max_constraint_number": {
            "max_reflection_constraint": 0,
            "max_friction_constraint": 0,
            "max_velocity_constraint": 0,
        },
    }
)

dempm.memory_allocate(
    memory={"body_coordination_number": 14, "wall_coordination_number": 80, "compaction_ratio": [1.0, 1.0]}
)


dempm.dem.add_attribute(
    materialID=0, attribute={"Density": 26500, "ForceLocalDamping": 0.15, "TorqueLocalDamping": 0.05}
)

dempm.dem.choose_contact_model(particle_particle_contact_model=None, particle_wall_contact_model=None)

dempm.dem.add_wall(
    body={
        "WallType": "Patch",
        "WallID": 0,
        "WallFile": f"{ROOT}/assets/mesh/MPM/plane.stl",
        "Translation": [0.0, 0.0, 0.05],
        "Counterclockwise": None,
        "MaterialID": 0,
        "Visualize": False,
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

dempm.mpm.add_element(element={"ElementType": "R8N3D", "ElementSize": ti.Vector([0.1, 0.1, 0.1])})

dempm.mpm.add_region(
    region={
        "Name": "region1",
        "Type": "Rectangle",
        "BoundingBoxPoint": ti.Vector([2.0, 2.0, 0.1]),
        "BoundingBoxSize": ti.Vector([1.0, 1.0, 1.0]),
    }
)

dempm.mpm.add_body(
    body={
        "Template": {
            "RegionName": "region1",
            "nParticlesPerCell": 2,
            "BodyID": 0,
            "MaterialID": 1,
            #   "InitialVelocity":[0, 0],
            "FixVelocity": ["Free", "Free", "Free"],
        }
    }
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

dempm.mpm.postprocessing()

# ti.profiler.print_kernel_profiler_info()

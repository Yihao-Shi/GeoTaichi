import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-l", type=float, default=0.005)
args = parser.parse_args()

from geotaichi import *

init()

dempm = DEMPM()

dempm.set_configuration(
    domain=ti.Vector([0.5, 0.2, 0.3]), coupling_scheme="MPDEM", particle_interaction=True, wall_interaction=True
)

dempm.mpm.set_configuration(
    background_damping=0.005, alphaPIC=0.001, mapping="USL", shape_function="GIMP", gravity=ti.Vector([0.0, 0.0, -9.8])
)

dempm.dem.set_configuration(
    boundary=["Destroy", "Destroy", "Destroy"],
    gravity=ti.Vector([0.0, 0.0, -9.8]),
    engine="VelocityVerlet",
    search="LinkedCell",
    scheme="LSDEM",
)


save_path = "OutputData"
if args.l != 0.01:
    save_path += "_l" + str(args.l)
dempm.set_solver({"Timestep": 2e-5, "SimulationTime": 0.5505, "SaveInterval": 0.05, "SavePath": save_path})

dempm.dem.memory_allocate(
    memory={
        "max_material_number": 3,
        "max_rigid_body_number": 6,
        "levelset_grid_number": 30923,
        "surface_node_number": 338,
        "max_plane_number": 6,
        "body_coordination_number": 6,
        "wall_coordination_number": 4,
        "verlet_distance_multiplier": [0.15, 0.2],
        "point_coordination_number": [3, 2],
        "compaction_ratio": [0.3, 0.5, 0.5, 0.5],
    }
)

dempm.mpm.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_particle_number": 256000,
        "verlet_distance_multiplier": 1.0,
        "max_constraint_number": {
            "max_reflection_constraint": 0,
            "max_friction_constraint": 0,
            "max_velocity_constraint": 12322,
        },
    }
)

dempm.memory_allocate(
    memory={"body_coordination_number": 6, "wall_coordination_number": 3, "compaction_ratio": [0.2, 0.15]}
)


dempm.dem.add_attribute(materialID=0, attribute={"Density": 500, "ForceLocalDamping": 0.15, "TorqueLocalDamping": 0.05})

dempm.dem.add_attribute(materialID=1, attribute={"Density": 8500, "ForceLocalDamping": 0.0, "TorqueLocalDamping": 0.0})

dempm.dem.add_template(
    template={
        "Name": "clump1",
        "Object": polyhedron(file=f"{ROOT}/assets/mesh/MPDEM/block.stl").grids(space=0.002, extent=3).reset(False),
        "SurfaceResolution": 4902,
    }
)
# polyhedron(file=f'{ROOT}/assets/mesh/MPDEM/block.stl').grids(space=0.002, extent=1).reset(False)
# box((0.02, 0.018, 0.018)).grids(space=0.002, extent=1).reset(False)

dempm.dem.create_body(
    body={
        "GenerateType": "Create",
        "BodyType": "RigidBody",
        "Template": [
            {
                "Name": "clump1",
                "GroupID": 0,
                "MaterialID": 0,
                "BodyPoint": [0.31, 0.1, 0.009],
                "ScaleFactor": 1.0,
                "FixMotion": ["Fix", "Fix", "Fix"],
            },
            {"Name": "clump1", "GroupID": 0, "MaterialID": 0, "BodyPoint": [0.31, 0.1, 0.027], "ScaleFactor": 1.0},
            {"Name": "clump1", "GroupID": 0, "MaterialID": 0, "BodyPoint": [0.31, 0.1, 0.045], "ScaleFactor": 1.0},
        ],
    }
)

dempm.dem.choose_contact_model(
    particle_particle_contact_model="Linear Model", particle_wall_contact_model="Linear Model"
)

dempm.dem.add_property(
    materialID1=0,
    materialID2=0,
    property={
        "NormalStiffness": 1e5,
        "TangentialStiffness": 1e5,
        "Friction": 0.6,
        "NormalViscousDamping": 0.15,
        "TangentialViscousDamping": 0.15,
    },
)

dempm.dem.add_property(
    materialID1=0,
    materialID2=1,
    property={
        "NormalStiffness": 8e5,
        "TangentialStiffness": 7e5,
        "Friction": 0.6,
        "NormalViscousDamping": 0.1,
        "TangentialViscousDamping": 0.1,
    },
)

dempm.dem.add_property(
    materialID1=0,
    materialID2=2,
    property={
        "NormalStiffness": 8e5,
        "TangentialStiffness": 7e5,
        "Friction": 0.0,
        "NormalViscousDamping": 0.0,
        "TangentialViscousDamping": 0.0,
    },
)


dempm.dem.add_wall(
    body={
        "WallType": "Plane",
        "MaterialID": 1,
        "WallCenter": ti.Vector([0.25, 0.1, 0.0]),
        "OuterNormal": ti.Vector([0.0, 0.0, 1.0]),
    }
)

dempm.dem.add_wall(
    body={
        "WallType": "Plane",
        "MaterialID": 1,
        "WallCenter": ti.Vector([0.0, 0.1, 0.15]),
        "OuterNormal": ti.Vector([1.0, 0.0, 0.0]),
    }
)

dempm.dem.add_wall(
    body={
        "WallType": "Plane",
        "MaterialID": 1,
        "WallCenter": ti.Vector([0.5, 0.1, 0.15]),
        "OuterNormal": ti.Vector([-1.0, 0.0, 0.0]),
    }
)

dempm.dem.add_wall(
    body={
        "WallType": "Plane",
        "MaterialID": 2,
        "WallCenter": ti.Vector([0.25, 0.0, 0.15]),
        "OuterNormal": ti.Vector([0.0, 1.0, 0.0]),
    }
)

dempm.dem.add_wall(
    body={
        "WallType": "Plane",
        "MaterialID": 2,
        "WallCenter": ti.Vector([0.25, 0.2, 0.15]),
        "OuterNormal": ti.Vector([0.0, -1.0, 0.0]),
    }
)


dempm.dem.select_save_data(surface=True)

dempm.mpm.add_material(
    model="DruckerPrager",
    material={
        "MaterialID": 1,
        "Density": 1300,
        "YoungModulus": 5e4,
        "PoissionRatio": 0.4,
        "Friction": 22,
        "Dilation": 0.0,
        "Cohesion": 0.0,
        "Tensile": 0.0,
    },
)

dempm.mpm.add_element(element={"ElementType": "R8N3D", "ElementSize": ti.Vector([args.l, args.l, args.l])})


dempm.mpm.add_region(
    region=[
        {
            "Name": "region1",
            "Type": "Rectangle",
            "BoundingBoxPoint": ti.Vector([0.0, 0.0, 0.0]),
            "BoundingBoxSize": ti.Vector([0.1, 0.2, 0.2]),
        }
    ]
)

dempm.mpm.add_body(
    body={
        "Template": [
            {
                "RegionName": "region1",
                "nParticlesPerCell": 2,
                "BodyID": 0,
                "MaterialID": 1,
                "InitialVelocity": ti.Vector([0, 0, 0]),
                "FixVelocity": ["Free", "Free", "Free"],
            }
        ]
    }
)

"""dempm.mpm.add_boundary_condition(boundary=[
                                    {
                                        "BoundaryType":   "VelocityConstraint",
                                        "Velocity":       [None, 0., None],
                                        "StartPoint":     [0., 0., 0.],
                                        "EndPoint":       [0.5, 0.0, 0.3]
                                    },
                                    
                                    {
                                        "BoundaryType":   "VelocityConstraint",
                                        "Velocity":       [None, 0., None],
                                        "StartPoint":     [0, 0.02, 0],
                                        "EndPoint":       [0.5, 0.02, 0.3]
                                    }])"""

dempm.mpm.select_save_data()

dempm.choose_contact_model(particle_particle_contact_model="Linear Model", particle_wall_contact_model="Linear Model")

dempm.add_property(
    DEMmaterial=0,
    MPMmaterial=1,
    property={
        "NormalStiffness": 9e3,
        "TangentialStiffness": 6e3,
        "Friction": 0.55,
        "NormalViscousDamping": 0.15,
        "TangentialViscousDamping": 0.15,
    },
)

dempm.add_property(
    DEMmaterial=1,
    MPMmaterial=1,
    property={
        "NormalStiffness": 5e4,
        "TangentialStiffness": 3e4,
        "Friction": 0.16,
        "NormalViscousDamping": 0.2,
        "TangentialViscousDamping": 0.2,
    },
)

dempm.add_property(
    DEMmaterial=2,
    MPMmaterial=1,
    property={
        "NormalStiffness": 5e4,
        "TangentialStiffness": 3e4,
        "Friction": 0.0,
        "NormalViscousDamping": 0.0,
        "TangentialViscousDamping": 0.0,
    },
)


dempm.run(mpm_gravity_field=True)

"""dempm.dem.update_material_properties(0, "ForceLocalDamping", 0.)
dempm.dem.update_material_properties(0, "TorqueLocalDamping", 0.)

print(dempm.dem.scene.material[0].fdamp)

dempm.modify_parameters(SimulationTime=0.55)

dempm.run()"""

dempm.mpm.postprocessing()

dempm.dem.postprocessing()

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(device_memory_GB=4, debug=False)

dempm = DEMPM()

dempm.set_configuration(
    domain=ti.Vector([0.004, 0.004, 0.08]), coupling_scheme="MPDEM", particle_interaction=True, wall_interaction=False
)

dempm.mpm.set_configuration(
    background_damping=0.025,
    alphaPIC=0.00,
    mapping="USL",
    shape_function="GIMP",
    gravity=ti.Vector([0.0, 0.0, -9.8]),
    material_type="Fluid",
)

dempm.dem.set_configuration(
    boundary=["Destroy", "Destroy", "Destroy"],
    gravity=ti.Vector([0.0, 0.0, -9.8]),
    engine="VelocityVerlet",
    search="LinkedCell",
    scheme="LSDEM",
)

dempm.set_solver({"Timestep": 1e-5, "SimulationTime": 0.401, "SaveInterval": 0.02, "SavePath": "OutputData"})

dempm.dem.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_rigid_body_number": 1,
        "levelset_grid_number": 4225,
        "surface_node_number": 20000,
        "max_plane_number": 0,
        "body_coordination_number": 1,
        "wall_coordination_number": 1,
        "verlet_distance_multiplier": [0.15, 0.1],
        "point_coordination_number": [1, 1],
        "compaction_ratio": [0.0, 0.0, 0.0, 0.0],
    }
)

dempm.mpm.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_particle_number": 200000,
        "verlet_distance_multiplier": 1.0,
        "max_constraint_number": {"max_velocity_constraint": 12322},
    }
)

dempm.memory_allocate(
    memory={"body_coordination_number": 1, "wall_coordination_number": 0, "compaction_ratio": [0.2, 0.0]}
)

dempm.dem.add_attribute(materialID=0, attribute={"Density": 1500, "ForceLocalDamping": 0.1, "TorqueLocalDamping": 0.05})

dempm.dem.add_template(
    template={
        "Name": "clump1",
        "Object": polyhedron(f"{ROOT}/assets/mesh/MPDEM/ellipsoid.stl"),
        "SurfaceResolution": 60000,
        "GridSpace": 0.0001,
        "Extent": 1,
        "WriteFile": True,
    }
)

dempm.dem.create_body(
    body={
        "GenerateType": "Create",
        "BodyType": "RigidBody",
        "Template": [
            {
                "Name": "clump1",
                "GroupID": 0,
                "MaterialID": 0,
                "BodyPoint": [0.002, 0.002, 0.06],
                "ScaleFactor": 1.0,
                "BodyOrientation": [0.0, 45.0, 0.0],
            }
        ],
    }
)

dempm.dem.choose_contact_model()

dempm.dem.select_save_data(surface=True)

dempm.mpm.add_material(
    model="Newtonian", material={"MaterialID": 1, "Density": 1000.0, "Modulus": 3.6e5, "Viscosity": 1e-3}
)

dempm.mpm.add_element(element={"ElementType": "R8N3D", "ElementSize": ti.Vector([0.0005, 0.0005, 0.0005])})

dempm.mpm.add_region(
    region=[
        {
            "Name": "region1",
            "Type": "Rectangle",
            "BoundingBoxPoint": ti.Vector([0.0, 0.0, 0.0]),
            "BoundingBoxSize": ti.Vector([0.004, 0.004, 0.07]),
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

dempm.mpm.add_boundary_condition(
    boundary=[
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [None, None, 0.0],
            "StartPoint": [0.0, 0.0, 0.0],
            "EndPoint": [0.004, 0.004, 0.0],
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [None, 0.0, None],
            "StartPoint": [0, 0.0, 0],
            "EndPoint": [0.004, 0.0, 0.08],
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [None, 0.0, None],
            "StartPoint": [0.0, 0.004, 0],
            "EndPoint": [0.004, 0.004, 0.08],
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, None, None],
            "StartPoint": [0, 0.0, 0],
            "EndPoint": [0.0, 0.004, 0.08],
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, None, None],
            "StartPoint": [0.004, 0.0, 0],
            "EndPoint": [0.004, 0.004, 0.08],
        },
    ]
)

dempm.mpm.select_save_data()

dempm.add_body(check_overlap=True)

dempm.choose_contact_model(particle_particle_contact_model="Linear Model", particle_wall_contact_model=None)

dempm.add_property(
    DEMmaterial=0,
    MPMmaterial=1,
    property={
        "NormalStiffness": 4e3,
        "TangentialStiffness": 2e3,
        "StaticFriction": 0.57,
        "DynamicFriction": 0.36,
        "NormalViscousDamping": 0.1,
        "TangentialViscousDamping": 0.1,
    },
    dType="particle-particle",
)

dempm.run(mpm_gravity_field=True)

dempm.mpm.postprocessing()

dempm.dem.postprocessing()

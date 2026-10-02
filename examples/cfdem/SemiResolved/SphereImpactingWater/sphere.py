import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(debug=False, kernel_profiler=True)

dempm = DEMPM()

dempm.set_configuration(
    domain=ti.Vector([0.1, 0.1, 0.15]),
    coupling_scheme="CFDEM",
    cfdem_resolution="SemiResolved",
    particle_interaction=True,
    wall_interaction=False,
)

dempm.mpm.set_configuration(
    background_damping=0.00,
    alphaPIC=0.000,
    mapping="USF",
    shape_function="QuadBSpline",
    gravity=ti.Vector([0.0, 0.0, -9.8]),
    material_type="Fluid",
    velocity_projection="Affine",
    solver_type="Implicit",
    discretization="FDM",
    fluid_level_set=True,
    particle_shifting=True,
    density_projection=True,
)

dempm.mpm.set_implicit_solver_parameters(linear_solver="PCG", max_iteration_number=200, residual_tolerance=1e-8)

dempm.dem.set_configuration(
    scheme="DEM",
    boundary=["Destroy", "Destroy", "Destroy"],
    gravity=ti.Vector([0.0, 0.0, -9.8]),
    engine="VelocityVerlet",
    search="LinkedCell",
)

dempm.set_solver({"Timestep": 5e-6, "SimulationTime": 0.06, "SaveInterval": 0.001, "SavePath": "ratio=10"})

dempm.dem.memory_allocate(
    memory={
        "max_material_number": 2,
        "max_particle_number": 1,
        "max_sphere_number": 1,
        "max_plane_number": 0,
        "body_coordination_number": 0,
        "wall_coordination_number": 0,
        "verlet_distance_multiplier": 0.2,
    }
)

dempm.mpm.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_particle_number": 4594637,
        "verlet_distance_multiplier": 0.4,
        "max_constraint_number": {
            "max_reflection_constraint": 197642,
            "max_friction_constraint": 0,
            "max_velocity_constraint": 0,
        },
    }
)

dempm.memory_allocate(
    memory={"body_coordination_number": 1, "wall_coordination_number": 0, "compaction_ratio": [0.02, 0.1]}
)


dempm.dem.add_attribute(materialID=0, attribute={"Density": 860, "ForceLocalDamping": 0.0, "TorqueLocalDamping": 0.0})

dempm.dem.add_attribute(materialID=1, attribute={"Density": 8500, "ForceLocalDamping": 0.0, "TorqueLocalDamping": 0.0})

dempm.dem.create_body(
    body={
        "BodyType": "Sphere",
        "Template": [
            {
                "GroupID": 0,
                "MaterialID": 0,
                "InitialVelocity": [0.0, 0.0, -2.17],
                "InitialAngularVelocity": [0.0, 0.0, 0.0],
                "BodyPoint": [0.05, 0.05, 0.1327],
                "FixVelocity": ["Free", "Free", "Free"],
                "FixAngularVelocity": ["Free", "Free", "Free"],
                "Radius": 0.0127,
                "BodyOrientation": "uniform",
            }
        ],
    }
)

dempm.dem.choose_contact_model(particle_particle_contact_model=None, particle_wall_contact_model=None)


dempm.dem.select_save_data(surface=True)

dempm.mpm.add_material(
    model="Newtonian",
    material={
        "MaterialID": 1,
        "Density": 1000.0,
        "Modulus": 3.6e5,
        "Viscosity": 1e-3,
        "ElementLength": 0.0,
        "cL": 0.1,
        "cQ": 2,
    },
)

dempm.mpm.add_element(
    element={
        "ElementType": "Staggered",
        "ElementSize": 1.0 / 6 * ti.Vector([0.0127, 0.0127, 0.0127]),
        "GhostCell": 1,
    }
)


dempm.mpm.add_region(
    region=[
        {
            "Name": "region1",
            "Type": "Rectangle",
            "BoundingBoxPoint": ti.Vector([0.0, 0.0, 0.0]),
            "BoundingBoxSize": ti.Vector([0.1, 0.1, 0.12]),
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
            "BoundaryType": "SolidCell",
            "Norm": [0.0, -1.0, 0.0],
            "StartPoint": [0.0, 0.0, 0.0],
            "EndPoint": [0.1, 0.0, 0.15],
        },
        {
            "BoundaryType": "SolidCell",
            "Norm": [0.0, 1.0, 0.0],
            "StartPoint": [0, 0.1, 0],
            "EndPoint": [0.1, 0.1, 0.15],
        },
        {
            "BoundaryType": "SolidCell",
            "Norm": [-1.0, 0.0, 0.0],
            "StartPoint": [0, 0.0, 0],
            "EndPoint": [0.0, 0.1, 0.15],
        },
        {
            "BoundaryType": "SolidCell",
            "Norm": [1.0, 0.0, 0.0],
            "StartPoint": [0.1, 0.0, 0],
            "EndPoint": [0.1, 0.1, 0.15],
        },
        {
            "BoundaryType": "SolidCell",
            "Norm": [0.0, 0.0, -1.0],
            "StartPoint": [0, 0.0, 0],
            "EndPoint": [0.1, 0.1, 0.0],
        },
    ]
)


dempm.mpm.select_save_data(grid=True)

dempm.choose_contact_model(None, None)

dempm.run(mpm_gravity_field=True, drag_model={"DragForceModel": "SchillerNaumannModel", "DragLaw": "Quadratic"})

dempm.mpm.postprocessing()

dempm.dem.postprocessing()

ti.profiler.print_kernel_profiler_info()

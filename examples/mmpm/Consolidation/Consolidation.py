import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

ARCH = os.environ.get("GEOTAICHI_SINGLE_POINT_CONSOLIDATION_ARCH", "cpu")
DT = float(os.environ.get("GEOTAICHI_SINGLE_POINT_CONSOLIDATION_DT", "1.0e-5"))
SIMULATION_TIME = float(os.environ.get("GEOTAICHI_SINGLE_POINT_CONSOLIDATION_TIME", "1.456"))
SAVE_INTERVAL = float(os.environ.get("GEOTAICHI_SINGLE_POINT_CONSOLIDATION_SAVE_INTERVAL", "0.0728"))
SAVE_PATH = os.environ.get("GEOTAICHI_SINGLE_POINT_CONSOLIDATION_OUTPUT", "1_Consolidation")
CELL_SIZE = float(os.environ.get("GEOTAICHI_SINGLE_POINT_CONSOLIDATION_CELL_SIZE", "0.05"))
PARTICLES_PER_CELL = int(os.environ.get("GEOTAICHI_SINGLE_POINT_CONSOLIDATION_PPC", "2"))

init(dim=2, arch=ARCH)

mpm = MPM()

mpm.set_configuration(
    domain=ti.Vector([0.2, 1.1]),
    background_damping=0.0,
    gravity=ti.Vector([0.0, 0.0]),
    alphaPIC=0.000,
    mapping="USF",
    shape_function="GIMP",
    # stabilize="B-Bar Method",
    material_type="TwoPhaseSingleLayer",
)

mpm.set_solver(
    solver={"Timestep": DT, "SimulationTime": SIMULATION_TIME, "SaveInterval": SAVE_INTERVAL, "SavePath": SAVE_PATH}
)

mpm.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_particle_number": 10000,
        "max_constraint_number": {
            "max_velocity_constraint": 134474,
            "max_absorbing_constraint": 134474,
            "max_particle_traction_constraint": 10000,
        },
    }
)

mpm.add_material(
    model="LinearElastic",  # fluid
    material={
        "MaterialID": 1,
        # "Density":             1500.,
        "SolidDensity": 2670.0,
        "FluidDensity": 1000.0,
        "Porosity": 0.40,
        "FluidBulkModulus": 2.2e8,
        "Permeability": 1e-3,
        "YoungModulus": 1e7,
        "PoissonRatio": 0.30,
    },
)

mpm.add_element(element={"ElementType": "Q4N2D", "ElementSize": ti.Vector([CELL_SIZE, CELL_SIZE]), "Contact": {}})

mpm.add_region(
    region=[
        {
            "Name": "region1",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": ti.Vector([0.0, 0.0]),
            "BoundingBoxSize": ti.Vector([0.2, 1.0]),
        },
        {
            "Name": "region2",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": ti.Vector([0.0, 0.975]),
            "BoundingBoxSize": ti.Vector([0.2, 0.025]),
        },
    ]
)

mpm.add_body(
    body={
        "Template": [
            {
                "RegionName": "region1",
                "nParticlesPerCell": PARTICLES_PER_CELL,
                "BodyID": 0,
                "MaterialID": 1,
                "ParticleStress": {"InternalStress": ti.Vector([-0, -0, -0, 0.0, 0.0, 0.0]), "PorePressure": 1e4},
                "Traction": [
                    {"Pressure": ti.Vector([0, -1e4]), "FluidPressure": ti.Vector([0, 0.0]), "RegionName": "region2"}
                ],
                # Traction f
                "InitialVelocity": ti.Vector([0.0, 0.0]),
                "FixVelocity": ["Free", "Free"],
            },
        ]
    }
)

mpm.add_boundary_condition(
    boundary=[
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, 0.0],
            "StartPoint": [0.0, 0.0],
            "EndPoint": [0.2, 0.0],
            "NLevel": 0,
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, None],
            "StartPoint": [0.0, 0.0],
            "EndPoint": [0.0, 1.1],
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, None],
            "StartPoint": [0.2, 0.0],
            "EndPoint": [0.2, 1.1],
        },
    ]
)


mpm.select_save_data()

mpm.run()

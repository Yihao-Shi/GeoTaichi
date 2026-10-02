import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *
from examples.mpm.Contact.CPT2D.cpt_reference import (
    ALPHA_PIC,
    BACKGROUND_DAMPING,
    DOMAIN,
    GRAVITY,
    GRID_SIZE,
    INITIAL_STRESS,
    MAPPING,
    MAX_MATERIAL_NUMBER,
    MAX_PARTICLE_NUMBER,
    MAX_PARTICLE_TRACTION_CONSTRAINT,
    MAX_VELOCITY_CONSTRAINT,
    MPM_DEM_CONTACT,
    PARTICLES_PER_CELL,
    PILE_SPEED,
    SAVE_INTERVAL as REFERENCE_SAVE_INTERVAL,
    SHAPE_FUNCTION,
    SIMULATION_TIME as REFERENCE_SIMULATION_TIME,
    SOIL_ORIGIN,
    SOIL_SIZE,
    SOIL_MATERIAL,
    STABILIZATION,
    SURFACE_ORIGIN,
    SURFACE_PRESSURE,
    SURFACE_SIZE,
    TIMESTEP as REFERENCE_TIMESTEP,
    environment_float,
)

ARCH = os.environ.get("GEOTAICHI_CPT_MPM_ARCH", "gpu")
DT = environment_float(os.environ, "GEOTAICHI_CPT_MPM_DT", REFERENCE_TIMESTEP)
SIMULATION_TIME = environment_float(os.environ, "GEOTAICHI_CPT_MPM_TIME", REFERENCE_SIMULATION_TIME)
SAVE_INTERVAL = environment_float(os.environ, "GEOTAICHI_CPT_MPM_SAVE_INTERVAL", REFERENCE_SAVE_INTERVAL)
SAVE_PATH = os.environ.get("GEOTAICHI_CPT_MPM_OUTPUT", "1_Pile2DAxisy_SDMC_dem")
PILE_PATH = os.environ.get(
    "GEOTAICHI_CPT_MPM_PILE",
    os.path.join(ROOT, "assets", "data", "MPM", "CPT2D", "pile.txt"),
)

init(dim=2, arch=ARCH, device_memory_GB=7.0)

mpm = MPM()

mpm.set_configuration(
    domain=ti.Vector(DOMAIN),
    is_2DAxisy=True,
    background_damping=BACKGROUND_DAMPING,
    gravity=ti.Vector(GRAVITY),
    alphaPIC=ALPHA_PIC,
    mapping=MAPPING,
    shape_function=SHAPE_FUNCTION,
    stabilize=STABILIZATION,
)

mpm.set_solver(
    solver={"Timestep": DT, "SimulationTime": SIMULATION_TIME, "SaveInterval": SAVE_INTERVAL, "SavePath": SAVE_PATH}
)

mpm.memory_allocate(
    memory={
        "max_material_number": MAX_MATERIAL_NUMBER,
        "max_particle_number": MAX_PARTICLE_NUMBER,
        "max_constraint_number": {
            "max_velocity_constraint": MAX_VELOCITY_CONSTRAINT,
            "max_particle_traction_constraint": MAX_PARTICLE_TRACTION_CONSTRAINT,
        },
    }
)

mpm.add_contact(
    contact_type="DEMContact",
    materialID=1,
    stiffness=list(MPM_DEM_CONTACT["stiffness"]),
    friction=MPM_DEM_CONTACT["friction"],
)


mpm.add_material(
    model="StateDependentMohrCoulomb",
    material=dict(SOIL_MATERIAL),
)

mpm.add_element(element={"ElementType": "Q4N2D", "ElementSize": ti.Vector([GRID_SIZE, GRID_SIZE])})

mpm.add_region(
    region=[
        {
            "Name": "region1",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": ti.Vector(SOIL_ORIGIN),
            "BoundingBoxSize": ti.Vector(SOIL_SIZE),
        },
        {
            "Name": "region3",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": ti.Vector(SURFACE_ORIGIN),
            "BoundingBoxSize": ti.Vector(SURFACE_SIZE),
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
                "ParticleStress": {"InternalStress": ti.Vector(INITIAL_STRESS)},
                "Traction": [{"Pressure": ti.Vector([0, -SURFACE_PRESSURE]), "RegionName": "region3"}],
                "InitialVelocity": ti.Vector([0.0, 0.0]),
                "FixVelocity": ["Free", "Free"],
            }
        ]
    }
)

mpm.add_polygons(body={"Vertices": PILE_PATH, "InitialVelocity": [0.0, -PILE_SPEED]})

mpm.add_boundary_condition(
    boundary=[
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, 0.0],
            "StartPoint": list(SOIL_ORIGIN),
            "EndPoint": [SOIL_SIZE[0], 0.0],
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, None],
            "StartPoint": list(SOIL_ORIGIN),
            "EndPoint": [0.0, DOMAIN[1]],
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, None],
            "StartPoint": [DOMAIN[0], 0.0],
            "EndPoint": list(DOMAIN),
        },
    ]
)


mpm.select_save_data(grid=True)

mpm.run(gravity_field=True)

# mpm.postprocessing(read_path='1_Pile2DAxisy_SDMC_dem', write_background_grid=True)

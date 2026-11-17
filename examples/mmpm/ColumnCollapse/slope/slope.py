import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(
    dim=2,
    arch=os.environ.get("GT_MMPM_ARCH", "gpu"),
    debug=os.environ.get("GT_MMPM_DEBUG", "1") == "1",
    device_memory_GB=float(os.environ.get("GT_MMPM_DEVICE_MEMORY_GB", "6")),
)

dt = float(os.environ.get("GT_MMPM_DT", "1e-5"))
initial_time = float(os.environ.get("GT_MMPM_INITIAL_TIME", str(dt)))
simulation_time = float(os.environ.get("GT_MMPM_SIMULATION_TIME", "1"))
save_interval = float(os.environ.get("GT_MMPM_SAVE_INTERVAL", str(dt)))
save_path = os.environ.get("GT_MMPM_SAVE_PATH", "SRF = 1.0")

mpm = MPM()

mpm.set_configuration(
    domain=ti.Vector([60.0, 20.0]),
    is_2DAxisy=False,
    background_damping=0.1,
    gravity=ti.Vector([0.0, -9.8]),
    alphaPIC=0.00,
    mapping="USL",
    shape_function="GIMP",
    material_type="TwoPhaseSingleLayer",
)

mpm.set_solver(
    solver={"Timestep": dt, "SimulationTime": initial_time, "SaveInterval": save_interval, "SavePath": save_path}
)

mpm.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_particle_number": 2e5,
        "max_constraint_number": {"max_velocity_constraint": 83000},
    }
)

mpm.add_material(
    model="MohrCoulomb",
    material={
        "MaterialID": 1,
        "SolidDensity": 2500.0,
        "FluidDensity": 1000.0,
        "Porosity": 0.40,
        "YoungModulus": 1e8,
        "FluidBulkModulus": 2.2e8,
        "Permeability": 1e-1,
        "PoissonRatio": 0.30,
        "Cohesion": 20000,
        "Friction": 30.0,
        "Dilation": 0.0,
    },
)

mpm.add_element(
    element={
        "ElementType": "Q4N2D",
        "ElementSize": ti.Vector([0.5, 0.5]),
    }
)


def region_func(new_position, new_radius=0.0):
    # assume that our target region is closed by x = 20; y = 5; and x + 2 * y - 50. <= 0
    return (
        1
        if (new_position[0] + 2 * new_position[1] - 50 <= 0) and (new_position[0] >= 20) and (new_position[1] >= 5)
        else 0
    )


def volume_func():
    # return the area of our target region
    return 100.0


mpm.add_region(
    region=[
        {
            "Name": "region1",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": ti.Vector([0, 0]),
            "BoundingBoxSize": ti.Vector([20, 15]),
        }
    ]
)

mpm.add_body(
    body={
        "Template": [
            {
                "RegionName": "region1",
                "nParticlesPerCell": 2,
                "BodyID": 0,
                "MaterialID": 1,
                "ParticleStress": {
                    "GravityField": True,
                    "InternalStress": ti.Vector([-0, -0, -0, 0.0, 0.0, 0.0]),
                    "Traction": {},
                },
                "InitialVelocity": ti.Vector([0, 0]),
                "FixVelocity": ["Free", "Free"],
            }
        ]
    }
)

mpm.add_region(
    region=[
        {
            "Name": "region2",
            "Type": "UserDefined",
            "BoundingBoxPoint": ti.Vector([20.0, 5.0]),
            "BoundingBoxSize": ti.Vector([20.0, 10.0]),
            "RegionVolume": volume_func,
            "RegionFunction": region_func,
        }
    ]
)


def get_gravity(points):
    # return the distance from the current material point to the free surface: y = 25. - 0.5 * x
    return 25.0 - 0.5 * points[:, 0] - points[:, 1]


mpm.add_body(
    body={
        "Template": [
            {
                "RegionName": "region2",
                "nParticlesPerCell": 2,
                "BodyID": 0,
                "MaterialID": 1,
                "InitialVelocity": ti.Vector([0, 0]),
                "FixVelocity": ["Free", "Free"],
            }
        ]
    }
)

mpm.add_region(
    region=[
        {
            "Name": "region3",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": ti.Vector([20, 0]),
            "BoundingBoxSize": ti.Vector([40, 5]),
        }
    ]
)

mpm.add_body(
    body={
        "Template": [
            {
                "RegionName": "region3",
                "nParticlesPerCell": 2,
                "BodyID": 0,
                "MaterialID": 1,
                "InitialVelocity": ti.Vector([0, 0]),
                "FixVelocity": ["Free", "Free"],
            }
        ]
    }
)
mpm.add_boundary_condition(
    boundary=[
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, 0.0],
            "StartPoint": [0.0, 0.0],
            "EndPoint": [60.0, 0.0],
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, None],
            "StartPoint": [0.0, 0.0],
            "EndPoint": [0.0, 15.0],
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, None],
            "StartPoint": [60.0, 0.0],
            "EndPoint": [60.0, 5],
        },
    ]
)

mpm.select_save_data()

mpm.run(gravity_field=get_gravity)

mpm.add_material(
    model="MohrCoulomb",
    material={
        "MaterialID": 1,
        "SolidDensity": 2500.0,
        "FluidDensity": 1000.0,
        "Porosity": 0.40,
        "YoungModulus": 1e8,
        "FluidBulkModulus": 2.2e8,
        "Permeability": 1e-5,
        "PoissonRatio": 0.30,
        "Cohesion": 20000,
        "Friction": 30.0,
        "Dilation": 0.0,
    },
)

mpm.modify_parameters(SimulationTime=simulation_time)

mpm.run()

if os.environ.get("GEOTAICHI_SKIP_POSTPROCESS", "0") != "1":
    mpm.postprocessing(read_path=save_path, write_background_grid=True)

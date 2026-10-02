import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(dim=2, debug=False, device_memory_GB=2)

mpm = MPM()

mpm.set_configuration(
    domain=ti.Vector([50.0, 20.0]),
    is_2DAxisy=False,
    # mode="Lightweight",
    background_damping=0.0,
    gravity=ti.Vector([0.0, -10.0]),
    alphaPIC=0.00,
    mapping="USF",
    shape_function="QuadBSpline",
)

mpm.set_solver(solver={"Timestep": 1e-4, "SimulationTime": 5, "SaveInterval": 0.2, "SavePath": "OutputData"})

mpm.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_particle_number": 5.12e5,
        "max_constraint_number": {"max_velocity_constraint": 83000, "max_friction_constraint": 83000},
    }
)

mpm.add_material(
    model="DruckerPrager",
    material={
        "MaterialID": 1,
        "Density": 1800.0,
        "YoungModulus": 100e6,
        "PoissonRatio": 0.3,
        "Cohesion": 6700.0,
        "Friction": 20.0,
        "Dilation": 0.0,
        "Tensile": 0.0,
    },
)

mpm.add_element(element={"ElementType": "Q4N2D", "ElementSize": ti.Vector([0.2, 0.2]), "Contact": {}})


def get_gravity(points):
    # return the distance from the current material point to the free surface: y = 1. - 0.5 * x
    import numpy as np

    return np.where(
        points[:, 0] < 20.0,
        20 - points[:, 1],
        np.where(points[:, 0] < 30.0, 40.0 - points[:, 0] - points[:, 1], 10 - points[:, 1]),
    )


mpm.add_body_from_file(
    body={
        "FileType": "TXT",
        "Template": [
            {
                "BodyID": 0,
                "MaterialID": 1,
                "ParticleFile": f"{ROOT}/assets/data/MPM/Particle2D.txt",
                "InitialVelocity": [0, 0, 0],
                "FixVelocity": ["Free", "Free", "Free"],
            }
        ],
    }
)

mpm.add_boundary_condition(
    boundary=[
        {"BoundaryType": "VelocityConstraint", "Velocity": [0.0, 0], "StartPoint": [0.0, 0.0], "EndPoint": [50.0, 0.0]},
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, None],
            "StartPoint": [0.0, 0.0],
            "EndPoint": [0.0, 20.0],
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, None],
            "StartPoint": [50.0, 0.0],
            "EndPoint": [50.0, 20.0],
        },
    ]
)

mpm.select_save_data(grid=False)

mpm.run(gravity_field=get_gravity)

mpm.postprocessing(read_path="DP2DPlane", write_background_grid=True)

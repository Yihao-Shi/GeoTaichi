import argparse
import os
import sys

from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description="LSMPM soft sphere rolling on a plane")
parser.add_argument("--arch", default="cpu")
parser.add_argument("--output-dir", default="OutputData/soft_sphere_rolling")
parser.add_argument("--friction", type=float, default=0.6)
parser.add_argument("--shape-function", default="QuadBSpline")
parser.add_argument("--soft-grid-type", default="Hexahedron")
parser.add_argument("--levelset-extent", type=int, default=3)
parser.add_argument("--particles-per-cell", type=int, default=2)
parser.add_argument("--grid-spacing-ratio", type=float, default=0.15)
parser.add_argument("--initial-speed", type=float, default=0.25)
parser.add_argument("--initial-spin", type=float, default=0.0)
parser.add_argument("--search", default="LinkedCell")
parser.add_argument("--dt", type=float, default=2.0e-5)
parser.add_argument("--time", type=float, default=0.02)
parser.add_argument("--save-interval", type=float, default=0.002)
parser.add_argument("--young-modulus", type=float, default=5.0e6)
parser.add_argument("--scene-manifest", help="Optional Blender SceneManifest metadata")
arguments = parser.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import MPDEM, init, polyhedron


init(arch=arguments.arch, log=True, debug=False, offline_cache=False)

mesh_path = Path(ROOT) / "assets/mesh/LSDEM/sphere.stl"
save_path = arguments.output_dir
friction = arguments.friction
shape_function = arguments.shape_function
soft_grid_type = arguments.soft_grid_type
sdf_extent = arguments.levelset_extent
material_points_per_cell = arguments.particles_per_cell
mechanical_grid_spacing_ratio = arguments.grid_spacing_ratio

radius = 0.12
floor_z = 0.12
initial_speed = arguments.initial_speed
initial_spin = arguments.initial_spin

mpdem = MPDEM(log=True)
mpdem.set_configuration(
    domain=[1.2, 0.7, 0.7],
    coupling_scheme="MPDEM",
    particle_interaction=True,
    wall_interaction=False,
    gravity=[0.0, 0.0, -9.81],
    search=arguments.search,
    visualize=True,
    log=True,
)

dem = mpdem.dem
dem.set_configuration(
    domain=[1.2, 0.7, 0.7],
    boundary=["Destroy", "Destroy", "Destroy"],
    gravity=[0.0, 0.0, -9.81],
    search=arguments.search,
    scheme="LSMPM",
    shape_function=shape_function,
    soft_grid_type=soft_grid_type,
    soft_mechanical_grid_spacing_ratio=mechanical_grid_spacing_ratio,
    visualize=True,
    log=True,
)

dem.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_rigid_body_number": 0,
        "max_soft_body_number": 1,
        "max_material_point_number": 12000,
        "max_rigid_template_number": 1,
        "levelset_grid_number": 8000,
        "soft_grid_number": 16000,
        "surface_node_number": 2000,
        "max_plane_number": 1,
        "body_coordination_number": 4,
        "wall_coordination_number": 4,
        "verlet_distance_multiplier": [0.2, 0.2],
        "point_coordination_number": [4, 4],
        "compaction_ratio": [1.0, 1.0],
        "wall_per_cell": 4,
    },
    log=True,
)

mpdem.set_solver(
    {
        "Timestep": arguments.dt,
        "SimulationTime": arguments.time,
        "SaveInterval": arguments.save_interval,
        "SavePath": save_path,
    },
    log=True,
)

dem.add_attribute(
    materialID=0,
    attribute={
        "Density": 1200.0,
        "ConstitutiveModel": "NeoHookean",
        "YoungModulus": arguments.young_modulus,
        "PoissonRatio": 0.3,
        "ForceLocalDamping": 0.02,
        "TorqueLocalDamping": 0.02,
    },
)

dem.add_template(
    template={
        "Name": "SphereTemplate",
        "Object": polyhedron(file=str(mesh_path)).grids(
            space=0.2, extent=sdf_extent
        ),
    }
)

dem.create_body(
    body={
        "BodyType": "SoftBody",
        "Template": {
            "Name": "SphereTemplate",
            "GroupID": 0,
            "MaterialID": 0,
            "BodyPoint": [0.35, 0.35, floor_z + radius - 0.004],
            "BoundingRadius": radius,
            "MaterialPointsPerCell": material_points_per_cell,
            "InitialVelocity": [initial_speed, 0.0, 0.0],
            "InitialAngularVelocity": [0.0, initial_spin, 0.0],
            "BodyOrientation": "constant",
            "FixMotion": ["Free", "Free", "Free"],
        },
    }
)

dem.add_wall(
    body={
        "WallType": "Plane",
        "WallID": 0,
        "MaterialID": 0,
        "WallCenter": np.array([0.0, 0.0, floor_z]),
        "OuterNormal": np.array([0.0, 0.0, 1.0]),
    }
)

dem.choose_contact_model(
    particle_particle_contact_model="Linear Model",
    particle_wall_contact_model="Linear Model",
)
dem.add_property(
    materialID1=0,
    materialID2=0,
    property={
        "NormalStiffness": 1.0e5,
        "TangentialStiffness": 5.0e4,
        "Friction": friction,
        "NormalViscousDamping": 0.02,
        "TangentialViscousDamping": 0.02,
    },
    dType="all",
)

dem.select_save_data(particle=True, surface=True, bounding=True, wall=True)
mpdem.run()

point_num = int(dem.scene.softPointNum[0])
soft_np = dem.scene.soft_point.to_numpy()
contact_force = soft_np["contact_force"][:point_num]
velocity = soft_np["v"][:point_num]
mass = soft_np["m"][:point_num]
mass_sum = np.sum(mass)
mean_velocity = np.sum(velocity * mass[:, None], axis=0) / mass_sum
total_contact_force = np.sum(contact_force, axis=0)
print(
    "SoftSphereRolling finished: "
    f"mu={friction}, grid={dem.sims.soft_grid_type}, "
    f"shape={dem.sims.soft_shape_function}, "
    f"mean_velocity={mean_velocity.tolist()}, "
    f"total_contact_force={total_contact_force.tolist()}, "
    "projection_uncovered="
    f"{dem.sims.soft_levelset_projection_uncovered_nodes}/"
    f"{dem.sims.soft_levelset_projection_band_nodes}, "
    "projection_max_support_loss="
    f"{dem.sims.soft_levelset_projection_max_support_loss:.3e}, "
    f"output={save_path}"
)

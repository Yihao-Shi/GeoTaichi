"""Two TET4 FEM soft particles with explicit PT/EE DEM contact."""

import argparse
import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--arch", default="gpu")
parser.add_argument("--default-fp", default="float64")
parser.add_argument("--search", default="BVH")
parser.add_argument("--dt", type=float, default=2.0e-5)
parser.add_argument("--time", type=float, default=0.05)
parser.add_argument("--output-interval", type=int, default=100)
parser.add_argument(
    "--output-dir",
    default=str(CASE_DIR / "OutputData" / "explicit_soft_particles"),
)
parser.add_argument("--scene-manifest", help="Optional Blender SceneManifest metadata")
arguments = parser.parse_args()

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np

from geotaichi import FEM, init


init(
    arch=arguments.arch,
    default_fp=arguments.default_fp,
    log=True,
    debug=False,
    offline_cache=False,
)

fem = FEM(log=True)
fem.set_configuration(dimension=3, solver_type="Explicit")
fem.add_soft_particle(
    fem.create_mesh(
        "box",
        origin=[0.20, 0.30, 0.30],
        size=[0.20, 0.20, 0.20],
        divisions=[2, 2, 2],
        element_type="TET4",
    )
)
fem.add_soft_particle(
    fem.create_mesh(
        "box",
        origin=[0.42, 0.30, 0.30],
        size=[0.20, 0.20, 0.20],
        divisions=[2, 2, 2],
        element_type="TET4",
    )
)
fem.add_material(
    "NeoHookean",
    density=1100.0,
    young_modulus=2.0e5,
    poisson_ratio=0.3,
)
fem.add_soft_particle_contact(
    "Linear",
    search=arguments.search,
    verlet_distance_multiplier=0.15,
    ContactThickness=0.025,
    NormalStiffness=2.0e6,
    TangentialStiffness=1.0e6,
    Friction=0.35,
    NormalViscousDamping=0.1,
    TangentialViscousDamping=0.1,
)

velocity = np.zeros_like(fem.scene.mesh.points)
velocity[fem.scene.mesh.node_body_ids == 0, 0] = 0.20
velocity[fem.scene.mesh.node_body_ids == 1, 0] = -0.20
fem.set_solver(
    dt=arguments.dt,
    simulation_time=arguments.time,
    output_interval=arguments.output_interval,
    path=arguments.output_dir,
    initial_velocity=velocity,
)
fem.run()

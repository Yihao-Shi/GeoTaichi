import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--arch", choices=("cpu", "gpu"), default="cpu")
parser.add_argument("--default-fp", default="float64")
parser.add_argument("--dt", type=float, default=1.0e-2)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--output-interval", type=int, default=10)
parser.add_argument("--preview", action="store_true")
parser.add_argument(
    "--output-dir",
    default=str(CASE_DIR / "OutputData" / "elastic_beam"),
)
arguments = parser.parse_args()

from geotaichi import *
from src.iga import DirichletBoundary, Primitives, Rectangle

import numpy as np


init(
    dim=2,
    arch=arguments.arch,
    default_fp=arguments.default_fp,
    debug=False,
    log=True,
)

iga = IGA(log=True)
iga.set_configuration(dimension=2, solver_type="Implicit")

# Time integration
dt = arguments.dt
gravity = [0., -10.]

# newmark integration parameters
#integration = [0.5, 0.25, 0.5]  # alpha, beta, gamma
integration = [1., 0.5, 1.]  # alpha, beta, gamma

# Initial settings for IGA
degree_u, degree_v = 2, 2
output_interval = arguments.output_interval
total_step = arguments.steps
output_path = arguments.output_dir

# Generate IGA body
body = Rectangle()
body.set_parameters(size=[10, 2])
body.generate_knot_u(degree=degree_u, num_ctrlpts=11)
body.generate_knot_v(degree=degree_v, num_ctrlpts=5)
body.generate_ctrlpts()
body.generate_weights()
body.activate_boundary()
body.gather_boundary_ctrlpts()
if arguments.preview:
    body.visualize(resolution=10)

# Material Parameters
young_modulus = 8e6
poisson_ratio = 0.3
density = 1000.

# Wrap body to primitive class
primitives = Primitives()
primitives.append(body, 'beam')
primitives.finialize()

# Dirichlet boundary condition
dirichlet_node = []
for name, meta in primitives.body.items(): 
    primitive = meta['primitive']
    index = np.where(primitive.control_points[:,0] == 0.)[0]
    dirichlet_node.append(list(2 * index))
    dirichlet_node.append(list(2 * index + 1))
dirichlet = DirichletBoundary()
total_len = 2 * len(index)
dirichlet.append(dirichlet_node,[0.0] * total_len)

iga.add_primitives(primitives)
iga.add_boundary_condition(dirichlet=dirichlet)
iga.add_element(degree=[degree_u, degree_v])
iga.add_material(young_modulus=young_modulus, poisson_ratio=poisson_ratio, density=density)
iga.set_solver(dt=dt, newmark=integration, residual=1e-4, gravity=gravity, interval=output_interval, step=total_step, path=output_path)
iga.run()

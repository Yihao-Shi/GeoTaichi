import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(
    dim=3,
    arch=os.environ.get("GT_DIRECT_ARCH", "gpu"),
    default_fp="float64",
    kernel_profiler=os.environ.get("GT_DIRECT_KERNEL_PROFILER", "1") == "1",
    debug=False,
    log=False,
)

from src.mpm.generator.Body import Body
from src.mpm.boundaries.BoundaryCondition import DirichletBoundary

dt = float(os.environ.get("GT_DIRECT_DT", "1e-3"))
gravity = [0., 0., -10.]
domain = [5., 5., 5.]
dx = float(os.environ.get("GT_DIRECT_DX", "0.05"))
total_steps = int(os.environ.get("GT_DIRECT_STEPS", "100"))
output_interval = int(os.environ.get("GT_DIRECT_INTERVAL", "10"))
output_path = os.environ.get("GT_DIRECT_OUTPUT", "OutputData/examples3d/ElasticBeam")

young_modulus = 8e6
poisson_ratio = 0.3
density = 1000.

# newmark integration parameters
#integration = [0.5, 0.25, 0.5]  # alpha, beta, gamma
integration = [1., 0.5, 1.]  # alpha, beta, gamma

# Dirichlet boundary condition
Z, Y, X = np.meshgrid(np.linspace(0., domain[2], int(domain[2]/dx)+1), np.linspace(0., domain[1], int(domain[1]/dx)+1), np.linspace(0., domain[0], int(domain[0]/dx)+1), indexing='ij') 
coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1) 
dirichlet_node = []
index = np.where(coords[:,0] <= 1.)[0]
dirichlet_node.append(list(3 * index))
dirichlet_node.append(list(3 * index + 1))
dirichlet_node.append(list(3 * index + 2))
dirichlet = DirichletBoundary()
total_len = 3 * len(index)
dirichlet.append(dirichlet_node,[0.0] * total_len)

body = Body()
body.add_cube([1., 2., 3.], [4., 2.5, 3.5], dx, 2)
mpm = MPM(log=False)
mpm.set_configuration(dimension=3, mpm_backend="Direct", solver_type="Implicit", configuration="TLMPM", domain=domain, gravity=gravity)
mpm.add_body(body)
mpm.add_boundary_condition(dirichlet=dirichlet)
mpm.add_material(young_modulus=young_modulus, poisson_ratio=poisson_ratio, density=density)
mpm.add_element({"ElementSize": dx, "ShapeFunction": "linear"})
mpm.set_solver({"dt": dt, "step": total_steps, "interval": output_interval, "newmark": integration, "residual": 1e-4, "line_search": True, "path": output_path})
mpm.run()

import os
import sys

import taichi as ti
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(dim=2, arch="gpu", default_fp="float64", kernel_profiler=True, debug=False, log=False)

from src.mpm.generator.Body import Body
from src.mpm.generator.Ground import Ground
from src.mpm.boundaries.BoundaryCondition import NeumannBoundary

dt = 1e-3
gravity = [0., -10.0]
domain = [20., 10.]
dx = 0.05

young_modulus = 1e5
poisson_ratio = 0.3
density = 1000.

# newmark integration parameters
#integration = [0.5, 0.25, 0.5]  # alpha, beta, gamma
integration = [1., 0.5, 1.]  # alpha, beta, gamma

# Ground
ground = Ground()
ground.append([2.5, 0.5], [0., 1.])

body = Body()
body.add_semi_circle([5., 8.5], 8.0, spacing=[100, 200], init_v=[0.0, 0.0])
body.add_rectangle([4., 0.], [6., 0.5], dx, 2, init_v=[0.0, 0.0])

# Neumann boundary condition
Y, X = np.meshgrid(np.linspace(0., domain[1], int(domain[1]/dx)+1), np.linspace(0., domain[0], int(domain[0]/dx)+1), indexing='ij') 
coords = np.stack([X.ravel(), Y.ravel()], axis=1) 
neumann_node = []
index = np.where(coords[:,0] <= 1.)[0]
neumann_node.append(list(2 * index))
neumann_node.append(list(2 * index + 1))
neumann = NeumannBoundary()
total_len = 2 * len(index)
neumann.append(neumann_node,[0.0] * total_len)

mpm = MPM(log=False)
mpm.set_configuration(dimension=2, mpm_backend="Direct", solver_type="Implicit", configuration="TLMPM", domain=domain, gravity=gravity, ipc=True)
mpm.add_body(body)
mpm.add_ground(ground)
mpm.add_material(young_modulus=young_modulus, poisson_ratio=poisson_ratio, density=density)
mpm.add_element({"ElementSize": dx, "ShapeFunction": "bspline"})
mpm.add_contact("IPC", kappa=1e5, dhat=0.01, mu=0.0, epsv=0.001)
mpm.set_solver({"dt": dt, "step": 100, "interval": 10, "newmark": integration, "residual": 1e-4, "line_search": True, "scale": 1.0, "path": "OutputData/examples2d/HertzContact"})
mpm.run()

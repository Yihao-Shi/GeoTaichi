import os
import sys

import taichi as ti
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(dim=3, arch="gpu", default_fp="float64", kernel_profiler=True, debug=False, log=False)

from src.mpm.generator.Body import Body
from src.mpm.generator.Ground import Ground


dt = 1e-2
gravity = [0., 0., 0.]
domain = [5., 5., 5.]
dx = 0.02
fric = 1.0

young_modulus = 1e7
poisson_ratio = 0.3
density = 1000.

# newmark integration parameters
#integration = [0.5, 0.25, 0.5]  # alpha, beta, gamma
integration = [1., 0.5, 1.]  # alpha, beta, gamma

# Ground
ground = Ground()
ground.append([2.5, 2.5, 0.1], [0., 0., 1.], [0., 0., 0.1])
ground.append([2.5, 2.5, 2.1], [0., 0., -1.], [0., 0., -0.1])

body = Body()
body.add_sphere([1.2, 2., 1.1], 1., 36995, init_v=[0., 0., 0.])
mpm = MPM(log=False)
mpm.set_configuration(dimension=3, mpm_backend="Direct", solver_type="Implicit", configuration="ULMPM", domain=domain, gravity=gravity, ipc=True)
mpm.add_body(body)
mpm.add_ground(ground)
mpm.add_material(young_modulus=young_modulus, poisson_ratio=poisson_ratio, density=density)
mpm.add_element({"ElementSize": dx, "ShapeFunction": "linear"})
mpm.add_contact("IPC", kappa=1e4, dhat=0.0001, mu=fric, epsv=0.00001, friction_residual=1e-4, friction_set=1600)
mpm.set_solver({"dt": dt, "step": 250, "interval": 1, "newmark": integration, "residual": 1e-4, "path": "OutputData/examples3d/HertzContact"})
mpm.run()

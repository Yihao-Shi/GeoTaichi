import os
import sys

import taichi as ti

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(dim=2, arch="gpu", default_fp="float64", kernel_profiler=True, debug=True, log=False)

from src.mpm.generator.Body import Body
from src.mpm.generator.Ground import Ground

dt = 1e-3
gravity = [0., -10.0]
domain = [5., 5.]
dx = 0.05

young_modulus = 1e5
poisson_ratio = 0.3
density = 1000.

# newmark integration parameters
#integration = [0.5, 0.25, 0.5]  # alpha, beta, gamma
integration = [1., 0.5, 1.]  # alpha, beta, gamma

# Ground
ground = Ground()
ground.append([2.5, 0.0], [0., 1.])
ground.append([0., 2.5], [1., 0.])
ground.append([5., 2.5], [-1., 0.])

body = Body()
body.add_circle([1.5, 1.5], 1.0, spacing=[100, 200], init_v=[0.0, 0.0])
body.add_circle([3.5, 1.5], 1.0, spacing=[100, 200], init_v=[0.0, 0.])
body.add_circle([2.5, 3.3], 1.0, spacing=[100, 200], init_v=[0.0, 0.])
#body.add_rectangle([0., 0.], [5., 0.5], dx, 4)
#body.add_rectangle([0., 0.5], [0.5, 4.5], dx, 4)
#body.add_rectangle([4.5, 0.5], [5., 4.5], dx, 4)
mpm = MPM(log=False)
mpm.set_configuration(dimension=2, mpm_backend="Direct", solver_type="Implicit", configuration="TLMPM", domain=domain, gravity=gravity, ipc=True)
mpm.add_body(body)
mpm.add_ground(ground)
mpm.add_material(young_modulus=young_modulus, poisson_ratio=poisson_ratio, density=density)
mpm.add_element({"ElementSize": dx, "ShapeFunction": "bspline"})
mpm.add_contact("IPC", kappa=1e5, dhat=0.01, mu=0.0, epsv=0.001)
mpm.set_solver({"dt": dt, "step": 100, "interval": 10, "newmark": integration, "residual": 1e-4, "line_search": True, "scale": 1.0, "path": "OutputData/examples2d/MultiBody"})
mpm.run()

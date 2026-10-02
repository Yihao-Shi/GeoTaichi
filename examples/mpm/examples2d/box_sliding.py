import os
import sys

import taichi as ti
import numpy as np
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(dim=2, arch="cpu", default_fp="float64", kernel_profiler=True, debug=False, log=False)

from src.mpm.generator.Body import Body
from src.mpm.generator.Ground import Ground

dt = 1e-3
gravity = [0., -6.929646456]
domain = [5., 5.]
dx = 0.05
fric = 1.
output_path = "OutputData/examples2d/BoxSliding1"

young_modulus = 1e12
poisson_ratio = 0.2
density = 1000.

# newmark integration parameters
#integration = [0.5, 0.25, 0.5]  # alpha, beta, gamma
integration = [1., 0.5, 1.]  # alpha, beta, gamma

# Ground
ground = Ground()
ground.append([2.5, 0.1], [0., 1.])

body = Body()
body.add_rectangle([1., 0.1-0.0115], [2., 0.6], dx, 2, init_v=[0., 0.])
mpm = MPM(log=False)
mpm.set_configuration(dimension=2, mpm_backend="Direct", solver_type="Implicit", configuration="ULMPM", domain=domain, gravity=gravity, ipc=True)
mpm.add_body(body)
mpm.add_ground(ground)
mpm.add_material(young_modulus=young_modulus, poisson_ratio=poisson_ratio, density=density)
mpm.add_element({"ElementSize": dx, "ShapeFunction": "linear"})
mpm.add_contact("IPC", kappa=1e6, dhat=0.001, mu=0.0, epsv=0.001, friction_residual=1e-8, activate_friction=(fric != 0.))
mpm.set_solver({"dt": dt, "step": 10, "interval": 20, "newmark": integration, "residual": 1e-8, "max_iters": 20, "line_search": True, "path": output_path})
mpm.run()

def write_data():
    ipcmpm = mpm.enginer
    vel = ipcmpm.mpm.calculate_mean_velocity()
    acc = ipcmpm.mpm.calculate_mean_acceleration()
    fn = ipcmpm.ipc.record_normal_force()
    ft = ipcmpm.ipc.record_tangential_force()
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, f'mu={fric}.txt'), 'a') as f:
        f.write(f"{vel[0]} {acc[0]} {fn[0][1]} {ft[0][0]}\n")

mpm.enginer.ipc.friction.friction = fric
mpm.enginer.mpm.damping = 0.
mpm.enginer.mpm.output_interval = 10
mpm.enginer.mpm.gravity = [6.929646456, -6.929646456]
mpm.enginer.mpm.total_step = 100
mpm.run(postprocessing=[write_data])

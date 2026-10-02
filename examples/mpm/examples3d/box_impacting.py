import os
import sys

import taichi as ti
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
from src.mpm.generator.Ground import Ground
from third_party.pyevtk.hl import unstructuredGridToVTK
from third_party.pyevtk.vtk import VtkQuad


dt = float(os.environ.get("GT_DIRECT_DT", "1e-3"))
gravity = [0., 0., -9.8]
domain = [5., 5., 5.]
dx = float(os.environ.get("GT_DIRECT_DX", "0.05"))
total_steps = int(os.environ.get("GT_DIRECT_STEPS", "10"))
output_interval = int(os.environ.get("GT_DIRECT_INTERVAL", "10"))
output_path = os.environ.get("GT_DIRECT_OUTPUT", "OutputData/examples3d/SphereImpacing")
sphere_points = int(os.environ.get("GT_DIRECT_SPHERE_POINTS", "36995"))

young_modulus = 1e5
poisson_ratio = 0.3
density = 1000.

# newmark integration parameters
#integration = [0.5, 0.25, 0.5]  # alpha, beta, gamma
integration = [1., 0.5, 1.]  # alpha, beta, gamma

# Ground
ground = Ground()
ground.append([2.5, 2.5, 0.1], [0., 0., 1.])

body = Body()
body.add_sphere([1.2, 2., 1.1], 1., sphere_points, init_v=[0., 0., -2.])
mpm = MPM(log=False)
mpm.set_configuration(dimension=3, mpm_backend="Direct", solver_type="Implicit", configuration="ULMPM", domain=domain, gravity=gravity, ipc=True)
mpm.add_body(body)
mpm.add_ground(ground)
mpm.add_material(young_modulus=young_modulus, poisson_ratio=poisson_ratio, density=density)
mpm.add_element({"ElementSize": dx, "ShapeFunction": "linear"})
mpm.add_contact("IPC", kappa=1e5, dhat=0.001, mu=0.0, epsv=0.001)
mpm.set_solver({"dt": dt, "step": total_steps, "interval": output_interval, "newmark": integration, "residual": 1e-4, "path": output_path})

# The IPC ground is solver state rather than a particle body.  Emit matching
# static geometry so a gallery animation remains understandable without
# reading the source code.
os.makedirs(output_path, exist_ok=True)
ground_x = np.ascontiguousarray([0.0, domain[0], domain[0], 0.0])
ground_y = np.ascontiguousarray([0.0, 0.0, domain[1], domain[1]])
ground_z = np.ascontiguousarray([0.1, 0.1, 0.1, 0.1])
unstructuredGridToVTK(
    os.path.join(output_path, "GalleryGround000000"),
    ground_x,
    ground_y,
    ground_z,
    connectivity=np.ascontiguousarray([0, 1, 2, 3], dtype=np.int64),
    offsets=np.ascontiguousarray([4], dtype=np.int64),
    cell_types=np.ascontiguousarray([VtkQuad.tid], dtype=np.uint8),
)
mpm.run()

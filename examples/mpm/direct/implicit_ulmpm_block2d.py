import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

import numpy as np


init(dim=2, arch="cpu", default_fp="float64", offline_cache=False, log=False)

mpm = MPM(log=False)
mpm.set_configuration(
    dimension=2,
    mpm_backend="Direct",
    solver_type="Implicit",
    configuration="ULMPM",
    domain=[1.0, 1.0],
    gravity=[0.0, -9.8],
)

body = mpm.create_body()
body.add_particles(
    np.array(
        [
            [0.25, 0.25],
            [0.35, 0.25],
            [0.25, 0.35],
            [0.35, 0.35],
        ],
        dtype=np.float64,
    ),
    volume=0.01,
    init_v=[0.0, 0.0],
    name="block",
)

mpm.add_material(
    model="LinearElastic",
    material={
        "YoungModulus": 1.0e5,
        "PoissonRatio": 0.3,
        "Density": 1000.0,
    },
)
mpm.add_element(element={"ElementSize": [0.1, 0.1], "ShapeFunction": "Linear"})
mpm.add_body(body)
mpm.set_implicit_solver_parameters(residual_tolerance=1.0e-4)
mpm.set_solver(
    {
        "Timestep": 1.0e-3,
        "SimulationTime": 1.0e-3,
        "SaveInterval": 1.0e-3,
        "SavePath": "OutputData/DirectMPMBlock2D",
    },
)

mpm.run(verbose=False)

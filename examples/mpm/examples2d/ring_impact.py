import os
import sys
import json
from pathlib import Path

import taichi as ti
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(
    dim=2,
    arch=os.environ.get("GT_DIRECT_ARCH", "gpu"),
    default_fp="float64",
    kernel_profiler=os.environ.get("GT_DIRECT_KERNEL_PROFILER", "1") == "1",
    debug=False,
    log=False,
)

from src.mpm.generator.Body import Body
from src.mpm.generator.Ground import Ground

dt = float(os.environ.get("GT_DIRECT_DT", "1e-3"))
gravity = [0.0, -0.0]
domain = [5.0, 5.0]
dx = float(os.environ.get("GT_DIRECT_DX", "0.05"))
total_steps = int(os.environ.get("GT_DIRECT_STEPS", "100"))
output_interval = int(os.environ.get("GT_DIRECT_INTERVAL", "10"))
output_path = os.environ.get("GT_DIRECT_OUTPUT", "OutputData/examples2d/RingImpacing")
radial_points = int(os.environ.get("GT_DIRECT_RING_RADIAL_POINTS", "20"))
circumferential_points = int(os.environ.get("GT_DIRECT_RING_CIRCUMFERENTIAL_POINTS", "520"))
contact_model = os.environ.get("GT_DIRECT_CONTACT_MODEL", "BarrierIPC")
if contact_model not in ("BarrierIPC", "SemiIPC"):
    raise ValueError("GT_DIRECT_CONTACT_MODEL must be BarrierIPC or SemiIPC")

young_modulus = 1e5
poisson_ratio = 0.3
density = 1000.0

# newmark integration parameters
integration = [0.5, 0.25, 0.5]  # alpha, beta, gamma
# integration = [1., 0.5, 1.]  # alpha, beta, gamma

# Ground
ground = Ground()
ground.append([2.5, 0.49], [0.0, 1.0])

body = Body()
body.add_ring([1.5, 2.5], 0.8, 1.0, spacing=[radial_points, circumferential_points], init_v=[1.0, -0.0])
body.add_ring([3.5, 2.5], 0.8, 1.0, spacing=[radial_points, circumferential_points], init_v=[-1.0, 0.0])
mpm = MPM(log=False)
mpm.set_configuration(
    dimension=2,
    mpm_backend="Direct",
    solver_type="Implicit",
    configuration="TLMPM",
    domain=domain,
    gravity=gravity,
    ipc=True,
)
mpm.add_body(body)
mpm.add_ground(ground)
mpm.add_material(young_modulus=young_modulus, poisson_ratio=poisson_ratio, density=density)
mpm.add_element({"ElementSize": dx, "ShapeFunction": "bspline"})
mpm.add_contact(contact_model, kappa=1e5, dhat=0.01, mu=0.0, epsv=0.001)
mpm.set_solver(
    {
        "dt": dt,
        "step": total_steps,
        "interval": output_interval,
        "newmark": integration,
        "residual": 1e-4,
        "damping": 0.0,
        "line_search": True,
        "scale": 1.0,
        "path": output_path,
    }
)
peak_particle_contacts = [0]


def sample_contacts():
    peak_particle_contacts[0] = max(peak_particle_contacts[0], int(mpm.enginer.ipc.pbarrierNum[0]))


mpm.run(postprocessing=[sample_contacts])
positions = mpm.enginer.mpm.particle.x.to_numpy()[: int(mpm.enginer.mpm.particleNum[0])]
summary = {
    "case": "direct_mpm_ring_self_contact",
    "particles": int(positions.shape[0]),
    "steps": int(mpm.enginer.mpm.step_count),
    "output_frames": int(total_steps),
    "completed_time": float(mpm.enginer.mpm.time),
    "finite": bool(np.isfinite(positions).all()),
    "contact_model": str(mpm.enginer.ipc.barrier.model),
    "maximum_particle_contacts": peak_particle_contacts[0],
}
Path(output_path).mkdir(parents=True, exist_ok=True)
(Path(output_path) / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if (
    not summary["finite"]
    or summary["steps"] != total_steps * output_interval
    or not summary["maximum_particle_contacts"]
):
    raise RuntimeError(summary)

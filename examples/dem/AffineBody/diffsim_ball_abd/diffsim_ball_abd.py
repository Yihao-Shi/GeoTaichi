"""Newton ``diffsim_ball`` reproduced with the device ABD adjoint.

The trainable control is the ball's initial velocity.  A fixed-step
BarrierIPC trajectory is replayed backwards on device and the terminal
distance-to-target loss supplies the seed for ``DifferentiableABD.backward``.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from geotaichi import DEM, init, polyhedron

DT = float(os.environ.get("GT_DIFFSIM_DT", "2.0833333333333334e-3"))
STEPS = int(os.environ.get("GT_DIFFSIM_STEPS", "96"))
TRAIN_ITERS = int(os.environ.get("GT_DIFFSIM_TRAIN_ITERS", "3"))
LEARNING_RATE = float(os.environ.get("GT_DIFFSIM_LEARNING_RATE", "2.0e-2"))
TARGET = np.asarray([0.0, 1.5, 1.5], dtype=np.float64)
SAVE_PATH = Path(os.environ.get("GT_DIFFSIM_SAVE_PATH", "diffsim_ball_abd"))
MESH = ROOT / "assets/mesh/AffineBody/lowpoly_sphere.obj"

init(
    arch=os.environ.get("GT_ARCH", "gpu"),
    default_fp=os.environ.get("GT_DEFAULT_FP", "float64"),
    log=False,
    debug=False,
    offline_cache=bool(int(os.environ.get("GT_OFFLINE_CACHE", "1"))),
)

dem = DEM(log=False)
dem.set_configuration(
    domain=[4.0, 4.0, 3.0],
    scheme="AffineBody",
    search="LinkedCell",
    gravity=[0.0, 0.0, -9.81],
    visualize=False,
    log=False,
)
dem.set_affine_body_parameters(
    assemble_type=os.environ.get("GT_AFFINE_ASSEMBLE_TYPE", "HashTriplet"),
    young_modulus=5.0e5,
    dhat=0.04,
    barrier_stiffness=5.0e5,
    friction_iterations=-1,
    friction_max_iterations=8,
    friction_tolerance=1.0e-7,
    newton_tolerance=1.0e-7,
    linear_tolerance=1.0e-10,
    max_newton_iteration=16,
    line_search_max_iteration=20,
    max_step=0.03,
    ccd_type="accd",
)
dem.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_affine_body_number": 1,
        "surface_node_number": 12,
        "max_plane_number": 2,
        "body_coordination_number": 4,
        "wall_coordination_number": 2,
        "affine_contact_block_capacity": 256,
        "compaction_ratio": [1.0, 1.0],
    },
    log=False,
)
dem.set_solver(
    {
        "Timestep": DT,
        "SimulationTime": DT * STEPS,
        "SaveInterval": DT * STEPS,
        "SavePath": str(SAVE_PATH),
    },
    log=False,
)
dem.add_attribute(
    materialID=0,
    attribute={
        "Density": 1200.0,
        "ForceLocalDamping": 0.0,
        "TorqueLocalDamping": 0.0,
    },
)
dem.add_template(
    template={
        "Name": "ball",
        "TemplateType": "AffineBody",
        "Object": polyhedron(file=str(MESH)),
    }
)
dem.create_body(
    body={
        "BodyType": "AffineBody",
        "Template": {
            "Name": "ball",
            "GroupID": 0,
            "MaterialID": 0,
            "BodyPoint": [0.0, -0.5, 1.0],
            "BoundingRadius": 0.18,
            "InitialVelocity": [0.0, 5.0, -5.0],
            "Friction": 0.0,
        },
    }
)
if os.environ.get("GT_DIFFSIM_NO_WALLS", "0") != "1":
    dem.add_wall(
        {
            "WallType": "Plane",
            "MaterialID": 0,
            "WallCenter": np.asarray([0.0, 0.0, 0.0]),
            "OuterNormal": np.asarray([0.0, 0.0, 1.0]),
        }
    )
    dem.add_wall(
        {
            "WallType": "Plane",
            "MaterialID": 0,
            "WallCenter": np.asarray([0.0, 2.0, 0.0]),
            "OuterNormal": np.asarray([0.0, -1.0, 0.0]),
        }
    )
dem.add_property(
    materialID1=0,
    materialID2=0,
    property={"Dhat": 0.04, "BarrierStiffness": 5.0e5, "Friction": 0.0},
    dType="all",
)

dem.add_essentials()
engine = dem.enginer
engine.initialize(dem.sims, dem.scene)
operator = engine.operator
initial_y = operator.y.to_numpy()[: operator.control_num].copy()
initial_time = float(dem.sims.current_time)
initial_step = int(dem.sims.current_step)


def rollout(initial_velocity):
    """Return loss and d(loss)/d(initial_velocity) for one fixed tape."""

    initial_velocity = np.ascontiguousarray(initial_velocity, dtype=np.float64)
    operator.y.from_numpy(np.ascontiguousarray(initial_y))
    operator.velocity_y.from_numpy(initial_velocity)
    dem.sims.current_time = initial_time
    dem.sims.current_step = initial_step
    trajectory = dem.differentiable_affine(steps=STEPS)
    for _ in range(STEPS):
        trajectory.step()
    terminal_y = operator.y.to_numpy()[: operator.control_num]
    center = np.mean(terminal_y[:4], axis=0)
    delta = center - TARGET
    seed_y = np.repeat((2.0 * delta / 4.0)[None, :], 4, axis=0)
    seed_v = np.zeros_like(initial_velocity)
    vjp = trajectory.backward(seed_y, seed_v)["initial_velocity"]
    vjp = np.asarray(vjp, dtype=np.float64).reshape(initial_velocity.shape)
    return float(delta.dot(delta)), vjp


velocity = operator.velocity_y.to_numpy()[: operator.control_num].copy()
history = []
for iteration in range(TRAIN_ITERS):
    loss, gradient = rollout(velocity)
    velocity -= LEARNING_RATE * gradient
    history.append({"iteration": iteration, "loss": loss})
    print(
        f"iter={iteration:03d} loss={loss:.8e} "
        f"grad_norm={np.linalg.norm(gradient):.8e} "
        f"velocity={velocity[0].tolist()}"
    )

SAVE_PATH.mkdir(parents=True, exist_ok=True)
with (SAVE_PATH / "optimization.json").open("w", encoding="utf-8") as stream:
    json.dump(
        {
            "steps": STEPS,
            "dt": DT,
            "target": TARGET.tolist(),
            "initial_velocity": [0.0, 5.0, -5.0],
            "final_velocity": velocity[0].tolist(),
            "history": history,
        },
        stream,
        indent=2,
    )

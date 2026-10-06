"""GPU validation: a planar affine multi-link robot arm.

The scene uses GeoTaichi's existing affine revolute-joint path: every joint
keeps its world anchor/axis coincident, drives a motor target, enforces an
angle limit, and disables collision only for the connected pair.  VTU/NPZ
frames are written below ``GT_ROBOT_SAVE_PATH`` (default ``robot_multilink``).
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from geotaichi import DEM, init, polyhedron

LINK_COUNT = int(os.environ.get("GT_ROBOT_LINK_COUNT", "7"))
if LINK_COUNT < 2:
    raise ValueError("GT_ROBOT_LINK_COUNT must be at least 2")
LINK_LENGTH = 0.9
LINK_Z = 0.14
BASE_X = 0.55
BASE_Y = 0.9
DT = float(os.environ.get("GT_ROBOT_DT", "1.0e-3"))
STEPS = int(os.environ.get("GT_ROBOT_STEPS", "400"))
SAVE_INTERVAL = float(os.environ.get("GT_ROBOT_SAVE_INTERVAL", "0.02"))
SAVE_PATH = os.environ.get("GT_ROBOT_SAVE_PATH", "robot_multilink")
DIFFERENTIABLE = bool(int(os.environ.get("GT_ROBOT_DIFFERENTIABLE", "0")))
FRICTION_MAX_ITERATIONS = int(os.environ.get("GT_ROBOT_FRICTION_MAX_ITERATIONS", "20"))
MESH = ROOT / "assets/mesh/AffineBody/robot_link.obj"

init(
    arch=os.environ.get("GT_ARCH", "gpu"),
    default_fp=os.environ.get("GT_DEFAULT_FP", "float64"),
    log=False,
    debug=False,
    offline_cache=bool(int(os.environ.get("GT_OFFLINE_CACHE", "1"))),
)

dem = DEM(log=False)
dem.set_configuration(
    domain=[LINK_COUNT * LINK_LENGTH + 0.8, 1.8, 1.0],
    scheme="AffineBody",
    search="LinkedCell",
    gravity=[0.0, 0.0, -9.81],
    visualize=False,
    log=False,
)
dem.set_affine_body_parameters(
    assemble_type=os.environ.get("GT_AFFINE_ASSEMBLE_TYPE", "HashTriplet"),
    young_modulus=5.0e5,
    dhat=0.035,
    barrier_stiffness=8.0e5,
    friction_epsv=1.0e-4,
    friction_iterations=-1,
    friction_max_iterations=FRICTION_MAX_ITERATIONS,
    friction_tolerance=1.0e-7,
    newton_tolerance=1.0e-6,
    max_newton_iteration=20,
    line_search_max_iteration=30,
    max_step=0.02,
    ccd_type="accd",
)
dem.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_affine_body_number": LINK_COUNT,
        "surface_node_number": 8,
        "body_coordination_number": 16,
        "wall_coordination_number": 4,
        "max_plane_number": 1,
        "max_point_triangle_pairs": 8 * 16 * LINK_COUNT,
        "max_edge_edge_pairs": 24 * 16 * LINK_COUNT,
        "affine_contact_block_capacity": 512,
        "compaction_ratio": [1.0, 1.0],
    },
    log=False,
)
dem.set_solver(
    {
        "Timestep": DT,
        "SimulationTime": DT * STEPS,
        "SaveInterval": SAVE_INTERVAL,
        "SavePath": SAVE_PATH,
    },
    log=False,
)
dem.add_attribute(
    materialID=0,
    attribute={"Density": 1200.0, "ForceLocalDamping": 0.02, "TorqueLocalDamping": 0.02},
)
dem.add_template(template={"Name": "robot_link", "TemplateType": "AffineBody", "Object": polyhedron(file=str(MESH))})

for link_id in range(LINK_COUNT):
    dem.create_body(
        body={
            "BodyType": "AffineBody",
            "Template": {
                "Name": "robot_link",
                "GroupID": 0,
                "MaterialID": 0,
                "BodyPoint": [BASE_X + (link_id + 0.5) * LINK_LENGTH, BASE_Y, LINK_Z],
                "InitialVelocity": [0.0, 0.0, 0.0],
                "FixMotion": ["Fix", "Fix", "Fix"] if link_id == 0 else ["Free", "Free", "Free"],
                "Friction": 0.25,
            },
        }
    )

# The first link is the fixed robot base; the joints connect adjacent moving
# links. Angles are degrees at the public API. CollideConnected is
# intentionally omitted (False).
for joint_id in range(LINK_COUNT - 1):
    target = 14.0 * np.sin(0.8 * joint_id)
    dem.add_joint(
        {
            "JointType": "Revolute",
            "BodyID1": joint_id + 1,
            "BodyID2": joint_id,
            "WorldAnchor": [BASE_X + (joint_id + 1) * LINK_LENGTH, BASE_Y, LINK_Z],
            "WorldAxis": [0.0, 0.0, 1.0],
            "PositionStiffness": 2.0e5,
            "AxisStiffness": 2.0e5,
            "MotorStiffness": 2.0e3,
            "TargetAngle": float(target),
            "AngleLimit": [-55.0, 55.0],
            "LimitStiffness": 5.0e4,
            "Damping": 4.0,
        }
    )

dem.add_wall(
    {
        "WallType": "Plane",
        "MaterialID": 0,
        "WallCenter": np.array([0.0, 0.0, 0.0]),
        "OuterNormal": np.array([0.0, 0.0, 1.0]),
    }
)
dem.add_property(
    materialID1=0,
    materialID2=0,
    property={"Dhat": 0.035, "BarrierStiffness": 8.0e5, "Friction": 0.25},
    dType="all",
)
if not DIFFERENTIABLE:
    dem.run()
else:
    dem.add_essentials()
    dem.enginer.initialize(dem.sims, dem.scene)
    operator = dem.enginer.operator
    trajectory = dem.differentiable_affine(steps=STEPS)
    started = time.perf_counter()
    for _ in range(STEPS):
        trajectory.step()
    forward_seconds = time.perf_counter() - started

    terminal = operator.y.to_numpy()[: operator.control_num]
    end_controls = slice(4 * (LINK_COUNT - 1), 4 * LINK_COUNT)
    end_center = terminal[end_controls].mean(axis=0)
    target = np.asarray(
        [BASE_X + (LINK_COUNT - 0.5) * LINK_LENGTH, BASE_Y + 0.2, LINK_Z + 0.2],
        dtype=np.float64,
    )
    delta = end_center - target
    terminal_seed = np.zeros_like(terminal)
    terminal_seed[end_controls] = 2.0 * delta / 4.0
    started = time.perf_counter()
    gradient = trajectory.backward(terminal_seed)
    backward_seconds = time.perf_counter() - started

    summary = {
        "case": "robot_multilink_affine_adjoint",
        "links": LINK_COUNT,
        "joints": LINK_COUNT - 1,
        "steps": STEPS,
        "dt": DT,
        "friction_max_iterations": FRICTION_MAX_ITERATIONS,
        "loss": float(delta @ delta),
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "finite": bool(all(np.isfinite(value).all() for value in gradient.values())),
        "initial_position_vjp_norm": float(np.linalg.norm(gradient["initial_position"])),
        "initial_velocity_vjp_norm": float(np.linalg.norm(gradient["initial_velocity"])),
        "joint_target_vjp": gradient["joint_target_angle_degrees"].tolist(),
        "joint_damping_vjp": gradient["joint_damping"].tolist(),
    }
    output = Path(SAVE_PATH)
    output.mkdir(parents=True, exist_ok=True)
    (output / "differentiable_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["finite"] or not summary["initial_velocity_vjp_norm"]:
        raise RuntimeError(summary)

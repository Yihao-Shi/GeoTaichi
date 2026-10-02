import argparse
import os
import sys

from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description="AffineBody cube sliding on an IPC incline")
parser.add_argument("--arch", default="cpu")
parser.add_argument("--friction", type=float, default=1.0)
parser.add_argument("--ccd-type", default="accd")
parser.add_argument("--incline-degrees", type=float, default=30.0)
parser.add_argument("--density", type=float, default=1000.0)
parser.add_argument("--dhat", type=float, default=0.03)
parser.add_argument("--barrier-stiffness", type=float, default=8.0e5)
parser.add_argument("--initial-gap", type=float)
parser.add_argument("--time", type=float, default=0.08)
parser.add_argument("--dt", type=float, default=1.0e-3)
parser.add_argument("--local-damping", type=float, default=0.0)
parser.add_argument("--contact-damping", type=float, default=0.0)
parser.add_argument("--search", default="LinkedCell")
parser.add_argument("--assemble-type", default="MatrixFree")
parser.add_argument("--friction-iterations", type=int, default=-1)
parser.add_argument("--friction-epsv", type=float, default=1.0e-4)
parser.add_argument("--friction-tolerance", type=float, default=1.0e-7)
parser.add_argument("--newton-tolerance", type=float, default=1.0e-4)
parser.add_argument("--line-search-max-iterations", type=int, default=50)
parser.add_argument("--output-dir")
parser.add_argument("--scene-manifest", help="Optional Blender SceneManifest metadata")
arguments = parser.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import DEM, init, polyhedron, python_callback
from src.physics_model.contact_model.ipc.IPC import (
    ipc_barrier_distance_terms_py,
)


mu = arguments.friction
ccd_type = arguments.ccd_type
init(arch=arguments.arch, log=False, debug=False, offline_cache=False)

incline_degrees = arguments.incline_degrees
theta = np.deg2rad(incline_degrees)
normal = np.array([-np.sin(theta), 0.0, np.cos(theta)], dtype=np.float64)
upslope = np.array([np.cos(theta), 0.0, np.sin(theta)], dtype=np.float64)
downslope = -upslope
scale = 0.25
support = 0.5 * scale
density = arguments.density
gravity = 9.81
dhat = arguments.dhat
barrier_stiffness = arguments.barrier_stiffness


def equilibrium_barrier_gap(normal_load, collision_weight):
    def weighted_normal_force(gap):
        _, gradient, _ = ipc_barrier_distance_terms_py(
            gap,
            dhat,
            kappa=barrier_stiffness,
        )
        return collision_weight * max(-gradient, 0.0)

    lower = max(1.0e-12 * dhat, np.finfo(float).tiny)
    upper = dhat * (1.0 - 1.0e-12)
    if not (weighted_normal_force(lower) > normal_load and weighted_normal_force(upper) < normal_load):
        raise RuntimeError("failed to bracket the IPC barrier equilibrium gap")
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if weighted_normal_force(midpoint) > normal_load:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


mass = density * scale**3
collision_weight = 3.0 * scale**2
balanced_gap = equilibrium_barrier_gap(
    mass * gravity * np.cos(theta),
    collision_weight,
)
initial_gap = balanced_gap if arguments.initial_gap is None else arguments.initial_gap
initial_center = 0.55 * upslope + np.array([0.0, 0.5, 0.0]) + normal * (support + initial_gap)
total_time = arguments.time
dt = arguments.dt
mesh_path = Path(ROOT) / "assets/mesh/AffineBody/cube.obj"
local_damping = arguments.local_damping
save_path = arguments.output_dir or f"AffineCubeIncline_mu_{mu:g}_{ccd_type}"

dem = DEM(log=False)
dem.set_configuration(
    domain=[2.0, 1.0, 2.0],
    scheme="AffineBody",
    search=arguments.search,
    gravity=[0.0, 0.0, -gravity],
    visualize=False,
    log=False,
)
dem.set_affine_body_parameters(
    assemble_type=arguments.assemble_type,
    young_modulus=5.0e5,
    # An extremely small epsv makes the regularized stick/slip transition
    # too stiff for this initially stress-free elastic transient.
    friction_epsv=arguments.friction_epsv,
    friction_iterations=arguments.friction_iterations,
    friction_max_iterations=20,
    friction_tolerance=arguments.friction_tolerance,
    # Reference IPC measures nonlinear convergence with the Newton correction
    # velocity max(abs(dx))/dt, in m/s.
    newton_tolerance=arguments.newton_tolerance,
    max_newton_iteration=20,
    line_search_max_iteration=arguments.line_search_max_iterations,
    max_step=0.02,
    ccd_type=ccd_type,
    ccd_eta=0.2,
    accd_tolerance=1.0e-6,
)
dem.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_affine_body_number": 1,
        "surface_node_number": 16,
        "max_plane_number": 1,
        "body_coordination_number": 4,
        "wall_coordination_number": 4,
        "compaction_ratio": [1.0, 1.0],
    },
    log=False,
)
dem.set_solver(
    {
        "Timestep": dt,
        "SimulationTime": total_time,
        "SaveInterval": total_time,
        "SavePath": save_path,
    },
    log=False,
)
dem.add_attribute(
    materialID=0,
    attribute={
        "Density": density,
        "ForceLocalDamping": local_damping,
        "TorqueLocalDamping": local_damping,
    },
)
dem.add_template(template={"Name": "cube", "TemplateType": "AffineBody", "Object": polyhedron(file=str(mesh_path))})
dem.create_body(
    body={
        "BodyType": "AffineBody",
        "Template": {
            "Name": "cube",
            "GroupID": 0,
            "MaterialID": 0,
            "BodyPoint": initial_center.tolist(),
            "ScaleFactor": scale,
            "BodyOrientation": [0.0, -incline_degrees, 0.0],
            "InitialVelocity": [0.0, 0.0, 0.0],
            "Friction": mu,
        },
    }
)
dem.add_wall(
    {
        "WallType": "Plane",
        "MaterialID": 0,
        "WallCenter": np.array([0.0, 0.0, 0.0]),
        "OuterNormal": normal,
    }
)
dem.add_property(
    materialID1=0,
    materialID2=0,
    property={
        "Dhat": dhat,
        "BarrierStiffness": barrier_stiffness,
        "ContactDampingStiffness": arguments.contact_damping,
        "Friction": mu,
    },
    dType="all",
)


def wall_friction_diagnostics():
    operator = dem.enginer.operator
    count = min(
        int(operator.friction_contact_count[0]),
        operator.friction_contact_capacity,
    )
    bodies = operator.friction_contact_bodies.to_numpy()[:count]
    contact_vertices = operator.friction_contact_vertices.to_numpy()[:count]
    weights = operator.friction_contact_weights.to_numpy()[:count]
    normals = operator.friction_contact_normal.to_numpy()[:count]
    hat_rel = operator.friction_contact_hat_rel.to_numpy()[:count]
    coefficients = operator.friction_contact_coeff.to_numpy()[:count]
    positions = operator.x.to_numpy()[: operator.vertex_num]
    friction_force = np.zeros(3, dtype=np.float64)
    maximum_increment = 0.0
    coulomb_limit = 0.0
    for contact in range(count):
        if not (
            np.all(bodies[contact] == bodies[contact, 0])
            and np.all(contact_vertices[contact] == contact_vertices[contact, 0])
        ):
            continue
        unit_normal = normals[contact] / np.linalg.norm(normals[contact])
        rel = np.sum(
            weights[contact, :, None] * positions[contact_vertices[contact]],
            axis=0,
        )
        increment = rel - hat_rel[contact]
        tangential_increment = increment - np.dot(increment, unit_normal) * unit_normal
        speed = np.linalg.norm(tangential_increment) / dt
        friction_ratio = 1.0
        if speed < operator.epsv:
            friction_ratio = speed * (2.0 * operator.epsv - speed) / operator.epsv**2
        if speed > 0.0:
            friction_force -= (
                coefficients[contact]
                / operator.scale
                * friction_ratio
                * tangential_increment
                / np.linalg.norm(tangential_increment)
            )
        coulomb_limit += coefficients[contact] / operator.scale
        maximum_increment = max(
            maximum_increment,
            float(np.linalg.norm(tangential_increment)),
        )
    return float(np.dot(friction_force, upslope)), coulomb_limit, maximum_increment


@python_callback
def report_convergence():
    engine = dem.enginer
    force, limit, increment = wall_friction_diagnostics()
    print(
        f"step={dem.sims.current_step + 1:04d} "
        f"newton_ok={engine.last_inner_converged} "
        f"newton_it={engine.last_newton_iterations} "
        f"newton_res={engine.last_inner_residual:.3e} "
        f"friction_ok={engine.last_friction_converged} "
        f"friction_it={engine.last_friction_iterations} "
        f"friction_res={engine.last_friction_residual:.3e} "
        f"linear_ok={engine.last_linear_converged} "
        f"linear_it={engine.last_linear_iterations} "
        f"linear_res={engine.last_linear_residual:.3e} "
        f"ls_ok={engine.step_line_search_converged} "
        f"ls_calls={engine.step_line_search_calls} "
        f"ls_alpha_min={engine.step_line_search_min_alpha:.3e} "
        f"ls_backtracks_max={engine.step_line_search_max_backtracks} "
        f"Ft={force:.6e} N "
        f"muN={limit:.6e} N "
        f"du_t={increment:.3e} m"
    )


dem.run(function=report_convergence)

vertices, _, _, _ = dem.enginer.state.surface_mesh()
final_center = np.mean(vertices, axis=0)
center_increment = final_center - initial_center
numeric = float(np.dot(center_increment, downslope))
normal_displacement = float(np.dot(center_increment, normal))
compensated = numeric + mu * normal_displacement
acceleration = max(
    0.0,
    gravity * (np.sin(theta) - mu * np.cos(theta)),
)
step_count = int(np.ceil(total_time / dt - 1.0e-12))
analytic = acceleration * dt * dt * step_count * (step_count + 1) / 2.0
regime = "sliding" if acceleration > 0.0 else "static"
if acceleration > 0.0:
    abs_error = abs(compensated - analytic)
    rel_error = abs_error / max(abs(analytic), 1.0e-12)
    print(
        f"mu={mu:g}, ccd_type={ccd_type}, regime={regime}, "
        f"initial_gap={initial_gap:.8e}, "
        f"numeric_downslope={numeric:.8e}, "
        f"normal_displacement={normal_displacement:.8e}, "
        f"compensated_displacement={compensated:.8e}, "
        f"backward_euler_oracle={analytic:.8e}, "
        f"abs_error={abs_error:.8e}, rel_error={rel_error:.8e}, "
        f"output={save_path}"
    )
else:
    operator = dem.enginer.operator
    resolved_tangential_force, coulomb_limit, last_step_tangential_displacement = wall_friction_diagnostics()
    initial_vertices = np.asarray(operator.rest_x_np, dtype=np.float64)
    initial_gaps = initial_vertices @ normal
    initial_face = np.isclose(
        initial_gaps,
        np.min(initial_gaps),
        rtol=0.0,
        atol=1.0e-12,
    )
    interface_displacement = vertices[initial_face] - initial_vertices[initial_face]
    interface_tangential_displacement = interface_displacement - np.outer(
        interface_displacement @ normal,
        normal,
    )
    interface_partial_slip = float(np.max(np.linalg.norm(interface_tangential_displacement, axis=1)))
    interface_partial_slip_ratio = interface_partial_slip / scale
    required_static_force = mass * gravity * np.sin(theta)
    force_balance_error = resolved_tangential_force - required_static_force
    mass_matrix = operator.mass.to_numpy()[: operator.body_num]
    velocity = operator.velocity_y.to_numpy()[: operator.control_num].reshape((-1, 4, 3))
    previous_velocity = operator.previous_velocity_y.to_numpy()[: operator.control_num].reshape((-1, 4, 3))
    momentum_increment = np.einsum(
        "bij,bjk->k",
        mass_matrix,
        velocity - previous_velocity,
    )
    downslope_inertial_force = float(np.dot(momentum_increment / dt, downslope))
    dynamic_balance_error = required_static_force - resolved_tangential_force - downslope_inertial_force
    print(
        f"mu={mu:g}, ccd_type={ccd_type}, regime={regime}, "
        f"initial_gap={initial_gap:.8e}, rigid_gross_sliding_oracle=0, "
        f"deformable_center_motion={numeric:.8e}, "
        f"interface_partial_slip={interface_partial_slip:.8e}, "
        f"interface_partial_slip_ratio={interface_partial_slip_ratio:.8e}, "
        f"last_step_tangential_displacement="
        f"{last_step_tangential_displacement:.8e}, "
        f"friction_transition_displacement={arguments.friction_epsv * dt:.8e}, "
        f"tangential_friction_force={resolved_tangential_force:.8e}, "
        f"required_static_force={required_static_force:.8e}, "
        f"force_balance_error={force_balance_error:.8e}, "
        f"downslope_inertial_force={downslope_inertial_force:.8e}, "
        f"dynamic_balance_error={dynamic_balance_error:.8e}, "
        f"coulomb_limit={coulomb_limit:.8e}, "
        "note='an initially stress-free deformable contact patch can undergo "
        "elastic motion and local partial slip below the gross Coulomb "
        f"threshold; use the maintained scalar-law test for the exact static "
        f"branch', output={save_path}"
    )

"""Two ordinary TET4 elastic bodies colliding through FEM IPC."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def hairpin_mesh(fem, divisions):
    arm_length = 0.4
    bend_radius = 0.1
    thickness = 0.12
    length = 2.0 * arm_length + np.pi * bend_radius
    longitudinal = 8 * divisions
    transverse = max(2, 3 * divisions // 4)
    mesh = fem.create_mesh(
        "box",
        size=(length, thickness, thickness),
        divisions=(longitudinal, transverse, transverse),
        origin=(0.0, -0.5 * thickness, -0.5 * thickness),
        name="hairpin",
    )
    material_coordinate = mesh.points.copy()
    s = material_coordinate[:, 0]
    q = material_coordinate[:, 1]
    center = np.zeros((mesh.number_of_nodes, 2), dtype=np.float64)
    tangent = np.zeros_like(center)
    left = s <= arm_length
    right = s >= arm_length + np.pi * bend_radius
    bend = ~(left | right)
    center[left, 1] = arm_length - s[left]
    tangent[left, 1] = -1.0
    angle = (s[bend] - arm_length) / bend_radius
    center[bend, 0] = bend_radius * (1.0 - np.cos(angle))
    center[bend, 1] = -bend_radius * np.sin(angle)
    tangent[bend, 0] = np.sin(angle)
    tangent[bend, 1] = -np.cos(angle)
    center[right, 0] = 2.0 * bend_radius
    center[right, 1] = s[right] - arm_length - np.pi * bend_radius
    tangent[right, 1] = 1.0
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    mesh.points[:, :2] = center + q[:, None] * normal

    velocity = np.zeros_like(mesh.points)
    velocity[left, 0] = 0.5
    velocity[right, 0] = -0.5
    velocity[bend, 0] = 0.5 * np.cos(angle)
    spacing = min(length / longitudinal, thickness / transverse)
    return mesh, velocity, spacing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="gpu")
    parser.add_argument("--divisions", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--scene", choices=("pair", "hairpin"), default="pair")
    parser.add_argument("--contact-model", choices=("BarrierIPC", "SemiIPC"), default="BarrierIPC")
    parser.add_argument("--assemble-type", choices=("COO", "HashTriplet"), default="HashTriplet")
    parser.add_argument("--friction-coefficient", type=float, default=0.2)
    parser.add_argument("--penalty", type=float, default=1.0e5)
    parser.add_argument("--max-penalty", type=float, default=1.0e12)
    parser.add_argument("--output-dir", default=str((Path(__file__).parents[1] / "OutputData") / "volume_self_contact"))
    args = parser.parse_args()
    if (
        args.divisions <= 0
        or args.steps <= 0
        or args.dt <= 0.0
        or args.friction_coefficient < 0.0
        or args.penalty <= 0.0
        or args.max_penalty < args.penalty
    ):
        parser.error("invalid time, mesh, friction, or SemiIPC penalty parameter")

    import geotaichi as gt
    from src.fem import FEMMesh

    gt.init(arch=args.arch, default_fp="float64", log=True)
    fem = gt.FEM("TET4 FEM IPC collision", log=True)
    fem.set_configuration(dimension=3, solver_type="Implicit")
    if args.scene == "pair":
        resolution = (2 * args.divisions, args.divisions, args.divisions)
        left = fem.create_mesh(
            "box", size=(0.6, 0.3, 0.3), divisions=resolution, origin=(0.25, 0.35, 0.35), name="left"
        )
        right = fem.create_mesh(
            "box", size=(0.6, 0.3, 0.3), divisions=resolution, origin=(0.95, 0.35, 0.35), name="right"
        )
        mesh = FEMMesh.concatenate((left, right), name="ordinary_tet4_pair")
        velocity = np.zeros_like(mesh.points)
        velocity[: left.number_of_nodes, 0] = 0.5
        velocity[left.number_of_nodes :, 0] = -0.5
        spacing = 0.3 / args.divisions
        dhat = 0.5 * spacing
        dmin = 0.1 * spacing
    else:
        mesh, velocity, spacing = hairpin_mesh(fem, args.divisions)
        dhat = 0.2 * spacing
        dmin = 0.05 * spacing
    fem.add_mesh(mesh)
    fem.add_material("NeoHookean", density=1000.0, young_modulus=1.0e5, poisson_ratio=0.3)
    fem.add_contact(
        args.contact_model,
        self_contact=True,
        broad_phase="BVH",
        dhat=dhat,
        dmin=dmin,
        kappa=1.0e5,
        penalty=args.penalty,
        max_penalty=args.max_penalty,
        friction_coefficient=args.friction_coefficient,
        epsv=1.0e-3,
        project_pd=True,
    )
    if args.scene == "pair":
        fem.add_contact_property(0, 1)
    fem.set_solver(
        dt=args.dt,
        step=args.steps,
        gravity=(0.0, 0.0, 0.0),
        damping=0.02,
        initial_velocity=velocity,
        max_iterations=100,
        residual_tolerance=1.0e-7,
        correction_velocity_tolerance=1.0e-3,
        line_search=True,
        assemble_type=args.assemble_type,
        linear_solver="PCG",
        linear_solver_relative_tolerance=1.0e-5,
        project_pd=True,
        output_interval=max(1, args.steps // 10),
        path=args.output_dir,
    )

    peak_contacts = [0]
    peak_violation = [0.0]
    peak_penalty = [args.penalty]
    first_contact_step = [None]

    def sample_contact(engine):
        contact = engine.last_step_record.get("contact", {})
        active_contacts = int(contact.get("active_contacts", 0))
        peak_contacts[0] = max(peak_contacts[0], active_contacts)
        peak_violation[0] = max(peak_violation[0], float(contact.get("contact_violation", 0.0)))
        peak_penalty[0] = max(peak_penalty[0], float(contact.get("contact_penalty", args.penalty)))
        if active_contacts and first_contact_step[0] is None:
            first_contact_step[0] = engine.step_count

    result = fem.run(verbose=False, postprocessing=[sample_contact])
    summary = {
        "case": "ordinary_tet4_fem_pair_ipc" if args.scene == "pair" else "ordinary_tet4_fem_self_contact",
        "scene": args.scene,
        "contact_model": args.contact_model,
        "assemble_type": args.assemble_type,
        "nodes": mesh.number_of_nodes,
        "tetrahedra": mesh.number_of_cells,
        "steps": args.steps,
        "dhat": dhat,
        "dmin": dmin,
        "friction_coefficient": args.friction_coefficient,
        "penalty": args.penalty,
        "max_penalty": args.max_penalty,
        "final_time": float(result.time),
        "converged": bool(result.converged),
        "finite_positions": bool(np.isfinite(result.positions).all()),
        "first_active_contact_step": first_contact_step[0],
        "first_active_contact_time": None if first_contact_step[0] is None else first_contact_step[0] * args.dt,
        "maximum_active_contacts": peak_contacts[0],
        "maximum_contact_violation": peak_violation[0],
        "maximum_contact_penalty": peak_penalty[0],
    }
    output = Path(args.output_dir)
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if (
        not summary["converged"]
        or not summary["finite_positions"]
        or not summary["maximum_active_contacts"]
        or not np.isclose(summary["final_time"], args.steps * args.dt)
    ):
        raise RuntimeError(summary)


if __name__ == "__main__":
    main()

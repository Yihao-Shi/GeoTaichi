"""GeoTaichi analogue of Newton's ``example_cloth_hanging``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def make_grid(nx, ny, width=6.3, height=3.1, z=4.0):
    points = []
    for i in range(nx):
        x = width * (i / (nx - 1) - 0.5)
        for j in range(ny):
            y = height * j / (ny - 1)
            points.append([x, y, z])
    cells = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            b = (i + 1) * ny + j
            cells.extend(((a, b, a + 1), (b, b + 1, a + 1)))
    return np.asarray(points, dtype=np.float64), np.asarray(cells, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="gpu")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--start-height", type=float, default=4.0)
    parser.add_argument("--spacing", type=float, default=0.1)
    parser.add_argument("--contact-model", choices=("BarrierIPC", "SemiIPC"), default="BarrierIPC")
    parser.add_argument("--max-iterations", type=int, default=250)
    parser.add_argument("--newton-velocity-tolerance", type=float, default=1.0e-2)
    parser.add_argument("--bending-stiffness", type=float, default=2.5e7)
    parser.add_argument("--friction-coefficient", type=float, default=1.0)
    parser.add_argument("--assemble-type", choices=("COO", "HashTriplet"), default="HashTriplet")
    parser.add_argument("--linear-solver", choices=("PCG", "BiCGSTAB", "Scipy"), default="PCG")
    parser.add_argument("--linear-relative-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--project-pd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--project-bending-pd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contact-project-pd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-dir", default=str(Path(__file__).with_name("OutputData") / "newton_cloth_hanging"))
    args = parser.parse_args()
    if (
        args.nx < 3
        or args.ny < 3
        or args.steps <= 0
        or args.dt <= 0.0
        or not np.isfinite(args.spacing)
        or args.spacing <= 0.0
        or args.max_iterations <= 0
        or args.newton_velocity_tolerance <= 0.0
        or not np.isfinite(args.bending_stiffness)
        or args.bending_stiffness <= 0.0
        or not np.isfinite(args.friction_coefficient)
        or args.friction_coefficient < 0.0
        or not np.isfinite(args.linear_relative_tolerance)
        or args.linear_relative_tolerance < 0.0
    ):
        parser.error("nx, ny >= 3, steps > 0, dt > 0, max-iterations > 0, and Newton tolerance > 0 are required")

    import geotaichi as gt
    from src.fem import DirichletBoundary, FEM
    from src.fem.generator import FEMMesh

    gt.init(arch=args.arch, default_fp="float64", log=True)
    width = args.spacing * (args.nx - 1)
    height = args.spacing * (args.ny - 1)
    points, cells = make_grid(args.nx, args.ny, width=width, height=height, z=args.start_height)
    cloth = FEM(title="Newton cloth hanging", log=True)
    cloth.set_configuration(dimension=3, solver_type="Implicit")
    cloth.add_mesh(FEMMesh(points, cells, "TRI3"))
    left = np.arange(args.ny, dtype=np.int32)
    cloth.add_boundary_condition(dirichlet=DirichletBoundary().add(left, "all", 0.0))
    cloth.add_material(
        "ClothARAP",
        stretch_stiffness=1.0e3,
        compression_stiffness=1.0e3,
        density=0.1,
        thickness=2.0e-2,
        bending_stiffness=args.bending_stiffness,
        bending_poisson_ratio=0.3,
        bending_model="Dihedral",
    )
    cloth.add_contact(
        args.contact_model,
        self_contact=True,
        planes=[((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))],
        dhat=2.0e-2,
        dmin=1.0e-3,
        kappa=5.0e2,
        friction_coefficient=args.friction_coefficient,
        project_pd=args.contact_project_pd,
    )
    cloth.set_solver(
        dt=args.dt,
        step=args.steps,
        gravity=(0.0, 0.0, -9.81),
        damping=0.1,
        max_iterations=args.max_iterations,
        residual_tolerance=1.0e-7,
        correction_velocity_tolerance=args.newton_velocity_tolerance,
        line_search=True,
        assemble_type=args.assemble_type,
        linear_solver=args.linear_solver,
        linear_solver_relative_tolerance=args.linear_relative_tolerance,
        project_pd=args.project_pd,
        project_bending_pd=args.project_bending_pd,
        output_interval=max(1, args.steps // 8),
        path=args.output_dir,
    )
    peak = {"active_contacts": 0, "pt_candidates": 0, "ee_candidates": 0}

    def sample_contact(engine):
        contact = engine.last_step_record.get("contact", {})
        for key in peak:
            peak[key] = max(peak[key], int(contact.get(key, 0)))

    result = cloth.run(verbose=args.verbose, postprocessing=[sample_contact])
    summary = {
        "case": "newton_cloth_hanging",
        "nodes": int(points.shape[0]),
        "triangles": int(cells.shape[0]),
        "steps": int(args.steps),
        "final_time": float(result.time),
        "spacing": float(args.spacing),
        "size": [float(width), float(height)],
        "assemble_type": args.assemble_type,
        "linear_solver": args.linear_solver,
        "linear_relative_tolerance": float(args.linear_relative_tolerance),
        "project_membrane_pd": bool(args.project_pd),
        "project_bending_pd": bool(args.project_bending_pd),
        "project_contact_pd": bool(args.contact_project_pd),
        "contact_model": args.contact_model,
        "bending_stiffness": float(args.bending_stiffness),
        "effective_bending_modulus": float(args.bending_stiffness * 2.0e-2**3 / (24.0 * (1.0 - 0.3**2))),
        "friction_coefficient": float(args.friction_coefficient),
        "converged": bool(result.converged),
        "finite_positions": bool(np.isfinite(result.positions).all()),
        "min_z": float(np.min(result.positions[:, 2])),
        "position_min": np.min(result.positions, axis=0).tolist(),
        "position_max": np.max(result.positions, axis=0).tolist(),
        "maximum_active_contacts": peak["active_contacts"],
        "maximum_pt_candidates": peak["pt_candidates"],
        "maximum_ee_candidates": peak["ee_candidates"],
    }
    (Path(args.output_dir) / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if (
        not summary["converged"]
        or not summary["finite_positions"]
        or not np.isclose(summary["final_time"], args.steps * args.dt)
    ):
        raise RuntimeError(summary)


if __name__ == "__main__":
    main()

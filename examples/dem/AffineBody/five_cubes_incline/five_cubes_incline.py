"""Five free ABD cubes on one incline, with converged outer friction iterations."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
MU = np.array([0.10, 0.30, 0.50, 0.70, 0.90])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="cpu")
    parser.add_argument("--time", type=float, default=1.2)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--output-dir", default="FiveCubesIncline")
    args = parser.parse_args()
    if not np.isfinite([args.time, args.dt]).all() or min(args.time, args.dt) <= 0:
        parser.error("time and dt must be finite and positive")
    steps = round(args.time / args.dt)
    if steps < 1 or not np.isclose(steps * args.dt, args.time, rtol=0, atol=1e-12):
        parser.error("time must be an integer multiple of dt")
    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    if args.arch == "cpu":
        os.environ["GEOTAICHI_FORCE_CPU"] = "1"
    sys.path.insert(0, str(ROOT))
    import geotaichi as gt
    import taichi as ti
    import trimesh
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from src.physics_model.contact_model.ipc.IPC import ipc_toolkit_barrier_distance_terms

    gt.init(arch=args.arch, default_fp="float64", cpu_max_num_threads=4, log=True)
    theta = np.pi / 6
    normal = np.array([-np.sin(theta), 0, np.cos(theta)])
    upslope = np.array([np.cos(theta), 0, np.sin(theta)])
    origin = np.array([0, 0, 0.1])
    edge, density, gravity = 0.25, 1000.0, 9.81
    dhat, kappa = 0.003, 8e8

    @ti.kernel
    def normal_force(gap: ti.f64) -> ti.f64:
        _, derivative, _ = ipc_toolkit_barrier_distance_terms(gap, dhat, kappa)
        return -3.0 * edge**2 * derivative

    # Start at normal equilibrium to avoid an artificial drop onto the ramp.
    load = density * edge**3 * gravity * np.cos(theta)
    lower, upper = 1e-12 * dhat, dhat
    assert normal_force(lower) > load > normal_force(upper)
    for _ in range(60):
        midpoint = (lower + upper) / 2
        if normal_force(midpoint) > load:
            lower = midpoint
        else:
            upper = midpoint
    gap = (lower + upper) / 2
    mesh_path = ROOT / "assets/mesh/AffineBody/cube.obj"
    cube = trimesh.load(mesh_path, process=False)
    local = cube.vertices * edge
    areas = np.bincount(cube.faces.ravel(), weights=np.repeat(cube.area_faces / 3, 3), minlength=8) * edge**2
    centers, orientations, setup_residuals = [], [], []
    for i, mu in enumerate(MU):
        effective_mu = min(mu, np.tan(theta))

        def pose(parameters):
            angles = np.array([gap / edge * parameters[1], -theta + gap / edge * parameters[2], 0])
            relative = local @ Rotation.from_euler("xyz", angles).as_matrix().T
            return angles, relative, edge / 2 + gap * parameters[0]

        def balance(parameters):
            _, relative, height = pose(parameters)
            distances = height + relative @ normal
            forces = areas * np.array([normal_force(max(d, 1e-12)) for d in distances]) / (3 * edge**2)
            moment = np.cross(relative, forces[:, None] * (normal + effective_mu * upslope)).sum(axis=0)
            return np.array([(forces.sum() - load) / load, moment @ upslope / (load * edge), moment[1] / (load * edge)])

        # Setup only: friction shifts the centre of pressure, so balance torque too.
        solution = least_squares(balance, [1, 0, 0], xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=100)
        angles, relative, height = pose(solution.x)
        residual = float(np.linalg.norm(balance(solution.x), ord=np.inf))
        assert solution.success and residual < 1e-8 and np.min(height + relative @ normal) > 0
        centers.append(origin + 3.5 * upslope + [0, 0.35 + 0.5 * i, 0] + height * normal)
        orientations.append(np.rad2deg(angles).tolist())
        setup_residuals.append(residual)
    centers = np.array(centers)
    dem = gt.DEM(log=False)
    dem.set_configuration(
        domain=[5, 3, 3], scheme="AffineBody", search="BVH", gravity=[0, 0, -gravity], visualize=False, log=False
    )
    dem.set_affine_body_parameters(
        assemble_type="MatrixFree",
        young_modulus=2e12,
        newmark=[1, 0.5, 1],
        friction_iterations=-1,
        friction_max_iterations=50,
        friction_tolerance=1e-7,
        friction_epsv=1e-4,
        newton_tolerance=1e-8,
        max_newton_iteration=1000,
        # The mass-norm bound gives <2.81e-10 m/s error, 2.81% of the Newton target.
        linear_tolerance=1e-13,
        linear_max_iteration=1000,
        line_search_max_iteration=50,
        ccd_type="accd",
        max_step=0.02,
    )
    dem.memory_allocate(
        {
            "max_material_number": 6,
            "max_affine_body_number": 5,
            "surface_node_number": 80,
            "max_plane_number": 1,
            "body_coordination_number": 8,
            "wall_coordination_number": 4,
            "compaction_ratio": [1.0, 1.0],
        },
        log=False,
    )
    dem.set_solver(
        {
            "Timestep": args.dt,
            "SimulationTime": args.time,
            "SaveInterval": min(0.01, args.time),
            "SavePath": args.output_dir,
        },
        log=False,
    )
    for material in range(6):
        dem.add_attribute(material, {"Density": density, "ForceLocalDamping": 0.0, "TorqueLocalDamping": 0.0})
    dem.add_template(
        {
            "Name": "cube",
            "TemplateType": "AffineBody",
            "Object": gt.polyhedron(file=str(mesh_path)),
        }
    )
    dem.create_body(
        {
            "BodyType": "AffineBody",
            "Template": [
                {
                    "Name": "cube",
                    "GroupID": i,
                    "MaterialID": i,
                    "BodyPoint": center.tolist(),
                    "ScaleFactor": edge,
                    "BodyOrientation": orientations[i],
                    "InitialVelocity": [0, 0, 0],
                    "Friction": float(MU[i]),
                }
                for i, center in enumerate(centers)
            ],
        }
    )
    dem.add_wall({"WallType": "Plane", "MaterialID": 5, "WallCenter": origin, "OuterNormal": normal})
    for i, mu in enumerate(MU):
        dem.add_property(
            i,
            5,
            {"Dhat": dhat, "BarrierStiffness": kappa, "ContactDampingStiffness": 0.0, "Friction": float(mu)},
            dType="particle-wall",
        )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    min_gap, max_stretch = float("inf"), 0.0

    @gt.python_callback
    def observe():
        nonlocal min_gap, max_stretch
        engine = dem.enginer
        # Output diagnostics only; the nonlinear solve and friction stay on device.
        engine.operator.sync_output_state()
        vertices = np.asarray(engine.state.world_vertices())
        min_gap = min(min_gap, float(np.min((vertices - origin) @ normal)))
        affine = np.transpose(engine.state.y[:, 1:] - engine.state.y[:, :1], (0, 2, 1))
        max_stretch = max(max_stretch, float(np.max(np.abs(np.linalg.svd(affine, compute_uv=False) - 1))))
        assert np.isfinite(vertices).all() and min_gap > 0
        assert engine.last_inner_converged and engine.last_linear_converged and engine.last_friction_converged
        diagnostics.append(
            {
                "time": engine.history[-1]["time"],
                "friction_iterations": engine.last_friction_iterations,
                "friction_residual": engine.last_friction_residual,
                "newton_iterations": engine.last_newton_iterations,
                "downslope_m": ((vertices.mean(axis=1) - centers) @ -upslope).tolist(),
            }
        )

    dem.run(function=observe)
    displacement = np.array(diagnostics[-1]["downslope_m"])
    acceleration = gravity * np.maximum(np.sin(theta) - MU * np.cos(theta), 0)
    expected = acceleration * args.dt**2 * steps * (steps + 1) / 2
    slide_error = np.max(np.abs(displacement[:3] - expected[:3]) / np.maximum(0.01 * expected[:3], 2e-5))
    normalized_error = max(float(slide_error), float(np.max(np.abs(displacement[3:]))) / 2e-4, max_stretch / 1e-4)
    summary = {
        "completed_time": dem.sims.current_time,
        "steps": len(diagnostics),
        "dt": args.dt,
        "mu": MU.tolist(),
        "initial_gap_m": gap,
        "initial_force_moment_relative_residuals": setup_residuals,
        "initial_orientation_degrees": orientations,
        "downslope_m": displacement.tolist(),
        "backward_euler_oracle_m": expected.tolist(),
        "minimum_gap_m": min_gap,
        "maximum_stretch_deviation": max_stretch,
        "maximum_friction_iterations": max(r["friction_iterations"] for r in diagnostics),
        "maximum_friction_residual": max(r["friction_residual"] for r in diagnostics),
        "normalized_error": normalized_error,
        "passed": normalized_error <= 1,
    }
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "history.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    evidence = {
        "schema_version": 1,
        "task_status": "completed",
        "finite_state": True,
        "capacity_overflow": False,
        "solver_converged": True,
        "timestep_consistent": len(diagnostics) == steps,
        "backend": str(ti.lang.impl.current_cfg().arch),
        "precision": "float64",
        "production_parameters": args.time == 1.2 and args.dt == 0.001,
        "command": " ".join(sys.argv),
        "checks": [
            {
                "name": "Incline motion and rigidity",
                "kind": "contract_observable",
                "observed": normalized_error,
                "expected": 0,
                "tolerance": 1.0,
                "evidence": str(output / "validation_summary.json"),
            },
            {
                "name": "Positive gap",
                "kind": "contact_gap",
                "status": "pass",
                "expected": "positive",
                "tolerance": 0,
                "evidence": str(output / "validation_summary.json"),
            },
        ],
    }
    (output / "validation-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    assert summary["passed"] and len(diagnostics) == steps, "Incline validation failed"


if __name__ == "__main__":
    main()

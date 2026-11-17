"""Evaluate saved column-twist results and render a preview without rerunning IGA."""

import argparse
import json
from pathlib import Path

import numpy as np


def evaluate(output_dir):
    output_dir = Path(output_dir)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    with np.load(output_dir / "reference.npz", allow_pickle=False) as data:
        reference, bottom, top = data["control_points"], data["bottom"], data["top"]
    steps, interval = summary["steps"], summary["output_interval"]
    expected_steps = [0, *range(interval, steps + 1, interval)]
    if expected_steps[-1] != steps:
        expected_steps.append(steps)
    files = sorted(output_dir.glob("state_*.npz"))
    if len(files) != len(expected_steps):
        raise ValueError("missing or stale state frames; use a separate output directory for each run")
    finite = bool(np.all(np.isfinite(reference)))
    bottom_error = top_error = time_error = 0.0
    minimum_jacobian = float("inf")
    angles, expected_angles, times = [], [], []
    interior = np.setdiff1d(np.arange(len(reference)), np.concatenate((bottom, top)))
    maximum_interior_displacement = 0.0
    for path, expected_step in zip(files, expected_steps):
        with np.load(path, allow_pickle=False) as data:
            if int(data["step"]) != expected_step:
                raise ValueError(f"unexpected step in {path.name}")
            current = data["control_points"]
            gradient, stress = data["deformation_gradient"], data["stress"]
            if current.shape != reference.shape or gradient.shape != (summary["elements"], 27, 3, 3):
                raise ValueError(f"unexpected control-point or quadrature shape in {path.name}")
            finite &= all(np.all(np.isfinite(values)) for values in (current, gradient, stress, data["time"]))
            if not finite:
                raise ValueError(f"non-finite state in {path.name}")
            fraction = expected_step / steps
            angle = summary["angle_degrees"] * fraction**2 * (3 - 2 * fraction)
            cosine, sine = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
            expected_top = reference[top].copy()
            expected_top[:, 0] = cosine * reference[top, 0] - sine * reference[top, 1]
            expected_top[:, 1] = sine * reference[top, 0] + cosine * reference[top, 1]
            bottom_error = max(bottom_error, float(np.max(np.abs(current[bottom] - reference[bottom]))))
            top_error = max(top_error, float(np.max(np.abs(current[top] - expected_top))))
            time_error = max(time_error, abs(float(data["time"]) - expected_step * summary["dt"]))
            jacobian = np.linalg.det(gradient)
            if not np.all(np.isfinite(jacobian)):
                raise ValueError(f"non-finite deformation determinant in {path.name}")
            minimum_jacobian = min(minimum_jacobian, float(jacobian.min()))
            maximum_interior_displacement = max(
                maximum_interior_displacement,
                float(np.max(np.linalg.norm(current[interior] - reference[interior], axis=1))),
            )
            angles.append(np.arctan2(current[top[0], 1], current[top[0], 0]))
            expected_angles.append(angle)
            times.append(float(data["time"]))
    resolved_rotation = bool(np.all(np.abs(np.diff(expected_angles)) < 180.0))
    measured_angles = np.rad2deg(np.unwrap(angles) - angles[0])
    angle_error = float(np.max(np.abs(measured_angles - expected_angles)))
    checks = {
        "finite_state": finite,
        "fixed_base": bottom_error <= 1e-10,
        "rotating_top": top_error <= 1e-10,
        "positive_quadrature_jacobian": minimum_jacobian > 0.0,
        "physical_time": time_error <= 1e-10 * max(1.0, steps * summary["dt"]),
        "rotation_history_resolved": resolved_rotation,
        "full_rotation_path": angle_error <= 1e-8,
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "solver": summary["solver"],
        "backend": summary["backend"],
        "precision": summary["precision"],
        "frames": len(files),
        "max_bottom_error_m": bottom_error,
        "max_top_error_m": top_error,
        "max_time_error_s": time_error,
        "minimum_quadrature_jacobian": minimum_jacobian,
        "max_rotation_error_degrees": angle_error,
        "measured_rotation_degrees": float(measured_angles[-1]),
        "maximum_interior_displacement_m": maximum_interior_displacement,
        "limitations": "Checks saved frames, not every substep; no mesh convergence or static torque benchmark.",
    }
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savetxt(
        output_dir / "rotation_history.csv",
        np.column_stack((times, expected_angles, measured_angles)),
        delimiter=",",
        header="time_s,expected_degrees,measured_degrees",
        comments="",
    )
    evidence = {
        "schema_version": 1,
        "task_status": "completed" if report["passed"] else "failed",
        "finite_state": finite,
        "capacity_overflow": False,
        "solver_converged": True if summary["solver"] == "implicit" else "not_applicable",
        "timestep_consistent": checks["physical_time"],
        "command": summary.get("command", ""),
        "backend": summary["backend"],
        "precision": summary["precision"],
        "production_parameters": bool(
            summary.get("degrees_of_freedom") == 1944
            and summary["angle_degrees"] == 450
            and np.isclose(steps * summary["dt"], 1.5)
        ),
        "checks": [
            {
                "name": "top rotation",
                "kind": "contract_observable",
                "observed": float(measured_angles[-1]),
                "expected": summary["angle_degrees"],
                "tolerance": {"absolute": 1e-8},
                "evidence": "rotation_history.csv",
            },
            {
                "name": "prescribed boundary coordinates",
                "kind": "analytical",
                "observed": max(bottom_error, top_error),
                "expected": 0,
                "tolerance": {"absolute": 1e-10},
                "evidence": "evaluation.json",
            },
        ],
    }
    (output_dir / "validation-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return report


def render(output_dir):
    import pyvista as pv

    files = sorted((output_dir / "vtks").glob("NurbsVolumetwisting_column*.vtu"))
    history = np.loadtxt(output_dir / "rotation_history.csv", delimiter=",", skiprows=1, ndmin=2)
    if len(files) != len(history):
        raise ValueError("VTU and state frame counts differ")
    indices = (0, len(files) // 2, len(files) - 1)
    meshes = [pv.read(files[index]) for index in indices]
    maximum_stress = max(float(mesh["stress"].max()) for mesh in meshes)
    plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(1500, 750))
    for panel, (index, mesh) in enumerate(zip(indices, meshes)):
        plotter.subplot(0, panel)
        plotter.set_background("white")
        plotter.add_mesh(
            mesh,
            scalars="stress",
            clim=(0, max(maximum_stress, 1.0)),
            cmap="viridis",
            show_edges=True,
            show_scalar_bar=False,
        )
        plotter.add_text(f"Top rotation: {history[index, 2]:.1f} deg", font_size=14, color="black")
        plotter.add_scalar_bar(
            title="von Mises stress (Pa)",
            color="black",
            n_labels=3,
            fmt="%.1e",
            width=0.7,
            position_x=0.15,
            position_y=0.03,
            title_font_size=14,
            label_font_size=12,
        )
        plotter.camera_position = [(5, -8, 4.5), (0, 0, 1.5), (0, 0, 1)]
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = 2.05
    plotter.show(screenshot=str(output_dir / "twist_preview.png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--no-plot", action="store_true", help="evaluate without a graphics context")
    args = parser.parse_args()
    report = evaluate(args.output_dir)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)
    if not args.no_plot:
        render(args.output_dir)


if __name__ == "__main__":
    main()

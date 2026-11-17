"""Extract the saved StripFootingMC reference and evaluate independent grid runs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def extract_reference(source, output):
    source, output = Path(source), Path(output)
    grids = sorted((source / "grids").glob("MPMGrid*.npz"))
    if len(grids) < 3:
        raise ValueError(f"Missing historical time series: {source}")
    rows, provenance = [], []
    initial_center = None
    for grid in grids:
        particle_path = source / "particles" / grid.name.replace("MPMGrid", "MPMParticle")
        with np.load(grid) as data, np.load(particle_path) as particles:
            positions = particles["position"]
            rigid = particles["bodyID"] == 1
            axis = positions.shape[1] - 1
            center = float(np.average(positions[rigid, axis], weights=particles["mass"][rigid]))
            if initial_center is None:
                initial_center = center
                thickness = 1.0
                if positions.shape[1] == 3:
                    thickness = float(np.ptp(positions[rigid, 1]) + 2.0 * particles["psize"][rigid, 1].max())
            if abs(float(data["t_current"]) - float(particles["t_current"])) > 1e-6:
                raise ValueError(f"Historical particle/grid times disagree: {grid}")
            rows.append(
                [
                    float(data["t_current"]),
                    initial_center - center,
                    float(data["contact_force"][:, 1, axis].sum()) / thickness,
                ]
            )
        provenance.append(
            {
                "grid": str(grid),
                "sha256": hashlib.sha256(grid.read_bytes()).hexdigest(),
                "particle": str(particle_path),
                "particle_sha256": hashlib.sha256(particle_path.read_bytes()).hexdigest(),
            }
        )
    values = np.asarray(rows)
    if not np.isfinite(values).all() or np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError("Historical reference contains nonfinite state or nonincreasing time")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, values, delimiter=",", header="time_s,settlement_m,footing_fy_N_per_m", comments="")
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "source": str(source),
                "thickness_m": thickness,
                "frames": len(rows),
                "final_time": values[-1, 0],
                "final_settlement": values[-1, 1],
                "normalization": "3D force divided by original thin-layer thickness",
                "provenance": provenance,
            },
            indent=2,
        )
        + "\n"
    )
    return {"frames": len(rows), "thickness_m": thickness, "settlement_max_m": values[-1, 1]}


def curve_errors(reference, candidate, bins=20):
    """Compare bin means over common settlement; preserve raw data separately."""
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    for curve in (reference, candidate):
        if len(curve) < 3 or not np.isfinite(curve).all() or np.any(np.diff(curve[:, 0]) < -1e-8):
            raise ValueError("Force curves must contain finite data and monotone measured settlement")
    low = max(reference[0, 0], candidate[0, 0])
    high = min(reference[-1, 0], candidate[-1, 0])
    if high - low <= 1e-8:
        raise ValueError("No usable common settlement interval")
    # Bins reduce contact-grid oscillations without fitting either curve to the other.
    edges = np.linspace(low, high, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = []
    for curve in (reference, candidate):
        average = []
        for left, right, center in zip(edges[:-1], edges[1:], centers):
            values = curve[(curve[:, 0] >= left) & (curve[:, 0] < right), 1]
            average.append(float(values.mean()) if len(values) else float(np.interp(center, curve[:, 0], curve[:, 1])))
        means.append(np.asarray(average))
    baseline, result = means
    scale = float(np.sqrt(np.mean(baseline**2)))
    if scale <= 1e-8:
        raise ValueError("Reference force is too small for relative comparison")
    tail = max(1, bins // 5)
    peak_ref, tail_ref = float(baseline.max()), float(baseline[-tail:].mean())
    if min(peak_ref, abs(tail_ref)) <= 1e-8:
        raise ValueError("Reference peak/tail cannot define a relative error")
    metrics = {
        "relative_rms": float(np.sqrt(np.mean((result - baseline) ** 2)) / scale),
        "peak_relative_error": float(abs(result.max() - peak_ref) / abs(peak_ref)),
        "tail_mean_relative_error": float(abs(result[-tail:].mean() - tail_ref) / abs(tail_ref)),
        "common_settlement_m": [float(low), float(high)],
        "bins": bins,
        "candidate_full_settlement_m": float(candidate[-1, 0]),
        "reference_full_settlement_m": float(reference[-1, 0]),
    }
    raw_peaks = [
        max(
            float(curve[(curve[:, 0] >= low) & (curve[:, 0] <= high), 1].max(initial=-np.inf)),
            float(np.interp(low, curve[:, 0], curve[:, 1])),
            float(np.interp(high, curve[:, 0], curve[:, 1])),
        )
        for curve in (reference, candidate)
    ]
    metrics["raw_peak_relative_error"] = abs(raw_peaks[1] - raw_peaks[0]) / max(abs(raw_peaks[0]), 1e-8)
    metrics["passed"] = all(
        metrics[key] <= 0.1
        for key in ["relative_rms", "peak_relative_error", "tail_mean_relative_error", "raw_peak_relative_error"]
    )
    return metrics, np.column_stack([centers, baseline, result])


def evaluate_run(output, require_refinement=True):
    output = Path(output)
    run = json.loads((output / "run.json").read_text())
    history = np.atleast_2d(np.genfromtxt(output / "contact_history.csv", delimiter=",", names=True))
    files = sorted((output / "particles").glob("MPMParticle*.npz"))
    masses, particle_counts, finite, positive = [], [], True, True
    minimum_volume_ratio, maximum_volume_ratio, maximum_speed = 1.0, 1.0, 0.0
    last = None
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            active = data["active"] == 1
            if last is None:
                rigid = data["bodyID"] == 1
                initial_footing_bottom = float(np.min(data["position"][rigid, 1] - data["psize"][rigid, 1]))
                initial_density = {}
                for body in np.unique(data["bodyID"]):
                    mask = (data["bodyID"] == body) & active
                    density = data["mass"][mask] / data["volume"][mask]
                    if not np.isfinite(density).all() or not np.allclose(density, density[0], rtol=1e-10):
                        raise ValueError("This footing evaluator requires uniform initial density in each body")
                    initial_density[int(body)] = float(density[0])
                initial_rigid_count = int(rigid.sum())
            particle_counts.append(int(active.sum()))
            masses.append(float(data["mass"][active].sum()))
            for key in ["position", "velocity", "stress", "mass", "volume"]:
                finite = finite and bool(np.isfinite(data[key][active]).all())
            positive = positive and bool((data["mass"][active] > 0).all() and (data["volume"][active] > 0).all())
            # Splitting divides both mass and reference volume; indices and counts may change.
            density0 = np.asarray([initial_density[int(body)] for body in data["bodyID"][active]])
            volume_ratio = data["volume"][active] * density0 / data["mass"][active]
            minimum_volume_ratio = min(minimum_volume_ratio, float(volume_ratio.min()))
            maximum_volume_ratio = max(maximum_volume_ratio, float(volume_ratio.max()))
            maximum_speed = max(maximum_speed, float(np.linalg.norm(data["velocity"][active], axis=1).max()))
            last = {key: data[key].copy() for key in ["position", "bodyID", "active"]}
            if "grid_level" in data:
                last["grid_level"] = data["grid_level"].copy()
            if "particle_refined" in data:
                last["particle_refined"] = data["particle_refined"].copy()
    if not masses:
        raise ValueError(f"Missing snapshots: {output}")
    force = history["footing_fy_N_per_m"].ravel()
    balance = np.max(np.abs(force + history["soil_fy_N_per_m"].ravel())) / max(1.0, np.max(np.abs(force)))
    mass_error = float(np.max(np.abs(np.asarray(masses) / masses[0] - 1)))
    time_ok = bool(abs(float(history["time_s"].ravel()[-1]) - run["duration"]) < 1e-8)
    report = {
        "completed": run["status"] == "completed",
        "finite_state": finite,
        "finite_history": bool(all(np.isfinite(history[name]).all() for name in history.dtype.names)),
        "positive_mass_volume": positive,
        "minimum_volume_ratio": minimum_volume_ratio,
        "maximum_volume_ratio": maximum_volume_ratio,
        "maximum_particle_speed_m_per_s": maximum_speed,
        "response_bounded": minimum_volume_ratio >= 0.5 and maximum_volume_ratio <= 2.0 and maximum_speed <= 1.0,
        "relative_mass_error": mass_error,
        "relative_action_reaction_error": float(balance),
        "time_complete": time_ok,
        "frames": len(files),
        "wall_seconds": run.get("wall_seconds"),
        "reused_baseline_from": run.get("reused_from"),
        "timing_comparable": run.get("timing_comparable", True),
        "final_settlement_m": float(history["settlement_m"].ravel()[-1]),
        "initial_particle_count": run["particle_count"],
        "final_particle_count": particle_counts[-1],
        "maximum_particle_count": max(particle_counts),
        "particle_count_matches_history": bool(
            "particle_count" not in history.dtype.names
            or int(history["particle_count"].ravel()[-1]) == particle_counts[-1]
        ),
        "rigid_particle_count_preserved": int(np.sum(last["bodyID"] == 1)) == initial_rigid_count,
        "level": run["level"],
        "maximum_l1_cells": float(history["l1_coarse_cells"].max()),
        "maximum_l2_cells": float(history["l2_coarse_cells"].max()),
    }
    if run["level"]:
        level = last["grid_level"]
        mask = (last["bodyID"] == 0) & (last["position"][:, 0] <= 1.4)
        mask &= last["position"][:, 1] > initial_footing_bottom - report["final_settlement_m"] - 0.4
        mask &= last["active"] == 1
        report["near_footing_particle_level_counts"] = {str(i): int(np.sum(level[mask] == i)) for i in range(3)}
        report["finest_level_reached"] = bool(np.any(level == run["level"]))
        if run.get("refine_particles", False):
            split_level = last["particle_refined"][last["bodyID"] == 0]
            report["particle_split_level_counts"] = {
                str(i): int(np.sum(split_level == i)) for i in range(run["level"] + 1)
            }
            report["particle_splitting_observed"] = bool(
                particle_counts[-1] > particle_counts[0] and np.any(split_level == run["level"])
            )
    report["passed"] = bool(
        report["completed"]
        and finite
        and report["finite_history"]
        and positive
        and report["response_bounded"]
        and mass_error <= 1e-10
        and balance <= 1e-8
        and time_ok
        and report["particle_count_matches_history"]
        and report["rigid_particle_count_preserved"]
        and (not require_refinement or report.get("finest_level_reached", True))
        and (not require_refinement or report.get("particle_splitting_observed", True))
    )
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    evidence = {
        "schema_version": 1,
        "task_status": run["status"],
        "finite_state": finite and report["finite_history"],
        "capacity_overflow": False,
        "solver_converged": "not_applicable",
        "timestep_consistent": bool(abs(run.get("effective_dt", 0) - run["dt"]) < 1e-10),
        "command": run["command"],
        "backend": run["backend"],
        "precision": run["precision"],
        "production_parameters": not run["smoke"] and run["duration"] == 17.5,
        "checks": [
            {
                "kind": "mass_balance",
                "observed": mass_error,
                "expected": 0,
                "tolerance": {"absolute": 1e-10},
                "evidence": str(output / "evaluation.json"),
            },
            {
                "kind": "contract_observable",
                "observed": float(balance),
                "expected": 0,
                "tolerance": {"absolute": 1e-8},
                "evidence": str(output / "contact_history.csv"),
            },
        ],
        "limitations": "This score checks numerical invariants only; historical and adaptive force agreement are separate required checks in comparison.json.",
    }
    (output / "validation-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return report, np.column_stack([history["settlement_m"].ravel(), force])


def initial_discretization_checks(root):
    """Compare the physical initial state while permitting coarser soil quadrature."""
    snapshots, runs = {}, {}
    for name in ("uniform", "adaptive_l1", "adaptive_l2"):
        runs[name] = json.loads((root / name / "run.json").read_text())
        with np.load(root / name / "particles/MPMParticle000000.npz") as data:
            snapshots[name] = {key: data[key].copy() for key in ("bodyID", "position", "mass", "volume", "psize")}
    reference = snapshots["uniform"]
    checks = {}
    for name, data in snapshots.items():
        soil, rigid = data["bodyID"] == 0, data["bodyID"] == 1
        reference_soil, reference_rigid = reference["bodyID"] == 0, reference["bodyID"] == 1
        run = runs[name]
        split = run.get("refine_particles", False)
        spacing = run["coarse_h"] / 4 if split else runs["uniform"]["coarse_h"] / 4
        checks[name] = bool(
            soil.sum() * (4 ** run["level"] if split else 1) == reference_soil.sum()
            and np.allclose(data["psize"][soil], spacing / 2, rtol=1e-10, atol=1e-12)
        )
        for body in (0, 1):
            mask, baseline = data["bodyID"] == body, reference["bodyID"] == body
            for key in ("mass", "volume"):
                checks[name] &= bool(np.isclose(data[key][mask].sum(), reference[key][baseline].sum(), rtol=1e-10))
            for side, reduction in ((-1, np.min), (1, np.max)):
                bounds = reduction(data["position"][mask] + side * data["psize"][mask], axis=0)
                expected = reduction(reference["position"][baseline] + side * reference["psize"][baseline], axis=0)
                checks[name] &= bool(np.allclose(bounds, expected, rtol=0, atol=1e-10))
        # The rigid body never splits; its contact geometry must match exactly.
        for key in ("position", "mass", "volume", "psize"):
            a, b = data[key][rigid], reference[key][reference_rigid]
            checks[name] &= bool(a.shape == b.shape and np.allclose(a, b, rtol=1e-10, atol=1e-12))
    return {name: bool(passed) for name, passed in checks.items()}


def preflight(output):
    report, _ = evaluate_run(output, require_refinement=False)
    # At <=0.1 s the prescribed displacement is <=0.0001 m, with no gravity or prestress.
    report["initial_response_bounded"] = bool(
        report["minimum_volume_ratio"] >= 0.5
        and report["maximum_volume_ratio"] <= 2.0
        and report["maximum_particle_speed_m_per_s"] <= 1.0
    )
    run = json.loads((Path(output) / "run.json").read_text())
    report["passed"] &= report["initial_response_bounded"] and run["duration"] <= 0.1
    (Path(output) / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def sampled_curve_errors(reference, candidate, sample_times):
    """Compare force samples at historical times; retain full-resolution diagnostics."""
    for history in (reference, candidate):
        if any(not np.isfinite(history[key]).all() for key in history.dtype.names):
            raise ValueError("Nonfinite force history")
        if np.any(np.diff(history["time_s"]) <= 0):
            raise ValueError("Force history times must increase")
    times = np.asarray(sample_times, dtype=float)
    low = max(reference["time_s"][0], candidate["time_s"][0])
    high = min(reference["time_s"][-1], candidate["time_s"][-1])
    times = times[(times >= low) & (times <= high)]
    if len(times) < 3 or np.any(np.diff(times) <= 0):
        raise ValueError("Insufficient common historical sampling times")
    ref_force = np.interp(times, reference["time_s"], reference["footing_fy_N_per_m"])
    force = np.interp(times, candidate["time_s"], candidate["footing_fy_N_per_m"])
    scale = float(np.linalg.norm(ref_force))
    if scale <= 1e-8 or ref_force.max() <= 1e-8:
        raise ValueError("Reference force is too small for relative comparison")
    tail = max(1, int(np.ceil(0.2 * len(times))))
    metrics = {
        "relative_rms": float(np.linalg.norm(force - ref_force) / scale),
        "peak_relative_error": float(abs(force.max() - ref_force.max()) / ref_force.max()),
        "tail_mean_relative_error": float(
            abs(force[-tail:].mean() - ref_force[-tail:].mean()) / max(abs(ref_force[-tail:].mean()), 1e-8)
        ),
        "sample_count": len(times),
        "sampling_interval_s": float(np.median(np.diff(times))),
    }
    # At constant speed, h/v = 0.1/0.1 = 1 s; compare complete cycles after the 5 s ramp.
    interval = metrics["sampling_interval_s"]
    cycles = np.floor(times + interval * 1e-3 - 5.0).astype(int)
    groups = [cycles == i for i in np.unique(cycles) if i >= 0 and np.count_nonzero(cycles == i) == round(1 / interval)]
    if groups:
        a = float(np.median([np.ptp(ref_force[g]) for g in groups]))
        b = float(np.median([np.ptp(force[g]) for g in groups]))
        metrics["oscillation_amplitude"] = {
            "method": "Median peak-to-peak force in complete 1 s windows after the velocity ramp",
            "cycles": len(groups),
            "reference_N_per_m": a,
            "candidate_N_per_m": b,
            "relative_difference": abs(b - a) / a if a > 1e-8 else None,
            "acceptance": "Reported separately for review; no separate amplitude tolerance was prescribed",
        }
    metrics["passed"] = all(
        metrics[k] <= 0.1 for k in ("relative_rms", "peak_relative_error", "tail_mean_relative_error")
    )
    raw = [np.column_stack((h["settlement_m"], h["footing_fy_N_per_m"])) for h in (reference, candidate)]
    metrics["unfiltered_diagnostics"], _ = curve_errors(*raw)
    values = np.column_stack(
        (
            times,
            np.interp(times, reference["time_s"], reference["settlement_m"]),
            ref_force,
            np.interp(times, candidate["time_s"], candidate["settlement_m"]),
            force,
        )
    )
    return metrics, values


def compare(root, reference, output=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    output = root if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if reference is None or not Path(reference).is_file():
        raise ValueError("Historical reference is required to define the sampling times")
    old = np.genfromtxt(reference, delimiter=",", names=True)
    reports, histories = {}, {}
    for name in ["uniform", "adaptive_l1", "adaptive_l2"]:
        reports[name], _ = evaluate_run(root / name)
        histories[name] = np.genfromtxt(root / name / "contact_history.csv", delimiter=",", names=True)
    initial_checks = initial_discretization_checks(root)
    historical, history_samples = sampled_curve_errors(old, histories["uniform"], old["time_s"])
    historical.update(status="passed" if historical["passed"] else "failed", reference=str(reference))
    curves = {"uniform": history_samples[:, 3:5]}
    comparisons = {}
    for name in ["adaptive_l1", "adaptive_l2"]:
        metrics, values = sampled_curve_errors(histories["uniform"], histories[name], old["time_s"])
        comparisons[name] = metrics
        curves[name] = values[:, 3:5]
        np.savetxt(
            output / f"{name}_comparison.csv",
            values,
            delimiter=",",
            comments="",
            header="time_s,uniform_settlement_m,uniform_force_N_per_m,adaptive_settlement_m,adaptive_force_N_per_m",
        )
    result = {
        "numerical_checks": reports,
        "initial_discretization_checks": initial_checks,
        "uniform_vs_historical": historical,
        "adaptive_vs_uniform": comparisons,
        "tolerance": 0.1,
        "source_root": str(root),
        "sampling_policy": "Original historical saved times (~0.25 s); linear interpolation in time, no averaging or smoothing. Unfiltered peaks are diagnostic only, as requested by the user.",
        "amplitude_policy": "Sampled oscillation amplitudes are reported separately for engineering review; curve pass checks RMS, sampled peak and tail mean.",
        "normalization": "Force per unit out-of-plane thickness; measured settlement",
        "passed": bool(
            all(initial_checks.values())
            and all(r["passed"] for r in reports.values())
            and historical["passed"]
            and all(r["passed"] for r in comparisons.values())
        ),
        "limitations": [
            "Historical control is a constrained 3D thin layer; new runs are 2D plane strain",
            "Historical source revision is unavailable; saved sliding-node forces imply friction 0.1, matching the current model",
            "Equal finest spacing does not prove spatial/time convergence; GPU contention prevents a controlled speed claim",
            "Historical-rate sampling can miss short contact impulses; their unfiltered diagnostics remain in this report",
        ],
    }
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for name, curve in curves.items():
        axes[0].plot(curve[:, 0], curve[:, 1] / 1000, label=name, linewidth=1.4)
        axes[1].plot(curve[:, 0] / 2, curve[:, 1] / 1000, label=name, linewidth=1.4)
    baseline = history_samples[:, 1:3]
    for ax, scale in zip(axes, [1, 2]):
        ax.plot(baseline[:, 0] / scale, baseline[:, 1] / 1000, "k--", label="historical normal", linewidth=1.4)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set(xlabel="Measured settlement (m)", ylabel="Vertical reaction (kN/m)")
    axes[1].set(xlabel="Settlement / full footing width", ylabel="q / cohesion (half width = 1 m)")
    fig.suptitle("Samples at historical saved times (~0.25 s), no smoothing")
    fig.savefig(output / "contact_force_comparison.png", dpi=180)
    plt.close(fig)
    lines = [
        "# Strip footing comparison",
        "",
        f"Force and numerical checks passed: **{result['passed']}**.",
        "",
        "All force comparisons use the original historical sampling times (~0.25 s).",
        "",
        "| Comparison | Pointwise RMS | Sampled peak error | Tail mean error | Amplitude difference (review) | Pass |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, metrics in [("uniform / saved", historical), *comparisons.items()]:
        amplitude = metrics.get("oscillation_amplitude", {}).get("relative_difference")
        values = [f"{metrics[k]:.2%}" for k in ("relative_rms", "peak_relative_error", "tail_mean_relative_error")]
        values.append("not available" if amplitude is None else f"{amplitude:.2%}")
        lines.append(f"| {name} | " + " | ".join(values) + f" | {metrics['passed']} |")
    lines += ["", result["sampling_policy"], "", result["amplitude_policy"]]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {"passed": result["passed"], "uniform_vs_historical": historical, "adaptive_vs_uniform": comparisons},
            indent=2,
        )
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract-reference")
    extract.add_argument("source", type=Path)
    extract.add_argument("output", type=Path)
    evaluate = sub.add_parser("compare")
    evaluate.add_argument("root", type=Path)
    evaluate.add_argument("--reference", type=Path)
    evaluate.add_argument("--output", type=Path, help="Write comparison artifacts separately from the input runs")
    check = sub.add_parser("preflight")
    check.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "extract-reference":
        print(json.dumps(extract_reference(args.source, args.output)))
    elif args.command == "preflight":
        result = preflight(args.output)
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit(1)
    else:
        result = compare(args.root, args.reference, args.output)
        if not result["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

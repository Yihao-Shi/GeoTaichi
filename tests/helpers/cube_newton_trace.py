"""Replay one failed CUDA cube step; retain trial diagnostics only on the remote host."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time
import types

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--restart-support", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--newton-tolerance", type=float, default=0.05)
    parser.add_argument("--linear-tolerance", type=float, default=1e-10)
    parser.add_argument("--trial", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--armijo", type=float, default=0.0)
    args = parser.parse_args()
    assert args.iterations > 0 and np.isfinite(args.dt) and args.dt > 0
    assert np.isfinite(args.newton_tolerance) and args.newton_tolerance > 0
    assert np.isfinite(args.linear_tolerance) and args.linear_tolerance > 0
    assert not args.output.exists(), args.output
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    os.environ.update(GEOTAICHI_REAL_DTYPE="float64", GT_AFFINE_MAX_HASH_TRIPLETS="8000000")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(args.restart_support))
    import geotaichi as gt
    import taichi as ti
    from restart import load_restart

    if args.armijo:
        import importlib
        from src.utils.SolverRuntime import accepts_monotone_energy

        assert 0.0 < args.armijo < 0.5
        engine_module = importlib.import_module("src.dem.engines.AffineBodyEngine")

        def sufficient_decrease(initial, trial, step, slope):
            roundoff = 1e-12 * max(abs(initial), abs(trial))
            return trial <= initial + args.armijo * step * slope or (
                -step * slope <= roundoff and accepts_monotone_energy(initial, trial, step, slope)
            )

        engine_module.accepts_monotone_energy = sufficient_decrease

    class Complete(BaseException):
        pass

    factory = gt.DEM

    def create(*factory_args, **factory_kwargs):
        dem = factory(*factory_args, **factory_kwargs)
        original_parameters, original_property = dem.set_affine_body_parameters, dem.add_property
        original_solver = dem.set_solver

        def solver(values, **options):
            return original_solver(dict(values, enable_step_retry=False), **options)

        def parameters(**values):
            values.update(
                barrier_stiffness=1e10,
                max_newton_iteration=args.iterations,
                newton_tolerance=args.newton_tolerance,
                linear_tolerance=args.linear_tolerance,
            )
            return original_parameters(**values)

        def property_(first, second, values, **options):
            return original_property(first, second, dict(values, BarrierStiffness=1e10), **options)

        def run(self, **unused):
            with np.load(args.checkpoint) as data:
                start = float(data["t_current"])
            metadata = dict(
                checkpoint=str(args.checkpoint),
                sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                dt=args.dt,
                step=args.checkpoint_step,
            )
            metadata_file = args.output / "restart_input.json"
            metadata_file.write_text(json.dumps(metadata))
            load_restart(self, metadata_file, start + args.dt, args.dt)
            engine, op = self.enginer, self.enginer.operator
            assert ti.lang.impl.current_cfg().arch == ti.cuda
            assert self.sims.affine_newton_tolerance == args.newton_tolerance and not engine.step_retry.enabled
            if args.audit_only:
                from src.dem.engines.AffineBodyOperator import MATRIX_HASH_TRIPLET

                op.set_timestep(args.dt)
                op.device_begin_step(args.dt)
                op.initialize_contact_damping_device()
                if args.trial:
                    with np.load(args.trial) as data:
                        assert float(data["dt"]) == args.dt
                        op.y.from_numpy(data["y"])
                energy = engine._assemble_system_device(self.sims, need_matrix=True)
                engine._solve_hash_direction_device(self.sims)
                y, gradient, direction = op.y.to_numpy(), op.grad.to_numpy(), op.direction_y.to_numpy()
                projected = engine.hash_triplet.to_scipy(op.control_num)
                residual = projected @ direction.ravel() + gradient.ravel()
                unit = direction / np.max(np.abs(direction))
                dx = np.sum(op.basis_np[:, :, None] * direction.reshape(op.body_num, 4, 3)[op.node2body_np], axis=1)
                worst_vertex = int(np.argmax(np.max(np.abs(dx), axis=1)))
                worst_body = int(op.node2body_np[worst_vertex])
                engine.hash_triplet.reset_system()
                op.assemble_device(
                    need_matrix=True, matrix_mode=MATRIX_HASH_TRIPLET, project_spd=False, solver_shift=False
                )
                engine.hash_triplet.finalize_taichi_assembly()
                exact = engine.hash_triplet.to_scipy(op.control_num)
                product = exact @ unit.ravel()
                projection_difference = projected - exact
                principal_minima = [
                    float(
                        np.linalg.eigvalsh(
                            projection_difference[12 * b : 12 * b + 12, 12 * b : 12 * b + 12].toarray()
                        ).min()
                    )
                    for b in range(op.body_num)
                ]
                probes = []
                for h in (1e-7, 1e-8, 1e-9):
                    op.y.from_numpy(y + h * unit)
                    plus = engine._assemble_system_device(self.sims, need_matrix=False)
                    plus_gradient = op.grad.to_numpy().ravel()
                    op.y.from_numpy(y - h * unit)
                    minus = engine._assemble_system_device(self.sims, need_matrix=False)
                    minus_gradient = op.grad.to_numpy().ravel()
                    numerical = (plus_gradient - minus_gradient) / (2 * h)
                    probes.append(
                        dict(
                            h=h,
                            energy_derivative=(plus - minus) / (2 * h),
                            tangent_error_inf=float(np.max(np.abs(numerical - product))),
                            tangent_scale=float(np.max(np.abs(product))),
                            tangent_relative_error=float(np.linalg.norm(numerical - product) / np.linalg.norm(product)),
                        )
                    )
                report = dict(
                    backend="cuda",
                    precision="float64",
                    start=start,
                    dt=args.dt,
                    energy=energy,
                    derivative=float(np.sum(gradient * unit)),
                    independent_linear_residual=float(np.linalg.norm(residual)),
                    reported_linear_residual=engine.last_linear_residual,
                    worst_body=worst_body,
                    worst_vertex=worst_vertex,
                    correction_velocity=float(np.max(np.abs(dx)) / args.dt),
                    worst_body_y=y.reshape(op.body_num, 4, 3)[worst_body].tolist(),
                    worst_body_direction=direction.reshape(op.body_num, 4, 3)[worst_body].tolist(),
                    worst_body_gradient=gradient.reshape(op.body_num, 4, 3)[worst_body].tolist(),
                    projected_curvature=float(unit.ravel() @ (projected @ unit.ravel())),
                    exact_curvature=float(unit.ravel() @ product),
                    projection_quadratic=float(unit.ravel() @ (projection_difference @ unit.ravel())),
                    minimum_principal_projection_eigenvalue=min(principal_minima),
                    worst_projection_body=int(np.argmin(principal_minima)),
                    probes=probes,
                    scope="Independent sparse-matrix residual and unprojected tangent derivative at unchanged trial",
                )
                assert np.isfinite(residual).all() and np.isfinite(product).all()
                (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(report), flush=True)
                raise Complete
            trace = []
            original_search = engine._line_search_cuda
            original_inner = engine._solve_lagged_inner_cuda

            def inner(sims, energy):
                if args.trial:
                    with np.load(args.trial) as data:
                        assert float(data["dt"]) == args.dt
                        op.y.from_numpy(data["y"])
                    energy = engine._assemble_system_device(sims, need_matrix=True)
                return original_inner(sims, energy)

            def search(sims, energy):
                iteration = len(trace)
                row = dict(
                    iteration=iteration,
                    energy=energy,
                    correction=engine.last_inner_residual,
                    slope=float(op.device_gradient_direction_dot()),
                    gradient_inf=float(op.device_gradient_inf_norm()),
                    linear_iterations=engine.last_linear_iterations,
                    linear_initial_residual=engine.last_linear_initial_residual,
                    linear_residual=engine.last_linear_residual,
                )
                if iteration % 50 == 0:
                    y, direction, gradient = op.y.to_numpy(), op.direction_y.to_numpy(), op.grad.to_numpy()
                    np.savez(
                        args.output / f"trial_{iteration:04d}.npz",
                        y=y,
                        direction=direction,
                        gradient=gradient,
                        dt=args.dt,
                        accepted=False,
                    )
                    norm = np.max(np.abs(direction))
                    assert np.isfinite(norm) and norm > 0
                    unit = direction / norm
                    row["derivative"] = float(np.sum(gradient * unit))
                    probes = []
                    for h in (1e-7, 1e-8, 1e-9):
                        op.y.from_numpy(y + h * unit)
                        plus = engine._assemble_system_device(sims, need_matrix=False)
                        op.y.from_numpy(y - h * unit)
                        minus = engine._assemble_system_device(sims, need_matrix=False)
                        probes.append(dict(h=h, plus=plus, minus=minus, derivative=(plus - minus) / (2 * h)))
                    row["probes"] = probes
                    op.y.from_numpy(y)
                    engine._assemble_system_device(sims, need_matrix=True)
                began = time.perf_counter()
                result = original_search(sims, energy)
                row.update(
                    trial_energy=result,
                    alpha=engine.last_line_search_alpha,
                    backtracks=engine.last_line_search_backtracks,
                    trials=engine.last_line_search_trials,
                    search_seconds=time.perf_counter() - began,
                    friction_majorizer=getattr(engine, "last_line_search_friction_majorizer", False),
                )
                trace.append(row)
                with (args.output / "trace.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                return result

            engine._line_search_cuda = search
            engine._solve_lagged_inner_cuda = inner
            report = dict(
                backend="cuda",
                precision="float64",
                start=start,
                dt=args.dt,
                scope="Diagnostic replay only; trial dumps are not accepted restart frames",
            )
            try:
                engine.step(self.sims, self.scene)
                report.update(converged=True, history=engine.history)
                if args.trial is None:
                    self.sims.current_time = start + args.dt
                    self.sims.current_step += 1
                    self.solver.save_file(self.scene)
                    report["accepted_restart"] = str(
                        args.output / "particles" / f"AffineBody{self.sims.current_print - 1:06d}.npz"
                    )
            except RuntimeError as error:
                report.update(converged=False, error=str(error), diagnostics=engine.diagnostics_snapshot())
            np.savez(
                args.output / "terminal_trial.npz",
                y=op.y.to_numpy(),
                gradient=op.grad.to_numpy(),
                dt=args.dt,
                accepted=False,
            )
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            raise Complete

        dem.set_affine_body_parameters = parameters
        dem.add_property = property_
        dem.set_solver = solver
        dem.run = types.MethodType(run, dem)
        return dem

    gt.DEM = create
    script = root / "examples/dem/AffineBody/thousand_cubes_heavy_ball_impact/thousand_cubes_heavy_ball_impact.py"
    sys.argv = [
        str(script),
        "--arch",
        "gpu",
        "--cube-count",
        "1000",
        "--dt",
        str(args.dt),
        "--output-dir",
        str(args.output),
    ]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except Complete:
        assert (args.output / "report.json").exists()


if __name__ == "__main__":
    main()

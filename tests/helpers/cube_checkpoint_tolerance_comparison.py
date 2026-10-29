"""Warm and compare fixed linear thresholds from one accepted cube-impact checkpoint."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from time import perf_counter
import types

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--restart-support", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-tolerance", type=float, required=True)
    parser.add_argument("--audit-initial-assembly", action="store_true")
    args = parser.parse_args()
    assert np.isfinite(args.relative_tolerance) and 0 < args.relative_tolerance < 1
    assert not args.output.exists(), args.output
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    os.environ.update(GEOTAICHI_REAL_DTYPE="float64", GT_AFFINE_MAX_HASH_TRIPLETS="8000000")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(args.restart_support))
    import geotaichi as gt
    import taichi as ti
    from restart import load_restart

    report = dict(
        backend="cuda",
        precision="float64",
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        scope="One-step paired replay; first complete step warms kernels and is excluded from timing comparison",
        passes=[],
    )
    result_path = args.output / "comparison.json"

    class Complete(BaseException):
        pass

    factory = gt.DEM

    def benchmark_factory(*factory_args, **factory_kwargs):
        dem = factory(*factory_args, **factory_kwargs)
        set_parameters, add_property = dem.set_affine_body_parameters, dem.add_property

        def parameters(**values):
            values.update(
                barrier_stiffness=1e10, max_newton_iteration=1000, linear_tolerance=1e-13, linear_relative_tolerance=0.0
            )
            return set_parameters(**values)

        def property_(first, second, values, **options):
            return add_property(first, second, dict(values, BarrierStiffness=1e10), **options)

        def run(self, *unused_args, **unused_kwargs):
            with np.load(args.checkpoint) as data:
                saved = {key: data[key] for key in data.files}
            start, dt = float(saved["t_current"]), 0.001
            metadata = dict(
                checkpoint=str(args.checkpoint), sha256=report["checkpoint_sha256"], dt=dt, step=args.checkpoint_step
            )
            metadata_file = args.output / "restart_input.json"
            metadata_file.write_text(json.dumps(metadata))
            load_restart(self, metadata_file, start + dt, dt)
            engine, op = self.enginer, self.enginer.operator
            assert not op.use_newmark and not op.fully_implicit and not op.levelset_contact
            assert self.sims.affine_newton_tolerance == 1e-7 and not engine.step_retry.enabled
            if args.audit_initial_assembly:
                from src.utils.FieldIO import field_to_numpy_prefix

                op.set_timestep(dt)
                op.device_begin_step(dt)
                snapshots = []
                report["scope"] = "Repeated initial assembly at unchanged checkpoint state; no Newton update"
                for repeat in range(3):
                    op.initialize_contact_damping_device()
                    count = int(op.friction_contact_count[0])
                    fields = [
                        field_to_numpy_prefix(getattr(op, "friction_contact_" + key), count)
                        for key in ("bodies", "vertices", "weights", "normal", "hat_rel", "coeff")
                    ]
                    contacts = np.column_stack(fields)
                    contacts = contacts[np.lexsort(contacts.T[::-1])]
                    energy = engine._assemble_system_device(self.sims, need_matrix=True)
                    engine.hash_triplet.finalize_taichi_assembly()
                    matrix = engine.hash_triplet.to_scipy(op.control_num)
                    gradient = op.grad.to_numpy()
                    row = dict(
                        repeat=repeat,
                        contacts=count,
                        energy=energy,
                        nnz=matrix.nnz,
                        gradient_max=float(np.abs(gradient).max()),
                        matrix_max=float(abs(matrix).max()),
                    )
                    if snapshots:
                        first_contacts, first_matrix, first_gradient = snapshots[0]
                        row.update(
                            contacts_equal=bool(np.array_equal(contacts, first_contacts)),
                            gradient_difference=float(np.abs(gradient - first_gradient).max()),
                            matrix_difference=float(abs(matrix - first_matrix).max()),
                        )
                        if contacts.shape == first_contacts.shape:
                            difference = np.abs(contacts - first_contacts)
                            row["contact_column_max_difference"] = difference.max(axis=0).tolist() if count else []
                            changed = np.flatnonzero(np.any(difference != 0, axis=1))
                            row["changed_contact_rows"] = len(changed)
                            row["first_changed_rows"] = [
                                dict(cold=first_contacts[i].tolist(), repeated=contacts[i].tolist())
                                for i in changed[:5]
                            ]
                    snapshots.append((contacts, matrix, gradient))
                    report["passes"].append(row)
                    result_path.write_text(json.dumps(report, indent=2) + "\n")
                report["passed"] = all(
                    row["contacts_equal"]
                    and row["gradient_difference"] <= 1e-12 * report["passes"][0]["gradient_max"]
                    and row["matrix_difference"] <= 1e-12 * report["passes"][0]["matrix_max"]
                    for row in report["passes"][1:]
                )
                result_path.write_text(json.dumps(report, indent=2) + "\n")
                assert report["passed"], report
                raise Complete
            basis, ids = op.basis.to_numpy(), saved["bodyID"].reshape(-1).astype(int)
            linear_records = []
            solve_direction = engine._solve_hash_direction_device

            def measured_direction(sims):
                result = solve_direction(sims)
                linear_records.append(
                    {
                        key: result[key]
                        for key in ("iterations", "residual", "initial_residual", "convergence_tolerance")
                    }
                )
                return result

            engine._solve_hash_direction_device = measured_direction
            outputs = {}
            for name, relative in (("warmup", 0.0), ("absolute", 0.0), ("relative", args.relative_tolerance)):
                for field in ("y", "hat_y", "tilde_y", "previous_y"):
                    getattr(op, field).from_numpy(saved["y"].reshape(-1, 3))
                for field in ("velocity_y", "previous_velocity_y"):
                    getattr(op, field).from_numpy(saved["v_y"].reshape(-1, 3))
                if "a_y" in saved:
                    for field in ("acceleration_y", "previous_acceleration_y"):
                        getattr(op, field).from_numpy(saved["a_y"].reshape(-1, 3))
                op.sync_output_state()
                np.testing.assert_array_equal(op.y.to_numpy(), saved["y"].reshape(-1, 3))
                np.testing.assert_array_equal(op.velocity_y.to_numpy(), saved["v_y"].reshape(-1, 3))
                # Test-only replay changes solver controls on the same allocated workspace.
                self.sims.affine_linear_tolerance = 1e-13
                self.sims.affine_linear_relative_tolerance = relative
                linear_records.clear()
                ti.sync()
                began = perf_counter()
                engine.step(self.sims, self.scene)
                ti.sync()
                row = dict(
                    engine.history[-1],
                    name=name,
                    relative_tolerance=relative,
                    wall_seconds=perf_counter() - began,
                    pcg_solves=len(linear_records),
                    pcg_iterations=sum(r["iterations"] for r in linear_records),
                )
                assert row["step_retry"]["retry_count"] == 0 and row["step_retry"]["accepted_timestep"] == dt
                assert all(r["residual"] <= r["convergence_tolerance"] for r in linear_records)
                outputs[name] = (op.y.to_numpy().reshape(-1, 4, 3), op.velocity_y.to_numpy().reshape(-1, 4, 3))
                assert all(np.isfinite(value).all() for value in outputs[name])
                report["passes"].append(row)
                result_path.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)
            comparisons = {}
            for name in ("warmup", "relative"):
                errors = [
                    float(np.abs(np.einsum("vi,vij->vj", basis, (value - reference)[ids])).max())
                    for value, reference in zip(outputs[name], outputs["absolute"])
                ]
                comparisons[name] = dict(surface_position_difference=errors[0], surface_velocity_difference=errors[1])
            limit = 2 * self.sims.affine_newton_tolerance
            report.update(
                comparisons=comparisons,
                position_limit=dt * limit,
                velocity_limit=limit,
                passed=all(
                    v["surface_position_difference"] <= dt * limit and v["surface_velocity_difference"] <= limit
                    for v in comparisons.values()
                ),
            )
            report["warm_speed_ratio"] = report["passes"][1]["wall_seconds"] / report["passes"][2]["wall_seconds"]
            result_path.write_text(json.dumps(report, indent=2) + "\n")
            assert report["passed"], comparisons
            raise Complete

        dem.set_affine_body_parameters, dem.add_property = parameters, property_
        dem.run = types.MethodType(run, dem)
        return dem

    gt.DEM = benchmark_factory
    script = root / "examples/dem/AffineBody/thousand_cubes_heavy_ball_impact/thousand_cubes_heavy_ball_impact.py"
    sys.argv = [str(script), "--arch", "gpu", "--cube-count", "1000", "--dt", ".001", "--output-dir", str(args.output)]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except Complete:
        pass
    except BaseException as error:
        report.update(passed=False, error=repr(error))
        result_path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    finally:
        gt.DEM = factory


if __name__ == "__main__":
    main()

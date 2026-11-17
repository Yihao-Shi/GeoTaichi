"""Replay a captured ABD matrix on CUDA; host calculations are independent test oracles."""

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.linalg import splu
import taichi as ti


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", type=Path, required=True)
    parser.add_argument("--geometry-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt", type=float, required=True)
    args = parser.parse_args()
    assert np.isfinite(args.dt) and args.dt > 0
    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    ti.init(arch=ti.cuda, enable_fallback=False, default_fp=ti.f64, offline_cache=False)
    from src.linear_solver.BuildTriplet import BuildTriplet

    matrix = load_npz(args.system / "failed_matrix.npz")
    with np.load(args.system / "failed_vectors.npz") as data:
        rhs = data["b"]
    extended = matrix.astype(np.longdouble)
    factor = splu(matrix.tocsc())
    reference = factor.solve(rhs)
    for _ in range(4):
        residual = rhs.astype(np.longdouble) - extended @ reference.astype(np.longdouble)
        reference += factor.solve(np.asarray(residual, dtype=float))
    with np.load(args.geometry_checkpoint) as data:
        controls, vertices, ids = data["y"], data["vertices"], data["bodyID"].reshape(-1).astype(int)
    homogeneous = np.concatenate((controls, np.ones((*controls.shape[:2], 1))), axis=2)
    basis = np.einsum(
        "vi,vij->vj", np.column_stack((vertices, np.ones(len(vertices)))), np.linalg.inv(homogeneous)[ids]
    )
    np.testing.assert_allclose(np.einsum("vi,vij->vj", basis, controls[ids]), vertices, rtol=0, atol=1e-13)
    nodes = rhs.size // 3
    system = BuildTriplet(
        3,
        matrix.nnz,
        matrix.nnz,
        nodes,
        symmetric=False,
        matrix_symmetric=True,
        solver="PCG",
        device_reduction=True,
        preconditioner_group_size=4,
    )
    coo = matrix.tocoo()
    keep = coo.row // 3 <= coo.col // 3
    system.reset_system()
    system.assemble_scalar_triplets(coo.row[keep], coo.col[keep], coo.data[keep])
    system.finalize_taichi_assembly()
    assert float(abs(system.to_scipy(nodes) - matrix).max()) == 0.0
    system.rhs.from_numpy(rhs.reshape(nodes, 3))
    system.solve(active_nodes=nodes, tol=1e-13, maxiter=1, return_solution=False)
    report = dict(
        backend="cuda",
        precision="float64",
        matrix=str(args.system),
        dt=args.dt,
        scope="Fixed-matrix calibration only; concurrent GPU work prevents a production speedup claim",
        comparisons=[],
    )
    for absolute, relative in [(1e-14, 0), (1e-13, 0), (0, 1e-13), (0, 1e-12), (0, 1e-10), (0, 1e-8)]:
        system.x.fill(0)
        ti.sync()
        started = perf_counter()
        result = system.solve(active_nodes=nodes, tol=absolute, rel_tol=relative, maxiter=3000, return_solution=False)
        ti.sync()
        elapsed = perf_counter() - started
        solution = system.x.to_numpy().ravel()
        error = (solution - reference).reshape(-1, 4, 3)
        velocity_error = np.abs(np.einsum("vi,vij->vj", basis, error[ids])).max() / args.dt
        result.update(
            absolute_tolerance=absolute,
            relative_tolerance=relative,
            warm_seconds=elapsed,
            surface_velocity_error=float(velocity_error),
            extended_true_residual=float(
                np.linalg.norm(rhs.astype(np.longdouble) - extended @ solution.astype(np.longdouble))
            ),
        )
        report["comparisons"].append(result)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

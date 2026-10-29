"""Compare host/device scalar recurrences; timings cover linear solves, not timesteps."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.linear_solver.BuildTriplet import BuildTriplet
from src.linear_solver.MatrixFreeCG import MatrixFreeCG
from src.linear_solver.MatrixFreePCG import MatrixFreePCG
from src.linear_solver.MatrixFreeBICGSTAB import MatrixFreeBICGSTAB
from src.linear_solver.MatrixFreePBICGSTAB import MatrixFreePBICGSTAB
from src.linear_solver.MultiGridPCG_mixture import MGPCGMixPoissonSolver
from src.linear_solver.MultiGridPCG_mixture_Axi import MGPCGMixPoissonSolver_Axi


@ti.data_oriented
class Stencil:
    def __init__(self, size, nonsymmetric):
        self.size = size
        self.left = -0.9 if nonsymmetric else -1.0

    @ti.kernel
    def matvec(self, x: ti.template(), result: ti.template()):
        for i in range(self.size):
            value = 2.1 * x[i]
            if i > 0:
                value += self.left * x[i - 1]
            if i + 1 < self.size:
                value -= x[i + 1]
            result[i] = value


def scalar_case(size, cls, preconditioned, nonsymmetric):
    solver, operator = cls(size), Stencil(size, nonsymmetric)
    b, x, diagonal = [ti.field(float, shape=size) for _ in range(3)]
    rhs = np.random.default_rng(41).normal(size=size)
    b.from_numpy(rhs)
    diagonal.fill(2.1)

    def solve(device):
        solver.device_execution = device
        x.fill(0.0)
        args = (operator, b, x, diagonal, size) if preconditioned else (operator, b, x, size)
        assert solver.solve(*args, tol=1e-9, rel_tol=1e-10, maxiter=400), solver.last_breakdown_reason
        return solver.last_iterations, solver.last_residual

    def residual(solution):
        value = 2.1 * solution - rhs
        value[1:] += operator.left * solution[:-1]
        value[:-1] -= solution[1:]
        return np.linalg.norm(value)

    return dict(
        name=cls.__name__,
        size=size,
        solve=solve,
        solution=x.to_numpy,
        residual=residual,
        tolerance=max(1e-9, 1e-10 * np.linalg.norm(rhs)),
    )


def block_case(nodes, nonsymmetric):
    solver = BuildTriplet(
        dim=3,
        max_pairs_num=2 * nodes,
        max_nonzeros=2 * nodes,
        max_active_nodes=nodes,
        symmetric=False,
        matrix_symmetric=not nonsymmetric,
        solver="BiCGSTAB" if nonsymmetric else "PCG",
    )
    block = np.array([[2.2, 0.01, 0.0], [0.01, 2.15, 0.02], [0.0, 0.02, 2.18]])
    if nonsymmetric:
        block[1, 0] = -0.02
    solver.diag.from_numpy(np.tile(block.ravel(), (nodes, 1)))
    row = np.arange(nodes - 1, dtype=np.int32)
    rows, columns = row, row + 1
    upper = -0.9 if nonsymmetric else -1.0
    entries = np.tile((upper * np.eye(3)).ravel(), (nodes - 1, 1))
    if nonsymmetric:
        rows, columns = np.concatenate((row, row + 1)), np.concatenate((row + 1, row))
        entries = np.concatenate((entries, np.tile((-np.eye(3)).ravel(), (nodes - 1, 1))))
    solver._set_reduced_non_diag(rows, columns, entries)
    rhs = np.random.default_rng(17).normal(size=(nodes, 3))
    solver.rhs.from_numpy(rhs)

    def solve(device):
        solver.device_execution = device
        result = solver.solve(active_nodes=nodes, tol=1e-9, rel_tol=1e-10, maxiter=400, return_solution=False)
        assert result["converged"], result
        return result["iterations"], result["residual"]

    def residual(solution):
        value = solution @ block.T - rhs
        value[:-1] += upper * solution[1:]
        value[1:] -= solution[:-1]
        return np.linalg.norm(value)

    return dict(
        name="BuildTriplet_" + ("BiCGSTAB" if nonsymmetric else "PCG"),
        size=3 * nodes,
        solve=solve,
        solution=solver.x.to_numpy,
        residual=residual,
        tolerance=max(1e-9, 1e-10 * np.linalg.norm(rhs)),
    )


def mixture_case(axisymmetric):
    dim = 2 if axisymmetric else 3
    shape = (128, 64) if axisymmetric else (32, 32, 32)
    cls = MGPCGMixPoissonSolver_Axi if axisymmetric else MGPCGMixPoissonSolver
    solver = cls(dim, shape, n_mg_levels=3, bottom_smoothing=10, **({"dr": 0.1} if axisymmetric else {}))
    for level in range(3):
        solver.grid_type[level].fill(solver.FLUID)
        solver.Adiag[level].fill(-2.0 * dim - 0.5)
        solver.Ax[level].fill(1.0)
        if axisymmetric:
            solver.Ax_neg[level].fill(1.0)
    rhs = np.random.default_rng(13).normal(size=shape)
    solver.b.from_numpy(rhs)

    def solve(device):
        solver.graph_dispatch = device
        solver.x.fill(0.0)
        assert solver.solve(400, rel_tol=1e-10, abs_tol=1e-20), solver.breakdown_reason
        return solver.last_iterations, np.sqrt(solver.final_residual)

    def residual(solution):
        value = (-2.0 * dim - 0.5) * solution - rhs
        for axis in range(dim):
            left, right = [slice(None)] * dim, [slice(None)] * dim
            left[axis], right[axis] = slice(None, -1), slice(1, None)
            value[tuple(left)] += solution[tuple(right)]
            value[tuple(right)] += solution[tuple(left)]
        return np.linalg.norm(value)

    return dict(
        name=cls.__name__,
        size=int(np.prod(shape)),
        solve=solve,
        solution=solver.x.to_numpy,
        residual=residual,
        tolerance=np.sqrt(max(1e-20, 1e-10 * np.sum(rhs**2))),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--sizes", type=int, nargs="+", default=(1024, 65536))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ti.init(arch=getattr(ti, args.arch), default_fp=ti.f64, offline_cache=False, device_memory_GB=0.5)
    cases = []
    for size in args.sizes:
        for cls, preconditioned, nonsymmetric in (
            (MatrixFreeCG, False, False),
            (MatrixFreePCG, True, False),
            (MatrixFreeBICGSTAB, False, True),
            (MatrixFreePBICGSTAB, True, True),
        ):
            cases.append(scalar_case(size, cls, preconditioned, nonsymmetric))
        for nonsymmetric in (False, True):
            cases.append(block_case(max(16, size // 3), nonsymmetric))
    cases.extend([mixture_case(False), mixture_case(True)])
    for case in cases:
        solutions, residuals = [], []
        for device in (False, True):
            for _ in range(2):
                case["solve"](device)
            solution = case["solution"]()
            solutions.append(solution)
            residuals.append(float(case["residual"](solution)))
        np.testing.assert_allclose(solutions[1], solutions[0], rtol=1e-7, atol=1e-8)
        assert max(residuals) <= case["tolerance"] * (1 + 1e-4), (case["name"], residuals, case["tolerance"])
        case["true_residuals"] = residuals
        case["solution_max_error"] = float(np.max(np.abs(solutions[1] - solutions[0])))
        print("WARMED", case["name"], case["size"], flush=True)
    print("DEVICE_KRYLOV_WARMUP_COMPLETE", flush=True)
    time.sleep(0.5)
    result = []
    for case in cases:
        samples, iterations = [[], []], [[], []]
        for repeat in range(args.repeats):
            for device in (False, True) if repeat % 2 == 0 else (True, False):
                ti.sync()
                began = time.perf_counter()
                count, residual = case["solve"](device)
                ti.sync()
                samples[int(device)].append(time.perf_counter() - began)
                iterations[int(device)].append(count)
        medians = [float(np.median(values)) for values in samples]
        row = dict(
            name=case["name"],
            size=case["size"],
            host_s=medians[0],
            device_s=medians[1],
            speedup=medians[0] / medians[1],
            iterations=iterations,
            samples_s=samples,
            true_residuals=case["true_residuals"],
            solution_max_error=case["solution_max_error"],
        )
        result.append(row)
        print("RESULT", json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(arch=args.arch, fp="float64", scope=__doc__, cases=result), indent=2) + "\n")


if __name__ == "__main__":
    main()

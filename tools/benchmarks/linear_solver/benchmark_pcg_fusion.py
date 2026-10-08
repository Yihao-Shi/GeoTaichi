"""Compare split and fused HashTriplet PCG kernels on the same SPD block chain.

Each sample synchronizes Taichi; compilation and solution downloads are excluded.
This measures a linear solve, not a complete coupled timestep.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.linear_solver.BuildTriplet import BuildTriplet, _dense_block_matvec


@ti.data_oriented
class SplitPCG(BuildTriplet):
    """Reference launch/transfer layout, with the same PCG stopping rules."""

    def PCGSolver(self, *args, **kwargs):
        self._solver_reset(kwargs["active_nodes"])
        return super().PCGSolver(*args, **kwargs)

    @ti.kernel
    def _split_init(self, active_nodes: int):
        for i in range(active_nodes):
            self.r[i] = self.rhs[i] - self.Ax[i]
            self.z[i] = _dense_block_matvec(self.diag_inverse[i], self.r[i], ti.static(self.dim))
            self.p[i] = self.z[i]

    def _init_pcg(self, active_nodes):
        self._split_init(active_nodes)
        return self._dot(active_nodes, self.r, self.z), self._dot(active_nodes, self.r, self.r)

    @ti.kernel
    def _split_update(self, active_nodes: int, alpha: float):
        for i in range(active_nodes):
            self.x[i] += alpha * self.p[i]
            self.r[i] -= alpha * self.Ap[i]

    def _pcg_update_and_reduce(self, active_nodes, alpha):
        self._split_update(active_nodes, alpha)
        self._apply_preconditioner(active_nodes, self.r, self.z)
        return self._dot(active_nodes, self.r, self.z), self._dot(active_nodes, self.r, self.r)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "cuda", "metal"), default="cuda")
    parser.add_argument("--nodes", type=int, nargs="+", default=(1024, 8192, 32768))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.nodes) < 2 or args.repeats < 1:
        parser.error("node counts must be at least two and repeats positive")
    ti.init(arch=getattr(ti, args.arch), default_fp=ti.f64, offline_cache=False)
    cases = []
    for nodes in args.nodes:
        systems = [
            cls(
                dim=3,
                max_pairs_num=nodes,
                max_nonzeros=nodes,
                max_active_nodes=nodes,
                symmetric=False,
                matrix_symmetric=True,
                solver="PCG",
            )
            for cls in (SplitPCG, BuildTriplet)
        ]
        diagonal_block = np.array([[2.05, 0.01, -0.005], [0.01, 2.08, 0.02], [-0.005, 0.02, 2.1]])
        diagonal = np.tile(diagonal_block.ravel(), (nodes, 1))
        upper = np.arange(nodes - 1, dtype=np.int32)
        blocks = np.tile((-np.eye(3)).ravel(), (nodes - 1, 1))
        rhs = np.random.default_rng(0).normal(size=(nodes, 3))
        for system in systems:
            system.diag.from_numpy(diagonal)
            system._set_reduced_non_diag(upper, upper + 1, blocks)
            system.rhs.from_numpy(rhs)

        def solve(index):
            result = systems[index].PCGSolver(
                active_nodes=nodes, tol=1e-9, rel_tol=1e-10, maxiter=400, return_solution=False
            )
            ti.sync()
            assert result["converged"], result
            return result

        for index in (0, 1):
            solve(index)
            solve(index)
        solutions = [system.x.to_numpy() for system in systems]
        np.testing.assert_allclose(solutions[1], solutions[0], rtol=2e-9, atol=2e-9)
        true_residuals = []
        for solution in solutions:
            residual = solution @ diagonal_block.T - rhs
            residual[1:] -= solution[:-1]
            residual[:-1] -= solution[1:]
            true_residuals.append(float(np.linalg.norm(residual)))
        assert max(true_residuals) <= max(1e-9, 1e-10 * np.linalg.norm(rhs)) * (1 + 1e-5)
        samples, iterations = [[], []], [[], []]
        for sample in range(args.repeats):
            for index in (0, 1) if sample % 2 == 0 else (1, 0):
                started = time.perf_counter()
                result = solve(index)
                samples[index].append(time.perf_counter() - started)
                iterations[index].append(result["iterations"])
        medians = [float(np.median(values)) for values in samples]
        cases.append(
            dict(
                nodes=nodes,
                dofs=3 * nodes,
                split_seconds=medians[0],
                fused_seconds=medians[1],
                speedup=medians[0] / medians[1],
                samples_seconds=samples,
                iterations=iterations,
                solution_max_difference=float(np.max(np.abs(solutions[1] - solutions[0]))),
                true_residuals=true_residuals,
            )
        )
    report = dict(arch=args.arch, precision="float64", offline_cache=False, scope=__doc__, cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

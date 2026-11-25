"""Diagonal congruence scaling with a true residual on the unchanged matrix."""

import math

import taichi as ti

from src.linear_solver.BuildTriplet import BuildTriplet
from src.utils.errors import NonfiniteStateError, SparseAssemblyError


@ti.data_oriented
class EquilibratedSolve:
    def __init__(self, matrix):
        if matrix.symmetric or matrix.matrix_symmetric:
            raise ValueError("EquilibratedSolve requires a dense-block nonsymmetric matrix")
        self.matrix = matrix
        self.dim = matrix.dim
        self.scaled_matrix = BuildTriplet(
            dim=self.dim,
            max_pairs_num=1,
            max_nonzeros=matrix.max_nonzeros,
            max_active_nodes=matrix.max_active_nodes,
            symmetric=False,
            solver="BiCGSTAB",
            matrix_symmetric=False,
            device_reduction=True,
            preconditioner_group_size=matrix.preconditioner_group_size,
            preconditioner_group_nodes=matrix.preconditioner_group_nodes,
            reduced_pattern_from=matrix,
        )
        self.scale = ti.Vector.field(self.dim, ti.f64, shape=matrix.max_active_nodes)
        self.scaled_rhs = ti.field(ti.f64, shape=self.dim * matrix.max_active_nodes)
        self.scaled_solution = ti.field(ti.f64, shape=self.dim * matrix.max_active_nodes)

    @ti.kernel
    def _prepare(self, rhs: ti.template(), active_nodes: ti.i32) -> ti.types.vector(3, ti.f64):
        minimum = 1.0e300
        squared = 0.0
        invalid = 0.0
        for block in range(active_nodes):
            for component in ti.static(range(self.dim)):
                value = ti.abs(self.matrix.diag[block][self.dim * component + component])
                factor = 1.0
                if value > 0.0:
                    factor = 1.0 / ti.sqrt(value)
                force = rhs[self.dim * block + component]
                if not (value == value and value < 1.0e300 and force == force and ti.abs(force) < 1.0e300):
                    ti.atomic_max(invalid, 1.0)
                self.scale[block][component] = factor
                self.scaled_rhs[self.dim * block + component] = factor * force
                ti.atomic_min(minimum, factor)
                squared += force * force
        return ti.Vector([minimum, squared, invalid])

    @ti.kernel
    def _scale_matrix(self, active_nodes: ti.i32, count: ti.i32):
        for block in range(active_nodes):
            for row, column in ti.static(ti.ndrange(self.dim, self.dim)):
                index = self.dim * row + column
                self.scaled_matrix.diag[block][index] = self.matrix.diag[block][index] * (
                    self.scale[block][row] * self.scale[block][column]
                )
        for contact in range(count):
            i = self.scaled_matrix.non_diag.tripletI[contact]
            j = self.scaled_matrix.non_diag.tripletJ[contact]
            if 0 <= i < active_nodes and 0 <= j < active_nodes:
                for row, column in ti.static(ti.ndrange(self.dim, self.dim)):
                    index = self.dim * row + column
                    factor = self.scale[i][row] * self.scale[j][column]
                    self.scaled_matrix.non_diag.tripletH[contact][index] = (
                        self.matrix.non_diag.tripletH[contact][index] * factor
                    )

    @ti.kernel
    def _recover(self, rhs: ti.template(), solution: ti.template(), active_nodes: ti.i32):
        for block in range(active_nodes):
            for component in ti.static(range(self.dim)):
                dof = self.dim * block + component
                value = self.scale[block][component] * self.scaled_solution[dof]
                solution[dof] = value
                self.matrix.x[block][component] = value
                self.matrix.rhs[block][component] = rhs[dof]

    def solve(self, rhs, solution, *, active_nodes, tol, maxiter, rel_tol=0.0, return_solution=False):
        if return_solution:
            raise ValueError("EquilibratedSolve writes its solution to the supplied Taichi field")
        tol, rel_tol = float(tol), float(rel_tol)
        if not math.isfinite(tol) or tol < 0.0 or not math.isfinite(rel_tol) or rel_tol < 0.0:
            raise ValueError("EquilibratedSolve tolerances must be finite and non-negative")
        if not 0 < active_nodes <= self.matrix.max_active_nodes:
            raise ValueError("EquilibratedSolve requires a positive active prefix within matrix capacity")
        maxiter = int(maxiter)
        if maxiter < 0:
            raise ValueError("EquilibratedSolve maxiter must be non-negative")
        if int(self.matrix.overflow[0]) != 0:
            raise SparseAssemblyError("Cannot equilibrate a matrix with an overflowed assembly")
        values = self._prepare(rhs, active_nodes)
        minimum, initial_residual = float(values[0]), math.sqrt(max(float(values[1]), 0.0))
        if values[2] != 0.0 or not math.isfinite(initial_residual) or not 0.0 < minimum < 1.0e300:
            raise NonfiniteStateError("Diagonal equilibration encountered a nonfinite matrix diagonal or rhs")
        threshold = max(float(tol), float(rel_tol) * initial_residual)
        count = int(self.scaled_matrix.non_diag.element_pair_num[0])
        self._scale_matrix(active_nodes, count)
        scaled_reference = math.sqrt(
            float(self.scaled_matrix._flat_squared_norm(active_nodes * self.dim, self.scaled_rhs))
        )
        scaled_tolerance = threshold * max(minimum, scaled_reference / max(initial_residual, 1e-300))
        iterations = 0
        for refinement in range(3):
            result = self.scaled_matrix.solve_flat_system(
                self.scaled_rhs,
                self.scaled_solution,
                active_nodes=active_nodes,
                tol=scaled_tolerance,
                maxiter=max(0, maxiter - iterations),
                return_solution=False,
            )
            iterations += result["iterations"]
            self._recover(rhs, solution, active_nodes)
            self.matrix.matvec(
                active_nodes, int(self.matrix.non_diag.element_pair_num[0]), self.matrix.x, self.matrix.Ax
            )
            residual = math.sqrt(max(float(self.matrix._init_bicgstab(active_nodes)), 0.0))
            if residual <= threshold or not result["converged"] or iterations >= maxiter:
                break
            # Correct the scaled target from the actual physical residual, with a guaranteed final bound.
            scaled_tolerance = max(
                threshold * minimum,
                min(scaled_tolerance * 0.1, threshold * result["residual"] / residual * 0.5),
            )
            if refinement == 1:
                scaled_tolerance = threshold * minimum
        result.update(
            equilibrated=True,
            equilibration_refinements=refinement,
            iterations=iterations,
            scaled_residual=result["residual"],
            scaled_convergence_tolerance=result["convergence_tolerance"],
            converged=math.isfinite(residual) and residual <= threshold,
            residual=residual,
            initial_residual=initial_residual,
            convergence_tolerance=threshold,
            solution_inf_norm=float(self.matrix._solution_inf_norm(active_nodes)),
        )
        return result

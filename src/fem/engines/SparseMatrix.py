"""Selectable COO/HashTriplet matrices for implicit FEM."""

import numpy as np
import taichi as ti

from src.linear_solver.BuildTriplet import BuildTriplet, solve_csr_system
from src.linear_solver.CoordinateSparseMatrix import CoordinateSparseMatrix
from src.linear_solver.MatrixFreePBICGSTAB import MatrixFreePBICGSTAB

from src.fem.engines.SparseMatrixKernel import (
    _copy_coo_triplets,
    _zero_constrained_coo_entries,
    _build_coo_diagonal,
    _apply_hash_constraints,
    _load_scalar_field,
    _write_coo_diagonal_tail,
    _add_hash_diagonal,
    _pack_vector_rhs,
    _unpack_vector_solution,
    _set_mass_diagonal,
)


def normalize_assemble_type(value):
    normalized = str(value).replace("_", "").replace("-", "").lower()
    if normalized in ("hash", "hashtriplet", "triplet", "buildtriplet"):
        return "Hash"
    if normalized in ("coo", "coordinatesparse", "coordinatesparsematrix"):
        return "COO"
    raise ValueError(f"Unsupported FEM assemble_type: {value}")


def normalize_linear_solver(value):
    normalized = str(value).replace("_", "").replace("-", "").lower()
    if normalized in ("scipy", "spsolve", "cpu", "direct"):
        return "Scipy"
    if normalized in ("pcg", "taichipcg", "matrixfreepcg"):
        return "PCG"
    if normalized in ("bicgstab", "taichibicgstab", "bicg"):
        return "BiCGSTAB"
    raise ValueError(f"Unsupported FEM linear_solver: {value}")


class FEMTripletContribution:
    """A SciPy-independent scalar triplet contribution."""

    def __init__(self, degree_of_freedom, rows=None, columns=None, values=None):
        self.degree_of_freedom = int(degree_of_freedom)
        self.rows = np.empty(0, dtype=np.int32) if rows is None else np.asarray(rows, dtype=np.int32)
        self.columns = np.empty(0, dtype=np.int32) if columns is None else np.asarray(columns, dtype=np.int32)
        self.values = np.empty(0, dtype=np.float64) if values is None else np.asarray(values, dtype=np.float64)

    @property
    def shape(self):
        return self.degree_of_freedom, self.degree_of_freedom

    def add_triplets(self, rows, columns, values):
        rows = np.asarray(rows, dtype=np.int32).reshape(-1)
        columns = np.asarray(columns, dtype=np.int32).reshape(-1)
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if not (rows.shape == columns.shape == values.shape):
            raise ValueError("FEM sparse triplets must have matching shapes")
        if rows.size:
            self.rows = np.concatenate((self.rows, rows))
            self.columns = np.concatenate((self.columns, columns))
            self.values = np.concatenate((self.values, values))
        return self

    def add_local_matrices(self, stencils, local_matrices):
        stencils = np.asarray(stencils, dtype=np.int32)
        local_matrices = np.asarray(local_matrices, dtype=np.float64)
        if stencils.size == 0:
            return self
        local_dofs = (3 * stencils[:, :, None] + np.arange(3, dtype=np.int32)[None, None, :]).reshape(
            stencils.shape[0], -1
        )
        local_size = local_dofs.shape[1]
        if local_matrices.shape != (stencils.shape[0], local_size, local_size):
            raise ValueError("local FEM matrices do not match their stencils")
        self.add_triplets(
            np.repeat(local_dofs, local_size, axis=1).reshape(-1),
            np.tile(local_dofs, (1, local_size)).reshape(-1),
            local_matrices.reshape(-1),
        )
        return self

    def add_diagonal(self, diagonal):
        diagonal = np.asarray(diagonal, dtype=np.float64).reshape(-1)
        if diagonal.size != self.degree_of_freedom:
            raise ValueError("FEM sparse diagonal has the wrong number of entries")
        indices = np.arange(self.degree_of_freedom, dtype=np.int32)
        return self.add_triplets(indices, indices, diagonal)

    def to_scipy(self):
        from scipy.sparse import coo_matrix

        matrix = coo_matrix((self.values, (self.rows, self.columns)), shape=self.shape).tocsr()
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        return matrix

    def toarray(self):
        return self.to_scipy().toarray()

    def __matmul__(self, vector):
        return self.to_scipy() @ vector


class FEMSparseMatrix(FEMTripletContribution):
    """Deferred FEM matrix assembled as COO or HashTriplet.

    Elasticity, contact and bending project complete kernel-local Hessians and
    immediately scatter scalar COO entries or 3x3 HashTriplet blocks. Inertia
    is appended before constraints and the linear solve are applied.
    """

    def __init__(
        self,
        degree_of_freedom,
        *,
        assemble_type="Hash",
        linear_solver="PCG",
        linear_solver_tolerance=1.0e-10,
        linear_solver_relative_tolerance=0.0,
        linear_solver_max_iters=500,
        base_assembler=None,
    ):
        super().__init__(degree_of_freedom)
        if self.degree_of_freedom % 3 != 0:
            raise ValueError("FEM sparse systems require three displacement components per node")
        self.assemble_type = normalize_assemble_type(assemble_type)
        self.linear_solver = normalize_linear_solver(linear_solver)
        self.linear_solver_tolerance = float(linear_solver_tolerance)
        self.linear_solver_relative_tolerance = float(linear_solver_relative_tolerance)
        self.linear_solver_max_iters = int(linear_solver_max_iters)
        if not np.isfinite(self.linear_solver_tolerance) or self.linear_solver_tolerance < 0.0:
            raise ValueError("FEM linear_solver_tolerance must be finite and non-negative")
        if not np.isfinite(self.linear_solver_relative_tolerance) or self.linear_solver_relative_tolerance < 0.0:
            raise ValueError("FEM linear_solver_relative_tolerance must be finite and non-negative")
        if self.linear_solver_tolerance + self.linear_solver_relative_tolerance <= 0.0:
            raise ValueError("FEM linear solver tolerances cannot both be zero")
        if self.linear_solver_max_iters <= 0:
            raise ValueError("FEM linear_solver_max_iters must be positive")
        self.base_assembler = base_assembler
        self.node_count = self.degree_of_freedom // 3
        self.device_contributions = []
        self.additional_diagonal = ti.field(
            dtype=ti.lang.impl.current_cfg().default_fp,
            shape=self.degree_of_freedom,
        )
        self.rhs_field = ti.field(
            dtype=ti.lang.impl.current_cfg().default_fp,
            shape=self.degree_of_freedom,
        )
        self.solution_field = ti.field(
            dtype=ti.lang.impl.current_cfg().default_fp,
            shape=self.degree_of_freedom,
        )
        self.coo_diagonal = ti.field(
            dtype=ti.lang.impl.current_cfg().default_fp,
            shape=self.degree_of_freedom,
        )
        self.coo_matrix = None
        self.coo_bicgstab = (
            MatrixFreePBICGSTAB(self.degree_of_freedom)
            if self.assemble_type == "COO" and self.linear_solver == "PCG"
            else None
        )
        self.coo_active_count = 0
        self.hash_matrix = None
        self.additional_diagonal.fill(0.0)

    def set_mass_diagonal(self, mass, factor):
        _set_mass_diagonal(mass, float(factor), self.additional_diagonal)

    def clear_additional_diagonal(self):
        self.additional_diagonal.fill(0.0)

    def add_device_contribution(self, contribution):
        if contribution is not None:
            self.device_contributions.append(contribution)
        return self

    def reset_device_assembly(self):
        self.device_contributions.clear()
        self.clear_additional_diagonal()

    @property
    def device_entry_count(self):
        return sum(int(value.stiffness_entry_count_device()) for value in self.device_contributions)

    @property
    def device_block_pair_count(self):
        return sum(int(value.stiffness_block_pair_count_device()) for value in self.device_contributions)

    @property
    def base_entry_count(self):
        if self.base_assembler is None:
            return 0
        return int(self.base_assembler.stiffness_entry_count)

    def add_contribution(self, contribution):
        if contribution is None:
            return self
        if isinstance(contribution, FEMTripletContribution):
            return self.add_triplets(contribution.rows, contribution.columns, contribution.values)
        matrix = contribution.tocoo()
        return self.add_triplets(matrix.row, matrix.col, matrix.data)

    def diagonal(self):
        return np.asarray(self.to_scipy().diagonal(), dtype=np.float64)

    def _constraint_mask(self, constrained_dofs):
        mask = np.zeros(self.degree_of_freedom, dtype=np.int32)
        mask[np.asarray(constrained_dofs, dtype=np.int64)] = 1
        return mask

    def _coo_matrix(self, constrained, diagonal_shift):
        extra_count = self.values.size
        device_count = self.device_entry_count
        # Shift entries are zero on constrained rows; identity entries live in
        # a separate tail so zeroing physical constrained entries cannot turn
        # every duplicate diagonal into an identity.
        physical_count = self.base_entry_count + device_count + extra_count + self.degree_of_freedom
        total_count = physical_count + self.degree_of_freedom
        matrix = self.coo_matrix
        if matrix is None or matrix.capacity < total_count:
            matrix = CoordinateSparseMatrix(
                1 << (total_count - 1).bit_length(),
                self.degree_of_freedom,
                preconditioned=True,
                symmetry=self.linear_solver != "BiCGSTAB",
                linear_solver=self.linear_solver != "Scipy",
            )
            self.coo_matrix = matrix
        else:
            matrix.reset()
        offset = 0
        if self.base_assembler is not None:
            self.base_assembler.scatter_stiffness_to_coo(matrix, offset)
            offset += self.base_entry_count
        for contribution in self.device_contributions:
            contribution.scatter_stiffness_to_coo(matrix, offset)
            offset += int(contribution.stiffness_entry_count_device())
        if extra_count:
            _copy_coo_triplets(
                offset,
                np.ascontiguousarray(self.rows),
                np.ascontiguousarray(self.columns),
                np.ascontiguousarray(self.values),
                matrix.rows,
                matrix.cols,
                matrix.data,
            )
            offset += extra_count
        _zero_constrained_coo_entries(physical_count, constrained, matrix.rows, matrix.cols, matrix.data)
        _write_coo_diagonal_tail(
            offset,
            float(diagonal_shift),
            constrained,
            self.additional_diagonal,
            matrix.rows,
            matrix.cols,
            matrix.data,
        )
        matrix.linear_operator.update_nnz(total_count)
        self.coo_active_count = total_count
        return matrix, total_count

    def _hash_matrix(self, constrained, diagonal_shift):
        base_pairs = 0
        if self.base_assembler is not None:
            base_pairs = int(self.base_assembler.stiffness_block_pair_count)
        extra_off_diagonal = int(np.count_nonzero(self.rows // 3 != self.columns // 3))
        capacity = max(
            1,
            base_pairs + self.device_block_pair_count + extra_off_diagonal,
        )
        # This matrix canonicalizes a complete symmetric input to one global
        # block triangle before reduction.  Body/device assemblers emit both
        # orientations, so only half of those directed raw block pairs can be
        # distinct after canonicalization.  Host scalar extras are kept at
        # their full count because callers may supply upper-only entries.
        base_unique_directed = int(
            getattr(
                self.base_assembler,
                "stiffness_unique_block_pair_count",
                base_pairs,
            )
            if self.base_assembler is not None
            else 0
        )
        reduced_capacity = max(
            1,
            min(
                capacity,
                (base_unique_directed + 1) // 2 + (self.device_block_pair_count + 1) // 2 + extra_off_diagonal,
            ),
        )
        matrix = self.hash_matrix
        if (
            matrix is None
            or int(matrix.non_diag.max_pairs_num) < capacity
            or int(matrix.max_nonzeros) < reduced_capacity
        ):
            matrix = BuildTriplet(
                dim=3,
                max_pairs_num=1 << (capacity - 1).bit_length(),
                max_nonzeros=1 << (reduced_capacity - 1).bit_length(),
                max_active_nodes=self.node_count,
                symmetric=False,
                solver="PCG" if self.linear_solver == "PCG" else "BiCGSTAB",
                matrix_symmetric=True,
                full_symmetric_input=True,
                device_reduction=True,
            )
            self.hash_matrix = matrix
        matrix.reset_system()
        if self.base_assembler is not None:
            self.base_assembler.scatter_stiffness_to_hash(matrix)
        for contribution in self.device_contributions:
            contribution.scatter_stiffness_to_hash(matrix)
        if self.values.size:
            matrix.assemble_scalar_triplets(self.rows, self.columns, self.values)
        _add_hash_diagonal(
            matrix,
            float(diagonal_shift),
            constrained,
            self.additional_diagonal,
        )
        matrix.canonicalize_full_symmetric_input()
        _apply_hash_constraints(matrix, constrained)
        matrix.finalize_taichi_assembly()
        return matrix

    def _materialize(
        self,
        constrained_dofs=(),
        diagonal_shift=0.0,
        constrained_field=None,
    ):
        if constrained_field is None:
            constrained_values = self._constraint_mask(constrained_dofs)
            constrained = ti.field(dtype=ti.i32, shape=self.degree_of_freedom)
            constrained.from_numpy(constrained_values)
        else:
            constrained = constrained_field
        if self.assemble_type == "COO":
            return self._coo_matrix(constrained, diagonal_shift)[0], constrained
        return self._hash_matrix(constrained, diagonal_shift), constrained

    def to_scipy(self):
        matrix, _ = self._materialize()
        if self.assemble_type == "COO":
            return matrix._to_scipy().tocsr()
        return matrix.to_scipy(self.node_count).tocsr()

    def solve(self, rhs, constrained_dofs=(), diagonal_shift=0.0):
        rhs = np.asarray(rhs, dtype=np.float64).reshape(-1).copy()
        matrix, constrained = self._materialize(constrained_dofs, diagonal_shift)
        constrained_values = constrained.to_numpy()
        rhs[constrained_values != 0] = 0.0
        if self.linear_solver == "Scipy":
            sparse = matrix._to_scipy().tocsr() if self.assemble_type == "COO" else matrix.to_scipy(self.node_count)
            return solve_csr_system(rhs, sparse)

        _load_scalar_field(np.ascontiguousarray(rhs), self.rhs_field)
        self.solution_field.fill(0.0)
        if self.assemble_type == "COO":
            _build_coo_diagonal(self.coo_active_count, matrix.rows, matrix.cols, matrix.data, self.coo_diagonal)
            converged = matrix.solve(
                self.rhs_field,
                self.solution_field,
                self.coo_diagonal,
                tol=self.linear_solver_tolerance,
                maxiter=self.linear_solver_max_iters,
                rel_tol=self.linear_solver_relative_tolerance,
            )
            solver = matrix.linear_solver
            residual = float(solver.last_residual)
            iterations = int(solver.last_iterations)
        else:
            result = matrix.solve_flat_system(
                self.rhs_field,
                self.solution_field,
                active_nodes=self.node_count,
                tol=self.linear_solver_tolerance,
                maxiter=self.linear_solver_max_iters,
                return_solution=False,
                rel_tol=self.linear_solver_relative_tolerance,
            )
            converged = result["converged"]
            residual = float(result["residual"])
            iterations = int(result["iterations"])
        if not converged:
            raise RuntimeError(
                f"FEM {self.assemble_type}/{self.linear_solver} solve did not converge: "
                f"residual={residual:.6e}, iterations={iterations}"
            )
        return self.solution_field.to_numpy()

    def solve_device(
        self,
        residual,
        direction,
        constrained,
        diagonal_shift=0.0,
        fallback_to_bicgstab=False,
    ):
        """Solve ``K direction = -residual`` without vector downloads.

        Scipy is the sole exception: selecting it explicitly converts the
        assembled matrix and right-hand side on the CPU, then uploads the
        solution. COO/PCG and Hash/PCG/BiCGSTAB retain full vectors in Taichi.
        """
        _pack_vector_rhs(residual, constrained, self.rhs_field)
        self.solution_field.fill(0.0)
        matrix, _ = self._materialize(
            diagonal_shift=diagonal_shift,
            constrained_field=constrained,
        )
        if self.linear_solver == "Scipy":
            sparse = matrix._to_scipy().tocsr() if self.assemble_type == "COO" else matrix.to_scipy(self.node_count)
            solution = solve_csr_system(self.rhs_field.to_numpy(), sparse)
            self.solution_field.from_numpy(np.ascontiguousarray(solution, dtype=np.float64))
            result = {"converged": True, "residual": 0.0, "iterations": 1}
        elif self.assemble_type == "COO":
            _build_coo_diagonal(
                self.coo_active_count,
                matrix.rows,
                matrix.cols,
                matrix.data,
                self.coo_diagonal,
            )
            converged = matrix.solve(
                self.rhs_field,
                self.solution_field,
                self.coo_diagonal,
                tol=self.linear_solver_tolerance,
                maxiter=self.linear_solver_max_iters,
                rel_tol=self.linear_solver_relative_tolerance,
            )
            result = {
                "converged": bool(converged),
                "initial_residual": float(matrix.linear_solver.last_initial_residual),
                "residual": float(matrix.linear_solver.last_residual),
                "iterations": int(matrix.linear_solver.last_iterations),
            }
            result["convergence_tolerance"] = max(
                self.linear_solver_tolerance,
                self.linear_solver_relative_tolerance * result["initial_residual"],
            )
            if fallback_to_bicgstab and not converged and self.linear_solver == "PCG":
                primary_result = result
                primary_solver = matrix.linear_solver
                matrix.linear_solver = self.coo_bicgstab
                self.solution_field.fill(0.0)
                try:
                    converged = matrix.solve(
                        self.rhs_field,
                        self.solution_field,
                        self.coo_diagonal,
                        tol=self.linear_solver_tolerance,
                        maxiter=self.linear_solver_max_iters,
                        rel_tol=self.linear_solver_relative_tolerance,
                    )
                finally:
                    matrix.linear_solver = primary_solver
                result = {
                    "converged": bool(converged),
                    "initial_residual": float(self.coo_bicgstab.last_initial_residual),
                    "residual": float(self.coo_bicgstab.last_residual),
                    "iterations": int(self.coo_bicgstab.last_iterations),
                    "fallback_from": "PCG",
                    "primary_iterations": int(primary_result["iterations"]),
                    "primary_residual": float(primary_result["residual"]),
                }
                result["convergence_tolerance"] = max(
                    self.linear_solver_tolerance,
                    self.linear_solver_relative_tolerance * result["initial_residual"],
                )
        else:
            result = matrix.solve_flat_system(
                self.rhs_field,
                self.solution_field,
                active_nodes=self.node_count,
                tol=self.linear_solver_tolerance,
                maxiter=self.linear_solver_max_iters,
                return_solution=False,
                rel_tol=self.linear_solver_relative_tolerance,
                fallback_to_bicgstab=bool(fallback_to_bicgstab),
            )
        if not result["converged"]:
            raise RuntimeError(
                f"FEM {self.assemble_type}/{self.linear_solver} solve did not "
                f"converge: initial_residual={float(result.get('initial_residual', float('nan'))):.6e}, "
                f"residual={float(result['residual']):.6e}, "
                f"target={float(result.get('convergence_tolerance', float('nan'))):.6e}, "
                f"iterations={int(result['iterations'])}"
            )
        _unpack_vector_solution(self.solution_field, constrained, direction)
        return result


__all__ = [
    "FEMSparseMatrix",
    "FEMTripletContribution",
    "normalize_assemble_type",
    "normalize_linear_solver",
]

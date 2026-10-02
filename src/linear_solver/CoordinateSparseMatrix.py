import taichi as ti
import numpy as np

from src.linear_solver.MatrixFreeCG import MatrixFreeCG
from src.linear_solver.MatrixFreeBICGSTAB import MatrixFreeBICGSTAB
from src.linear_solver.MatrixFreePCG import MatrixFreePCG
from src.linear_solver.MatrixFreePBICGSTAB import MatrixFreePBICGSTAB
from src.linear_solver.HashReduction import HashReduction
from src.linear_solver.ScipyKrylov import checked_scipy_krylov
from src.utils.constants import WARP_SZ, BLOCK_SZ
from src.utils.BitFunction import ballot, clz, brev
from src.utils.sorting.RadixSort import RadixSort


from src.linear_solver.CoordinateSparseMatrixKernel import (
    set_from_triplets,
    compute_Ap,
    _copy_triplets_to_hash_reducer_impl,
)


class CoordinateSparseMatrix(object):
    def __init__(
        self, nonzeros, degree_of_freedom, is_sparse=False, preconditioned=True, symmetry=True, linear_solver=True
    ) -> None:
        self.coo_matrix = None
        self.hash_reducer = None
        self.is_sparse = is_sparse
        self.preconditioned = preconditioned
        self.symmetry = symmetry
        self.nonzeros = int(nonzeros)
        self.dofs = int(degree_of_freedom)
        if self.dofs <= 0:
            raise ValueError("CoordinateSparseMatrix degree_of_freedom must be positive")
        if self.dofs > int(np.iinfo(np.int32).max):
            raise ValueError("CoordinateSparseMatrix degree_of_freedom exceeds the " "int32 row/column index range")
        # self.radix_sort = RadixSort(initial_size, ti.f64)

        self.linear_operator = COOOperator()
        if is_sparse:
            self._sparse_field_build()
            self.reset = self.sparse_reset
        else:
            self._field_build()
            self.reset = self.csr_reset

        # SciPy is loaded only when the user explicitly calls ``spsolve``.
        # Device Krylov backends therefore have no host-solver dependency in
        # their construction or time-stepping path.
        self.cpu_solver_name = "spsolve" if degree_of_freedom < 10000 else ("cg" if symmetry else "bicgstab")

        if linear_solver:
            if preconditioned:
                if symmetry:
                    self.linear_solver = MatrixFreePCG(degree_of_freedom)
                else:
                    self.linear_solver = MatrixFreePBICGSTAB(degree_of_freedom)
                self.solve = self.solve2
            else:
                if symmetry:
                    self.linear_solver = MatrixFreeCG(degree_of_freedom)
                else:
                    self.linear_solver = MatrixFreeBICGSTAB(degree_of_freedom)
                self.solve = self.solve1

    @property
    def capacity(self):
        """Allocated scalar-triplet capacity.

        Coupled device assemblers use this value for explicit overflow checks.
        Keep it tied to ``nonzeros`` so a matrix rebuilt by ``_from_scipy``
        reports its new allocation rather than a stale constructor value.
        """
        return self.nonzeros

    def _field_build(self, nonzeros=None):
        if nonzeros is not None:
            self.nonzeros = int(nonzeros)
        if self.nonzeros <= 0:
            raise ValueError("CoordinateSparseMatrix capacity must be positive")
        int32_max = int(np.iinfo(np.int32).max)
        if self.nonzeros > int32_max:
            runtime = ti.lang.impl.get_runtime()
            value_bytes = 8
            if runtime.prog is not None:
                value_bytes = 8 if runtime.prog.config().default_fp == ti.f64 else 4
            estimated_gib = (8 + value_bytes) * self.nonzeros / float(1 << 30)
            raise ValueError(
                "CoordinateSparseMatrix scalar-triplet capacity "
                f"{self.nonzeros} exceeds Taichi's dense SNode int32 limit "
                f"{int32_max} (rows/cols/data alone would require about "
                f"{estimated_gib:.1f} GiB); use HashTriplet or reduce the "
                "assembly capacity"
            )
        self.rows = ti.field(int)
        self.cols = ti.field(int)
        self.data = ti.field(float)
        builder = ti.FieldsBuilder()
        builder.dense(ti.i, self.nonzeros).place(self.rows, self.cols, self.data)
        self.builder = builder.finalize()
        self.linear_operator.link_ptrs(self.rows, self.cols, self.data)
        self.linear_operator.update_active_dofs(self.dofs)
        self.linear_operator.update_nnz(self.nonzeros)

    def _sparse_field_build(self):
        self.sparse_matrix = ti.field(float)
        self.grandparent = ti.root.pointer(ti.ij, (self.dofs, self.dofs))
        self.grandparent.place(self.sparse_matrix)

    def clear(self):
        if self.is_sparse:
            self.sparse_reset()
        else:
            self.builder.destroy()

    def sparse_reset(self):
        self.grandparent.deactivate_all()

    def csr_reset(self):
        self.rows.fill(0)
        self.cols.fill(0)
        self.data.fill(0)

    def clean(self, sparse_matrix):
        sparse_matrix.data[np.abs(sparse_matrix.data) < 2.2204460492503131e-15] = 0
        sparse_matrix.eliminate_zeros()
        return sparse_matrix

    def _to_text(self, path=None):
        import os, sys

        if path is None:
            path = os.path.dirname(os.path.abspath(sys.argv[0]))
        if not os.path.exists(path):
            os.makedirs(path)
        dense_matrix = self._to_numpy()
        return np.savetxt(path + "/kmatrix.txt", dense_matrix, fmt="%.6f", delimiter=" ")

    def _to_numpy(self):
        if self.is_sparse:
            return self.sparse_matrix.to_numpy()
        else:
            return self._to_scipy().toarray()

    def _to_scipy(self):
        from scipy.sparse import coo_matrix

        if self.is_sparse:
            dense_matrix = self.sparse_matrix.to_numpy()
            self.coo_matrix = coo_matrix(dense_matrix)
        else:
            rows = self.rows.to_numpy()
            cols = self.cols.to_numpy()
            data = self.data.to_numpy()
            self.coo_matrix = coo_matrix((data, (rows, cols)), shape=(self.dofs, self.dofs))
        self.clean(self.coo_matrix)
        return self.coo_matrix

    def _to_scipy_hash(self):
        from scipy.sparse import coo_matrix

        if self.is_sparse:
            return self._to_scipy()
        if self.hash_reducer is None:
            self.hash_reducer = HashReduction(self.nonzeros, dim=1, max_nnz=self.nonzeros, hessian_size=1)
        self._copy_triplets_to_hash_reducer()
        self.hash_reducer.go(self.nonzeros)
        rows, cols, data = self.hash_reducer.get_reduced_triplets_numpy()
        self.coo_matrix = coo_matrix((data[:, 0], (rows, cols)), shape=(self.dofs, self.dofs))
        self.clean(self.coo_matrix)
        return self.coo_matrix

    def _copy_triplets_to_hash_reducer(self):
        _copy_triplets_to_hash_reducer_impl(
            self.nonzeros,
            self.rows,
            self.cols,
            self.data,
            self.hash_reducer.blockI,
            self.hash_reducer.blockJ,
            self.hash_reducer.blockH,
        )

    def _from_scipy(self, coo_matrixes):
        if (
            coo_matrixes.col.shape[0] > self.cols.shape[0]
            or coo_matrixes.row.shape[0] > self.rows.shape[0]
            or coo_matrixes.data.shape[0] > self.data.shape[0]
        ):
            self.clear()
            self._field_build(coo_matrixes.row.shape[0])
        self.coo_matrix = coo_matrixes

        if self.is_sparse:
            pass
        else:
            self.rows.from_numpy(coo_matrixes.row)
            self.cols.from_numpy(coo_matrixes.col)
            self.data.from_numpy(coo_matrixes.data)

    def _from_numpy(self, K: np.ndarray):
        from scipy.sparse import csr_matrix

        coo_matrixes = csr_matrix(K).tocoo()
        self._from_scipy(coo_matrixes)

    def spsolve(self, rhs, sparse_matrixes=None):
        from scipy.sparse import coo_matrix
        import scipy.sparse.linalg as sl

        if isinstance(rhs, ti.ScalarField):
            rhs = rhs.to_numpy()
        if isinstance(sparse_matrixes, coo_matrix):
            sparse_matrixes = sparse_matrixes.tocsr()
        if sparse_matrixes is None and self.coo_matrix is None:
            self.coo_matrix = self._to_scipy()
        sparse_matrixes = self.coo_matrix.tocsr() if sparse_matrixes is None else sparse_matrixes
        assert not sparse_matrixes is None
        self.clean(sparse_matrixes)
        non_zero_rows = sparse_matrixes.getnnz(axis=1) != 0
        non_zero_cols = sparse_matrixes.getnnz(axis=0) != 0
        csr_matrixes_reduced = sparse_matrixes[non_zero_rows, :][:, non_zero_cols]
        rhs_reduced = rhs[non_zero_rows]
        if self.cpu_solver_name == "spsolve":
            x_reduced = sl.spsolve(csr_matrixes_reduced, rhs_reduced)
        elif self.cpu_solver_name == "cg":
            x_reduced = checked_scipy_krylov(sl.cg, csr_matrixes_reduced, rhs_reduced, solver_name="SciPy CG")
        else:
            x_reduced = checked_scipy_krylov(
                sl.bicgstab, csr_matrixes_reduced, rhs_reduced, solver_name="SciPy BiCGSTAB"
            )
        x_full = np.zeros(sparse_matrixes.shape[1])
        x_full[non_zero_cols] = x_reduced
        return x_full

    def solve1(self, b, x, tol=1e-6, maxiter=5000):
        # set_from_triplets(self.bsr_matrix, self.rows, self.cols, self.data)
        return self.linear_solver.solve(
            self.linear_operator,
            b,
            x,
            self.linear_operator.active_dofs,
            tol,
            maxiter,
        )

    def solve2(self, b, x, diagA, tol=1e-6, maxiter=5000, rel_tol=0.0):
        # set_from_triplets(self.bsr_matrix, self.rows, self.cols, self.data)
        return self.linear_solver.solve(
            self.linear_operator,
            b,
            x,
            diagA,
            self.linear_operator.active_dofs,
            tol=tol,
            maxiter=maxiter,
            rel_tol=rel_tol,
        )


class COOOperator(object):
    def __init__(self):
        pass

    def link_ptrs(self, *args):
        self.row = args[0]
        self.col = args[1]
        self.value = args[2]

    def update_active_dofs(self, active_dofs):
        self.active_dofs = int(active_dofs)

    def update_nnz(self, nnz):
        self.nnz = int(nnz)

    def matvec(self, x, Ax):
        Ax.fill(0)
        compute_Ap(self.nnz, self.row, self.col, self.value, x, Ax)

import taichi as ti
import numpy as np

# import pypardiso

from src.linear_solver.MatrixFreeCG import MatrixFreeCG
from src.linear_solver.MatrixFreeBICGSTAB import MatrixFreeBICGSTAB
from src.linear_solver.MatrixFreePCG import MatrixFreePCG
from src.linear_solver.MatrixFreePBICGSTAB import MatrixFreePBICGSTAB
from src.linear_solver.ScipyKrylov import checked_scipy_krylov
from src.utils.constants import WARP_SZ
from src.utils.linalg import round32
from src.utils.TypeDefination import u1
from src.utils.WarpReduce import warp_scan_up_f32

EPSILON = 2.2204460492503131e-15


from src.linear_solver.CompressedSparseRowKernel import (
    compute_Ap,
    compute_Ap_warp_reduce,
    compute_Ap_shared_reduce,
)


class CompressedSparseRow(object):
    def __init__(self, nonzeros, degree_of_freedom, is_sparse=False, preconditioned=True, symmetry=True) -> None:
        self.csr_matrix = None
        self.is_sparse = is_sparse
        self.preconditioned = preconditioned
        self.symmetry = symmetry

        self.linear_operator = CSROperator()
        if is_sparse:
            self._sparse_field_build(nonzeros, degree_of_freedom)
            self.reset = self.sparse_reset
        else:
            self._field_build(nonzeros, degree_of_freedom)
            self.reset = self.csr_reset

        # SciPy is an opt-in linear solve backend, not part of the Taichi
        # matrix assembly or Krylov runtime.
        self.cpu_solver_name = "spsolve" if nonzeros < 10000 else ("cg" if symmetry else "bicgstab")

        if preconditioned:
            if symmetry:
                self.linear_solver = MatrixFreePCG(nonzeros)
            else:
                self.linear_solver = MatrixFreePBICGSTAB(nonzeros)
            self.solve = self.solve2
        else:
            if symmetry:
                self.linear_solver = MatrixFreeCG(nonzeros)
            else:
                self.linear_solver = MatrixFreeBICGSTAB(nonzeros)
            self.solve = self.solve1

    def _field_build(self, nonzeros, degree_of_freedom):
        self.offsets = ti.field(int)
        builder = ti.FieldsBuilder()
        builder.dense(ti.i, degree_of_freedom + 1).place(self.offsets)
        self.values = ti.field(float)
        self.indices = ti.field(int)
        builder.dense(ti.i, nonzeros).place(self.values, self.indices)
        self.grid_active = ti.field(u1)
        ti.root.dense(ti.i, round32(degree_of_freedom) // 32).quant_array(ti.i, dimensions=32, max_num_bits=32).place(
            self.grid_active
        )
        self.builder = builder.finalize()
        self.linear_operator.link_ptrs(self.offsets, self.indices, self.values, degree_of_freedom)

    def _sparse_field_build(self, nonzeros, degree_of_freedom):
        self.sparse_matrix = ti.field(float)
        self.grandparent = ti.root.pointer(ti.ij, (degree_of_freedom, degree_of_freedom))
        self.grandparent.place(self.sparse_matrix)

    def clear(self):
        if self.is_sparse:
            self.sparse_reset()
        else:
            self.builder.destroy()

    def sparse_reset(self):
        self.grandparent.deactivate_all()

    def csr_reset(self):
        self.offsets.fill(0)
        self.indices.fill(0)
        self.grid_active.fill(0)

    def csr_clean(self, csr_matrix):
        csr_matrix.data[np.abs(csr_matrix.data) < EPSILON] = 0
        csr_matrix.eliminate_zeros()
        return csr_matrix

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
        from scipy.sparse import csr_matrix

        if self.is_sparse:
            dense_matrix = self.sparse_matrix.to_numpy()
            self.csr_matrix = csr_matrix(dense_matrix)
        else:
            indptr = self.offsets.to_numpy()
            indices = self.indices.to_numpy()
            data = self.values.to_numpy()
            self.csr_matrix = csr_matrix((data, indices, indptr))
        self.csr_clean(self.csr_matrix)
        return self.csr_matrix

    def _from_scipy(self, csr_matrixes):
        if (
            csr_matrixes.indptr.shape[0] > self.offsets.shape[0]
            or csr_matrixes.indices.shape[0] > self.indices.shape[0]
            or csr_matrixes.data.shape[0] > self.values.shape[0]
        ):
            self.clear()
            self._field_build(csr_matrixes.data.shape[0], csr_matrixes.indptr.shape[0])
        self.csr_matrix = csr_matrixes

        if self.is_sparse:
            pass
        else:
            self.offsets.from_numpy(csr_matrixes.indptr)
            self.indices.from_numpy(csr_matrixes.indices)
            self.values.from_numpy(csr_matrixes.data)

    def _from_numpy(self, K: np.ndarray):
        from scipy.sparse import csr_matrix

        csr_matrixes = csr_matrix(K)
        self._from_scipy(csr_matrixes)

    def spsolve(self, rhs, csr_matrixes=None):
        import scipy.sparse.linalg as sl

        if isinstance(rhs, ti.ScalarField):
            rhs = rhs.to_numpy()
        if csr_matrixes is None and self.csr_matrix is None:
            self.csr_matrix = self._to_scipy()
        csr_matrixes = self.csr_matrix if csr_matrixes is None else csr_matrixes
        self.csr_clean(csr_matrixes)
        assert not csr_matrixes is None
        non_zero_rows = csr_matrixes.getnnz(axis=1) != 0
        non_zero_cols = csr_matrixes.getnnz(axis=0) != 0
        csr_matrixes_reduced = csr_matrixes[non_zero_rows, :][:, non_zero_cols]
        rhs_reduced = rhs[non_zero_rows]
        if self.cpu_solver_name == "spsolve":
            x_reduced = sl.spsolve(csr_matrixes_reduced, rhs_reduced)
        elif self.cpu_solver_name == "cg":
            x_reduced = checked_scipy_krylov(sl.cg, csr_matrixes_reduced, rhs_reduced, solver_name="SciPy CG")
        else:
            x_reduced = checked_scipy_krylov(
                sl.bicgstab, csr_matrixes_reduced, rhs_reduced, solver_name="SciPy BiCGSTAB"
            )
        x_full = np.zeros(csr_matrixes.shape[1])
        x_full[non_zero_cols] = x_reduced
        return x_full

    def solve1(self, b, x, tol=1e-6, maxiter=5000):
        return self.linear_solver.solve(
            self.linear_operator,
            b,
            x,
            self.linear_operator.active_dofs,
            tol,
            maxiter,
        )

    def solve2(self, b, x, diagA, tol=1e-6, maxiter=5000):
        return self.linear_solver.solve(
            self.linear_operator,
            b,
            x,
            diagA,
            self.linear_operator.active_dofs,
            tol,
            maxiter,
        )


class CSROperator(object):
    def __init__(self):
        pass

    def link_ptrs(self, *args):
        self.offset = args[0]
        self.indices = args[1]
        self.data = args[2]
        self.active_dofs = args[3]

    def update_active_dofs(self, active_dofs):
        self.active_dofs = active_dofs

    def matvec(self, x, Ax):
        Ax.fill(0)
        compute_Ap(self.active_dofs, self.data, self.indices, self.offset, x, Ax)

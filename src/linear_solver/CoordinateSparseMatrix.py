import taichi as ti
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import scipy.sparse.linalg as sl

from src.linear_solver.MatrixFreeCG import MatrixFreeCG
from src.linear_solver.MatrixFreeBICGSTAB import MatrixFreeBICGSTAB
from src.linear_solver.MatrixFreePCG import MatrixFreePCG
from src.linear_solver.MatrixFreePBICGSTAB import MatrixFreePBICGSTAB
from src.utils.constants import WARP_SZ, BLOCK_SZ
from src.utils.BitFunction import ballot, clz, brev
from src.utils.sorting.RadixSort import RadixSort


EPSILON = 2.2204460492503131e-15
class CoordinateSparseMatrix(object):
    def __init__(self, initial_size, nonzeros, degree_of_freedom, is_sparse=False, preconditioned=True, symmetry=True) -> None:
        self.coo_matrix = None
        self.is_sparse = is_sparse
        self.preconditioned = preconditioned
        self.symmetry = symmetry
        self.initial_size = initial_size
        self.bsr_size = nonzeros
        self.dofs = degree_of_freedom
        self.radix_sort = RadixSort(initial_size, ti.f64)

        self.linear_operator = COOOperator()
        if is_sparse:
            self._sparse_field_build()
            self.reset = self.sparse_reset
        else:
            self._field_build()
            self.reset = self.csr_reset

        self.cpu_solver = None
        if nonzeros < 10000:
            self.cpu_solver = sl.spsolve
        else:
            if symmetry:
                def cg(A, b): return sl.cg(A, b)[0]
                self.cpu_solver = cg
            else:
                def bicgstab(A, b): return sl.bicgstab(A, b)[0]
                self.cpu_solver = bicgstab

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

    def _field_build(self):
        self.rows = ti.field(int)
        self.cols = ti.field(int)
        self.data = ti.field(float)
        self.bsr_matrix = ti.field(float)
        builder = ti.FieldsBuilder()
        builder.dense(ti.i, self.initial_size).place(self.rows, self.cols, self.data)
        builder.dense(ti.ij, (self.bsr_size, 3)).place(self.bsr_matrix)
        self.builder = builder.finalize()
        self.linear_operator.link_ptrs(self.bsr_matrix)

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
        sparse_matrix.data[np.abs(sparse_matrix.data) < EPSILON] = 0
        sparse_matrix.eliminate_zeros()
        return sparse_matrix
    
    def _to_text(self, path=None):
        import os, sys
        if path is None:
            path = os.path.dirname(os.path.abspath(sys.argv[0]))
        if not os.path.exists(path):
            os.makedirs(path)   
        dense_matrix = self._to_numpy()
        return np.savetxt(path+'/kmatrix.txt', dense_matrix, fmt='%.6f', delimiter=' ')

    def _to_numpy(self):
        if self.is_sparse:
            return self.sparse_matrix.to_numpy()
        else:
            return self._to_scipy().toarray()
    
    def _to_scipy(self):
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
    
    def _from_scipy(self, coo_matrixes: coo_matrix):
        if coo_matrixes.col.shape[0] > self.cols.shape[0] or coo_matrixes.row.shape[0] > self.rows.shape[0] or coo_matrixes.data.shape[0] > self.data.shape[0]:
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
        coo_matrixes = csr_matrix(K).tocoo()
        self._from_scipy(coo_matrixes)

    def spsolve(self, rhs, sparse_matrixes: csr_matrix=None):
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
        x_reduced = self.cpu_solver(csr_matrixes_reduced, rhs_reduced)
        x_full = np.zeros(sparse_matrixes.shape[1])
        x_full[non_zero_cols] = x_reduced
        return x_full
    
    def solve1(self, b, x, tol=1e-6, maxiter=5000):
        #set_from_triplets(self.bsr_matrix, self.rows, self.cols, self.data)
        self.linear_solver.solve(self.linear_operator, b, x, self.linear_operator.active_dofs, tol, maxiter)

    def solve2(self, b, x, diagA, tol=1e-6, maxiter=5000):
        #set_from_triplets(self.bsr_matrix, self.rows, self.cols, self.data)
        self.linear_solver.solve(self.linear_operator, b, x, diagA, self.linear_operator.active_dofs, tol, maxiter)

class COOOperator(object):
    def __init__(self):
        pass

    def link_ptrs(self, *args):
        self.bsr_matrix = args[0]

    def update_active_dofs(self, active_dofs):
        self.active_dofs = active_dofs

    def matvec(self, x, Ax):
        compute_Ap(self.active_dofs, self.bsr_matrix, x, Ax)

@ti.kernel
def set_from_triplets(bsr_matrix: ti.template(), rows: ti.template(), cols: ti.template(), values: ti.template()):
    ti.loop_config(block_dim=BLOCK_SZ)
    for idx in range(rows.shape[0]):
        offset = ti.simt.block.SharedArray((1, ), ti.i32)
        thread_id = idx % BLOCK_SZ
        block_id = idx // BLOCK_SZ
        lane_id = thread_id & 0x1f
        warp_id = thread_id // WARP_SZ

        row_ind = rows[idx]
        col_ind = cols[idx]
        rdata = values[idx]

        mask = ti.simt.warp.active_mask()
        prev_endID1 = ti.simt.warp.shfl_up_i32(mask, row_ind, 1)
        ti.simt.block.sync()

        bBoundary = (lane_id == 0) or (row_ind != prev_endID1)
        mark = ballot(bBoundary)
        mark = brev(mark)
        interval = ti.min(clz(mark << (lane_id + 1)), 31 - lane_id)
        ti.simt.block.sync()

        index = 1
        while index < min(WARP_SZ, 12):
            if interval >= index: 
                pass
            index <<= 1

        if bBoundary == 1:
            pass

@ti.kernel
def compute_Ap(total_row: int, bsr_matrix: ti.template(), p: ti.template(), Ap: ti.template()):
    for i in range(total_row):
        Ap[bsr_matrix[i, 0]] += p[bsr_matrix[i, 1]] * bsr_matrix[i, 2]

@ti.kernel
def compute_Ap_original(rows: ti.template(), cols: ti.template(), values: ti.template(), p: ti.template(), Ap: ti.template()):
    for i in range(rows.shape[0]):
        Ap[rows[i]] += p[cols[i]] * values[i]

if __name__ == '__main__':
    ti.init()
    sparse_matrix = CoordinateSparseMatrix(8, 8, 5, preconditioned=False)
    sparse_matrix.rows.from_numpy(np.array([0, 0, 1, 2, 3, 4, 4, 1]))
    sparse_matrix.cols.from_numpy(np.array([0, 2, 1, 3, 2, 0, 4, 3]))
    sparse_matrix.data.from_numpy(np.array([10, 3, 5, 7, 2, 6, 9, 4], dtype=np.float64))
    b = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    x = sparse_matrix.spsolve(b)

    print("求解结果 x =", x)
import taichi as ti
import numpy as np
# import pypardiso
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as sl

from src.linear_solver.MatrixFreeCG import MatrixFreeCG
from src.linear_solver.MatrixFreeBICGSTAB import MatrixFreeBICGSTAB
from src.linear_solver.MatrixFreePCG import MatrixFreePCG
from src.linear_solver.MatrixFreePBICGSTAB import MatrixFreePBICGSTAB
from src.utils.constants import WARP_SZ
from src.utils.linalg import round32
from src.utils.TypeDefination import u1
from src.utils.WarpReduce import warp_shfl_up_f32


EPSILON = 2.2204460492503131e-15
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

    def _field_build(self, nonzeros, degree_of_freedom):
        self.offsets = ti.field(int)
        builder = ti.FieldsBuilder()
        builder.dense(ti.i, degree_of_freedom + 1).place(self.offsets)
        self.values = ti.field(float)
        self.indices = ti.field(int)
        builder.dense(ti.i, nonzeros).place(self.values, self.indices)
        self.grid_active = ti.field(u1)
        ti.root.dense(ti.i, round32(degree_of_freedom)//32).quant_array(ti.i, dimensions=32, max_num_bits=32).place(self.grid_active)
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
        csr_matrix.data[np.abs(self.csr_matrix.data) < EPSILON] = 0
        csr_matrix.eliminate_zeros()
        return csr_matrix
    
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
            self.csr_matrix = csr_matrix(dense_matrix)
        else:
            indptr = self.offsets.to_numpy()
            indices = self.indices.to_numpy()
            data = self.values.to_numpy()
            self.csr_matrix = csr_matrix((data, indices, indptr))
        self.csr_clean(self.csr_matrix)
        return self.csr_matrix
    
    def _from_scipy(self, csr_matrixes: csr_matrix):
        if csr_matrixes.indptr.shape[0] > self.offsets.shape[0] or csr_matrixes.indices.shape[0] > self.indices.shape[0] or csr_matrixes.data.shape[0] > self.values.shape[0]:
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
        csr_matrixes = csr_matrix(K)
        self._from_scipy(csr_matrixes)

    def spsolve(self, rhs, csr_matrixes: csr_matrix=None):
        if isinstance(rhs, ti.ScalarField):
            rhs = rhs.to_numpy()
        csr_matrixes = self.csr_matrix if csr_matrixes is None else csr_matrixes
        self.csr_clean(csr_matrixes)
        assert not csr_matrixes is None
        non_zero_rows = csr_matrixes.getnnz(axis=1) != 0
        non_zero_cols = csr_matrixes.getnnz(axis=0) != 0
        csr_matrixes_reduced = csr_matrixes[non_zero_rows, :][:, non_zero_cols]
        rhs_reduced = rhs[non_zero_rows]
        x_reduced = self.cpu_solver(csr_matrixes_reduced, rhs_reduced)
        x_full = np.zeros(csr_matrixes.shape[1])
        x_full[non_zero_cols] = x_reduced
        return x_full
    
    def solve1(self, b, x, tol=1e-6, maxiter=5000):
        self.linear_solver.solve(self.linear_operator, b, x, self.linear_operator.active_dofs, tol, maxiter)

    def solve2(self, b, x, diagA, tol=1e-6, maxiter=5000):
        self.linear_solver.solve(self.linear_operator, b, x, diagA, self.linear_operator.active_dofs, tol, maxiter)

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
        compute_Ap(self.active_dofs, self.data, self.indices, self.offset, x, Ax)

@ti.kernel
def compute_Ap(total_row: int, values: ti.template(), indices: ti.template(), offsets: ti.template(), p: ti.template(), Ap: ti.template()):
    for i in range(total_row):
        sums = 0.
        for index in range(offsets[i], offsets[i + 1]):
            sums += p[indices[index]] * values[index]
        Ap[i] = sums

@ti.kernel
def compute_Ap_warp_reduce(total_row: int, values: ti.template(), indices: ti.template(), offsets: ti.template(), p: ti.template(), Ap: ti.template()):
    for thread_id in range(WARP_SZ * total_row):
        warp_id = thread_id // WARP_SZ
        lane_id = thread_id % WARP_SZ

        sums = ti.cast(0., ti.f32)
        if warp_id < total_row:
            index = offsets[warp_id] + lane_id
            while index < offsets[warp_id + 1]:
                sums += p[indices[index]] * values[index]
                index += WARP_SZ

        sums = warp_shfl_up_f32(lane_id, sums)
        ti.simt.block.sync()
        if lane_id == 0 and warp_id < total_row:
            Ap[warp_id] = sums

@ti.kernel
def compute_Ap_shared_reduce(total_row: int, values: ti.template(), indices: ti.template(), offsets: ti.template(), p: ti.template(), Ap: ti.template()):
    ti.loop_config(block_dim=WARP_SZ)
    for thread_id in range(WARP_SZ * total_row):
        warp_id = thread_id // WARP_SZ
        lane_id = thread_id % WARP_SZ
        sdata = ti.simt.block.SharedArray((WARP_SZ, ), ti.f64)

        if warp_id < total_row:
            index = offsets[warp_id] + lane_id
            while index < offsets[warp_id + 1]:
                sdata[warp_id] += p[indices[index]] * values[index]
                index += WARP_SZ
        ti.simt.block.sync()

        temp = int(0.5 * WARP_SZ)
        while temp > 1:
            if lane_id < temp:
                sdata[lane_id] += sdata[lane_id + temp]
                temp = int(0.5 * temp)

        if lane_id == 0 and warp_id < total_row:
            Ap[warp_id] = sdata[0]

import taichi as ti
import numpy as np
import math
from taichi.lang.impl import current_cfg

from src.linear_solver.HashReduction import HashReduction
from src.utils.FieldIO import field_to_numpy_prefix


def solve_csr_system(rhs, sparse_matrixes):
    from scipy.sparse.linalg import spsolve

    if isinstance(rhs, ti.ScalarField):
        rhs = rhs.to_numpy()
    sparse_matrixes = sparse_matrixes.tocsr()
    sparse_matrixes.data[np.abs(sparse_matrixes.data) < 2.2204460492503131e-15] = 0
    sparse_matrixes.eliminate_zeros()
    non_zero_rows = sparse_matrixes.getnnz(axis=1) != 0
    non_zero_cols = sparse_matrixes.getnnz(axis=0) != 0
    csr_matrixes_reduced = sparse_matrixes[non_zero_rows, :][:, non_zero_cols]
    rhs_reduced = rhs[non_zero_rows]
    x_reduced = spsolve(csr_matrixes_reduced, rhs_reduced)
    x_full = np.zeros(sparse_matrixes.shape[1])
    x_full[non_zero_cols] = x_reduced
    return x_full


@ti.func
def _sym_block_matvec(h, x, dim: ti.template()):
    y = ti.Vector.zero(float, dim)
    y[0] = h[0] * x[0]
    if ti.static(dim >= 2):
        y[0] += h[3] * x[1]
        y[1] = h[3] * x[0] + h[1] * x[1]
    if ti.static(dim == 3):
        y[0] += h[5] * x[2]
        y[1] += h[4] * x[2]
        y[2] = h[5] * x[0] + h[4] * x[1] + h[2] * x[2]
    return y


@ti.func
def _dense_block_matvec(h, x, dim: ti.template()):
    y = ti.Vector.zero(float, dim)
    for i in range(dim):
        for j in range(dim):
            y[i] += h[i * dim + j] * x[j]
    return y


@ti.func
def _dense_block_transpose_matvec(h, x, dim: ti.template()):
    y = ti.Vector.zero(float, dim)
    for i in range(dim):
        for j in range(dim):
            y[i] += h[j * dim + i] * x[j]
    return y


@ti.func
def _apply_sym_block_jacobi(h, r, dim: ti.template()):
    """Apply a scale-invariant inverse or a bounded diagonal fallback.

    Determinants carry ``dim`` powers of the physical matrix scale, so an
    absolute determinant threshold incorrectly classifies small, otherwise
    well-conditioned blocks as singular.  Normalize first and accept the
    inverse only when an infinity-norm reciprocal-condition estimate is
    sufficiently large.  The fallback uses the original scale and never
    returns an all-zero preconditioner for a zero block (which would make PCG
    report a false zero preconditioned residual).
    """
    z = r
    scale = 0.0
    finite_input = 1
    for component in ti.static(range(6)):
        value = h[component]
        if not (-ti.math.inf < value and value < ti.math.inf):
            finite_input = 0
        scale = ti.max(scale, ti.abs(value))

    condition_tolerance = 1.0e-12
    determinant_floor = 1.0e-30
    if finite_input != 0 and scale > 0.0 and scale < ti.math.inf:
        # A scale-aware diagonal fallback is also the safe path for a block
        # whose normalized inverse is too ill-conditioned.  Tiny diagonals are
        # replaced by the block scale instead of producing an unbounded inverse.
        for row in ti.static(range(4)):
            if ti.static(row < dim):
                diagonal = h[row]
                denominator = scale
                if ti.abs(diagonal) > condition_tolerance * scale:
                    denominator = diagonal
                value = r[row] / denominator
                if -ti.math.inf < value and value < ti.math.inf:
                    z[row] = value

    if ti.static(dim == 2):
        if finite_input != 0 and scale > 0.0 and scale < ti.math.inf:
            a00 = h[0] / scale
            a11 = h[1] / scale
            a01 = h[3] / scale
            det = a00 * a11 - a01 * a01
            if ti.abs(det) > determinant_floor:
                inverse00 = a11 / det
                inverse01 = -a01 / det
                inverse11 = a00 / det
                matrix_norm = ti.max(
                    ti.abs(a00) + ti.abs(a01),
                    ti.abs(a01) + ti.abs(a11),
                )
                inverse_norm = ti.max(
                    ti.abs(inverse00) + ti.abs(inverse01),
                    ti.abs(inverse01) + ti.abs(inverse11),
                )
                rcond = 0.0
                if matrix_norm > 0.0 and inverse_norm > 0.0:
                    rcond = 1.0 / (matrix_norm * inverse_norm)
                if rcond > condition_tolerance:
                    candidate0 = (inverse00 * r[0] + inverse01 * r[1]) / scale
                    candidate1 = (inverse01 * r[0] + inverse11 * r[1]) / scale
                    if (
                        -ti.math.inf < candidate0
                        and candidate0 < ti.math.inf
                        and -ti.math.inf < candidate1
                        and candidate1 < ti.math.inf
                    ):
                        z[0] = candidate0
                        z[1] = candidate1
    else:
        if finite_input != 0 and scale > 0.0 and scale < ti.math.inf:
            a00, a11, a22 = h[0] / scale, h[1] / scale, h[2] / scale
            a01, a12, a02 = h[3] / scale, h[4] / scale, h[5] / scale
            c00 = a11 * a22 - a12 * a12
            c01 = a02 * a12 - a01 * a22
            c02 = a01 * a12 - a02 * a11
            c11 = a00 * a22 - a02 * a02
            c12 = a01 * a02 - a00 * a12
            c22 = a00 * a11 - a01 * a01
            det = a00 * c00 + a01 * c01 + a02 * c02
            if ti.abs(det) > determinant_floor:
                inverse00, inverse01, inverse02 = c00 / det, c01 / det, c02 / det
                inverse11, inverse12, inverse22 = c11 / det, c12 / det, c22 / det
                matrix_norm = ti.max(
                    ti.abs(a00) + ti.abs(a01) + ti.abs(a02),
                    ti.abs(a01) + ti.abs(a11) + ti.abs(a12),
                    ti.abs(a02) + ti.abs(a12) + ti.abs(a22),
                )
                inverse_norm = ti.max(
                    ti.abs(inverse00) + ti.abs(inverse01) + ti.abs(inverse02),
                    ti.abs(inverse01) + ti.abs(inverse11) + ti.abs(inverse12),
                    ti.abs(inverse02) + ti.abs(inverse12) + ti.abs(inverse22),
                )
                rcond = 0.0
                if matrix_norm > 0.0 and inverse_norm > 0.0:
                    rcond = 1.0 / (matrix_norm * inverse_norm)
                if rcond > condition_tolerance:
                    candidate0 = (inverse00 * r[0] + inverse01 * r[1] + inverse02 * r[2]) / scale
                    candidate1 = (inverse01 * r[0] + inverse11 * r[1] + inverse12 * r[2]) / scale
                    candidate2 = (inverse02 * r[0] + inverse12 * r[1] + inverse22 * r[2]) / scale
                    if (
                        -ti.math.inf < candidate0
                        and candidate0 < ti.math.inf
                        and -ti.math.inf < candidate1
                        and candidate1 < ti.math.inf
                        and -ti.math.inf < candidate2
                        and candidate2 < ti.math.inf
                    ):
                        z[0] = candidate0
                        z[1] = candidate1
                        z[2] = candidate2
    return z


@ti.func
def _apply_dense_block_jacobi(h, r, dim: ti.template()):
    """Dense counterpart of the scale-aware block-Jacobi solve above."""
    z = r
    scale = 0.0
    finite_input = 1
    for row in range(dim):
        for column in range(dim):
            value = h[row * dim + column]
            if not (-ti.math.inf < value and value < ti.math.inf):
                finite_input = 0
            scale = ti.max(scale, ti.abs(value))

    condition_tolerance = 1.0e-12
    determinant_floor = 1.0e-30
    if finite_input != 0 and scale > 0.0 and scale < ti.math.inf:
        for row in ti.static(range(4)):
            if ti.static(row < dim):
                diagonal = h[row * dim + row]
                denominator = scale
                if ti.abs(diagonal) > condition_tolerance * scale:
                    denominator = diagonal
                value = r[row] / denominator
                if -ti.math.inf < value and value < ti.math.inf:
                    z[row] = value

    if ti.static(dim == 1):
        if finite_input != 0 and scale > 0.0 and scale < ti.math.inf:
            z[0] = r[0] / h[0]
    elif ti.static(dim == 2):
        if finite_input != 0 and scale > 0.0 and scale < ti.math.inf:
            a00, a01 = h[0] / scale, h[1] / scale
            a10, a11 = h[2] / scale, h[3] / scale
            det = a00 * a11 - a01 * a10
            if ti.abs(det) > determinant_floor:
                inverse00, inverse01 = a11 / det, -a01 / det
                inverse10, inverse11 = -a10 / det, a00 / det
                matrix_norm = ti.max(
                    ti.abs(a00) + ti.abs(a01),
                    ti.abs(a10) + ti.abs(a11),
                )
                inverse_norm = ti.max(
                    ti.abs(inverse00) + ti.abs(inverse01),
                    ti.abs(inverse10) + ti.abs(inverse11),
                )
                rcond = 0.0
                if matrix_norm > 0.0 and inverse_norm > 0.0:
                    rcond = 1.0 / (matrix_norm * inverse_norm)
                if rcond > condition_tolerance:
                    candidate0 = (inverse00 * r[0] + inverse01 * r[1]) / scale
                    candidate1 = (inverse10 * r[0] + inverse11 * r[1]) / scale
                    if (
                        -ti.math.inf < candidate0
                        and candidate0 < ti.math.inf
                        and -ti.math.inf < candidate1
                        and candidate1 < ti.math.inf
                    ):
                        z[0] = candidate0
                        z[1] = candidate1
    elif ti.static(dim == 3):
        if finite_input != 0 and scale > 0.0 and scale < ti.math.inf:
            a00, a01, a02 = h[0] / scale, h[1] / scale, h[2] / scale
            a10, a11, a12 = h[3] / scale, h[4] / scale, h[5] / scale
            a20, a21, a22 = h[6] / scale, h[7] / scale, h[8] / scale
            c00 = a11 * a22 - a12 * a21
            c01 = a02 * a21 - a01 * a22
            c02 = a01 * a12 - a02 * a11
            c10 = a12 * a20 - a10 * a22
            c11 = a00 * a22 - a02 * a20
            c12 = a02 * a10 - a00 * a12
            c20 = a10 * a21 - a11 * a20
            c21 = a01 * a20 - a00 * a21
            c22 = a00 * a11 - a01 * a10
            det = a00 * c00 + a01 * c10 + a02 * c20
            if ti.abs(det) > determinant_floor:
                inverse00, inverse01, inverse02 = c00 / det, c01 / det, c02 / det
                inverse10, inverse11, inverse12 = c10 / det, c11 / det, c12 / det
                inverse20, inverse21, inverse22 = c20 / det, c21 / det, c22 / det
                matrix_norm = ti.max(
                    ti.abs(a00) + ti.abs(a01) + ti.abs(a02),
                    ti.abs(a10) + ti.abs(a11) + ti.abs(a12),
                    ti.abs(a20) + ti.abs(a21) + ti.abs(a22),
                )
                inverse_norm = ti.max(
                    ti.abs(inverse00) + ti.abs(inverse01) + ti.abs(inverse02),
                    ti.abs(inverse10) + ti.abs(inverse11) + ti.abs(inverse12),
                    ti.abs(inverse20) + ti.abs(inverse21) + ti.abs(inverse22),
                )
                rcond = 0.0
                if matrix_norm > 0.0 and inverse_norm > 0.0:
                    rcond = 1.0 / (matrix_norm * inverse_norm)
                if rcond > condition_tolerance:
                    candidate0 = (inverse00 * r[0] + inverse01 * r[1] + inverse02 * r[2]) / scale
                    candidate1 = (inverse10 * r[0] + inverse11 * r[1] + inverse12 * r[2]) / scale
                    candidate2 = (inverse20 * r[0] + inverse21 * r[1] + inverse22 * r[2]) / scale
                    if (
                        -ti.math.inf < candidate0
                        and candidate0 < ti.math.inf
                        and -ti.math.inf < candidate1
                        and candidate1 < ti.math.inf
                        and -ti.math.inf < candidate2
                        and candidate2 < ti.math.inf
                    ):
                        z[0] = candidate0
                        z[1] = candidate1
                        z[2] = candidate2
    else:
        if finite_input != 0 and scale > 0.0 and scale < ti.math.inf:
            A = ti.Matrix.zero(float, 4, 4)
            for i in range(4):
                for j in range(4):
                    A[i, j] = h[i * 4 + j] / scale
            det = A.determinant()
            if ti.abs(det) > determinant_floor:
                inverse = A.inverse()
                matrix_norm = 0.0
                inverse_norm = 0.0
                finite_inverse = 1
                for row in range(4):
                    row_norm = 0.0
                    inverse_row_norm = 0.0
                    for column in range(4):
                        row_norm += ti.abs(A[row, column])
                        inverse_row_norm += ti.abs(inverse[row, column])
                        if not (-ti.math.inf < inverse[row, column] and inverse[row, column] < ti.math.inf):
                            finite_inverse = 0
                    matrix_norm = ti.max(matrix_norm, row_norm)
                    inverse_norm = ti.max(inverse_norm, inverse_row_norm)
                rcond = 0.0
                if matrix_norm > 0.0 and inverse_norm > 0.0:
                    rcond = 1.0 / (matrix_norm * inverse_norm)
                if finite_inverse != 0 and rcond > condition_tolerance:
                    candidate = (inverse @ r) / scale
                    finite_candidate = 1
                    for row in ti.static(range(4)):
                        if not (-ti.math.inf < candidate[row] and candidate[row] < ti.math.inf):
                            finite_candidate = 0
                    if finite_candidate != 0:
                        z = candidate
    return z


@ti.data_oriented
class BuildTriplet:
    """Block triplet solver.

    symmetric=True packs each individual block into vector6 and is valid only
    when every stored block is itself symmetric.  Global matrix symmetry alone
    is represented by matrix_symmetric=True: one off-diagonal triangle is
    stored, but each retained off-diagonal block remains dense dim x dim.
    symmetric=False uses dense dim x dim blocks and block-Jacobi-preconditioned
    BiCGSTAB (scalar Jacobi when ``dim == 1``).
    matrix_symmetric=True stores only one global off-diagonal block triangle and
    applies the transposed contribution during matvec.
    full_symmetric_input=True accepts a complete two-triangle raw assembly,
    canonicalizes it to structurally symmetric upper-triangle storage before
    mirror-aware Dirichlet elimination and sparse reduction.  This is useful
    for projected-Newton IPC systems whose source assemblers are also reused by
    a nonsymmetric fully implicit solve.
    """

    def __init__(
        self,
        dim,
        max_pairs_num,
        max_nonzeros,
        max_active_nodes,
        symmetric=True,
        solver=None,
        matrix_symmetric=False,
        full_symmetric_input=False,
        pattern_cache=True,
        pattern_cache_extra_fraction=0.01,
        pattern_cache_max_age=25,
        device_reduction=None,
        raw_only=False,
        reduction="hash",
    ):
        if dim not in (1, 2, 3, 4):
            raise ValueError(
                "BuildTriplet currently supports dim=1, dim=2, dim=3, or nonsymmetric dim=4 block systems."
            )
        if symmetric and dim == 1:
            raise ValueError("BuildTriplet dim=1 uses dense scalar blocks; set symmetric=False.")
        if symmetric and dim == 4:
            raise ValueError("BuildTriplet symmetric=True uses vec6 blocks and only supports dim=2 or dim=3.")
        self.dim = dim
        self.symmetric = bool(symmetric)
        self.matrix_symmetric = bool(matrix_symmetric)
        self.full_symmetric_input = bool(full_symmetric_input)
        if self.full_symmetric_input and not self.matrix_symmetric:
            raise ValueError("full_symmetric_input=True requires matrix_symmetric=True")
        if self.full_symmetric_input and self.symmetric:
            raise ValueError("full_symmetric_input=True requires dense block storage " "(symmetric=False)")
        max_pairs_num = int(max_pairs_num)
        max_nonzeros = int(max_nonzeros)
        max_active_nodes = int(max_active_nodes)
        self.raw_only = bool(raw_only)
        if self.raw_only:
            # Coupled assemblers use source matrices only as deterministic
            # diagonal/raw-block streams and reduce once in the monolithic
            # destination.  Do not duplicate a large reduced pattern/hash
            # workspace that can never be consumed by the source itself.
            max_nonzeros = 1
            pattern_cache = False
        int32_max = int(np.iinfo(np.int32).max)
        if max_pairs_num <= 0 or max_nonzeros <= 0:
            raise ValueError("BuildTriplet raw-pair and reduced-nonzero capacities must " "be positive")
        if max_active_nodes <= 0:
            raise ValueError("BuildTriplet max_active_nodes must be positive")
        if max_active_nodes > int32_max:
            raise ValueError("BuildTriplet max_active_nodes exceeds the int32 block-index " "range")
        if max_active_nodes * int(dim) > int32_max:
            raise ValueError("BuildTriplet scalar degree-of-freedom capacity exceeds the " "int32 index range")
        self.hessian_size = 6 if self.symmetric else dim * dim
        self.max_active_nodes = max_active_nodes
        self.max_nonzeros = max_nonzeros
        self.solver = self._normalize_solver(solver or ("PCG" if self.symmetric else "BiCGSTAB"))

        self.diag = ti.Vector.field(self.hessian_size, dtype=float, shape=max_active_nodes)
        # Dense inverse diagonal blocks are rebuilt once per linear solve and
        # reused by every Krylov iteration.  This avoids repeating small
        # determinant/inverse calculations in each preconditioner application.
        self.diag_inverse = ti.Vector.field(self.dim * self.dim, dtype=float, shape=max_active_nodes)
        if reduction not in ("hash", "bucket"):
            raise ValueError("reduction must be 'hash' or 'bucket'")
        if reduction == "bucket" and self.raw_only:
            raise ValueError("bucket sources must be reducible")
        reducer = HashReduction
        if reduction == "bucket":
            from src.linear_solver.BucketReduction import BucketReduction

            reducer = BucketReduction
        self.non_diag = reducer(
            max_pairs_num,
            dim,
            max_nonzeros,
            hessian_size=self.hessian_size,
            pattern_cache=pattern_cache,
            pattern_cache_extra_fraction=pattern_cache_extra_fraction,
            pattern_cache_max_age=pattern_cache_max_age,
            device_reduction=device_reduction,
        )
        self.raw_non_diag_count = ti.field(ti.i32, shape=1)
        self.overflow = ti.field(ti.i32, shape=1)

        self.rhs = ti.Vector.field(self.dim, dtype=float, shape=max_active_nodes)
        self.x = ti.Vector.field(self.dim, dtype=float, shape=max_active_nodes)
        self.Ax = ti.Vector.field(self.dim, dtype=float, shape=max_active_nodes)
        self.r = ti.Vector.field(self.dim, dtype=float, shape=max_active_nodes)
        self.z = ti.Vector.field(self.dim, dtype=float, shape=max_active_nodes)
        self.p = ti.Vector.field(self.dim, dtype=float, shape=max_active_nodes)
        self.Ap = ti.Vector.field(self.dim, dtype=float, shape=max_active_nodes)

        self.r_hat = ti.Vector.field(self.dim, dtype=float, shape=max_active_nodes)
        self.v = ti.Vector.field(self.dim, dtype=float, shape=max_active_nodes)
        self.s = ti.Vector.field(self.dim, dtype=float, shape=max_active_nodes)
        self.t = ti.Vector.field(self.dim, dtype=float, shape=max_active_nodes)
        self.p_hat = ti.Vector.field(self.dim, dtype=float, shape=max_active_nodes)
        self.s_hat = ti.Vector.field(self.dim, dtype=float, shape=max_active_nodes)

        # Cached block-to-scalar CSR conversion plan for the CPU direct-solve
        # compatibility path.  CUDA Krylov solvers stay entirely in Taichi and
        # do not use this plan.
        self._scipy_plan_key = None
        self._scipy_plan_inverse = None
        self._scipy_plan_indices = None
        self._scipy_plan_indptr = None
        self.scipy_pattern_rebuilds = 0
        self.scipy_pattern_hits = 0
        # Python-side lifecycle flag for full-input matrices. Production
        # assembly mutates them only through reset/append/direct scalar APIs,
        # so a pre-DBC canonicalization can be reused by finalization without
        # launching a second O(raw blocks) device sweep in the same Newton
        # iteration.
        self._full_input_canonicalized = False

    def reduce_non_diag(self, pairs_num):
        if getattr(self.non_diag, "fixed_count", 0) and getattr(self, "_fixed_finalized", False):
            return
        if self.raw_only:
            raise RuntimeError(
                "raw-only BuildTriplet sources cannot be reduced directly; "
                "append them to a reducible destination first"
            )
        self.non_diag.go(pairs_num)
        self._fixed_finalized = bool(getattr(self.non_diag, "fixed_count", 0))

    def reset_system(self):
        if getattr(self.non_diag, "fixed_count", 0):
            self.non_diag.reset_fixed_values()
        self._reset_system()
        self._full_input_canonicalized = False
        self._fixed_finalized = False

    def install_fixed_pattern(self, coordinates):
        if self.raw_only or not isinstance(self.non_diag, HashReduction):
            raise ValueError("fixed assembly requires a reducible HashTriplet matrix")
        coordinates = np.asarray(coordinates)
        if self.symmetric or coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("fixed assembly requires dense blocks and an (n, 2) coordinate array")
        if np.any(coordinates >= self.max_active_nodes):
            raise ValueError("fixed coordinates exceed node capacity")
        if self.matrix_symmetric and np.any(coordinates[:, 0] >= coordinates[:, 1]):
            raise ValueError("symmetric fixed slots must contain the strict upper block triangle")
        if np.any(coordinates[:, 0] == coordinates[:, 1]):
            raise ValueError("diagonal blocks use the dedicated diagonal field")
        self.non_diag.install_fixed_pattern(coordinates)

    @ti.func
    def add_fixed_block(self, slot, block_i, block_j, block):
        if block_i == block_j:
            for row, column in ti.static(ti.ndrange(self.dim, self.dim)):
                ti.atomic_add(self.diag[block_i][self.scalar_component_index(row, column)], block[row, column])
        elif slot >= 0:
            for row, column in ti.static(ti.ndrange(self.dim, self.dim)):
                ti.atomic_add(
                    self.non_diag.tripletH[slot][self.scalar_component_index(row, column)], block[row, column]
                )

    @ti.kernel
    def eliminate_fixed_constraints(self, fixed: ti.template(), correction: ti.template(), rhs: ti.template()):
        for slot in range(self.non_diag.fixed_count):
            i, j = self.non_diag.tripletI[slot], self.non_diag.tripletJ[slot]
            for row, column in ti.static(ti.ndrange(self.dim, self.dim)):
                r, c = self.dim * i + row, self.dim * j + column
                entry = self.scalar_component_index(row, column)
                value = self.non_diag.tripletH[slot][entry]
                if fixed[c]:
                    ti.atomic_add(rhs[r], -value * correction[c])
                if ti.static(self.matrix_symmetric):
                    if fixed[r]:
                        ti.atomic_add(rhs[c], -value * correction[r])
                if fixed[r] or fixed[c]:
                    self.non_diag.tripletH[slot][entry] = 0.0

    def reserve_raw_block_slots(self, count):
        """Reserve deterministic off-diagonal slots for direct device assembly."""
        count = int(count)
        start = int(self.raw_non_diag_count[0])
        end = start + count
        if end > int(self.non_diag.max_pairs_num):
            raise RuntimeError(
                "BuildTriplet non-diagonal triplet buffer overflow: "
                f"need {end}, capacity {self.non_diag.max_pairs_num}."
            )
        self.raw_non_diag_count[0] = end
        self._full_input_canonicalized = False
        return start

    def append_raw_from(self, source, *, active_nodes=None, block_offset=0, scale=1.0):
        """Append another block system without leaving the Taichi device.

        This is the monolithic assembly primitive used by coupled IPC
        solvers.  Both diagonal blocks and unreduced off-diagonal blocks are
        copied directly between Taichi fields; the destination is reduced
        only once after all subsystems have been appended.
        """
        self._append_source(source, active_nodes, block_offset, scale, reduced=False)

    def append_reduced_from(self, source, *, active_nodes=None, block_offset=0, scale=1.0):
        """Reduce a source on device, then append its diagonal and unique blocks."""
        self._append_source(source, active_nodes, block_offset, scale, reduced=True)

    def _append_source(self, source, active_nodes, block_offset, scale, *, reduced):
        if getattr(self, "_fixed_finalized", False):
            raise RuntimeError("reset a finalized fixed-slot matrix before appending new sources")
        if not isinstance(source, BuildTriplet):
            raise TypeError("source must be a BuildTriplet")
        if self.full_symmetric_input:
            # A full-input destination must see the source's two physical
            # triangles.  Accepting an already mirrored/upper-only source
            # would make the later lower-triangle discard ambiguous.
            compatible_matrix_storage = not source.matrix_symmetric
        else:
            compatible_matrix_storage = (
                source.matrix_symmetric == self.matrix_symmetric and not source.full_symmetric_input
            )
        if (
            source.dim != self.dim
            or source.hessian_size != self.hessian_size
            or source.symmetric != self.symmetric
            or not compatible_matrix_storage
        ):
            raise ValueError("source and destination block storage must match")
        active_nodes = source.max_active_nodes if active_nodes is None else int(active_nodes)
        block_offset = int(block_offset)
        scale = float(scale)
        if not np.isfinite(scale):
            raise ValueError("source matrix scale must be finite")
        if active_nodes < 0 or active_nodes > source.max_active_nodes:
            raise ValueError(
                f"active_nodes={active_nodes} is outside source capacity " f"[0, {source.max_active_nodes}]"
            )
        if block_offset < 0 or block_offset + active_nodes > self.max_active_nodes:
            raise ValueError("shifted source blocks exceed destination node capacity")
        if reduced:
            source.finalize_taichi_assembly()
        self._append_raw_fields(
            active_nodes,
            block_offset,
            source.diag,
            source.non_diag.tripletI if reduced else source.non_diag.blockI,
            source.non_diag.tripletJ if reduced else source.non_diag.blockJ,
            source.non_diag.tripletH if reduced else source.non_diag.blockH,
            source.non_diag.element_pair_num if reduced else source.raw_non_diag_count,
            source.overflow,
            scale,
        )
        self._full_input_canonicalized = False

    def solve(
        self,
        rhs=None,
        x=None,
        active_nodes=None,
        tol=1.0e-10,
        maxiter=500,
        return_solution=True,
        rel_tol=0.0,
        transpose=False,
    ):
        if current_cfg().arch == ti.cuda:
            if rhs is not None or x is not None:
                raise RuntimeError(
                    "CUDA BuildTriplet.solve requires rhs/x to be preloaded "
                    "in Taichi fields; host vectors are not supported"
                )
            if return_solution:
                raise RuntimeError(
                    "CUDA BuildTriplet.solve cannot return a full host solution; "
                    "use return_solution=False and keep x in its Taichi field"
                )
        if self.solver in ("CG", "PCG"):
            if transpose and not self.matrix_symmetric:
                raise ValueError("transpose=True for a nonsymmetric matrix requires BiCGSTAB")
            return self.PCGSolver(
                rhs=rhs,
                x=x,
                active_nodes=active_nodes,
                tol=tol,
                rel_tol=rel_tol,
                maxiter=maxiter,
                return_solution=return_solution,
            )
        return self.BiCGSTABSolver(
            rhs=rhs,
            x=x,
            active_nodes=active_nodes,
            tol=tol,
            rel_tol=rel_tol,
            maxiter=maxiter,
            return_solution=return_solution,
            transpose=transpose,
        )

    def solve_flat_system(
        self,
        rhs,
        solution=None,
        *,
        active_nodes=None,
        tol=1.0e-10,
        maxiter=500,
        return_solution=False,
        rel_tol=0.0,
        transpose=False,
        fallback_to_bicgstab=False,
    ):
        """Solve from/to flat scalar fields without a CPU matrix conversion.

        This is the common CUDA route for implicit MPM, IGA, and IPC coupled
        systems. ``rhs`` and ``solution`` may be Taichi scalar fields; only the
        small convergence scalars cross the host boundary during Krylov.
        ``transpose=True`` solves the transposed device operator without
        assembling another triplet matrix. Exact symmetric adjoints may set
        ``fallback_to_bicgstab``: PCG remains the fast path and a curvature or
        convergence failure restarts BiCGSTAB on the same device matrix.
        """
        active_nodes = self.max_active_nodes if active_nodes is None else int(active_nodes)
        if current_cfg().arch == ti.cuda:
            if not isinstance(rhs, ti.ScalarField):
                raise RuntimeError("CUDA solve_flat_system requires a Taichi scalar rhs field")
            if return_solution:
                raise RuntimeError(
                    "CUDA solve_flat_system cannot return a full host solution; "
                    "provide a Taichi solution field instead"
                )
        if active_nodes < 0 or active_nodes > self.max_active_nodes:
            raise ValueError(f"active_nodes={active_nodes} is outside [0, {self.max_active_nodes}]")
        expected = active_nodes * self.dim
        if isinstance(rhs, ti.ScalarField):
            if len(rhs.shape) != 1 or int(rhs.shape[0]) < expected:
                raise ValueError(f"flat rhs field has shape {rhs.shape}, expected at least ({expected},)")
            self._load_flat_rhs_field(active_nodes, rhs)
        else:
            values = np.asarray(rhs, dtype=np.float64).reshape(-1)
            if values.size < expected:
                raise ValueError(f"flat rhs has {values.size} values, expected at least {expected}")
            self.rhs.from_numpy(self._pad_vector_array(values[:expected].reshape((active_nodes, self.dim))))
        result = self.solve(
            active_nodes=active_nodes,
            tol=tol,
            rel_tol=rel_tol,
            maxiter=maxiter,
            return_solution=False,
            transpose=transpose,
        )
        if fallback_to_bicgstab and not result["converged"] and self.solver in ("CG", "PCG"):
            primary_result = result
            primary_solver = self.solver
            self.solver = "BiCGSTAB"
            try:
                result = self.solve(
                    active_nodes=active_nodes,
                    tol=tol,
                    rel_tol=rel_tol,
                    maxiter=maxiter,
                    return_solution=False,
                    transpose=transpose,
                )
            finally:
                self.solver = primary_solver
            result["fallback_from"] = primary_solver
            result["primary_iterations"] = int(primary_result["iterations"])
            result["primary_residual"] = float(primary_result["residual"])
        result["solution_inf_norm"] = float(self._solution_inf_norm(active_nodes))
        if solution is not None:
            if not isinstance(solution, ti.ScalarField):
                raise TypeError("solution must be a Taichi scalar field")
            if len(solution.shape) != 1 or int(solution.shape[0]) < expected:
                raise ValueError(f"flat solution field has shape {solution.shape}, expected at least ({expected},)")
            self._store_flat_solution_field(active_nodes, solution)
        if return_solution:
            result["x"] = field_to_numpy_prefix(self.x, active_nodes).reshape(-1).copy()
        return result

    def flat_l2_norm(self, values, active_dofs):
        """Return only a scalar norm while keeping the full vector on device."""
        if not isinstance(values, ti.ScalarField):
            raise TypeError("values must be a Taichi scalar field")
        active_dofs = int(active_dofs)
        if active_dofs < 0 or active_dofs > int(values.shape[0]):
            raise ValueError(f"active_dofs={active_dofs} is outside field capacity " f"[0, {int(values.shape[0])}]")
        return math.sqrt(max(float(self._flat_squared_norm(active_dofs, values)), 0.0))

    def PCGSolver(
        self,
        rhs=None,
        x=None,
        active_nodes=None,
        tol=1.0e-10,
        maxiter=500,
        return_solution=True,
        rel_tol=0.0,
    ):
        tol = float(tol)
        rel_tol = float(rel_tol)
        maxiter = int(maxiter)
        if not math.isfinite(tol) or tol < 0.0:
            raise ValueError("PCG absolute tolerance must be finite and non-negative")
        if not math.isfinite(rel_tol) or rel_tol < 0.0:
            raise ValueError("PCG relative tolerance must be finite and non-negative")
        if maxiter < 0:
            raise ValueError("PCG maxiter must be non-negative")
        self._load_rhs_x(rhs, x)
        active_nodes = self.max_active_nodes if active_nodes is None else int(active_nodes)
        nnz = int(self.non_diag.element_pair_num[0])

        # Matvec and initialization overwrite every PCG workspace before use.
        self._build_block_jacobi(active_nodes)
        self.matvec(active_nodes, nnz, self.x, self.Ax)
        rz_old, residual_squared = self._init_pcg(active_nodes)
        # ``tol`` is an absolute residual tolerance, so convergence must be
        # measured in the unpreconditioned Euclidean norm.  The PCG recurrence
        # still uses r^T M^-1 r below, but that quantity changes under a simple
        # scaling of A/M.  Comparing sqrt(r^T M^-1 r) with ``tol`` can
        # incorrectly accept x=0 for a high-stiffness contact system whose
        # true residual is still orders of magnitude above tolerance.
        residual = math.sqrt(max(float(residual_squared), 0.0))
        initial_residual = residual
        convergence_tolerance = max(tol, rel_tol * initial_residual)
        converged = residual <= convergence_tolerance
        iterations = 0
        for iteration in range(maxiter):
            if converged:
                break
            self.matvec(active_nodes, nnz, self.p, self.Ap)
            denom = self._dot(active_nodes, self.p, self.Ap)
            if not math.isfinite(denom) or denom <= 0.0:
                break
            alpha = rz_old / denom
            if not math.isfinite(alpha):
                break
            rz_new, residual_squared = self._pcg_update_and_reduce(active_nodes, alpha)
            residual = math.sqrt(max(float(residual_squared), 0.0))
            iterations = iteration + 1
            verify_residual = residual <= convergence_tolerance
            if verify_residual:
                # Recursive CG residuals are not a convergence oracle for
                # strongly scaled IPC systems.  Recompute r=b-Ax and restart
                # p=M^-1r so a cancellation in ``r -= alpha Ap`` cannot report
                # false convergence.
                self.matvec(active_nodes, nnz, self.x, self.Ax)
                rz_old, residual_squared = self._init_pcg(active_nodes)
                residual = math.sqrt(max(float(residual_squared), 0.0))
                converged = residual <= convergence_tolerance
                if converged:
                    break
                if not math.isfinite(rz_old) or rz_old <= 0.0:
                    break
                continue
            beta = rz_new / rz_old
            if not math.isfinite(beta):
                break
            self._pcg_update_p(active_nodes, beta)
            rz_old = rz_new

        recursive_residual = residual
        if not converged:
            self.matvec(active_nodes, nnz, self.x, self.Ax)
            _, residual_squared = self._init_pcg(active_nodes)
            residual = math.sqrt(max(float(residual_squared), 0.0))
            converged = residual <= convergence_tolerance
        result = self._result(
            active_nodes,
            converged,
            iterations,
            residual,
            return_solution,
            initial_residual=initial_residual,
            convergence_tolerance=convergence_tolerance,
        )
        result["recursive_residual"] = float(recursive_residual)
        return result

    def BiCGSTABSolver(
        self,
        rhs=None,
        x=None,
        active_nodes=None,
        tol=1.0e-10,
        maxiter=500,
        return_solution=True,
        rel_tol=0.0,
        transpose=False,
    ):
        tol = float(tol)
        rel_tol = float(rel_tol)
        maxiter = int(maxiter)
        if not math.isfinite(tol) or tol < 0.0:
            raise ValueError("BiCGSTAB absolute tolerance must be finite and non-negative")
        if not math.isfinite(rel_tol) or rel_tol < 0.0:
            raise ValueError("BiCGSTAB relative tolerance must be finite and non-negative")
        if maxiter < 0:
            raise ValueError("BiCGSTAB maxiter must be non-negative")
        self._load_rhs_x(rhs, x)
        active_nodes = self.max_active_nodes if active_nodes is None else int(active_nodes)
        nnz = int(self.non_diag.element_pair_num[0])
        matvec = self.transpose_matvec if transpose else self.matvec
        apply_preconditioner = self._apply_transpose_preconditioner if transpose else self._apply_preconditioner

        self._solver_reset(active_nodes)
        self._build_block_jacobi(active_nodes)
        matvec(active_nodes, nnz, self.x, self.Ax)
        self._init_bicgstab(active_nodes)

        rho_old = 1.0
        alpha = 1.0
        omega = 1.0
        residual = math.sqrt(max(float(self._dot(active_nodes, self.r, self.r)), 0.0))
        initial_residual = residual
        convergence_tolerance = max(tol, rel_tol * initial_residual)
        converged = residual <= convergence_tolerance
        iterations = 0
        reliable_update_interval = 32

        for iteration in range(maxiter):
            if converged:
                break
            rho = self._dot(active_nodes, self.r_hat, self.r)
            if abs(rho) < 1.0e-30:
                break
            if iteration == 0:
                self._copy(active_nodes, self.r, self.p)
            else:
                beta = (rho / rho_old) * (alpha / omega)
                self._bicg_update_p(active_nodes, beta, omega)

            apply_preconditioner(active_nodes, self.p, self.p_hat)
            matvec(active_nodes, nnz, self.p_hat, self.v)
            denom = self._dot(active_nodes, self.r_hat, self.v)
            if abs(denom) < 1.0e-30:
                break
            alpha = rho / denom
            self._bicg_update_s(active_nodes, alpha)
            s_norm = math.sqrt(max(float(self._dot(active_nodes, self.s, self.s)), 0.0))
            if s_norm <= convergence_tolerance:
                self._bicg_update_x_alpha(active_nodes, alpha)
                iterations = iteration + 1
                matvec(active_nodes, nnz, self.x, self.Ax)
                self._init_bicgstab(active_nodes)
                residual = math.sqrt(max(float(self._dot(active_nodes, self.r, self.r)), 0.0))
                converged = residual <= convergence_tolerance
                if converged:
                    break
                rho_old = self._dot(active_nodes, self.r_hat, self.r)
                alpha = 1.0
                omega = 1.0
                if not math.isfinite(rho_old) or abs(rho_old) < 1.0e-30:
                    break
                continue

            apply_preconditioner(active_nodes, self.s, self.s_hat)
            matvec(active_nodes, nnz, self.s_hat, self.t)
            tt = self._dot(active_nodes, self.t, self.t)
            if abs(tt) < 1.0e-30:
                break
            omega = self._dot(active_nodes, self.t, self.s) / tt
            self._bicg_update_x_r(active_nodes, alpha, omega)
            residual = math.sqrt(max(float(self._dot(active_nodes, self.r, self.r)), 0.0))
            iterations = iteration + 1
            verify_residual = residual <= convergence_tolerance or (iteration + 1) % reliable_update_interval == 0
            if verify_residual:
                matvec(active_nodes, nnz, self.x, self.Ax)
                self._init_bicgstab(active_nodes)
                residual = math.sqrt(max(float(self._dot(active_nodes, self.r, self.r)), 0.0))
                converged = residual <= convergence_tolerance
                if converged:
                    break
                rho_old = self._dot(active_nodes, self.r_hat, self.r)
                alpha = 1.0
                omega = 1.0
                if not math.isfinite(rho_old) or abs(rho_old) < 1.0e-30:
                    break
                continue
            if abs(omega) < 1.0e-30:
                break
            rho_old = rho

        if not converged:
            matvec(active_nodes, nnz, self.x, self.Ax)
            self._init_bicgstab(active_nodes)
            residual = math.sqrt(max(float(self._dot(active_nodes, self.r, self.r)), 0.0))
            converged = residual <= convergence_tolerance

        return self._result(
            active_nodes,
            converged,
            iterations,
            residual,
            return_solution,
            initial_residual=initial_residual,
            convergence_tolerance=convergence_tolerance,
        )

    @staticmethod
    def _normalize_solver(solver):
        solver = str(solver).upper()
        if solver in ("CG", "PCG"):
            return solver
        if solver in ("BICG", "BICGSTAB", "BI-CGSTAB", "STABBICG"):
            return "BiCGSTAB"
        raise ValueError(f"Unknown BuildTriplet solver: {solver}")

    def matvec(self, active_nodes, nnz, x, Ax):
        self._matvec(int(active_nodes), int(nnz), x, Ax)

    def transpose_matvec(self, active_nodes, nnz, x, Ax):
        kernel = self._matvec if self.matrix_symmetric else self._transpose_matvec
        kernel(int(active_nodes), int(nnz), x, Ax)

    def assemble_scalar_triplets(self, rows, cols, vals):
        rows = np.asarray(rows, dtype=np.int32)
        cols = np.asarray(cols, dtype=np.int32)
        vals = np.asarray(vals, dtype=np.float64)
        if rows.shape != cols.shape or rows.shape != vals.shape:
            raise ValueError("rows, cols, and vals must have the same 1D shape.")
        self._full_input_canonicalized = False
        self._assemble_scalar_triplets(rows, cols, vals, int(vals.shape[0]))

    @ti.kernel
    def _assemble_scalar_triplets(
        self, rows: ti.types.ndarray(), cols: ti.types.ndarray(), vals: ti.types.ndarray(), n: ti.i32
    ):
        for i in range(n):
            row = rows[i]
            col = cols[i]
            block_i = row // self.dim
            block_j = col // self.dim
            row_comp = row - block_i * self.dim
            col_comp = col - block_j * self.dim
            value = vals[i]
            h_index = self.scalar_component_index(row_comp, col_comp)
            if block_i == block_j:
                h_value = value
                if ti.static(self.symmetric):
                    if row_comp != col_comp:
                        h_value *= 0.5
                ti.atomic_add(self.diag[block_i][h_index], h_value)
            else:
                store_i = block_i
                store_j = block_j
                store_row = row_comp
                store_col = col_comp
                if ti.static(self.matrix_symmetric and not self.full_symmetric_input):
                    if block_i > block_j:
                        store_i = block_j
                        store_j = block_i
                        store_row = col_comp
                        store_col = row_comp
                h_index = self.scalar_component_index(store_row, store_col)
                idx = ti.atomic_add(self.raw_non_diag_count[0], 1)
                if idx < self.non_diag.blockI.shape[0]:
                    self.non_diag.blockI[idx] = store_i
                    self.non_diag.blockJ[idx] = store_j
                    for k in range(self.hessian_size):
                        self.non_diag.blockH[idx][k] = 0.0
                    h_value = value
                    if ti.static(self.symmetric):
                        if row_comp != col_comp:
                            h_value *= 0.5
                    self.non_diag.blockH[idx][h_index] = h_value
                else:
                    self.overflow[0] = 1

    @ti.kernel
    def _canonicalize_full_symmetric_input(self, pairs_num: ti.i32):
        """Convert a complete raw matrix to one structurally symmetric half.

        PCG must not depend on two independently reduced floating-point copies
        being bitwise transposes. Keep the upper block triangle, mirror it in
        the matrix-vector product, and explicitly symmetrize every dense
        diagonal block. Mirror-aware Dirichlet elimination must run after this
        operation so its right-hand side correction uses the identical
        canonical operator. Raw slots are only invalidated, never compacted,
        so persistent pattern-cache slot identities remain deterministic
        across Newton steps.
        """
        for block in self.diag:
            for row in range(self.dim):
                for column in range(self.dim):
                    if row < column:
                        upper = row * self.dim + column
                        lower = column * self.dim + row
                        value = 0.5 * (self.diag[block][upper] + self.diag[block][lower])
                        self.diag[block][upper] = value
                        self.diag[block][lower] = value

        for raw_index in range(pairs_num):
            if raw_index < self.non_diag.blockI.shape[0]:
                block_i = self.non_diag.blockI[raw_index]
                block_j = self.non_diag.blockJ[raw_index]
                if block_i >= block_j:
                    self.non_diag.blockI[raw_index] = -1
                    self.non_diag.blockJ[raw_index] = -1
                    self.non_diag.blockH[raw_index] = ti.Vector.zero(float, self.hessian_size)

    def canonicalize_full_symmetric_input(self):
        """Make a full projected-Newton matrix structurally symmetric.

        Call this after every source has been appended and before any
        mirror-aware Dirichlet elimination.  ``finalize_taichi_assembly`` calls
        it again defensively; the operation is idempotent.
        """
        if not self.full_symmetric_input:
            return int(self.raw_non_diag_count[0])
        if self._full_input_canonicalized:
            return int(self.raw_non_diag_count[0])
        if int(self.overflow[0]) != 0:
            raise RuntimeError("BuildTriplet cannot canonicalize an overflowed raw matrix")
        pairs_num = int(self.raw_non_diag_count[0])
        self._canonicalize_full_symmetric_input(pairs_num)
        self._full_input_canonicalized = True
        return pairs_num

    def finalize_taichi_assembly(self):
        fixed = getattr(self.non_diag, "fixed_count", 0)
        if fixed and getattr(self, "_fixed_finalized", False):
            return int(self.raw_non_diag_count[0])
        if self.raw_only:
            raise RuntimeError(
                "raw-only BuildTriplet sources cannot be finalized directly; "
                "append them to a reducible destination first"
            )
        if int(self.overflow[0]) != 0:
            raise RuntimeError(
                f"BuildTriplet non-diagonal triplet buffer overflow: used {int(self.raw_non_diag_count[0])}, "
                f"capacity {self.non_diag.max_pairs_num}."
            )
        pairs_num = int(self.raw_non_diag_count[0])
        if self.full_symmetric_input and not self._full_input_canonicalized:
            self._canonicalize_full_symmetric_input(pairs_num)
            self._full_input_canonicalized = True
        self.reduce_non_diag(pairs_num)
        self._fixed_finalized = bool(fixed)
        return pairs_num

    def to_scipy(self, active_nodes=None):
        if self.raw_only:
            raise RuntimeError(
                "raw-only BuildTriplet sources cannot be converted directly; "
                "append them to a reducible destination first"
            )
        from scipy.sparse import csr_matrix

        active_nodes = self.max_active_nodes if active_nodes is None else int(active_nodes)
        if active_nodes < 0 or active_nodes > self.max_active_nodes:
            raise ValueError(f"active_nodes={active_nodes} is outside [0, {self.max_active_nodes}]")
        diag = field_to_numpy_prefix(self.diag, active_nodes)
        diag_blocks = self._unpack_blocks(diag)
        triplet_i, triplet_j, triplet_h = self.non_diag.get_reduced_triplets_numpy()
        keep = (triplet_i >= 0) & (triplet_i < active_nodes)
        keep &= (triplet_j >= 0) & (triplet_j < active_nodes)
        triplet_i = triplet_i[keep]
        triplet_j = triplet_j[keep]
        offdiag_blocks = self._unpack_blocks(triplet_h[keep])

        block_rows = [np.arange(active_nodes, dtype=np.int32), triplet_i]
        block_cols = [np.arange(active_nodes, dtype=np.int32), triplet_j]
        blocks = [diag_blocks, offdiag_blocks]
        if self.matrix_symmetric and triplet_i.size:
            mirror = triplet_i != triplet_j
            block_rows.append(triplet_j[mirror])
            block_cols.append(triplet_i[mirror])
            blocks.append(np.swapaxes(offdiag_blocks[mirror], 1, 2))

        block_rows = np.concatenate(block_rows)
        block_cols = np.concatenate(block_cols)
        blocks = np.concatenate(blocks, axis=0)
        size = active_nodes * self.dim
        if block_rows.size == 0:
            return csr_matrix((size, size), dtype=np.float64)

        entries_per_block = self.dim * self.dim
        row_component = np.repeat(np.arange(self.dim, dtype=np.int64), self.dim)
        col_component = np.tile(np.arange(self.dim, dtype=np.int64), self.dim)
        rows = np.repeat(block_rows.astype(np.int64), entries_per_block) * self.dim + np.tile(
            row_component, block_rows.size
        )
        cols = np.repeat(block_cols.astype(np.int64), entries_per_block) * self.dim + np.tile(
            col_component, block_cols.size
        )
        data = np.ascontiguousarray(blocks.reshape(-1), dtype=np.float64)

        pattern_stats = self.non_diag.pattern_cache_statistics()
        plan_key = (
            active_nodes,
            int(pattern_stats.get("pattern_version", -1)),
            int(triplet_i.size),
            int(data.size),
            bool(self.matrix_symmetric),
        )
        if self._scipy_plan_key == plan_key and self._scipy_plan_inverse.size == data.size:
            inverse = self._scipy_plan_inverse
            self.scipy_pattern_hits += 1
        else:
            scalar_keys = rows * np.int64(max(size, 1)) + cols
            unique_keys, inverse = np.unique(scalar_keys, return_inverse=True)
            unique_rows = unique_keys // np.int64(max(size, 1))
            unique_cols = unique_keys - unique_rows * np.int64(max(size, 1))
            row_counts = np.bincount(unique_rows, minlength=size)
            indptr = np.empty(size + 1, dtype=np.int64)
            indptr[0] = 0
            np.cumsum(row_counts, out=indptr[1:])
            self._scipy_plan_key = plan_key
            self._scipy_plan_inverse = np.ascontiguousarray(inverse, dtype=np.int64)
            self._scipy_plan_indices = np.ascontiguousarray(unique_cols, dtype=np.int32)
            self._scipy_plan_indptr = indptr
            self.scipy_pattern_rebuilds += 1
            inverse = self._scipy_plan_inverse

        reduced_data = np.bincount(
            inverse,
            weights=data,
            minlength=self._scipy_plan_indices.size,
        )
        # Callers may eliminate zeros or add matrices in-place, so protect the
        # immutable conversion plan from SciPy's structural mutations.
        return csr_matrix(
            (
                reduced_data,
                self._scipy_plan_indices.copy(),
                self._scipy_plan_indptr.copy(),
            ),
            shape=(size, size),
            copy=False,
        )

    def _unpack_blocks(self, packed):
        packed = np.asarray(packed, dtype=np.float64)
        if packed.size == 0:
            return np.empty((0, self.dim, self.dim), dtype=np.float64)
        if not self.symmetric:
            return np.ascontiguousarray(packed.reshape((-1, self.dim, self.dim)))
        blocks = np.zeros((packed.shape[0], self.dim, self.dim), dtype=np.float64)
        blocks[:, 0, 0] = packed[:, 0]
        blocks[:, 1, 1] = packed[:, 1]
        blocks[:, 0, 1] = blocks[:, 1, 0] = packed[:, 3]
        if self.dim == 3:
            blocks[:, 2, 2] = packed[:, 2]
            blocks[:, 1, 2] = blocks[:, 2, 1] = packed[:, 4]
            blocks[:, 0, 2] = blocks[:, 2, 0] = packed[:, 5]
        return blocks

    def acceleration_statistics(self):
        return {
            "block_pattern": self.non_diag.pattern_cache_statistics(),
            "scipy_pattern_rebuilds": int(self.scipy_pattern_rebuilds),
            "scipy_pattern_hits": int(self.scipy_pattern_hits),
        }

    def load_from_scipy_blocks(self, K, active_nodes=None):
        if self.full_symmetric_input:
            raise RuntimeError(
                "full_symmetric_input matrices require raw two-triangle "
                "assembly followed by canonicalization; reduced SciPy "
                "loading bypasses that lifecycle"
            )
        active_nodes = self.max_active_nodes if active_nodes is None else int(active_nodes)
        K = K.tocoo()
        if K.shape[0] != K.shape[1]:
            raise ValueError("BuildTriplet expects a square matrix.")
        if K.shape[0] > active_nodes * self.dim:
            raise ValueError("The scipy matrix is larger than active_nodes * dim.")

        diag = np.zeros((self.max_active_nodes, self.hessian_size), dtype=np.float64)
        blocks = {}
        for row, col, value in zip(K.row, K.col, K.data):
            bi = int(row) // self.dim
            bj = int(col) // self.dim
            ci = int(row) % self.dim
            cj = int(col) % self.dim
            if bi >= active_nodes or bj >= active_nodes:
                continue
            if bi == bj:
                if self.symmetric:
                    self._accumulate_sym_entry(diag[bi], ci, cj, value)
                else:
                    diag[bi, ci * self.dim + cj] += value
            else:
                if self.matrix_symmetric and bi > bj:
                    continue
                block = blocks.setdefault((bi, bj), np.zeros((self.dim, self.dim), dtype=np.float64))
                block[ci, cj] += value

        self.diag.from_numpy(diag)
        block_i = []
        block_j = []
        block_h = []
        for (bi, bj), block in blocks.items():
            block_i.append(bi)
            block_j.append(bj)
            if self.symmetric:
                block_h.append(self._pack_sym_block(block))
            else:
                block_h.append(block.reshape(-1))
        if block_i:
            self._set_reduced_non_diag(
                np.asarray(block_i, dtype=np.int32),
                np.asarray(block_j, dtype=np.int32),
                np.asarray(block_h, dtype=np.float64),
            )
        else:
            self.non_diag.element_pair_num[0] = 0

    @staticmethod
    def _accumulate_sym_entry(h, row, col, value):
        if row == 0 and col == 0:
            h[0] += value
        elif row == 1 and col == 1:
            h[1] += value
        elif row == 2 and col == 2:
            h[2] += value
        elif (row == 0 and col == 1) or (row == 1 and col == 0):
            h[3] += 0.5 * value
        elif (row == 1 and col == 2) or (row == 2 and col == 1):
            h[4] += 0.5 * value
        elif (row == 0 and col == 2) or (row == 2 and col == 0):
            h[5] += 0.5 * value

    def _load_rhs_x(self, rhs, x):
        if rhs is not None:
            self.rhs.from_numpy(self._pad_vector_array(rhs))
        if x is not None:
            self.x.from_numpy(self._pad_vector_array(x))
        else:
            self.x.fill(0.0)

    def _result(
        self,
        active_nodes,
        converged,
        iterations,
        residual,
        return_solution=True,
        *,
        initial_residual=None,
        convergence_tolerance=None,
    ):
        result = {
            "converged": bool(converged),
            "iterations": int(iterations),
            "residual": float(residual),
        }
        if initial_residual is not None:
            result["initial_residual"] = float(initial_residual)
        if convergence_tolerance is not None:
            result["convergence_tolerance"] = float(convergence_tolerance)
        if return_solution:
            result["x"] = self.x.to_numpy()[:active_nodes, : self.dim].copy()
        return result

    @ti.kernel
    def _load_flat_rhs_field(self, active_nodes: int, rhs: ti.template()):
        for block in range(active_nodes):
            for component in ti.static(range(4)):
                if ti.static(component < self.dim):
                    self.rhs[block][component] = rhs[block * self.dim + component]

    @ti.kernel
    def _store_flat_solution_field(self, active_nodes: int, solution: ti.template()):
        for block in range(active_nodes):
            for component in ti.static(range(4)):
                if ti.static(component < self.dim):
                    solution[block * self.dim + component] = self.x[block][component]

    @ti.kernel
    def _flat_squared_norm(self, active_dofs: int, values: ti.template()) -> float:
        norm_squared = 0.0
        for dof in range(active_dofs):
            norm_squared += values[dof] * values[dof]
        return norm_squared

    @ti.kernel
    def _solution_inf_norm(self, active_nodes: int) -> float:
        maximum = 0.0
        for block in range(active_nodes):
            for component in ti.static(range(4)):
                if ti.static(component < self.dim):
                    ti.atomic_max(maximum, ti.abs(self.x[block][component]))
        return maximum

    def _set_reduced_non_diag(self, block_i, block_j, block_h):
        pairs_num = int(np.asarray(block_i).shape[0])
        self.non_diag.set_reduced_triplets_from_numpy(block_i, block_j, block_h)
        return pairs_num

    def _make_symmetric_spd_case(self, rng, active_nodes, offdiag_pairs):
        dense = np.zeros((active_nodes * self.dim, active_nodes * self.dim), dtype=np.float64)
        block_i = rng.integers(0, active_nodes, size=offdiag_pairs, dtype=np.int32)
        block_j = rng.integers(0, active_nodes, size=offdiag_pairs, dtype=np.int32)
        mask = block_i != block_j
        block_i = block_i[mask]
        block_j = block_j[mask]
        block_h = np.zeros((block_i.shape[0], 6), dtype=np.float64)
        for k, (bi, bj) in enumerate(zip(block_i, block_j)):
            a = rng.normal(scale=0.01, size=(self.dim, self.dim))
            block = 0.5 * (a + a.T)
            block_h[k] = self._pack_sym_block(block)
            dense[bi * self.dim : (bi + 1) * self.dim, bj * self.dim : (bj + 1) * self.dim] += block
        dense = 0.5 * (dense + dense.T)
        dense += (2.0 + np.sum(np.abs(dense), axis=1).max()) * np.eye(dense.shape[0])
        diag = np.zeros((self.max_active_nodes, 6), dtype=np.float64)
        for i in range(active_nodes):
            diag[i] = self._pack_sym_block(dense[i * self.dim : (i + 1) * self.dim, i * self.dim : (i + 1) * self.dim])
        # Keep the off-diagonal input consistent with the final symmetric dense matrix.
        off_i = []
        off_j = []
        off_h = []
        for i in range(active_nodes):
            for j in range(active_nodes):
                if i == j:
                    continue
                block = dense[i * self.dim : (i + 1) * self.dim, j * self.dim : (j + 1) * self.dim]
                if np.linalg.norm(block) > 0.0:
                    off_i.append(i)
                    off_j.append(j)
                    off_h.append(self._pack_sym_block(block))
        return (
            diag,
            np.asarray(off_i, dtype=np.int32),
            np.asarray(off_j, dtype=np.int32),
            np.asarray(off_h, dtype=np.float64),
        )

    def _make_nonsymmetric_case(self, rng, active_nodes, offdiag_pairs):
        diag = np.zeros((self.max_active_nodes, self.hessian_size), dtype=np.float64)
        for i in range(active_nodes):
            block = rng.normal(scale=0.05, size=(self.dim, self.dim))
            block += (2.0 + self.dim) * np.eye(self.dim)
            diag[i] = block.reshape(-1)
        block_i = rng.integers(0, active_nodes, size=offdiag_pairs, dtype=np.int32)
        block_j = rng.integers(0, active_nodes, size=offdiag_pairs, dtype=np.int32)
        keep = block_i != block_j
        block_i = block_i[keep]
        block_j = block_j[keep]
        block_h = rng.normal(scale=0.02, size=(block_i.shape[0], self.hessian_size))
        return diag, block_i, block_j, block_h

    @staticmethod
    def _pack_sym_block(mat):
        h = np.zeros(6, dtype=np.float64)
        h[0] = mat[0, 0]
        h[1] = mat[1, 1]
        if mat.shape[0] == 3:
            h[2] = mat[2, 2]
            h[4] = 0.5 * (mat[1, 2] + mat[2, 1])
            h[5] = 0.5 * (mat[0, 2] + mat[2, 0])
        h[3] = 0.5 * (mat[0, 1] + mat[1, 0])
        return h

    def _append_block(self, rows, cols, data, bi, bj, h, transpose=False):
        base_i = bi * self.dim
        base_j = bj * self.dim
        if self.symmetric:
            block = np.zeros((self.dim, self.dim), dtype=np.float64)
            block[0, 0] = h[0]
            if self.dim >= 2:
                block[1, 1] = h[1]
                block[0, 1] = block[1, 0] = h[3]
            if self.dim == 3:
                block[2, 2] = h[2]
                block[1, 2] = block[2, 1] = h[4]
                block[0, 2] = block[2, 0] = h[5]
        else:
            block = np.asarray(h, dtype=np.float64).reshape(self.dim, self.dim)
        if transpose:
            block = block.T
        for i in range(self.dim):
            for j in range(self.dim):
                rows.append(base_i + i)
                cols.append(base_j + j)
                data.append(block[i, j])

    def _pad_vector_array(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape((-1, self.dim))
        out = np.zeros((self.max_active_nodes, self.dim), dtype=np.float64)
        out[: values.shape[0], : self.dim] = values[:, : self.dim]
        return out

    @ti.func
    def add_scalar_entry(self, row, col, value, row_comp, col_comp):
        block_i = row // self.dim
        block_j = col // self.dim
        h_index = self.scalar_component_index(row_comp, col_comp)
        h_value = value
        if ti.static(self.symmetric):
            if row_comp != col_comp:
                h_value = 0.5 * value
        if block_i == block_j:
            ti.atomic_add(self.diag[block_i][h_index], h_value)
        else:
            store_i = block_i
            store_j = block_j
            store_row = row_comp
            store_col = col_comp
            if ti.static(self.matrix_symmetric and not self.full_symmetric_input):
                if block_i > block_j:
                    store_i = block_j
                    store_j = block_i
                    store_row = col_comp
                    store_col = row_comp
            h_index = self.scalar_component_index(store_row, store_col)
            idx = ti.atomic_add(self.raw_non_diag_count[0], 1)
            if idx < self.non_diag.blockI.shape[0]:
                self.non_diag.blockI[idx] = store_i
                self.non_diag.blockJ[idx] = store_j
                for k in range(self.hessian_size):
                    self.non_diag.blockH[idx][k] = 0.0
                self.non_diag.blockH[idx][h_index] = h_value
            else:
                self.overflow[0] = 1

    @ti.func
    def add_block_entry(self, block_i, block_j, block):
        if block_i >= 0 and block_j >= 0:
            if block_i == block_j:
                if ti.static(self.symmetric):
                    # Packed symmetric storage has at most 3x3 entries.  Keep
                    # this bounded specialization so scalar-component mapping
                    # remains compile-time exact.
                    for i in ti.static(range(self.dim)):
                        for j in ti.static(range(self.dim)):
                            h_index = self.scalar_component_index(i, j)
                            h_value = block[i, j]
                            if i != j:
                                h_value *= 0.5
                            ti.atomic_add(self.diag[block_i][h_index], h_value)
                else:
                    # A vector atomic keeps all entries in one device
                    # operation.  Taichi 1.7 cannot safely lower a dynamic
                    # component lookup into a vector field from an inlined
                    # contact scatter (LLVM dominance failure on CPU).
                    dense_block = ti.Vector(
                        [block[row, column] for row in range(self.dim) for column in range(self.dim)]
                    )
                    ti.atomic_add(self.diag[block_i], dense_block)
            else:
                store_i = block_i
                store_j = block_j
                transpose_block = False
                if ti.static(self.matrix_symmetric and not self.full_symmetric_input):
                    if block_i > block_j:
                        store_i = block_j
                        store_j = block_i
                        transpose_block = True
                idx = ti.atomic_add(self.raw_non_diag_count[0], 1)
                if idx < self.non_diag.blockI.shape[0]:
                    self.non_diag.blockI[idx] = store_i
                    self.non_diag.blockJ[idx] = store_j
                    if ti.static(self.symmetric):
                        self.non_diag.blockH[idx] = ti.Vector.zero(float, self.hessian_size)
                        for i in ti.static(range(self.dim)):
                            for j in ti.static(range(self.dim)):
                                row_comp = i
                                col_comp = j
                                h_value = block[i, j]
                                if ti.static(self.matrix_symmetric and not self.full_symmetric_input):
                                    if transpose_block:
                                        row_comp = j
                                        col_comp = i
                                h_index = self.scalar_component_index(row_comp, col_comp)
                                if row_comp != col_comp:
                                    h_value *= 0.5
                                self.non_diag.blockH[idx][h_index] = h_value
                    else:
                        dense_block = ti.Vector(
                            [block[row, column] for row in range(self.dim) for column in range(self.dim)]
                        )
                        if ti.static(self.matrix_symmetric and not self.full_symmetric_input):
                            if transpose_block:
                                dense_block = ti.Vector(
                                    [block[column, row] for row in range(self.dim) for column in range(self.dim)]
                                )
                        self.non_diag.blockH[idx] = dense_block
                else:
                    self.overflow[0] = 1

    @ti.func
    def initialize_raw_block_slot(self, raw_index, block_i, block_j):
        """Initialize one already-reserved directed off-diagonal block."""
        if raw_index < self.non_diag.blockI.shape[0]:
            self.non_diag.blockI[raw_index] = block_i
            self.non_diag.blockJ[raw_index] = block_j
            self.non_diag.blockH[raw_index] = ti.Vector.zero(float, self.hessian_size)
        else:
            self.overflow[0] = 1

    @ti.func
    def atomic_add_raw_block_slot(self, raw_index, block):
        """Accumulate a dense block without allocating another raw triplet."""
        if raw_index < self.non_diag.blockH.shape[0]:
            if ti.static(self.symmetric):
                for row, column in ti.static(ti.ndrange(self.dim, self.dim)):
                    value = block[row, column]
                    if row != column:
                        value *= 0.5
                    ti.atomic_add(
                        self.non_diag.blockH[raw_index][self.scalar_component_index(row, column)],
                        value,
                    )
            else:
                dense_block = ti.Vector([block[row, column] for row in range(self.dim) for column in range(self.dim)])
                ti.atomic_add(self.non_diag.blockH[raw_index], dense_block)
        else:
            self.overflow[0] = 1

    @ti.func
    def scalar_component_index(self, row_comp, col_comp) -> ti.i32:
        index = row_comp * self.dim + col_comp
        if ti.static(self.symmetric):
            index = 0
            if row_comp == 0 and col_comp == 0:
                index = 0
            elif row_comp == 1 and col_comp == 1:
                index = 1
            elif row_comp == 2 and col_comp == 2:
                index = 2
            elif (row_comp == 0 and col_comp == 1) or (row_comp == 1 and col_comp == 0):
                index = 3
            elif (row_comp == 1 and col_comp == 2) or (row_comp == 2 and col_comp == 1):
                index = 4
            elif (row_comp == 0 and col_comp == 2) or (row_comp == 2 and col_comp == 0):
                index = 5
        return index

    @ti.kernel
    def _reset_system(self):
        self.raw_non_diag_count[0] = 0
        self.overflow[0] = 0
        for i in self.diag:
            self.diag[i] = ti.Vector.zero(float, self.hessian_size)

    @ti.kernel
    def _append_raw_fields(
        self,
        active_nodes: int,
        block_offset: int,
        source_diag: ti.template(),
        source_i: ti.template(),
        source_j: ti.template(),
        source_h: ti.template(),
        source_count: ti.template(),
        source_overflow: ti.template(),
        scale: float,
    ):
        if source_overflow[0] != 0:
            self.overflow[0] = 1
        # Each source entry owns a deterministic destination slot.  A
        # parallel atomic append makes the raw order nondeterministic on CUDA
        # and defeats the persistent raw-to-reduced pattern cache even when
        # every subsystem stencil is unchanged.
        destination_base = self.raw_non_diag_count[0]
        source_entries = source_count[0]
        destination_end = destination_base + source_entries
        self.raw_non_diag_count[0] = destination_end
        if destination_end > self.non_diag.blockI.shape[0]:
            self.overflow[0] = 1
        for block in range(active_nodes):
            for component in range(self.hessian_size):
                self.diag[block + block_offset][component] += scale * source_diag[block][component]
        for source_index in range(source_count[0]):
            if source_index < source_i.shape[0]:
                destination = destination_base + source_index
                if destination < self.non_diag.blockI.shape[0]:
                    source_block_i = source_i[source_index]
                    source_block_j = source_j[source_index]
                    if source_block_i >= 0 and source_block_j >= 0:
                        self.non_diag.blockI[destination] = source_block_i + block_offset
                        self.non_diag.blockJ[destination] = source_block_j + block_offset
                    else:
                        # Preserve deterministic-stencil holes across a shifted
                        # subsystem merge.  Adding a positive block offset to
                        # the -1 sentinel would otherwise create a fake block.
                        self.non_diag.blockI[destination] = -1
                        self.non_diag.blockJ[destination] = -1
                    for component in range(self.hessian_size):
                        self.non_diag.blockH[destination][component] = scale * source_h[source_index][component]

    @ti.kernel
    def _solver_reset(self, active_nodes: int):
        for i in range(active_nodes):
            z = ti.Vector.zero(float, self.dim)
            self.Ax[i] = z
            self.r[i] = z
            self.z[i] = z
            self.p[i] = z
            self.Ap[i] = z
            self.r_hat[i] = z
            self.v[i] = z
            self.s[i] = z
            self.t[i] = z
            self.p_hat[i] = z
            self.s_hat[i] = z

    @ti.kernel
    def _matvec(self, active_nodes: int, nnz: int, x: ti.template(), Ax: ti.template()):
        for i in range(active_nodes):
            if ti.static(self.symmetric):
                Ax[i] = _sym_block_matvec(self.diag[i], x[i], ti.static(self.dim))
            else:
                Ax[i] = _dense_block_matvec(self.diag[i], x[i], ti.static(self.dim))
        for k in range(nnz):
            i = self.non_diag.tripletI[k]
            j = self.non_diag.tripletJ[k]
            if i < active_nodes and j < active_nodes:
                if ti.static(self.symmetric):
                    Ax[i] += _sym_block_matvec(self.non_diag.tripletH[k], x[j], ti.static(self.dim))
                    if ti.static(self.matrix_symmetric):
                        Ax[j] += _sym_block_matvec(self.non_diag.tripletH[k], x[i], ti.static(self.dim))
                else:
                    Ax[i] += _dense_block_matvec(self.non_diag.tripletH[k], x[j], ti.static(self.dim))
                    if ti.static(self.matrix_symmetric):
                        Ax[j] += _dense_block_transpose_matvec(self.non_diag.tripletH[k], x[i], ti.static(self.dim))

    @ti.kernel
    def _transpose_matvec(self, active_nodes: int, nnz: int, x: ti.template(), Ax: ti.template()):
        for i in range(active_nodes):
            if ti.static(self.symmetric):
                Ax[i] = _sym_block_matvec(self.diag[i], x[i], ti.static(self.dim))
            else:
                Ax[i] = _dense_block_transpose_matvec(self.diag[i], x[i], ti.static(self.dim))
        for k in range(nnz):
            i = self.non_diag.tripletI[k]
            j = self.non_diag.tripletJ[k]
            if i < active_nodes and j < active_nodes:
                if ti.static(self.symmetric):
                    Ax[j] += _sym_block_matvec(self.non_diag.tripletH[k], x[i], ti.static(self.dim))
                else:
                    Ax[j] += _dense_block_transpose_matvec(self.non_diag.tripletH[k], x[i], ti.static(self.dim))

    @ti.kernel
    def _apply_preconditioner(self, active_nodes: int, src: ti.template(), dst: ti.template()):
        for i in range(active_nodes):
            dst[i] = _dense_block_matvec(self.diag_inverse[i], src[i], ti.static(self.dim))

    @ti.kernel
    def _apply_transpose_preconditioner(self, active_nodes: int, src: ti.template(), dst: ti.template()):
        for i in range(active_nodes):
            dst[i] = _dense_block_transpose_matvec(self.diag_inverse[i], src[i], ti.static(self.dim))

    @ti.kernel
    def _build_block_jacobi(self, active_nodes: int):
        for block in range(active_nodes):
            inverse = ti.Vector.zero(float, self.dim * self.dim)
            for column in range(self.dim):
                basis = ti.Vector.zero(float, self.dim)
                basis[column] = 1.0
                result = ti.Vector.zero(float, self.dim)
                if ti.static(self.symmetric):
                    result = _apply_sym_block_jacobi(self.diag[block], basis, ti.static(self.dim))
                else:
                    result = _apply_dense_block_jacobi(self.diag[block], basis, ti.static(self.dim))
                for row in range(self.dim):
                    inverse[row * self.dim + column] = result[row]
            self.diag_inverse[block] = inverse

    @ti.kernel
    def _dot(self, active_nodes: int, a: ti.template(), b: ti.template()) -> float:
        result = 0.0
        for i in range(active_nodes):
            for d in ti.static(range(4)):
                if ti.static(d < self.dim):
                    result += a[i][d] * b[i][d]
        return result

    @ti.kernel
    def _copy(self, active_nodes: int, src: ti.template(), dst: ti.template()):
        for i in range(active_nodes):
            dst[i] = src[i]

    @ti.kernel
    def _init_pcg(self, active_nodes: int) -> tuple[float, float]:
        rz, rr = 0.0, 0.0
        for i in range(active_nodes):
            residual = self.rhs[i] - self.Ax[i]
            preconditioned = _dense_block_matvec(self.diag_inverse[i], residual, ti.static(self.dim))
            self.r[i], self.z[i], self.p[i] = residual, preconditioned, preconditioned
            for d in ti.static(range(self.dim)):
                rz += residual[d] * preconditioned[d]
                rr += residual[d] * residual[d]
        return rz, rr

    @ti.kernel
    def _pcg_update_and_reduce(self, active_nodes: int, alpha: float) -> tuple[float, float]:
        rz, rr = 0.0, 0.0
        for i in range(active_nodes):
            self.x[i] += alpha * self.p[i]
            residual = self.r[i] - alpha * self.Ap[i]
            preconditioned = _dense_block_matvec(self.diag_inverse[i], residual, ti.static(self.dim))
            self.r[i], self.z[i] = residual, preconditioned
            for d in ti.static(range(self.dim)):
                rz += residual[d] * preconditioned[d]
                rr += residual[d] * residual[d]
        return rz, rr

    @ti.kernel
    def _pcg_update_p(self, active_nodes: int, beta: float):
        for i in range(active_nodes):
            self.p[i] = self.z[i] + beta * self.p[i]

    @ti.kernel
    def _init_bicgstab(self, active_nodes: int):
        for i in range(active_nodes):
            self.r[i] = self.rhs[i] - self.Ax[i]
            self.r_hat[i] = self.r[i]
            self.p[i] = self.r[i]

    @ti.kernel
    def _bicg_update_p(self, active_nodes: int, beta: float, omega: float):
        for i in range(active_nodes):
            self.p[i] = self.r[i] + beta * (self.p[i] - omega * self.v[i])

    @ti.kernel
    def _bicg_update_s(self, active_nodes: int, alpha: float):
        for i in range(active_nodes):
            self.s[i] = self.r[i] - alpha * self.v[i]

    @ti.kernel
    def _bicg_update_x_alpha(self, active_nodes: int, alpha: float):
        for i in range(active_nodes):
            self.x[i] += alpha * self.p_hat[i]

    @ti.kernel
    def _bicg_update_x_r(self, active_nodes: int, alpha: float, omega: float):
        for i in range(active_nodes):
            self.x[i] += alpha * self.p_hat[i] + omega * self.s_hat[i]
            self.r[i] = self.s[i] - omega * self.t[i]

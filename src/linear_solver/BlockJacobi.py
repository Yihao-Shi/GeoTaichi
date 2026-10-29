"""Device block Jacobi for an affine-control prefix and ordinary node blocks."""

import taichi as ti

from src.linear_solver.BuildTriplet import _apply_dense_block_jacobi


class BlockJacobiError(RuntimeError):
    pass


@ti.data_oriented
class BlockJacobi:
    def __init__(self, dim, max_nodes, group_size, group_nodes=None, diagonal_inverse=None):
        self.dim, self.group_size = dim, group_size
        self.max_nodes = max_nodes
        self.group_nodes = group_nodes
        capacity = max_nodes if group_nodes is None else group_nodes
        self.factor = ti.field(float, shape=(max(capacity // group_size, 1), dim * group_size, dim * group_size))
        self.scale = ti.field(float, shape=max(capacity // group_size, 1))
        self.permutation = ti.field(ti.i32, shape=(max(capacity // group_size, 1), dim * group_size))
        self.error = ti.field(ti.i32, shape=())
        self.diagonal_inverse = diagonal_inverse
        if diagonal_inverse is None:
            self.diagonal_inverse = ti.Vector.field(dim * dim, float, shape=max_nodes)
        self.cholesky = True

    def grouped_nodes(self, active_nodes):
        if not 0 <= active_nodes <= self.max_nodes:
            raise ValueError("Block preconditioner active nodes exceed matrix capacity")
        nodes = active_nodes if self.group_nodes is None else self.group_nodes
        if nodes > active_nodes or nodes % self.group_size:
            raise ValueError("Block preconditioning requires a complete affine prefix within the active nodes")
        return nodes

    def build_triplet(self, matrix, active_nodes, nnz, cholesky=True):
        nodes = self.grouped_nodes(active_nodes)
        self._clear(active_nodes, nodes)
        self._extract_triplet(matrix, active_nodes, nodes, nnz)
        self._finish(active_nodes, nodes, cholesky)

    def build_coo(self, matrix, active_nodes, cholesky=True):
        nodes = self.grouped_nodes(active_nodes)
        self._clear(active_nodes, nodes)
        self._extract_coo(matrix, nodes, int(matrix.linear_operator.nnz))
        self._finish(active_nodes, nodes, cholesky)

    def _finish(self, active_nodes, nodes, cholesky):
        self.cholesky = bool(cholesky)
        self._factor(nodes, self.cholesky)
        if self.error[None]:
            kind = "non-finite or non-positive Cholesky pivot" if cholesky else "non-finite or singular LU pivot"
            raise BlockJacobiError(f"Block Jacobi linear solve preconditioner contains a {kind}")
        self._invert_nodes(active_nodes, nodes)

    def apply(self, active_nodes, source, destination, transpose=False, flat=False):
        self._apply(active_nodes, self.grouped_nodes(active_nodes), source, destination, transpose, flat, self.cholesky)

    @ti.kernel
    def _clear(self, active_nodes: int, nodes: int):
        self.error[None] = 0
        for node in range(active_nodes):
            self.diagonal_inverse[node] = ti.Vector.zero(float, self.dim * self.dim)
        width = ti.static(self.dim * self.group_size)
        for group, i, j in ti.ndrange(nodes // self.group_size, width, width):
            self.factor[group, i, j] = 0.0

    @ti.kernel
    def _extract_triplet(self, matrix: ti.template(), active_nodes: int, nodes: int, nnz: int):
        for node in range(active_nodes):
            self.diagonal_inverse[node] = matrix.diag[node]
            if node < nodes:
                group, offset = node // self.group_size, (node % self.group_size) * self.dim
                for i, j in ti.ndrange(self.dim, self.dim):
                    self.factor[group, offset + i, offset + j] = matrix.diag[node][i * self.dim + j]
        for slot in range(nnz):
            row, column = matrix.non_diag.tripletI[slot], matrix.non_diag.tripletJ[slot]
            if row < nodes and column < nodes and row // self.group_size == column // self.group_size:
                group = row // self.group_size
                r, c = (row % self.group_size) * self.dim, (column % self.group_size) * self.dim
                for i, j in ti.ndrange(self.dim, self.dim):
                    value = matrix.non_diag.tripletH[slot][i * self.dim + j]
                    self.factor[group, r + i, c + j] += value
                    if ti.static(matrix.matrix_symmetric):
                        self.factor[group, c + j, r + i] += value

    @ti.kernel
    def _extract_coo(self, matrix: ti.template(), nodes: int, nnz: int):
        for slot in range(nnz):
            row, column, value = matrix.rows[slot], matrix.cols[slot], matrix.data[slot]
            r, c = row // self.dim, column // self.dim
            if r == c:
                self.diagonal_inverse[r][(row % self.dim) * self.dim + column % self.dim] += value
            if r < nodes and c < nodes and r // self.group_size == c // self.group_size:
                width = ti.static(self.dim * self.group_size)
                self.factor[r // self.group_size, row % width, column % width] += value

    @ti.kernel
    def _factor(self, nodes: int, cholesky: ti.template()):
        width = ti.static(self.dim * self.group_size)
        for group in range(nodes // self.group_size):
            scale = 0.0
            for i, j in ti.ndrange(width, width):
                scale = ti.max(scale, ti.abs(self.factor[group, i, j]))
            if not (0.0 < scale and scale < ti.math.inf):
                ti.atomic_max(self.error[None], 1)
                scale = 1.0
            self.scale[group] = scale
            for i, j in ti.ndrange(width, width):
                self.factor[group, i, j] /= scale
            if ti.static(cholesky):
                for i in range(width):
                    for j in range(i + 1):
                        value = self.factor[group, i, j]
                        for k in range(j):
                            value -= self.factor[group, i, k] * self.factor[group, j, k]
                        if i == j:
                            if not (0.0 < value and value < ti.math.inf):
                                ti.atomic_max(self.error[None], 1)
                                value = 1.0
                            self.factor[group, i, j] = ti.sqrt(value)
                        else:
                            if not (-ti.math.inf < value and value < ti.math.inf):
                                ti.atomic_max(self.error[None], 1)
                                value = 0.0
                            self.factor[group, i, j] = value / self.factor[group, j, j]
            else:
                for i in range(width):
                    self.permutation[group, i] = i
                for k in range(width):
                    pivot = k
                    magnitude = ti.abs(self.factor[group, k, k])
                    for i in range(k + 1, width):
                        candidate = ti.abs(self.factor[group, i, k])
                        if candidate > magnitude:
                            pivot, magnitude = i, candidate
                    if not (0.0 < magnitude and magnitude < ti.math.inf):
                        ti.atomic_max(self.error[None], 1)
                    if pivot != k:
                        for j in range(width):
                            value = self.factor[group, k, j]
                            self.factor[group, k, j] = self.factor[group, pivot, j]
                            self.factor[group, pivot, j] = value
                        value = self.permutation[group, k]
                        self.permutation[group, k] = self.permutation[group, pivot]
                        self.permutation[group, pivot] = value
                    diagonal = self.factor[group, k, k]
                    if not (0.0 < ti.abs(diagonal) < ti.math.inf):
                        ti.atomic_max(self.error[None], 1)
                        diagonal = 1.0
                    for i in range(k + 1, width):
                        self.factor[group, i, k] /= diagonal
                        for j in range(k + 1, width):
                            self.factor[group, i, j] -= self.factor[group, i, k] * self.factor[group, k, j]
                for i, j in ti.ndrange(width, width):
                    if not (-ti.math.inf < self.factor[group, i, j] < ti.math.inf):
                        ti.atomic_max(self.error[None], 1)

    @ti.kernel
    def _invert_nodes(self, active_nodes: int, nodes: int):
        for node in range(nodes, active_nodes):
            diagonal = self.diagonal_inverse[node]
            inverse = ti.Vector.zero(float, self.dim * self.dim)
            for column in range(self.dim):
                basis = ti.Vector.zero(float, self.dim)
                basis[column] = 1.0
                result = _apply_dense_block_jacobi(diagonal, basis, ti.static(self.dim))
                for row in range(self.dim):
                    inverse[row * self.dim + column] = result[row]
            self.diagonal_inverse[node] = inverse

    @ti.func
    def _read(self, source: ti.template(), node, component, flat: ti.template()):
        value = 0.0
        if ti.static(flat):
            value = source[node * self.dim + component]
        else:
            value = source[node][component]
        return value

    @ti.func
    def _write(self, destination: ti.template(), node, component, value, flat: ti.template()):
        if ti.static(flat):
            destination[node * self.dim + component] = value
        else:
            destination[node][component] = value

    @ti.kernel
    def _apply(
        self,
        active_nodes: int,
        nodes: int,
        source: ti.template(),
        destination: ti.template(),
        transpose: ti.template(),
        flat: ti.template(),
        cholesky: ti.template(),
    ):
        width = ti.static(self.dim * self.group_size)
        for group in range(nodes // self.group_size):
            values = ti.Vector.zero(float, width)
            for i in range(width):
                row = i
                if ti.static(not cholesky and not transpose):
                    row = self.permutation[group, i]
                value = self._read(source, group * self.group_size + row // self.dim, row % self.dim, flat)
                value /= self.scale[group]
                for j in range(i):
                    coefficient = self.factor[group, i, j]
                    if ti.static(not cholesky and transpose):
                        coefficient = self.factor[group, j, i]
                    value -= coefficient * values[j]
                if ti.static(cholesky or transpose):
                    value /= self.factor[group, i, i]
                values[i] = value
            for reverse in range(width):
                i = width - 1 - reverse
                value = values[i]
                for j in range(i + 1, width):
                    coefficient = self.factor[group, j, i]
                    if ti.static(not cholesky and not transpose):
                        coefficient = self.factor[group, i, j]
                    value -= coefficient * values[j]
                if ti.static(cholesky or not transpose):
                    value /= self.factor[group, i, i]
                values[i] = value
            for i in range(width):
                row = i
                if ti.static(not cholesky and transpose):
                    row = self.permutation[group, i]
                self._write(destination, group * self.group_size + row // self.dim, row % self.dim, values[i], flat)
        for node in range(nodes, active_nodes):
            input_value = ti.Vector.zero(float, self.dim)
            for j in ti.static(range(self.dim)):
                input_value[j] = self._read(source, node, j, flat)
            for i in ti.static(range(self.dim)):
                value = 0.0
                for j in ti.static(range(self.dim)):
                    entry = i * self.dim + j
                    if ti.static(transpose):
                        entry = j * self.dim + i
                    value += self.diagonal_inverse[node][entry] * input_value[j]
                self._write(destination, node, i, value, flat)

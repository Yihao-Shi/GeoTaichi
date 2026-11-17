import numpy as np


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

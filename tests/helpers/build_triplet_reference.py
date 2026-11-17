"""Small block systems for solver validation."""

import numpy as np
from src.linear_solver.BuildTriplet import BuildTriplet


class ReferenceBuildTriplet(BuildTriplet):
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

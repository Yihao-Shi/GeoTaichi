"""Dense oracles for runtime-loop block assembly and reduction."""

import numpy as np
import pytest
import taichi as ti

from src.linear_solver.BuildTriplet import BuildTriplet
from src.linear_solver.HashReduction import HashReduction


pytestmark = [
    pytest.mark.cpu,
    pytest.mark.linear_solver,
    pytest.mark.hash_triplet,
]


def _accumulate_blocks(block_i, block_j, block_h, block_count, dimension):
    dense = np.zeros(
        (block_count * dimension, block_count * dimension), dtype=np.float64
    )
    for row, column, block in zip(block_i, block_j, block_h):
        if row < 0 or column < 0:
            continue
        row_slice = slice(row * dimension, (row + 1) * dimension)
        column_slice = slice(
            column * dimension, (column + 1) * dimension
        )
        dense[row_slice, column_slice] += block.reshape(
            (dimension, dimension)
        )
    return dense


@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_random_runtime_block_pipeline_matches_dense_oracle(
    taichi_runtime, dimension
):
    """raw/upload/add/append/reduce/matvec/Jacobi agree for dim=2,3,4."""

    rng = np.random.default_rng(4100 + dimension)
    block_count = 3
    hessian_size = dimension * dimension
    raw_i = np.asarray([0, 0, 1, 1, 2, 2, 2, -1], dtype=np.int32)
    raw_j = np.asarray([1, 1, 0, 2, 0, 0, 1, -1], dtype=np.int32)
    raw_blocks = rng.normal(
        scale=0.2, size=(raw_i.size, dimension, dimension)
    )

    diagonal_blocks = np.empty(
        (block_count, dimension, dimension), dtype=np.float64
    )
    for node in range(block_count):
        factor = rng.normal(size=(dimension, dimension))
        diagonal_blocks[node] = (
            factor.T @ factor
            + (dimension + node + 2.0) * np.eye(dimension)
        )

    source = BuildTriplet(
        dim=dimension,
        max_pairs_num=raw_i.size,
        max_nonzeros=raw_i.size,
        max_active_nodes=block_count,
        symmetric=False,
        device_reduction=False,
    )
    assembled = BuildTriplet(
        dim=dimension,
        max_pairs_num=raw_i.size,
        max_nonzeros=raw_i.size,
        max_active_nodes=block_count,
        symmetric=False,
        device_reduction=False,
    )

    @ti.kernel
    def add_raw_blocks(
        diagonal: ti.types.ndarray(),
        rows: ti.types.ndarray(),
        columns: ti.types.ndarray(),
        values: ti.types.ndarray(),
        raw_count: ti.i32,
    ):
        for node in range(block_count):
            block = ti.Matrix(
                [
                    [
                        diagonal[node, row, column]
                        for column in range(dimension)
                    ]
                    for row in range(dimension)
                ]
            )
            source.add_block_entry(node, node, block)

        for raw in range(raw_count):
            block = ti.Matrix(
                [
                    [
                        values[raw, row, column]
                        for column in range(dimension)
                    ]
                    for row in range(dimension)
                ]
            )
            source.add_block_entry(rows[raw], columns[raw], block)

    source.reset_system()
    assembled.reset_system()
    add_raw_blocks(
        diagonal_blocks,
        raw_i,
        raw_j,
        raw_blocks,
        raw_i.size,
    )
    assert int(source.raw_non_diag_count[0]) == raw_i.size - 1

    # Coupled systems append raw Taichi fields first, then reduce only once.
    assembled.append_raw_from(source, active_nodes=block_count)
    assert int(assembled.raw_non_diag_count[0]) == raw_i.size - 1
    assembled.finalize_taichi_assembly()

    expected = _accumulate_blocks(
        np.arange(block_count, dtype=np.int32),
        np.arange(block_count, dtype=np.int32),
        diagonal_blocks,
        block_count,
        dimension,
    )
    expected += _accumulate_blocks(
        raw_i, raw_j, raw_blocks, block_count, dimension
    )
    np.testing.assert_allclose(
        assembled.to_scipy(block_count).toarray(),
        expected,
        rtol=2.0e-13,
        atol=2.0e-13,
    )

    # Exercise the raw ndarray upload and the device-resident duplicate
    # reduction, including all 16 components for the dim=4 specialization.
    reduction = HashReduction(
        max_pairs_num=raw_i.size,
        dim=dimension,
        max_nnz=raw_i.size,
        hessian_size=hessian_size,
        device_reduction=True,
    )
    uploaded_count = reduction.set_triplets_from_numpy(
        raw_i, raw_j, raw_blocks.reshape((raw_i.size, hessian_size))
    )
    reduction.go(uploaded_count)
    reduced_i, reduced_j, reduced_h = reduction.get_reduced_triplets_numpy()
    reduced_dense = _accumulate_blocks(
        reduced_i,
        reduced_j,
        reduced_h,
        block_count,
        dimension,
    )
    expected_off_diagonal = _accumulate_blocks(
        raw_i, raw_j, raw_blocks, block_count, dimension
    )
    np.testing.assert_allclose(
        reduced_dense,
        expected_off_diagonal,
        rtol=2.0e-13,
        atol=2.0e-13,
    )

    vector = rng.normal(size=(block_count, dimension))
    assembled.x.from_numpy(vector)
    nnz = int(assembled.non_diag.element_pair_num[0])
    assembled.matvec(block_count, nnz, assembled.x, assembled.Ax)
    np.testing.assert_allclose(
        assembled.Ax.to_numpy()[:block_count],
        (expected @ vector.reshape(-1)).reshape((block_count, dimension)),
        rtol=2.0e-13,
        atol=2.0e-13,
    )

    residual = rng.normal(size=(block_count, dimension))
    assembled.r.from_numpy(residual)
    assembled._build_block_jacobi(block_count)
    assembled._apply_preconditioner(
        block_count, assembled.r, assembled.z
    )
    expected_preconditioned = np.stack(
        [
            np.linalg.solve(diagonal_blocks[node], residual[node])
            for node in range(block_count)
        ]
    )
    np.testing.assert_allclose(
        assembled.z.to_numpy()[:block_count],
        expected_preconditioned,
        rtol=2.0e-12,
        atol=2.0e-12,
    )

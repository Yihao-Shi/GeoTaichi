"""Correctness checks for the Taichi block-pattern cache."""

import numpy as np
import pytest
import taichi as ti
from scipy.sparse import coo_matrix

from src.linear_solver.BuildTriplet import BuildTriplet
from src.linear_solver.HashReduction import HashReduction


def _component_matrices(block_i, block_j, block_h, block_count):
    return [
        coo_matrix(
            (block_h[:, component], (block_i, block_j)),
            shape=(block_count, block_count),
        ).tocsr()
        for component in range(block_h.shape[1])
    ]


def _assert_reduction_matches(reduction, block_i, block_j, block_h, block_count):
    out_i, out_j, out_h = reduction.get_reduced_triplets_numpy()
    actual = _component_matrices(out_i, out_j, out_h, block_count)
    expected = _component_matrices(block_i, block_j, block_h, block_count)
    maximum_error = 0.0
    for actual_component, expected_component in zip(actual, expected):
        difference = (actual_component - expected_component).tocoo()
        if difference.nnz:
            maximum_error = max(maximum_error, float(np.max(np.abs(difference.data))))
    assert maximum_error <= 1.0e-11, maximum_error


def _run_device_pattern_checks():
    rng = np.random.default_rng(23)
    block_count = 31
    hessian_size = 9
    pair_count = 512
    block_i = rng.integers(0, block_count, size=pair_count, dtype=np.int32)
    block_j = rng.integers(0, block_count, size=pair_count, dtype=np.int32)
    block_h = rng.normal(size=(pair_count, hessian_size))

    reduction = HashReduction(
        max_pairs_num=pair_count + 64,
        max_nnz=block_count * block_count,
        hessian_size=hessian_size,
        pattern_cache=True,
        pattern_cache_extra_fraction=0.01,
        pattern_cache_max_age=3,
        # Exercise the same Taichi kernels on CPU-only CI. On CUDA this flag
        # selects the actual GPU implementation used by every BuildTriplet.
        device_reduction=True,
    )

    reduction.set_triplets_from_numpy(block_i, block_j, block_h)
    reduction.go(pair_count)
    _assert_reduction_matches(reduction, block_i, block_j, block_h, block_count)
    first = reduction.pattern_cache_statistics()
    assert first["pattern_rebuilds"] == 1
    assert first["last_new_entries"] > 0
    assert first["last_mapping_misses"] == pair_count

    # Stable FEM/MPM raw ordering must bypass the open-addressing table. Only
    # the cached raw->reduced slot is validated before direct value scatter.
    stable_h = 0.5 * block_h
    reduction.set_triplets_from_numpy(block_i, block_j, stable_h)
    reduction.go(pair_count)
    _assert_reduction_matches(reduction, block_i, block_j, stable_h, block_count)
    stable = reduction.pattern_cache_statistics()
    assert stable["pattern_version"] == first["pattern_version"]
    assert stable["pattern_hits"] == first["pattern_hits"] + 1
    assert stable["last_mapping_misses"] == 0

    permutation = rng.permutation(pair_count)
    second_h = 1.7 * block_h[permutation]
    second_i = block_i[permutation]
    second_j = block_j[permutation]
    reduction.set_triplets_from_numpy(second_i, second_j, second_h)
    reduction.go(pair_count)
    _assert_reduction_matches(reduction, second_i, second_j, second_h, block_count)
    second = reduction.pattern_cache_statistics()
    assert second["pattern_version"] == stable["pattern_version"]
    assert second["pattern_hits"] == stable["pattern_hits"] + 1
    assert second["last_mapping_misses"] > 0

    # Removing most entries forces bounded-superset compaction. Old cached
    # blocks must not retain values from the previous Newton iteration.
    keep = np.arange(pair_count) < pair_count // 5
    reduced_i = second_i[keep]
    reduced_j = second_j[keep]
    reduced_h = second_h[keep]
    reduction.set_triplets_from_numpy(reduced_i, reduced_j, reduced_h)
    reduction.go(reduced_i.size)
    _assert_reduction_matches(reduction, reduced_i, reduced_j, reduced_h, block_count)
    compacted = reduction.pattern_cache_statistics()
    assert compacted["pattern_rebuilds"] > second["pattern_rebuilds"]
    assert compacted["pattern_nonzeros"] == compacted["current_nonzeros"]

    # A genuinely new block changes the pattern once and is accumulated on
    # device together with duplicate contributions.
    new_i = np.concatenate((reduced_i, np.asarray([30, 30], dtype=np.int32)))
    new_j = np.concatenate((reduced_j, np.asarray([0, 0], dtype=np.int32)))
    new_h = np.concatenate((reduced_h, np.ones((2, hessian_size))), axis=0)
    reduction.set_triplets_from_numpy(new_i, new_j, new_h)
    reduction.go(new_i.size)
    _assert_reduction_matches(reduction, new_i, new_j, new_h, block_count)
    extended = reduction.pattern_cache_statistics()
    assert extended["pattern_version"] > compacted["pattern_version"]

    # A genuinely oversized current pattern must fail promptly.  In
    # particular, retrying after dropping stale cache entries must not turn an
    # overflow into an unbounded probing loop on the device.
    overflow = HashReduction(
        max_pairs_num=32,
        max_nnz=8,
        hessian_size=hessian_size,
        pattern_cache=True,
        device_reduction=True,
    )
    overflow_i = np.arange(16, dtype=np.int32)
    overflow_j = np.zeros(16, dtype=np.int32)
    overflow_h = np.ones((16, hessian_size), dtype=np.float64)
    overflow.set_triplets_from_numpy(overflow_i, overflow_j, overflow_h)
    try:
        overflow.go(overflow_i.size)
    except RuntimeError as error:
        assert "overflow" in str(error).lower()
    else:
        raise AssertionError("device pattern overflow was not reported")

    # Many CUDA blocks may publish the same coordinate concurrently.  The
    # reduced pattern still contains exactly one entry.
    duplicate_count = 4096
    duplicate = HashReduction(
        max_pairs_num=duplicate_count,
        max_nnz=1,
        hessian_size=1,
        pattern_cache=False,
        device_reduction=True,
    )
    duplicate_i = np.zeros(duplicate_count, dtype=np.int32)
    duplicate_j = np.ones(duplicate_count, dtype=np.int32)
    duplicate_h = np.ones((duplicate_count, 1), dtype=np.float64)
    duplicate.set_triplets_from_numpy(duplicate_i, duplicate_j, duplicate_h)
    duplicate.go(duplicate_count)
    assert int(duplicate.element_pair_num[0]) == 1
    assert float(duplicate.tripletH[0][0]) == duplicate_count

    # The coupled MPM--ABD assembly mixes tens of thousands of mostly unique
    # stencil blocks with duplicates.  It must not report a false hash overflow.
    stress_unique = 27000
    stress_count = 35000
    stress_i = np.arange(stress_unique, dtype=np.int32)
    stress_j = (17 * stress_i + 3).astype(np.int32)
    duplicate_indices = rng.integers(0, stress_unique, size=stress_count - stress_unique)
    stress_i = np.concatenate((stress_i, stress_i[duplicate_indices]))
    stress_j = np.concatenate((stress_j, stress_j[duplicate_indices]))
    permutation = rng.permutation(stress_count)
    stress = HashReduction(
        max_pairs_num=stress_count,
        max_nnz=stress_unique,
        hessian_size=1,
        pattern_cache=False,
        device_reduction=True,
    )
    stress.set_triplets_from_numpy(
        stress_i[permutation],
        stress_j[permutation],
        np.ones((stress_count, 1), dtype=np.float64),
    )
    stress.go(stress_count)
    assert int(stress.element_pair_num[0]) == stress_unique
    assert np.isclose(float(stress.tripletH.to_numpy().sum()), stress_count)


def _run_build_triplet_checks():
    rng = np.random.default_rng(9)
    active_nodes = 8
    matrix = BuildTriplet(
        dim=3,
        max_pairs_num=96,
        max_nonzeros=64,
        max_active_nodes=active_nodes,
        symmetric=False,
        matrix_symmetric=False,
        device_reduction=True,
    )
    diagonal = np.zeros((active_nodes, 9), dtype=np.float64)
    for node in range(active_nodes):
        block = rng.normal(size=(3, 3))
        block += (5.0 + node) * np.eye(3)
        diagonal[node] = block.reshape(-1)
    matrix.diag.from_numpy(diagonal)

    pair_count = 72
    block_i = rng.integers(0, active_nodes, size=pair_count, dtype=np.int32)
    block_j = rng.integers(0, active_nodes, size=pair_count, dtype=np.int32)
    off_diagonal = block_i != block_j
    block_i = block_i[off_diagonal]
    block_j = block_j[off_diagonal]
    block_h = rng.normal(size=(block_i.size, 9))
    matrix.non_diag.set_triplets_from_numpy(block_i, block_j, block_h)
    matrix.non_diag.go(block_i.size)

    assembled = matrix.to_scipy(active_nodes)
    reference = np.zeros((3 * active_nodes, 3 * active_nodes), dtype=np.float64)
    for node in range(active_nodes):
        reference[3 * node : 3 * node + 3, 3 * node : 3 * node + 3] += diagonal[node].reshape(3, 3)
    for row, col, values in zip(block_i, block_j, block_h):
        reference[3 * row : 3 * row + 3, 3 * col : 3 * col + 3] += values.reshape(3, 3)
    np.testing.assert_allclose(assembled.toarray(), reference, rtol=1.0e-12, atol=1.0e-12)

    # Reusing the same pattern must reuse both the device block pattern and the
    # scalar conversion plan used only by direct SciPy compatibility solves.
    matrix.non_diag.set_triplets_from_numpy(block_i[::-1], block_j[::-1], block_h[::-1])
    matrix.non_diag.go(block_i.size)
    matrix.to_scipy(active_nodes)
    statistics = matrix.acceleration_statistics()
    assert statistics["block_pattern"]["pattern_hits"] >= 1
    assert statistics["scipy_pattern_hits"] >= 1

    flat_rhs = ti.field(ti.f64, shape=3 * active_nodes)
    flat_solution = ti.field(ti.f64, shape=3 * active_nodes)
    rhs_values = rng.normal(size=3 * active_nodes)
    flat_rhs.from_numpy(rhs_values)
    solve_result = matrix.solve_flat_system(
        flat_rhs,
        flat_solution,
        active_nodes=active_nodes,
        tol=1.0e-11,
        maxiter=500,
    )
    assert solve_result["converged"], solve_result
    reference_solution = np.linalg.solve(reference, rhs_values)
    np.testing.assert_allclose(flat_solution.to_numpy(), reference_solution, rtol=1.0e-8, atol=1.0e-9)

    # Coupled solvers append body/contact systems directly between Taichi
    # fields and perform one reduction for the monolithic matrix.
    source_a = BuildTriplet(
        dim=3,
        max_pairs_num=8,
        max_nonzeros=8,
        max_active_nodes=2,
        symmetric=False,
        device_reduction=True,
    )
    source_b = BuildTriplet(
        dim=3,
        max_pairs_num=8,
        max_nonzeros=8,
        max_active_nodes=2,
        symmetric=False,
        device_reduction=True,
    )
    coupled = BuildTriplet(
        dim=3,
        max_pairs_num=16,
        max_nonzeros=16,
        max_active_nodes=4,
        symmetric=False,
        device_reduction=True,
    )
    a_diag = np.zeros((2, 9), dtype=np.float64)
    b_diag = np.zeros((2, 9), dtype=np.float64)
    a_diag[0] = (2.0 * np.eye(3)).reshape(-1)
    a_diag[1] = (3.0 * np.eye(3)).reshape(-1)
    b_diag[0] = (5.0 * np.eye(3)).reshape(-1)
    b_diag[1] = (7.0 * np.eye(3)).reshape(-1)
    source_a.diag.from_numpy(a_diag)
    source_b.diag.from_numpy(b_diag)
    a_block = np.arange(1.0, 10.0, dtype=np.float64).reshape(1, 9)
    b_block = -0.25 * a_block
    source_a.non_diag.set_triplets_from_numpy(
        np.asarray([0], dtype=np.int32),
        np.asarray([1], dtype=np.int32),
        a_block,
    )
    source_b.non_diag.set_triplets_from_numpy(
        np.asarray([1], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
        b_block,
    )
    # Fixed stencils retain unused slots as (-1, -1). Appending with a block
    # offset must preserve that sentinel instead of turning it into a real row.
    source_a.non_diag.blockI[1] = -1
    source_a.non_diag.blockJ[1] = -1
    source_a.raw_non_diag_count[0] = 2
    source_b.raw_non_diag_count[0] = 1
    coupled.reset_system()
    coupled.append_raw_from(source_a, active_nodes=2)
    coupled.append_raw_from(source_b, active_nodes=2, block_offset=2)
    first_raw_count = int(coupled.raw_non_diag_count[0])
    first_raw_i = coupled.non_diag.blockI.to_numpy()[:first_raw_count].copy()
    first_raw_j = coupled.non_diag.blockJ.to_numpy()[:first_raw_count].copy()
    # Each source keeps its own source-index order, and successive systems own
    # consecutive ranges. This must not depend on CUDA thread scheduling.
    np.testing.assert_array_equal(first_raw_i, np.asarray([0, -1, 3], dtype=np.int32))
    np.testing.assert_array_equal(first_raw_j, np.asarray([1, -1, 2], dtype=np.int32))
    coupled.finalize_taichi_assembly()
    coupled_dense = coupled.to_scipy(active_nodes=4).toarray()
    expected_coupled = np.zeros((12, 12), dtype=np.float64)
    expected_coupled[0:3, 0:3] = 2.0 * np.eye(3)
    expected_coupled[3:6, 3:6] = 3.0 * np.eye(3)
    expected_coupled[6:9, 6:9] = 5.0 * np.eye(3)
    expected_coupled[9:12, 9:12] = 7.0 * np.eye(3)
    expected_coupled[0:3, 3:6] = a_block.reshape(3, 3)
    expected_coupled[9:12, 6:9] = b_block.reshape(3, 3)
    np.testing.assert_allclose(coupled_dense, expected_coupled, atol=1.0e-12)

    scaled = BuildTriplet(
        dim=3,
        max_pairs_num=8,
        max_nonzeros=8,
        max_active_nodes=2,
        symmetric=False,
        device_reduction=True,
    )
    scaled.reset_system()
    scaled.append_raw_from(source_a, active_nodes=2, scale=-0.25)
    scaled.finalize_taichi_assembly()
    expected_scaled = np.zeros((6, 6), dtype=np.float64)
    expected_scaled[0:3, 0:3] = -0.5 * np.eye(3)
    expected_scaled[3:6, 3:6] = -0.75 * np.eye(3)
    expected_scaled[0:3, 3:6] = -0.25 * a_block.reshape(3, 3)
    np.testing.assert_allclose(
        scaled.to_scipy(active_nodes=2).toarray(),
        expected_scaled,
        atol=1.0e-12,
    )

    first_coupled_stats = coupled.acceleration_statistics()["block_pattern"]
    source_a.diag.from_numpy(2.0 * a_diag)
    source_b.diag.from_numpy(2.0 * b_diag)
    source_a.non_diag.set_triplets_from_numpy(
        np.asarray([0], dtype=np.int32),
        np.asarray([1], dtype=np.int32),
        2.0 * a_block,
    )
    source_b.non_diag.set_triplets_from_numpy(
        np.asarray([1], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
        2.0 * b_block,
    )
    coupled.reset_system()
    coupled.append_raw_from(source_a, active_nodes=2)
    coupled.append_raw_from(source_b, active_nodes=2, block_offset=2)
    second_raw_count = int(coupled.raw_non_diag_count[0])
    np.testing.assert_array_equal(coupled.non_diag.blockI.to_numpy()[:second_raw_count], first_raw_i)
    np.testing.assert_array_equal(coupled.non_diag.blockJ.to_numpy()[:second_raw_count], first_raw_j)
    coupled.finalize_taichi_assembly()
    second_coupled_stats = coupled.acceleration_statistics()["block_pattern"]
    assert second_coupled_stats["pattern_version"] == first_coupled_stats["pattern_version"]
    assert second_coupled_stats["last_mapping_misses"] == 0
    np.testing.assert_allclose(
        coupled.to_scipy(active_nodes=4).toarray(),
        2.0 * expected_coupled,
        atol=1.0e-12,
    )

    flat_values = ti.field(ti.f64, shape=12)
    norm_values = np.linspace(-1.0, 1.0, 12)
    flat_values.from_numpy(norm_values)
    np.testing.assert_allclose(
        coupled.flat_l2_norm(flat_values, 12),
        np.linalg.norm(norm_values),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def _run_full_symmetric_input_checks():
    """A full two-triangle source must become one exact symmetric structure."""
    active_nodes = 3
    source = BuildTriplet(
        dim=3,
        max_pairs_num=8,
        max_nonzeros=8,
        max_active_nodes=active_nodes,
        symmetric=False,
        matrix_symmetric=False,
        device_reduction=True,
    )
    destination = BuildTriplet(
        dim=3,
        max_pairs_num=8,
        max_nonzeros=8,
        max_active_nodes=active_nodes,
        symmetric=False,
        solver="PCG",
        matrix_symmetric=True,
        full_symmetric_input=True,
        device_reduction=True,
    )

    diagonal_blocks = np.asarray(
        [
            [[8.0, 1.0, -2.0], [3.0, 9.0, 4.0], [5.0, -6.0, 10.0]],
            [[11.0, -3.0, 2.0], [7.0, 12.0, -5.0], [1.0, 9.0, 13.0]],
            [[14.0, 6.0, -4.0], [-2.0, 15.0, 8.0], [3.0, 1.0, 16.0]],
        ],
        dtype=np.float64,
    )
    upper_01 = np.asarray(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=np.float64,
    )
    upper_12 = np.asarray(
        [[-2.0, 1.5, 0.25], [3.0, -4.0, 5.0], [6.5, -7.0, 8.0]],
        dtype=np.float64,
    )
    # Deliberately disagree with the upper blocks. Canonical storage has a
    # documented, deterministic choice: retain block_i < block_j and obtain
    # the lower triangle only by transposed matvec/serialization.
    ignored_lower_10 = 100.0 + np.arange(9, dtype=np.float64).reshape(3, 3)
    ignored_lower_21 = -200.0 - np.arange(9, dtype=np.float64).reshape(3, 3)
    ignored_same_block = 300.0 + np.arange(9, dtype=np.float64).reshape(3, 3)
    block_i = np.asarray([1, 0, 2, 1, 1, -1], dtype=np.int32)
    block_j = np.asarray([0, 1, 1, 2, 1, -1], dtype=np.int32)
    block_h = np.asarray(
        [
            ignored_lower_10,
            upper_01,
            ignored_lower_21,
            upper_12,
            ignored_same_block,
            np.full((3, 3), 400.0, dtype=np.float64),
        ],
        dtype=np.float64,
    ).reshape((-1, 9))

    def load_source(scale):
        source.diag.from_numpy((scale * diagonal_blocks).reshape((-1, 9)))
        source.non_diag.set_triplets_from_numpy(block_i, block_j, scale * block_h)
        source.raw_non_diag_count[0] = block_i.size
        source.overflow[0] = 0

    def expected_dense(scale):
        expected = np.zeros((9, 9), dtype=np.float64)
        for node, diagonal in enumerate(diagonal_blocks):
            symmetric_diagonal = 0.5 * (diagonal + diagonal.T)
            expected[
                3 * node : 3 * node + 3,
                3 * node : 3 * node + 3,
            ] = (
                scale * symmetric_diagonal
            )
        expected[0:3, 3:6] = scale * upper_01
        expected[3:6, 0:3] = scale * upper_01.T
        expected[3:6, 6:9] = scale * upper_12
        expected[6:9, 3:6] = scale * upper_12.T
        return expected

    load_source(1.0)
    destination.reset_system()
    destination.append_raw_from(source, active_nodes=active_nodes)
    assert not destination._full_input_canonicalized
    assert destination.canonicalize_full_symmetric_input() == block_i.size
    assert destination._full_input_canonicalized

    expected_raw_i = np.asarray([-1, 0, -1, 1, -1, -1], dtype=np.int32)
    expected_raw_j = np.asarray([-1, 1, -1, 2, -1, -1], dtype=np.int32)
    raw_i = destination.non_diag.blockI.to_numpy()[: block_i.size].copy()
    raw_j = destination.non_diag.blockJ.to_numpy()[: block_i.size].copy()
    raw_h = destination.non_diag.blockH.to_numpy()[: block_i.size].copy()
    np.testing.assert_array_equal(raw_i, expected_raw_i)
    np.testing.assert_array_equal(raw_j, expected_raw_j)
    np.testing.assert_array_equal(raw_h[1], upper_01.reshape(-1))
    np.testing.assert_array_equal(raw_h[3], upper_12.reshape(-1))
    np.testing.assert_array_equal(raw_h[[0, 2, 4, 5]], 0.0)

    destination.finalize_taichi_assembly()
    first_dense = destination.to_scipy(active_nodes=active_nodes).toarray()
    np.testing.assert_array_equal(first_dense, first_dense.T)
    np.testing.assert_allclose(first_dense, expected_dense(1.0), rtol=0.0, atol=0.0)
    reduced_i, reduced_j, _ = destination.non_diag.get_reduced_triplets_numpy()
    assert np.all(reduced_i < reduced_j)
    assert set(zip(reduced_i.tolist(), reduced_j.tolist())) == {(0, 1), (1, 2)}

    # Finalization defensively canonicalizes again. It must neither discard an
    # already-canonical upper block nor alter values on a second invocation.
    first_statistics = destination.acceleration_statistics()["block_pattern"]
    destination.finalize_taichi_assembly()
    second_dense = destination.to_scipy(active_nodes=active_nodes).toarray()
    np.testing.assert_array_equal(second_dense, first_dense)
    np.testing.assert_array_equal(destination.non_diag.blockI.to_numpy()[: block_i.size], expected_raw_i)
    np.testing.assert_array_equal(destination.non_diag.blockJ.to_numpy()[: block_i.size], expected_raw_j)
    second_statistics = destination.acceleration_statistics()["block_pattern"]
    assert second_statistics["pattern_version"] == first_statistics["pattern_version"]
    assert second_statistics["last_mapping_misses"] == 0

    # Resetting and appending the same stencil must retain the raw slot layout
    # and cached reduced pattern while replacing every numerical value.
    load_source(2.0)
    destination.reset_system()
    assert not destination._full_input_canonicalized
    destination.append_raw_from(source, active_nodes=active_nodes)
    destination.canonicalize_full_symmetric_input()
    np.testing.assert_array_equal(destination.non_diag.blockI.to_numpy()[: block_i.size], expected_raw_i)
    np.testing.assert_array_equal(destination.non_diag.blockJ.to_numpy()[: block_i.size], expected_raw_j)
    destination.finalize_taichi_assembly()
    reset_dense = destination.to_scipy(active_nodes=active_nodes).toarray()
    np.testing.assert_array_equal(reset_dense, reset_dense.T)
    np.testing.assert_allclose(reset_dense, expected_dense(2.0), rtol=0.0, atol=0.0)
    reset_statistics = destination.acceleration_statistics()["block_pattern"]
    assert reset_statistics["pattern_version"] == second_statistics["pattern_version"]
    assert reset_statistics["last_mapping_misses"] == 0

    # Direct scalar insertion must also retain physical row orientation until
    # canonicalization; otherwise matrix_symmetric insertion would fold both
    # input triangles into the upper block and double the coefficient.
    destination.reset_system()
    destination.assemble_scalar_triplets(
        np.asarray([0, 3, 0, 3], dtype=np.int32),
        np.asarray([3, 0, 0, 3], dtype=np.int32),
        np.asarray([2.0, 99.0, 4.0, 5.0], dtype=np.float64),
    )
    destination.finalize_taichi_assembly()
    directly_assembled = destination.to_scipy(active_nodes=active_nodes).toarray()
    assert directly_assembled[0, 3] == 2.0
    assert directly_assembled[3, 0] == 2.0
    assert directly_assembled[0, 0] == 4.0
    assert directly_assembled[3, 3] == 5.0
    np.testing.assert_array_equal(directly_assembled, directly_assembled.T)

    with pytest.raises(ValueError, match="requires matrix_symmetric=True"):
        BuildTriplet(
            dim=3,
            max_pairs_num=1,
            max_nonzeros=1,
            max_active_nodes=1,
            symmetric=False,
            matrix_symmetric=False,
            full_symmetric_input=True,
        )
    with pytest.raises(ValueError, match="requires dense block storage"):
        BuildTriplet(
            dim=3,
            max_pairs_num=1,
            max_nonzeros=1,
            max_active_nodes=1,
            symmetric=True,
            matrix_symmetric=True,
            full_symmetric_input=True,
        )

    upper_only_source = BuildTriplet(
        dim=3,
        max_pairs_num=1,
        max_nonzeros=1,
        max_active_nodes=active_nodes,
        symmetric=False,
        matrix_symmetric=True,
    )
    destination.reset_system()
    with pytest.raises(ValueError, match="source and destination block storage"):
        destination.append_raw_from(upper_only_source, active_nodes=active_nodes)

    # A full-input source may still contain both raw triangles before its own
    # finalization. Appending it to an ordinary mirrored destination would
    # mirror both copies and silently double every off-diagonal coefficient.
    half_storage_destination = BuildTriplet(
        dim=3,
        max_pairs_num=8,
        max_nonzeros=8,
        max_active_nodes=active_nodes,
        symmetric=False,
        matrix_symmetric=True,
    )
    destination.reset_system()
    destination.assemble_scalar_triplets(
        np.asarray([0, 3], dtype=np.int32),
        np.asarray([3, 0], dtype=np.int32),
        np.asarray([3.0, 3.0], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="source and destination block storage"):
        half_storage_destination.append_raw_from(destination, active_nodes=active_nodes)
    with pytest.raises(RuntimeError, match="raw two-triangle assembly"):
        destination.load_from_scipy_blocks(
            coo_matrix(np.eye(3 * active_nodes)),
            active_nodes=active_nodes,
        )


def _run_overflow_recovery_checks():
    reduction = HashReduction(
        max_pairs_num=8,
        max_nnz=4,
        hessian_size=1,
        device_reduction=True,
    )
    block_i = np.arange(5, dtype=np.int32)
    block_j = np.zeros(5, dtype=np.int32)
    block_h = np.ones((5, 1), dtype=np.float64)
    reduction.set_triplets_from_numpy(block_i, block_j, block_h)
    try:
        reduction.go(5)
    except RuntimeError as exc:
        assert "overflow" in str(exc)
    else:
        raise AssertionError("max_nnz overflow was not reported")

    # A failed reduction must leave a coherent empty state so callers can
    # resize/retry instead of reusing an out-of-range element_pair_num.
    assert int(reduction.element_pair_num[0]) == 0
    assert not reduction.device_pattern_initialized
    reduction.set_triplets_from_numpy(block_i[:4], block_j[:4], block_h[:4])
    reduction.go(4)
    _assert_reduction_matches(reduction, block_i[:4], block_j[:4], block_h[:4], block_count=5)

    # A full cached superset may overflow when one contact is replaced even
    # though the exact current pattern still fits. It must compact and retry.
    replacement_i = np.arange(1, 5, dtype=np.int32)
    replacement_j = np.zeros(4, dtype=np.int32)
    replacement_h = 2.0 * np.ones((4, 1), dtype=np.float64)
    reduction.set_triplets_from_numpy(replacement_i, replacement_j, replacement_h)
    reduction.go(4)
    _assert_reduction_matches(
        reduction,
        replacement_i,
        replacement_j,
        replacement_h,
        block_count=5,
    )

    no_superset = HashReduction(
        max_pairs_num=4,
        max_nnz=4,
        hessian_size=1,
        pattern_cache=True,
        pattern_cache_extra_fraction=0.0,
        pattern_cache_max_age=25,
        device_reduction=True,
    )
    no_superset.set_triplets_from_numpy(block_i[:2], block_j[:2], block_h[:2])
    no_superset.go(2)
    no_superset.set_triplets_from_numpy(block_i[:1], block_j[:1], block_h[:1])
    no_superset.go(1)
    assert no_superset.pattern_cache_statistics()["pattern_nonzeros"] == 1


def _run_invalid_fixed_slot_checks():
    """Reserved fixed-stencil slots must be invisible to both reducers."""
    block_i = np.asarray([-1, 0, 0, -1, 2, 1, -1, 2], dtype=np.int32)
    block_j = np.asarray([-1, 1, 1, -1, 0, 2, -1, 0], dtype=np.int32)
    block_h = np.arange(1.0, 1.0 + 8 * 4, dtype=np.float64).reshape(8, 4)
    valid = (block_i >= 0) & (block_j >= 0)

    # CUDA is intentionally device-only: selecting host reduction there would
    # download every raw block and upload every reduced block each Newton step.
    reduction_modes = (True,) if ti.lang.impl.current_cfg().arch == ti.cuda else (False, True)
    for device_reduction in reduction_modes:
        reduction = HashReduction(
            max_pairs_num=block_i.size,
            max_nnz=9,
            hessian_size=4,
            pattern_cache=True,
            device_reduction=device_reduction,
        )
        reduction.set_triplets_from_numpy(block_i, block_j, block_h)
        reduction.go(block_i.size)
        _assert_reduction_matches(
            reduction,
            block_i[valid],
            block_j[valid],
            block_h[valid],
            block_count=3,
        )
        out_i, out_j, _ = reduction.get_reduced_triplets_numpy()
        assert np.all(out_i >= 0)
        assert np.all(out_j >= 0)

        if device_reduction:
            first = reduction.pattern_cache_statistics()
            assert first["last_mapping_misses"] == int(np.count_nonzero(valid))
            reduction.set_triplets_from_numpy(block_i, block_j, 0.25 * block_h)
            reduction.go(block_i.size)
            _assert_reduction_matches(
                reduction,
                block_i[valid],
                block_j[valid],
                0.25 * block_h[valid],
                block_count=3,
            )
            assert reduction.pattern_cache_statistics()["last_mapping_misses"] == 0


def test_cuda_explicit_host_reduction_fails_fast(monkeypatch):
    import src.linear_solver.HashReduction as reduction_module

    monkeypatch.setattr(
        reduction_module,
        "current_cfg",
        lambda: type("Cfg", (), {"arch": ti.cuda})(),
    )
    with pytest.raises(RuntimeError, match="device_reduction=False"):
        HashReduction(
            max_pairs_num=1,
            max_nnz=1,
            hessian_size=1,
            device_reduction=False,
        )


def test_cuda_krylov_api_rejects_full_host_vectors(monkeypatch):
    import src.linear_solver.BuildTriplet as triplet_module

    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)
    matrix = BuildTriplet(
        dim=3,
        max_pairs_num=1,
        max_nonzeros=1,
        max_active_nodes=1,
        symmetric=False,
        device_reduction=True,
    )
    monkeypatch.setattr(
        triplet_module,
        "current_cfg",
        lambda: type("Cfg", (), {"arch": ti.cuda})(),
    )

    with pytest.raises(RuntimeError, match="preloaded.*Taichi fields"):
        matrix.solve(
            rhs=np.ones(3, dtype=np.float64),
            active_nodes=1,
            return_solution=False,
        )
    with pytest.raises(RuntimeError, match="full host solution"):
        matrix.solve(active_nodes=1, return_solution=True)
    with pytest.raises(RuntimeError, match="Taichi scalar rhs field"):
        matrix.solve_flat_system(
            np.ones(3, dtype=np.float64),
            active_nodes=1,
            return_solution=False,
        )


def test_taichi_device_block_pattern_cache_and_krylov():
    """Pytest entry point; run the real Taichi device kernels on any backend."""
    runtime = ti.lang.impl.get_runtime()
    if runtime.prog is None:
        arch = ti.cuda if ti._lib.core.with_cuda() else ti.cpu
        ti.init(arch=arch, default_fp=ti.f64, offline_cache=False)
    _run_device_pattern_checks()
    _run_build_triplet_checks()
    _run_full_symmetric_input_checks()
    _run_overflow_recovery_checks()
    _run_invalid_fixed_slot_checks()

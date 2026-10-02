"""Deterministic unit contracts for the multi-batch LBVH implementation."""

import numpy as np
import pytest
import taichi as ti

from src.contact_detection.bounding_volume_hierarchy.MultiPatchLBVH import (
    AABB,
    LBVH,
)
from src.utils.BitFunction import expandBits
from src.utils.TypeDefination import vec3f


pytestmark = [pytest.mark.contact, pytest.mark.geometry, pytest.mark.cpu]


@ti.kernel
def _expand_bits(values: ti.template(), expanded: ti.template()):
    for i in values:
        expanded[i] = expandBits(values[i])


def _build_random_tree(count, seed=731):
    rng = np.random.default_rng(seed)
    lower = rng.uniform(0.0, 8.0, size=(count, 3)).astype(np.float32)
    upper = lower + rng.uniform(0.05, 0.45, size=(count, 3)).astype(
        np.float32
    )
    aabb = AABB(n_aabbs=[count])
    aabb.aabbs.min.from_numpy(lower)
    aabb.aabbs.max.from_numpy(upper)
    tree = LBVH(aabb=aabb)
    tree.initialize()
    tree.build()
    return tree


def test_expand_bits_inserts_two_zero_bits(taichi_runtime):
    values_np = np.array(
        [0, 1, 2, 3, 7, 31, 255, 511, 1022, 1023], dtype=np.uint32
    )
    values = ti.field(ti.u32, shape=values_np.shape)
    expanded = ti.field(ti.u32, shape=values_np.shape)
    values.from_numpy(values_np)

    _expand_bits(values, expanded)

    expected = np.array(
        [
            sum(((int(value) >> bit) & 1) << (3 * bit) for bit in range(10))
            for value in values_np
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(expanded.to_numpy(), expected)


def test_morton_codes_are_strictly_sorted(taichi_runtime):
    tree = _build_random_tree(32)

    codes = tree.morton_codes.to_numpy()

    assert np.all(codes[1:] > codes[:-1])


def test_tree_parent_child_links_and_bounds_are_consistent(taichi_runtime):
    tree = _build_random_tree(24)
    nodes = tree.nodes.to_numpy()
    prefix = tree.prefix_batch_size.to_numpy()

    for batch_id in range(tree.aabb.n_batches):
        batch_size = int(prefix[batch_id + 1] - prefix[batch_id])
        node_start = int(2 * prefix[batch_id] - batch_id)
        for local_index in range(2 * batch_size - 1):
            node_index = node_start + local_index
            parent = int(nodes["parent"][node_index])
            left = int(nodes["left"][node_index])
            right = int(nodes["right"][node_index])

            if local_index == 0:
                assert parent == -1
            else:
                assert local_index in (
                    int(nodes["left"][node_start + parent]),
                    int(nodes["right"][node_start + parent]),
                )

            for child in (left, right):
                if child != -1:
                    assert int(nodes["parent"][node_start + child]) == local_index

            if left != -1 and right != -1:
                expected_min = np.minimum(
                    nodes["bound"]["min"][node_start + left],
                    nodes["bound"]["min"][node_start + right],
                )
                expected_max = np.maximum(
                    nodes["bound"]["max"][node_start + left],
                    nodes["bound"]["max"][node_start + right],
                )
                np.testing.assert_allclose(
                    nodes["bound"]["min"][node_index],
                    expected_min,
                    rtol=1.0e-5,
                    atol=1.0e-6,
                )
                np.testing.assert_allclose(
                    nodes["bound"]["max"][node_index],
                    expected_max,
                    rtol=1.0e-5,
                    atol=1.0e-6,
                )


def test_self_query_matches_brute_force_intersections(taichi_runtime):
    lower = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.4, 0.2, 0.2],
            [0.0, 3.0, 0.0],
            [0.2, 3.2, 0.2],
            [6.0, 6.0, 6.0],
        ],
        dtype=np.float32,
    )
    upper = lower + 1.0
    count = lower.shape[0]
    aabb = AABB(n_aabbs=[count])
    aabb.aabbs.min.from_numpy(lower)
    aabb.aabbs.max.from_numpy(upper)
    tree = LBVH(aabb=aabb)
    tree.initialize()
    tree.build()
    candidates = ti.field(ti.i32, shape=count * count)
    counts = ti.field(ti.i32, shape=count + 1)

    tree.self_query(count, count, 0, 0, aabb.aabbs, candidates, counts)

    candidates_np = candidates.to_numpy()
    counts_np = counts.to_numpy()
    for master in range(count):
        actual = set(
            candidates_np[
                master * count : master * count + counts_np[master + 1]
            ].tolist()
        )
        expected = {
            slave
            for slave in range(count)
            if np.all(lower[master] <= upper[slave])
            and np.all(upper[master] >= lower[slave])
        }
        assert actual == expected


def test_active_batches_use_current_prefix(taichi_runtime):
    aabb = AABB(n_aabbs=[8, 5])
    aabb.aabbs.min.from_numpy(np.zeros((13, 3), dtype=np.float32))
    aabb.aabbs.max.from_numpy(np.ones((13, 3), dtype=np.float32))
    tree = LBVH(aabb=aabb)

    tree.initialize(active_aabbs=[3, 2])

    assert aabb.prefix_current_batch_size == [0, 3, 5]
    assert tree.prefix_batch_size.to_numpy().tolist() == [0, 3, 5]
    assert tree.leaf2batch.to_numpy()[:5].tolist() == [0, 0, 0, 1, 1]
    assert tree.internal2batch.to_numpy()[:3].tolist() == [0, 0, 1]
    tree.build()


def test_degenerate_triangle_aabb_has_positive_extent(taichi_runtime):
    @ti.dataclass
    class Triangle:
        vertice1: vec3f
        vertice2: vec3f
        vertice3: vec3f

    @ti.kernel
    def set_triangle(triangles: ti.template()):
        triangles[0].vertice1 = [1.0, 2.0, 3.0]
        triangles[0].vertice2 = [4.0, 2.0, 3.0]
        triangles[0].vertice3 = [1.0, 5.0, 3.0]

    triangles = Triangle.field(shape=1)
    set_triangle(triangles)
    aabb = AABB(n_aabbs=[1])

    aabb.set_triangle_aabbs(1, 0, 0.0, triangles)

    extent = aabb.aabbs.max.to_numpy()[0] - aabb.aabbs.min.to_numpy()[0]
    assert np.all(extent > 0.0)


def test_prefix_build_preserves_unrebuilt_batch(taichi_runtime):
    lower = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [2.0, 5.0, 0.0],
        ],
        dtype=np.float32,
    )
    upper = lower + np.array([0.5, 0.5, 0.5], dtype=np.float32)
    aabb = AABB(n_aabbs=[3, 2])
    aabb.aabbs.min.from_numpy(lower)
    aabb.aabbs.max.from_numpy(upper)
    tree = LBVH(aabb=aabb)
    tree.initialize(active_aabbs=[3, 2])
    tree.build()

    wall_prefix = int(tree.prefix_batch_size.to_numpy()[1])
    wall_start = 2 * wall_prefix - 1
    wall_stop = wall_start + 3
    before = tree.nodes.to_numpy()
    before_codes = tree.morton_codes.to_numpy()[wall_prefix : wall_prefix + 2]

    lower[:3] += np.array([0.25, 0.0, 0.0], dtype=np.float32)
    upper[:3] += np.array([0.25, 0.0, 0.0], dtype=np.float32)
    aabb.aabbs.min.from_numpy(lower)
    aabb.aabbs.max.from_numpy(upper)
    tree.build_prefix(1, 3)

    after = tree.nodes.to_numpy()
    np.testing.assert_array_equal(
        tree.morton_codes.to_numpy()[wall_prefix : wall_prefix + 2],
        before_codes,
    )
    for member in ("left", "right", "parent"):
        np.testing.assert_array_equal(
            after[member][wall_start:wall_stop],
            before[member][wall_start:wall_stop],
        )
    for member in ("min", "max"):
        np.testing.assert_allclose(
            after["bound"][member][wall_start:wall_stop],
            before["bound"][member][wall_start:wall_stop],
        )

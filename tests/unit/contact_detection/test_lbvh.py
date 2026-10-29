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
    upper = lower + rng.uniform(0.05, 0.45, size=(count, 3)).astype(np.float32)
    aabb = AABB(n_aabbs=[count])
    aabb.aabbs.min.from_numpy(lower)
    aabb.aabbs.max.from_numpy(upper)
    tree = LBVH(aabb=aabb)
    tree.initialize()
    tree.build()
    return tree


def test_expand_bits_inserts_two_zero_bits(taichi_runtime):
    values_np = np.array([0, 1, 2, 3, 7, 31, 255, 511, 1022, 1023], dtype=np.uint32)
    values = ti.field(ti.u32, shape=values_np.shape)
    expanded = ti.field(ti.u32, shape=values_np.shape)
    values.from_numpy(values_np)

    _expand_bits(values, expanded)

    expected = np.array(
        [sum(((int(value) >> bit) & 1) << (3 * bit) for bit in range(10)) for value in values_np],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(expanded.to_numpy(), expected)


def test_morton_codes_are_strictly_sorted(taichi_runtime):
    tree = _build_random_tree(32)

    codes = tree.morton_codes.to_numpy()

    assert np.all(codes[1:] > codes[:-1])


def test_chunk_radix_sort_preserves_stability_active_batches_and_full_width_keys(taichi_runtime):
    capacities = [259, 1031, 521]
    tree = LBVH(aabb=AABB(capacities), extended_morton=True)
    rng = np.random.default_rng(473)
    for sizes, batches in [([259, 1031, 521], 3), ([1, 513, 257], 2), ([257, 1025, 1], 3)]:
        tree.initialize(active_aabbs=sizes)
        np.testing.assert_array_equal(
            tree.sort_chunk_prefix.to_numpy(), np.cumsum([0, *((size + 255) // 256 for size in sizes)])
        )
        keys = rng.integers(0, 2**64, sum(capacities), dtype=np.uint64)
        # Equal full-width keys must retain their associated primitive order.
        keys[::3] = np.uint64(2**64 - 1)
        ids = np.arange(sum(capacities), dtype=np.int32)
        expected_keys, expected_ids = keys.copy(), ids.copy()
        prefix = np.cumsum([0, *sizes])
        for batch in range(batches):
            begin, end = prefix[batch : batch + 2]
            order = np.argsort(keys[begin:end], kind="stable")
            expected_keys[begin:end] = keys[begin:end][order]
            expected_ids[begin:end] = ids[begin:end][order]
        tree.morton_codes.from_numpy(keys)
        tree.primitive_ids.from_numpy(ids)
        tree.radix_sort_morton_codes(batches, int(prefix[batches]))
        np.testing.assert_array_equal(tree.morton_codes.to_numpy(), expected_keys)
        np.testing.assert_array_equal(tree.primitive_ids.to_numpy(), expected_ids)


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


@pytest.mark.parametrize("rebuild", ["build", "refit"])
def test_repeated_multibatch_rebuilds_publish_all_child_bounds(taichi_runtime, rebuild):
    sizes = [257, 513, 1025]
    prefix = np.cumsum([0, *sizes])
    rng = np.random.default_rng(651)
    lower = rng.uniform(0.1, 0.9, (sum(sizes), 3))
    aabb = AABB(n_aabbs=sizes)
    tree = LBVH(aabb=aabb)
    tree.initialize()
    for iteration in range(12):
        padding = 0.006 if iteration % 2 == 0 else 0.0
        aabb.aabbs.min.from_numpy(lower - padding)
        aabb.aabbs.max.from_numpy(lower + 0.001 + padding)
        if iteration == 0:
            tree.build()
        else:
            getattr(tree, rebuild)()
        nodes = tree.nodes.to_numpy()
        for batch, size in enumerate(sizes):
            start = 2 * prefix[batch] - batch
            internal = np.arange(start, start + size - 1)
            left = start + nodes["left"][internal]
            right = start + nodes["right"][internal]
            np.testing.assert_array_equal(
                nodes["bound"]["min"][internal], np.minimum(nodes["bound"]["min"][left], nodes["bound"]["min"][right])
            )
            np.testing.assert_array_equal(
                nodes["bound"]["max"][internal], np.maximum(nodes["bound"]["max"][left], nodes["bound"]["max"][right])
            )


def test_bound_build_rejects_depth_exceeding_query_stack(taichi_runtime):
    aabb = AABB(n_aabbs=[3])
    lower = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    aabb.aabbs.min.from_numpy(lower)
    aabb.aabbs.max.from_numpy(lower + 0.1)
    tree = LBVH(aabb=aabb)
    tree.initialize()
    tree.max_stack_depth = 2
    with pytest.raises(RuntimeError, match="traversal capacity"):
        tree.build()


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
        actual = set(candidates_np[master * count : master * count + counts_np[master + 1]].tolist())
        expected = {
            slave
            for slave in range(count)
            if np.all(lower[master] <= upper[slave]) and np.all(upper[master] >= lower[slave])
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


@pytest.mark.parametrize("rebuild", ["build_prefix", "refit_prefix"])
def test_prefix_build_preserves_unrebuilt_batch(taichi_runtime, rebuild):
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
    getattr(tree, rebuild)(1, 3)

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


def test_explicit_bvh_schedule_handles_sparse_queries_static_batches_and_reinitialization(taichi_runtime, monkeypatch):
    lower = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [10, 0, 0], [11, 0, 0]], dtype=np.float64)
    tree = LBVH(aabb=AABB([4, 2], dtype=ti.f64), extended_morton=True)
    tree.initialize()
    calls = []
    for method in ("build_prefix", "refit_prefix"):
        original = getattr(tree, method)

        def observed(batches, count, original=original, method=method):
            calls.append((method, batches, count))
            original(batches, count)

        monkeypatch.setattr(tree, method, observed)
    wall_before = None
    for step in (0, 2, 6, 6, 11, 3):
        current = lower.copy()
        current[:4, 0] += 0.1 * step
        tree.aabb.aabbs.min.from_numpy(current)
        tree.aabb.aabbs.max.from_numpy(current + 0.1)
        tree.update(step, 5, n_batches=1)
        nodes = tree.nodes.to_numpy()
        np.testing.assert_allclose(nodes["bound"]["min"][0], current[:4].min(axis=0))
        np.testing.assert_allclose(nodes["bound"]["max"][0], (current[:4] + 0.1).max(axis=0))
        wall = {name: nodes[name][7:].copy() for name in ("left", "right", "parent")}
        wall.update({name: nodes["bound"][name][7:].copy() for name in ("min", "max")})
        if wall_before is not None:
            for name in wall:
                np.testing.assert_array_equal(wall[name], wall_before[name])
        wall_before = wall
    assert calls == [
        ("build_prefix", 2, 6),
        ("refit_prefix", 1, 4),
        ("build_prefix", 1, 4),
        ("refit_prefix", 1, 4),
        ("build_prefix", 1, 4),
        ("build_prefix", 1, 4),
    ]
    tree.initialize(active_aabbs=[3, 2])
    tree.update(4, 5, n_batches=1)
    assert calls[-1] == ("build_prefix", 2, 5)


@pytest.mark.parametrize("value", [0, -1, 1.5, True, np.bool_(True), float("inf"), "5", None])
def test_bvh_rebuild_interval_rejects_invalid_values(value):
    from src.contact_detection.bounding_volume_hierarchy.MultiPatchLBVH import validate_bvh_rebuild_interval

    with pytest.raises(ValueError, match="positive integer"):
        validate_bvh_rebuild_interval(value)
    assert validate_bvh_rebuild_interval(np.int64(5)) == 5


def test_dem_active_count_change_rebuilds_shifted_static_wall_batch(taichi_runtime):
    from types import SimpleNamespace
    from src.dem.neighbor.BoundingVolumeHierarchy import BoundingVolumeHierarchy

    neighbor = object.__new__(BoundingVolumeHierarchy)
    neighbor.sims = SimpleNamespace(current_step=0, bvh_rebuild_interval=100, wall_type=1, static_wall=True)
    neighbor.lbvh = LBVH(aabb=AABB([3, 2], dtype=ti.f64), extended_morton=True)
    scene = SimpleNamespace(particleNum=[2], wallNum=[1])

    @ti.kernel
    def fill_boxes(begin: ti.i32, count: ti.i32, origin: ti.f64):
        for i in range(count):
            lower = ti.Vector([origin + i, 0.0, 0.0])
            neighbor.lbvh.aabb.aabbs[begin + i].min = lower
            neighbor.lbvh.aabb.aabbs[begin + i].max = lower + 0.1

    neighbor.place_particle_to_cell = lambda scene: fill_boxes(0, scene.particleNum[0], 0.0)
    neighbor.place_wall_to_cells = lambda scene: fill_boxes(scene.particleNum[0], scene.wallNum[0], 10.0)
    place_walls = neighbor.place_wall_to_cells
    neighbor.manage_wall_function = lambda: setattr(neighbor, "place_wall_to_cells", place_walls)
    neighbor.update_verlet_tables_particle_particle = lambda scene: None
    neighbor.update_verlet_tables_particle_wall = lambda scene: None
    neighbor.reset_verletDisp = lambda: None
    neighbor.pre_neighbor(scene)
    assert neighbor.lbvh.aabb.current_batch_size == [2, 1]
    scene.particleNum[0], scene.wallNum[0] = 3, 2
    neighbor.sims.current_step = 1
    neighbor.update_verlet_table(scene)
    assert neighbor.lbvh.aabb.current_batch_size == [3, 2]
    assert neighbor.lbvh._last_rebuild_step == 1
    bounds = neighbor.lbvh.nodes.to_numpy()["bound"]
    np.testing.assert_allclose(bounds["min"][5], [10.0, 0.0, 0.0])
    np.testing.assert_allclose(bounds["max"][5], [11.1, 0.1, 0.1])
    neighbor.sims.current_step = 2
    neighbor.update_verlet_table(scene)
    assert neighbor.lbvh._last_rebuild_step == 1


def test_stackless_export_visits_each_primitive_across_many_batches(taichi_runtime):
    counts = [1, 7, 3] + [1] * 127
    total = sum(counts)
    boxes = AABB(counts, dtype=ti.f64)
    lower = np.random.default_rng(831).random((total, 3))
    boxes.aabbs.min.from_numpy(lower)
    boxes.aabbs.max.from_numpy(lower + 0.01)
    tree = LBVH(aabb=boxes, extended_morton=True)
    tree.initialize()
    tree.build()
    nodes = ti.Vector.field(4, ti.i32, shape=2 * total - len(counts))
    prefix = ti.field(ti.i32, shape=len(counts) + 1)
    tree.export_stackless(nodes, prefix)
    escapes = ti.field(ti.i32, shape=nodes.shape)
    tree.export_escape_links(escapes)
    exported, starts = nodes.to_numpy(), prefix.to_numpy()
    np.testing.assert_array_equal(escapes.to_numpy(), exported[:, 2])
    bounds = tree.nodes.to_numpy()["bound"]
    primitive_begin = 0
    for batch, count in enumerate(counts):
        cursor, stop = starts[batch : batch + 2]
        visited, leaves = set(), []
        while cursor < stop:
            assert cursor not in visited
            assert starts[batch] <= cursor < stop
            visited.add(cursor)
            left, right, escape, primitive = exported[cursor]
            if primitive >= 0:
                leaves.append(primitive)
                cursor = escape
            else:
                assert starts[batch] <= right < stop
                cursor = left
        assert cursor == stop
        assert len(visited) == 2 * count - 1
        assert sorted(leaves) == list(range(primitive_begin, primitive_begin + count))
        for query in lower[primitive_begin : primitive_begin + min(count, 2)]:
            query_max = query + 0.1
            cursor, stop = starts[batch : batch + 2]
            hits = []
            while cursor < stop:
                left, _, escape, primitive = exported[cursor]
                overlap = np.all(bounds["min"][cursor] <= query_max) and np.all(query <= bounds["max"][cursor])
                if overlap and primitive < 0:
                    cursor = left
                else:
                    if overlap:
                        hits.append(primitive)
                    cursor = escape
            batch_ids = np.arange(primitive_begin, primitive_begin + count)
            expected = batch_ids[
                np.all(lower[batch_ids] <= query_max, axis=1) & np.all(query <= lower[batch_ids] + 0.01, axis=1)
            ]
            assert sorted(hits) == list(expected)
        primitive_begin += count


@pytest.mark.parametrize("rebuild_interval", [None, 5])
def test_fem_contact_bvh_rebuilds_current_and_swept_queries_on_device(taichi_runtime, monkeypatch, rebuild_interval):
    from src.fem.contact.BVHBroadPhase import DynamicBVHBroadPhase
    from src.fem.contact.ContactTopology import build_contact_surface
    from tests.helpers.fem_broad_phase_reference import _reference_broad_phase_candidates
    from src.fem.generator import FEMMesh

    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.1, 0.1, 0.05], [1.1, 0.1, 0.05], [0.1, 1.1, 0.05]]
    )
    mesh = FEMMesh(points, np.array([[0, 1, 2], [3, 4, 5]]), "TRI3")
    surface = build_contact_surface(mesh)
    positions = ti.Vector.field(3, ti.f64, shape=6)
    ends = ti.Vector.field(3, ti.f64, shape=6)
    positions.from_numpy(points)

    def forbid_host_sort(*args, **kwargs):
        raise AssertionError("BVH topology must be built on device")

    monkeypatch.setattr(np, "argsort", forbid_host_sort)
    broad = DynamicBVHBroadPhase(
        surface.faces,
        surface.edges,
        surface.vertices,
        surface.node_area,
        surface.edge_area,
        points,
        bvh_rebuild_interval=rebuild_interval,
    )
    assert isinstance(broad.face_bvh, LBVH)
    original_build = LBVH.build_prefix
    builds = []

    def build(tree, *args):
        builds.append(tree)
        original_build(tree, *args)

    monkeypatch.setattr(LBVH, "build_prefix", build)
    orders = []
    for separation in (0.05, -0.5, 0.02):
        current = points.copy()
        current[3:, 2] = separation
        if separation < 0:
            current[3:, :2] -= 0.2
        positions.from_numpy(current)
        for swept in (False, True):
            broad.current_step = 2 * len(orders)
            end = current.copy()
            end[3:, 2] -= 0.2
            ends.from_numpy(end)
            pt_count, ee_count = broad.rebuild(positions, 0.1, ends if swept else None)
            expected_pt, expected_ee = _reference_broad_phase_candidates(
                surface, current, radius=0.1, end_positions=end if swept else None
            )
            assert set(map(tuple, broad.point_triangle.to_numpy()[:pt_count])) == set(map(tuple, expected_pt))
            assert set(map(tuple, broad.edge_edge.to_numpy()[:ee_count])) == set(map(tuple, expected_ee))
            orders.append(tuple(broad.face_bvh.primitive_ids.to_numpy()[:2]))
    assert builds == [broad.face_bvh, broad.edge_bvh] * (6 if rebuild_interval is None else 2)
    assert len(set(orders)) > 1


def test_abd_rebuilds_topology_and_preserves_current_and_swept_candidates(taichi_runtime, monkeypatch):
    from types import SimpleNamespace
    from src.dem.neighbor.AffineBodyNeighbor import AffineBodyBoundingVolumeHierarchy

    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.1, 0.1, 0.05], [1.1, 0.1, 0.05], [0.1, 1.1, 0.05]]
    )
    face_indices = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    edge_indices = np.array([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]], dtype=np.int32)
    body_ids = np.repeat(np.arange(2, dtype=np.int32), 3)
    x, dx = ti.Vector.field(3, ti.f64, shape=6), ti.Vector.field(3, ti.f64, shape=6)
    faces, edges = ti.Vector.field(3, ti.i32, shape=2), ti.Vector.field(2, ti.i32, shape=6)
    node_body, face_body, edge_body = ti.field(ti.i32, shape=6), ti.field(ti.i32, shape=2), ti.field(ti.i32, shape=6)
    faces.from_numpy(face_indices)
    edges.from_numpy(edge_indices)
    node_body.from_numpy(body_ids)
    face_body.from_numpy(np.arange(2, dtype=np.int32))
    edge_body.from_numpy(body_ids)
    broad = AffineBodyBoundingVolumeHierarchy(
        SimpleNamespace(search="BVH", max_point_triangle_pairs=128, max_edge_edge_pairs=128), 6, 2, 6, 0.1
    )
    original_build = broad.lbvh.build
    builds = []

    def build(count):
        builds.append(count)
        original_build(count)

    monkeypatch.setattr(broad.lbvh, "build", build)
    orders = []
    for separation in (0.5, -0.05, 0.02):
        current = points.copy()
        current[3:, 2] = separation
        direction = np.zeros_like(current)
        direction[3:, 2] = -0.4
        x.from_numpy(current)
        dx.from_numpy(direction)
        for swept in (False, True):
            broad.update(x, dx, faces, edges, node_body, face_body, edge_body, 0.1, swept=swept)
            end = current + direction if swept else current
            lo, hi = np.minimum(current, end) - 0.1, np.maximum(current, end) + 0.1
            flo, fhi = lo[face_indices].min(axis=1), hi[face_indices].max(axis=1)
            elo, ehi = lo[edge_indices].min(axis=1), hi[edge_indices].max(axis=1)

            def overlaps(a, b, c, d):
                return np.all(a <= d) and np.all(c <= b)

            pt = {
                (i, j)
                for i in range(6)
                for j in range(2)
                if body_ids[i] != j and overlaps(lo[i], hi[i], flo[j], fhi[j])
            }
            ee = {
                (i, j)
                for i in range(6)
                for j in range(i + 1, 6)
                if body_ids[i] != body_ids[j] and overlaps(elo[i], ehi[i], elo[j], ehi[j])
            }
            actual_pt = set(
                zip(
                    broad.candidate_vertex.to_numpy()[: broad.last_vertex_face_candidate_pairs],
                    broad.candidate_face.to_numpy()[: broad.last_vertex_face_candidate_pairs],
                )
            )
            actual_ee = set(
                zip(
                    broad.candidate_edge0.to_numpy()[: broad.last_edge_edge_candidate_pairs],
                    broad.candidate_edge1.to_numpy()[: broad.last_edge_edge_candidate_pairs],
                )
            )
            assert actual_pt == pt and actual_ee == ee
            orders.append(tuple(broad.lbvh.primitive_ids.to_numpy()[:14]))
    assert builds == [14] * 6
    assert len(set(orders)) > 1

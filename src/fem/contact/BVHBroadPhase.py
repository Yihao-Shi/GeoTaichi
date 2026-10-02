"""Refitted Taichi BVH broad phase for deforming FEM contact surfaces."""

import numpy as np
import taichi as ti

from src.fem.contact.LinkedCellBroadPhase import _grown_capacity
from src.physics_model.contact_model.ipc.ContactDistance import (
    edge_edge_distance2,
    point_triangle_distance2,
)
from src.utils.PrefixSum import PrefixSumExecutor


def _build_binary_topology(primitives, reference_positions):
    """Build a balanced connectivity tree once; runtime AABBs are device-refit."""
    primitive_count = int(primitives.shape[0])
    if primitive_count == 0:
        return (
            np.full((1, 2), -1, dtype=np.int32),
            np.full(1, -1, dtype=np.int32),
            np.zeros(1, dtype=np.int32),
            np.zeros(1, dtype=np.int32),
            1,
        )
    centroids = np.mean(reference_positions[primitives], axis=1)
    children = []
    primitive = []
    primitive_leaf = np.empty(primitive_count, dtype=np.int32)
    postorder = []
    maximum_depth = 1

    def build(indices, depth):
        nonlocal maximum_depth
        node = len(children)
        children.append([-1, -1])
        primitive.append(-1)
        maximum_depth = max(maximum_depth, depth)
        if indices.size == 1:
            primitive_id = int(indices[0])
            primitive[node] = primitive_id
            primitive_leaf[primitive_id] = node
            return node
        bounds = np.ptp(centroids[indices], axis=0)
        axis = int(np.argmax(bounds))
        ordered = indices[np.argsort(centroids[indices, axis], kind="stable")]
        middle = ordered.size // 2
        left = build(ordered[:middle], depth + 1)
        right = build(ordered[middle:], depth + 1)
        children[node] = [left, right]
        postorder.append(node)
        return node

    build(np.arange(primitive_count, dtype=np.int32), 1)
    return (
        np.asarray(children, dtype=np.int32),
        np.asarray(primitive, dtype=np.int32),
        primitive_leaf,
        np.asarray(postorder, dtype=np.int32),
        maximum_depth + 1,
    )


@ti.data_oriented
class DynamicBVHBroadPhase:
    """Static balanced topology with current/swept AABBs refitted each query.

    Candidate output uses per-query count -> prefix sum -> fill, so neither
    leaves nor contact pairs have a user-chosen per-node capacity.
    """

    def __init__(
        self,
        faces,
        edges,
        vertices,
        node_area,
        edge_area,
        reference_positions,
        *,
        max_point_triangle_pairs=None,
        max_edge_edge_pairs=None,
        node_system_ids=None,
        cross_system_only=False,
        exact_proximity=False,
        **_,
    ):
        faces = np.ascontiguousarray(faces, dtype=np.int32)
        edges = np.ascontiguousarray(edges, dtype=np.int32)
        vertices = np.ascontiguousarray(vertices, dtype=np.int32)
        reference_positions = np.asarray(reference_positions, dtype=np.float64)
        self.face_count = int(faces.shape[0])
        self.edge_count = int(edges.shape[0])
        self.vertex_count = int(vertices.size)
        self.real_type = ti.lang.impl.current_cfg().default_fp
        self.numpy_type = np.float64 if self.real_type == ti.f64 else np.float32
        self.cross_system_only = bool(cross_system_only)
        self.exact_proximity = bool(exact_proximity)
        system_values = (
            np.zeros(reference_positions.shape[0], dtype=np.int32)
            if node_system_ids is None
            else np.asarray(node_system_ids, dtype=np.int32)
        )
        if system_values.shape != (reference_positions.shape[0],):
            raise ValueError("broad-phase node_system_ids must contain one ID per node")
        self.system_count = max(
            int(np.max(system_values, initial=0)) + 1,
            1,
        )
        face_system_values = system_values[faces[:, 0]] if self.face_count else np.zeros(1, dtype=np.int32)
        edge_system_values = system_values[edges[:, 0]] if self.edge_count else np.zeros(1, dtype=np.int32)
        self.node_system = ti.field(dtype=ti.i32, shape=reference_positions.shape[0])
        self.face_system = ti.field(dtype=ti.i32, shape=max(self.face_count, 1))
        self.edge_system = ti.field(dtype=ti.i32, shape=max(self.edge_count, 1))
        self.node_system.from_numpy(np.ascontiguousarray(system_values))
        self.face_system.from_numpy(np.ascontiguousarray(face_system_values, dtype=np.int32))
        self.edge_system.from_numpy(np.ascontiguousarray(edge_system_values, dtype=np.int32))

        self.faces = ti.Vector.field(3, dtype=ti.i32, shape=max(self.face_count, 1))
        self.edges = ti.Vector.field(2, dtype=ti.i32, shape=max(self.edge_count, 1))
        self.vertices = ti.field(dtype=ti.i32, shape=max(self.vertex_count, 1))
        self.node_area = ti.field(dtype=self.real_type, shape=reference_positions.shape[0])
        self.edge_area = ti.field(dtype=self.real_type, shape=max(self.edge_count, 1))
        self.reference_position = ti.Vector.field(3, dtype=self.real_type, shape=reference_positions.shape[0])
        if self.face_count:
            buffer = np.zeros((max(self.face_count, 1), 3), np.int32)
            buffer[: self.face_count] = faces
            self.faces.from_numpy(buffer)
        if self.edge_count:
            buffer = np.zeros((max(self.edge_count, 1), 2), np.int32)
            buffer[: self.edge_count] = edges
            self.edges.from_numpy(buffer)
        if self.vertex_count:
            buffer = np.zeros(max(self.vertex_count, 1), np.int32)
            buffer[: self.vertex_count] = vertices
            self.vertices.from_numpy(buffer)
        self.node_area.from_numpy(np.ascontiguousarray(node_area, dtype=self.numpy_type))
        edge_area_buffer = np.zeros(max(self.edge_count, 1), dtype=self.numpy_type)
        edge_area_buffer[: self.edge_count] = edge_area
        self.edge_area.from_numpy(edge_area_buffer)
        self.reference_position.from_numpy(np.ascontiguousarray(reference_positions, dtype=self.numpy_type))

        face_topology = _build_binary_topology(faces, reference_positions)
        edge_topology = _build_binary_topology(edges, reference_positions)
        self.face_node_count = int(face_topology[0].shape[0])
        self.edge_node_count = int(edge_topology[0].shape[0])
        self.face_internal_count = int(face_topology[3].size)
        self.edge_internal_count = int(edge_topology[3].size)
        self.face_stack_depth = int(face_topology[4])
        self.edge_stack_depth = int(edge_topology[4])

        self.face_child = ti.Vector.field(2, dtype=ti.i32, shape=max(self.face_node_count, 1))
        self.edge_child = ti.Vector.field(2, dtype=ti.i32, shape=max(self.edge_node_count, 1))
        self.face_primitive = ti.field(dtype=ti.i32, shape=max(self.face_node_count, 1))
        self.edge_primitive = ti.field(dtype=ti.i32, shape=max(self.edge_node_count, 1))
        self.face_primitive_leaf = ti.field(dtype=ti.i32, shape=max(self.face_count, 1))
        self.edge_primitive_leaf = ti.field(dtype=ti.i32, shape=max(self.edge_count, 1))
        self.face_postorder = ti.field(dtype=ti.i32, shape=max(self.face_internal_count, 1))
        self.edge_postorder = ti.field(dtype=ti.i32, shape=max(self.edge_internal_count, 1))
        self._load_topology(
            face_topology,
            self.face_child,
            self.face_primitive,
            self.face_primitive_leaf,
            self.face_postorder,
        )
        self._load_topology(
            edge_topology,
            self.edge_child,
            self.edge_primitive,
            self.edge_primitive_leaf,
            self.edge_postorder,
        )

        self.face_lower = ti.Vector.field(3, dtype=self.real_type, shape=max(self.face_node_count, 1))
        self.face_upper = ti.Vector.field(3, dtype=self.real_type, shape=max(self.face_node_count, 1))
        self.edge_lower = ti.Vector.field(3, dtype=self.real_type, shape=max(self.edge_node_count, 1))
        self.edge_upper = ti.Vector.field(3, dtype=self.real_type, shape=max(self.edge_node_count, 1))
        self.radius = ti.field(dtype=self.real_type, shape=())
        self.edge_radius = ti.field(dtype=self.real_type, shape=())
        self.supports_separate_proximity_radius = True
        self.pt_scan = PrefixSumExecutor(max(self.vertex_count + 1, 1))
        self.ee_scan = PrefixSumExecutor(max(self.edge_count + 1, 1))
        # The CUDA scan stores block totals after the logical input range.
        # Allocate its padded working length, while kernels continue to use
        # vertex_count + 1 / edge_count + 1 as the logical prefix ranges.
        self.pt_prefix = ti.field(dtype=ti.i32, shape=self.pt_scan.get_length())
        self.ee_prefix = ti.field(dtype=ti.i32, shape=self.ee_scan.get_length())
        self.pt_stack = ti.field(
            dtype=ti.i32,
            shape=(max(self.vertex_count, 1), self.face_stack_depth),
        )
        self.ee_stack = ti.field(
            dtype=ti.i32,
            shape=(max(self.edge_count, 1), self.edge_stack_depth),
        )
        nearest_capacity = self.vertex_count * self.system_count if self.exact_proximity else 1
        self.pt_nearest_face = ti.field(
            dtype=ti.i32,
            shape=max(nearest_capacity, 1),
        )
        self.pt_nearest_distance2 = ti.field(
            dtype=self.real_type,
            shape=max(nearest_capacity, 1),
        )

        self.fixed_pt_capacity = max_point_triangle_pairs is not None
        self.fixed_ee_capacity = max_edge_edge_pairs is not None
        self.pt_capacity = max(int(max_point_triangle_pairs or 1), 1)
        self.ee_capacity = max(int(max_edge_edge_pairs or 1), 1)
        self.point_triangle = ti.Vector.field(4, dtype=ti.i32, shape=self.pt_capacity)
        self.point_triangle_primitive = ti.Vector.field(2, dtype=ti.i32, shape=self.pt_capacity)
        self.edge_edge = ti.Vector.field(4, dtype=ti.i32, shape=self.ee_capacity)
        self.point_triangle_measure = ti.field(dtype=self.real_type, shape=self.pt_capacity)
        self.edge_edge_measure = ti.field(dtype=self.real_type, shape=self.ee_capacity)
        self.point_triangle_count = ti.field(dtype=ti.i32, shape=())
        self.edge_edge_count = ti.field(dtype=ti.i32, shape=())

    def _load_topology(self, topology, children, primitive, primitive_leaf, postorder):
        child_values, primitive_values, leaves, internal, _ = topology
        children.from_numpy(child_values)
        primitive.from_numpy(primitive_values)
        leaf_buffer = np.zeros(primitive_leaf.shape[0], dtype=np.int32)
        leaf_buffer[: leaves.size] = leaves
        primitive_leaf.from_numpy(leaf_buffer)
        postorder_buffer = np.zeros(postorder.shape[0], dtype=np.int32)
        postorder_buffer[: internal.size] = internal
        postorder.from_numpy(postorder_buffer)

    @ti.func
    def _point_bounds(self, node, positions, end_positions, swept):
        lower = positions[node]
        upper = positions[node]
        if swept != 0:
            lower = ti.min(lower, end_positions[node])
            upper = ti.max(upper, end_positions[node])
        return lower, upper

    @ti.func
    def _overlap(self, lower_a, upper_a, lower_b, upper_b, radius):
        return (
            lower_a[0] <= upper_b[0] + radius
            and upper_a[0] + radius >= lower_b[0]
            and lower_a[1] <= upper_b[1] + radius
            and upper_a[1] + radius >= lower_b[1]
            and lower_a[2] <= upper_b[2] + radius
            and upper_a[2] + radius >= lower_b[2]
        )

    @ti.kernel
    def _refit_leaves(
        self,
        positions: ti.template(),
        end_positions: ti.template(),
        swept: ti.i32,
    ):
        for face_id in range(self.face_count):
            face = self.faces[face_id]
            lower, upper = self._point_bounds(face[0], positions, end_positions, swept)
            for local in ti.static(range(1, 3)):
                value_lower, value_upper = self._point_bounds(face[local], positions, end_positions, swept)
                lower = ti.min(lower, value_lower)
                upper = ti.max(upper, value_upper)
            node = self.face_primitive_leaf[face_id]
            self.face_lower[node] = lower
            self.face_upper[node] = upper
        for edge_id in range(self.edge_count):
            edge = self.edges[edge_id]
            lower0, upper0 = self._point_bounds(edge[0], positions, end_positions, swept)
            lower1, upper1 = self._point_bounds(edge[1], positions, end_positions, swept)
            node = self.edge_primitive_leaf[edge_id]
            self.edge_lower[node] = ti.min(lower0, lower1)
            self.edge_upper[node] = ti.max(upper0, upper1)

    @ti.kernel
    def _refit_internal_nodes(self):
        ti.loop_config(serialize=True)
        for order in range(self.face_internal_count):
            node = self.face_postorder[order]
            children = self.face_child[node]
            self.face_lower[node] = ti.min(self.face_lower[children[0]], self.face_lower[children[1]])
            self.face_upper[node] = ti.max(self.face_upper[children[0]], self.face_upper[children[1]])
        ti.loop_config(serialize=True)
        for order in range(self.edge_internal_count):
            node = self.edge_postorder[order]
            children = self.edge_child[node]
            self.edge_lower[node] = ti.min(self.edge_lower[children[0]], self.edge_lower[children[1]])
            self.edge_upper[node] = ti.max(self.edge_upper[children[0]], self.edge_upper[children[1]])

    @ti.func
    def _valid_pt(self, node, face_id):
        face = self.faces[face_id]
        valid = node != face[0] and node != face[1] and node != face[2]
        if ti.static(self.cross_system_only):
            valid = valid and self.node_system[node] != self.face_system[face_id]
        return valid

    @ti.func
    def _valid_ee(self, first_id, second_id):
        first = self.edges[first_id]
        second = self.edges[second_id]
        valid = (
            first_id < second_id
            and first[0] != second[0]
            and first[0] != second[1]
            and first[1] != second[0]
            and first[1] != second[1]
        )
        if ti.static(self.cross_system_only):
            valid = valid and self.edge_system[first_id] != self.edge_system[second_id]
        return valid

    @ti.func
    def _valid_ee_proximity(
        self,
        first_id,
        second_id,
        positions,
        swept,
    ):
        valid = self._valid_ee(first_id, second_id)
        if ti.static(self.exact_proximity):
            if valid and swept == 0:
                first = self.edges[first_id]
                second = self.edges[second_id]
                valid = (
                    edge_edge_distance2(
                        positions[first[0]],
                        positions[first[1]],
                        positions[second[0]],
                        positions[second[1]],
                    )
                    < self.edge_radius[None] * self.edge_radius[None]
                )
        return valid

    @ti.func
    def _record_nearest_point_triangle(
        self,
        local_vertex,
        node,
        face_id,
        positions,
    ):
        face = self.faces[face_id]
        distance2 = point_triangle_distance2(
            positions[node],
            positions[face[0]],
            positions[face[1]],
            positions[face[2]],
        )
        if distance2 < self.radius[None] * self.radius[None]:
            target_system = self.face_system[face_id]
            slot = local_vertex * self.system_count + target_system
            previous_face = self.pt_nearest_face[slot]
            previous_distance2 = self.pt_nearest_distance2[slot]
            tolerance = 1.0e-14 * ti.max(1.0, previous_distance2)
            if previous_face < 0 or distance2 < previous_distance2 - tolerance:
                self.pt_nearest_face[slot] = face_id
                self.pt_nearest_distance2[slot] = distance2

    @ti.kernel
    def _count_candidates(
        self,
        positions: ti.template(),
        end_positions: ti.template(),
        swept: ti.i32,
    ):
        self.pt_prefix[0] = 0
        self.ee_prefix[0] = 0
        for local_vertex in range(self.vertex_count):
            node = self.vertices[local_vertex]
            if ti.static(self.exact_proximity):
                if swept == 0:
                    for target_system in range(self.system_count):
                        slot = local_vertex * self.system_count + target_system
                        self.pt_nearest_face[slot] = -1
                        self.pt_nearest_distance2[slot] = 1.0e30
            lower, upper = self._point_bounds(node, positions, end_positions, swept)
            count = 0
            stack_size = 1
            self.pt_stack[local_vertex, 0] = 0
            while stack_size > 0:
                stack_size -= 1
                tree_node = self.pt_stack[local_vertex, stack_size]
                if self._overlap(
                    lower,
                    upper,
                    self.face_lower[tree_node],
                    self.face_upper[tree_node],
                    self.radius[None],
                ):
                    primitive = self.face_primitive[tree_node]
                    if primitive >= 0:
                        if ti.static(self.exact_proximity):
                            if swept == 0:
                                if self._valid_pt(node, primitive):
                                    self._record_nearest_point_triangle(
                                        local_vertex,
                                        node,
                                        primitive,
                                        positions,
                                    )
                            elif self._valid_pt(node, primitive):
                                count += 1
                        elif self._valid_pt(node, primitive):
                            count += 1
                    else:
                        children = self.face_child[tree_node]
                        self.pt_stack[local_vertex, stack_size] = children[0]
                        stack_size += 1
                        self.pt_stack[local_vertex, stack_size] = children[1]
                        stack_size += 1
            if ti.static(self.exact_proximity):
                if swept == 0:
                    for target_system in range(self.system_count):
                        slot = local_vertex * self.system_count + target_system
                        if self.pt_nearest_face[slot] >= 0:
                            count += 1
            self.pt_prefix[local_vertex + 1] = count
        for first_id in range(self.edge_count):
            leaf = self.edge_primitive_leaf[first_id]
            lower = self.edge_lower[leaf]
            upper = self.edge_upper[leaf]
            count = 0
            stack_size = 1
            self.ee_stack[first_id, 0] = 0
            while stack_size > 0:
                stack_size -= 1
                tree_node = self.ee_stack[first_id, stack_size]
                if self._overlap(
                    lower,
                    upper,
                    self.edge_lower[tree_node],
                    self.edge_upper[tree_node],
                    self.edge_radius[None],
                ):
                    primitive = self.edge_primitive[tree_node]
                    if primitive >= 0:
                        if self._valid_ee_proximity(
                            first_id,
                            primitive,
                            positions,
                            swept,
                        ):
                            count += 1
                    else:
                        children = self.edge_child[tree_node]
                        self.ee_stack[first_id, stack_size] = children[0]
                        stack_size += 1
                        self.ee_stack[first_id, stack_size] = children[1]
                        stack_size += 1
            self.ee_prefix[first_id + 1] = count

    def _resize_candidates(self):
        pt_required = int(self.pt_prefix[self.vertex_count])
        ee_required = int(self.ee_prefix[self.edge_count])
        if self.fixed_pt_capacity and pt_required > self.pt_capacity:
            raise RuntimeError(
                "point-triangle broad-phase capacity is too small: " f"need {pt_required}, allocated {self.pt_capacity}"
            )
        if self.fixed_ee_capacity and ee_required > self.ee_capacity:
            raise RuntimeError(
                "edge-edge broad-phase capacity is too small: " f"need {ee_required}, allocated {self.ee_capacity}"
            )
        capacity = _grown_capacity(pt_required, self.pt_capacity)
        if capacity != self.pt_capacity:
            self.pt_capacity = capacity
            self.point_triangle = ti.Vector.field(4, dtype=ti.i32, shape=capacity)
            self.point_triangle_primitive = ti.Vector.field(2, dtype=ti.i32, shape=capacity)
            self.point_triangle_measure = ti.field(dtype=self.real_type, shape=capacity)
        capacity = _grown_capacity(ee_required, self.ee_capacity)
        if capacity != self.ee_capacity:
            self.ee_capacity = capacity
            self.edge_edge = ti.Vector.field(4, dtype=ti.i32, shape=capacity)
            self.edge_edge_measure = ti.field(dtype=self.real_type, shape=capacity)

    @ti.kernel
    def _fill_candidates(
        self,
        positions: ti.template(),
        end_positions: ti.template(),
        swept: ti.i32,
        point_triangle: ti.template(),
        point_triangle_primitive: ti.template(),
        point_triangle_measure: ti.template(),
        edge_edge: ti.template(),
        edge_edge_measure: ti.template(),
    ):
        for local_vertex in range(self.vertex_count):
            node = self.vertices[local_vertex]
            lower, upper = self._point_bounds(node, positions, end_positions, swept)
            output = self.pt_prefix[local_vertex]
            use_nearest = 0
            if ti.static(self.exact_proximity):
                use_nearest = swept == 0
            if use_nearest != 0:
                for target_system in range(self.system_count):
                    slot = local_vertex * self.system_count + target_system
                    primitive = self.pt_nearest_face[slot]
                    if primitive >= 0:
                        face = self.faces[primitive]
                        point_triangle[output] = ti.Vector([node, face[0], face[1], face[2]])
                        point_triangle_primitive[output] = ti.Vector([local_vertex, primitive])
                        point_triangle_measure[output] = 0.25 * self.node_area[node]
                        output += 1
            else:
                stack_size = 1
                self.pt_stack[local_vertex, 0] = 0
                while stack_size > 0:
                    stack_size -= 1
                    tree_node = self.pt_stack[local_vertex, stack_size]
                    if self._overlap(
                        lower,
                        upper,
                        self.face_lower[tree_node],
                        self.face_upper[tree_node],
                        self.radius[None],
                    ):
                        primitive = self.face_primitive[tree_node]
                        if primitive >= 0:
                            if self._valid_pt(node, primitive):
                                face = self.faces[primitive]
                                point_triangle[output] = ti.Vector([node, face[0], face[1], face[2]])
                                point_triangle_primitive[output] = ti.Vector([local_vertex, primitive])
                                point_triangle_measure[output] = 0.25 * self.node_area[node]
                                output += 1
                        else:
                            children = self.face_child[tree_node]
                            self.pt_stack[local_vertex, stack_size] = children[0]
                            stack_size += 1
                            self.pt_stack[local_vertex, stack_size] = children[1]
                            stack_size += 1
        for first_id in range(self.edge_count):
            leaf = self.edge_primitive_leaf[first_id]
            lower = self.edge_lower[leaf]
            upper = self.edge_upper[leaf]
            output = self.ee_prefix[first_id]
            stack_size = 1
            self.ee_stack[first_id, 0] = 0
            while stack_size > 0:
                stack_size -= 1
                tree_node = self.ee_stack[first_id, stack_size]
                if self._overlap(
                    lower,
                    upper,
                    self.edge_lower[tree_node],
                    self.edge_upper[tree_node],
                    self.edge_radius[None],
                ):
                    second_id = self.edge_primitive[tree_node]
                    if second_id >= 0:
                        if self._valid_ee_proximity(
                            first_id,
                            second_id,
                            positions,
                            swept,
                        ):
                            first = self.edges[first_id]
                            second = self.edges[second_id]
                            edge_edge[output] = ti.Vector([first[0], first[1], second[0], second[1]])
                            edge_edge_measure[output] = 0.25 * (self.edge_area[first_id] + self.edge_area[second_id])
                            output += 1
                    else:
                        children = self.edge_child[tree_node]
                        self.ee_stack[first_id, stack_size] = children[0]
                        stack_size += 1
                        self.ee_stack[first_id, stack_size] = children[1]
                        stack_size += 1

    @ti.kernel
    def _set_candidate_counts(self):
        self.point_triangle_count[None] = self.pt_prefix[self.vertex_count]
        self.edge_edge_count[None] = self.ee_prefix[self.edge_count]

    def rebuild(
        self,
        positions,
        radius,
        end_positions=None,
        edge_radius=None,
    ):
        swept = int(end_positions is not None)
        end_positions = positions if end_positions is None else end_positions
        self.radius[None] = float(radius)
        self.edge_radius[None] = float(radius if edge_radius is None else edge_radius)
        self._refit_leaves(positions, end_positions, swept)
        self._refit_internal_nodes()
        self._count_candidates(positions, end_positions, swept)
        self.pt_scan.run(self.pt_prefix)
        self.ee_scan.run(self.ee_prefix)
        self._resize_candidates()
        self._fill_candidates(
            positions,
            end_positions,
            swept,
            self.point_triangle,
            self.point_triangle_primitive,
            self.point_triangle_measure,
            self.edge_edge,
            self.edge_edge_measure,
        )
        self._set_candidate_counts()
        return (
            int(self.point_triangle_count[None]),
            int(self.edge_edge_count[None]),
        )


__all__ = ["DynamicBVHBroadPhase"]

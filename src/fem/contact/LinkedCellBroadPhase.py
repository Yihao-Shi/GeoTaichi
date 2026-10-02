"""Dynamic linked-cell broad phase for deforming FEM contact surfaces."""

import math

import numpy as np
import taichi as ti

from src.utils.PrefixSum import PrefixSumExecutor


def _grown_capacity(required, current):
    required = max(int(required), 1)
    current = max(int(current), 1)
    if required <= current:
        return current
    return 1 << (required - 1).bit_length()


@ti.data_oriented
class DynamicLinkedCellBroadPhase:
    """Compact deforming-surface broad phase without per-cell capacities.

    Every rebuild performs count -> prefix sum -> fill for face, vertex and
    edge memberships.  Candidate pairs use the same three-pass construction.
    Only scalar required-size values are read by Python when a global compact
    buffer has to grow; primitive data and contact pairs never leave Taichi.
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
        target_cell_count=None,
        max_point_triangle_pairs=None,
        max_edge_edge_pairs=None,
        node_system_ids=None,
        cross_system_only=False,
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
        system_values = (
            np.zeros(reference_positions.shape[0], dtype=np.int32)
            if node_system_ids is None
            else np.asarray(node_system_ids, dtype=np.int32)
        )
        if system_values.shape != (reference_positions.shape[0],):
            raise ValueError("broad-phase node_system_ids must contain one ID per node")
        face_system_values = system_values[faces[:, 0]] if self.face_count else np.zeros(1, dtype=np.int32)
        edge_system_values = system_values[edges[:, 0]] if self.edge_count else np.zeros(1, dtype=np.int32)
        self.node_system = ti.field(dtype=ti.i32, shape=reference_positions.shape[0])
        self.face_system = ti.field(dtype=ti.i32, shape=max(self.face_count, 1))
        self.edge_system = ti.field(dtype=ti.i32, shape=max(self.edge_count, 1))
        self.node_system.from_numpy(np.ascontiguousarray(system_values))
        self.face_system.from_numpy(np.ascontiguousarray(face_system_values, dtype=np.int32))
        self.edge_system.from_numpy(np.ascontiguousarray(edge_system_values, dtype=np.int32))

        primitive_count = max(self.face_count + self.edge_count, 1)
        requested_cells = max(8, 2 * primitive_count) if target_cell_count is None else max(int(target_cell_count), 1)
        extent = np.ptp(reference_positions, axis=0)
        positive = extent[extent > 1.0e-12]
        scale = float(np.prod(positive) ** (1.0 / max(positive.size, 1))) if positive.size else 1.0
        aspect = np.maximum(extent / max(scale, 1.0e-12), 0.25)
        multiplier = (requested_cells / float(np.prod(aspect))) ** (1.0 / 3.0)
        dims = np.maximum(np.rint(multiplier * aspect).astype(np.int32), 1)
        while int(np.prod(dims)) > max(4 * requested_cells, requested_cells + 64):
            dims[np.argmax(dims)] -= 1
            dims = np.maximum(dims, 1)
        self.grid_x, self.grid_y, self.grid_z = map(int, dims)
        self.cell_count = int(np.prod(dims))

        self.faces = ti.Vector.field(3, dtype=ti.i32, shape=max(self.face_count, 1))
        self.edges = ti.Vector.field(2, dtype=ti.i32, shape=max(self.edge_count, 1))
        self.vertices = ti.field(dtype=ti.i32, shape=max(self.vertex_count, 1))
        self.node_area = ti.field(dtype=self.real_type, shape=reference_positions.shape[0])
        self.edge_area = ti.field(dtype=self.real_type, shape=max(self.edge_count, 1))
        self.reference_position = ti.Vector.field(3, dtype=self.real_type, shape=reference_positions.shape[0])
        if self.face_count:
            face_buffer = np.zeros((max(self.face_count, 1), 3), dtype=np.int32)
            face_buffer[: self.face_count] = faces
            self.faces.from_numpy(face_buffer)
        if self.edge_count:
            edge_buffer = np.zeros((max(self.edge_count, 1), 2), dtype=np.int32)
            edge_buffer[: self.edge_count] = edges
            self.edges.from_numpy(edge_buffer)
        if self.vertex_count:
            vertex_buffer = np.zeros(max(self.vertex_count, 1), dtype=np.int32)
            vertex_buffer[: self.vertex_count] = vertices
            self.vertices.from_numpy(vertex_buffer)
        self.node_area.from_numpy(np.ascontiguousarray(node_area, dtype=self.numpy_type))
        edge_area_buffer = np.zeros(max(self.edge_count, 1), dtype=self.numpy_type)
        edge_area_buffer[: self.edge_count] = edge_area
        self.edge_area.from_numpy(edge_area_buffer)
        self.reference_position.from_numpy(np.ascontiguousarray(reference_positions, dtype=self.numpy_type))

        self.grid_lower = ti.Vector.field(3, dtype=self.real_type, shape=())
        self.grid_upper = ti.Vector.field(3, dtype=self.real_type, shape=())
        self.inverse_cell_size = ti.Vector.field(3, dtype=self.real_type, shape=())
        self.radius = ti.field(dtype=self.real_type, shape=())

        # Prefix arrays contain an initial zero, hence cell_count + 1.
        prefix_shape = self.cell_count + 1
        self.prefix_sum = PrefixSumExecutor(prefix_shape)
        scan_shape = self.prefix_sum.get_length()
        self.face_prefix = ti.field(dtype=ti.i32, shape=scan_shape)
        self.vertex_prefix = ti.field(dtype=ti.i32, shape=scan_shape)
        self.edge_prefix = ti.field(dtype=ti.i32, shape=scan_shape)
        self.face_cursor = ti.field(dtype=ti.i32, shape=self.cell_count)
        self.vertex_cursor = ti.field(dtype=ti.i32, shape=self.cell_count)
        self.edge_cursor = ti.field(dtype=ti.i32, shape=self.cell_count)
        self.pt_prefix = ti.field(dtype=ti.i32, shape=scan_shape)
        self.ee_prefix = ti.field(dtype=ti.i32, shape=scan_shape)

        self.face_member_capacity = 1
        self.vertex_member_capacity = 1
        self.edge_member_capacity = 1
        self.fixed_pt_capacity = max_point_triangle_pairs is not None
        self.fixed_ee_capacity = max_edge_edge_pairs is not None
        self.pt_capacity = max(int(max_point_triangle_pairs or 1), 1)
        self.ee_capacity = max(int(max_edge_edge_pairs or 1), 1)
        self.face_members = ti.field(dtype=ti.i32, shape=1)
        self.vertex_members = ti.field(dtype=ti.i32, shape=1)
        self.edge_members = ti.field(dtype=ti.i32, shape=1)
        self.point_triangle = ti.Vector.field(4, dtype=ti.i32, shape=self.pt_capacity)
        self.point_triangle_primitive = ti.Vector.field(2, dtype=ti.i32, shape=self.pt_capacity)
        self.edge_edge = ti.Vector.field(4, dtype=ti.i32, shape=self.ee_capacity)
        self.point_triangle_measure = ti.field(dtype=self.real_type, shape=self.pt_capacity)
        self.edge_edge_measure = ti.field(dtype=self.real_type, shape=self.ee_capacity)
        self.point_triangle_count = ti.field(dtype=ti.i32, shape=())
        self.edge_edge_count = ti.field(dtype=ti.i32, shape=())

    @ti.func
    def _cell_coordinate(self, point):
        coordinate = ti.cast(
            ti.floor((point - self.grid_lower[None]) * self.inverse_cell_size[None]),
            ti.i32,
        )
        coordinate[0] = ti.max(0, ti.min(coordinate[0], self.grid_x - 1))
        coordinate[1] = ti.max(0, ti.min(coordinate[1], self.grid_y - 1))
        coordinate[2] = ti.max(0, ti.min(coordinate[2], self.grid_z - 1))
        return coordinate

    @ti.func
    def _cell_id(self, coordinate):
        return coordinate[0] + self.grid_x * (coordinate[1] + self.grid_y * coordinate[2])

    @ti.func
    def _point_bounds(self, node, positions, end_positions, swept):
        lower = positions[node]
        upper = positions[node]
        if swept != 0:
            lower = ti.min(lower, end_positions[node])
            upper = ti.max(upper, end_positions[node])
        return lower, upper

    @ti.func
    def _face_bounds(self, face_id, positions, end_positions, swept, expand):
        nodes = self.faces[face_id]
        lower, upper = self._point_bounds(nodes[0], positions, end_positions, swept)
        for local in ti.static(range(1, 3)):
            node_lower, node_upper = self._point_bounds(nodes[local], positions, end_positions, swept)
            lower = ti.min(lower, node_lower)
            upper = ti.max(upper, node_upper)
        lower -= expand
        upper += expand
        return lower, upper

    @ti.func
    def _edge_bounds(self, edge_id, positions, end_positions, swept, expand):
        nodes = self.edges[edge_id]
        lower0, upper0 = self._point_bounds(nodes[0], positions, end_positions, swept)
        lower1, upper1 = self._point_bounds(nodes[1], positions, end_positions, swept)
        # Expanding both edge boxes by the full search distance would accept
        # pairs separated by twice the requested radius. Split the distance
        # evenly so the pairwise Minkowski expansion remains ``expand``.
        half_expand = 0.5 * expand
        return (
            ti.min(lower0, lower1) - half_expand,
            ti.max(upper0, upper1) + half_expand,
        )

    @ti.func
    def _overlap(self, lower_a, upper_a, lower_b, upper_b):
        return (
            upper_a[0] >= lower_b[0]
            and lower_a[0] <= upper_b[0]
            and upper_a[1] >= lower_b[1]
            and lower_a[1] <= upper_b[1]
            and upper_a[2] >= lower_b[2]
            and lower_a[2] <= upper_b[2]
        )

    @ti.kernel
    def _initialize_grid(
        self,
        positions: ti.template(),
        end_positions: ti.template(),
        swept: ti.i32,
        radius: ti.f64,
    ):
        for component in ti.static(range(3)):
            self.grid_lower[None][component] = 1.0e30
            self.grid_upper[None][component] = -1.0e30
        for node in range(positions.shape[0]):
            for component in ti.static(range(3)):
                ti.atomic_min(self.grid_lower[None][component], positions[node][component])
                ti.atomic_max(self.grid_upper[None][component], positions[node][component])
                if swept != 0:
                    ti.atomic_min(
                        self.grid_lower[None][component],
                        end_positions[node][component],
                    )
                    ti.atomic_max(
                        self.grid_upper[None][component],
                        end_positions[node][component],
                    )
        dimensions = ti.Vector([self.grid_x, self.grid_y, self.grid_z])
        for component in ti.static(range(3)):
            self.grid_lower[None][component] -= radius + 1.0e-12
            self.grid_upper[None][component] += radius + 1.0e-12
            extent = self.grid_upper[None][component] - self.grid_lower[None][component]
            self.inverse_cell_size[None][component] = dimensions[component] / ti.max(extent, 1.0e-12)
        self.radius[None] = radius

    @ti.kernel
    def _count_memberships(
        self,
        positions: ti.template(),
        end_positions: ti.template(),
        swept: ti.i32,
    ):
        for index in range(self.cell_count + 1):
            self.face_prefix[index] = 0
            self.vertex_prefix[index] = 0
            self.edge_prefix[index] = 0
        for face_id in range(self.face_count):
            lower, upper = self._face_bounds(face_id, positions, end_positions, swept, self.radius[None])
            first, last = self._cell_coordinate(lower), self._cell_coordinate(upper)
            for x, y, z in ti.ndrange(
                (first[0], last[0] + 1),
                (first[1], last[1] + 1),
                (first[2], last[2] + 1),
            ):
                ti.atomic_add(
                    self.face_prefix[self._cell_id(ti.Vector([x, y, z])) + 1],
                    1,
                )
        for local_vertex in range(self.vertex_count):
            node = self.vertices[local_vertex]
            lower, upper = self._point_bounds(node, positions, end_positions, swept)
            first, last = self._cell_coordinate(lower), self._cell_coordinate(upper)
            for x, y, z in ti.ndrange(
                (first[0], last[0] + 1),
                (first[1], last[1] + 1),
                (first[2], last[2] + 1),
            ):
                ti.atomic_add(
                    self.vertex_prefix[self._cell_id(ti.Vector([x, y, z])) + 1],
                    1,
                )
        for edge_id in range(self.edge_count):
            lower, upper = self._edge_bounds(edge_id, positions, end_positions, swept, self.radius[None])
            first, last = self._cell_coordinate(lower), self._cell_coordinate(upper)
            for x, y, z in ti.ndrange(
                (first[0], last[0] + 1),
                (first[1], last[1] + 1),
                (first[2], last[2] + 1),
            ):
                ti.atomic_add(
                    self.edge_prefix[self._cell_id(ti.Vector([x, y, z])) + 1],
                    1,
                )

    def _scan_memberships(self):
        self.prefix_sum.run(self.face_prefix)
        self.prefix_sum.run(self.vertex_prefix)
        self.prefix_sum.run(self.edge_prefix)

    def _resize_memberships(self):
        face_required = int(self.face_prefix[self.cell_count])
        vertex_required = int(self.vertex_prefix[self.cell_count])
        edge_required = int(self.edge_prefix[self.cell_count])
        new_capacity = _grown_capacity(face_required, self.face_member_capacity)
        if new_capacity != self.face_member_capacity:
            self.face_member_capacity = new_capacity
            self.face_members = ti.field(dtype=ti.i32, shape=new_capacity)
        new_capacity = _grown_capacity(vertex_required, self.vertex_member_capacity)
        if new_capacity != self.vertex_member_capacity:
            self.vertex_member_capacity = new_capacity
            self.vertex_members = ti.field(dtype=ti.i32, shape=new_capacity)
        new_capacity = _grown_capacity(edge_required, self.edge_member_capacity)
        if new_capacity != self.edge_member_capacity:
            self.edge_member_capacity = new_capacity
            self.edge_members = ti.field(dtype=ti.i32, shape=new_capacity)

    @ti.kernel
    def _initialize_cursors(self):
        for cell in range(self.cell_count):
            self.face_cursor[cell] = self.face_prefix[cell]
            self.vertex_cursor[cell] = self.vertex_prefix[cell]
            self.edge_cursor[cell] = self.edge_prefix[cell]

    @ti.kernel
    def _fill_memberships(
        self,
        positions: ti.template(),
        end_positions: ti.template(),
        swept: ti.i32,
        face_members: ti.template(),
        vertex_members: ti.template(),
        edge_members: ti.template(),
    ):
        for face_id in range(self.face_count):
            lower, upper = self._face_bounds(face_id, positions, end_positions, swept, self.radius[None])
            first, last = self._cell_coordinate(lower), self._cell_coordinate(upper)
            for x, y, z in ti.ndrange(
                (first[0], last[0] + 1),
                (first[1], last[1] + 1),
                (first[2], last[2] + 1),
            ):
                cell = self._cell_id(ti.Vector([x, y, z]))
                slot = ti.atomic_add(self.face_cursor[cell], 1)
                face_members[slot] = face_id
        for local_vertex in range(self.vertex_count):
            node = self.vertices[local_vertex]
            lower, upper = self._point_bounds(node, positions, end_positions, swept)
            first, last = self._cell_coordinate(lower), self._cell_coordinate(upper)
            for x, y, z in ti.ndrange(
                (first[0], last[0] + 1),
                (first[1], last[1] + 1),
                (first[2], last[2] + 1),
            ):
                cell = self._cell_id(ti.Vector([x, y, z]))
                slot = ti.atomic_add(self.vertex_cursor[cell], 1)
                vertex_members[slot] = local_vertex
        for edge_id in range(self.edge_count):
            lower, upper = self._edge_bounds(edge_id, positions, end_positions, swept, self.radius[None])
            first, last = self._cell_coordinate(lower), self._cell_coordinate(upper)
            for x, y, z in ti.ndrange(
                (first[0], last[0] + 1),
                (first[1], last[1] + 1),
                (first[2], last[2] + 1),
            ):
                cell = self._cell_id(ti.Vector([x, y, z]))
                slot = ti.atomic_add(self.edge_cursor[cell], 1)
                edge_members[slot] = edge_id

    @ti.func
    def _canonical_cell(self, lower_a, lower_b):
        first_a = self._cell_coordinate(lower_a)
        first_b = self._cell_coordinate(lower_b)
        return self._cell_id(ti.max(first_a, first_b))

    @ti.func
    def _valid_pt(
        self,
        cell,
        local_vertex,
        face_id,
        positions,
        end_positions,
        swept,
    ):
        node = self.vertices[local_vertex]
        face = self.faces[face_id]
        valid = node != face[0] and node != face[1] and node != face[2]
        if ti.static(self.cross_system_only):
            valid = valid and self.node_system[node] != self.face_system[face_id]
        point_lower, point_upper = self._point_bounds(node, positions, end_positions, swept)
        face_lower, face_upper = self._face_bounds(face_id, positions, end_positions, swept, self.radius[None])
        valid = valid and self._overlap(point_lower, point_upper, face_lower, face_upper)
        valid = valid and cell == self._canonical_cell(point_lower, face_lower)
        return valid

    @ti.func
    def _valid_ee(
        self,
        cell,
        first_edge,
        second_edge,
        positions,
        end_positions,
        swept,
    ):
        first = self.edges[first_edge]
        second = self.edges[second_edge]
        valid = first_edge < second_edge
        if ti.static(self.cross_system_only):
            valid = valid and self.edge_system[first_edge] != self.edge_system[second_edge]
        valid = valid and first[0] != second[0] and first[0] != second[1]
        valid = valid and first[1] != second[0] and first[1] != second[1]
        first_lower, first_upper = self._edge_bounds(first_edge, positions, end_positions, swept, self.radius[None])
        second_lower, second_upper = self._edge_bounds(second_edge, positions, end_positions, swept, self.radius[None])
        valid = valid and self._overlap(first_lower, first_upper, second_lower, second_upper)
        valid = valid and cell == self._canonical_cell(first_lower, second_lower)
        return valid

    @ti.kernel
    def _count_candidates(
        self,
        positions: ti.template(),
        end_positions: ti.template(),
        swept: ti.i32,
        face_members: ti.template(),
        vertex_members: ti.template(),
        edge_members: ti.template(),
    ):
        for index in range(self.cell_count + 1):
            self.pt_prefix[index] = 0
            self.ee_prefix[index] = 0
        for cell in range(self.cell_count):
            pt_count = 0
            for vertex_slot in range(self.vertex_prefix[cell], self.vertex_prefix[cell + 1]):
                local_vertex = vertex_members[vertex_slot]
                for face_slot in range(self.face_prefix[cell], self.face_prefix[cell + 1]):
                    face_id = face_members[face_slot]
                    if self._valid_pt(
                        cell,
                        local_vertex,
                        face_id,
                        positions,
                        end_positions,
                        swept,
                    ):
                        pt_count += 1
            ee_count = 0
            for first_slot in range(self.edge_prefix[cell], self.edge_prefix[cell + 1]):
                first_edge = edge_members[first_slot]
                for second_slot in range(first_slot + 1, self.edge_prefix[cell + 1]):
                    second_edge = edge_members[second_slot]
                    if self._valid_ee(
                        cell,
                        first_edge,
                        second_edge,
                        positions,
                        end_positions,
                        swept,
                    ):
                        ee_count += 1
            self.pt_prefix[cell + 1] = pt_count
            self.ee_prefix[cell + 1] = ee_count

    def _resize_candidates(self):
        pt_required = int(self.pt_prefix[self.cell_count])
        ee_required = int(self.ee_prefix[self.cell_count])
        if self.fixed_pt_capacity and pt_required > self.pt_capacity:
            raise RuntimeError(
                "point-triangle broad-phase capacity is too small: " f"need {pt_required}, allocated {self.pt_capacity}"
            )
        if self.fixed_ee_capacity and ee_required > self.ee_capacity:
            raise RuntimeError(
                "edge-edge broad-phase capacity is too small: " f"need {ee_required}, allocated {self.ee_capacity}"
            )
        new_capacity = _grown_capacity(pt_required, self.pt_capacity)
        if new_capacity != self.pt_capacity:
            self.pt_capacity = new_capacity
            self.point_triangle = ti.Vector.field(4, dtype=ti.i32, shape=new_capacity)
            self.point_triangle_primitive = ti.Vector.field(2, dtype=ti.i32, shape=new_capacity)
            self.point_triangle_measure = ti.field(dtype=self.real_type, shape=new_capacity)
        new_capacity = _grown_capacity(ee_required, self.ee_capacity)
        if new_capacity != self.ee_capacity:
            self.ee_capacity = new_capacity
            self.edge_edge = ti.Vector.field(4, dtype=ti.i32, shape=new_capacity)
            self.edge_edge_measure = ti.field(dtype=self.real_type, shape=new_capacity)

    @ti.kernel
    def _fill_candidates(
        self,
        positions: ti.template(),
        end_positions: ti.template(),
        swept: ti.i32,
        face_members: ti.template(),
        vertex_members: ti.template(),
        edge_members: ti.template(),
        point_triangle: ti.template(),
        point_triangle_primitive: ti.template(),
        point_triangle_measure: ti.template(),
        edge_edge: ti.template(),
        edge_edge_measure: ti.template(),
    ):
        for cell in range(self.cell_count):
            pt_output = self.pt_prefix[cell]
            for vertex_slot in range(self.vertex_prefix[cell], self.vertex_prefix[cell + 1]):
                local_vertex = vertex_members[vertex_slot]
                node = self.vertices[local_vertex]
                for face_slot in range(self.face_prefix[cell], self.face_prefix[cell + 1]):
                    face_id = face_members[face_slot]
                    if self._valid_pt(
                        cell,
                        local_vertex,
                        face_id,
                        positions,
                        end_positions,
                        swept,
                    ):
                        face = self.faces[face_id]
                        point_triangle[pt_output] = ti.Vector([node, face[0], face[1], face[2]])
                        # Preserve the source primitive ids for coupled
                        # systems whose point and triangle use different DOF
                        # maps.  Existing IPC consumers can ignore this field.
                        point_triangle_primitive[pt_output] = ti.Vector([local_vertex, face_id])
                        point_triangle_measure[pt_output] = 0.25 * self.node_area[node]
                        pt_output += 1
            ee_output = self.ee_prefix[cell]
            for first_slot in range(self.edge_prefix[cell], self.edge_prefix[cell + 1]):
                first_edge = edge_members[first_slot]
                for second_slot in range(first_slot + 1, self.edge_prefix[cell + 1]):
                    second_edge = edge_members[second_slot]
                    if self._valid_ee(
                        cell,
                        first_edge,
                        second_edge,
                        positions,
                        end_positions,
                        swept,
                    ):
                        first, second = self.edges[first_edge], self.edges[second_edge]
                        edge_edge[ee_output] = ti.Vector([first[0], first[1], second[0], second[1]])
                        edge_edge_measure[ee_output] = 0.25 * (self.edge_area[first_edge] + self.edge_area[second_edge])
                        ee_output += 1

    @ti.kernel
    def _set_candidate_counts(self):
        self.point_triangle_count[None] = self.pt_prefix[self.cell_count]
        self.edge_edge_count[None] = self.ee_prefix[self.cell_count]

    def rebuild(self, positions, radius, end_positions=None):
        """Rebuild current or swept candidates; no Verlet padding is used."""
        swept = int(end_positions is not None)
        end_positions = positions if end_positions is None else end_positions
        self._initialize_grid(positions, end_positions, swept, float(radius))
        self._count_memberships(positions, end_positions, swept)
        self._scan_memberships()
        self._resize_memberships()
        self._initialize_cursors()
        self._fill_memberships(
            positions,
            end_positions,
            swept,
            self.face_members,
            self.vertex_members,
            self.edge_members,
        )
        self._count_candidates(
            positions,
            end_positions,
            swept,
            self.face_members,
            self.vertex_members,
            self.edge_members,
        )
        self.prefix_sum.run(self.pt_prefix)
        self.prefix_sum.run(self.ee_prefix)
        self._resize_candidates()
        self._fill_candidates(
            positions,
            end_positions,
            swept,
            self.face_members,
            self.vertex_members,
            self.edge_members,
            self.point_triangle,
            self.point_triangle_primitive,
            self.point_triangle_measure,
            self.edge_edge,
            self.edge_edge_measure,
        )
        self._set_candidate_counts()
        return int(self.point_triangle_count[None]), int(self.edge_edge_count[None])


__all__ = ["DynamicLinkedCellBroadPhase"]

from collections import deque
from itertools import product
from math import ceil

import numpy as np


def build_bridging_node_weights(refined_cell, coarse_cnum, bridging_cells):
    coarse_cnum = np.asarray(coarse_cnum, dtype=np.int32)
    coarse_gnum = coarse_cnum + 1
    dimension = coarse_cnum.size
    refined = np.asarray(refined_cell, dtype=np.uint8).reshape(
        tuple(coarse_cnum),
        order="F",
    )
    node_has_refined_cell = np.zeros(tuple(coarse_gnum), dtype=bool)
    node_has_coarse_cell = np.zeros(tuple(coarse_gnum), dtype=bool)

    for cell in np.ndindex(*coarse_cnum):
        target = node_has_refined_cell if refined[cell] else node_has_coarse_cell
        for offset in product((0, 1), repeat=dimension):
            node = tuple(cell[d] + offset[d] for d in range(dimension))
            target[node] = True

    interface = node_has_refined_cell & node_has_coarse_cell
    distance = np.full(tuple(coarse_gnum), -1, dtype=np.int32)
    frontier = deque()
    for node in zip(*np.nonzero(interface)):
        distance[node] = 0
        frontier.append(node)

    while frontier:
        node = frontier.popleft()
        if distance[node] >= bridging_cells:
            continue
        for axis in range(dimension):
            for direction in (-1, 1):
                neighbor = list(node)
                neighbor[axis] += direction
                if not 0 <= neighbor[axis] < coarse_gnum[axis]:
                    continue
                neighbor = tuple(neighbor)
                if node_has_refined_cell[neighbor] and distance[neighbor] < 0:
                    distance[neighbor] = distance[node] + 1
                    frontier.append(neighbor)

    weights = np.zeros(tuple(coarse_gnum), dtype=np.float64)
    in_bridge = (distance >= 0) & (distance <= bridging_cells)
    weights[in_bridge] = 1.0 - distance[in_bridge] / float(bridging_cells)
    return np.ascontiguousarray(weights.flatten(order="F"))


def build_hanging_constraint_table(
    refined_cell,
    coarse_cnum,
    fine_gnum,
    logical_to_compact,
    compact_to_logical,
    active_compact_nodes,
    refinement_ratio=2,
    max_level=1,
):
    coarse_cnum = np.asarray(coarse_cnum, dtype=np.int32)
    fine_gnum = np.asarray(fine_gnum, dtype=np.int32)
    refined_cell = np.asarray(refined_cell, dtype=np.uint8).reshape(
        tuple(coarse_cnum),
        order="F",
    )
    logical_to_compact = np.asarray(logical_to_compact, dtype=np.int32)
    compact_to_logical = np.asarray(compact_to_logical, dtype=np.int32)
    dimension = coarse_cnum.size
    corner_count = 1 << dimension
    finest_ratio = int(refinement_ratio) ** int(max_level)

    slave_ids = []
    master_ids = []
    master_weights = []
    touched_nodes = set()

    for node_id in range(active_compact_nodes):
        logical_id = int(compact_to_logical[node_id])
        if logical_id < 0:
            continue
        fine_node = np.asarray(
            np.unravel_index(logical_id, tuple(fine_gnum), order="F"),
            dtype=np.int32,
        )

        native_level = 0
        for level in range(1, max_level + 1):
            stride = finest_ratio // (refinement_ratio**level)
            if np.all(fine_node % stride == 0):
                native_level = level
        if native_level == 0:
            continue

        adjacent_levels = []
        has_adjacent_cell = False
        base_cell = fine_node // finest_ratio
        for offset in product((0, 1), repeat=dimension):
            adjacent_cell = base_cell.copy()
            for d in range(dimension):
                if fine_node[d] % finest_ratio == 0:
                    adjacent_cell[d] += offset[d] - 1
            if np.all(adjacent_cell >= 0) and np.all(adjacent_cell < coarse_cnum):
                has_adjacent_cell = True
                adjacent_levels.append(int(refined_cell[tuple(adjacent_cell)]))

        if not has_adjacent_cell:
            continue
        master_level = min(adjacent_levels)
        if master_level >= native_level:
            continue

        master_divisions = refinement_ratio**master_level
        master_stride = finest_ratio // master_divisions
        coarse_cell = np.minimum(fine_node // finest_ratio, coarse_cnum - 1)
        master_cell = np.minimum(fine_node // master_stride, coarse_cnum * master_divisions - 1)
        local_coord = (fine_node.astype(np.float64) - master_stride * master_cell.astype(np.float64)) / float(
            master_stride
        )
        ids = np.full(corner_count, -1, dtype=np.int32)
        weights = np.zeros(corner_count, dtype=np.float64)
        for corner, offset in enumerate(product((0, 1), repeat=dimension)):
            weight = 1.0
            for d in range(dimension):
                weight *= local_coord[d] if offset[d] else 1.0 - local_coord[d]
            weights[corner] = weight
            if weight > 0.0:
                master_node = master_stride * (master_cell + np.asarray(offset, dtype=np.int32))
                master_logical_id = np.ravel_multi_index(
                    tuple(master_node),
                    tuple(fine_gnum),
                    order="F",
                )
                master_id = int(logical_to_compact[master_logical_id])
                assert master_id >= 0, "AdaptiveGrid coarse node map is incomplete"
                ids[corner] = master_id
                touched_nodes.add(master_id)

        nonzero_master_ids = ids[weights > 0.0]
        if nonzero_master_ids.size == 1 and int(nonzero_master_ids[0]) == node_id:
            continue

        slave_ids.append(node_id)
        master_ids.append(ids)
        master_weights.append(weights)
        touched_nodes.add(node_id)

    return (
        np.ascontiguousarray(slave_ids, dtype=np.int32),
        np.ascontiguousarray(master_ids, dtype=np.int32).reshape(-1, corner_count),
        np.ascontiguousarray(master_weights, dtype=np.float64).reshape(
            -1,
            corner_count,
        ),
        np.ascontiguousarray(sorted(touched_nodes), dtype=np.int32),
    )


class AdaptiveNodeMap:
    def __init__(
        self,
        coarse_cnum,
        ghost_cell,
        max_refined_ratio,
        bridging_domain=False,
        max_level=1,
    ):
        self.coarse_cnum = np.asarray(coarse_cnum, dtype=np.int32)
        self.dimension = self.coarse_cnum.size
        self.bridging_domain = bridging_domain
        self.refinement_ratio = 2
        self.max_level = int(max_level)
        self.finest_ratio = self.refinement_ratio**self.max_level
        self.fine_gnum = self.finest_ratio * self.coarse_cnum + 1
        self.full_logical_node_count = int(np.prod(self.fine_gnum))
        self.coarse_gnum = self.coarse_cnum + 1
        self.coarse_node_count = int(np.prod(self.coarse_gnum))

        boundary_reserve = self._physical_boundary_fine_node_count(ghost_cell)
        refinable_node_count = self.full_logical_node_count
        maximum_node_count = self.coarse_node_count + self.full_logical_node_count
        if not self.bridging_domain:
            refinable_node_count -= self.coarse_node_count
            maximum_node_count = self.full_logical_node_count
        requested_capacity = self.coarse_node_count + ceil(max_refined_ratio * refinable_node_count) + boundary_reserve
        self.capacity = int(min(maximum_node_count, requested_capacity))

        self.logical_to_compact = np.full(
            self.full_logical_node_count,
            -1,
            dtype=np.int32,
        )
        self.fine_logical_to_compact = None
        if self.bridging_domain:
            self.fine_logical_to_compact = np.full(
                self.full_logical_node_count,
                -1,
                dtype=np.int32,
            )
        self.compact_to_logical = np.full(self.capacity, -1, dtype=np.int32)
        self.next_compact_id = 0
        self._initialize_coarse_nodes()

        self.logical_to_compact_field = None
        self.fine_logical_to_compact_field = None
        self.compact_to_logical_field = None

    def _physical_boundary_fine_node_count(self, ghost_cell):
        physical_cnum = np.maximum(self.coarse_cnum - 2 * ghost_cell, 0)
        fine_nodes = self.finest_ratio * physical_cnum + 1
        coarse_nodes = physical_cnum + 1
        if self.dimension == 2:
            fine_boundary = 2 * fine_nodes[0] + 2 * max(fine_nodes[1] - 2, 0)
            coarse_boundary = 2 * coarse_nodes[0] + 2 * max(coarse_nodes[1] - 2, 0)
        else:
            fine_boundary = int(np.prod(fine_nodes) - np.prod(np.maximum(fine_nodes - 2, 0)))
            coarse_boundary = int(np.prod(coarse_nodes) - np.prod(np.maximum(coarse_nodes - 2, 0)))
        if self.bridging_domain:
            return fine_boundary
        return max(fine_boundary - coarse_boundary, 0)

    def _logical_id(self, index):
        logical_id = int(index[0] + index[1] * self.fine_gnum[0])
        if self.dimension == 3:
            logical_id += int(index[2] * self.fine_gnum[0] * self.fine_gnum[1])
        return logical_id

    def _coarse_index(self, coarse_id):
        x = coarse_id % self.coarse_gnum[0]
        y = (coarse_id // self.coarse_gnum[0]) % self.coarse_gnum[1]
        if self.dimension == 2:
            return np.array([x, y], dtype=np.int32)
        z = coarse_id // (self.coarse_gnum[0] * self.coarse_gnum[1])
        return np.array([x, y, z], dtype=np.int32)

    def _coarse_cell_index(self, cell_id):
        x = cell_id % self.coarse_cnum[0]
        y = (cell_id // self.coarse_cnum[0]) % self.coarse_cnum[1]
        if self.dimension == 2:
            return np.array([x, y], dtype=np.int32)
        z = cell_id // (self.coarse_cnum[0] * self.coarse_cnum[1])
        return np.array([x, y, z], dtype=np.int32)

    def _initialize_coarse_nodes(self):
        for coarse_id in range(self.coarse_node_count):
            fine_index = self.finest_ratio * self._coarse_index(coarse_id)
            logical_id = self._logical_id(fine_index)
            self.logical_to_compact[logical_id] = coarse_id
            self.compact_to_logical[coarse_id] = logical_id
        self.next_compact_id = self.coarse_node_count

    def bind_fields(
        self,
        logical_to_compact_field,
        compact_to_logical_field,
        fine_logical_to_compact_field=None,
    ):
        self.logical_to_compact_field = logical_to_compact_field
        self.fine_logical_to_compact_field = fine_logical_to_compact_field
        self.compact_to_logical_field = compact_to_logical_field
        self.sync_fields()

    def sync_fields(self):
        if self.logical_to_compact_field is not None:
            self.logical_to_compact_field.from_numpy(self.logical_to_compact)
            if self.fine_logical_to_compact_field is not None:
                self.fine_logical_to_compact_field.from_numpy(self.fine_logical_to_compact)
            self.compact_to_logical_field.from_numpy(self.compact_to_logical)

    def _ensure_ids(self, logical_ids, mapping):
        logical_ids = np.unique(np.asarray(logical_ids, dtype=np.int64))
        missing = logical_ids[mapping[logical_ids] < 0]
        assert self.next_compact_id + missing.size <= self.capacity, (
            f"AdaptiveGrid dense node capacity {self.capacity} exceeded: "
            f"{self.next_compact_id + missing.size} nodes are required. "
            "Increase AdaptiveGrid/RefineRatio."
        )
        for logical_id in missing:
            compact_id = self.next_compact_id
            mapping[logical_id] = compact_id
            self.compact_to_logical[compact_id] = logical_id
            self.next_compact_id += 1
        if missing.size > 0:
            self.sync_fields()
        return mapping[logical_ids]

    def ensure_logical_ids(self, logical_ids):
        return self._ensure_ids(logical_ids, self.logical_to_compact)

    def ensure_fine_logical_ids(self, logical_ids):
        if not self.bridging_domain:
            return self.ensure_logical_ids(logical_ids)
        return self._ensure_ids(logical_ids, self.fine_logical_to_compact)

    def ensure_refined_cells(self, refined_cell, shape_function):
        halo = 1 if shape_function in ("GIMP", "QuadBSpline", "CubicBSpline") else 0
        logical_ids = []
        refined_cell = np.asarray(refined_cell, dtype=np.uint8)
        for cell_id in np.flatnonzero(refined_cell):
            cell_level = min(int(refined_cell[cell_id]), self.max_level)
            coarse_cell = self._coarse_cell_index(int(cell_id))
            for level in range(1, cell_level + 1):
                divisions = self.refinement_ratio**level
                stride = self.finest_ratio // divisions
                origin = divisions * coarse_cell
                level_gnum = self.coarse_cnum * divisions + 1
                ranges = [
                    range(
                        max(int(origin[d]) - halo, 0),
                        min(
                            int(origin[d]) + divisions + halo + 1,
                            int(level_gnum[d]),
                        ),
                    )
                    for d in range(self.dimension)
                ]
                for level_index in product(*ranges):
                    finest_index = stride * np.asarray(level_index, dtype=np.int32)
                    if self.bridging_domain or any(value % self.finest_ratio != 0 for value in finest_index):
                        logical_ids.append(self._logical_id(finest_index))
        if logical_ids:
            if self.bridging_domain:
                self.ensure_fine_logical_ids(logical_ids)
            else:
                self.ensure_logical_ids(logical_ids)

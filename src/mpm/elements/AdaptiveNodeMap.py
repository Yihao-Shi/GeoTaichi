from itertools import product
from math import ceil
import numpy as np
import taichi as ti
from src.mpm.elements.AdaptiveTopologyKernel import AdaptiveTopology


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

        self._logical_to_compact = np.full(
            self.full_logical_node_count,
            -1,
            dtype=np.int32,
        )
        self._fine_logical_to_compact = None
        if self.bridging_domain:
            self._fine_logical_to_compact = np.full(
                self.full_logical_node_count,
                -1,
                dtype=np.int32,
            )
        self._compact_to_logical = np.full(self.capacity, -1, dtype=np.int32)
        self._next_compact_id = 0
        self._initialize_coarse_nodes()

        self.logical_to_compact_field = None
        self.fine_logical_to_compact_field = None
        self.compact_to_logical_field = None
        self.device = None
        self._device_snapshot_current = True
        self._host_refinement_input = None

    def _sync_host_snapshot(self):
        """Explicit inspection/setup only; numerical refinement keeps fields on device."""
        if self.device is not None and not self._device_snapshot_current:
            self._logical_to_compact = self.logical_to_compact_field.to_numpy()
            if self.fine_logical_to_compact_field is not None:
                self._fine_logical_to_compact = self.fine_logical_to_compact_field.to_numpy()
            self._compact_to_logical = self.compact_to_logical_field.to_numpy()
            self._next_compact_id = int(self.device.node_count[None])
            self._device_snapshot_current = True

    @property
    def logical_to_compact(self):
        self._sync_host_snapshot()
        return self._logical_to_compact

    @property
    def fine_logical_to_compact(self):
        self._sync_host_snapshot()
        return self._fine_logical_to_compact

    @property
    def compact_to_logical(self):
        self._sync_host_snapshot()
        return self._compact_to_logical

    @property
    def next_compact_id(self):
        return self._next_compact_id if self.device is None else int(self.device.node_count[None])

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
            self._logical_to_compact[logical_id] = coarse_id
            self._compact_to_logical[coarse_id] = logical_id
        self._next_compact_id = self.coarse_node_count

    def bind_fields(
        self,
        logical_to_compact_field,
        compact_to_logical_field,
        fine_logical_to_compact_field=None,
    ):
        self._sync_host_snapshot()
        self.logical_to_compact_field = logical_to_compact_field
        self.fine_logical_to_compact_field = fine_logical_to_compact_field
        self.compact_to_logical_field = compact_to_logical_field
        self.sync_fields()
        mapping = self.fine_logical_to_compact_field if self.bridging_domain else self.logical_to_compact_field
        self.device = AdaptiveTopology(
            self.coarse_cnum,
            self.max_level,
            self.capacity,
            mapping,
            self.compact_to_logical_field,
            self._next_compact_id,
            self.bridging_domain,
        )

    def sync_fields(self):
        self._sync_host_snapshot()
        if self.logical_to_compact_field is not None:
            self.logical_to_compact_field.from_numpy(self._logical_to_compact)
            if self.fine_logical_to_compact_field is not None:
                self.fine_logical_to_compact_field.from_numpy(self._fine_logical_to_compact)
            self.compact_to_logical_field.from_numpy(self._compact_to_logical)
            if self.device is not None:
                self.device.node_count[None] = self._next_compact_id
            self._device_snapshot_current = True

    def _ensure_ids(self, logical_ids, mapping):
        self._sync_host_snapshot()
        logical_ids = np.unique(np.asarray(logical_ids, dtype=np.int64))
        missing = logical_ids[mapping[logical_ids] < 0]
        assert self._next_compact_id + missing.size <= self.capacity, (
            f"AdaptiveGrid dense node capacity {self.capacity} exceeded: "
            f"{self._next_compact_id + missing.size} nodes are required. "
            "Increase AdaptiveGrid/RefineRatio."
        )
        for logical_id in missing:
            compact_id = self._next_compact_id
            mapping[logical_id] = compact_id
            self._compact_to_logical[compact_id] = logical_id
            self._next_compact_id += 1
        if missing.size > 0:
            self.sync_fields()
        return mapping[logical_ids]

    def ensure_logical_ids(self, logical_ids):
        self._sync_host_snapshot()
        return self._ensure_ids(logical_ids, self._logical_to_compact)

    def ensure_fine_logical_ids(self, logical_ids):
        self._sync_host_snapshot()
        if not self.bridging_domain:
            return self.ensure_logical_ids(logical_ids)
        return self._ensure_ids(logical_ids, self._fine_logical_to_compact)

    def ensure_refined_cells_device(self, refined_cell, shape_function):
        self.device.ensure_refined_cells(refined_cell, shape_function)
        self._device_snapshot_current = False

    def ensure_refined_cells(self, refined_cell, shape_function):
        # Array input is retained for setup and reference tests, never for runtime updates.
        if self.device is not None:
            if self._host_refinement_input is None:
                self._host_refinement_input = ti.field(ti.u8, shape=int(np.prod(self.coarse_cnum)))
            self._host_refinement_input.from_numpy(np.asarray(refined_cell, dtype=np.uint8).reshape(-1))
            self.ensure_refined_cells_device(self._host_refinement_input, shape_function)
            return
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

"""Planar point--edge culling on the shared dynamic linked-cell/BVH backends."""

import taichi as ti

from src.fem.contact.LinkedCellBroadPhase import _grown_capacity
from src.physics_model.contact_model.ipc.ContactDistance import (
    point_edge_distance2,
)
from src.utils.PrefixSum import PrefixSumExecutor


@ti.data_oriented
class FEMPMPlanarCulling:
    """Compact MPM-point/FEM-edge candidates from shared edge AABBs.

    MPM samples are represented only in the broad phase as zero-length edges.
    The compacted narrow-phase stencil is the true three-site ``(p,e0,e1)``
    primitive, so distance derivatives and CCD never evaluate a degenerate
    edge--edge formula.
    """

    def __init__(self, broad_phase, sample_count, max_point_edge_pairs=None):
        self.broad_phase = broad_phase
        self.sample_count = int(sample_count)
        self.real_type = broad_phase.real_type
        self.point_edge_count = ti.field(dtype=ti.i32, shape=())
        self.raw_count = ti.field(dtype=ti.i32, shape=())
        self.fixed_capacity = max_point_edge_pairs is not None
        self.capacity = max(int(max_point_edge_pairs or 1), 1)
        self.point_edge = ti.Vector.field(3, dtype=ti.i32, shape=self.capacity)
        self.prefix = ti.field(dtype=ti.i32, shape=self.capacity + 1)
        self.scan = PrefixSumExecutor(self.capacity + 1)

    def _resize(self, raw_count):
        if self.fixed_capacity and raw_count > self.capacity:
            raise RuntimeError(
                "point-edge culling capacity is too small: " f"need {raw_count}, allocated {self.capacity}"
            )
        capacity = _grown_capacity(raw_count, self.capacity)
        if capacity == self.capacity:
            return
        self.capacity = capacity
        self.point_edge = ti.Vector.field(3, dtype=ti.i32, shape=capacity)
        self.prefix = ti.field(dtype=ti.i32, shape=capacity + 1)
        self.scan = PrefixSumExecutor(capacity + 1)

    @ti.func
    def _decode(self, stencil):
        point = -1
        endpoint0 = -1
        endpoint1 = -1
        if (
            stencil[0] == stencil[1]
            and stencil[0] < ti.static(self.sample_count)
            and stencil[2] >= ti.static(self.sample_count)
            and stencil[2] != stencil[3]
        ):
            point = stencil[0]
            endpoint0 = stencil[2]
            endpoint1 = stencil[3]
        elif (
            stencil[2] == stencil[3]
            and stencil[2] < ti.static(self.sample_count)
            and stencil[0] >= ti.static(self.sample_count)
            and stencil[0] != stencil[1]
        ):
            point = stencil[2]
            endpoint0 = stencil[0]
            endpoint1 = stencil[1]
        return point, endpoint0, endpoint1

    @ti.kernel
    def _mark(
        self,
        positions: ti.template(),
        candidates: ti.template(),
        raw_count: ti.i32,
        active_distance2: ti.f64,
        exact: ti.i32,
    ):
        self.prefix[0] = 0
        for contact_id in range(raw_count):
            point, endpoint0, endpoint1 = self._decode(candidates[contact_id])
            valid = point >= 0
            if valid and exact != 0:
                distance2 = point_edge_distance2(
                    positions[point],
                    positions[endpoint0],
                    positions[endpoint1],
                )
                valid = distance2 < active_distance2
            self.prefix[contact_id + 1] = ti.cast(valid, ti.i32)

    @ti.kernel
    def _compact(self, candidates: ti.template(), raw_count: ti.i32):
        for contact_id in range(raw_count):
            if self.prefix[contact_id + 1] > self.prefix[contact_id]:
                point, endpoint0, endpoint1 = self._decode(candidates[contact_id])
                self.point_edge[self.prefix[contact_id]] = ti.Vector([point, endpoint0, endpoint1])

    @ti.kernel
    def _set_counts(self, raw_count: ti.i32):
        self.raw_count[None] = raw_count
        self.point_edge_count[None] = self.prefix[raw_count]

    def rebuild_proximity(self, positions, active_distance):
        _, raw_count = self.broad_phase.rebuild(positions, float(active_distance))
        self._resize(raw_count)
        self._mark(
            positions,
            self.broad_phase.edge_edge,
            raw_count,
            float(active_distance) ** 2,
            1,
        )
        self.scan.run(self.prefix)
        self._compact(self.broad_phase.edge_edge, raw_count)
        self._set_counts(raw_count)
        return int(self.point_edge_count[None]), 0

    def rebuild_swept_candidates(self, positions, end_positions, search_distance):
        _, raw_count = self.broad_phase.rebuild(
            positions,
            float(search_distance),
            end_positions=end_positions,
        )
        self._resize(raw_count)
        self._mark(
            positions,
            self.broad_phase.edge_edge,
            raw_count,
            float(search_distance) ** 2,
            0,
        )
        self.scan.run(self.prefix)
        self._compact(self.broad_phase.edge_edge, raw_count)
        self._set_counts(raw_count)
        return int(self.point_edge_count[None]), 0


__all__ = ["FEMPMPlanarCulling"]

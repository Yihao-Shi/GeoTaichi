"""Device collision-culling pipeline for deforming FEM contact surfaces."""

import numpy as np
import taichi as ti

from src.contact_detection.continuous_contact_detection.CCD import (
    edge_edge_ccd,
    point_triangle_ccd,
)
from src.contact_detection.continuous_contact_detection.AdditiveCCD import (
    edge_edge_accd,
    point_triangle_accd,
)
from src.fem.contact.LinkedCellBroadPhase import _grown_capacity
from src.fem.contact.ExplicitGeometry import signed_point_triangle_gap
from src.physics_model.contact_model.ipc.ContactDistance import (
    edge_edge_distance2,
    point_triangle_distance2,
)
from src.utils.PrefixSum import PrefixSumExecutor


@ti.data_oriented
class FEMCollisionCulling:
    """Compact proximity and swept-CCD stencils over a spatial backend.

    The selected linked-cell or BVH object only produces conservative AABB
    overlaps.  This layer applies stitch/topology exclusions, exact proximity
    tests or conservative CCD, and prefix-sum compaction without moving
    primitive or candidate arrays through the host.
    """

    def __init__(
        self,
        broad_phase,
        node_count,
        node_body_ids=None,
        body_pair=None,
        node_system_ids=None,
        cross_system_only=False,
        exclude_same_body=False,
        max_point_triangle_pairs=None,
        max_edge_edge_pairs=None,
        max_active_point_triangle_pairs=None,
        max_active_edge_edge_pairs=None,
        enable_ccd=True,
        signed_point_triangle=False,
        point_triangle_pseudonormals=None,
    ):
        self.broad_phase = broad_phase
        self.node_count = int(node_count)
        self.real_type = broad_phase.real_type
        self.reference_position = broad_phase.reference_position
        self.filter_body_pair = body_pair is not None
        self.body_pair = (-1, -1) if body_pair is None else tuple(sorted(map(int, body_pair)))
        self.cross_system_only = bool(cross_system_only)
        self.exclude_same_body = bool(exclude_same_body)
        self.enable_ccd = bool(enable_ccd)
        self.signed_point_triangle = bool(signed_point_triangle)
        if self.signed_point_triangle and point_triangle_pseudonormals is None:
            raise ValueError("signed point-triangle culling requires surface pseudonormals")
        self.point_triangle_pseudonormals = point_triangle_pseudonormals
        body_values = (
            np.zeros(self.node_count, dtype=np.int32)
            if node_body_ids is None
            else np.asarray(node_body_ids, dtype=np.int32)
        )
        if body_values.shape != (self.node_count,):
            raise ValueError("FEM contact node_body_ids must contain one ID per node")
        self.node_body = ti.field(dtype=ti.i32, shape=self.node_count)
        self.node_body.from_numpy(np.ascontiguousarray(body_values))
        system_values = (
            np.zeros(self.node_count, dtype=np.int32)
            if node_system_ids is None
            else np.asarray(node_system_ids, dtype=np.int32)
        )
        if system_values.shape != (self.node_count,):
            raise ValueError("FEM contact node_system_ids must contain one ID per node")
        self.node_system = ti.field(dtype=ti.i32, shape=self.node_count)
        self.node_system.from_numpy(np.ascontiguousarray(system_values))

        self.point_triangle_count = ti.field(dtype=ti.i32, shape=())
        self.edge_edge_count = ti.field(dtype=ti.i32, shape=())
        self.raw_point_triangle_count = ti.field(dtype=ti.i32, shape=())
        self.raw_edge_edge_count = ti.field(dtype=ti.i32, shape=())
        self.ccd_point_triangle_count = ti.field(dtype=ti.i32, shape=())
        self.ccd_edge_edge_count = ti.field(dtype=ti.i32, shape=())
        self.minimum_step = ti.field(dtype=self.real_type, shape=())

        active_pt_request = (
            max_active_point_triangle_pairs if max_active_point_triangle_pairs is not None else max_point_triangle_pairs
        )
        active_ee_request = (
            max_active_edge_edge_pairs if max_active_edge_edge_pairs is not None else max_edge_edge_pairs
        )
        self.fixed_pt_capacity = active_pt_request is not None
        self.fixed_ee_capacity = active_ee_request is not None
        self.active_pt_capacity = max(int(active_pt_request or 1), 1)
        self.active_ee_capacity = max(int(active_ee_request or 1), 1)
        self.point_triangle = ti.Vector.field(4, dtype=ti.i32, shape=self.active_pt_capacity)
        self.edge_edge = ti.Vector.field(4, dtype=ti.i32, shape=self.active_ee_capacity)
        self.point_triangle_measure = ti.field(dtype=self.real_type, shape=self.active_pt_capacity)
        self.edge_edge_measure = ti.field(dtype=self.real_type, shape=self.active_ee_capacity)
        self.filter_pt_capacity = max(int(getattr(broad_phase, "pt_capacity", 1)), 1)
        self.filter_ee_capacity = max(int(getattr(broad_phase, "ee_capacity", 1)), 1)
        self.pt_scan = PrefixSumExecutor(self.filter_pt_capacity + 1)
        self.ee_scan = PrefixSumExecutor(self.filter_ee_capacity + 1)
        self.pt_prefix = ti.field(dtype=ti.i32, shape=self.pt_scan.get_length())
        self.ee_prefix = ti.field(dtype=ti.i32, shape=self.ee_scan.get_length())

        # Explicit penalty contact never invokes continuous collision
        # detection.  Keep one-element placeholders so diagnostics retain a
        # uniform interface without reserving another pair of full-capacity
        # PT/EE candidate, TOI and prefix buffers.
        self.ccd_pt_capacity = self.active_pt_capacity if self.enable_ccd else 1
        self.ccd_ee_capacity = self.active_ee_capacity if self.enable_ccd else 1
        self.ccd_point_triangle = ti.Vector.field(4, dtype=ti.i32, shape=self.ccd_pt_capacity)
        self.ccd_edge_edge = ti.Vector.field(4, dtype=ti.i32, shape=self.ccd_ee_capacity)
        self.ccd_point_triangle_toi = ti.field(dtype=self.real_type, shape=self.ccd_pt_capacity)
        self.ccd_edge_edge_toi = ti.field(dtype=self.real_type, shape=self.ccd_ee_capacity)
        self.raw_point_triangle_toi = ti.field(dtype=self.real_type, shape=self.ccd_pt_capacity)
        self.raw_edge_edge_toi = ti.field(dtype=self.real_type, shape=self.ccd_ee_capacity)
        self.ccd_pt_scan = PrefixSumExecutor(self.ccd_pt_capacity + 1)
        self.ccd_ee_scan = PrefixSumExecutor(self.ccd_ee_capacity + 1)
        self.ccd_pt_prefix = ti.field(dtype=ti.i32, shape=self.ccd_pt_scan.get_length())
        self.ccd_ee_prefix = ti.field(dtype=ti.i32, shape=self.ccd_ee_scan.get_length())

        self.stitch_prefix = ti.field(dtype=ti.i32, shape=max(self.node_count + 1, 1))
        self.stitch_neighbor_capacity = 1
        self.stitch_neighbors = ti.field(dtype=ti.i32, shape=1)
        self.stitch_prefix.fill(0)
        self.stitch_neighbors.fill(-1)
        self._set_empty_counts()

    def set_stitch_exclusions(self, stitches):
        """Exclude a stitched point from primitives containing its partners."""
        stencils = np.asarray(stitches, dtype=np.int32)
        if stencils.size == 0:
            self.stitch_prefix.fill(0)
            self.stitch_neighbors.fill(-1)
            return
        stencils = stencils.reshape(-1, 3)
        if np.any(stencils < 0) or np.any(stencils >= self.node_count):
            raise ValueError("stitch exclusion contains an out-of-range node")
        relations = set()
        for point, edge0, edge1 in stencils:
            for partner in (edge0, edge1):
                if int(point) != int(partner):
                    relations.add((int(point), int(partner)))
                    relations.add((int(partner), int(point)))
        ordered = sorted(relations)
        prefix = np.zeros(self.node_count + 1, dtype=np.int32)
        for first, _ in ordered:
            prefix[first + 1] += 1
        np.cumsum(prefix, out=prefix)
        required = max(len(ordered), 1)
        capacity = _grown_capacity(required, self.stitch_neighbor_capacity)
        if capacity != self.stitch_neighbor_capacity:
            self.stitch_neighbor_capacity = capacity
            self.stitch_neighbors = ti.field(dtype=ti.i32, shape=capacity)
        neighbors = np.full(self.stitch_neighbor_capacity, -1, dtype=np.int32)
        if ordered:
            neighbors[: len(ordered)] = np.asarray(ordered, dtype=np.int32)[:, 1]
        self.stitch_prefix.from_numpy(prefix)
        self.stitch_neighbors.from_numpy(neighbors)

    def _resize_filter_scratch(self, raw_pt_count, raw_ee_count):
        required = max(int(raw_pt_count), 1)
        capacity = _grown_capacity(required, self.filter_pt_capacity)
        if capacity != self.filter_pt_capacity:
            self.filter_pt_capacity = capacity
            self.pt_scan = PrefixSumExecutor(capacity + 1)
            self.pt_prefix = ti.field(dtype=ti.i32, shape=self.pt_scan.get_length())
        required = max(int(raw_ee_count), 1)
        capacity = _grown_capacity(required, self.filter_ee_capacity)
        if capacity != self.filter_ee_capacity:
            self.filter_ee_capacity = capacity
            self.ee_scan = PrefixSumExecutor(capacity + 1)
            self.ee_prefix = ti.field(dtype=ti.i32, shape=self.ee_scan.get_length())

    def _resize_active(self, active_pt_count, active_ee_count):
        if self.fixed_pt_capacity and active_pt_count > self.active_pt_capacity:
            raise RuntimeError(
                "point-triangle culling capacity is too small: "
                f"need {active_pt_count}, allocated {self.active_pt_capacity}"
            )
        if self.fixed_ee_capacity and active_ee_count > self.active_ee_capacity:
            raise RuntimeError(
                "edge-edge culling capacity is too small: "
                f"need {active_ee_count}, allocated {self.active_ee_capacity}"
            )
        required = max(int(active_pt_count), 1)
        capacity = _grown_capacity(required, self.active_pt_capacity)
        if capacity != self.active_pt_capacity:
            self.active_pt_capacity = capacity
            self.point_triangle = ti.Vector.field(4, dtype=ti.i32, shape=capacity)
            self.point_triangle_measure = ti.field(dtype=self.real_type, shape=capacity)

        required = max(int(active_ee_count), 1)
        capacity = _grown_capacity(required, self.active_ee_capacity)
        if capacity != self.active_ee_capacity:
            self.active_ee_capacity = capacity
            self.edge_edge = ti.Vector.field(4, dtype=ti.i32, shape=capacity)
            self.edge_edge_measure = ti.field(dtype=self.real_type, shape=capacity)

    def _resize_ccd(self, raw_pt_count, raw_ee_count):
        if not self.enable_ccd:
            raise RuntimeError("CCD storage is disabled for this explicit contact pipeline")
        if self.fixed_pt_capacity and raw_pt_count > self.ccd_pt_capacity:
            raise RuntimeError(
                "point-triangle CCD capacity is too small: " f"need {raw_pt_count}, allocated {self.ccd_pt_capacity}"
            )
        if self.fixed_ee_capacity and raw_ee_count > self.ccd_ee_capacity:
            raise RuntimeError(
                "edge-edge CCD capacity is too small: " f"need {raw_ee_count}, allocated {self.ccd_ee_capacity}"
            )
        required = max(int(raw_pt_count), 1)
        capacity = _grown_capacity(required, self.ccd_pt_capacity)
        if capacity != self.ccd_pt_capacity:
            self.ccd_pt_capacity = capacity
            self.ccd_point_triangle = ti.Vector.field(4, dtype=ti.i32, shape=capacity)
            self.ccd_point_triangle_toi = ti.field(dtype=self.real_type, shape=capacity)
            self.raw_point_triangle_toi = ti.field(dtype=self.real_type, shape=capacity)
            self.ccd_pt_scan = PrefixSumExecutor(capacity + 1)
            self.ccd_pt_prefix = ti.field(dtype=ti.i32, shape=self.ccd_pt_scan.get_length())

        required = max(int(raw_ee_count), 1)
        capacity = _grown_capacity(required, self.ccd_ee_capacity)
        if capacity != self.ccd_ee_capacity:
            self.ccd_ee_capacity = capacity
            self.ccd_edge_edge = ti.Vector.field(4, dtype=ti.i32, shape=capacity)
            self.ccd_edge_edge_toi = ti.field(dtype=self.real_type, shape=capacity)
            self.raw_edge_edge_toi = ti.field(dtype=self.real_type, shape=capacity)
            self.ccd_ee_scan = PrefixSumExecutor(capacity + 1)
            self.ccd_ee_prefix = ti.field(dtype=ti.i32, shape=self.ccd_ee_scan.get_length())

    @ti.func
    def _stitched(self, first, second, stitch_prefix, stitch_neighbors):
        connected = False
        for slot in range(stitch_prefix[first], stitch_prefix[first + 1]):
            connected = connected or stitch_neighbors[slot] == second
        return connected

    @ti.func
    def _exclude_point_triangle(self, stencil, stitch_prefix, stitch_neighbors):
        return (
            self._stitched(stencil[0], stencil[1], stitch_prefix, stitch_neighbors)
            or self._stitched(stencil[0], stencil[2], stitch_prefix, stitch_neighbors)
            or self._stitched(stencil[0], stencil[3], stitch_prefix, stitch_neighbors)
        )

    @ti.func
    def _exclude_edge_edge(self, stencil, stitch_prefix, stitch_neighbors):
        return (
            self._stitched(stencil[0], stencil[2], stitch_prefix, stitch_neighbors)
            or self._stitched(stencil[0], stencil[3], stitch_prefix, stitch_neighbors)
            or self._stitched(stencil[1], stencil[2], stitch_prefix, stitch_neighbors)
            or self._stitched(stencil[1], stencil[3], stitch_prefix, stitch_neighbors)
        )

    @ti.func
    def _allowed_body_pair(self, stencil):
        allowed = True
        if ti.static(self.cross_system_only):
            allowed = self.node_system[stencil[0]] != self.node_system[stencil[2]]
        if ti.static(self.filter_body_pair):
            first = self.node_body[stencil[0]]
            second = self.node_body[stencil[2]]
            lower = ti.min(first, second)
            upper = ti.max(first, second)
            allowed = allowed and (lower == self.body_pair[0] and upper == self.body_pair[1])
        if ti.static(self.exclude_same_body):
            allowed = allowed and (self.node_body[stencil[0]] != self.node_body[stencil[2]])
        return allowed

    @ti.kernel
    def _mark_active_candidates(
        self,
        positions: ti.template(),
        raw_point_triangle: ti.template(),
        raw_edge_edge: ti.template(),
        pt_prefix: ti.template(),
        ee_prefix: ti.template(),
        raw_pt_count: ti.i32,
        raw_ee_count: ti.i32,
        active_distance: ti.f64,
        edge_active_distance2: ti.f64,
        exact_distance: ti.i32,
        stitch_prefix: ti.template(),
        stitch_neighbors: ti.template(),
    ):
        pt_prefix[0] = 0
        ee_prefix[0] = 0
        for contact_id in range(raw_pt_count):
            stencil = raw_point_triangle[contact_id]
            valid = self._allowed_body_pair(stencil) and not self._exclude_point_triangle(
                stencil, stitch_prefix, stitch_neighbors
            )
            if exact_distance != 0 and valid:
                if ti.static(self.signed_point_triangle):
                    gap = signed_point_triangle_gap(
                        positions[stencil[0]],
                        positions[stencil[1]],
                        positions[stencil[2]],
                        positions[stencil[3]],
                        stencil,
                        self.point_triangle_pseudonormals,
                    )
                    valid = gap < active_distance
                else:
                    distance2 = point_triangle_distance2(
                        positions[stencil[0]],
                        positions[stencil[1]],
                        positions[stencil[2]],
                        positions[stencil[3]],
                    )
                    valid = distance2 < active_distance * active_distance
            pt_prefix[contact_id + 1] = ti.cast(valid, ti.i32)
        for contact_id in range(raw_ee_count):
            stencil = raw_edge_edge[contact_id]
            valid = self._allowed_body_pair(stencil) and not self._exclude_edge_edge(
                stencil, stitch_prefix, stitch_neighbors
            )
            if exact_distance != 0 and valid:
                distance2 = edge_edge_distance2(
                    positions[stencil[0]],
                    positions[stencil[1]],
                    positions[stencil[2]],
                    positions[stencil[3]],
                )
                valid = distance2 < edge_active_distance2
            ee_prefix[contact_id + 1] = ti.cast(valid, ti.i32)

    @ti.kernel
    def _compact_active_candidates(
        self,
        raw_point_triangle: ti.template(),
        raw_point_triangle_measure: ti.template(),
        raw_edge_edge: ti.template(),
        raw_edge_edge_measure: ti.template(),
        pt_prefix: ti.template(),
        ee_prefix: ti.template(),
        point_triangle: ti.template(),
        point_triangle_measure: ti.template(),
        edge_edge: ti.template(),
        edge_edge_measure: ti.template(),
        raw_pt_count: ti.i32,
        raw_ee_count: ti.i32,
    ):
        for contact_id in range(raw_pt_count):
            if pt_prefix[contact_id + 1] > pt_prefix[contact_id]:
                output = pt_prefix[contact_id]
                point_triangle[output] = raw_point_triangle[contact_id]
                point_triangle_measure[output] = raw_point_triangle_measure[contact_id]
        for contact_id in range(raw_ee_count):
            if ee_prefix[contact_id + 1] > ee_prefix[contact_id]:
                output = ee_prefix[contact_id]
                edge_edge[output] = raw_edge_edge[contact_id]
                edge_edge_measure[output] = raw_edge_edge_measure[contact_id]

    @ti.kernel
    def _set_active_counts(
        self,
        pt_prefix: ti.template(),
        ee_prefix: ti.template(),
        raw_pt_count: ti.i32,
        raw_ee_count: ti.i32,
    ):
        self.raw_point_triangle_count[None] = raw_pt_count
        self.raw_edge_edge_count[None] = raw_ee_count
        self.point_triangle_count[None] = pt_prefix[raw_pt_count]
        self.edge_edge_count[None] = ee_prefix[raw_ee_count]

    def _rebuild_active(
        self,
        positions,
        search_distance,
        *,
        end_positions=None,
        edge_search_distance=None,
        exact_distance,
    ):
        edge_search_distance = float(search_distance) if edge_search_distance is None else float(edge_search_distance)
        if getattr(
            self.broad_phase,
            "supports_separate_proximity_radius",
            False,
        ):
            raw_pt_count, raw_ee_count = self.broad_phase.rebuild(
                positions,
                float(search_distance),
                end_positions=end_positions,
                edge_radius=edge_search_distance,
            )
        else:
            raw_pt_count, raw_ee_count = self.broad_phase.rebuild(
                positions,
                float(search_distance),
                end_positions=end_positions,
            )
        broad_phase_exact = bool(
            exact_distance and end_positions is None and getattr(self.broad_phase, "exact_proximity", False)
        )
        self._resize_filter_scratch(raw_pt_count, raw_ee_count)
        self._mark_active_candidates(
            positions,
            self.broad_phase.point_triangle,
            self.broad_phase.edge_edge,
            self.pt_prefix,
            self.ee_prefix,
            raw_pt_count,
            raw_ee_count,
            float(search_distance),
            edge_search_distance**2,
            int(exact_distance and not broad_phase_exact),
            self.stitch_prefix,
            self.stitch_neighbors,
        )
        self.pt_scan.run(self.pt_prefix)
        self.ee_scan.run(self.ee_prefix)
        active_pt_count = int(self.pt_prefix[raw_pt_count])
        active_ee_count = int(self.ee_prefix[raw_ee_count])
        self._resize_active(active_pt_count, active_ee_count)
        self._compact_active_candidates(
            self.broad_phase.point_triangle,
            self.broad_phase.point_triangle_measure,
            self.broad_phase.edge_edge,
            self.broad_phase.edge_edge_measure,
            self.pt_prefix,
            self.ee_prefix,
            self.point_triangle,
            self.point_triangle_measure,
            self.edge_edge,
            self.edge_edge_measure,
            raw_pt_count,
            raw_ee_count,
        )
        self._set_active_counts(
            self.pt_prefix,
            self.ee_prefix,
            raw_pt_count,
            raw_ee_count,
        )
        return (
            int(self.point_triangle_count[None]),
            int(self.edge_edge_count[None]),
        )

    def rebuild_proximity(
        self,
        positions,
        active_distance,
        edge_active_distance=None,
    ):
        return self._rebuild_active(
            positions,
            active_distance,
            edge_search_distance=edge_active_distance,
            exact_distance=True,
        )

    def rebuild_swept_candidates(self, positions, end_positions, search_distance):
        return self._rebuild_active(
            positions,
            search_distance,
            end_positions=end_positions,
            exact_distance=False,
        )

    @ti.kernel
    def _mark_ccd_candidates(
        self,
        positions: ti.template(),
        end_positions: ti.template(),
        raw_point_triangle: ti.template(),
        raw_edge_edge: ti.template(),
        raw_point_triangle_toi: ti.template(),
        raw_edge_edge_toi: ti.template(),
        ccd_pt_prefix: ti.template(),
        ccd_ee_prefix: ti.template(),
        raw_pt_count: ti.i32,
        raw_ee_count: ti.i32,
        eta: ti.f64,
        thickness: ti.f64,
        max_iterations: ti.i32,
        stitch_prefix: ti.template(),
        stitch_neighbors: ti.template(),
    ):
        ccd_pt_prefix[0] = 0
        ccd_ee_prefix[0] = 0
        self.minimum_step[None] = 1.0
        for contact_id in range(raw_pt_count):
            stencil = raw_point_triangle[contact_id]
            valid = self._allowed_body_pair(stencil) and not self._exclude_point_triangle(
                stencil, stitch_prefix, stitch_neighbors
            )
            toi = 1.0
            if valid:
                if thickness > 0.0:
                    toi = point_triangle_accd(
                        positions[stencil[0]],
                        positions[stencil[1]],
                        positions[stencil[2]],
                        positions[stencil[3]],
                        end_positions[stencil[0]] - positions[stencil[0]],
                        end_positions[stencil[1]] - positions[stencil[1]],
                        end_positions[stencil[2]] - positions[stencil[2]],
                        end_positions[stencil[3]] - positions[stencil[3]],
                        eta,
                        thickness,
                        max_iterations,
                    )
                else:
                    toi = point_triangle_ccd(
                        positions[stencil[0]],
                        positions[stencil[1]],
                        positions[stencil[2]],
                        positions[stencil[3]],
                        end_positions[stencil[0]] - positions[stencil[0]],
                        end_positions[stencil[1]] - positions[stencil[1]],
                        end_positions[stencil[2]] - positions[stencil[2]],
                        end_positions[stencil[3]] - positions[stencil[3]],
                        eta,
                        max_iterations,
                    )
                ti.atomic_min(self.minimum_step[None], toi)
            raw_point_triangle_toi[contact_id] = toi
            ccd_pt_prefix[contact_id + 1] = ti.cast(valid and toi < 1.0, ti.i32)
        for contact_id in range(raw_ee_count):
            stencil = raw_edge_edge[contact_id]
            valid = self._allowed_body_pair(stencil) and not self._exclude_edge_edge(
                stencil, stitch_prefix, stitch_neighbors
            )
            toi = 1.0
            if valid:
                if thickness > 0.0:
                    toi = edge_edge_accd(
                        positions[stencil[0]],
                        positions[stencil[1]],
                        positions[stencil[2]],
                        positions[stencil[3]],
                        end_positions[stencil[0]] - positions[stencil[0]],
                        end_positions[stencil[1]] - positions[stencil[1]],
                        end_positions[stencil[2]] - positions[stencil[2]],
                        end_positions[stencil[3]] - positions[stencil[3]],
                        eta,
                        thickness,
                        max_iterations,
                    )
                else:
                    toi = edge_edge_ccd(
                        positions[stencil[0]],
                        positions[stencil[1]],
                        positions[stencil[2]],
                        positions[stencil[3]],
                        end_positions[stencil[0]] - positions[stencil[0]],
                        end_positions[stencil[1]] - positions[stencil[1]],
                        end_positions[stencil[2]] - positions[stencil[2]],
                        end_positions[stencil[3]] - positions[stencil[3]],
                        eta,
                        max_iterations,
                    )
                ti.atomic_min(self.minimum_step[None], toi)
            raw_edge_edge_toi[contact_id] = toi
            ccd_ee_prefix[contact_id + 1] = ti.cast(valid and toi < 1.0, ti.i32)

    @ti.kernel
    def _compact_ccd_candidates(
        self,
        raw_point_triangle: ti.template(),
        raw_edge_edge: ti.template(),
        raw_point_triangle_toi: ti.template(),
        raw_edge_edge_toi: ti.template(),
        ccd_pt_prefix: ti.template(),
        ccd_ee_prefix: ti.template(),
        ccd_point_triangle: ti.template(),
        ccd_edge_edge: ti.template(),
        ccd_point_triangle_toi: ti.template(),
        ccd_edge_edge_toi: ti.template(),
        raw_pt_count: ti.i32,
        raw_ee_count: ti.i32,
    ):
        for contact_id in range(raw_pt_count):
            if ccd_pt_prefix[contact_id + 1] > ccd_pt_prefix[contact_id]:
                output = ccd_pt_prefix[contact_id]
                ccd_point_triangle[output] = raw_point_triangle[contact_id]
                ccd_point_triangle_toi[output] = raw_point_triangle_toi[contact_id]
        for contact_id in range(raw_ee_count):
            if ccd_ee_prefix[contact_id + 1] > ccd_ee_prefix[contact_id]:
                output = ccd_ee_prefix[contact_id]
                ccd_edge_edge[output] = raw_edge_edge[contact_id]
                ccd_edge_edge_toi[output] = raw_edge_edge_toi[contact_id]

    @ti.kernel
    def _set_ccd_counts(
        self,
        ccd_pt_prefix: ti.template(),
        ccd_ee_prefix: ti.template(),
        raw_pt_count: ti.i32,
        raw_ee_count: ti.i32,
    ):
        self.raw_point_triangle_count[None] = raw_pt_count
        self.raw_edge_edge_count[None] = raw_ee_count
        self.ccd_point_triangle_count[None] = ccd_pt_prefix[raw_pt_count]
        self.ccd_edge_edge_count[None] = ccd_ee_prefix[raw_ee_count]

    def compute_ccd(
        self,
        positions,
        end_positions,
        search_distance,
        *,
        eta,
        thickness,
        max_iterations,
    ):
        if not self.enable_ccd:
            raise RuntimeError("CCD storage is disabled for this explicit contact pipeline")
        raw_pt_count, raw_ee_count = self.broad_phase.rebuild(
            positions,
            float(search_distance),
            end_positions=end_positions,
        )
        self._resize_ccd(raw_pt_count, raw_ee_count)
        self._mark_ccd_candidates(
            positions,
            end_positions,
            self.broad_phase.point_triangle,
            self.broad_phase.edge_edge,
            self.raw_point_triangle_toi,
            self.raw_edge_edge_toi,
            self.ccd_pt_prefix,
            self.ccd_ee_prefix,
            raw_pt_count,
            raw_ee_count,
            float(eta),
            float(thickness),
            int(max_iterations),
            self.stitch_prefix,
            self.stitch_neighbors,
        )
        self.ccd_pt_scan.run(self.ccd_pt_prefix)
        self.ccd_ee_scan.run(self.ccd_ee_prefix)
        self._compact_ccd_candidates(
            self.broad_phase.point_triangle,
            self.broad_phase.edge_edge,
            self.raw_point_triangle_toi,
            self.raw_edge_edge_toi,
            self.ccd_pt_prefix,
            self.ccd_ee_prefix,
            self.ccd_point_triangle,
            self.ccd_edge_edge,
            self.ccd_point_triangle_toi,
            self.ccd_edge_edge_toi,
            raw_pt_count,
            raw_ee_count,
        )
        self._set_ccd_counts(
            self.ccd_pt_prefix,
            self.ccd_ee_prefix,
            raw_pt_count,
            raw_ee_count,
        )
        return (
            int(self.ccd_point_triangle_count[None]),
            int(self.ccd_edge_edge_count[None]),
            float(self.minimum_step[None]),
        )

    @ti.kernel
    def _set_empty_counts(self):
        self.point_triangle_count[None] = 0
        self.edge_edge_count[None] = 0
        self.raw_point_triangle_count[None] = 0
        self.raw_edge_edge_count[None] = 0
        self.ccd_point_triangle_count[None] = 0
        self.ccd_edge_edge_count[None] = 0
        self.minimum_step[None] = 1.0

    @ti.kernel
    def _clear_active(self):
        self.point_triangle_count[None] = 0
        self.edge_edge_count[None] = 0

    def clear_active(self):
        self._clear_active()

    def diagnostics(self):
        return {
            "raw_pt_candidates": int(self.raw_point_triangle_count[None]),
            "raw_ee_candidates": int(self.raw_edge_edge_count[None]),
            "pt_candidates": int(self.point_triangle_count[None]),
            "ee_candidates": int(self.edge_edge_count[None]),
            "ccd_pt_candidates": int(self.ccd_point_triangle_count[None]),
            "ccd_ee_candidates": int(self.ccd_edge_edge_count[None]),
        }


__all__ = ["FEMCollisionCulling"]

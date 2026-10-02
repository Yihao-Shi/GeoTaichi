import numpy as np
import taichi as ti

from src.contact_detection.bounding_volume_hierarchy.AABB import AABB
from src.contact_detection.bounding_volume_hierarchy.MultiPatchLBVH import LBVH
from src.dem.neighbor.BoundingVolumeHierarchy import BoundingVolumeHierarchy
from src.dem.neighbor.HierarchicalLinkedCell import HierarchicalLinkedCell
from src.dem.neighbor.LinkedCell import LinkedCell
from src.dem.neighbor.neighbor_kernel import (
    board_search_affine_edge_edge_bvh_,
    board_search_affine_edge_edge_linked_cell_,
    board_search_affine_edge_edge_linked_cell_hierarchical_,
    board_search_affine_vertex_face_bvh_,
    board_search_affine_vertex_face_linked_cell_,
    board_search_affine_vertex_face_linked_cell_hierarchical_,
    initialize_affine_edge_bounds_,
    initialize_affine_edge_levels_,
    initialize_affine_face_bounds_,
    initialize_affine_face_levels_,
    initialize_affine_vertex_bounds_,
    initialize_grid_information,
    insert_affine_edges_to_cell_,
    insert_affine_edges_to_cell_hierarchical_,
    insert_affine_faces_to_cell_,
    insert_affine_faces_to_cell_hierarchical_,
    set_affine_bvh_aabbs_,
)
from src.dem.structs.BaseStruct import HierarchicalCell
from src.utils.TypeDefination import vec3i


def _coordination_value(value, default=16):
    if isinstance(value, (list, tuple, np.ndarray)):
        return max(int(v) for v in value) if len(value) > 0 else int(default)
    return int(value) if value is not None else int(default)


def _as_domain(domain):
    return np.maximum(np.asarray(domain, dtype=np.float64).reshape(3), 1.0e-6)


class AffineBodyNeighborBase(object):
    def __init__(self, sims, vertex_num, face_num, edge_num, dhat):
        self.sims = sims
        self.vertex_num = int(vertex_num)
        self.face_num = int(face_num)
        self.edge_num = int(edge_num)
        self.dhat = float(dhat)
        self.last_candidate_pairs = 0
        self.last_vertex_face_candidate_pairs = 0
        self.last_edge_edge_candidate_pairs = 0
        self.last_cell_overflow = 0
        self.last_candidate_overflow = 0
        self.last_edge_candidate_overflow = 0
        self.last_mode = str(sims.search)

        self.vertex_min = ti.Vector.field(3, dtype=float, shape=max(self.vertex_num, 1))
        self.vertex_max = ti.Vector.field(3, dtype=float, shape=max(self.vertex_num, 1))
        self.face_min = ti.Vector.field(3, dtype=float, shape=max(self.face_num, 1))
        self.face_max = ti.Vector.field(3, dtype=float, shape=max(self.face_num, 1))
        self.face_center = ti.Vector.field(3, dtype=float, shape=max(self.face_num, 1))
        self.face_radius = ti.field(dtype=float, shape=max(self.face_num, 1))
        self.edge_min = ti.Vector.field(3, dtype=float, shape=max(self.edge_num, 1))
        self.edge_max = ti.Vector.field(3, dtype=float, shape=max(self.edge_num, 1))
        self.edge_center = ti.Vector.field(3, dtype=float, shape=max(self.edge_num, 1))
        self.edge_radius = ti.field(dtype=float, shape=max(self.edge_num, 1))

        configured_pt = int(getattr(sims, "max_point_triangle_pairs", 0))
        configured_ee = int(getattr(sims, "max_edge_edge_pairs", 0))
        self.fixed_candidate_capacity = configured_pt >= 0
        self.fixed_edge_candidate_capacity = configured_ee >= 0
        vf_capacity = max(configured_pt, 1)
        ee_capacity = max(configured_ee, 1)
        self._allocate_candidates(vf_capacity)
        self._allocate_edge_candidates(ee_capacity)

    def _allocate_candidates(self, capacity):
        self.candidate_capacity = int(max(capacity, 1))
        self.candidate_vertex = ti.field(dtype=ti.i32, shape=self.candidate_capacity)
        self.candidate_face = ti.field(dtype=ti.i32, shape=self.candidate_capacity)
        self.candidate_count = ti.field(dtype=ti.i32, shape=())
        self.candidate_overflow = ti.field(dtype=ti.i32, shape=())

    def _allocate_edge_candidates(self, capacity):
        self.edge_candidate_capacity = int(max(capacity, 1))
        self.candidate_edge0 = ti.field(dtype=ti.i32, shape=self.edge_candidate_capacity)
        self.candidate_edge1 = ti.field(dtype=ti.i32, shape=self.edge_candidate_capacity)
        self.edge_candidate_count = ti.field(dtype=ti.i32, shape=())
        self.edge_candidate_overflow = ti.field(dtype=ti.i32, shape=())

    def _prepare_bounds(self, x, dx, faces, edges, margin, swept):
        initialize_affine_vertex_bounds_(
            self.vertex_num, float(margin), bool(swept), x, dx, self.vertex_min, self.vertex_max
        )
        initialize_affine_face_bounds_(
            self.face_num,
            float(margin),
            bool(swept),
            x,
            dx,
            faces,
            self.face_min,
            self.face_max,
            self.face_center,
            self.face_radius,
        )
        initialize_affine_edge_bounds_(
            self.edge_num,
            float(margin),
            bool(swept),
            x,
            dx,
            edges,
            self.edge_min,
            self.edge_max,
            self.edge_center,
            self.edge_radius,
        )

    def _finish_update(self):
        self.last_vertex_face_candidate_pairs = int(self.candidate_count[None])
        self.last_edge_edge_candidate_pairs = int(self.edge_candidate_count[None])
        self.last_candidate_pairs = self.last_vertex_face_candidate_pairs + self.last_edge_edge_candidate_pairs
        self.last_candidate_overflow = int(self.candidate_overflow[None])
        self.last_edge_candidate_overflow = int(self.edge_candidate_overflow[None])
        return self.last_candidate_pairs

    def _grow_candidates(self, requested_count=None):
        requested = int(requested_count) if requested_count is not None else self.candidate_capacity * 2
        if self.fixed_candidate_capacity:
            raise RuntimeError(
                "AffineBody max_point_triangle_pairs is too small: "
                f"need {requested}, allocated {self.candidate_capacity}"
            )
        max_pairs = max(self.vertex_num * self.face_num, 1)
        new_capacity = min(max(self.candidate_capacity * 2, requested, 1), max_pairs)
        if new_capacity <= self.candidate_capacity and self.candidate_capacity < max_pairs:
            new_capacity = min(self.candidate_capacity + max(self.vertex_num, 1), max_pairs)
        if new_capacity <= self.candidate_capacity:
            raise RuntimeError(
                "Affine body neighbor candidate buffer overflowed even at the dense vertex-face capacity. "
                "The contact broadphase generated too many pairs."
            )
        self._allocate_candidates(new_capacity)

    def _grow_edge_candidates(self, requested_count=None):
        requested = int(requested_count) if requested_count is not None else self.edge_candidate_capacity * 2
        if self.fixed_edge_candidate_capacity:
            raise RuntimeError(
                "AffineBody max_edge_edge_pairs is too small: "
                f"need {requested}, allocated {self.edge_candidate_capacity}"
            )
        max_pairs = max(self.edge_num * max(self.edge_num - 1, 1) // 2, 1)
        new_capacity = min(max(self.edge_candidate_capacity * 2, requested, 1), max_pairs)
        if new_capacity <= self.edge_candidate_capacity and self.edge_candidate_capacity < max_pairs:
            new_capacity = min(self.edge_candidate_capacity + max(self.edge_num, 1), max_pairs)
        if new_capacity <= self.edge_candidate_capacity:
            raise RuntimeError(
                "Affine body neighbor edge-edge candidate buffer overflowed even at the dense edge-edge capacity. "
                "The contact broadphase generated too many pairs."
            )
        self._allocate_edge_candidates(new_capacity)

    def update(self, x, dx, faces, edges, node2body, face2body, edge2body, margin, swept=False):
        raise NotImplementedError


class AffineBodyLinkedCell(LinkedCell, AffineBodyNeighborBase):
    def __init__(self, sims, vertex_num, face_num, edge_num, dhat):
        AffineBodyNeighborBase.__init__(self, sims, vertex_num, face_num, edge_num, dhat)
        self.domain = _as_domain(sims.domain)
        self.grid_size = self._choose_grid_size()
        self.igrid_size = 1.0 / self.grid_size
        self.cnum = vec3i([max(int(self.domain[i] * self.igrid_size), 1) for i in range(3)])
        self.cell_sum = int(self.cnum[0] * self.cnum[1] * self.cnum[2])
        coord = _coordination_value(getattr(sims, "body_coordination_number", 16), 16)
        wall_per_cell = _coordination_value(getattr(sims, "wall_per_cell", 4), 4)
        self.item_per_cell = max(32, 2 * coord, wall_per_cell)
        self._allocate_cells(self.item_per_cell)

    def _choose_grid_size(self):
        grid_size = max(2.0 * self.dhat, min(self.domain) / 64.0, 1.0e-5)
        max_cells = max(4096, min(400000, max(64 * (self.vertex_num + self.face_num + self.edge_num), 4096)))
        while np.prod(np.maximum((self.domain * (1.0 / grid_size)).astype(np.int64), 1)) > max_cells:
            grid_size *= 1.25
        return float(grid_size)

    def _allocate_cells(self, item_per_cell):
        self.face_per_cell = int(max(item_per_cell, 1))
        self.edge_per_cell = self.face_per_cell
        self.cell_count = ti.field(dtype=ti.i32, shape=max(self.cell_sum, 1))
        self.cell_face = ti.field(dtype=ti.i32, shape=max(self.cell_sum * self.face_per_cell, 1))
        self.edge_cell_count = ti.field(dtype=ti.i32, shape=max(self.cell_sum, 1))
        self.cell_edge = ti.field(dtype=ti.i32, shape=max(self.cell_sum * self.edge_per_cell, 1))
        self.cell_overflow = ti.field(dtype=ti.i32, shape=())
        self.edge_cell_overflow = ti.field(dtype=ti.i32, shape=())

    def _grow_cells(self):
        self._allocate_cells(self.face_per_cell * 2)

    def update(self, x, dx, faces, edges, node2body, face2body, edge2body, margin, swept=False):
        self._prepare_bounds(x, dx, faces, edges, margin, swept)
        while True:
            insert_affine_faces_to_cell_(
                self.face_num,
                self.igrid_size,
                self.face_per_cell,
                self.face_min,
                self.face_max,
                self.cell_count,
                self.cell_face,
                self.cell_overflow,
                self.cnum,
            )
            insert_affine_edges_to_cell_(
                self.edge_num,
                self.igrid_size,
                self.edge_per_cell,
                self.edge_min,
                self.edge_max,
                self.edge_cell_count,
                self.cell_edge,
                self.edge_cell_overflow,
                self.cnum,
            )
            board_search_affine_vertex_face_linked_cell_(
                self.vertex_num,
                self.face_per_cell,
                self.candidate_capacity,
                self.igrid_size,
                self.cell_count,
                self.cell_face,
                self.vertex_min,
                self.vertex_max,
                self.face_min,
                self.face_max,
                node2body,
                face2body,
                self.candidate_vertex,
                self.candidate_face,
                self.candidate_count,
                self.candidate_overflow,
                self.cnum,
            )
            board_search_affine_edge_edge_linked_cell_(
                self.edge_num,
                self.edge_per_cell,
                self.edge_candidate_capacity,
                self.igrid_size,
                self.edge_cell_count,
                self.cell_edge,
                self.edge_min,
                self.edge_max,
                edge2body,
                self.candidate_edge0,
                self.candidate_edge1,
                self.edge_candidate_count,
                self.edge_candidate_overflow,
                self.cnum,
            )
            count = self._finish_update()
            self.last_cell_overflow = max(int(self.cell_overflow[None]), int(self.edge_cell_overflow[None]))
            if self.last_cell_overflow:
                self._grow_cells()
                continue
            if self.last_candidate_overflow:
                self._grow_candidates(self.last_vertex_face_candidate_pairs)
                continue
            if self.last_edge_candidate_overflow:
                self._grow_edge_candidates(self.last_edge_edge_candidate_pairs)
                continue
            return count


class AffineBodyHierarchicalLinkedCell(HierarchicalLinkedCell, AffineBodyNeighborBase):
    def __init__(self, sims, vertex_num, face_num, edge_num, dhat):
        AffineBodyNeighborBase.__init__(self, sims, vertex_num, face_num, edge_num, dhat)
        self.domain = _as_domain(sims.domain)
        self.levels = max(int(getattr(sims, "hierarchical_level", 1)), 1)
        sizes = list(getattr(sims, "hierarchical_size", []))
        if len(sizes) == 0:
            sizes = [max(self.dhat, min(self.domain) / 64.0)]
        sizes = sorted(float(v) for v in sizes)
        while len(sizes) < self.levels:
            sizes.append(sizes[-1] * 2.0)
        self.hierarchical_size = np.asarray(sizes[: self.levels], dtype=np.float64)
        self.face_level = ti.field(dtype=ti.i32, shape=max(self.face_num, 1))
        self.edge_level = ti.field(dtype=ti.i32, shape=max(self.edge_num, 1))
        self.grid = HierarchicalCell.field(shape=self.levels)
        self._initialize_grid_storage()

    def _wall_per_cell_values(self):
        base = getattr(self.sims, "wall_per_cell", 4)
        if isinstance(base, (list, tuple, np.ndarray)):
            values = [int(max(v, 1)) for v in base]
        else:
            values = [int(max(base, 1)) for _ in range(self.levels)]
        coord = _coordination_value(getattr(self.sims, "body_coordination_number", 16), 16)
        values = [max(v, 32, 2 * coord) for v in values]
        while len(values) < self.levels:
            values.append(values[-1])
        return values[: self.levels]

    def _initialize_grid_storage(self):
        gsize = []
        cnum = []
        csum = []
        factor = []
        for level in range(self.levels):
            grid_size = max(2.0 * (self.hierarchical_size[level] + self.dhat), 1.0e-5)
            igrid = 1.0 / grid_size
            cells = [max(int(self.domain[i] * igrid), 1) for i in range(3)]
            gsize.append(grid_size)
            cnum.append(cells)
            csum.append(int(cells[0] * cells[1] * cells[2]))
            factor.append(0.5 + self.hierarchical_size[level] / grid_size)
        self.gsize_np = np.asarray(gsize, dtype=np.float64)
        self.cnum_np = np.asarray(cnum, dtype=np.int32)
        self.csum_np = np.asarray(csum, dtype=np.int32)
        self.factor_np = np.asarray(factor, dtype=np.float64)
        self.cell_sum = int(sum(csum))
        wall_per_cell = self._wall_per_cell_values()
        self.wall_per_cell_np = np.asarray(wall_per_cell, dtype=np.int32)
        self.wall_in_cell = int(
            initialize_grid_information(
                self.levels,
                self.gsize_np,
                self.cnum_np,
                self.csum_np,
                self.factor_np,
                self.wall_per_cell_np,
                self.grid,
            )
        )
        self.cell_count = ti.field(dtype=ti.i32, shape=max(self.cell_sum, 1))
        self.cell_face = ti.field(dtype=ti.i32, shape=max(self.wall_in_cell, 1))
        self.edge_cell_count = ti.field(dtype=ti.i32, shape=max(self.cell_sum, 1))
        self.cell_edge = ti.field(dtype=ti.i32, shape=max(self.wall_in_cell, 1))
        self.cell_overflow = ti.field(dtype=ti.i32, shape=())
        self.edge_cell_overflow = ti.field(dtype=ti.i32, shape=())

    def _grow_cells(self):
        self.wall_per_cell_np *= 2
        self.wall_in_cell = int(
            initialize_grid_information(
                self.levels,
                self.gsize_np,
                self.cnum_np,
                self.csum_np,
                self.factor_np,
                self.wall_per_cell_np,
                self.grid,
            )
        )
        self.cell_face = ti.field(dtype=ti.i32, shape=max(self.wall_in_cell, 1))
        self.cell_edge = ti.field(dtype=ti.i32, shape=max(self.wall_in_cell, 1))

    def update(self, x, dx, faces, edges, node2body, face2body, edge2body, margin, swept=False):
        self._prepare_bounds(x, dx, faces, edges, margin, swept)
        initialize_affine_face_levels_(
            self.face_num, self.levels, self.hierarchical_size, self.face_radius, self.face_level
        )
        initialize_affine_edge_levels_(
            self.edge_num, self.levels, self.hierarchical_size, self.edge_radius, self.edge_level
        )
        while True:
            insert_affine_faces_to_cell_hierarchical_(
                self.face_num,
                self.face_min,
                self.face_max,
                self.face_level,
                self.cell_count,
                self.cell_face,
                self.cell_overflow,
                self.grid,
            )
            insert_affine_edges_to_cell_hierarchical_(
                self.edge_num,
                self.edge_min,
                self.edge_max,
                self.edge_level,
                self.edge_cell_count,
                self.cell_edge,
                self.edge_cell_overflow,
                self.grid,
            )
            board_search_affine_vertex_face_linked_cell_hierarchical_(
                self.vertex_num,
                self.levels,
                self.candidate_capacity,
                self.cell_count,
                self.cell_face,
                self.vertex_min,
                self.vertex_max,
                self.face_min,
                self.face_max,
                node2body,
                face2body,
                self.candidate_vertex,
                self.candidate_face,
                self.candidate_count,
                self.candidate_overflow,
                self.grid,
            )
            board_search_affine_edge_edge_linked_cell_hierarchical_(
                self.edge_num,
                self.levels,
                self.edge_candidate_capacity,
                self.edge_cell_count,
                self.cell_edge,
                self.edge_min,
                self.edge_max,
                edge2body,
                self.candidate_edge0,
                self.candidate_edge1,
                self.edge_candidate_count,
                self.edge_candidate_overflow,
                self.grid,
            )
            count = self._finish_update()
            self.last_cell_overflow = max(int(self.cell_overflow[None]), int(self.edge_cell_overflow[None]))
            if self.last_cell_overflow:
                self._grow_cells()
                continue
            if self.last_candidate_overflow:
                self._grow_candidates(self.last_vertex_face_candidate_pairs)
                continue
            if self.last_edge_candidate_overflow:
                self._grow_edge_candidates(self.last_edge_edge_candidate_pairs)
                continue
            return count


class AffineBodyBoundingVolumeHierarchy(BoundingVolumeHierarchy, AffineBodyNeighborBase):
    def __init__(self, sims, vertex_num, face_num, edge_num, dhat):
        AffineBodyNeighborBase.__init__(self, sims, vertex_num, face_num, edge_num, dhat)
        self.aabb = AABB([max(self.vertex_num, 1), max(self.face_num, 1), max(self.edge_num, 1)])
        self.lbvh = LBVH(
            n_aabbs=[max(self.vertex_num, 1), max(self.face_num, 1), max(self.edge_num, 1)],
            aabb=self.aabb,
            extended_morton=True,
        )
        self.lbvh.initialize(active_aabbs=[max(self.vertex_num, 1), max(self.face_num, 1), max(self.edge_num, 1)])
        self.vertex_batch_id = 0
        self.face_batch_id = 1
        self.edge_batch_id = 2

    def update(self, x, dx, faces, edges, node2body, face2body, edge2body, margin, swept=False):
        self._prepare_bounds(x, dx, faces, edges, margin, swept)
        vertex_prefix = int(self.aabb.prefix_batch_size[self.vertex_batch_id])
        face_prefix = int(self.aabb.prefix_batch_size[self.face_batch_id])
        edge_prefix = int(self.aabb.prefix_batch_size[self.edge_batch_id])
        set_affine_bvh_aabbs_(
            self.vertex_num,
            self.face_num,
            self.edge_num,
            vertex_prefix,
            face_prefix,
            edge_prefix,
            self.vertex_min,
            self.vertex_max,
            self.face_min,
            self.face_max,
            self.edge_min,
            self.edge_max,
            self.lbvh.aabb.aabbs,
        )
        self.lbvh.build(self.vertex_num + self.face_num + self.edge_num)
        while True:
            board_search_affine_vertex_face_bvh_(
                self.vertex_batch_id,
                self.face_batch_id,
                self.vertex_num,
                self.candidate_capacity,
                self.lbvh.prefix_batch_size,
                self.lbvh.nodes,
                self.lbvh.morton_codes,
                self.lbvh.primitive_ids,
                self.lbvh.extended_morton,
                self.lbvh.aabb.aabbs,
                node2body,
                face2body,
                self.candidate_vertex,
                self.candidate_face,
                self.candidate_count,
                self.candidate_overflow,
            )
            board_search_affine_edge_edge_bvh_(
                self.edge_batch_id,
                self.edge_num,
                self.edge_candidate_capacity,
                self.lbvh.prefix_batch_size,
                self.lbvh.nodes,
                self.lbvh.morton_codes,
                self.lbvh.primitive_ids,
                self.lbvh.extended_morton,
                self.lbvh.aabb.aabbs,
                edge2body,
                self.candidate_edge0,
                self.candidate_edge1,
                self.edge_candidate_count,
                self.edge_candidate_overflow,
            )
            count = self._finish_update()
            self.last_cell_overflow = 0
            if self.last_candidate_overflow:
                self._grow_candidates(self.last_vertex_face_candidate_pairs)
                continue
            if self.last_edge_candidate_overflow:
                self._grow_edge_candidates(self.last_edge_edge_candidate_pairs)
                continue
            return count


def make_affine_body_neighbor(sims, vertex_num, face_num, edge_num, dhat):
    search = str(getattr(sims, "search", "LinkedCell"))
    if search == "BVH":
        return AffineBodyBoundingVolumeHierarchy(sims, vertex_num, face_num, edge_num, dhat)
    if search == "HierarchicalLinkedCell":
        return AffineBodyHierarchicalLinkedCell(sims, vertex_num, face_num, edge_num, dhat)
    return AffineBodyLinkedCell(sims, vertex_num, face_num, edge_num, dhat)

from src.utils.errors import CapacityError

import taichi as ti
from src.dem.neighbor.NeighborKernel import (
    initialize_affine_edge_bounds,
    initialize_affine_face_bounds,
    initialize_affine_vertex_bounds,
)


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
        initialize_affine_vertex_bounds(
            self.vertex_num, float(margin), bool(swept), x, dx, self.vertex_min, self.vertex_max
        )
        initialize_affine_face_bounds(
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
        initialize_affine_edge_bounds(
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
            raise CapacityError(
                "AffineBody max_point_triangle_pairs is too small: "
                f"need {requested}, allocated {self.candidate_capacity}"
            )
        max_pairs = max(self.vertex_num * self.face_num, 1)
        new_capacity = min(max(self.candidate_capacity * 2, requested, 1), max_pairs)
        if new_capacity <= self.candidate_capacity and self.candidate_capacity < max_pairs:
            new_capacity = min(self.candidate_capacity + max(self.vertex_num, 1), max_pairs)
        if new_capacity <= self.candidate_capacity:
            raise CapacityError(
                "Affine body neighbor candidate buffer overflowed even at the dense vertex-face capacity. "
                "The contact broadphase generated too many pairs."
            )
        self._allocate_candidates(new_capacity)

    def _grow_edge_candidates(self, requested_count=None):
        requested = int(requested_count) if requested_count is not None else self.edge_candidate_capacity * 2
        if self.fixed_edge_candidate_capacity:
            raise CapacityError(
                "AffineBody max_edge_edge_pairs is too small: "
                f"need {requested}, allocated {self.edge_candidate_capacity}"
            )
        max_pairs = max(self.edge_num * max(self.edge_num - 1, 1) // 2, 1)
        new_capacity = min(max(self.edge_candidate_capacity * 2, requested, 1), max_pairs)
        if new_capacity <= self.edge_candidate_capacity and self.edge_candidate_capacity < max_pairs:
            new_capacity = min(self.edge_candidate_capacity + max(self.edge_num, 1), max_pairs)
        if new_capacity <= self.edge_candidate_capacity:
            raise CapacityError(
                "Affine body neighbor edge-edge candidate buffer overflowed even at the dense edge-edge capacity. "
                "The contact broadphase generated too many pairs."
            )
        self._allocate_edge_candidates(new_capacity)

    def update(self, x, dx, faces, edges, node2body, face2body, edge2body, margin, swept=False):
        raise NotImplementedError

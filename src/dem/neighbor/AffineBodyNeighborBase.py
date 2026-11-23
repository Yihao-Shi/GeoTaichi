from src.contact_detection.Capacity import contact_pair_capacity
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

        self._allocate_candidates(contact_pair_capacity(self.face_num, sims.point_triangle_coordination_number))
        self._allocate_edge_candidates(contact_pair_capacity(self.face_num, sims.edge_edge_coordination_number))

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

    def _require_candidate_capacity(self, count):
        raise CapacityError(
            f"AffineBody point–triangle pairs need {count}, allocated {self.candidate_capacity}. "
            "Increase point_triangle_coordination_number."
        )

    def _require_edge_candidate_capacity(self, count):
        raise CapacityError(
            f"AffineBody edge–edge pairs need {count}, allocated {self.edge_candidate_capacity}. "
            "Increase edge_edge_coordination_number."
        )

    def update(self, x, dx, faces, edges, node2body, face2body, edge2body, margin, swept=False):
        raise NotImplementedError

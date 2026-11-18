from src.dem.neighbor.Utilities import _as_domain
from src.dem.neighbor.Utilities import _coordination_value
from src.dem.neighbor.AffineBodyNeighborBase import AffineBodyNeighborBase

import numpy as np
import taichi as ti
from src.dem.neighbor.LinkedCell import LinkedCell
from src.dem.neighbor.NeighborKernel import (
    board_search_affine_edge_edge_linked_cell,
    board_search_affine_vertex_face_linked_cell,
    insert_affine_edges_to_cell,
    insert_affine_faces_to_cell,
)
from src.utils.TypeDefinition import vec3i


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
            insert_affine_faces_to_cell(
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
            insert_affine_edges_to_cell(
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
            board_search_affine_vertex_face_linked_cell(
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
            board_search_affine_edge_edge_linked_cell(
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

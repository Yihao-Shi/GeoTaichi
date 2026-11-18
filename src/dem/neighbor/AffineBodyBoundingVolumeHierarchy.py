from src.dem.neighbor.AffineBodyNeighborBase import AffineBodyNeighborBase

import taichi as ti
from src.contact_detection.bounding_volume_hierarchy.AABB import AABB
from src.contact_detection.bounding_volume_hierarchy.MultiPatchLBVH import LBVH
from src.dem.neighbor.BoundingVolumeHierarchy import BoundingVolumeHierarchy
from src.dem.neighbor.NeighborKernel import (
    board_search_affine_edge_edge_bvh,
    board_search_affine_vertex_face_bvh,
    set_affine_bvh_aabbs,
)


@ti.data_oriented
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
        self.node_owner = ti.field(ti.i32, shape=self.lbvh.nodes.shape)
        self.node_escape = ti.field(ti.i32, shape=self.lbvh.nodes.shape)

    @ti.kernel
    def _initialize_leaf_owners(self, node2body: ti.template(), face2body: ti.template(), edge2body: ti.template()):
        for leaf in range(self.lbvh.aabb.current_n_aabbs):
            batch = self.lbvh.leaf2batch[leaf]
            begin = self.lbvh.prefix_batch_size[batch]
            size = self.lbvh.prefix_batch_size[batch + 1] - begin
            primitive = self.lbvh.primitive_index(leaf)
            owner = -1
            if batch == self.vertex_batch_id:
                owner = node2body[primitive]
            elif batch == self.face_batch_id:
                owner = face2body[primitive]
            else:
                owner = edge2body[primitive]
            self.node_owner[2 * begin - batch + size - 1 + leaf - begin] = owner

    @ti.kernel
    def _initialize_internal_owners(self, depth: ti.i32):
        for internal in range(self.lbvh.aabb.current_n_aabbs - self.lbvh.aabb.n_batches):
            if self.lbvh.internal_node_depth[internal] == depth:
                batch = self.lbvh.internal2batch[internal]
                begin = self.lbvh.prefix_batch_size[batch]
                root = 2 * begin - batch
                index = root + internal - begin + batch
                node = self.lbvh.nodes[index]
                left = self.node_owner[root + node.left]
                right = self.node_owner[root + node.right]
                self.node_owner[index] = left if left == right else -1

    def update(self, x, dx, faces, edges, node2body, face2body, edge2body, margin, swept=False):
        self._prepare_bounds(x, dx, faces, edges, margin, swept)
        vertex_prefix = int(self.aabb.prefix_batch_size[self.vertex_batch_id])
        face_prefix = int(self.aabb.prefix_batch_size[self.face_batch_id])
        edge_prefix = int(self.aabb.prefix_batch_size[self.edge_batch_id])
        set_affine_bvh_aabbs(
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
        primitive_count = self.vertex_num + self.face_num + self.edge_num
        self.lbvh.build(primitive_count)
        self.lbvh.export_escape_links(self.node_escape)
        self._initialize_leaf_owners(node2body, face2body, edge2body)
        for depth in range(self.lbvh.maximum_depth, -1, -1):
            self._initialize_internal_owners(depth)
        while True:
            board_search_affine_vertex_face_bvh(
                self.vertex_batch_id,
                self.face_batch_id,
                self.vertex_num,
                self.candidate_capacity,
                self.lbvh.prefix_batch_size,
                self.lbvh.nodes,
                self.node_owner,
                self.node_escape,
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
            board_search_affine_edge_edge_bvh(
                self.edge_batch_id,
                self.edge_num,
                self.edge_candidate_capacity,
                self.lbvh.prefix_batch_size,
                self.lbvh.nodes,
                self.node_owner,
                self.node_escape,
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

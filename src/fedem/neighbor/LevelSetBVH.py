from src.utils.errors import CapacityError
from src.fedem.neighbor.LevelSet import LevelSetCandidateBase

import taichi as ti
from src.contact_detection.bounding_volume_hierarchy.MultiPatchLBVH import LBVH


@ti.data_oriented
class LevelSetBVH(LevelSetCandidateBase):
    """LBVH over current rotated rigid level-set AABBs."""

    def __init__(self, simulation, patch, rigid_count):
        super().__init__(simulation, patch, rigid_count)
        self.bvh = LBVH(aabb=self.aabbs, extended_morton=True)
        self.bvh.initialize(active_aabbs=[self.rigid_count])
        self.stack_depth = 64
        self.query_stack = ti.field(
            dtype=ti.i32,
            shape=(max(self.vertex_count, 1), self.stack_depth),
        )

    @ti.func
    def _inside(self, point, bound):
        inside = True
        for component in ti.static(range(3)):
            inside = inside and bound.min[component] <= point[component] and point[component] <= bound.max[component]
        return inside

    @ti.kernel
    def _count_candidates(self, fem_position: ti.template()):
        self.node_prefix[0] = 0
        self.overflow[None] = 0
        for local_vertex in range(self.vertex_count):
            node = self.patch.surface_vertices[local_vertex]
            point = fem_position[node]
            count = 0
            stack_size = 1
            self.query_stack[local_vertex, 0] = 0
            while stack_size > 0:
                stack_size -= 1
                node_index = self.query_stack[local_vertex, stack_size]
                tree_node = self.bvh.nodes[node_index]
                if self._inside(point, tree_node.bound):
                    if tree_node.left == -1 and tree_node.right == -1:
                        morton_index = node_index - (self.rigid_count - 1)
                        rigid_id = self.bvh.primitive_index(morton_index)
                        count += 1
                    else:
                        if tree_node.right != -1:
                            if stack_size < ti.static(self.stack_depth):
                                self.query_stack[local_vertex, stack_size] = tree_node.right
                                stack_size += 1
                            else:
                                self.overflow[None] = 1
                        if tree_node.left != -1:
                            if stack_size < ti.static(self.stack_depth):
                                self.query_stack[local_vertex, stack_size] = tree_node.left
                                stack_size += 1
                            else:
                                self.overflow[None] = 1
            self.node_prefix[local_vertex + 1] = count

    @ti.kernel
    def _fill_candidates(self, fem_position: ti.template()):
        for local_vertex in range(self.vertex_count):
            node = self.patch.surface_vertices[local_vertex]
            point = fem_position[node]
            output = self.node_prefix[local_vertex]
            stack_size = 1
            self.query_stack[local_vertex, 0] = 0
            while stack_size > 0:
                stack_size -= 1
                node_index = self.query_stack[local_vertex, stack_size]
                tree_node = self.bvh.nodes[node_index]
                if self._inside(point, tree_node.bound):
                    if tree_node.left == -1 and tree_node.right == -1:
                        morton_index = node_index - (self.rigid_count - 1)
                        rigid_id = self.bvh.primitive_index(morton_index)
                        self.candidate_node[output] = node
                        self.candidate_rigid[output] = rigid_id
                        output += 1
                    else:
                        if tree_node.right != -1:
                            self.query_stack[local_vertex, stack_size] = tree_node.right
                            stack_size += 1
                        if tree_node.left != -1:
                            self.query_stack[local_vertex, stack_size] = tree_node.left
                            stack_size += 1

    def rebuild(self, fem_position, rigid, box, skin=0.0):
        self.aabbs.set_levelset_body_aabbs(self.rigid_count, 0, skin, rigid, box)
        self.bvh.update(self.simulation.current_step, self.simulation.bvh_rebuild_interval)
        self._count_candidates(fem_position)
        if int(self.overflow[None]) != 0:
            raise CapacityError("FEM--LSDEM BVH traversal stack overflow")
        count = self._candidate_count()
        self._fill_candidates(fem_position)
        return count

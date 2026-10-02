"""Refitted linear BVH for DEM sphere--FEM triangle candidates."""

import taichi as ti

from src.contact_detection.bounding_volume_hierarchy.AABB import AABB
from src.contact_detection.bounding_volume_hierarchy.MultiPatchLBVH import LBVH
from src.fedem.neighbor.NeighborBase import FEDEMNeighborBase
from src.utils.GeometryFunction import DistanceFromPointToTriangle


@ti.data_oriented
class FEDEMBVH(FEDEMNeighborBase):
    def __init__(self, simulation, patch):
        super().__init__(simulation, patch)
        self.face_aabbs = AABB(max(simulation.max_surface_facet_num, 2))
        self.bvh = LBVH(aabb=self.face_aabbs, extended_morton=True)
        self.bvh.initialize(active_aabbs=[patch.face_count])

    @ti.kernel
    def _update_face_aabbs(self):
        for face in range(self.patch.face_count):
            ids = self.patch.faces[face]
            self.face_aabbs.update_triangle_aabb(
                face,
                self.patch.nodes[ids[0]],
                self.patch.nodes[ids[1]],
                self.patch.nodes[ids[2]],
                0.0,
            )

    @ti.kernel
    def _search(self, particle_count: ti.i32, particles: ti.template()):
        self.particle_offsets.fill(0)
        self.overflow[None] = 0
        coordination = self.simulation.contact_coordination_number
        face_count = self.patch.face_count
        for particle in range(particle_count):
            local_count = 0
            if particles[particle].active != 0:
                position = particles[particle].x
                expansion = particles[particle].rad + 2.0 * self.skin
                query_min = position - expansion
                query_max = position + expansion
                query_stack = ti.Vector.zero(ti.i32, 64)
                stack_depth = 1
                while stack_depth > 0:
                    stack_depth -= 1
                    node_index = query_stack[stack_depth]
                    node = self.bvh.nodes[node_index]
                    intersects = True
                    for component in ti.static(range(3)):
                        intersects = (
                            intersects
                            and query_min[component] <= node.bound.max[component]
                            and query_max[component] >= node.bound.min[component]
                        )
                    if intersects:
                        if node.left == -1 and node.right == -1:
                            morton_index = node_index - (face_count - 1)
                            face = self.bvh.primitive_index(morton_index)
                            ids = self.patch.faces[face]
                            distance = ti.abs(
                                DistanceFromPointToTriangle(
                                    position,
                                    self.patch.nodes[ids[0]],
                                    self.patch.nodes[ids[1]],
                                    self.patch.nodes[ids[2]],
                                    self.patch.normals[face],
                                )
                            )
                            if distance <= expansion:
                                if local_count < coordination:
                                    self.potential_faces[particle * coordination + local_count] = face
                                else:
                                    self.overflow[None] = 1
                                local_count += 1
                        else:
                            if node.right != -1:
                                if stack_depth < 64:
                                    query_stack[stack_depth] = node.right
                                    stack_depth += 1
                                else:
                                    self.overflow[None] = 2
                            if node.left != -1:
                                if stack_depth < 64:
                                    query_stack[stack_depth] = node.left
                                    stack_depth += 1
                                else:
                                    self.overflow[None] = 2
            self.particle_offsets[particle + 1] = ti.min(local_count, coordination)

    def rebuild(self, particle_count, particles):
        particle_count = self._begin_rebuild(particle_count)
        self._update_face_aabbs()
        self.bvh.build(self.patch.face_count)
        self._search(particle_count, particles)
        return self._finish_rebuild(particle_count)


__all__ = ["FEDEMBVH"]

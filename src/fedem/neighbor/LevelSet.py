"""Dynamic FEM-boundary-node--rigid-level-set broad phases."""

from src.utils.errors import SimulationStateError

import taichi as ti
from src.contact_detection.bounding_volume_hierarchy.AABB import AABB
from src.utils.PrefixSum import PrefixSumExecutor


@ti.data_oriented
class LevelSetCandidateBase:
    """Preallocated compact node--rigid candidate output."""

    def __init__(self, simulation, patch, rigid_count):
        self.simulation = simulation
        self.patch = patch
        self.rigid_count = int(rigid_count)
        self.vertex_count = int(patch.surface_vertex_count)
        self.max_contact_pairs = int(simulation.max_contact_pairs)
        self.node_scan = PrefixSumExecutor(max(self.vertex_count + 1, 1))
        self.node_prefix = ti.field(dtype=ti.i32, shape=self.node_scan.get_length())
        self.candidate_node = ti.field(dtype=ti.i32, shape=self.max_contact_pairs)
        self.candidate_rigid = ti.field(dtype=ti.i32, shape=self.max_contact_pairs)
        self.overflow = ti.field(dtype=ti.i32, shape=())
        self.aabbs = AABB(max(self.rigid_count, 2))

    def _candidate_count(self):
        self.node_scan.run(self.node_prefix)
        count = int(self.node_prefix[self.vertex_count])
        if count > self.max_contact_pairs:
            raise SimulationStateError(
                "FEM--LSDEM max_contact_pairs is too small: " f"need {count}, allocated {self.max_contact_pairs}"
            )
        return count

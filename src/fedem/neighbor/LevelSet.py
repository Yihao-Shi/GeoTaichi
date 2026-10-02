"""Dynamic FEM-boundary-node--rigid-level-set broad phases."""

import math

import numpy as np
import taichi as ti

from src.contact_detection.bounding_volume_hierarchy.AABB import AABB
from src.contact_detection.bounding_volume_hierarchy.MultiPatchLBVH import LBVH
from src.fedem.neighbor.LinkedCell import _linear_cell_id
from src.fedem.structs import FEMLevelSetContact
from src.utils.PrefixSum import PrefixSumExecutor


def _power_of_two(value):
    value = max(int(value), 1)
    return 1 << (value - 1).bit_length()


@ti.data_oriented
class _LevelSetCandidateBase:
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
            raise RuntimeError(
                "FEM--LSDEM max_contact_pairs is too small: " f"need {count}, allocated {self.max_contact_pairs}"
            )
        return count


@ti.data_oriented
class _LevelSetLinkedCell(_LevelSetCandidateBase):
    """Per-rebuild rigid-AABB cell count, prefix sum, fill, and node query."""

    def __init__(self, simulation, patch, rigid_count, maximum_radius):
        super().__init__(simulation, patch, rigid_count)
        self.domain = tuple(float(value) for value in simulation.domain)
        if len(self.domain) != 3 or min(self.domain) <= 0.0:
            raise ValueError("FEM--LSDEM linked cell requires a positive 3D domain")
        self.grid_size = 2.0 * float(maximum_radius)
        if not math.isfinite(self.grid_size) or self.grid_size <= 0.0:
            raise ValueError("FEM--LSDEM linked-cell size must be positive")
        self.inverse_grid_size = 1.0 / self.grid_size
        self.cell_number = tuple(max(int(math.floor(length * self.inverse_grid_size)) + 1, 1) for length in self.domain)
        self.cell_count = int(np.prod(self.cell_number))
        self.cell_scan = PrefixSumExecutor(self.cell_count + 1)
        self.cell_offsets = ti.field(dtype=ti.i32, shape=self.cell_scan.get_length())
        self.cell_cursor = ti.field(dtype=ti.i32, shape=self.cell_count)
        self.membership_capacity = int(
            getattr(
                simulation,
                "max_levelset_cell_pairs",
                max(8 * self.max_contact_pairs, self.rigid_count, 1),
            )
        )
        self.cell_rigid = ti.field(dtype=ti.i32, shape=self.membership_capacity)

    @ti.func
    def _cell_span(self, rigid_id, domain, cell_number):
        lower = ti.max(self.aabbs.aabbs[rigid_id].min, 0.0)
        upper = ti.min(self.aabbs.aabbs[rigid_id].max, domain)
        begin = ti.cast(ti.floor(lower * self.inverse_grid_size), ti.i32)
        end = ti.cast(ti.floor(upper * self.inverse_grid_size), ti.i32)
        begin = ti.max(begin, 0)
        end = ti.min(end, cell_number - 1)
        valid = True
        for component in ti.static(range(3)):
            valid = valid and lower[component] <= upper[component]
        return begin, end, valid

    @ti.kernel
    def _count_memberships(
        self,
        domain: ti.types.vector(3, float),
        cell_number: ti.types.vector(3, ti.i32),
    ):
        self.cell_offsets.fill(0)
        for rigid_id in range(self.rigid_count):
            begin, end, valid = self._cell_span(rigid_id, domain, cell_number)
            if valid:
                for i, j, k in ti.ndrange(
                    (begin[0], end[0] + 1),
                    (begin[1], end[1] + 1),
                    (begin[2], end[2] + 1),
                ):
                    cell = _linear_cell_id(ti.Vector([i, j, k]), cell_number)
                    ti.atomic_add(self.cell_offsets[cell + 1], 1)

    @ti.kernel
    def _fill_memberships(
        self,
        domain: ti.types.vector(3, float),
        cell_number: ti.types.vector(3, ti.i32),
    ):
        self.cell_cursor.fill(0)
        for rigid_id in range(self.rigid_count):
            begin, end, valid = self._cell_span(rigid_id, domain, cell_number)
            if valid:
                for i, j, k in ti.ndrange(
                    (begin[0], end[0] + 1),
                    (begin[1], end[1] + 1),
                    (begin[2], end[2] + 1),
                ):
                    cell = _linear_cell_id(ti.Vector([i, j, k]), cell_number)
                    offset = ti.atomic_add(self.cell_cursor[cell], 1)
                    self.cell_rigid[self.cell_offsets[cell] + offset] = rigid_id

    @ti.kernel
    def _count_candidates(
        self,
        fem_position: ti.template(),
        domain: ti.types.vector(3, float),
        cell_number: ti.types.vector(3, ti.i32),
    ):
        self.node_prefix[0] = 0
        for local_vertex in range(self.vertex_count):
            node = self.patch.surface_vertices[local_vertex]
            point = fem_position[node]
            inside = True
            for component in ti.static(range(3)):
                inside = inside and 0.0 <= point[component] <= domain[component]
            count = 0
            if inside:
                index = ti.cast(ti.floor(point * self.inverse_grid_size), ti.i32)
                index = ti.max(0, ti.min(index, cell_number - 1))
                cell = _linear_cell_id(index, cell_number)
                for cursor in range(self.cell_offsets[cell], self.cell_offsets[cell + 1]):
                    rigid_id = self.cell_rigid[cursor]
                    if self.aabbs.aabbs[rigid_id].inside(point, 0.0):
                        count += 1
            self.node_prefix[local_vertex + 1] = count

    @ti.kernel
    def _fill_candidates(
        self,
        fem_position: ti.template(),
        domain: ti.types.vector(3, float),
        cell_number: ti.types.vector(3, ti.i32),
    ):
        for local_vertex in range(self.vertex_count):
            node = self.patch.surface_vertices[local_vertex]
            point = fem_position[node]
            inside = True
            for component in ti.static(range(3)):
                inside = inside and 0.0 <= point[component] <= domain[component]
            if inside:
                index = ti.cast(ti.floor(point * self.inverse_grid_size), ti.i32)
                index = ti.max(0, ti.min(index, cell_number - 1))
                cell = _linear_cell_id(index, cell_number)
                output = self.node_prefix[local_vertex]
                for cursor in range(self.cell_offsets[cell], self.cell_offsets[cell + 1]):
                    rigid_id = self.cell_rigid[cursor]
                    if self.aabbs.aabbs[rigid_id].inside(point, 0.0):
                        self.candidate_node[output] = node
                        self.candidate_rigid[output] = rigid_id
                        output += 1

    def rebuild(self, fem_position, rigid, box, skin=0.0):
        self.aabbs.set_levelset_body_aabbs(self.rigid_count, 0, skin, rigid, box)
        self._count_memberships(self.domain, self.cell_number)
        self.cell_scan.run(self.cell_offsets)
        memberships = int(self.cell_offsets[self.cell_count])
        if memberships > self.membership_capacity:
            raise RuntimeError(
                "FEM--LSDEM max_levelset_cell_pairs is too small: "
                f"need {memberships}, allocated {self.membership_capacity}"
            )
        self._fill_memberships(self.domain, self.cell_number)
        self._count_candidates(fem_position, self.domain, self.cell_number)
        count = self._candidate_count()
        self._fill_candidates(fem_position, self.domain, self.cell_number)
        return count


@ti.data_oriented
class _LevelSetBVH(_LevelSetCandidateBase):
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
        self.bvh.build(self.rigid_count)
        self._count_candidates(fem_position)
        if int(self.overflow[None]) != 0:
            raise RuntimeError("FEM--LSDEM BVH traversal stack overflow")
        count = self._candidate_count()
        self._fill_candidates(fem_position)
        return count


@ti.data_oriented
class FEMLevelSetBroadPhase:
    """Cull FEM boundary nodes against current rigid level-set AABBs."""

    def __init__(
        self,
        simulation,
        patch,
        dem_sims,
        dem_scene,
        interaction_distance=0.0,
    ):
        self.simulation = simulation
        self.patch = patch
        self.node_count = int(patch.node_count)
        self.rigid_count = int(dem_scene.particleNum[0])
        if self.rigid_count <= 0:
            raise RuntimeError("FEM soft-particle--LSDEM requires at least one rigid body")
        maximum_radius = float(dem_scene.find_bounding_sphere_max_radius(dem_sims))
        if maximum_radius <= 0.0:
            raise RuntimeError("FEM--LSDEM level-set bounding radius must be positive")
        self.skin = float(getattr(simulation, "verlet_distance", 0.0))
        self.interaction_distance = float(interaction_distance)
        if self.interaction_distance < 0.0:
            raise ValueError("FEM--LSDEM interaction distance must be non-negative")
        self.rebuild_required = ti.field(dtype=ti.i32, shape=())
        self.maximum_fem_sweep = ti.field(dtype=float, shape=())
        self.maximum_rigid_sweep = ti.field(dtype=float, shape=())
        self.search_mass_center = ti.Vector.field(3, dtype=float, shape=max(self.rigid_count, 1))
        self.search_orientation = ti.Vector.field(4, dtype=float, shape=max(self.rigid_count, 1))
        if simulation.search == "BVH":
            self.broad_phase = _LevelSetBVH(simulation, patch, self.rigid_count)
        else:
            self.broad_phase = _LevelSetLinkedCell(simulation, patch, self.rigid_count, maximum_radius)

        self.contact_capacity = int(
            getattr(
                simulation,
                "max_filtered_contact_pairs",
                simulation.max_contact_pairs,
            )
        )
        self.contacts = FEMLevelSetContact.field(shape=self.contact_capacity)
        self.contact_count = 0
        self.history_entry_capacity = int(
            getattr(
                simulation,
                "max_contact_history_pairs",
                simulation.max_contact_pairs,
            )
        )
        self.history_capacity = _power_of_two(4 * self.history_entry_capacity)
        self.history_state = ti.field(dtype=ti.i32, shape=self.history_capacity)
        self.history_key = ti.field(dtype=ti.i64, shape=self.history_capacity)
        self.history_overlap = ti.Vector.field(3, dtype=float, shape=self.history_capacity)
        self.history_overflow = ti.field(dtype=ti.i32, shape=())
        self.history_entry_count = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def _measure_rebuild_requirement(
        self,
        threshold: float,
        rigid: ti.template(),
    ):
        self.rebuild_required[None] = 0
        self.maximum_fem_sweep[None] = 0.0
        self.maximum_rigid_sweep[None] = 0.0
        reference = self.patch.surface_vertices[0]
        common_translation = self.patch.nodes[reference] - self.patch.search_positions[0]
        for local in range(self.patch.surface_vertex_count):
            node = self.patch.surface_vertices[local]
            relative_sweep = (self.patch.nodes[node] - self.patch.search_positions[local] - common_translation).norm()
            ti.atomic_max(self.maximum_fem_sweep[None], relative_sweep)
            if relative_sweep > threshold:
                self.rebuild_required[None] = 1
        for body in range(self.rigid_count):
            translation = (rigid[body].mass_center - self.search_mass_center[body] - common_translation).norm()
            current_q = rigid[body].q
            search_q = self.search_orientation[body]
            current_norm = current_q.norm()
            search_norm = search_q.norm()
            rotational_sweep = 2.0 * rigid[body].equi_r
            if current_norm > 1.0e-30 and search_norm > 1.0e-30:
                cosine_half_angle = ti.min(
                    ti.abs(current_q.dot(search_q)) / (current_norm * search_norm),
                    1.0,
                )
                sine_half_angle = ti.sqrt(ti.max(1.0 - cosine_half_angle * cosine_half_angle, 0.0))
                rotational_sweep = 2.0 * rigid[body].equi_r * sine_half_angle
            swept_distance = translation + rotational_sweep
            ti.atomic_max(self.maximum_rigid_sweep[None], swept_distance)
            if swept_distance > threshold:
                self.rebuild_required[None] = 1

    @ti.kernel
    def _commit_rigid_search_state(self, rigid: ti.template()):
        for body in range(self.rigid_count):
            self.search_mass_center[body] = rigid[body].mass_center
            self.search_orientation[body] = rigid[body].q
        self.maximum_rigid_sweep[None] = 0.0

    def requires_rebuild(self, rigid):
        """Return whether the padded level-set candidate list is exhausted.

        The list is built with a full Verlet skin.  Rebuilding after either
        the FEM surface or the maximum rigid surface sweep relative to a
        shared translation exceeds half of that skin bounds pairwise relative
        motion by one full skin.  Removing the shared translation avoids
        rebuilding when every grain undergoes the same free-fall increment.
        """
        if self.skin <= 0.0:
            return True
        self._measure_rebuild_requirement(0.5 * self.skin, rigid)
        return int(self.rebuild_required[None]) != 0

    @ti.kernel
    def _clear_history(self):
        self.history_overflow[None] = 0
        for slot in range(self.history_capacity):
            self.history_state[slot] = 2

    @ti.kernel
    def _count_history(self, old_count: ti.i32):
        self.history_entry_count[None] = 0
        for contact in range(old_count):
            if self.contacts[contact].old_tangential_overlap.norm_sqr() > 0.0:
                ti.atomic_add(self.history_entry_count[None], 1)

    @ti.kernel
    def _save_history(self, old_count: ti.i32):
        mask = ti.i64(self.history_capacity - 1)
        for contact in range(old_count):
            if self.contacts[contact].old_tangential_overlap.norm_sqr() <= 0.0:
                continue
            body = self.contacts[contact].rigid_id
            node = self.contacts[contact].node_id
            key = ti.i64(node) * ti.i64(self.rigid_count) + ti.i64(body)
            slot = int((key * ti.i64(1140071481932319845)) & mask)
            inserted = 0
            probe = 0
            while probe < self.history_capacity and inserted == 0:
                candidate = (slot + probe) & (self.history_capacity - 1)
                state = self.history_state[candidate]
                if state == 0:
                    if self.history_key[candidate] == key:
                        self.history_overlap[candidate] = self.contacts[contact].old_tangential_overlap
                        inserted = 1
                    else:
                        probe += 1
                elif state == 2:
                    previous = ti.atomic_min(self.history_state[candidate], 1)
                    if previous == 2:
                        self.history_key[candidate] = key
                        self.history_overlap[candidate] = self.contacts[contact].old_tangential_overlap
                        self.history_state[candidate] = 0
                        inserted = 1
                    elif previous == 0:
                        self.history_state[candidate] = 0
                    else:
                        # A different GPU thread owns this reserved slot.
                        # Probing onward avoids a warp-level spin deadlock.
                        probe += 1
                else:
                    probe += 1
            if inserted == 0:
                self.history_overflow[None] = 1

    @ti.func
    def _history_value(self, key):
        value = ti.Vector.zero(float, 3)
        found = 0
        mask = ti.i64(self.history_capacity - 1)
        slot = int((key * ti.i64(1140071481932319845)) & mask)
        probe = 0
        while probe < self.history_capacity and found == 0:
            candidate = (slot + probe) & (self.history_capacity - 1)
            state = self.history_state[candidate]
            if state == 0:
                if self.history_key[candidate] == key:
                    value = self.history_overlap[candidate]
                    found = 1
                else:
                    probe += 1
            elif state == 2:
                found = 1
            else:
                probe += 1
        return value

    @ti.kernel
    def _build_contacts(
        self,
        count: ti.i32,
        candidate_node: ti.template(),
        candidate_rigid: ti.template(),
    ):
        for contact in range(count):
            node = candidate_node[contact]
            body = candidate_rigid[contact]
            key = ti.i64(node) * ti.i64(self.rigid_count) + ti.i64(body)
            self.contacts[contact].rigid_id = body
            self.contacts[contact].node_id = node
            self.contacts[contact].active = 0
            self.contacts[contact].normal_force = ti.Vector.zero(float, 3)
            self.contacts[contact].tangential_force = ti.Vector.zero(float, 3)
            self.contacts[contact].old_tangential_overlap = self._history_value(key)

    def rebuild(self, fem_position, rigid, box):
        self._count_history(self.contact_count)
        history_entries = int(self.history_entry_count[None])
        if history_entries > self.history_entry_capacity:
            raise RuntimeError(
                "FEM--LSDEM max_contact_history_pairs is too small: "
                f"need {history_entries}, allocated "
                f"{self.history_entry_capacity}"
            )
        self._clear_history()
        self._save_history(self.contact_count)
        if int(self.history_overflow[None]) != 0:
            raise RuntimeError("FEM--LSDEM contact-history hash table overflow")
        count = self.broad_phase.rebuild(
            fem_position,
            rigid,
            box,
            skin=self.skin + self.interaction_distance,
        )
        if count > self.contact_capacity:
            raise RuntimeError(
                "FEM--LSDEM max_filtered_contact_pairs is too small: "
                f"need {count}, allocated {self.contact_capacity}"
            )
        self.contact_count = count
        self._build_contacts(
            count,
            self.broad_phase.candidate_node,
            self.broad_phase.candidate_rigid,
        )
        self.patch.commit_search_positions()
        self._commit_rigid_search_state(rigid)
        return count


__all__ = ["FEMLevelSetBroadPhase"]

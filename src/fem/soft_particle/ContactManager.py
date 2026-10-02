"""Lifecycle and persistent history for FEM soft-particle PT/EE contact."""

import math

import numpy as np
import taichi as ti

from src.fem.contact.BVHBroadPhase import DynamicBVHBroadPhase
from src.fem.contact.CollisionCulling import FEMCollisionCulling
from src.fem.contact.ContactTopology import build_contact_surface
from src.fem.contact.LinkedCellBroadPhase import DynamicLinkedCellBroadPhase
from src.fem.soft_particle.ContactKernel import (
    build_point_triangle_ranges,
    measure_active_point_triangle_penetration,
    resolve_edge_edge_contact,
    resolve_point_triangle_contact,
    update_surface_pseudonormals,
)
from src.fem.soft_particle.ContactModel import FEMSoftParticleProperty
from src.utils.FieldIO import field_to_numpy_prefix


def _power_of_two(value):
    value = max(int(value), 1)
    return 1 << (value - 1).bit_length()


@ti.data_oriented
class FEMSoftParticleContactManager:
    """Device-resident explicit contact between disconnected FEM bodies."""

    def __init__(self, mesh, state, contact, track_energy=True):
        if not mesh.is_volume:
            raise ValueError("FEM soft particles require a TET4 or HEX8 volume mesh")
        self.mesh = mesh
        self.state = state
        self.contact = contact
        self.track_energy = bool(track_energy)
        self.surface = build_contact_surface(mesh)
        self.body_count = int(np.max(mesh.cell_body_ids)) + 1
        self.real_type = ti.lang.impl.current_cfg().default_fp
        self.numpy_type = np.float64 if self.real_type == ti.f64 else np.float32
        self.node_count = int(mesh.number_of_nodes)
        self.face_count = int(self.surface.faces.shape[0])
        self.edge_count = int(self.surface.edges.shape[0])
        self.model_type = int(contact.model_type)
        self.node_pseudonormal = ti.Vector.field(3, dtype=self.real_type, shape=self.node_count)

        broad_phase_type = DynamicBVHBroadPhase if contact.search == "BVH" else DynamicLinkedCellBroadPhase
        capacities = contact.pair_capacities(self.face_count)
        fixed_pt_capacity = capacities["raw_point_triangle"]
        fixed_ee_capacity = capacities["raw_edge_edge"]
        filtered_pt_capacity = capacities["verlet_point_triangle"]
        filtered_ee_capacity = capacities["verlet_edge_edge"]
        broad_phase_options = {
            "max_point_triangle_pairs": fixed_pt_capacity,
            "max_edge_edge_pairs": fixed_ee_capacity,
            "node_system_ids": mesh.node_body_ids,
            "cross_system_only": True,
        }
        if contact.search == "BVH":
            # Keep every point--triangle stencil inside the Verlet radius.
            # Retaining only the face that is nearest at rebuild time is not
            # a valid Verlet list: the nearest face can change before the
            # displacement threshold is reached, especially at polyhedral
            # edges and corners.  CollisionCulling applies the exact-distance
            # filter to the complete BVH candidate set below.
            broad_phase_options["exact_proximity"] = False
        self.broad_phase = broad_phase_type(
            self.surface.faces,
            self.surface.edges,
            self.surface.vertices,
            self.surface.node_area,
            self.surface.edge_area,
            mesh.points,
            **broad_phase_options,
        )
        self.culling = FEMCollisionCulling(
            self.broad_phase,
            self.node_count,
            node_body_ids=mesh.node_body_ids,
            exclude_same_body=True,
            max_point_triangle_pairs=fixed_pt_capacity,
            max_edge_edge_pairs=fixed_ee_capacity,
            max_active_point_triangle_pairs=filtered_pt_capacity,
            max_active_edge_edge_pairs=filtered_ee_capacity,
            enable_ccd=False,
            signed_point_triangle=True,
            point_triangle_pseudonormals=self.node_pseudonormal,
        )
        self.faces = self.broad_phase.faces
        self.node_body = self.culling.node_body
        self.surface_vertices = ti.field(dtype=ti.i32, shape=max(int(self.surface.vertices.size), 1))
        vertex_buffer = np.zeros(self.surface_vertices.shape[0], dtype=np.int32)
        vertex_buffer[: self.surface.vertices.size] = self.surface.vertices
        self.surface_vertices.from_numpy(vertex_buffer)
        self.surface_vertex_count = int(self.surface.vertices.size)
        self.node_area = ti.field(dtype=self.real_type, shape=self.node_count)
        self.node_area.from_numpy(np.ascontiguousarray(self.surface.node_area, dtype=self.numpy_type))
        self.point_triangle_start = ti.field(dtype=ti.i32, shape=self.node_count)
        self.point_triangle_end = ti.field(dtype=ti.i32, shape=self.node_count)
        self.search_position = ti.Vector.field(3, dtype=self.real_type, shape=self.node_count)
        self.maximum_displacement = ti.field(dtype=self.real_type, shape=())
        self.maximum_point_triangle_penetration = ti.field(dtype=self.real_type, shape=())
        self.elastic_energy = ti.field(dtype=self.real_type, shape=())
        self.friction_dissipation = ti.field(dtype=self.real_type, shape=())
        self.damping_dissipation = ti.field(dtype=self.real_type, shape=())
        self.elastic_energy.fill(0.0)
        self.friction_dissipation.fill(0.0)
        self.damping_dissipation.fill(0.0)

        self.properties = FEMSoftParticleProperty.field(shape=(self.body_count, self.body_count))
        inactive = {
            "active": 0,
            "thickness": 0.0,
            "friction": 0.0,
            "kn": 0.0,
            "ks": 0.0,
            "normal_damping": 0.0,
            "tangential_damping": 0.0,
            "effective_young": 0.0,
            "effective_shear": 0.0,
            "restitution_damping": 0.0,
            "barrier_kappa": 0.0,
            "barrier_cutoff": 0.0,
            "barrier_stiffness_ratio": 0.0,
        }
        default_values, pair_values = contact.compact_pair_values(self.body_count)
        initial_values = default_values or inactive
        self._initialize_properties(
            int(initial_values["active"]),
            float(initial_values["thickness"]),
            float(initial_values["friction"]),
            float(initial_values["kn"]),
            float(initial_values["ks"]),
            float(initial_values["normal_damping"]),
            float(initial_values["tangential_damping"]),
            float(initial_values["effective_young"]),
            float(initial_values["effective_shear"]),
            float(initial_values["restitution_damping"]),
            float(initial_values["barrier_kappa"]),
            float(initial_values["barrier_cutoff"]),
            float(initial_values["barrier_stiffness_ratio"]),
        )
        for (first, second), values in pair_values.items():
            self.properties[first, second] = values
            self.properties[second, first] = values
        self.maximum_thickness = max(
            (
                float(values["thickness"])
                for values in (([] if default_values is None else [default_values]) + list(pair_values.values()))
            ),
            default=0.0,
        )
        self.maximum_interaction_distance = max(
            (
                float(values["thickness"]) + float(values["barrier_cutoff"])
                for values in (([] if default_values is None else [default_values]) + list(pair_values.values()))
            ),
            default=self.maximum_thickness,
        )

        minimum_edge = self._minimum_surface_edge(mesh.points)
        self.verlet_distance = (
            float(contact.verlet_distance)
            if contact.verlet_distance is not None
            else contact.verlet_distance_multiplier * minimum_edge
        )
        self.total_verlet_distance = 2.0 * self.verlet_distance
        self.search_distance = self.maximum_interaction_distance + self.total_verlet_distance
        self.edge_verlet_distance = min(
            self.verlet_distance,
            4.0 * minimum_edge,
        )
        self.total_edge_verlet_distance = 2.0 * self.edge_verlet_distance
        self.edge_search_distance = self.maximum_interaction_distance + self.total_edge_verlet_distance
        self.maximum_penetration = contact.maximum_penetration_fraction * minimum_edge
        if self.search_distance <= 0.0:
            raise ValueError("FEM soft-particle contact search distance must be positive")

        self.fixed_pt_capacity = filtered_pt_capacity is not None
        self.fixed_ee_capacity = filtered_ee_capacity is not None
        self.pt_capacity = max(int(filtered_pt_capacity or 1), 1)
        self.ee_capacity = max(int(filtered_ee_capacity or 1), 1)
        self.pt_count = 0
        self.ee_count = 0
        self.pt_active = ti.field(dtype=ti.i32, shape=self.pt_capacity)
        self.ee_active = ti.field(dtype=ti.i32, shape=self.ee_capacity)
        self.pt_normal_force = ti.Vector.field(3, dtype=self.real_type, shape=self.pt_capacity)
        self.ee_normal_force = ti.Vector.field(3, dtype=self.real_type, shape=self.ee_capacity)
        self.pt_tangential_force = ti.Vector.field(3, dtype=self.real_type, shape=self.pt_capacity)
        self.ee_tangential_force = ti.Vector.field(3, dtype=self.real_type, shape=self.ee_capacity)
        self.pt_tangential_overlap = ti.Vector.field(3, dtype=self.real_type, shape=self.pt_capacity)
        self.ee_tangential_overlap = ti.Vector.field(3, dtype=self.real_type, shape=self.ee_capacity)

        self.fixed_history_capacity = contact.contact_history_capacity is not None
        self.history_load_factor = contact.contact_history_load_factor
        self.history_capacity = _power_of_two(int(contact.contact_history_capacity or 1))
        self.history_state = ti.field(dtype=ti.i32, shape=self.history_capacity)
        self.history_kind = ti.field(dtype=ti.i32, shape=self.history_capacity)
        self.history_stencil = ti.Vector.field(4, dtype=ti.i32, shape=self.history_capacity)
        self.history_overlap = ti.Vector.field(3, dtype=self.real_type, shape=self.history_capacity)
        self.history_overflow = ti.field(dtype=ti.i32, shape=())
        self.history_entry_count = ti.field(dtype=ti.i32, shape=())
        self.peak_raw_pt_count = 0
        self.peak_raw_ee_count = 0
        self.peak_verlet_pt_count = 0
        self.peak_verlet_ee_count = 0
        self.rebuild(state.position)

    @ti.kernel
    def _initialize_properties(
        self,
        active: ti.i32,
        thickness: float,
        friction: float,
        kn: float,
        ks: float,
        normal_damping: float,
        tangential_damping: float,
        effective_young: float,
        effective_shear: float,
        restitution_damping: float,
        barrier_kappa: float,
        barrier_cutoff: float,
        barrier_stiffness_ratio: float,
    ):
        """Upload a uniform default contact law in one device kernel."""

        for first, second in ti.ndrange(self.body_count, self.body_count):
            self.properties[first, second].active = ti.select(first != second, active, 0)
            self.properties[first, second].thickness = thickness
            self.properties[first, second].friction = friction
            self.properties[first, second].kn = kn
            self.properties[first, second].ks = ks
            self.properties[first, second].normal_damping = normal_damping
            self.properties[first, second].tangential_damping = tangential_damping
            self.properties[first, second].effective_young = effective_young
            self.properties[first, second].effective_shear = effective_shear
            self.properties[first, second].restitution_damping = restitution_damping
            self.properties[first, second].barrier_kappa = barrier_kappa
            self.properties[first, second].barrier_cutoff = barrier_cutoff
            self.properties[first, second].barrier_stiffness_ratio = barrier_stiffness_ratio

    def _minimum_surface_edge(self, positions):
        edges = self.surface.edges
        lengths = np.linalg.norm(positions[edges[:, 1]] - positions[edges[:, 0]], axis=1)
        minimum = float(np.min(lengths))
        if not math.isfinite(minimum) or minimum <= 0.0:
            raise ValueError("FEM soft-particle surface contains a degenerate edge")
        return minimum

    @ti.func
    def _hash_stencil(self, kind, stencil, capacity):
        value = ti.u32(kind + 1) * ti.u32(2654435761)
        value ^= ti.u32(stencil[0] + 1) * ti.u32(73856093)
        value ^= ti.u32(stencil[1] + 1) * ti.u32(19349663)
        value ^= ti.u32(stencil[2] + 1) * ti.u32(83492791)
        value ^= ti.u32(stencil[3] + 1) * ti.u32(2166136261)
        return int(value & ti.u32(capacity - 1))

    @ti.func
    def _same_history(self, slot, kind, stencil, history_kind, history_stencil):
        same = history_kind[slot] == kind
        for local in ti.static(range(4)):
            same = same and history_stencil[slot][local] == stencil[local]
        return same

    @ti.func
    def _insert_history(
        self,
        kind,
        stencil,
        overlap,
        capacity,
        history_state,
        history_kind,
        history_stencil,
        history_overlap,
        history_overflow,
    ):
        slot = self._hash_stencil(kind, stencil, capacity)
        inserted = 0
        probe = 0
        while probe < capacity and inserted == 0:
            candidate = (slot + probe) & (capacity - 1)
            state = history_state[candidate]
            if state == 0:
                if self._same_history(
                    candidate,
                    kind,
                    stencil,
                    history_kind,
                    history_stencil,
                ):
                    history_overlap[candidate] = overlap
                    inserted = 1
                else:
                    probe += 1
            elif state == 2:
                previous = ti.atomic_min(history_state[candidate], 1)
                if previous == 2:
                    history_kind[candidate] = kind
                    history_stencil[candidate] = stencil
                    history_overlap[candidate] = overlap
                    history_state[candidate] = 0
                    inserted = 1
                elif previous == 0:
                    history_state[candidate] = 0
                else:
                    # Another GPU thread reserved this slot (state == 1).
                    # Do not spin on it: threads in the same warp can
                    # otherwise wait for one another forever.  Continue the
                    # linear probe and leave the owner to publish state 0.
                    probe += 1
            else:
                probe += 1
        if inserted == 0:
            history_overflow[None] = 1

    @ti.func
    def _find_history(
        self,
        kind,
        stencil,
        capacity,
        history_state,
        history_kind,
        history_stencil,
        history_overlap,
    ):
        overlap = ti.Vector.zero(float, 3)
        slot = self._hash_stencil(kind, stencil, capacity)
        finished = 0
        probe = 0
        while probe < capacity and finished == 0:
            candidate = (slot + probe) & (capacity - 1)
            state = history_state[candidate]
            if state == 0:
                if self._same_history(
                    candidate,
                    kind,
                    stencil,
                    history_kind,
                    history_stencil,
                ):
                    overlap = history_overlap[candidate]
                    finished = 1
                else:
                    probe += 1
            elif state == 2:
                finished = 1
            else:
                # A reserved slot should not normally survive the completed
                # save kernel, but advancing keeps corrupted/interrupted
                # history from turning a lookup into an infinite device loop.
                probe += 1
        return overlap

    @ti.kernel
    def _clear_history(
        self,
        capacity: ti.i32,
        history_state: ti.template(),
        history_overflow: ti.template(),
    ):
        history_overflow[None] = 0
        for slot in range(capacity):
            history_state[slot] = 2

    @ti.kernel
    def _count_history_entries(
        self,
        pt_count: ti.i32,
        ee_count: ti.i32,
        pt_overlap: ti.template(),
        ee_overlap: ti.template(),
        entry_count: ti.template(),
    ):
        """Count only contacts that carry persistent tangential state."""

        entry_count[None] = 0
        for contact in range(pt_count):
            if pt_overlap[contact].norm_sqr() > 0.0:
                ti.atomic_add(entry_count[None], 1)
        for contact in range(ee_count):
            if ee_overlap[contact].norm_sqr() > 0.0:
                ti.atomic_add(entry_count[None], 1)

    @ti.kernel
    def _save_history(
        self,
        pt_count: ti.i32,
        ee_count: ti.i32,
        point_triangle: ti.template(),
        edge_edge: ti.template(),
        pt_overlap: ti.template(),
        ee_overlap: ti.template(),
        capacity: ti.i32,
        history_state: ti.template(),
        history_kind: ti.template(),
        history_stencil: ti.template(),
        history_overlap: ti.template(),
        history_overflow: ti.template(),
    ):
        for contact in range(pt_count):
            if pt_overlap[contact].norm_sqr() > 0.0:
                self._insert_history(
                    0,
                    point_triangle[contact],
                    pt_overlap[contact],
                    capacity,
                    history_state,
                    history_kind,
                    history_stencil,
                    history_overlap,
                    history_overflow,
                )
        for contact in range(ee_count):
            if ee_overlap[contact].norm_sqr() > 0.0:
                self._insert_history(
                    1,
                    edge_edge[contact],
                    ee_overlap[contact],
                    capacity,
                    history_state,
                    history_kind,
                    history_stencil,
                    history_overlap,
                    history_overflow,
                )

    @ti.kernel
    def _load_history(
        self,
        pt_count: ti.i32,
        ee_count: ti.i32,
        point_triangle: ti.template(),
        edge_edge: ti.template(),
        pt_overlap: ti.template(),
        ee_overlap: ti.template(),
        capacity: ti.i32,
        history_state: ti.template(),
        history_kind: ti.template(),
        history_stencil: ti.template(),
        history_overlap: ti.template(),
    ):
        for contact in range(pt_count):
            pt_overlap[contact] = self._find_history(
                0,
                point_triangle[contact],
                capacity,
                history_state,
                history_kind,
                history_stencil,
                history_overlap,
            )
        for contact in range(ee_count):
            ee_overlap[contact] = self._find_history(
                1,
                edge_edge[contact],
                capacity,
                history_state,
                history_kind,
                history_stencil,
                history_overlap,
            )

    @ti.kernel
    def _commit_search_positions(self, positions: ti.template()):
        self.maximum_displacement[None] = 0.0
        for node in range(self.node_count):
            self.search_position[node] = positions[node]

    @ti.kernel
    def _measure_search_displacement(self, positions: ti.template()):
        self.maximum_displacement[None] = 0.0
        reference = self.surface_vertices[0]
        common_translation = positions[reference] - self.search_position[reference]
        for local in range(self.surface_vertex_count):
            node = self.surface_vertices[local]
            ti.atomic_max(
                self.maximum_displacement[None],
                (positions[node] - self.search_position[node] - common_translation).norm(),
            )

    def _resize_history(self, required):
        capacity = _power_of_two(
            max(
                int(math.ceil(int(required) / self.history_load_factor)),
                1,
            )
        )
        if capacity <= self.history_capacity:
            return
        if self.fixed_history_capacity:
            raise RuntimeError(
                "FEM soft-particle contact_history_capacity is too small: "
                f"need at least {capacity}, allocated {self.history_capacity}"
            )
        self.history_capacity = capacity
        self.history_state = ti.field(dtype=ti.i32, shape=capacity)
        self.history_kind = ti.field(dtype=ti.i32, shape=capacity)
        self.history_stencil = ti.Vector.field(4, dtype=ti.i32, shape=capacity)
        self.history_overlap = ti.Vector.field(3, dtype=self.real_type, shape=capacity)

    def _resize_contacts(self, pt_count, ee_count):
        required = _power_of_two(max(pt_count, 1))
        if required > self.pt_capacity:
            if self.fixed_pt_capacity:
                raise RuntimeError(
                    "FEM soft-particle max_point_triangle_pairs is too small: "
                    f"need {pt_count}, allocated {self.pt_capacity}"
                )
            self.pt_capacity = required
            self.pt_active = ti.field(dtype=ti.i32, shape=required)
            self.pt_normal_force = ti.Vector.field(3, dtype=self.real_type, shape=required)
            self.pt_tangential_force = ti.Vector.field(3, dtype=self.real_type, shape=required)
            self.pt_tangential_overlap = ti.Vector.field(3, dtype=self.real_type, shape=required)
        required = _power_of_two(max(ee_count, 1))
        if required > self.ee_capacity:
            if self.fixed_ee_capacity:
                raise RuntimeError(
                    "FEM soft-particle max_edge_edge_pairs is too small: "
                    f"need {ee_count}, allocated {self.ee_capacity}"
                )
            self.ee_capacity = required
            self.ee_active = ti.field(dtype=ti.i32, shape=required)
            self.ee_normal_force = ti.Vector.field(3, dtype=self.real_type, shape=required)
            self.ee_tangential_force = ti.Vector.field(3, dtype=self.real_type, shape=required)
            self.ee_tangential_overlap = ti.Vector.field(3, dtype=self.real_type, shape=required)

    def needs_rebuild(self, positions):
        self._measure_search_displacement(positions)
        return float(self.maximum_displacement[None]) > 0.5 * self.total_edge_verlet_distance

    def rebuild(self, positions):
        update_surface_pseudonormals(
            self.face_count,
            self.faces,
            positions,
            self.node_pseudonormal,
        )
        old_pt_count, old_ee_count = self.pt_count, self.ee_count
        if old_pt_count:
            measure_active_point_triangle_penetration(
                old_pt_count,
                self.culling.point_triangle,
                positions,
                self.node_pseudonormal,
                self.pt_active,
                self.maximum_point_triangle_penetration,
            )
            penetration = float(self.maximum_point_triangle_penetration[None])
            if penetration > self.maximum_penetration:
                raise RuntimeError(
                    "FEM soft-particle penetration exceeded the mesh-scale "
                    "quality limit before a Verlet rebuild: "
                    f"depth={penetration:.6g}, quality limit="
                    f"{self.maximum_penetration:.6g}. Increase contact "
                    "stiffness or reduce the time step; changing the Verlet "
                    "distance does not relax this physical accuracy gate."
                )
        self._count_history_entries(
            old_pt_count,
            old_ee_count,
            self.pt_tangential_overlap,
            self.ee_tangential_overlap,
            self.history_entry_count,
        )
        self._resize_history(int(self.history_entry_count[None]))
        self._clear_history(
            self.history_capacity,
            self.history_state,
            self.history_overflow,
        )
        if old_pt_count or old_ee_count:
            self._save_history(
                old_pt_count,
                old_ee_count,
                self.culling.point_triangle,
                self.culling.edge_edge,
                self.pt_tangential_overlap,
                self.ee_tangential_overlap,
                self.history_capacity,
                self.history_state,
                self.history_kind,
                self.history_stencil,
                self.history_overlap,
                self.history_overflow,
            )
            if int(self.history_overflow[None]) != 0:
                raise RuntimeError("FEM soft-particle contact-history hash overflow")
        pt_count, ee_count = self.culling.rebuild_proximity(
            positions,
            self.search_distance,
            self.edge_search_distance,
        )
        raw_pt_count = int(self.culling.raw_point_triangle_count[None])
        raw_ee_count = int(self.culling.raw_edge_edge_count[None])
        self.peak_raw_pt_count = max(self.peak_raw_pt_count, raw_pt_count)
        self.peak_raw_ee_count = max(self.peak_raw_ee_count, raw_ee_count)
        self.peak_verlet_pt_count = max(self.peak_verlet_pt_count, pt_count)
        self.peak_verlet_ee_count = max(self.peak_verlet_ee_count, ee_count)
        self._resize_contacts(pt_count, ee_count)
        self.pt_count, self.ee_count = pt_count, ee_count
        self._load_history(
            pt_count,
            ee_count,
            self.culling.point_triangle,
            self.culling.edge_edge,
            self.pt_tangential_overlap,
            self.ee_tangential_overlap,
            self.history_capacity,
            self.history_state,
            self.history_kind,
            self.history_stencil,
            self.history_overlap,
        )
        self.rebuild_point_triangle_ranges()
        self._commit_search_positions(positions)
        return pt_count, ee_count

    def rebuild_point_triangle_ranges(self):
        build_point_triangle_ranges(
            self.node_count,
            self.pt_count,
            self.culling.point_triangle,
            self.point_triangle_start,
            self.point_triangle_end,
        )

    def rebuild_restart_candidates(self, positions):
        """Reconstruct legacy-checkpoint stencils without erasing history.

        Checkpoints written before compact PT/EE stencils and their Python
        counts were archived still contain the persistent history hash.  A
        normal ``rebuild`` would first clear that hash and, because the
        reconstructed manager starts with zero Python counts, would lose all
        recoverable tangential history.  Rebuild the derived candidates from
        the restored nodal positions and load the archived hash directly.
        """
        pt_count, ee_count = self.culling.rebuild_proximity(
            positions,
            self.search_distance,
            self.edge_search_distance,
        )
        raw_pt_count = int(self.culling.raw_point_triangle_count[None])
        raw_ee_count = int(self.culling.raw_edge_edge_count[None])
        self.peak_raw_pt_count = max(self.peak_raw_pt_count, raw_pt_count)
        self.peak_raw_ee_count = max(self.peak_raw_ee_count, raw_ee_count)
        self.peak_verlet_pt_count = max(self.peak_verlet_pt_count, pt_count)
        self.peak_verlet_ee_count = max(self.peak_verlet_ee_count, ee_count)
        self._resize_contacts(pt_count, ee_count)
        self.pt_count, self.ee_count = pt_count, ee_count
        self._load_history(
            pt_count,
            ee_count,
            self.culling.point_triangle,
            self.culling.edge_edge,
            self.pt_tangential_overlap,
            self.ee_tangential_overlap,
            self.history_capacity,
            self.history_state,
            self.history_kind,
            self.history_stencil,
            self.history_overlap,
        )
        self.rebuild_point_triangle_ranges()
        self._commit_search_positions(positions)
        return pt_count, ee_count

    def resolve(
        self,
        dt,
        *,
        advance_history=True,
        check_rebuild=True,
    ):
        # Stored contact energy is instantaneous; dissipated work is
        # cumulative.  All three reductions stay device resident.
        self.elastic_energy.fill(0.0)
        update_surface_pseudonormals(
            self.face_count,
            self.faces,
            self.state.position,
            self.node_pseudonormal,
        )
        if check_rebuild and self.needs_rebuild(self.state.position):
            self.rebuild(self.state.position)
        resolve_point_triangle_contact(
            self.model_type,
            self.track_energy,
            self.pt_count,
            float(dt),
            int(advance_history),
            self.culling.point_triangle,
            self.node_body,
            self.properties,
            self.state.position,
            self.state.velocity,
            self.state.mass,
            self.node_area,
            self.node_pseudonormal,
            self.point_triangle_start,
            self.point_triangle_end,
            self.state.external_force,
            self.pt_active,
            self.pt_normal_force,
            self.pt_tangential_force,
            self.pt_tangential_overlap,
            self.elastic_energy,
            self.friction_dissipation,
            self.damping_dissipation,
        )
        resolve_edge_edge_contact(
            self.model_type,
            self.track_energy,
            self.ee_count,
            float(dt),
            int(advance_history),
            self.culling.edge_edge,
            self.node_body,
            self.properties,
            self.culling.edge_edge_measure,
            self.state.position,
            self.state.velocity,
            self.state.mass,
            self.state.external_force,
            self.ee_active,
            self.ee_normal_force,
            self.ee_tangential_force,
            self.ee_tangential_overlap,
            self.elastic_energy,
            self.friction_dissipation,
            self.damping_dissipation,
        )

    def energy_diagnostics(self):
        """Return one output-frame snapshot of the device energy ledger."""
        return {
            "elastic_energy": float(self.elastic_energy[None]),
            "friction_dissipation": float(self.friction_dissipation[None]),
            "damping_dissipation": float(self.damping_dissipation[None]),
        }

    def critical_timestep(self):
        if self.body_count <= 1:
            return math.inf
        minimum_mass = float(np.min(self.state.mass.to_numpy()))
        maximum_area = float(np.max(self.surface.node_area))
        default_values, overrides = self.contact.compact_pair_values(self.body_count)
        pair_values = ([] if default_values is None else [default_values]) + list(overrides.values())
        if self.model_type == 0:
            maximum_stiffness = max(
                (maximum_area * max(float(value["kn"]), float(value["ks"])) for value in pair_values),
                default=0.0,
            )
        elif self.model_type == 1:
            maximum_stiffness = max(
                (
                    maximum_area
                    * max(float(value["effective_young"]), 4.0 * float(value["effective_shear"]))
                    * max(math.sqrt(max(float(value["thickness"]), 1.0e-12)), 1.0e-6)
                    for value in pair_values
                ),
                default=0.0,
            )
        else:
            maximum_stiffness = max(
                (maximum_area * (4.0 + 2.0 * math.log(2.0)) * float(value["barrier_kappa"]) for value in pair_values),
                default=0.0,
            )
        return math.inf if maximum_stiffness <= 0.0 else math.sqrt(minimum_mass / maximum_stiffness)

    def diagnostics(self):
        return {
            "raw_point_triangle_candidates": int(self.culling.raw_point_triangle_count[None]),
            "raw_edge_edge_candidates": int(self.culling.raw_edge_edge_count[None]),
            "point_triangle_candidates": self.pt_count,
            "edge_edge_candidates": self.ee_count,
            "point_triangle_active": int(np.count_nonzero(field_to_numpy_prefix(self.pt_active, self.pt_count))),
            "edge_edge_active": int(np.count_nonzero(field_to_numpy_prefix(self.ee_active, self.ee_count))),
            "point_triangle_capacity": self.pt_capacity,
            "edge_edge_capacity": self.ee_capacity,
            "raw_point_triangle_capacity": self.broad_phase.pt_capacity,
            "raw_edge_edge_capacity": self.broad_phase.ee_capacity,
            "peak_raw_point_triangle_candidates": self.peak_raw_pt_count,
            "peak_raw_edge_edge_candidates": self.peak_raw_ee_count,
            "peak_point_triangle_candidates": self.peak_verlet_pt_count,
            "peak_edge_edge_candidates": self.peak_verlet_ee_count,
            "history_capacity": self.history_capacity,
            "history_entry_count": int(self.history_entry_count[None]),
            "history_load_factor": (int(self.history_entry_count[None]) / self.history_capacity),
            "maximum_point_triangle_penetration": float(self.maximum_point_triangle_penetration[None]),
            "maximum_penetration_limit": self.maximum_penetration,
            "point_triangle_search_distance": self.search_distance,
            "edge_edge_search_distance": self.edge_search_distance,
        }


__all__ = ["FEMSoftParticleContactManager"]

"""Shared compact contact-list storage for FEM--MPM broad phases."""

import taichi as ti

from src.fempm.structs import FEMPMContact, FEMPMHistoryContact
from src.utils.PrefixSum import PrefixSumExecutor


@ti.data_oriented
class FEMPMNeighborBase:
    def __init__(self, simulation, patch):
        self.simulation = simulation
        self.patch = patch
        self.skin = float(simulation.verlet_distance)
        self.particle_scan = PrefixSumExecutor(simulation.max_particle_num + 1)
        self.particle_offsets = ti.field(dtype=ti.i32, shape=self.particle_scan.get_length())
        self.history_offsets = ti.field(dtype=ti.i32, shape=self.particle_scan.get_length())
        self.potential_faces = ti.field(
            dtype=ti.i32,
            shape=(simulation.max_particle_num * simulation.contact_coordination_number),
        )
        self.contacts = FEMPMContact.field(shape=simulation.max_contact_pairs)
        self.history_contacts = FEMPMHistoryContact.field(shape=simulation.max_contact_pairs)
        self.overflow = ti.field(dtype=ti.i32, shape=())
        self.maximum_particle_displacement = ti.field(dtype=float, shape=())
        self.contact_count = 0

    @ti.kernel
    def _save_history(self, particle_count: ti.i32, old_contact_count: ti.i32):
        for particle in range(particle_count + 1):
            self.history_offsets[particle] = self.particle_offsets[particle]
        for contact in range(old_contact_count):
            self.history_contacts[contact].face_id = self.contacts[contact].face_id
            self.history_contacts[contact].old_tangential_overlap = self.contacts[contact].old_tangential_overlap

    @ti.kernel
    def _build_contacts(self, particle_count: ti.i32):
        coordination = self.simulation.contact_coordination_number
        for particle in range(particle_count):
            number = self.particle_offsets[particle + 1] - self.particle_offsets[particle]
            for offset in range(number):
                contact = self.particle_offsets[particle] + offset
                face = self.potential_faces[particle * coordination + offset]
                overlap = ti.Vector.zero(float, 3)
                for old_contact in range(
                    self.history_offsets[particle],
                    self.history_offsets[particle + 1],
                ):
                    if self.history_contacts[old_contact].face_id == face:
                        overlap = self.history_contacts[old_contact].old_tangential_overlap
                        break
                self.contacts[contact].particle_id = particle
                self.contacts[contact].face_id = face
                self.contacts[contact].active = 0
                self.contacts[contact].normal_force = ti.Vector.zero(float, 3)
                self.contacts[contact].tangential_force = ti.Vector.zero(float, 3)
                self.contacts[contact].old_tangential_overlap = overlap

    @ti.kernel
    def _measure_particle_displacement(self, particle_count: ti.i32, particles: ti.template()):
        self.maximum_particle_displacement[None] = 0.0
        for particle in range(particle_count):
            if particles[particle].active != 0 and particles[particle].coupling != 0:
                ti.atomic_max(
                    self.maximum_particle_displacement[None],
                    particles[particle].verletDisp.norm(),
                )

    def particle_requires_rebuild(self, particle_count, particles):
        self._measure_particle_displacement(int(particle_count), particles)
        return float(self.maximum_particle_displacement[None]) > 0.5 * self.simulation.verlet_distance

    def _begin_rebuild(self, particle_count):
        particle_count = int(particle_count)
        self._save_history(particle_count, self.contact_count)
        return particle_count

    def _finish_rebuild(self, particle_count):
        if int(self.overflow[None]) != 0:
            reason = "BVH traversal stack" if int(self.overflow[None]) == 2 else "contact_coordination_number"
            raise RuntimeError(f"FEMPM {reason} capacity is too small")
        self.particle_scan.run(self.particle_offsets)
        self.contact_count = int(self.particle_offsets[particle_count])
        if self.contact_count > self.simulation.max_contact_pairs:
            raise RuntimeError(
                "FEMPM max_contact_pairs/compaction_ratio is too small: "
                f"need {self.contact_count}, allocated "
                f"{self.simulation.max_contact_pairs}"
            )
        self._build_contacts(particle_count)
        self.patch.commit_search_positions()
        return self.contact_count


__all__ = ["FEMPMNeighborBase"]

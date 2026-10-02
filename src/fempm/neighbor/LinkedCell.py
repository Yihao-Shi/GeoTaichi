"""Dynamic prefix-sum linked cells for deforming FEM--MPM contact."""

import math

import numpy as np
import taichi as ti

from src.fempm.neighbor.NeighborBase import FEMPMNeighborBase
from src.utils.GeometryFunction import DistanceFromPointToTriangle
from src.utils.PrefixSum import PrefixSumExecutor


@ti.func
def _linear_cell_id(index, cell_number):
    return index[0] + cell_number[0] * (index[1] + cell_number[1] * index[2])


@ti.data_oriented
class FEMPMLinkedCell(FEMPMNeighborBase):
    def __init__(self, simulation, patch, maximum_particle_radius, face_radius):
        super().__init__(simulation, patch)
        self.maximum_particle_radius = float(maximum_particle_radius)
        if self.maximum_particle_radius <= 0.0:
            raise ValueError("FEMPM requires positive MPM particle radii")
        self.grid_size = self.maximum_particle_radius + 1.1 * float(face_radius) + self.skin
        if self.grid_size <= 1.0e-15:
            raise ValueError("FEMPM linked-cell size is zero")
        self.inverse_grid_size = 1.0 / self.grid_size
        self.cell_number = tuple(
            max(int(math.floor(length * self.inverse_grid_size)) + 1, 1) for length in simulation.domain
        )
        self.cell_count = int(np.prod(self.cell_number))
        self.cell_scan = PrefixSumExecutor(self.cell_count + 1)
        self.cell_offsets = ti.field(dtype=ti.i32, shape=self.cell_scan.get_length())
        self.cell_cursor = ti.field(dtype=ti.i32, shape=self.cell_count)
        self.cell_faces = ti.field(dtype=ti.i32, shape=simulation.max_facet_cell_pairs)

    @ti.kernel
    def _count_face_cells(
        self,
        domain: ti.types.vector(3, float),
        cell_number: ti.types.vector(3, ti.i32),
    ):
        self.cell_offsets.fill(0)
        expansion = self.maximum_particle_radius + 2.0 * self.skin
        for face in range(self.patch.face_count):
            ids = self.patch.faces[face]
            first = self.patch.nodes[ids[0]]
            second = self.patch.nodes[ids[1]]
            third = self.patch.nodes[ids[2]]
            lower = ti.max(ti.min(first, second, third) - expansion, 0.0)
            upper = ti.min(ti.max(first, second, third) + expansion, domain)
            begin = ti.cast(ti.floor(lower * self.inverse_grid_size), ti.i32)
            end = ti.cast(ti.floor(upper * self.inverse_grid_size), ti.i32)
            begin = ti.max(begin, 0)
            end = ti.min(end, cell_number - 1)
            for i, j, k in ti.ndrange(
                (begin[0], end[0] + 1),
                (begin[1], end[1] + 1),
                (begin[2], end[2] + 1),
            ):
                cell = _linear_cell_id(ti.Vector([i, j, k]), cell_number)
                ti.atomic_add(self.cell_offsets[cell + 1], 1)

    @ti.kernel
    def _insert_face_cells(
        self,
        domain: ti.types.vector(3, float),
        cell_number: ti.types.vector(3, ti.i32),
    ):
        self.cell_cursor.fill(0)
        expansion = self.maximum_particle_radius + 2.0 * self.skin
        for face in range(self.patch.face_count):
            ids = self.patch.faces[face]
            first = self.patch.nodes[ids[0]]
            second = self.patch.nodes[ids[1]]
            third = self.patch.nodes[ids[2]]
            lower = ti.max(ti.min(first, second, third) - expansion, 0.0)
            upper = ti.min(ti.max(first, second, third) + expansion, domain)
            begin = ti.cast(ti.floor(lower * self.inverse_grid_size), ti.i32)
            end = ti.cast(ti.floor(upper * self.inverse_grid_size), ti.i32)
            begin = ti.max(begin, 0)
            end = ti.min(end, cell_number - 1)
            for i, j, k in ti.ndrange(
                (begin[0], end[0] + 1),
                (begin[1], end[1] + 1),
                (begin[2], end[2] + 1),
            ):
                cell = _linear_cell_id(ti.Vector([i, j, k]), cell_number)
                offset = ti.atomic_add(self.cell_cursor[cell], 1)
                self.cell_faces[self.cell_offsets[cell] + offset] = face

    @ti.kernel
    def _search(
        self,
        particle_count: ti.i32,
        particles: ti.template(),
        domain: ti.types.vector(3, float),
        cell_number: ti.types.vector(3, ti.i32),
    ):
        self.particle_offsets.fill(0)
        self.overflow[None] = 0
        coordination = self.simulation.contact_coordination_number
        for particle in range(particle_count):
            position = particles[particle].x
            local_count = 0
            inside = True
            for component in ti.static(range(3)):
                inside = inside and 0.0 <= position[component] <= domain[component]
            if particles[particle].active != 0 and particles[particle].coupling != 0 and inside:
                index = ti.cast(ti.floor(position * self.inverse_grid_size), ti.i32)
                index = ti.max(0, ti.min(index, cell_number - 1))
                cell = _linear_cell_id(index, cell_number)
                for cursor in range(self.cell_offsets[cell], self.cell_offsets[cell + 1]):
                    face = self.cell_faces[cursor]
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
                    if distance <= particles[particle].rad + 2.0 * self.skin:
                        if local_count < coordination:
                            self.potential_faces[particle * coordination + local_count] = face
                        else:
                            self.overflow[None] = 1
                        local_count += 1
            self.particle_offsets[particle + 1] = ti.min(local_count, coordination)

    def rebuild(self, particle_count, particles):
        particle_count = self._begin_rebuild(particle_count)
        self._count_face_cells(self.simulation.domain, self.cell_number)
        self.cell_scan.run(self.cell_offsets)
        facet_cell_pairs = int(self.cell_offsets[self.cell_count])
        if facet_cell_pairs > self.simulation.max_facet_cell_pairs:
            raise RuntimeError(
                "FEMPM max_facet_cell_pairs is too small: "
                f"need {facet_cell_pairs}, allocated "
                f"{self.simulation.max_facet_cell_pairs}"
            )
        self._insert_face_cells(self.simulation.domain, self.cell_number)
        self._search(particle_count, particles, self.simulation.domain, self.cell_number)
        return self._finish_rebuild(particle_count)


__all__ = ["FEMPMLinkedCell"]

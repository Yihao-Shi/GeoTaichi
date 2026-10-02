import taichi as ti
import numpy as np

from src.mpm.Simulation import Simulation


@ti.data_oriented
class StaggeredGrid:
    def __init__(self, sims: Simulation, cnum, ghost_cell=1) -> None:
        self.dimension = sims.dimension
        self.force = [ti.field(dtype=float, 
            shape=([cnum[i] + (d == i) + 2 * ghost_cell for i in range(sims.dimension)]), 
            offset=([-ghost_cell for _ in range(sims.dimension)])) 
            for d in range(sims.dimension)]
        self.velocity = [ti.field(dtype=float, 
            shape=([cnum[i] + (d == i) + 2 * ghost_cell for i in range(sims.dimension)]), 
            offset=([-ghost_cell for _ in range(sims.dimension)])) 
            for d in range(sims.dimension)]
        self.m = [ti.field(dtype=float, 
            shape=([cnum[i] + (d == i) + 2 * ghost_cell for i in range(sims.dimension)]), 
            offset=([-ghost_cell for _ in range(sims.dimension)])) 
            for d in range(sims.dimension)]
        if sims.coupling:
            indice = ti.ij if self.dimension == 2 else ti.ijk
            self.sdf = ti.field(dtype=float)
            self.sdfID = ti.field(dtype=float)
            ti.root.dense(indice, ([cnum[i] + 1 for i in range(sims.dimension)])).place(self.sdf, self.sdfID)

    @ti.kernel
    def grid_reset(self, cutoff: float):
        for d in ti.static(range(self.dimension)):
            for I in ti.grouped(self.m[d]):
                self.m[d][I] = 0.
                self.velocity[d][I] = 0.
                self.force[d][I] = 0.

    @ti.func
    def _compute_nodal_acceleration(self, dt):
        self.force = self.velocity - self.force

    @ti.func
    def velocity_constraint(self, dirs, prescribed_velocity):
        self.velocity[dirs] = prescribed_velocity
        self.force[dirs] = 0.

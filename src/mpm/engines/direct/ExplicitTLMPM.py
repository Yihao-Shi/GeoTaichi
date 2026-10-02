import taichi as ti
import numpy as np

import src.mpm.config as config
from src.mpm.engines.direct.ExplicitMPM import ExplicitMPM
from src.mpm.utils import vectorize_id


@ti.data_oriented
class ExplicitTLMPM(ExplicitMPM):
    def __init__(self, bodies, dirichlet=None, neumann=None, name="case", **kwargs):
        super().__init__(bodies, dirichlet, neumann, name, configuration="TL", **kwargs)

    @ti.kernel
    def grid_reset(self):
        for i in self.grid:
            if self.grid[i].m > self.val_lim:
                self.grid[i].v = ti.Vector.zero(ti.f64, config.DIM)
                self.grid[i].a = ti.Vector.zero(ti.f64, config.DIM)

    @ti.kernel
    def mass_p2g(self):
        for i in range(self.particleNum[0]):
            p_mass = self.particle[i].m
            for j in range(self.offset[i]):
                nodeID = self.LnID[i, j]
                shape_fn = self.shape[i, j]
                self.grid[nodeID].m += shape_fn * p_mass

    @ti.kernel
    def vel_p2g(self):
        for i in range(self.particleNum[0]):
            bodyID = self.particle[i].bodyID
            goffset = self.body[bodyID].goffset
            grid_num = self.body[bodyID].grid_num
            grid_size = self.body[bodyID].grid_size
            xmin = self.body[bodyID].xmin

            p_pos = self.particle[i].x
            p_mass = self.particle[i].m
            p_vel = self.particle[i].v
            for j in range(self.offset[i]):
                nodeID = self.LnID[i, j]
                shape_fn = self.shape[i, j]
                v_p2g = p_vel
                if ti.static(self.velocity_proj):
                    node_pos = xmin + grid_size * ti.Vector(vectorize_id(nodeID - goffset, grid_num))
                    v_p2g += self.gradv[i] @ (node_pos - p_pos)
                self.grid[nodeID].v += shape_fn * p_mass * v_p2g
        for grid_id in self.grid:
            if self.grid[grid_id].m > self.val_lim:
                self.grid[grid_id].v /= self.grid[grid_id].m

    @ti.kernel
    def stress_update(self):
        for i in range(self.particleNum[0]):
            gradv = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
            for j in range(self.offset[i]):
                grid_id = self.LnID[i, j]
                grid_v = self.grid[grid_id].v
                gradv += grid_v.outer_product(self.dshape[i, j])
            if ti.static(config.DIM == 2):
                self.material.update_particle_volume_2D(i, gradv, self.stateVars, self.TIdt)
                self.particle[i].stress = self.material.ComputePKStress2D(
                    i, self.particle[i].stress, gradv, self.stateVars, self.TIdt
                )
            else:
                self.material.update_particle_volume(i, gradv, self.stateVars, self.TIdt)
                self.particle[i].stress = self.material.ComputePKStress(
                    i, self.particle[i].stress, gradv, self.stateVars, self.TIdt
                )

    def initial_simulation(self):
        super().initial_simulation()
        self.initial_material()
        self.compute_shapefn()
        self.mass_p2g()

    def substep(self, verbose=True):
        self.grid_reset()
        self.vel_p2g()
        self.stress_update()
        self.force_p2g(self.gravity)
        self.assemble_traction_step()
        self.grid_kinematics(self.damping)
        self.apply_neumann_step()
        self.apply_dirichlet_step()
        self.advent_particles(self.coeffPIC)

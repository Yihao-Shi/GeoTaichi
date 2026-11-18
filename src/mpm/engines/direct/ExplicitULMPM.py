import taichi as ti

import src.mpm.config as config
from src.mpm.engines.direct.ExplicitMPM import ExplicitMPM
from src.utils.MatrixFunction import matrix2vigot, vigot2matrix
from src.utils.ScalarFunction import vectorize_id


@ti.data_oriented
class ExplicitULMPM(ExplicitMPM):
    def __init__(self, bodies, dirichlet=None, neumann=None, name="case", **kwargs):
        super().__init__(bodies, dirichlet, neumann, name, configuration="UL", **kwargs)

    def initial_simulation(self):
        super().initial_simulation()
        self.initial_material()

    @ti.kernel
    def grid_reset(self):
        for i in self.grid:
            if self.grid[i].m > self.val_lim:
                self.grid[i].m = 0.0
                self.grid[i].v = ti.Vector.zero(ti.f64, config.DIM)
                self.grid[i].a = ti.Vector.zero(ti.f64, config.DIM)

    @ti.kernel
    def mass_vel_p2g(self):
        for i in range(self.particleNum[0]):
            body_id = self.particle[i].bodyID
            goffset = self.body[body_id].goffset
            grid_num = self.body[body_id].grid_num
            grid_size = self.body[body_id].grid_size
            xmin = self.body[body_id].xmin

            p_pos = self.particle[i].x
            p_mass = self.particle[i].m
            p_vel = self.particle[i].v
            for j in range(self.offset[i]):
                node_id = self.LnID[i, j]
                shape_fn = self.shape[i, j]
                v_p2g = p_vel
                if ti.static(self.velocity_proj):
                    node_pos = xmin + grid_size * ti.Vector(vectorize_id(node_id - goffset, grid_num))
                    v_p2g += self.gradv[i] @ (node_pos - p_pos)
                self.grid[node_id].m += shape_fn * p_mass
                self.grid[node_id].v += shape_fn * p_mass * v_p2g
        for grid_id in self.grid:
            if self.grid[grid_id].m > self.val_lim:
                self.grid[grid_id].v /= self.grid[grid_id].m

    @ti.kernel
    def stress_update(self):
        for i in range(self.particleNum[0]):
            previous_stress = matrix2vigot(self.particle[i].stress)
            gradv = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
            for j in range(self.offset[i]):
                grid_id = self.LnID[i, j]
                grid_v = self.grid[grid_id].v
                gradv += grid_v.outer_product(self.dshape[i, j])
            if ti.static(config.DIM == 2):
                self.particle[i].vol0 *= self.material.update_particle_volume_2d(i, gradv, self.stateVars, self.TIdt)
                self.particle[i].stress = vigot2matrix(
                    self.material.compute_stress2d(i, previous_stress, gradv, self.stateVars, self.TIdt)
                )
            else:
                self.particle[i].vol0 *= self.material.update_particle_volume(i, gradv, self.stateVars, self.TIdt)
                self.particle[i].stress = vigot2matrix(
                    self.material.compute_stress(i, previous_stress, gradv, self.stateVars, self.TIdt)
                )

    def substep(self, verbose=True):
        self.grid_reset()
        self.compute_shapefn()
        self.mass_vel_p2g()
        self.stress_update()
        self.force_p2g(self.gravity)
        self.assemble_traction_step()
        self.grid_kinematics(self.damping)
        self.apply_neumann_step()
        self.apply_dirichlet_step()
        self.advent_particles(self.coeffPIC)

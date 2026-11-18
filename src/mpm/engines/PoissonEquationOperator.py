from src.linear_solver.LinearOperator import LinearOperator
from src.mpm.SceneManager import SceneManager
from src.mpm.engines.AssembleMatrixKernel import (
    kernel_poisson_equation_cg,
    kernel_poisson_equation_cg_coupled_3d,
    kernel_poisson_equation_cg_cut_cell,
)


class PoissonEquationOperator(LinearOperator):
    def __init__(self, dimension):
        self.active_dofs = 0
        self.dimension = dimension
        self.use_cut_cell = False
        self.coupling_mode = 0
        self.use_free_surface_theta = True
        self.matvec = self.matvec_mf
        self.link_ptrs = self.link_ptrs_mf

    def link_ptrs_mf(self, scene: SceneManager, *args):
        self.ghost_cell = scene.element.ghost_cell
        self.cnum = scene.element.cnum
        self.grid_size = scene.element.grid_size
        self.igrid_size = scene.element.igrid_size
        self.cell_type = scene.element.cell.type
        self.fluid_sdf = scene.element.cell.fluid_sdf
        self.solid_sdf = scene.element.cell.solid_sdf
        self.cell_flag = scene.element.flag
        self.use_cut_cell = len(args) > 0 and args[0] is not None
        if len(args) > 1:
            self.use_free_surface_theta = bool(args[1])
        if len(args) > 2:
            self.coupling_mode = int(args[2])
        if self.coupling_mode > 0:
            self.solid_fraction = args[3]
            self.previous_solid_fraction = args[4]
            self.solid_density = args[5]
            self.mat_props = args[6]
        if self.use_cut_cell:
            self.face_fraction0 = args[0][0]
            self.face_fraction1 = args[0][1]
            self.face_fraction2 = args[0][2] if self.dimension == 3 else args[0][1]

    def update_active_dofs(self, active_dofs):
        self.active_dofs = active_dofs

    def matvec_mf(self, x, Ax):
        if self.coupling_mode > 0:
            face_fraction0 = self.face_fraction0 if self.use_cut_cell else self.solid_fraction
            face_fraction1 = self.face_fraction1 if self.use_cut_cell else self.solid_fraction
            face_fraction2 = self.face_fraction2 if self.use_cut_cell else self.solid_fraction
            kernel_poisson_equation_cg_coupled_3d(
                self.ghost_cell,
                self.cnum,
                self.igrid_size,
                self.cell_flag,
                self.cell_type,
                self.fluid_sdf,
                face_fraction0,
                face_fraction1,
                face_fraction2,
                self.solid_fraction,
                self.solid_density,
                self.mat_props,
                self.coupling_mode,
                self.use_cut_cell,
                self.use_free_surface_theta,
                x,
                Ax,
            )
        elif self.use_cut_cell:
            kernel_poisson_equation_cg_cut_cell(
                self.ghost_cell,
                self.cnum,
                self.grid_size,
                self.igrid_size,
                self.cell_flag,
                self.cell_type,
                self.fluid_sdf,
                self.solid_sdf,
                self.face_fraction0,
                self.face_fraction1,
                self.face_fraction2,
                self.use_free_surface_theta,
                x,
                Ax,
            )
        else:
            kernel_poisson_equation_cg(
                self.ghost_cell,
                self.cnum,
                self.grid_size,
                self.igrid_size,
                self.cell_flag,
                self.cell_type,
                self.fluid_sdf,
                self.use_free_surface_theta,
                x,
                Ax,
            )

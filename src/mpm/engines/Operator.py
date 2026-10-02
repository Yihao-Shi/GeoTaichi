from src.linear_solver.LinearOperator import LinearOperator
from src.mpm.SceneManager import myScene
from src.mpm.engines.AssembleMatrixKernel import (
    kernel_mass_balance_cg_poisson2D,
    kernel_moment_balance_cg,
    kernel_moment_balance_cg_2D,
    kernel_poisson_equation_cg,
    kernel_poisson_equation_cg_coupled_3d,
    kernel_poisson_equation_cg_cut_cell,
)
from src.utils.linalg import no_operation


class MomentBalanceDynamicOperator(LinearOperator):
    def __init__(self, dimension, assemble_type, solver_type=None):
        self.active_dofs = 0
        if assemble_type == "MatrixFree":
            if solver_type == "SemiImplicit" or solver_type == "SemiImplicit_u_p":
                self.matvec = self.matvecMF_poisson
            elif dimension == 2:
                self.matvec = self.matvecMF2D
            elif dimension == 3:
                self.matvec = self.matvecMF
            self.link_ptrs = self.link_ptrsMF
        elif assemble_type == "COO":
            self.matvec = None
            self.link_ptrs = no_operation
        elif assemble_type == "HashTriplet":
            self.matvec = None
            self.link_ptrs = no_operation

    def link_ptrsMF(self, scene: myScene, *args):
        self.cut_off = scene.mass_cut_off
        self.gridSum = scene.sparse_grid.node_capacity if scene.sparse_grid is not None else scene.element.gridSum
        self.flag = scene.element.flag
        self.total_nodes = scene.element.grid_nodes
        self.node = scene.node
        self.particleNum = scene.particleNum

        self.particle = scene.particle
        self.node_size = scene.element.node_size
        self.LnID = scene.element.LnID
        self.mass_matrix = args[0]
        self.local_stiffness = args[1]

    def update_active_dofs(self, active_dofs):
        self.active_dofs = active_dofs

    def matvecMF2D(self, x, Ax):
        kernel_moment_balance_cg_2D(
            self.gridSum,
            self.total_nodes,
            self.active_dofs,
            int(self.particleNum[0]),
            self.particle,
            self.node_size,
            self.LnID,
            self.flag,
            self.mass_matrix,
            self.local_stiffness,
            x,
            Ax,
        )

    def matvecMF(self, x, Ax):
        kernel_moment_balance_cg(
            self.gridSum,
            self.total_nodes,
            self.active_dofs,
            int(self.particleNum[0]),
            self.particle,
            self.node_size,
            self.LnID,
            self.flag,
            self.mass_matrix,
            self.local_stiffness,
            x,
            Ax,
        )

    def matvecMF_poisson(self, x, Ax):
        kernel_mass_balance_cg_poisson2D(
            self.total_nodes,
            self.active_dofs,
            int(self.particleNum[0]),
            self.particle,
            self.node_size,
            self.LnID,
            self.node,
            self.mass_matrix,
            self.local_stiffness,
            x,
            Ax,
        )


class PoissonEquationOperator(LinearOperator):
    def __init__(self, dimension):
        self.active_dofs = 0
        self.dimension = dimension
        self.use_cut_cell = False
        self.coupling_mode = 0
        self.use_free_surface_theta = True
        self.matvec = self.matvecMF
        self.link_ptrs = self.link_ptrsMF

    def link_ptrsMF(self, scene: myScene, *args):
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

    def matvecMF(self, x, Ax):
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

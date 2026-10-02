
from src.linear_solver.LinearOperator import LinearOperator
from src.mpm.SceneManager import myScene
from src.mpm.engines.AssembleMatrixKernel import *


class MomentBalanceDynamicOperator(LinearOperator):
    def __init__(self, dimension, assemble_type):
        self.active_dofs = 0
        if assemble_type == "MatrixFree":
            if dimension == 2:
                self.matvec = self.matvecMF2D
            elif dimension == 3:
                self.matvec = self.matvecMF
            self.link_ptrs = self.link_ptrsMF
        elif assemble_type == "CSR":
            self.matvec = self.matvecCSR
            self.link_ptrs = self.link_ptrsCSR

    def link_ptrsMF(self, scene: myScene, *args):
        self.cut_off = scene.mass_cut_off
        self.gridSum = scene.element.gridSum
        self.flag = scene.element.flag
        self.total_nodes = scene.element.grid_nodes
        self.particleNum = scene.particleNum
        
        self.particle = scene.particle
        self.node_size = scene.element.node_size
        self.LnID = scene.element.LnID
        self.mass_matrix = args[0]
        self.local_stiffness = args[1]

    def link_ptrsCSR(self, scene: myScene, *args):
        self.offset = args[0]
        self.indices = args[1]
        self.data = args[2]

    def update_active_dofs(self, active_dofs):
        self.active_dofs = active_dofs

    def matvecMF2D(self, x, Ax):
        kernel_moment_balance_cg_2D(self.gridSum, self.total_nodes, self.active_dofs, int(self.particleNum[0]), self.particle, self.node_size, self.LnID, self.flag, self.mass_matrix, self.local_stiffness, x, Ax)

    def matvecMF(self, x, Ax):
        kernel_moment_balance_cg(self.gridSum, self.total_nodes, self.active_dofs, int(self.particleNum[0]), self.particle, self.node_size, self.LnID, self.flag, self.mass_matrix, self.local_stiffness, x, Ax)

    def matvecCSR(self, x, Ax):
        kernel_moment_balance_cg_sparse_matrix(self.offset, self.indices, self.data, x, Ax)


class PoissonEquationOperator(LinearOperator):
    def __init__(self, dimension):
        self.active_dofs = 0
        self.matvec = self.matvecMF
        self.link_ptrs = self.link_ptrsMF

    def link_ptrsMF(self, scene: myScene, *args):
        self.ghost_cell = scene.element.ghost_cell
        self.cnum = scene.element.cnum
        self.grid_size = scene.element.grid_size
        self.igrid_size = scene.element.igrid_size
        self.cell_type = scene.element.cell.type
        self.cell_flag = scene.element.flag

    def update_active_dofs(self, active_dofs):
        self.active_dofs = active_dofs

    def matvecMF(self, x, Ax):
        kernel_poisson_equation_cg(self.ghost_cell, self.cnum, self.grid_size, self.igrid_size, self.cell_flag, self.cell_type, x, Ax)

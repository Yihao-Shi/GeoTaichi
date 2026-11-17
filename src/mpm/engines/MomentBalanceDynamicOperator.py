from src.linear_solver.LinearOperator import LinearOperator
from src.mpm.SceneManager import SceneManager
from src.mpm.engines.AssembleMatrixKernel import (
    kernel_mass_balance_cg_poisson2d,
    kernel_moment_balance_cg,
    kernel_moment_balance_cg_2d,
)
from src.utils.linalg import no_operation


class MomentBalanceDynamicOperator(LinearOperator):
    def __init__(self, dimension, assemble_type, solver_type=None):
        self.active_dofs = 0
        if assemble_type == "MatrixFree":
            if solver_type == "SemiImplicit" or solver_type == "SemiImplicit_u_p":
                self.matvec = self.matvec_mf_poisson
            elif dimension == 2:
                self.matvec = self.matvec_mf2d
            elif dimension == 3:
                self.matvec = self.matvec_mf
            self.link_ptrs = self.link_ptrs_mf
        elif assemble_type == "COO":
            self.matvec = None
            self.link_ptrs = no_operation
        elif assemble_type == "HashTriplet":
            self.matvec = None
            self.link_ptrs = no_operation

    def link_ptrs_mf(self, scene: SceneManager, *args):
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

    def matvec_mf2d(self, x, Ax):
        kernel_moment_balance_cg_2d(
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

    def matvec_mf(self, x, Ax):
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

    def matvec_mf_poisson(self, x, Ax):
        kernel_mass_balance_cg_poisson2d(
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

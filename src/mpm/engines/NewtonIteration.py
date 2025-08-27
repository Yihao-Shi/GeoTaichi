import taichi as ti

from src.mpm.Simulation import Simulation
from src.mpm.SceneManager import myScene
from src.mpm.engines.AssembleMatrixKernel import *
from src.mpm.engines.Operator import MomentBalanceDynamicOperator
from src.linear_solver.CompressedSparseRow import CompressedSparseRow
from src.linear_solver.MatrixFreePCG import MatrixFreePCG
from src.utils.linalg import no_operation


class MomentumConservation(object):
    def __init__(self) -> None:
        self.operator = None
        self.assemble_residual_force = None
        self.preconditioning_matrix = None
        self.assemble_mass_matrix = None
        self.assemble_stiffness_matrix = None
        self.assemble_diagonal_stiffness_matrix = None
        self.assemble_element_local_stiffnesses = None
        self.compute_residual_error = None
        self.cg = None

        self.unknow_vector = None
        self.right_hand_vector = None
        self.diag_A = None
        self.mass_matrix = None

        self.calculate_reaction_force = None
        self.accmulated_reaction_forces = None
        self.local_stiffness = None
        self.sparse_matrix = None

    def manage_function(self, sims: Simulation, scene: myScene):
        self.operator = MomentBalanceDynamicOperator(sims.dimension, sims.assemble_type)
        if sims.quasi_static:
            self.assemble_mass_matrix = self.assemble_mass_matrix_quasi_static
        else:
            self.assemble_mass_matrix = self.assemble_mass_matrix_dynamic

        self.calculate_reaction_force = no_operation
        if sims.calculate_reaction_force:
            self.calculate_reaction_force = self.calculate_reaction_forces

        if sims.dimension == 2:
            self.preconditioning_matrix = self.preconditioning_matrix_2D
            if sims.quasi_static:
                self.assemble_residual_force = self.assemble_residual_force_quasi_static_2D
            else:
                self.assemble_residual_force = self.assemble_residual_force_dynamic_2D

            self.compute_residual_error = compute_disp_error_2D
            self.assemble_element_local_stiffnesses = self.assemble_element_local_stiffness_2D
            if scene.check_elastic_material():
                self.assemble_diagonal_stiffness_matrix = assemble_elastic_diagonal_stiffness_matrix_2D
                self.assemble_stiffness_matrix = assemble_elastic_stiffness_matrix_2D
            else:
                self.assemble_diagonal_stiffness_matrix = assemble_diagonal_stiffness_matrix_2D
                self.assemble_stiffness_matrix = assemble_stiffness_matrix_2D
        elif sims.dimension == 3:
            self.preconditioning_matrix = self.preconditioning_matrix_
            if sims.quasi_static:
                self.assemble_residual_force = self.assemble_residual_force_quasi_static
            else:
                self.assemble_residual_force = self.assemble_residual_force_dynamic

            self.compute_residual_error = compute_disp_error
            self.assemble_element_local_stiffnesses = self.assemble_element_local_stiffness
            if scene.check_elastic_material():
                self.assemble_diagonal_stiffness_matrix = assemble_elastic_diagonal_stiffness_matrix
                self.assemble_stiffness_matrix = assemble_elastic_stiffness_matrix
            else:
                self.assemble_diagonal_stiffness_matrix = assemble_diagonal_stiffness_matrix
                self.assemble_stiffness_matrix = assemble_stiffness_matrix

    def manage_operator(self, scene):
        self.operator.link_ptrs(scene, self.mass_matrix, self.local_stiffness)

    def set_matrix_vector(self, dofs, sims: Simulation, scene: myScene):
        if sims.assemble_type == "CSR":
            self.sparse_matrix = CompressedSparseRow(scene.element.get_nonzero_grids_per_row() * int(sims.dof_multiplier * dofs), int(sims.dof_multiplier * dofs))
        else:
            self.cg = MatrixFreePCG(int(sims.dof_multiplier * dofs))
            self.unknow_vector = ti.field(dtype=float)                               
            self.right_hand_vector = ti.field(dtype=float)                               
            self.diag_A = ti.field(dtype=float)   
            self.mass_matrix = ti.field(dtype=float)
            if sims.calculate_reaction_force:
                self.accmulated_reaction_forces = ti.Vector.field(sims.dimension, float)
                ti.root.dense(ti.i, int(sims.dof_multiplier * dofs)).place(self.unknow_vector, self.right_hand_vector, self.diag_A, self.mass_matrix, self.accmulated_reaction_forces)
            else:
                ti.root.dense(ti.i, int(sims.dof_multiplier * dofs)).place(self.unknow_vector, self.right_hand_vector, self.diag_A, self.mass_matrix)
            if sims.assemble_type == "MatrixFree":
                self.local_stiffness = ti.field(float)
                ti.root.dense(ti.ijk, (sims.max_particle_num, scene.element.influenced_dofs, scene.element.influenced_dofs)).place(self.local_stiffness)

    def reset_matrix(self):
        unknow_reset(self.operator.active_dofs, self.unknow_vector)

    def assemble_residual_force_quasi_static(self, sims: Simulation, scene: myScene):
        kernel_assemble_residual_force_quasi_static(scene.element.gridSum, scene.mass_cut_off, scene.node, scene.element.flag, self.right_hand_vector)
        kernel_assemble_displacement_load(scene.element.gridSum, scene.mass_cut_off, int(scene.boundary.displacement_list[0]), scene.boundary.displacement_boundary, scene.node, scene.element.flag, self.right_hand_vector, self.diag_A)
    
    def assemble_residual_force_dynamic(self, sims: Simulation, scene: myScene):
        kernel_assemble_residual_force_dynamic(scene.element.gridSum, scene.mass_cut_off, sims.newmark_beta, scene.node, scene.element.flag, self.right_hand_vector, sims.dt)
        kernel_assemble_displacement_load(scene.element.gridSum, scene.mass_cut_off, int(scene.boundary.displacement_list[0]), scene.boundary.displacement_boundary, scene.node, scene.element.flag, self.right_hand_vector, self.diag_A)

    def assemble_residual_force_quasi_static_2D(self, sims: Simulation, scene: myScene):
        kernel_assemble_residual_force_quasi_static_2D(scene.element.gridSum, scene.mass_cut_off, scene.node, scene.element.flag, self.right_hand_vector)
        kernel_assemble_displacement_load(scene.element.gridSum, scene.mass_cut_off, int(scene.boundary.displacement_list[0]), scene.boundary.displacement_boundary, scene.node, scene.element.flag, self.right_hand_vector, self.diag_A)
    
    def assemble_residual_force_dynamic_2D(self, sims: Simulation, scene: myScene):
        kernel_assemble_residual_force_dynamic_2D(scene.element.gridSum, scene.mass_cut_off, sims.newmark_beta, scene.node, scene.element.flag, self.right_hand_vector, sims.dt)
        kernel_assemble_displacement_load(scene.element.gridSum, scene.mass_cut_off, int(scene.boundary.displacement_list[0]), scene.boundary.displacement_boundary, scene.node, scene.element.flag, self.right_hand_vector, self.diag_A)

    def assemble_mass_matrix_quasi_static(self, sims: Simulation, scene: myScene):
        kernel_compute_penalty_matrix(scene.element.gridSum, scene.mass_cut_off, int(scene.boundary.displacement_list[0]), scene.boundary.displacement_boundary, scene.node, scene.element.flag, self.mass_matrix)
    
    def assemble_mass_matrix_dynamic(self, sims: Simulation, scene: myScene):
        kernel_compute_mass_matrix(scene.element.gridSum, sims.newmark_beta, scene.mass_cut_off, sims.dt, scene.node, scene.element.flag, self.mass_matrix)
        kernel_compute_penalty_matrix(scene.element.gridSum, scene.mass_cut_off, int(scene.boundary.displacement_list[0]), scene.boundary.displacement_boundary, scene.node, scene.element.flag, self.mass_matrix)
    
    def preconditioning_matrix_(self, sims: Simulation, scene: myScene):
        matrix_reset_(self.operator.active_dofs, self.diag_A, self.mass_matrix, self.unknow_vector)
        kernel_preconditioning_matrix(scene.element.gridSum, scene.element.grid_nodes, int(scene.particleNum[0]), scene.particle, scene.element.node_size, 
                                      self.diag_A, scene.element.LnID, scene.element.flag, self.local_stiffness)
        
    def preconditioning_matrix_2D(self, sims: Simulation, scene: myScene):
        matrix_reset_(self.operator.active_dofs, self.diag_A, self.mass_matrix, self.unknow_vector)
        kernel_preconditioning_matrix_2D(scene.element.gridSum, scene.element.grid_nodes, int(scene.particleNum[0]), scene.particle, scene.element.node_size, 
                                         self.diag_A, scene.element.LnID, scene.element.flag, self.local_stiffness)
        
    def assemble_element_local_stiffness(self, scene: myScene):
        kernel_assemble_local_stiffness(scene.element.grid_nodes, int(scene.particleNum[0]), scene.particle, scene.element.dshape_fn, scene.element.node_size, 
                                        scene.material.stiffness_matrix, self.local_stiffness, self.assemble_stiffness_matrix)
        
    def assemble_element_local_stiffness_2D(self, scene: myScene):
        kernel_assemble_local_stiffness_2D(scene.element.grid_nodes, int(scene.particleNum[0]), scene.particle, scene.element.dshape_fn, scene.element.node_size, 
                                           scene.material.stiffness_matrix, self.local_stiffness, self.assemble_stiffness_matrix)
        
    def assemble_global_stiffness_2D(self, scene: myScene):
        pass

    def assemble_global_stiffness(self, scene: myScene):
        kernel_counting_row_offsets(scene.element.influenced_node, scene.element.gridSum, scene.mass_cut_off, scene.element.gnum, scene.node, scene.element.flag, self.sparse_matrix.offsets)
        self.pse.run(self.sparse_matrix.offsets)
        kernel_assemble_global_stiffness(scene.element.gridSum, scene.element.grid_nodes, int(scene.particleNum[0]), scene.particle, scene.element.dshape_fn, scene.element.node_size, scene.element.flag, 
                                        scene.material.stiffness_matrix, self.local_stiffness, self.assemble_stiffness_matrix)
        
    def calculate_reaction_forces(self, scene: myScene):
        kernel_calculate_reaction_forces(scene.element.gridSum, scene.mass_cut_off, int(scene.boundary.displacement_list[0]), scene.boundary.displacement_boundary, scene.node, scene.element.flag, self.accmulated_reaction_forces)

    def run(self, sims: Simulation, scene: myScene):
        self.assemble_element_local_stiffnesses(scene)
        self.preconditioning_matrix(sims, scene)
        self.assemble_residual_force(sims, scene)
        self.cg.solve(self.operator, self.right_hand_vector, self.unknow_vector, self.diag_A, self.operator.active_dofs, maxiter=self.operator.active_dofs, tol=sims.residual_tolerance)
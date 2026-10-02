import taichi as ti

from src.linear_solver.MatrixFreePCG import MatrixFreePCG
from src.linear_solver.CoordinateSparseMatrix import CoordinateSparseMatrix
from src.mpm.Simulation import Simulation
from src.mpm.SceneManager import myScene
from src.mpm.engines.Operator import MomentBalanceDynamicOperator
from src.mpm.engines.AssembleMatrixKernel import *

from src.mpm.engines.PoissonEquationKernel import (
    build_coo_jacobi_diagonal,
)


class MatrixFree(object):
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

    def _iter_twophase_materials(self, scene: myScene):
        mapping = getattr(scene.material, "mapping", None)
        if mapping is None:
            particle_num = int(scene.particleNum[0])
            for material_id in range(1, scene.material.matProps.size()):
                yield material_id, 0, particle_num, scene.material.matProps[material_id]
            return

        for material_offset in range(mapping.shape[0] - 1):
            material_id = material_offset + 1
            if material_id >= scene.material.matProps.size():
                continue
            start_index = int(mapping[material_offset])
            end_index = int(mapping[material_offset + 1])
            if end_index > start_index:
                yield material_id, start_index, end_index, scene.material.matProps[material_id]

    def manage_function(self, sims: Simulation, scene: myScene):
        self.operator = MomentBalanceDynamicOperator(sims.dimension, sims.assemble_type, sims.solver_type)
        if sims.quasi_static:
            self.assemble_mass_matrix = self.assemble_mass_matrix_quasi_static
        else:
            self.assemble_mass_matrix = self.assemble_mass_matrix_dynamic

        self.calculate_reaction_force = self.no_operation
        if sims.calculate_reaction_force:
            self.calculate_reaction_force = self.calculate_reaction_forces

        self.compute_residual_error = compute_disp_error_2D
        self.assemble_element_local_stiffnesses = self.assemble_element_local_stiffness_2D

        if sims.dimension == 2:
            if sims.quasi_static:
                self.assemble_residual_force = self.assemble_residual_force_quasi_static_2D
                self.preconditioning_matrix = self.preconditioning_matrix_quasi_static_2D
            else:
                self.assemble_residual_force = self.assemble_residual_force_dynamic_2D
                self.preconditioning_matrix = self.preconditioning_matrix_dynamic_2D

            if scene.check_elastic_material():
                if sims.assemble_type == "MatrixFree":
                    self.assemble_diagonal_stiffness_matrix = assemble_elastic_diagonal_stiffness_matrix_2D
                self.assemble_stiffness_matrix = assemble_elastic_stiffness_matrix_2D
            else:
                if sims.assemble_type == "MatrixFree":
                    self.assemble_diagonal_stiffness_matrix = assemble_diagonal_stiffness_matrix_2D
                self.assemble_stiffness_matrix = assemble_stiffness_matrix_2D
        elif sims.dimension == 3:
            if sims.quasi_static:
                self.assemble_residual_force = self.assemble_residual_force_quasi_static
                self.preconditioning_matrix = self.preconditioning_matrix_quasi_static
            else:
                self.assemble_residual_force = self.assemble_residual_force_dynamic
                self.preconditioning_matrix = self.preconditioning_matrix_dynamic

            self.compute_residual_error = compute_disp_error
            self.assemble_element_local_stiffnesses = self.assemble_element_local_stiffness
            if scene.check_elastic_material():
                if sims.assemble_type == "MatrixFree":
                    self.assemble_diagonal_stiffness_matrix = assemble_elastic_diagonal_stiffness_matrix
                self.assemble_stiffness_matrix = assemble_elastic_stiffness_matrix
            else:
                if sims.assemble_type == "MatrixFree":
                    self.assemble_diagonal_stiffness_matrix = assemble_diagonal_stiffness_matrix
                self.assemble_stiffness_matrix = assemble_stiffness_matrix

    def manage_function_poisson(self, sims: Simulation, scene: myScene):
        self.operator = MomentBalanceDynamicOperator(sims.dimension, sims.assemble_type, sims.solver_type)
        self.calculate_reaction_force = self.no_operation
        if sims.calculate_reaction_force:
            self.calculate_reaction_force = self.calculate_reaction_forces
        # self.compute_residual_error = compute_disp_error_2D
        if sims.dimension == 2:
            self.preconditioning_matrix = self.preconditioning_matrix_poisson_2D
            if sims.poisson_equation:
                if sims.is_2DAxisy:
                    self.assemble_residual_force = self.assemble_residual_poisson_2DAxisy
                else:
                    self.assemble_residual_force = self.assemble_residual_poisson_2D
                self.assemble_element_local_stiffnesses = self.assemble_element_local_stiffness_poisson2D
                if sims.pressure_stabilize == "FIC":
                    self.assemble_residual_force = self.assemble_residual_poisson_FIC_2D
                    self.assemble_element_local_stiffnesses = self.assemble_element_local_stiffness_poisson_FIC_2D
            elif sims.poisson_equation_u_p:
                self.assemble_residual_force = self.assemble_residual_poisson_2D_u_p
                self.assemble_element_local_stiffnesses = self.assemble_element_local_stiffness_poisson2D_u_p

        elif sims.dimension == 3:
            pass

    def manage_operator(self, scene):
        self.operator.link_ptrs(scene, self.mass_matrix, self.local_stiffness)

    def set_matrix_vector(self, dofs, sims: Simulation, scene: myScene):
        if sims.poisson_equation or sims.poisson_equation_u_p:
            sims.dof_multiplier = 1
            # Moving particles can activate more nodes than at initialization.
            # A pressure DOF needs both a grid/body slot and particle support.
            dofs = min(
                scene.node.shape[0] * scene.node.shape[1],
                int(scene.particleNum[0]) * scene.element.influenced_dofs,
            )
        ifnode = sims.dof_multiplier * scene.element.influenced_dofs
        if sims.assemble_type == "MatrixFree":
            self.cg = MatrixFreePCG(int(sims.dof_multiplier * dofs))
        self.unknow_vector = ti.field(dtype=float)
        self.right_hand_vector = ti.field(dtype=float)
        self.diag_A = ti.field(dtype=float)
        self.mass_matrix = ti.field(dtype=float)
        if sims.calculate_reaction_force:
            self.accmulated_reaction_forces = ti.Vector.field(sims.dimension, float)
            ti.root.dense(ti.i, int(sims.dof_multiplier * dofs)).place(
                self.unknow_vector,
                self.right_hand_vector,
                self.diag_A,
                self.mass_matrix,
                self.accmulated_reaction_forces,
            )
        else:
            ti.root.dense(ti.i, int(sims.dof_multiplier * dofs)).place(
                self.unknow_vector, self.right_hand_vector, self.diag_A, self.mass_matrix
            )
        if sims.assemble_type == "MatrixFree":
            self.local_stiffness = ti.field(float)
            ti.root.dense(
                ti.ijk, (int(scene.particleNum[0]), scene.element.influenced_dofs, scene.element.influenced_dofs)
            ).place(self.local_stiffness)
        elif sims.assemble_type == "COO":
            self.sparse_matrix = CoordinateSparseMatrix(int(ifnode * ifnode * scene.particleNum[0]), int(dofs))
        else:
            raise RuntimeError(f"SemiImplicit pressure assembly does not support assemble_type='{sims.assemble_type}'")

    def no_operation(self, scene):
        pass

    def reset_matrix(self):
        unknow_reset(self.operator.active_dofs, self.unknow_vector)

    def assemble_residual_force_quasi_static(self, sims: Simulation, scene: myScene):
        kernel_assemble_residual_force_quasi_static(scene.mass_cut_off, scene.node, self.right_hand_vector)
        kernel_assemble_displacement_load(
            scene.mass_cut_off,
            int(scene.boundary.displacement_list[0]),
            scene.boundary.displacement_boundary,
            scene.node,
            self.right_hand_vector,
            self.diag_A,
        )

    def assemble_residual_force_dynamic(self, sims: Simulation, scene: myScene):
        kernel_assemble_residual_force_dynamic(
            scene.mass_cut_off, sims.newmark_beta, scene.node, self.right_hand_vector, sims.dt
        )
        kernel_assemble_displacement_load(
            scene.mass_cut_off,
            int(scene.boundary.displacement_list[0]),
            scene.boundary.displacement_boundary,
            scene.node,
            self.right_hand_vector,
            self.diag_A,
        )

    def assemble_residual_force_quasi_static_2D(self, sims: Simulation, scene: myScene):
        kernel_assemble_residual_force_quasi_static_2D(scene.mass_cut_off, scene.node, self.right_hand_vector)
        kernel_assemble_displacement_load(
            scene.mass_cut_off,
            int(scene.boundary.displacement_list[0]),
            scene.boundary.displacement_boundary,
            scene.node,
            self.right_hand_vector,
            self.diag_A,
        )

    def assemble_residual_force_dynamic_2D(self, sims: Simulation, scene: myScene):
        kernel_assemble_residual_force_dynamic_2D(
            scene.mass_cut_off, sims.newmark_beta, scene.node, self.right_hand_vector, sims.dt
        )
        kernel_assemble_displacement_load(
            scene.mass_cut_off,
            int(scene.boundary.displacement_list[0]),
            scene.boundary.displacement_boundary,
            scene.node,
            self.right_hand_vector,
            self.diag_A,
        )

    def assemble_residual_poisson_2D(self, sims: Simulation, scene: myScene):
        kernel_assemble_residual_poisson_2D(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.node_size,
            scene.element.LnID,
            scene.node,
            scene.element.dshape_fn,
            scene.element.shape_fn,
            self.right_hand_vector,
        )
        # kernel_assemble_pressure_load(scene.mass_cut_off, int(sims.npressure[None]), sims.pressure_constraint_list, scene.node, self.right_hand_vector, self.diag_A)

    def assemble_residual_poisson_2DAxisy(self, sims: Simulation, scene: myScene):
        kernel_assemble_residual_poisson_2DAxisy(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.node_size,
            scene.element.LnID,
            scene.node,
            scene.element.dshape_fn,
            scene.element.shape_fn,
            self.right_hand_vector,
        )

    def assemble_residual_poisson_FIC_2D(self, sims: Simulation, scene: myScene):
        self.right_hand_vector.fill(0)
        for _, start_index, end_index, mat_prop in self._iter_twophase_materials(scene):
            kernel_assemble_residual_poisson_FIC_2D(
                scene.element.grid_nodes,
                start_index,
                end_index,
                scene.particle,
                scene.material.materialID,
                scene.element.node_size,
                scene.element.LnID,
                scene.node,
                mat_prop,
                scene.element.dshape_fn,
                scene.element.shape_fn,
                self.right_hand_vector,
                sims.dt,
                scene.element.grid_size,
                sims.pressure_beta,
            )

    def assemble_residual_poisson_2D_u_p(self, sims: Simulation, scene: myScene):
        self.right_hand_vector.fill(0)
        for _, start_index, end_index, mat_prop in self._iter_twophase_materials(scene):
            kernel_assemble_residual_poisson_2D_u_p(
                scene.element.grid_nodes,
                start_index,
                end_index,
                scene.particle,
                scene.material.materialID,
                scene.element.node_size,
                scene.element.LnID,
                scene.node,
                mat_prop,
                sims.gravity,
                scene.element.dshape_fn,
                scene.element.shape_fn,
                self.right_hand_vector,
                sims.dt,
                sims.pressure_beta,
            )

    def assemble_mass_matrix_quasi_static(self, sims: Simulation, scene: myScene):
        kernel_compute_penalty_matrix(
            scene.mass_cut_off,
            int(scene.boundary.displacement_list[0]),
            scene.boundary.displacement_boundary,
            scene.node,
            self.mass_matrix,
        )

    def assemble_mass_matrix_dynamic(self, sims: Simulation, scene: myScene):
        kernel_compute_mass_matrix(sims.newmark_beta, scene.mass_cut_off, sims.dt, scene.node, self.mass_matrix)
        kernel_compute_penalty_matrix(
            scene.mass_cut_off,
            int(scene.boundary.displacement_list[0]),
            scene.boundary.displacement_boundary,
            scene.node,
            self.mass_matrix,
        )

    def assemble_compute_penalty_matrix(self, sims: Simulation, scene: myScene):
        self.mass_matrix.fill(0)
        kernel_compute_penalty_matrix_poisson(
            scene.mass_cut_off, int(sims.npressure[None]), sims.pressure_constraint_list, scene.node, self.mass_matrix
        )

    def preconditioning_matrix_quasi_static(self, sims: Simulation, scene: myScene):
        matrix_reset_(self.operator.active_dofs, self.diag_A, self.unknow_vector)
        kernel_preconditioning_matrix(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.node_size,
            self.diag_A,
            scene.element.LnID,
            scene.node,
            self.local_stiffness,
        )

    def preconditioning_matrix_dynamic(self, sims: Simulation, scene: myScene):
        matrix_reset_(self.operator.active_dofs, self.diag_A, self.mass_matrix, self.unknow_vector)
        kernel_preconditioning_matrix(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.node_size,
            self.diag_A,
            scene.element.LnID,
            scene.node,
            self.local_stiffness,
        )

    def preconditioning_matrix_quasi_static_2D(self, sims: Simulation, scene: myScene):
        matrix_reset_(self.operator.active_dofs, self.diag_A, self.unknow_vector)
        kernel_preconditioning_matrix_2D(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.node_size,
            self.diag_A,
            scene.element.LnID,
            scene.node,
            self.local_stiffness,
        )

    def preconditioning_matrix_dynamic_2D(self, sims: Simulation, scene: myScene):
        matrix_reset_(self.operator.active_dofs, self.diag_A, self.mass_matrix, self.unknow_vector)
        kernel_preconditioning_matrix_2D(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.node_size,
            self.diag_A,
            scene.element.LnID,
            scene.node,
            self.local_stiffness,
        )

    def preconditioning_matrix_poisson_2D(self, sims: Simulation, scene: myScene):
        matrix_reset_(self.operator.active_dofs, self.diag_A, self.mass_matrix, self.unknow_vector)
        kernel_preconditioning_matrix_poisson2D(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.node_size,
            self.diag_A,
            scene.element.LnID,
            scene.node,
            self.local_stiffness,
        )

    def assemble_element_local_stiffness(self, scene: myScene):
        kernel_assemble_local_stiffness(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.dshape_fn,
            scene.element.node_size,
            scene.material.stiffness_matrix,
            self.local_stiffness,
            self.assemble_stiffness_matrix,
        )

    def assemble_element_local_stiffness_2D(self, scene: myScene):
        kernel_assemble_local_stiffness_2D(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.dshape_fn,
            scene.element.node_size,
            scene.material.stiffness_matrix,
            self.local_stiffness,
            self.assemble_stiffness_matrix,
        )

    def assemble_element_local_stiffness_poisson2D(self, sims: Simulation, scene: myScene):
        for _, start_index, end_index, mat_prop in self._iter_twophase_materials(scene):
            kernel_assemble_local_stiffness_poisson2D(
                scene.element.grid_nodes,
                start_index,
                end_index,
                scene.particle,
                scene.material.materialID,
                scene.element.dshape_fn,
                scene.element.node_size,
                mat_prop,
                self.local_stiffness,
                sims.dt,
                scene.element.grid_size,
                scene.material.stateVars,
            )

    def assemble_element_local_stiffness_poisson_FIC_2D(self, sims: Simulation, scene: myScene):
        for _, start_index, end_index, mat_prop in self._iter_twophase_materials(scene):
            kernel_assemble_local_stiffness_poisson_FIC_2D(
                scene.element.grid_nodes,
                start_index,
                end_index,
                scene.particle,
                scene.material.materialID,
                scene.element.shape_fn,
                scene.element.dshape_fn,
                scene.element.node_size,
                mat_prop,
                self.local_stiffness,
                sims.dt,
                scene.element.grid_size,
            )

    def assemble_element_local_stiffness_poisson2D_u_p(self, sims: Simulation, scene: myScene):
        for _, start_index, end_index, mat_prop in self._iter_twophase_materials(scene):
            kernel_assemble_local_stiffness_poisson2D_u_p(
                scene.element.grid_nodes,
                start_index,
                end_index,
                scene.particle,
                scene.material.materialID,
                scene.element.shape_fn,
                scene.element.dshape_fn,
                scene.element.node_size,
                mat_prop,
                self.local_stiffness,
                sims.dt,
            )

    def assemble_stiffness_poisson2D(self, sims: Simulation, scene: myScene):
        self.sparse_matrix.rows.fill(0)
        self.sparse_matrix.cols.fill(0)
        self.sparse_matrix.data.fill(0)
        boundary_pressure = 0.0  # Atmospheric gauge pressure, independent of cavitation.
        for _, start_index, end_index, mat_prop in self._iter_twophase_materials(scene):
            kernel_assemble_stiffness_poisson2D(
                scene.element.influenced_dofs,
                scene.element.grid_nodes,
                start_index,
                end_index,
                scene.particle,
                scene.material.materialID,
                scene.element.dshape_fn,
                scene.element.node_size,
                scene.element.LnID,
                scene.node,
                mat_prop,
                self.sparse_matrix,
                sims.dt,
                scene.element.grid_size,
                scene.material.stateVars,
                scene.mass_cut_off,
                sims.pressure_beta,
                boundary_pressure,
                self.right_hand_vector,
                sims.pressure_stabilize == "FIC",
            )

    def eliminate_pressure_increment_dirichlet(self, sims: Simulation, scene: myScene):
        boundary_pressure = 0.0
        kernel_eliminate_pressure_increment_dirichlet(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.node_size,
            scene.element.LnID,
            scene.node,
            self.local_stiffness,
            scene.mass_cut_off,
            sims.pressure_beta,
            boundary_pressure,
            self.right_hand_vector,
        )

    def calculate_reaction_forces(self, scene: myScene):
        kernel_calculate_reaction_forces(
            scene.mass_cut_off,
            int(scene.boundary.displacement_list[0]),
            scene.boundary.displacement_boundary,
            scene.node,
            self.accmulated_reaction_forces,
        )

    def run(self, sims: Simulation, scene: myScene):
        total_dof = self.checked_active_dofs()
        self.assemble_element_local_stiffnesses(scene)
        self.preconditioning_matrix(sims, scene)
        self.assemble_residual_force(sims, scene)
        solved = self.cg.solve(
            self.operator,
            self.right_hand_vector,
            self.unknow_vector,
            self.diag_A,
            total_dof,
            maxiter=10 * total_dof,
            tol=sims.residual_tolerance,
        )
        if not solved:
            raise RuntimeError(
                f"MatrixFree {type(self.cg).__name__} failed to converge after "
                f"{self.cg.last_iterations} iterations (initial residual={self.cg.last_initial_residual:.6e}, "
                f"final residual={self.cg.last_residual:.6e}, reason={self.cg.last_breakdown_reason})"
            )
        return (
            self.compute_residual_error(scene.mass_cut_off, scene.node, self.unknow_vector)
            < sims.displacement_tolerance
        )

    def checked_active_dofs(self):
        total_dof = self.operator.active_dofs
        if not 0 <= total_dof <= self.right_hand_vector.shape[0]:
            raise RuntimeError(
                f"Pressure active DOFs {total_dof} exceed allocated capacity " f"{self.right_hand_vector.shape[0]}"
            )
        return total_dof

    def run_poisson(self, sims: Simulation, scene: myScene):
        total_dof = self.checked_active_dofs()
        self.assemble_element_local_stiffnesses(sims, scene)
        self.preconditioning_matrix(sims, scene)
        self.assemble_residual_force(sims, scene)
        self.eliminate_pressure_increment_dirichlet(sims, scene)
        solved = self.cg.solve(
            self.operator,
            self.right_hand_vector,
            self.unknow_vector,
            self.diag_A,
            total_dof,
            maxiter=max(1, sims.iter_max),
            tol=1.0e-10,
            rel_tol=sims.residual_tolerance,
        )
        if not solved:
            diagonal = self.diag_A.to_numpy()[:total_dof]
            diagonal_min = diagonal.min() if diagonal.size else float("nan")
            diagonal_max = diagonal.max() if diagonal.size else float("nan")
            raise RuntimeError(
                "SemiImplicit pressure PCG failed to converge "
                f"after {self.cg.last_iterations} iterations "
                f"(initial residual={self.cg.last_initial_residual:.6e}, "
                f"final residual={self.cg.last_residual:.6e}, "
                f"reason={self.cg.last_breakdown_reason}, "
                f"Jacobi diagonal min={diagonal_min:.6e}, "
                f"max={diagonal_max:.6e}, "
                f"nonpositive={int((diagonal <= 0.0).sum())})"
            )

    def run_poisson_coo(self, sims: Simulation, scene: myScene):
        total_dof = self.checked_active_dofs()
        # Pressure increments belong to the current DOF numbering. Match the
        # MatrixFree initial guess; old increments have not been remapped.
        self.unknow_vector.fill(0.0)
        self.assemble_residual_force(sims, scene)
        self.assemble_stiffness_poisson2D(sims, scene)
        self.sparse_matrix.linear_operator.update_active_dofs(total_dof)
        self.sparse_matrix.linear_operator.update_nnz(self.sparse_matrix.nonzeros)
        build_coo_jacobi_diagonal(
            total_dof,
            self.sparse_matrix.nonzeros,
            self.sparse_matrix.rows,
            self.sparse_matrix.cols,
            self.sparse_matrix.data,
            self.diag_A,
        )
        solved = self.sparse_matrix.solve(
            self.right_hand_vector,
            self.unknow_vector,
            self.diag_A,
            tol=1.0e-10,
            maxiter=max(1, sims.iter_max),
            rel_tol=sims.residual_tolerance,
        )
        if not solved:
            cg = self.sparse_matrix.linear_solver
            raise RuntimeError(
                "SemiImplicit pressure COO-PCG failed to converge "
                f"after {cg.last_iterations} iterations "
                f"(initial residual={cg.last_initial_residual:.6e}, "
                f"final residual={cg.last_residual:.6e}, "
                f"reason={cg.last_breakdown_reason})"
            )

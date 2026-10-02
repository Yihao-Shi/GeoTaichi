import taichi as ti

from src.mpm.Simulation import Simulation
from src.mpm.SceneManager import myScene
from src.mpm.engines.AssembleMatrixKernel import *
from src.mpm.engines.Operator import MomentBalanceDynamicOperator
from src.linear_solver.CoordinateSparseMatrix import CoordinateSparseMatrix
from src.linear_solver.BuildTriplet import BuildTriplet
from src.linear_solver.MatrixFreePCG import MatrixFreePCG
from src.linear_solver.MatrixFreePBICGSTAB import MatrixFreePBICGSTAB
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
        self.assemble_global_matrix = no_operation
        self.assemble_displacement_load = no_operation
        self.apply_dirichlet_boundary = no_operation
        self.compute_residual_error = None
        self.solve_linear_system = None
        self.cg = None
        self.hash_triplet = None
        self.hash_triplet_rhs_constrained = False

        self.unknow_vector = None
        self.right_hand_vector = None
        self.diag_A = None
        self.mass_matrix = None

        self.calculate_reaction_force = None
        self.calculate_penalty_reaction_force = no_operation
        self.extract_matrix_free_reaction_force = no_operation
        self.extract_assembled_reaction_force = no_operation
        self.accmulated_reaction_forces = None
        self.local_stiffness = None
        self.sparse_matrix = None
        self.coo_base_nnz = 0
        self.reaction_rhs_vector = None
        self.reaction_vector = None
        self.hash_triplet_solver = "BiCGSTAB"
        self.hash_triplet_matrix_symmetric = False

    def manage_function(self, sims: Simulation, scene: myScene):
        self.operator = MomentBalanceDynamicOperator(sims.dimension, sims.assemble_type)
        self.assemble_displacement_load = no_operation
        self.apply_dirichlet_boundary = no_operation
        if sims.assemble_type == "MatrixFree":
            self.assemble_displacement_load = self.assemble_displacement_load_penalty
        elif sims.assemble_type == "COO":
            self.apply_dirichlet_boundary = self.apply_dirichlet_boundary_coo
        elif sims.assemble_type == "HashTriplet":
            self.apply_dirichlet_boundary = self.apply_dirichlet_boundary_hash_triplet

        if sims.quasi_static:
            self.assemble_mass_matrix = self.assemble_mass_matrix_quasi_static
        else:
            self.assemble_mass_matrix = self.assemble_mass_matrix_dynamic

        self.calculate_reaction_force = no_operation
        if scene.sparse_grid is None:
            self.calculate_penalty_reaction_force = self._calculate_penalty_reaction_force_dense
            self.extract_matrix_free_reaction_force = self._extract_matrix_free_reaction_force_dense
            self.extract_assembled_reaction_force = self._extract_assembled_reaction_force_dense
        else:
            self.calculate_penalty_reaction_force = self._calculate_penalty_reaction_force_sparse
            self.extract_matrix_free_reaction_force = self._extract_matrix_free_reaction_force_sparse
            self.extract_assembled_reaction_force = self._extract_assembled_reaction_force_sparse
        if sims.calculate_reaction_force:
            if sims.assemble_type == "MatrixFree":
                if sims.dimension == 2:
                    self.calculate_reaction_force = self.calculate_reaction_forces_matrix_free_2D
                else:
                    self.calculate_reaction_force = self.calculate_reaction_forces_matrix_free
            elif sims.dimension == 2:
                self.calculate_reaction_force = self.calculate_reaction_forces_assembled_2D
            else:
                self.calculate_reaction_force = self.calculate_reaction_forces_assembled

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
            if sims.assemble_type == "COO":
                self.assemble_global_matrix = self.assemble_coo_matrix
            elif sims.assemble_type == "HashTriplet":
                self.assemble_global_matrix = self.assemble_hash_triplet_matrix_2D
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
            if sims.assemble_type == "COO":
                self.assemble_global_matrix = self.assemble_coo_matrix
            elif sims.assemble_type == "HashTriplet":
                self.assemble_global_matrix = self.assemble_hash_triplet_matrix

        if sims.assemble_type != "MatrixFree":
            self.assemble_element_local_stiffnesses = no_operation
            self.preconditioning_matrix = (
                self.preconditioning_matrix_direct_2D if sims.dimension == 2 else self.preconditioning_matrix_direct
            )

        self.solve_linear_system = self.solve_matrix_free
        if sims.assemble_type == "COO":
            self.solve_linear_system = self.solve_coo
        elif sims.assemble_type == "HashTriplet":
            self.solve_linear_system = self.solve_hash_triplet

    def manage_operator(self, scene):
        self.operator.link_ptrs(scene, self.mass_matrix, self.local_stiffness)

    def node_stride(self, scene: myScene):
        return scene.sparse_grid.node_capacity if scene.sparse_grid is not None else scene.element.gridSum

    def sparse_grid_args(self, scene: myScene):
        sparse_grid = scene.sparse_grid
        return (
            scene.element.gnum,
            sparse_grid.block_count,
            sparse_grid.block_size,
            sparse_grid.block_volume,
            sparse_grid.block_map,
        )

    def hash_triplet_particle_capacity(self, sims: Simulation, scene: myScene):
        if getattr(scene.element, "adaptive", False):
            return max(1, int(sims.max_particle_num))
        particle_capacity = int(scene.particleNum[0])
        if particle_capacity <= 0:
            particle_capacity = int(sims.max_particle_num)
        return max(1, particle_capacity)

    def hash_triplet_support_nodes(self, sims: Simulation, scene: myScene):
        support_nodes = int(getattr(scene.element, "grid_nodes", 0))
        if support_nodes <= 0:
            influenced_node = int(getattr(scene.element, "influenced_node", 1))
            support_nodes = max(1, influenced_node**sims.dimension)
        return support_nodes

    def hash_triplet_pair_capacity(self, sims: Simulation, scene: myScene, matrix_symmetric=False):
        support_nodes = self.hash_triplet_support_nodes(sims, scene)
        off_diagonal_pairs = support_nodes * max(0, support_nodes - 1)
        if matrix_symmetric:
            off_diagonal_pairs //= 2
        return max(1, self.hash_triplet_particle_capacity(sims, scene) * off_diagonal_pairs)

    def hash_triplet_neighbor_nodes(self, sims: Simulation, scene: myScene):
        get_nonzero_grids = getattr(scene.element, "get_nonzero_grids_per_row", None)
        if get_nonzero_grids is not None:
            nonzero_grids = get_nonzero_grids()
            if nonzero_grids is not None:
                return max(1, int(nonzero_grids))
        influenced_node = int(getattr(scene.element, "influenced_node", 1))
        return max(1, (2 * influenced_node - 1) ** sims.dimension)

    def hash_triplet_nonzeros_capacity(
        self, sims: Simulation, scene: myScene, active_node_capacity, max_pairs_num, matrix_symmetric=False
    ):
        neighbor_nodes = self.hash_triplet_neighbor_nodes(sims, scene)
        off_diagonal_per_row = max(0, neighbor_nodes - 1)
        max_nonzeros = int(active_node_capacity) * off_diagonal_per_row
        if matrix_symmetric:
            max_nonzeros = (max_nonzeros + 1) // 2
        return max(1, min(int(max_pairs_num), int(max_nonzeros)))

    def use_symmetric_linear_system(self, sims: Simulation, scene: myScene):
        matrix_symmetric = getattr(sims, "hash_triplet_matrix_symmetric", None)
        if matrix_symmetric is not None:
            return bool(matrix_symmetric)
        return False

    def use_symmetric_hash_triplet_matrix(self, sims: Simulation, scene: myScene):
        return self.use_symmetric_linear_system(sims, scene)

    def use_bicg_matrix_free_solver(self, sims: Simulation):
        return sims.linear_solver == "BiCG"

    def choose_hash_triplet_solver(self, sims: Simulation, scene: myScene):
        if sims.linear_solver == "BiCG":
            return "BiCGSTAB"
        if self.use_symmetric_hash_triplet_matrix(sims, scene):
            if sims.linear_solver == "MGPCG":
                return "PCG"
            return sims.linear_solver
        return "BiCGSTAB"

    def set_matrix_vector(self, dofs, sims: Simulation, scene: myScene):
        if sims.assemble_type == "HashTriplet":
            active_node_capacity = max(1, int(sims.dof_multiplier * dofs) // sims.dimension)
            self.hash_triplet_matrix_symmetric = self.use_symmetric_hash_triplet_matrix(sims, scene)
            max_pairs_num = self.hash_triplet_pair_capacity(sims, scene, self.hash_triplet_matrix_symmetric)
            max_nonzeros = self.hash_triplet_nonzeros_capacity(
                sims, scene, active_node_capacity, max_pairs_num, self.hash_triplet_matrix_symmetric
            )
            self.hash_triplet_solver = self.choose_hash_triplet_solver(sims, scene)
            self.hash_triplet = BuildTriplet(
                sims.dimension,
                max_pairs_num,
                max_nonzeros,
                active_node_capacity,
                symmetric=False,
                solver=self.hash_triplet_solver,
                matrix_symmetric=self.hash_triplet_matrix_symmetric,
                device_reduction=True,
            )
            self.unknow_vector = ti.field(dtype=float)
            self.right_hand_vector = ti.field(dtype=float)
            self.diag_A = ti.field(dtype=float)
            self.mass_matrix = ti.field(dtype=float)
            if sims.calculate_reaction_force:
                self.accmulated_reaction_forces = ti.field(dtype=float)
                self.reaction_rhs_vector = ti.field(dtype=float)
                self.reaction_vector = ti.field(dtype=float)
                ti.root.dense(ti.i, int(sims.dof_multiplier * dofs)).place(
                    self.unknow_vector,
                    self.right_hand_vector,
                    self.diag_A,
                    self.mass_matrix,
                    self.accmulated_reaction_forces,
                    self.reaction_rhs_vector,
                    self.reaction_vector,
                )
            else:
                ti.root.dense(ti.i, int(sims.dof_multiplier * dofs)).place(
                    self.unknow_vector, self.right_hand_vector, self.diag_A, self.mass_matrix
                )
        else:
            active_dofs = int(sims.dof_multiplier * dofs)
            if sims.assemble_type == "MatrixFree":
                if self.use_bicg_matrix_free_solver(sims):
                    self.cg = MatrixFreePBICGSTAB(active_dofs)
                else:
                    self.cg = MatrixFreePCG(active_dofs)
            self.unknow_vector = ti.field(dtype=float)
            self.right_hand_vector = ti.field(dtype=float)
            self.diag_A = ti.field(dtype=float)
            self.mass_matrix = ti.field(dtype=float)
            if sims.calculate_reaction_force:
                self.accmulated_reaction_forces = ti.field(dtype=float)
                self.reaction_rhs_vector = ti.field(dtype=float)
                self.reaction_vector = ti.field(dtype=float)
                ti.root.dense(ti.i, active_dofs).place(
                    self.unknow_vector,
                    self.right_hand_vector,
                    self.diag_A,
                    self.mass_matrix,
                    self.accmulated_reaction_forces,
                    self.reaction_rhs_vector,
                    self.reaction_vector,
                )
            else:
                ti.root.dense(ti.i, active_dofs).place(
                    self.unknow_vector, self.right_hand_vector, self.diag_A, self.mass_matrix
                )
            if sims.assemble_type == "MatrixFree":
                self.local_stiffness = ti.field(float)
                ti.root.dense(
                    ti.ijk, (sims.max_particle_num, scene.element.influenced_dofs, scene.element.influenced_dofs)
                ).place(self.local_stiffness)
            if sims.assemble_type == "COO":
                active_node_capacity = max(1, active_dofs // sims.dimension)
                neighbor_nodes = self.hash_triplet_neighbor_nodes(sims, scene)
                stiffness_triplets = active_node_capacity * neighbor_nodes * sims.dimension * sims.dimension
                constraint_triplets = int(getattr(sims, "ndisplacement", 0))
                self.sparse_matrix = CoordinateSparseMatrix(
                    stiffness_triplets + constraint_triplets,
                    active_dofs,
                    symmetry=self.use_symmetric_linear_system(sims, scene),
                )

    def reset_matrix(self):
        unknow_reset(self.operator.active_dofs, self.unknow_vector)

    def assemble_residual_force_quasi_static(self, sims: Simulation, scene: myScene):
        kernel_assemble_residual_force_quasi_static(
            self.node_stride(scene), scene.mass_cut_off, scene.node, scene.element.flag, self.right_hand_vector
        )
        self.assemble_displacement_load(sims, scene)

    def assemble_residual_force_dynamic(self, sims: Simulation, scene: myScene):
        kernel_assemble_residual_force_dynamic(
            self.node_stride(scene),
            scene.mass_cut_off,
            sims.newmark_beta,
            scene.node,
            scene.element.flag,
            self.right_hand_vector,
            sims.dt,
        )
        self.assemble_displacement_load(sims, scene)

    def assemble_residual_force_quasi_static_2D(self, sims: Simulation, scene: myScene):
        kernel_assemble_residual_force_quasi_static_2D(
            self.node_stride(scene), scene.mass_cut_off, scene.node, scene.element.flag, self.right_hand_vector
        )
        self.assemble_displacement_load(sims, scene)

    def assemble_residual_force_dynamic_2D(self, sims: Simulation, scene: myScene):
        kernel_assemble_residual_force_dynamic_2D(
            self.node_stride(scene),
            scene.mass_cut_off,
            sims.newmark_beta,
            scene.node,
            scene.element.flag,
            self.right_hand_vector,
            sims.dt,
        )
        self.assemble_displacement_load(sims, scene)

    def assemble_displacement_load_penalty(self, sims: Simulation, scene: myScene):
        if self.reaction_rhs_vector is not None:
            kernel_copy_scalar_field(self.operator.active_dofs, self.right_hand_vector, self.reaction_rhs_vector)
        displacement_num = int(scene.boundary.displacement_list[0])
        if scene.sparse_grid is not None:
            kernel_assemble_displacement_load_sparse(
                self.node_stride(scene),
                scene.mass_cut_off,
                displacement_num,
                scene.boundary.displacement_boundary,
                scene.node,
                scene.element.flag,
                self.right_hand_vector,
                self.diag_A,
                *self.sparse_grid_args(scene),
            )
        else:
            kernel_assemble_displacement_load(
                scene.element.gridSum,
                scene.mass_cut_off,
                displacement_num,
                scene.boundary.displacement_boundary,
                scene.node,
                scene.element.flag,
                self.right_hand_vector,
                self.diag_A,
            )

    def assemble_mass_matrix_quasi_static(self, sims: Simulation, scene: myScene):
        if sims.assemble_type == "MatrixFree":
            displacement_num = int(scene.boundary.displacement_list[0])
            if scene.sparse_grid is not None:
                kernel_compute_penalty_matrix_sparse(
                    self.node_stride(scene),
                    scene.mass_cut_off,
                    displacement_num,
                    scene.boundary.displacement_boundary,
                    scene.node,
                    scene.element.flag,
                    self.mass_matrix,
                    *self.sparse_grid_args(scene),
                )
            else:
                kernel_compute_penalty_matrix(
                    scene.element.gridSum,
                    scene.mass_cut_off,
                    displacement_num,
                    scene.boundary.displacement_boundary,
                    scene.node,
                    scene.element.flag,
                    self.mass_matrix,
                )

    def assemble_mass_matrix_dynamic(self, sims: Simulation, scene: myScene):
        kernel_compute_mass_matrix(
            self.node_stride(scene),
            sims.newmark_beta,
            scene.mass_cut_off,
            sims.dt,
            scene.node,
            scene.element.flag,
            self.mass_matrix,
        )
        if sims.assemble_type == "MatrixFree":
            displacement_num = int(scene.boundary.displacement_list[0])
            if scene.sparse_grid is not None:
                kernel_compute_penalty_matrix_sparse(
                    self.node_stride(scene),
                    scene.mass_cut_off,
                    displacement_num,
                    scene.boundary.displacement_boundary,
                    scene.node,
                    scene.element.flag,
                    self.mass_matrix,
                    *self.sparse_grid_args(scene),
                )
            else:
                kernel_compute_penalty_matrix(
                    scene.element.gridSum,
                    scene.mass_cut_off,
                    displacement_num,
                    scene.boundary.displacement_boundary,
                    scene.node,
                    scene.element.flag,
                    self.mass_matrix,
                )

    def preconditioning_matrix_(self, sims: Simulation, scene: myScene):
        sims.timer.begin("Precondition reset")
        matrix_reset_(self.operator.active_dofs, self.diag_A, self.mass_matrix, self.unknow_vector)
        sims.timer.end("Precondition reset")
        sims.timer.begin("Precondition kernel")
        kernel_preconditioning_matrix(
            self.node_stride(scene),
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.node_size,
            self.diag_A,
            scene.element.LnID,
            scene.element.flag,
            self.local_stiffness,
        )
        sims.timer.end("Precondition kernel")

    def preconditioning_matrix_2D(self, sims: Simulation, scene: myScene):
        sims.timer.begin("Precondition reset")
        matrix_reset_(self.operator.active_dofs, self.diag_A, self.mass_matrix, self.unknow_vector)
        sims.timer.end("Precondition reset")
        sims.timer.begin("Precondition kernel")
        kernel_preconditioning_matrix_2D(
            self.node_stride(scene),
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.node_size,
            self.diag_A,
            scene.element.LnID,
            scene.element.flag,
            self.local_stiffness,
        )
        sims.timer.end("Precondition kernel")

    def preconditioning_matrix_direct(self, sims: Simulation, scene: myScene):
        sims.timer.begin("Precondition reset")
        matrix_reset_(self.operator.active_dofs, self.diag_A, self.mass_matrix, self.unknow_vector)
        sims.timer.end("Precondition reset")
        sims.timer.begin("Precondition kernel")
        kernel_preconditioning_matrix_direct(
            self.node_stride(scene),
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.dshape_fn,
            scene.element.node_size,
            self.diag_A,
            scene.element.LnID,
            scene.element.flag,
            scene.material.stiffness_matrix,
            self.assemble_stiffness_matrix,
        )
        sims.timer.end("Precondition kernel")

    def preconditioning_matrix_direct_2D(self, sims: Simulation, scene: myScene):
        sims.timer.begin("Precondition reset")
        matrix_reset_(self.operator.active_dofs, self.diag_A, self.mass_matrix, self.unknow_vector)
        sims.timer.end("Precondition reset")
        sims.timer.begin("Precondition kernel")
        kernel_preconditioning_matrix_direct_2D(
            self.node_stride(scene),
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.dshape_fn,
            scene.element.node_size,
            self.diag_A,
            scene.element.LnID,
            scene.element.flag,
            scene.material.stiffness_matrix,
            self.assemble_stiffness_matrix,
        )
        sims.timer.end("Precondition kernel")

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

    def assemble_coo_matrix(self, sims: Simulation, scene: myScene):
        sims.timer.begin("COO reset")
        self.sparse_matrix.reset()
        sims.timer.end("COO reset")
        active_dofs = self.operator.active_dofs
        active_nodes = active_dofs // sims.dimension
        particle_num = int(scene.particleNum[0])
        influenced_node = int(getattr(scene.element, "influenced_node", 1))
        neighbor_nodes = self.hash_triplet_neighbor_nodes(sims, scene)
        sims.timer.begin("COO assemble mass")
        kernel_assemble_coo_mass(
            active_dofs,
            influenced_node,
            self.mass_matrix,
            self.sparse_matrix.rows,
            self.sparse_matrix.cols,
            self.sparse_matrix.data,
        )
        sims.timer.end("COO assemble mass")
        sims.timer.begin("COO assemble stiffness")
        if scene.sparse_grid is not None:
            sparse_grid = scene.sparse_grid
            kernel_assemble_coo_stiffness_sparse(
                self.node_stride(scene),
                scene.element.grid_nodes,
                particle_num,
                influenced_node,
                scene.particle,
                scene.element.dshape_fn,
                scene.element.node_size,
                scene.element.LnID,
                scene.element.flag,
                scene.material.stiffness_matrix,
                self.assemble_stiffness_matrix,
                self.sparse_matrix.rows,
                self.sparse_matrix.cols,
                self.sparse_matrix.data,
                sparse_grid.block_count,
                sparse_grid.block_size,
                sparse_grid.block_volume,
                sparse_grid.active_block_ids,
            )
        else:
            kernel_assemble_coo_stiffness(
                self.node_stride(scene),
                scene.element.grid_nodes,
                particle_num,
                influenced_node,
                scene.element.gnum,
                scene.particle,
                scene.element.dshape_fn,
                scene.element.node_size,
                scene.element.LnID,
                scene.element.flag,
                scene.material.stiffness_matrix,
                self.assemble_stiffness_matrix,
                self.sparse_matrix.rows,
                self.sparse_matrix.cols,
                self.sparse_matrix.data,
            )
        sims.timer.end("COO assemble stiffness")
        nnz = active_nodes * neighbor_nodes * sims.dimension * sims.dimension
        self.coo_base_nnz = nnz
        self.sparse_matrix.linear_operator.update_active_dofs(active_dofs)
        self.sparse_matrix.linear_operator.update_nnz(nnz)

    def assemble_hash_triplet_matrix_2D(self, sims: Simulation, scene: myScene):
        sims.timer.begin("HashTriplet reset")
        self.hash_triplet.reset_system()
        sims.timer.end("HashTriplet reset")
        self.hash_triplet_rhs_constrained = False
        sims.timer.begin("HashTriplet assemble mass")
        kernel_assemble_hash_triplet_mass(self.operator.active_dofs, self.mass_matrix, self.hash_triplet)
        sims.timer.end("HashTriplet assemble mass")
        sims.timer.begin("HashTriplet assemble stiffness")
        kernel_assemble_hash_triplet_stiffness_2D(
            self.node_stride(scene),
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.node_size,
            scene.element.LnID,
            scene.element.dshape_fn,
            scene.element.flag,
            scene.material.stiffness_matrix,
            self.assemble_stiffness_matrix,
            self.hash_triplet,
        )
        sims.timer.end("HashTriplet assemble stiffness")
        sims.timer.begin("HashTriplet finalize")
        self.hash_triplet.finalize_taichi_assembly()
        sims.timer.end("HashTriplet finalize")

    def assemble_hash_triplet_matrix(self, sims: Simulation, scene: myScene):
        sims.timer.begin("HashTriplet reset")
        self.hash_triplet.reset_system()
        sims.timer.end("HashTriplet reset")
        self.hash_triplet_rhs_constrained = False
        sims.timer.begin("HashTriplet assemble mass")
        kernel_assemble_hash_triplet_mass(self.operator.active_dofs, self.mass_matrix, self.hash_triplet)
        sims.timer.end("HashTriplet assemble mass")
        sims.timer.begin("HashTriplet assemble stiffness")
        kernel_assemble_hash_triplet_stiffness(
            self.node_stride(scene),
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.node_size,
            scene.element.LnID,
            scene.element.dshape_fn,
            scene.element.flag,
            scene.material.stiffness_matrix,
            self.assemble_stiffness_matrix,
            self.hash_triplet,
        )
        sims.timer.end("HashTriplet assemble stiffness")
        sims.timer.begin("HashTriplet finalize")
        self.hash_triplet.finalize_taichi_assembly()
        sims.timer.end("HashTriplet finalize")

    def calculate_reaction_forces_penalty(self, scene: myScene):
        self.accmulated_reaction_forces.fill(0.0)
        displacement_num = int(scene.boundary.displacement_list[0])
        self.calculate_penalty_reaction_force(scene, displacement_num)

    def _calculate_penalty_reaction_force_dense(self, scene: myScene, displacement_num):
        kernel_calculate_reaction_forces(
            scene.element.gridSum,
            scene.mass_cut_off,
            displacement_num,
            scene.boundary.displacement_boundary,
            scene.node,
            scene.element.flag,
            self.accmulated_reaction_forces,
        )

    def _calculate_penalty_reaction_force_sparse(self, scene: myScene, displacement_num):
        kernel_calculate_reaction_forces_sparse(
            self.node_stride(scene),
            scene.mass_cut_off,
            displacement_num,
            scene.boundary.displacement_boundary,
            scene.node,
            scene.element.flag,
            self.accmulated_reaction_forces,
            *self.sparse_grid_args(scene),
        )

    def _extract_matrix_free_reaction_force_dense(self, scene: myScene, displacement_num):
        kernel_extract_matrix_free_reaction_forces(
            scene.element.gridSum,
            scene.mass_cut_off,
            displacement_num,
            scene.boundary.displacement_boundary,
            scene.node,
            scene.element.flag,
            self.reaction_vector,
            self.reaction_rhs_vector,
            self.unknow_vector,
            self.accmulated_reaction_forces,
        )

    def _extract_matrix_free_reaction_force_sparse(self, scene: myScene, displacement_num):
        kernel_extract_matrix_free_reaction_forces_sparse(
            self.node_stride(scene),
            scene.mass_cut_off,
            displacement_num,
            scene.boundary.displacement_boundary,
            scene.node,
            scene.element.flag,
            self.reaction_vector,
            self.reaction_rhs_vector,
            self.unknow_vector,
            self.accmulated_reaction_forces,
            *self.sparse_grid_args(scene),
        )

    def _extract_assembled_reaction_force_dense(self, scene: myScene, displacement_num):
        kernel_extract_assembled_reaction_forces(
            scene.element.gridSum,
            scene.mass_cut_off,
            displacement_num,
            scene.boundary.displacement_boundary,
            scene.node,
            scene.element.flag,
            self.reaction_vector,
            self.reaction_rhs_vector,
            self.accmulated_reaction_forces,
        )

    def _extract_assembled_reaction_force_sparse(self, scene: myScene, displacement_num):
        kernel_extract_assembled_reaction_forces_sparse(
            self.node_stride(scene),
            scene.mass_cut_off,
            displacement_num,
            scene.boundary.displacement_boundary,
            scene.node,
            scene.element.flag,
            self.reaction_vector,
            self.reaction_rhs_vector,
            self.accmulated_reaction_forces,
            *self.sparse_grid_args(scene),
        )

    def calculate_reaction_forces_matrix_free_2D(self, scene: myScene):
        self.accmulated_reaction_forces.fill(0.0)
        kernel_moment_balance_cg_2D(
            self.node_stride(scene),
            scene.element.grid_nodes,
            self.operator.active_dofs,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.node_size,
            scene.element.LnID,
            scene.element.flag,
            self.mass_matrix,
            self.local_stiffness,
            self.unknow_vector,
            self.reaction_vector,
        )
        displacement_num = int(scene.boundary.displacement_list[0])
        self.extract_matrix_free_reaction_force(scene, displacement_num)

    def calculate_reaction_forces_matrix_free(self, scene: myScene):
        self.accmulated_reaction_forces.fill(0.0)
        kernel_moment_balance_cg(
            self.node_stride(scene),
            scene.element.grid_nodes,
            self.operator.active_dofs,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.node_size,
            scene.element.LnID,
            scene.element.flag,
            self.mass_matrix,
            self.local_stiffness,
            self.unknow_vector,
            self.reaction_vector,
        )
        displacement_num = int(scene.boundary.displacement_list[0])
        self.extract_matrix_free_reaction_force(scene, displacement_num)

    def calculate_reaction_forces_assembled_2D(self, scene: myScene):
        self.accmulated_reaction_forces.fill(0.0)
        kernel_moment_balance_direct_2D(
            self.node_stride(scene),
            scene.element.grid_nodes,
            self.operator.active_dofs,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.dshape_fn,
            scene.element.node_size,
            scene.element.LnID,
            scene.element.flag,
            scene.material.stiffness_matrix,
            self.assemble_stiffness_matrix,
            self.mass_matrix,
            self.unknow_vector,
            self.reaction_vector,
        )
        displacement_num = int(scene.boundary.displacement_list[0])
        self.extract_assembled_reaction_force(scene, displacement_num)

    def calculate_reaction_forces_assembled(self, scene: myScene):
        self.accmulated_reaction_forces.fill(0.0)
        kernel_moment_balance_direct(
            self.node_stride(scene),
            scene.element.grid_nodes,
            self.operator.active_dofs,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.dshape_fn,
            scene.element.node_size,
            scene.element.LnID,
            scene.element.flag,
            scene.material.stiffness_matrix,
            self.assemble_stiffness_matrix,
            self.mass_matrix,
            self.unknow_vector,
            self.reaction_vector,
        )
        displacement_num = int(scene.boundary.displacement_list[0])
        self.extract_assembled_reaction_force(scene, displacement_num)

    def apply_dirichlet_boundary_coo(self, sims: Simulation, scene: myScene):
        displacement_num = int(scene.boundary.displacement_list[0])
        if displacement_num <= 0:
            return
        if self.reaction_rhs_vector is not None:
            sims.timer.begin("COO reaction rhs copy")
            kernel_copy_scalar_field(self.operator.active_dofs, self.right_hand_vector, self.reaction_rhs_vector)
            sims.timer.end("COO reaction rhs copy")
        sims.timer.begin("COO apply Dirichlet")
        if scene.sparse_grid is not None:
            kernel_apply_dirichlet_coo_sparse(
                self.node_stride(scene),
                scene.mass_cut_off,
                displacement_num,
                self.coo_base_nnz,
                scene.boundary.displacement_boundary,
                scene.node,
                scene.element.flag,
                self.sparse_matrix.rows,
                self.sparse_matrix.cols,
                self.sparse_matrix.data,
                self.right_hand_vector,
                self.diag_A,
                *self.sparse_grid_args(scene),
            )
        else:
            kernel_apply_dirichlet_coo(
                scene.element.gridSum,
                scene.mass_cut_off,
                displacement_num,
                self.coo_base_nnz,
                scene.boundary.displacement_boundary,
                scene.node,
                scene.element.flag,
                self.sparse_matrix.rows,
                self.sparse_matrix.cols,
                self.sparse_matrix.data,
                self.right_hand_vector,
                self.diag_A,
            )
        sims.timer.end("COO apply Dirichlet")
        self.sparse_matrix.linear_operator.update_nnz(self.coo_base_nnz + displacement_num)

    def apply_dirichlet_boundary_hash_triplet(self, sims: Simulation, scene: myScene):
        displacement_num = int(scene.boundary.displacement_list[0])
        if self.reaction_rhs_vector is not None:
            sims.timer.begin("HashTriplet reaction rhs copy")
            kernel_copy_scalar_field(self.operator.active_dofs, self.right_hand_vector, self.reaction_rhs_vector)
            sims.timer.end("HashTriplet reaction rhs copy")
        sims.timer.begin("HashTriplet rhs copy")
        kernel_copy_flat_rhs_to_hash_triplet(self.operator.active_dofs, self.right_hand_vector, self.hash_triplet.rhs)
        sims.timer.end("HashTriplet rhs copy")
        if displacement_num > 0:
            active_nodes = self.operator.active_dofs // sims.dimension
            sims.timer.begin("HashTriplet apply Dirichlet")
            if scene.sparse_grid is not None:
                kernel_apply_dirichlet_hash_triplet_sparse(
                    self.node_stride(scene),
                    scene.mass_cut_off,
                    displacement_num,
                    active_nodes,
                    scene.boundary.displacement_boundary,
                    scene.node,
                    scene.element.flag,
                    self.hash_triplet,
                    self.diag_A,
                    *self.sparse_grid_args(scene),
                )
            else:
                kernel_apply_dirichlet_hash_triplet(
                    scene.element.gridSum,
                    scene.mass_cut_off,
                    displacement_num,
                    active_nodes,
                    scene.boundary.displacement_boundary,
                    scene.node,
                    scene.element.flag,
                    self.hash_triplet,
                    self.diag_A,
                )
            sims.timer.end("HashTriplet apply Dirichlet")
        self.hash_triplet_rhs_constrained = True

    def solve_matrix_free(self, sims: Simulation, scene: myScene):
        result = self.cg.solve(
            self.operator,
            self.right_hand_vector,
            self.unknow_vector,
            self.diag_A,
            self.operator.active_dofs,
            maxiter=max(1, 5 * self.operator.active_dofs),
            tol=sims.residual_tolerance,
            rel_tol=sims.linear_solver_relative_tolerance,
        )
        if result is False:
            raise RuntimeError(
                f"MatrixFree {type(self.cg).__name__} failed to converge. "
                f"Initial residual is {self.cg.last_initial_residual}, final residual is {self.cg.last_residual}, "
                f"iterations = {self.cg.last_iterations}."
            )

    def solve_coo(self, sims: Simulation, scene: myScene):
        sims.timer.begin("COO solve")
        result = self.sparse_matrix.solve(
            self.right_hand_vector,
            self.unknow_vector,
            self.diag_A,
            maxiter=max(1, self.operator.active_dofs),
            tol=sims.residual_tolerance,
        )
        sims.timer.end("COO solve")
        if not result:
            solver = self.sparse_matrix.linear_solver
            raise RuntimeError(
                f"COO {type(solver).__name__} failed to converge. "
                f"Initial residual is {solver.last_initial_residual}, final residual is {solver.last_residual}, "
                f"iterations = {solver.last_iterations}, reason = {solver.last_breakdown_reason}."
            )

    def solve_hash_triplet(self, sims: Simulation, scene: myScene):
        active_nodes = self.operator.active_dofs // sims.dimension
        if not self.hash_triplet_rhs_constrained:
            sims.timer.begin("HashTriplet rhs copy")
            kernel_copy_flat_rhs_to_hash_triplet(
                self.operator.active_dofs, self.right_hand_vector, self.hash_triplet.rhs
            )
            sims.timer.end("HashTriplet rhs copy")
        sims.timer.begin("HashTriplet solve")
        result = self.hash_triplet.solve(
            active_nodes=active_nodes,
            tol=sims.residual_tolerance,
            rel_tol=sims.linear_solver_relative_tolerance,
            maxiter=max(1, 5 * self.operator.active_dofs),
            return_solution=False,
        )
        sims.timer.end("HashTriplet solve")
        if not result["converged"]:
            raise RuntimeError(
                f"HashTriplet {self.hash_triplet_solver} failed to converge. Final residual is {result['residual']}"
            )
        sims.timer.begin("HashTriplet solution copy")
        kernel_copy_hash_triplet_solution_to_flat(self.operator.active_dofs, self.hash_triplet.x, self.unknow_vector)
        sims.timer.end("HashTriplet solution copy")
        self.hash_triplet_rhs_constrained = False

    def run(self, sims: Simulation, scene: myScene):
        sims.timer.begin("Assemble matrix")
        self.assemble_element_local_stiffnesses(scene)
        if sims.assemble_type == "MatrixFree" and sims.symmetrize_matrix_free_tangent:
            kernel_symmetrize_local_stiffness(
                int(scene.particleNum[0]), scene.element.influenced_dofs, self.local_stiffness
            )
        sims.timer.end("Assemble matrix")
        sims.timer.begin("Preconditioning matrix")
        self.preconditioning_matrix(sims, scene)
        sims.timer.end("Preconditioning matrix")
        sims.timer.begin("Assemble global matrix")
        self.assemble_global_matrix(sims, scene)
        sims.timer.end("Assemble global matrix")
        sims.timer.begin("Assemble residual")
        self.assemble_residual_force(sims, scene)
        sims.timer.end("Assemble residual")
        sims.timer.begin("Apply Dirichlet boundary")
        self.apply_dirichlet_boundary(sims, scene)
        sims.timer.end("Apply Dirichlet boundary")
        sims.timer.begin("Solve linear system")
        self.solve_linear_system(sims, scene)
        sims.timer.end("Solve linear system")

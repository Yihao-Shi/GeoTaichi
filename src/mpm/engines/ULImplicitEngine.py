import math

from src.mpm.engines.NewtonIteration import MomentumConservation
from src.mpm.engines.Engine import Engine
from src.mpm.engines.EngineKernel import *
from src.mpm.SceneManager import myScene
from src.mpm.Simulation import Simulation
from src.mpm.SpatialHashGrid import SpatialHashGrid
from src.mpm.engines.FreeSurfaceDetection import *
from src.utils.linalg import no_operation


class ImplicitEngine(Engine):
    def __init__(self, sims) -> None:
        self.update_nodal_displacements = None
        self.compute_internal_forces = None
        self.update_adaptive_grid = no_operation
        super().__init__(sims)

    def manage_function(self, sims: Simulation):
        super().manage_function(sims)
        if sims.dimension == 2:
            self.update_nodal_displacements = self.update_nodal_displacement_2D
            self.compute_internal_forces = self.compute_internal_force_2D
            if sims.use_elastic_matrix_free_tangent:
                self.compute_stress_strains = self.compute_stress_strain_elastic_tangent_2D
        elif sims.dimension == 3:
            self.update_nodal_displacements = self.update_nodal_displacement
            self.compute_internal_forces = self.compute_internal_force
            if sims.use_elastic_matrix_free_tangent:
                self.compute_stress_strains = self.compute_stress_strain_elastic_tangent

    def choose_engine(self, sims: Simulation):
        if sims.integration_scheme == "Newmark":
            self.compute = self.newmark_integrate
        else:
            raise ValueError(f"The integration scheme {sims.integration_scheme} is not supported yet")

    def choose_boundary_constraints(self, sims: Simulation, scene: myScene):
        self.apply_traction_constraints = no_operation
        self.apply_particle_traction_constraints = no_operation
        self.update_adaptive_grid = no_operation
        if scene.boundary.traction_list[None] > 0:
            self.apply_traction_constraints = self.traction_constraints
        if int(scene.boundary.ptraction_list[0]) > 0:
            self.apply_particle_traction_constraints = self.particle_traction_constraints
        if getattr(scene.element, "adaptive", False):
            if not getattr(scene.element, "strong_hanging_constraint", False):
                raise RuntimeError("Implicit solid AdaptiveGrid requires " "HangingConstraintMode=ShapeFunction.")
            self.update_adaptive_grid = self.update_grid_refinement

    def reset_iterative_grid_message(self, scene: myScene):
        grid_internal_force_reset(scene.mass_cut_off, scene.node)

    def compute_nodal_mass(self, sims: Simulation, scene: myScene):
        kernel_mass_p2g(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.element.gnum,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )

    def compute_nodal_kinematics(self, sims: Simulation, scene: myScene):
        kernel_mass_momentum_acceleration_force_ip2g(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            sims.gravity,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )

    def compute_grid_velcity(self, sims: Simulation, scene: myScene):
        kernel_compute_grid_velocity_acceleration(scene.mass_cut_off, scene.node)

    def compute_nodal_kinematics_newmark(self, sims: Simulation, scene: myScene):
        kernel_compute_nodal_kinematics_newmark(
            sims.newmark_beta, sims.newmark_gamma, scene.mass_cut_off, scene.node, sims.dt
        )

    def compute_internal_force(self, sims: Simulation, scene: myScene):
        kernel_internal_force_p2g(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.dshape_fn,
            scene.element.node_size,
        )

    def compute_internal_force_2D(self, sims: Simulation, scene: myScene):
        kernel_internal_force_p2g_2D(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.dshape_fn,
            scene.element.node_size,
        )

    def update_nodal_displacement(self, scene: myScene):
        kernel_update_nodal_disp(
            self.iterator.node_stride(scene),
            scene.mass_cut_off,
            scene.node,
            scene.element.flag,
            self.iterator.unknow_vector,
        )

    def update_nodal_displacement_2D(self, scene: myScene):
        kernel_update_nodal_disp_2D(
            self.iterator.node_stride(scene),
            scene.mass_cut_off,
            scene.node,
            scene.element.flag,
            self.iterator.unknow_vector,
        )

    def compute_stress_strain(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_compute_stress_strain_newmark(
                sims.dt,
                start_index,
                end_index,
                scene.particle,
                scene.material.materialID,
                scene.material.matProps[materialID + 1],
                scene.material.stateVars,
                scene.material.stiffness_matrix,
            )

    def compute_stress_strain_elastic_tangent(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_compute_stress_strain_newmark_elastic_tangent(
                sims.dt,
                start_index,
                end_index,
                scene.particle,
                scene.material.materialID,
                scene.material.matProps[materialID + 1],
                scene.material.stateVars,
                scene.material.stiffness_matrix,
            )

    def compute_stress_strain_2D(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_compute_stress_strain_newmark_2D(
                sims.dt,
                start_index,
                end_index,
                scene.particle,
                scene.material.materialID,
                scene.material.matProps[materialID + 1],
                scene.material.stateVars,
                scene.material.stiffness_matrix,
            )

    def compute_stress_strain_elastic_tangent_2D(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_compute_stress_strain_newmark_elastic_tangent_2D(
                sims.dt,
                start_index,
                end_index,
                scene.particle,
                scene.material.materialID,
                scene.material.matProps[materialID + 1],
                scene.material.stateVars,
                scene.material.stiffness_matrix,
            )

    def update_velocity_gradient_2D(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_update_displacement_gradient_2D(
                scene.element.grid_nodes,
                start_index,
                end_index,
                sims.dt,
                scene.node,
                scene.particle,
                scene.material.materialID,
                scene.material.matProps[materialID + 1],
                scene.material.stateVars,
                scene.element.LnID,
                scene.element.dshape_fn,
                scene.element.node_size,
            )

    def update_velocity_gradient(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_update_displacement_gradient(
                scene.element.grid_nodes,
                start_index,
                end_index,
                sims.dt,
                scene.node,
                scene.particle,
                scene.material.materialID,
                scene.material.matProps[materialID + 1],
                scene.material.stateVars,
                scene.element.LnID,
                scene.element.dshape_fn,
                scene.element.node_size,
            )

    def update_velocity_gradient_affine_2D(self, sims: Simulation, scene: myScene):
        kernel_update_displacement_gradient_affine_2D(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.element.gnum,
            scene.element.grid_size,
            sims.dt,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )

    def update_velocity_gradient_affine(self, sims: Simulation, scene: myScene):
        kernel_update_displacement_gradient_affine(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.element.gnum,
            scene.element.grid_size,
            sims.dt,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )

    def update_velocity_gradient_bbar_2D(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_update_displacement_gradient_bbar_2D(
                scene.element.grid_nodes,
                start_index,
                end_index,
                sims.dt,
                scene.node,
                scene.particle,
                scene.material.materialID,
                scene.material.matProps[materialID + 1],
                scene.material.stateVars,
                scene.element.LnID,
                scene.element.dshape_fn,
                scene.element.dshape_fnc,
                scene.element.node_size,
            )

    def update_velocity_gradient_bbar(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_update_displacement_gradient_bbar(
                scene.element.grid_nodes,
                start_index,
                end_index,
                sims.dt,
                scene.node,
                scene.particle,
                scene.material.materialID,
                scene.material.matProps[materialID + 1],
                scene.material.stateVars,
                scene.element.LnID,
                scene.element.dshape_fn,
                scene.element.dshape_fnc,
                scene.element.node_size,
            )

    def compute_particle_kinematics(self, sims: Simulation, scene: myScene):
        kernel_kinemaitc_ig2p(
            scene.element.grid_nodes,
            sims.alphaPIC,
            sims.dt,
            int(scene.particleNum[0]),
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )

    def finalize_newton_raphson_iteration(self, sims: Simulation, scene: myScene):
        kernel_update_stress_strain_newmark(int(scene.particleNum[0]), scene.particle, sims.dt)

    def update_grid_refinement(self, sims: Simulation, scene: myScene):
        scene.element.update_refinement(sims, scene)

    def update_grid_refinement_profiled(self, sims: Simulation, scene: myScene):
        if getattr(scene.element, "adaptive", False) and scene.element.should_update_refinement(
            sims, scene.particleNum
        ):
            sims.timer.begin("Adaptive grid")
            self.update_adaptive_grid(sims, scene)
            sims.timer.end("Adaptive grid")

    def find_active_nodes(self, scene: myScene):
        find_active_node(self.iterator.node_stride(scene), scene.mass_cut_off, scene.node, scene.element.flag)
        scene.element.pse.run(scene.element.flag)
        return set_active_dofs(self.iterator.node_stride(scene), scene.mass_cut_off, scene.node, scene.element.flag)

    def pre_calculation(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        scene.element.calculate_characteristic_length(sims, int(scene.particleNum[0]), scene.particle, scene.psize)
        if sims.free_surface_detection:
            grid_mass_reset(scene.mass_cut_off, scene.node)
            scene.check_in_domain(sims)
            self.find_free_surface_by_density(scene)
            neighbor.place_particles(scene)
            self.detection_free_surface(scene, neighbor)
            grid_mass_reset(scene.mass_cut_off, scene.node)
        self.limit = sims.verlet_distance * sims.verlet_distance

        scene.element.calculate(scene.particleNum, scene.particle)
        if scene.sparse_grid is not None:
            scene.sparse_grid.rebuild(
                int(scene.particleNum[0]),
                scene.element.grid_nodes,
                scene.element.LnID,
                scene.element.node_size,
                scene.element.gnum,
            )
        self.compute_nodal_mass(sims, scene)
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            scene.material.compute_elasto_plastic_stiffness(
                materialID + 1, start_index, end_index, scene.particle, scene.material.materialID
            )
        total_dofs = estimate_active_grid_dofs(scene.mass_cut_off, scene.node)
        grid_mass_reset(scene.mass_cut_off, scene.node)

        self.iterator = MomentumConservation()
        self.iterator.manage_function(sims, scene)
        matrix_dofs = total_dofs
        if getattr(scene.element, "adaptive", False):
            max_adaptive_dofs = int(scene.element.gridSum) * sims.dimension
            multiplier = max(1, int(sims.dof_multiplier))
            matrix_dofs = max(matrix_dofs, (max_adaptive_dofs + multiplier - 1) // multiplier)
        self.iterator.set_matrix_vector(matrix_dofs, sims, scene)
        self.iterator.manage_operator(scene)
        self.iterator.operator.update_active_dofs(total_dofs)

    def newmark_integrate(self, sims: Simulation, scene: myScene, neighbor=None):
        sims.timer.begin("Shape function")
        self.calculate_interpolation(sims, scene)
        sims.timer.end("Shape function")
        sims.timer.begin("P2G")
        self.compute_nodal_kinematics(sims, scene)
        sims.timer.end("P2G")
        sims.timer.begin("Traction boundary")
        self.apply_particle_traction_constraints(sims, scene)
        sims.timer.end("Traction boundary")
        sims.timer.begin("Grid kinematic")
        self.compute_grid_velcity(sims, scene)
        sims.timer.end("Grid kinematic")
        sims.timer.begin("Compute dofs")
        total_dofs = self.find_active_nodes(scene)
        self.iterator.operator.update_active_dofs(total_dofs)
        sims.timer.end("Compute dofs")
        sims.timer.begin("Mass matrix")
        self.iterator.assemble_mass_matrix(sims, scene)
        sims.timer.end("Mass matrix")

        iter_num = 0
        convergence = False
        correction_error = float("inf")
        while not convergence and iter_num < sims.iter_max:
            sims.timer.begin("Grid reset-2")
            self.reset_iterative_grid_message(scene)
            sims.timer.end("Grid reset-2")
            sims.timer.begin("Internal force")
            self.compute_internal_forces(sims, scene)
            sims.timer.end("Internal force")
            self.iterator.run(sims, scene)
            self.update_nodal_displacements(scene)
            sims.timer.begin("Stress update")
            self.compute_velocity_gradient(sims, scene)
            self.compute_stress_strains(sims, scene)
            sims.timer.end("Stress update")
            sims.timer.begin("Compute residual")
            correction_error = self.iterator.compute_residual_error(
                self.iterator.node_stride(scene),
                scene.mass_cut_off,
                scene.node,
                scene.element.flag,
                self.iterator.unknow_vector,
            )
            convergence = math.isfinite(correction_error) and correction_error < sims.displacement_tolerance
            sims.timer.end("Compute residual")
            iter_num += 1

        if not convergence:
            raise RuntimeError(
                "Implicit MPM Newton solve did not converge: "
                f"relative_displacement_correction={correction_error:.6e}, "
                f"tolerance={sims.displacement_tolerance:.6e}, "
                f"iterations={iter_num}"
            )

        sims.timer.begin("Compute residual")
        self.finalize_newton_raphson_iteration(sims, scene)
        sims.timer.end("Compute residual")
        sims.timer.begin("Time integration")
        self.compute_nodal_kinematics_newmark(sims, scene)
        sims.timer.end("Time integration")
        sims.timer.begin("G2P")
        self.compute_particle_kinematic(sims, scene)
        sims.timer.end("G2P")
        sims.timer.begin("Compute residual")
        self.iterator.calculate_reaction_force(scene)
        sims.timer.end("Compute residual")
        self.update_grid_refinement_profiled(sims, scene)

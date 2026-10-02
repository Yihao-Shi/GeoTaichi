from src.mpm.boundaries.BoundaryCore import *
from src.mpm.engines.Engine import Engine
from src.mpm.engines.EngineKernel import *
from src.mpm.SceneManager import myScene
from src.mpm.Simulation import Simulation
from src.mpm.SpatialHashGrid import SpatialHashGrid
from src.utils.linalg import no_operation

@ti.data_oriented
class ULExplicitEngine(Engine):
    def __init__(self, sims) -> None:
        self.input_layer = 0
        self.compute = None
        self.compute_stress_strains = None
        self.bulid_neighbor_list = None
        self.apply_traction_constraints = None
        self.apply_absorbing_constraints = None
        self.apply_velocity_constraints = None
        self.apply_friction_constraints = None
        self.apply_reflection_constraints = None

        self.pre_contact_calculate = None
        self.compute_contact_force_ = None
        self.apply_contact_velocity_constraints = None
        self.apply_contact_reflection_constraints = None

        self.pressure_smoothing_ = None
        self.compute_nodal_kinematic = None
        self.compute_forces = None
        self.execute_board_serach = None
        self.calculate_interpolation = None
        self.free_surface_by_geometry = None
        self.compute_velocity_gradient = None
        self.calculate_velocity_gradient = None
        self.apply_adaptive_constraints = no_operation
        self.apply_adaptive_velocity_constraints = no_operation
        self.update_adaptive_grid = no_operation
        super().__init__(sims)

    def choose_boundary_constraints(self, sims: Simulation, scene: myScene):
        super().choose_boundary_constraints(sims, scene)
        if getattr(scene.element, "adaptive", False):
            if hasattr(scene.element, "configure_penalty_scale"):
                scene.element.configure_penalty_scale(scene.material)
            self.apply_adaptive_constraints = self.hanging_node_constraints
            self.apply_adaptive_velocity_constraints = self.hanging_node_velocity_constraints
            self.update_adaptive_grid = self.update_grid_refinement
        if int(scene.boundary.velocity_list[0]) > 0:
            if sims.contact_detection:
                self.apply_contact_velocity_constraints = self.contact_velocity_constraints
        if int(scene.boundary.reflection_list[0]) > 0:
            self.apply_reflection_constraints = self.reflection_constraints
            if sims.contact_detection:
                self.apply_contact_reflection_constraints = self.contact_reflection_constraints
        if int(scene.boundary.friction_list[0]) > 0:
            self.apply_friction_constraints = self.friction_constraints
        if int(scene.boundary.absorbing_list[0]) > 0:
            self.apply_absorbing_constraints = self.absorbing_constraints

    def calculate_precontact_2DAxisy(self, sims: Simulation, scene: myScene):
        kernel_calc_contact_normal_2DAxisy(scene.element.grid_nodes, int(scene.particleNum[0]), scene.element.gnum, scene.node, scene.particle, scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)

    def calculate_precontact(self, sims: Simulation, scene: myScene):
        kernel_calc_contact_normal(scene.element.grid_nodes, int(scene.particleNum[0]), scene.element.gnum, scene.node, scene.particle, scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)

    def compute_geocontact_force(self, sims: Simulation, scene: myScene):
        kernel_calc_contact_displacement(scene.element.grid_nodes, int(scene.particleNum[0]), scene.mass_cut_off, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)
        kernel_calc_geocontact(scene.mass_cut_off, scene.contact.friction, scene.contact.alpha, scene.contact.beta, scene.contact.cut_off, 
                               scene.element.gnum, scene.element.grid_size, sims.dt, scene.is_rigid, scene.node)
        self.apply_contact_velocity_constraints(sims, scene)
        self.apply_contact_reflection_constraints(sims, scene)
        kernel_assemble_contact_force(scene.mass_cut_off, sims.dt, scene.node)

    def compute_demcontact_force_2D(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_calc_demcontact_2D(scene.element.grid_nodes, start_index, end_index, scene.element.gnum, scene.element.grid_size, scene.contact.velocity, scene.particle, scene.material.materialID, scene.material.matProps[materialID + 1], sims.dt, 
                                      scene.contact.polygon_vertices, scene.element.LnID, scene.element.shape_fn, scene.element.node_size, scene.node)
        kernel_assemble_contact_force(scene.mass_cut_off, sims.dt, scene.node)

    def compute_contact_force(self, sims: Simulation, scene: myScene):
        kernel_calc_friction_contact(scene.mass_cut_off, scene.contact.friction, scene.element.gnum, sims.dt, scene.is_rigid, scene.node)
        self.apply_contact_velocity_constraints(sims, scene)
        self.apply_contact_reflection_constraints(sims, scene)
        kernel_assemble_contact_force(scene.mass_cut_off, sims.dt, scene.node)

    def compute_grid_velcity(self, sims: Simulation, scene: myScene):
        kernel_compute_grid_velocity(scene.mass_cut_off, scene.node)

    def compute_stress_strain(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_compute_stress_strain(start_index, end_index, sims.dt, scene.particle, scene.material.materialID, scene.material.matProps[materialID + 1], scene.material.stateVars)
        
    def compute_stress_strain_2D(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_compute_stress_strain_2D(start_index, end_index, sims.dt, scene.particle, scene.material.materialID, scene.material.matProps[materialID + 1], scene.material.stateVars)

    def compute_stress_strain_velocity_projection_2D(self, sims: Simulation, scene: myScene):
        scene.node.vol.fill(0)
        scene.node.jacobian.fill(0)
        scene.node.pressure.fill(0)
        scene.element.calc_shape_fn_spline_lower_order(scene.particleNum, scene.particle)
        kernel_dilatational_velocity_p2g(scene.element.grid_nodes_lower_order, int(scene.particleNum[0]), scene.element.gnum, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)
        kernel_grid_jacobian(scene.volume_cut_off, scene.is_rigid, scene.node)
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_gradient_velocity_projection_correction_2D(scene.element.grid_nodes_lower_order, start_index, end_index, scene.element.gnum, scene.node, scene.particle, scene.material.materialID, scene.element.LnID, scene.element.shape_fn, 
                                                              scene.element.node_size, scene.material.matProps[materialID + 1], scene.material.stateVars, sims.dt)
        kernel_grid_pressure_volume(scene.volume_cut_off, scene.is_rigid, scene.node)
        kernel_pressure_correction(scene.element.grid_nodes_lower_order, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def compute_stress_strain_velocity_projection(self, sims: Simulation, scene: myScene):
        scene.node.vol.fill(0)
        scene.node.jacobian.fill(0)
        scene.node.pressure.fill(0)
        scene.element.calc_shape_fn_spline_lower_order(scene.particleNum, scene.particle)
        kernel_dilatational_velocity_p2g(scene.element.grid_nodes_lower_order, int(scene.particleNum[0]), scene.element.gnum, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)
        kernel_grid_jacobian(scene.volume_cut_off, scene.is_rigid, scene.node)
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_gradient_velocity_projection_correction(scene.element.grid_nodes_lower_order, start_index, end_index, scene.element.gnum, scene.node, scene.particle, scene.material.materialID, scene.element.LnID, scene.element.shape_fn, 
                                                           scene.element.node_size, scene.material.matProps[materialID + 1], scene.material.stateVars, sims.dt)
        kernel_grid_pressure_volume(scene.volume_cut_off, scene.is_rigid, scene.node)
        kernel_pressure_correction(scene.element.grid_nodes_lower_order, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)
        
    def update_velocity_gradient_2D(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_update_velocity_gradient_2D(scene.element.grid_nodes, start_index, end_index, sims.dt, scene.node, scene.particle, scene.material.materialID, scene.material.matProps[materialID + 1], scene.material.stateVars,
                                               scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)
    
    def update_velocity_gradient(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_update_velocity_gradient(scene.element.grid_nodes, start_index, end_index, sims.dt, scene.node, scene.particle, scene.material.materialID, scene.material.matProps[materialID + 1], scene.material.stateVars,
                                            scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)
        
    def update_velocity_gradient_affine_2D(self, sims: Simulation, scene: myScene):
        if scene.sparse_grid is not None:
            kernel_update_velocity_gradient_affine_2D_sparse(
                scene.element.grid_nodes,
                int(scene.particleNum[0]),
                scene.element.grid_size,
                scene.node,
                scene.particle,
                scene.element.LnID,
                scene.element.shape_fn,
                scene.element.node_size,
                scene.sparse_grid.block_count,
                scene.sparse_grid.block_size,
                scene.sparse_grid.block_volume,
                scene.sparse_grid.active_block_ids,
            )
        else:
            kernel_update_velocity_gradient_affine_2D(scene.element.grid_nodes, int(scene.particleNum[0]), scene.element.gnum, scene.element.grid_size, 
                                                      scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)
        
    def update_velocity_gradient_affine(self, sims: Simulation, scene: myScene):
        if scene.sparse_grid is not None:
            kernel_update_velocity_gradient_affine_sparse(
                scene.element.grid_nodes,
                int(scene.particleNum[0]),
                scene.element.grid_size,
                scene.node,
                scene.particle,
                scene.element.LnID,
                scene.element.shape_fn,
                scene.element.node_size,
                scene.sparse_grid.block_count,
                scene.sparse_grid.block_size,
                scene.sparse_grid.block_volume,
                scene.sparse_grid.active_block_ids,
            )
        else:
            kernel_update_velocity_gradient_affine(scene.element.grid_nodes, int(scene.particleNum[0]), scene.element.gnum, scene.element.grid_size, 
                                                   scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def update_velocity_gradient_bbar_2D(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_update_velocity_gradient_bbar_2D(scene.element.grid_nodes, start_index, end_index, sims.dt, scene.node, scene.particle, scene.material.materialID, scene.material.matProps[materialID + 1], scene.material.stateVars,
                                                    scene.element.LnID, scene.element.dshape_fn, scene.element.dshape_fnc, scene.element.node_size)
    
    def update_velocity_gradient_bbar(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_update_velocity_gradient_bbar(scene.element.grid_nodes, start_index, end_index, sims.dt, scene.node, scene.particle, scene.material.materialID, scene.material.matProps[materialID + 1], scene.material.stateVars,
                                                 scene.element.LnID, scene.element.dshape_fn, scene.element.dshape_fnc, scene.element.node_size)
        
    def update_velocity_gradient_axisy_2D(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_update_velocity_gradient_2DAxisy(scene.element.grid_nodes, start_index, end_index, sims.dt, scene.node, scene.particle, scene.material.materialID, scene.material.matProps[materialID + 1], scene.material.stateVars,
                                                    scene.element.LnID, scene.element.shape_fn, scene.element.dshape_fn, scene.element.node_size)
        
    def update_velocity_gradient_bbar_axisy_2D(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_update_velocity_gradient_bbar_2DAxisy(scene.element.grid_nodes, start_index, end_index, sims.dt, scene.node, scene.particle, scene.material.materialID, scene.material.matProps[materialID + 1], scene.material.stateVars,
                                                         scene.element.LnID, scene.element.shape_fn, scene.element.shape_fnc, scene.element.dshape_fn, scene.element.dshape_fnc, scene.element.node_size)

    def particle_shifting(self, sims: Simulation, scene: myScene):
        scene.node.vol.fill(0)
        kernel_volume_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), scene.element.gnum, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)
        kernel_particle_shifting_delta_correction(scene.element.grid_nodes, int(scene.particleNum[0]), scene.element.grid_size, scene.node, scene.particle, scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)
        
    def compute_force(self, sims: Simulation, scene: myScene):
        kernel_force_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), scene.element.gnum, sims.gravity, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.dshape_fn, scene.element.node_size)

    def compute_force_2D(self, sims: Simulation, scene: myScene):
        kernel_force_p2g_2D(scene.element.grid_nodes, int(scene.particleNum[0]), scene.element.gnum, sims.gravity, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.dshape_fn, scene.element.node_size)

    def compute_force_2DAxisy(self, sims: Simulation, scene: myScene):
        kernel_force_p2g_2DAxisy(scene.element.grid_nodes, int(scene.particleNum[0]), scene.element.gnum, sims.gravity, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.dshape_fn, scene.element.node_size)

    def compute_force_bbar_2DAxisy(self, sims: Simulation, scene: myScene):
        kernel_force_bbar_p2g_2DAxisy(scene.element.grid_nodes, int(scene.particleNum[0]), scene.element.gnum, sims.gravity, scene.node, scene.particle, scene.element.LnID, 
                                      scene.element.shape_fn, scene.element.shape_fnc, scene.element.dshape_fn, scene.element.dshape_fnc, scene.element.node_size)
    
    def compute_force_mls(self, sims: Simulation, scene: myScene):
        kernel_force_mls_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), sims.gravity, scene.element.gnum, scene.element.grid_size, scene.element.inertia_tensor, scene.node, scene.particle, 
                             scene.element.LnID, scene.element.node_size)
        
    def compute_force_mls_2D(self, sims: Simulation, scene: myScene):
        kernel_force_mls_p2g_2D(scene.element.grid_nodes, int(scene.particleNum[0]), sims.gravity, scene.element.gnum, scene.element.grid_size, scene.element.inertia_tensor, scene.node, scene.particle, 
                                scene.element.LnID, scene.element.node_size)

    def compute_force_gauss(self, sims: Simulation, scene: myScene):
        kernel_external_force_p2g(scene.element.grid_nodes, scene.element.gnum, sims.gravity, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)
        gauss_cell_reset(scene.element.cell, scene.element.gauss_cell)
        kernel_volume_p2c(scene.element.cnum, scene.element.igrid_size, int(scene.particleNum[0]), scene.element.cell, scene.particle)
        kernel_find_valid_element(scene.element.cell_volume, scene.element.cell)
        kernel_sum_cell_stress(sims.gauss_number, scene.element.grid_size, scene.element.igrid_size, scene.element.cnum, int(scene.particleNum[0]), scene.particle, scene.element.cell, scene.element.gauss_cell)
        kernel_compute_gauss_average_stress(sims.gauss_number, scene.volume_cut_off, scene.element.cell, scene.element.gauss_cell)
        kernel_average_pressure(sims.gauss_number, scene.element.cell, scene.element.gauss_cell)
        kernel_internal_force_on_gauss_point_p2g(sims.gauss_number, scene.element.cnum, scene.element.gnum, scene.element.grid_size, scene.element.igrid_size, scene.node, 
                                                 scene.element.cell, scene.element.gauss_cell, scene.element.gauss_point.gpcoords, scene.element.gauss_point.weight)
        kernel_internal_force_on_material_point_p2g(scene.element.cnum, scene.element.gnum, scene.element.grid_size, scene.element.igrid_size, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.cell)

    def compute_force_gauss_2D(self, sims: Simulation, scene: myScene):
        kernel_external_force_p2g(scene.element.grid_nodes, scene.element.gnum, sims.gravity, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)
        gauss_cell_reset(scene.element.cell, scene.element.gauss_cell)
        kernel_volume_p2c_2D(scene.element.cnum, scene.element.igrid_size, int(scene.particleNum[0]), scene.element.cell, scene.particle)
        kernel_find_valid_element(scene.element.cell_volume, scene.element.cell)
        kernel_sum_cell_stress_2D(sims.gauss_number, scene.element.grid_size, scene.element.igrid_size, scene.element.cnum, int(scene.particleNum[0]), scene.particle, scene.element.cell, scene.element.gauss_cell)
        kernel_compute_gauss_average_stress_2D(sims.gauss_number, scene.volume_cut_off, scene.element.cell, scene.element.gauss_cell)
        kernel_average_pressure_2D(sims.gauss_number, scene.element.cell, scene.element.gauss_cell)
        kernel_internal_force_on_gauss_point_p2g_2D(sims.gauss_number, scene.element.cnum, scene.element.gnum, scene.element.grid_size, scene.element.igrid_size, scene.node,
                                                    scene.element.cell, scene.element.gauss_cell, scene.element.gauss_point.gpcoords, scene.element.gauss_point.weight)
        kernel_internal_force_on_material_point_p2g_2D(scene.element.cnum, scene.element.gnum, scene.element.grid_size, scene.element.igrid_size, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.cell)

    def compute_force_bbar(self, sims: Simulation, scene: myScene):
        kernel_force_bbar_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), scene.element.gnum, sims.gravity, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.dshape_fn, scene.element.dshape_fnc, scene.element.node_size)

    def compute_force_bbar_2D(self, sims: Simulation, scene: myScene):
        kernel_force_bbar_p2g_2D(scene.element.grid_nodes, int(scene.particleNum[0]), scene.element.gnum, sims.gravity, scene.node,scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.dshape_fn, scene.element.dshape_fnc, scene.element.node_size)

    def compute_grid_kinematic(self, sims: Simulation, scene: myScene):
        kernel_compute_grid_kinematic(scene.mass_cut_off, sims.background_damping, scene.node, sims.dt)

    def apply_kinematic_constraints(self, sims: Simulation, scene: myScene):
        self.apply_friction_constraints(sims, scene)
        self.apply_reflection_constraints(sims, scene)
        self.apply_velocity_constraints(sims, scene)

    def apply_dirichlet_constraints(self, sims: Simulation, scene: myScene):
        self.apply_reflection_constraints(sims, scene)
        self.apply_velocity_constraints(sims, scene)

    def traction_constraints(self, sims: Simulation, scene: myScene):
        if scene.sparse_grid is not None:
            apply_traction_constraint_sparse(
                int(scene.boundary.traction_list[0]),
                scene.boundary.traction_boundary,
                scene.node,
                scene.element.gnum,
                scene.sparse_grid.block_count,
                scene.sparse_grid.block_size,
                scene.sparse_grid.block_volume,
                scene.sparse_grid.block_map,
            )
        else:
            apply_traction_constraint(int(scene.boundary.traction_list[0]), scene.boundary.traction_boundary, scene.node)

    def traction_constraints_2D(self, sims: Simulation, scene: myScene):
        if scene.sparse_grid is not None:
            apply_traction_constraint_sparse(
                int(scene.boundary.traction_list[0]),
                scene.boundary.traction_boundary,
                scene.node,
                scene.element.gnum,
                scene.sparse_grid.block_count,
                scene.sparse_grid.block_size,
                scene.sparse_grid.block_volume,
                scene.sparse_grid.block_map,
            )
        else:
            apply_traction_constraint_2D(int(scene.boundary.traction_list[0]), scene.boundary.traction_boundary, scene.node)
    
    def absorbing_constraints(self, sims: Simulation, scene: myScene):
        apply_absorbing_constraint(int(scene.boundary.absorbing_list[0]), scene.boundary.absorbing_boundary, scene.material, scene.node)

    def velocity_constraints(self, sims: Simulation, scene: myScene):
        if scene.sparse_grid is not None:
            apply_velocity_constraint_sparse(
                scene.mass_cut_off,
                int(scene.boundary.velocity_list[0]),
                scene.boundary.velocity_boundary,
                scene.is_rigid,
                scene.node,
                scene.element.gnum,
                scene.sparse_grid.block_count,
                scene.sparse_grid.block_size,
                scene.sparse_grid.block_volume,
                scene.sparse_grid.block_map,
            )
        else:
            apply_velocity_constraint(scene.mass_cut_off, int(scene.boundary.velocity_list[0]), scene.boundary.velocity_boundary, scene.is_rigid, scene.node)

    def reflection_constraints(self, sims: Simulation, scene: myScene):
        if scene.sparse_grid is not None:
            apply_reflection_constraint_sparse(
                scene.mass_cut_off,
                int(scene.boundary.reflection_list[0]),
                scene.boundary.reflection_boundary,
                scene.is_rigid,
                scene.node,
                scene.element.gnum,
                scene.sparse_grid.block_count,
                scene.sparse_grid.block_size,
                scene.sparse_grid.block_volume,
                scene.sparse_grid.block_map,
            )
        else:
            apply_reflection_constraint(scene.mass_cut_off, int(scene.boundary.reflection_list[0]), scene.boundary.reflection_boundary, scene.is_rigid, scene.node)

    def contact_velocity_constraints(self, sims: Simulation, scene: myScene):
        apply_contact_velocity_constraint(scene.mass_cut_off, int(scene.boundary.velocity_list[0]), scene.boundary.velocity_boundary, scene.node)

    def contact_reflection_constraints(self, sims: Simulation, scene: myScene):
        apply_contact_reflection_constraint(scene.mass_cut_off, int(scene.boundary.reflection_list[0]), scene.boundary.reflection_boundary, scene.node)

    def friction_constraints(self, sims: Simulation, scene: myScene):
        if scene.sparse_grid is not None:
            apply_friction_constraint_sparse(
                scene.mass_cut_off,
                int(scene.boundary.friction_list[0]),
                scene.boundary.friction_boundary,
                scene.is_rigid,
                scene.node,
                sims.dt,
                scene.element.gnum,
                scene.sparse_grid.block_count,
                scene.sparse_grid.block_size,
                scene.sparse_grid.block_volume,
                scene.sparse_grid.block_map,
            )
        else:
            apply_friction_constraint(scene.mass_cut_off, int(scene.boundary.friction_list[0]), scene.boundary.friction_boundary, scene.is_rigid, scene.node, sims.dt)

    def hanging_node_constraints(self, sims: Simulation, scene: myScene):
        scene.element.apply_hanging_constraints(scene.mass_cut_off, sims.dt, scene.node, scene.particle)

    def hanging_node_velocity_constraints(self, sims: Simulation, scene: myScene):
        scene.element.apply_hanging_velocity_constraints(scene.mass_cut_off, sims.dt, scene.node, scene.particle)

    def update_grid_refinement(self, sims: Simulation, scene: myScene):
        scene.element.update_refinement(sims, scene)

    def update_grid_refinement_profiled(self, sims: Simulation, scene: myScene):
        if getattr(scene.element, "adaptive", False) and scene.element.should_update_refinement(
            sims, scene.particleNum
        ):
            sims.timer.begin('Adaptive grid')
            self.update_adaptive_grid(sims, scene)
            sims.timer.end('Adaptive grid')
        
    def compute_particle_kinematics(self, sims: Simulation, scene: myScene):
        kernel_kinemaitc_g2p(scene.element.grid_nodes, sims.alphaPIC, sims.dt, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def postmapping_grid_velocity(self, sims: Simulation, scene: myScene):
        kernel_reset_grid_velocity(scene.node)
        kernel_postmapping_kinemaitc(scene.element.grid_nodes, int(scene.particleNum[0]), scene.element.gnum, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def pre_calculation(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        scene.element.calculate_characteristic_length(sims, int(scene.particleNum[0]), scene.particle, scene.psize)
        if sims.neighbor_detection:
            scene.check_in_domain(sims)
            if sims.free_surface_detection:
                self.reset_grid_messages(scene)
                self.find_free_surface_by_density(sims, scene)
            neighbor.place_particles(scene)
            self.compute_boundary_direction(scene, neighbor)
            self.free_surface_by_geometry(scene, neighbor)
            if sims.free_surface_detection:
                self.reset_grid_messages(scene)
        self.limit = sims.verlet_distance * sims.verlet_distance

    def usl_updating(self, sims: Simulation, scene: myScene, neighbor=None):
        sims.timer.begin('Shape function')
        self.calculate_interpolation(sims, scene)
        sims.timer.end('Shape function')
        sims.timer.begin('P2G')
        self.compute_nodal_kinematic(sims, scene)
        self.compute_grid_velcity(sims, scene)
        self.compute_forces(sims, scene)
        sims.timer.end('P2G')
        sims.timer.begin('Traction boundary')
        self.apply_particle_traction_constraints(sims, scene)
        self.apply_traction_constraints(sims, scene)
        self.apply_absorbing_constraints(sims, scene)
        sims.timer.end('Traction boundary')
        sims.timer.begin('Grid kinematic')
        self.compute_grid_kinematic(sims, scene)
        sims.timer.end('Grid kinematic')
        sims.timer.begin('Contact & Kinematic boundary')
        self.pre_contact_calculate(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.apply_adaptive_constraints(sims, scene)
        self.apply_dirichlet_constraints(sims, scene)
        self.compute_contact_force_(sims, scene)
        sims.timer.end('Contact & Kinematic boundary')
        sims.timer.begin('G2P')
        self.compute_particle_kinematic(sims, scene)
        sims.timer.end('G2P')
        sims.timer.begin('Stress update')
        self.compute_velocity_gradient(sims, scene)
        self.compute_stress_strains(sims, scene)
        self.compute_affine_velocity_gradient(sims, scene)
        sims.timer.end('Stress update')
        self.update_grid_refinement_profiled(sims, scene)
        sims.timer.begin('Misc')
        self.pressure_smoothing_(sims, scene)
        sims.timer.end('Misc')

    def usf_updating(self, sims: Simulation, scene: myScene, neighbor=None):
        sims.timer.begin('Shape function')
        self.calculate_interpolation(sims, scene)
        sims.timer.end('Shape function')
        sims.timer.begin('P2G-1')
        self.compute_nodal_kinematic(sims, scene)
        self.compute_grid_velcity(sims, scene)
        self.apply_dirichlet_constraints(sims, scene)
        self.apply_adaptive_velocity_constraints(sims, scene)
        self.apply_dirichlet_constraints(sims, scene)
        sims.timer.end('P2G-1')
        sims.timer.begin('Stress update')
        self.compute_velocity_gradient(sims, scene)
        self.compute_stress_strains(sims, scene)
        self.compute_affine_velocity_gradient(sims, scene)
        sims.timer.end('Stress update')
        sims.timer.begin('Misc')
        self.pressure_smoothing_(sims, scene)
        sims.timer.end('Misc')
        sims.timer.begin('P2G-2')
        self.compute_forces(sims, scene)
        sims.timer.end('P2G-2')
        sims.timer.begin('Traction boundary')
        self.apply_particle_traction_constraints(sims, scene)
        self.apply_traction_constraints(sims, scene)
        self.apply_absorbing_constraints(sims, scene)
        sims.timer.end('Traction boundary')
        sims.timer.begin('Grid kinematic')
        self.compute_grid_kinematic(sims, scene)
        sims.timer.end('Grid kinematic')
        sims.timer.begin('Contact & Kinematic boundary')
        self.pre_contact_calculate(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.apply_adaptive_constraints(sims, scene)
        self.apply_dirichlet_constraints(sims, scene)
        self.compute_contact_force_(sims, scene)
        sims.timer.end('Contact & Kinematic boundary')
        sims.timer.begin('G2P')
        self.compute_particle_kinematic(sims, scene)
        sims.timer.end('G2P')
        self.update_grid_refinement_profiled(sims, scene)

    def musl_updating(self, sims: Simulation, scene: myScene, neighbor=None):
        sims.timer.begin('Shape function')
        self.calculate_interpolation(sims, scene)
        sims.timer.end('Shape function')
        sims.timer.begin('P2G')
        self.compute_nodal_kinematic(sims, scene)
        self.compute_grid_velcity(sims, scene)
        self.compute_forces(sims, scene)
        sims.timer.end('P2G')
        sims.timer.begin('Traction boundary')
        self.apply_particle_traction_constraints(sims, scene)
        self.apply_traction_constraints(sims, scene)
        self.apply_absorbing_constraints(sims, scene)
        sims.timer.end('Traction boundary')
        sims.timer.begin('Grid kinematic')
        self.compute_grid_kinematic(sims, scene)
        sims.timer.end('Grid kinematic')
        sims.timer.begin('Contact & Kinematic boundary')
        self.pre_contact_calculate(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.apply_adaptive_constraints(sims, scene)
        self.apply_dirichlet_constraints(sims, scene)
        self.compute_contact_force_(sims, scene)
        sims.timer.end('Contact & Kinematic boundary')
        sims.timer.begin('G2P')
        self.compute_particle_kinematic(sims, scene)
        sims.timer.end('G2P')
        sims.timer.begin('Velocity remapping')
        self.postmapping_grid_velocity(sims, scene)
        self.compute_grid_velcity(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.apply_adaptive_velocity_constraints(sims, scene)
        self.apply_dirichlet_constraints(sims, scene)
        sims.timer.end('Velocity remapping')
        sims.timer.begin('Stress update')
        self.compute_velocity_gradient(sims, scene)
        self.compute_stress_strains(sims, scene)
        self.compute_affine_velocity_gradient(sims, scene)
        sims.timer.end('Stress update')
        self.update_grid_refinement_profiled(sims, scene)
        sims.timer.begin('Misc')
        self.pressure_smoothing_(sims, scene)
        sims.timer.end('Misc')

    def first_substep_cfd_coupling(self, sims: Simulation, scene: myScene):
        self.calculate_interpolation(sims, scene)
        self.compute_nodal_kinematic(sims, scene)
        self.compute_grid_velcity(sims, scene)
        self.apply_dirichlet_constraints(sims, scene)
        self.compute_stress_strains(sims, scene)
        self.pressure_smoothing_(sims, scene)
        self.compute_forces(sims, scene)

    def second_substep_cfd_coupling(self, sims: Simulation, scene: myScene):
        self.apply_particle_traction_constraints(sims, scene)
        self.apply_traction_constraints(sims, scene)
        self.apply_absorbing_constraints(sims, scene)
        self.compute_grid_kinematic(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.compute_velocity_gradient(sims, scene)
        self.compute_particle_kinematic(sims, scene)

    def g2p2g(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        self.output_layer = 1 - self.input_layer
        g2p2g(scene.element.grid_nodes, int(scene.particleNum[0]), scene.element.grid_size, scene.element.igrid_size, sims.gravity, sims.dt, scene.node[self.output_layer],
              scene.node[self.input_layer], scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size, scene.material.matProps, scene.material.stateVars, neighbor.sorted.object_list)
        self.compute_grid_kinematic(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.input_layer = self.output_layer

    def lightweight(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        sims.timer.begin('P2G')
        lightweight_p2g(int(scene.particleNum[0]), scene.element.gnum, scene.element.grid_size, scene.element.igrid_size, sims.gravity, scene.element.calLength, scene.element.boundary_type, scene.node, scene.particle)
        sims.timer.end('P2G')
        sims.timer.begin('Traction boundary')
        self.apply_particle_traction_constraints(sims, scene)
        self.apply_traction_constraints(sims, scene)
        self.apply_absorbing_constraints(sims, scene)
        sims.timer.end('Traction boundary')
        sims.timer.begin('Grid kinematic')
        lightweight_grid_operation(scene.mass_cut_off, sims.background_damping, scene.node, sims.dt)
        sims.timer.end('Grid kinematic')
        sims.timer.begin('Contact & Kinematic boundary')
        self.pre_contact_calculate(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.compute_contact_force_(sims, scene)
        sims.timer.end('Contact & Kinematic boundary')
        sims.timer.begin('G2P')
        lightweight_g2p(int(scene.particleNum[0]), sims.alphaPIC, scene.mass_cut_off, sims.fbar_fraction, scene.element.gnum, scene.element.grid_size, scene.element.igrid_size, sims.dt, scene.element.calLength, scene.element.boundary_type, scene.node, scene.particle, scene.material.matProps, scene.material.stateVars)
        sims.timer.end('G2P')

    def test(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        pass

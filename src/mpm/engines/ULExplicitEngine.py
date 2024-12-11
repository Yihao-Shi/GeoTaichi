from src.mpm.boundaries.BoundaryCore import *
from src.mpm.engines.Engine import Engine
from src.mpm.engines.EngineKernel import *
from src.mpm.SceneManager import myScene
from src.mpm.Simulation import Simulation
from src.mpm.SpatialHashGrid import SpatialHashGrid
from src.utils.FreeSurfaceDetection import *
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
        super().__init__(sims)
        
    def choose_engine(self, sims: Simulation):
        if sims.mapping == "USL":
            self.compute = self.usl_updating
        elif sims.mapping == "USF":
            if sims.velocity_projection_scheme == "Taylor":
                self.compute = self.velocity_projection_updating
            else:
                self.compute = self.usf_updating
        elif sims.mapping == "MUSL":
            self.compute = self.musl_updating
        elif sims.mapping == "G2P2G":
            self.compute = self.g2p2g
        else:
            raise ValueError(f"The mapping scheme {sims.mapping} is not supported yet")
        
        if sims.velocity_projection_scheme == "Affine":
            self.compute = self.velocity_projection_updating

        if sims.TESTMODE:
            self.compute = self.test

    def choose_boundary_constraints(self, sims: Simulation, scene: myScene):
        self.apply_traction_constraints = no_operation
        self.apply_absorbing_constraints = no_operation
        self.apply_friction_constraints = no_operation
        self.apply_velocity_constraints = no_operation
        self.apply_reflection_constraints = no_operation
        self.apply_contact_velocity_constraints = no_operation
        self.apply_contact_reflection_constraints = no_operation
        self.apply_particle_traction_constraints = no_operation

        if int(scene.boundary.velocity_list[0]) > 0:
            self.apply_velocity_constraints = self.velocity_constraints
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
        if int(scene.boundary.traction_list[0]) > 0:
            if sims.dimension == 2:
                self.apply_traction_constraints = self.traction_constraints_2D
            elif sims.dimension == 3:
                self.apply_traction_constraints = self.traction_constraints
        if int(scene.boundary.ptraction_list[0]) > 0:
            if sims.material_type == "TwoPhaseSingleLayer":
                self.apply_particle_traction_constraints = self.particle_traction_constraints_twophase
            else:
                self.apply_particle_traction_constraints = self.particle_traction_constraints

    def manage_function(self, sims: Simulation):
        self.pre_contact_calculate = no_operation
        self.compute_nodal_kinematic = self.compute_nodal_kinematics
        self.compute_contact_force_ = no_operation
        self.pressure_smoothing_ = no_operation
        self.compute_forces = no_operation
        self.compute_stress_strains = no_operation
        self.bulid_neighbor_list = no_operation
        self.execute_board_serach = no_operation
        self.system_resolve = no_operation
        self.calculate_interpolation = self.calculate_interpolations
        self.compute_velocity_gradient = no_operation
        self.calculate_velocity_gradient = no_operation
        self.compute_particle_kinematic = no_operation

        if sims.dimension == 3:
            self.compute_particle_kinematic = self.compute_particle_kinematics
            self.compute_stress_strains = self.compute_stress_strain
            if sims.stabilize == "B-Bar Method":
                self.compute_forces = self.compute_force_bbar
                self.compute_velocity_gradient = self.update_velocity_gradient_bbar
            elif sims.stabilize == "F-Bar Method":
                self.compute_forces = self.compute_force
                if sims.velocity_projection_scheme == "Affine":
                    self.calculate_velocity_gradient = self.update_velocity_gradient_affine
                else:
                    self.calculate_velocity_gradient = self.update_velocity_gradient
                if sims.mls:
                    self.compute_forces = self.compute_force_mls
                if sims.material_type == "Solid":
                    self.compute_velocity_gradient = self.update_velocity_gradient_fbar
                elif sims.material_type == "Fluid":
                    self.compute_velocity_gradient = self.update_velocity_gradient
                    self.compute_stress_strains = self.compute_stress_strain_velocity_projection
            else:
                self.compute_forces = self.compute_force
                self.compute_velocity_gradient = self.update_velocity_gradient
                if sims.mls:
                    self.compute_forces = self.compute_force_mls

            if sims.gauss_number > 0:
                self.compute_forces = self.compute_force_gauss

            if sims.contact_detection:
                self.pre_contact_calculate = self.calculate_precontact
                if sims.contact_detection == "MPMContact":
                    self.compute_contact_force_ = self.compute_contact_force
                elif sims.contact_detection == "GeoContact":
                    self.compute_contact_force_ = self.compute_geocontact_force
                else:
                    raise RuntimeError("Wrong contact type!")

            if sims.pressure_smoothing:
                self.pressure_smoothing_ = self.pressure_smoothing

            if sims.velocity_projection_scheme == "Affine" or sims.velocity_projection_scheme == "Taylor":
                self.compute_nodal_kinematic = self.compute_nodal_kinematics_taylor
                if sims.velocity_projection_scheme == "Affine":
                    self.compute_velocity_gradient = self.update_velocity_gradient_affine

            if sims.neighbor_detection:
                self.calculate_interpolation = no_operation
                if sims.coupling == "Lagrangian":
                    self.execute_board_serach = self.update_verlet_table
                    self.system_resolve = self.compute_nodal_kinematic
                else:
                    self.bulid_neighbor_list = self.board_search
                self.compute_nodal_kinematic = no_operation

                self.free_surface_by_geometry = no_operation
                if sims.free_surface_detection:
                    self.free_surface_by_geometry = self.detection_free_surface

                self.compute_boundary_direction = no_operation
                if sims.boundary_direction_detection:
                    self.compute_boundary_direction = self.detection_boundary_direction
        elif sims.dimension == 2:
            self.compute_particle_kinematic = self.compute_particle_kinematics_2D
            if not sims.is_2DAxisy:
                self.compute_stress_strains = self.compute_stress_strain_2D
                if sims.stabilize == "B-Bar Method":
                    self.compute_forces = self.compute_force_bbar_2D
                    self.compute_velocity_gradient = self.update_velocity_gradient_bbar_2D
                elif sims.stabilize == "F-Bar Method":
                    self.compute_forces = self.compute_force_2D
                    if sims.mls:
                        self.compute_forces = self.compute_force_mls_2D
                    if sims.material_type == "Solid":
                        self.compute_velocity_gradient = self.update_velocity_gradient_fbar
                    elif sims.material_type == "Fluid":
                        self.compute_velocity_gradient = self.update_velocity_gradient_2D
                        self.compute_stress_strains = self.compute_stress_strain_velocity_projection_2D
                    if sims.velocity_projection_scheme == "Affine":
                        self.calculate_velocity_gradient = self.update_velocity_gradient_affine_2D
                    else:
                        self.calculate_velocity_gradient = self.update_velocity_gradient_2D
                else:
                    self.compute_forces = self.compute_force_2D
                    self.compute_velocity_gradient = self.update_velocity_gradient_2D
                    if sims.mls:
                        self.compute_forces = self.compute_force_mls_2D

                if sims.gauss_number > 0:
                    self.compute_forces = self.compute_force_gauss_2D

                if sims.contact_detection:
                    self.pre_contact_calculate = self.calculate_precontact
                    if sims.contact_detection == "MPMContact":
                        self.compute_contact_force_ = self.compute_contact_force_2D
                    elif sims.contact_detection == "GeoContact":
                        self.compute_contact_force_ = self.compute_geocontact_force_2D
                    elif sims.contact_detection == "DEMContact":
                        self.pre_contact_calculate = no_operation
                        self.compute_contact_force_ = self.compute_demcontact_force_2D
                    else:
                        raise RuntimeError("Wrong contact type!")
            elif sims.is_2DAxisy:
                self.compute_stress_strains = self.compute_stress_strain
                if sims.stabilize == "B-Bar Method":
                    self.compute_forces = self.compute_force_bbar_2DAxisy
                    self.compute_velocity_gradient = self.update_velocity_gradient_bbar_axisy_2D
                elif sims.stabilize == "F-Bar Method":
                    self.compute_forces = self.compute_force_2DAxisy
                    if sims.mls:
                        raise RuntimeError("2D condition do not support moving least squares")
                    if sims.material_type == "Fluid":
                        raise RuntimeError("2D condition do not support F-bar method for fluid materials")
                    if sims.velocity_projection_scheme == "Affine":
                        raise RuntimeError("2D condition do not support affine projection")
                else:
                    self.compute_forces = self.compute_force_2DAxisy
                    self.compute_velocity_gradient = self.update_velocity_gradient_axisy_2D
                    if sims.mls:
                        raise RuntimeError("2D condition do not support moving least squares")

                if sims.gauss_number > 0:
                    self.compute_forces = self.compute_force_gauss_2D

                if sims.contact_detection:
                    self.pre_contact_calculate = self.calculate_precontact_2DAxisy
                    if sims.contact_detection == "MPMContact":
                        self.compute_contact_force_ = self.compute_contact_force_2DAxisy
                    elif sims.contact_detection == "GeoContact":
                        self.compute_contact_force_ = self.compute_geocontact_force_2DAxisy
                    elif sims.contact_detection == "DEMContact":
                        self.pre_contact_calculate = no_operation
                        self.compute_contact_force_ = self.compute_demcontact_force_2D
                    else:
                        raise RuntimeError("Wrong contact type!")

            if sims.pressure_smoothing:
                self.pressure_smoothing_ = self.pressure_smoothing_2D

            if sims.velocity_projection_scheme == "Affine" or sims.velocity_projection_scheme == "Taylor":
                self.compute_nodal_kinematic = self.compute_nodal_kinematics_taylor_2D
                if sims.velocity_projection_scheme == "Affine":
                    if sims.is_2DAxisy: raise RuntimeError("2D condition do not support affine projection")
                    self.compute_velocity_gradient = self.update_velocity_gradient_affine_2D

            if sims.neighbor_detection:
                self.calculate_interpolation = no_operation
                if sims.coupling == "Lagrangian":
                    self.execute_board_serach = self.update_verlet_table
                    self.system_resolve = self.compute_nodal_kinematic
                else:
                    self.bulid_neighbor_list = self.board_search
                self.compute_nodal_kinematic = no_operation

                self.free_surface_by_geometry = no_operation
                if sims.free_surface_detection:
                    self.free_surface_by_geometry = self.detection_free_surface

                self.compute_boundary_direction = no_operation
                if sims.boundary_direction_detection:
                    self.compute_boundary_direction = self.detection_boundary_direction

    def valid_contact(self, sims: Simulation, scene: myScene):
        if sims.contact_detection == "GeoContact":
            if scene.is_rigid[0] == 0 and scene.is_rigid[1] == 0:
                raise RuntimeError("GeoContact is only suitable for soil-structure interaction")

    def calculate_precontact_2DAxisy(self, sims: Simulation, scene: myScene):
        kernel_calc_contact_normal_2DAxisy(scene.element.grid_nodes, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)

    def calculate_precontact(self, sims: Simulation, scene: myScene):
        kernel_calc_contact_normal(scene.element.grid_nodes, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)

    def compute_geocontact_force(self, sims: Simulation, scene: myScene):
        kernel_calc_contact_displacement(scene.element.grid_nodes, int(scene.particleNum[0]), scene.mass_cut_off, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)
        kernel_calc_geocontact(scene.mass_cut_off, scene.contact_parameter.friction, scene.contact_parameter.alpha, scene.contact_parameter.beta, scene.contact_parameter.cut_off, 
                               scene.element.gnum, scene.element.grid_size, sims.dt, scene.is_rigid, scene.node)
        self.apply_contact_velocity_constraints(sims, scene)
        self.apply_contact_reflection_constraints(sims, scene)
        kernel_assemble_contact_force(scene.mass_cut_off, sims.dt, scene.node)

    def compute_geocontact_force_2D(self, sims: Simulation, scene: myScene):
        kernel_calc_contact_displacement(scene.element.grid_nodes, int(scene.particleNum[0]), scene.mass_cut_off, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)
        kernel_calc_geocontact_2D(scene.mass_cut_off, scene.contact_parameter.friction, scene.contact_parameter.alpha, scene.contact_parameter.beta, scene.contact_parameter.cut_off, 
                                  scene.element.gnum,scene.element.grid_size,sims.dt, scene.is_rigid, scene.node)
        self.apply_contact_velocity_constraints(sims, scene)
        self.apply_contact_reflection_constraints(sims, scene)
        kernel_assemble_contact_force(scene.mass_cut_off, sims.dt, scene.node)

    def compute_geocontact_force_2DAxisy(self, sims: Simulation, scene: myScene):
        kernel_calc_contact_displacement(scene.element.grid_nodes, int(scene.particleNum[0]), scene.mass_cut_off, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)
        kernel_calc_geocontact_2DAxisy(scene.mass_cut_off, scene.contact_parameter.friction, scene.contact_parameter.alpha, scene.contact_parameter.beta, scene.contact_parameter.cut_off, 
                                       scene.element.gnum, scene.element.grid_size,sims.dt, scene.is_rigid, scene.node)
        self.apply_contact_velocity_constraints(sims, scene)
        self.apply_contact_reflection_constraints(sims, scene)
        kernel_assemble_contact_force(scene.mass_cut_off, sims.dt, scene.node)

    def compute_demcontact_force_2D(self, sims: Simulation, scene: myScene):
        kernel_calc_demcontact_2D(scene.element.grid_nodes, int(scene.particleNum[0]), scene.element.grid_size, scene.contact_parameter.velocity, scene.particle, scene.material.matProps, sims.dt, scene.contact_parameter.polygon_vertices,
                                  scene.element.LnID, scene.element.shape_fn, scene.element.node_size, scene.node)
        kernel_assemble_contact_force(scene.mass_cut_off, sims.dt, scene.node)

    def compute_contact_force(self, sims: Simulation, scene: myScene):
        kernel_calc_friction_contact(scene.mass_cut_off, scene.contact_parameter.friction, sims.dt, scene.is_rigid, scene.node)
        self.apply_contact_velocity_constraints(sims, scene)
        self.apply_contact_reflection_constraints(sims, scene)
        kernel_assemble_contact_force(scene.mass_cut_off, sims.dt, scene.node)

    def compute_contact_force_2D(self, sims: Simulation, scene: myScene):
        kernel_calc_friction_contact_2D(scene.mass_cut_off, scene.contact_parameter.friction, sims.dt, scene.is_rigid, scene.node)
        self.apply_contact_velocity_constraints(sims, scene)
        self.apply_contact_reflection_constraints(sims, scene)
        kernel_assemble_contact_force(scene.mass_cut_off, sims.dt, scene.node)

    def compute_contact_force_2DAxisy(self, sims: Simulation, scene: myScene):
        kernel_calc_friction_contact_2DAxisy(scene.mass_cut_off, scene.contact_parameter.friction, sims.dt, scene.is_rigid, scene.node)
        self.apply_contact_velocity_constraints(sims, scene)
        self.apply_contact_reflection_constraints(sims, scene)
        kernel_assemble_contact_force(scene.mass_cut_off, sims.dt, scene.node)

    def pressure_smoothing(self, scene: myScene):
        scene.extra_node.fill(0)
        kernel_pressure_p2g(scene.element.gnum, scene.element.igrid_size, int(scene.particleNum[0]), scene.extra_node, scene.particle)
        kernel_grid_pressure(scene.mass_cut_off, scene.is_rigid, scene.node, scene.extra_node)
        kernel_pressure_g2p(scene.element.gnum, scene.element.igrid_size, scene.extra_node, int(scene.particleNum[0]), scene.particle)

    def pressure_smoothing_2D(self, scene: myScene):
        scene.extra_node.fill(0)
        kernel_pressure_p2g_2D(scene.element.gnum, scene.element.igrid_size, int(scene.particleNum[0]), scene.extra_node, scene.particle)
        kernel_grid_pressure(scene.mass_cut_off, scene.is_rigid, scene.node, scene.extra_node)
        kernel_pressure_g2p_2D(scene.element.gnum, scene.element.igrid_size, scene.extra_node, int(scene.particleNum[0]), scene.particle)

    def compute_nodal_kinematics(self, sims: Simulation, scene: myScene):
        kernel_mass_momentum_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def compute_nodal_kinematics_taylor(self, sims: Simulation, scene: myScene):
        kernel_mass_momentum_taylor_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), scene.element.gnum, scene.element.grid_size, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def compute_nodal_kinematics_taylor_2D(self, sims: Simulation, scene: myScene):
        kernel_mass_momentum_taylor_2D_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), scene.element.gnum, scene.element.grid_size, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def compute_grid_velcity(self, sims: Simulation, scene: myScene):
        kernel_compute_grid_velocity(scene.mass_cut_off, scene.node)

    def compute_stress_strain(self, sims: Simulation, scene: myScene):
        kernel_compute_stress_strain(scene.element.grid_nodes, sims.dt, int(scene.particleNum[0]), scene.node, scene.particle, scene.material.matProps, scene.material.stateVars,
                                     scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)
        
    def compute_stress_strain_2D(self, sims: Simulation, scene: myScene):
        kernel_compute_stress_strain_2D(scene.element.grid_nodes, sims.dt, int(scene.particleNum[0]), scene.node, scene.particle, scene.material.matProps, scene.material.stateVars,
                                        scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)

    def compute_stress_strain_velocity_projection_2D(self, sims: Simulation, scene: myScene):
        scene.extra_node.fill(0)
        scene.element.calc_shape_fn_spline_lower_order(scene.particleNum, scene.particle)
        kernel_dilatational_velocity_p2g(scene.element.grid_nodes_lower_order, int(scene.particleNum[0]), scene.extra_node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)
        kernel_grid_jacobian(scene.volume_cut_off, scene.is_rigid, scene.extra_node)
        kernel_gradient_velocity_projection_correction_2D(scene.element.grid_nodes_lower_order, int(scene.particleNum[0]), scene.extra_node, scene.particle, scene.element.LnID, scene.element.shape_fn, 
                                                          scene.element.node_size, scene.material.matProps, scene.material.stateVars, sims.dt)
        kernel_grid_pressure_volume(scene.volume_cut_off, scene.is_rigid, scene.extra_node)
        kernel_pressure_correction(scene.element.grid_nodes_lower_order, int(scene.particleNum[0]), scene.extra_node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def compute_stress_strain_velocity_projection(self, sims: Simulation, scene: myScene):
        scene.extra_node.fill(0)
        scene.element.calc_shape_fn_spline_lower_order(scene.particleNum, scene.particle)
        kernel_dilatational_velocity_p2g(scene.element.grid_nodes_lower_order, int(scene.particleNum[0]), scene.extra_node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)
        kernel_grid_jacobian(scene.volume_cut_off, scene.is_rigid, scene.extra_node)
        kernel_gradient_velocity_projection_correction(scene.element.grid_nodes_lower_order, int(scene.particleNum[0]), scene.extra_node, scene.particle, scene.element.LnID, scene.element.shape_fn, 
                                                       scene.element.node_size, scene.material.matProps, scene.material.stateVars, sims.dt)
        kernel_grid_pressure_volume(scene.volume_cut_off, scene.is_rigid, scene.extra_node)
        kernel_pressure_correction(scene.element.grid_nodes_lower_order, int(scene.particleNum[0]), scene.extra_node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def update_velocity_gradient_fbar(self, sims: Simulation, scene: myScene):
        scene.extra_node.fill(0)
        self.calculate_velocity_gradient(sims, scene)
        kernel_volume_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), scene.extra_node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)
        kernel_jacobian_p2g(scene.element.grid_nodes, sims.dt, int(scene.particleNum[0]), scene.extra_node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)
        kernel_grid_jacobian(scene.volume_cut_off, scene.is_rigid, scene.extra_node)
        kernel_update_velocity_gradient_fbar(sims.dimension, scene.element.grid_nodes, sims.dt, int(scene.particleNum[0]), scene.extra_node, scene.particle, 
                                             scene.material.matProps, scene.material.stateVars, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)
        
    def update_velocity_gradient_2D(self, sims: Simulation, scene: myScene):
        kernel_update_velocity_gradient_2D(scene.element.grid_nodes, int(scene.particleNum[0]), sims.dt, scene.node, scene.particle, scene.material.matProps, scene.material.stateVars,
                                        scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)
    
    def update_velocity_gradient(self, sims: Simulation, scene: myScene):
        kernel_update_velocity_gradient(scene.element.grid_nodes, int(scene.particleNum[0]), sims.dt, scene.node, scene.particle, scene.material.matProps, scene.material.stateVars,
                                        scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)
        
    def update_velocity_gradient_affine_2D(self, sims: Simulation, scene: myScene):
        kernel_update_velocity_gradient_affine_2D(scene.element.grid_nodes, int(scene.particleNum[0]), scene.element.gnum, scene.element.grid_size, sims.dt, scene.node, scene.particle, 
                                                  scene.material.matProps, scene.material.stateVars, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)
        
    def update_velocity_gradient_affine(self, sims: Simulation, scene: myScene):
        kernel_update_velocity_gradient_affine(scene.element.grid_nodes, int(scene.particleNum[0]), scene.element.gnum, scene.element.grid_size, sims.dt, scene.node, scene.particle, 
                                               scene.material.matProps, scene.material.stateVars, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def update_velocity_gradient_bbar_2D(self, sims: Simulation, scene: myScene):
        kernel_update_velocity_gradient_bbar_2D(scene.element.grid_nodes, int(scene.particleNum[0]), sims.dt, scene.node, scene.particle, scene.material.matProps, scene.material.stateVars,
                                                scene.element.LnID, scene.element.dshape_fn, scene.element.dshape_fnc, scene.element.node_size)
    
    def update_velocity_gradient_bbar(self, sims: Simulation, scene: myScene):
        kernel_update_velocity_gradient_bbar(scene.element.grid_nodes, int(scene.particleNum[0]), sims.dt, scene.node, scene.particle, scene.material.matProps, scene.material.stateVars,
                                             scene.element.LnID, scene.element.dshape_fn, scene.element.dshape_fnc, scene.element.node_size)
        
    def update_velocity_gradient_axisy_2D(self, sims: Simulation, scene: myScene):
        kernel_update_velocity_gradient_2DAxisy(scene.element.grid_nodes, int(scene.particleNum[0]), sims.dt, scene.node, scene.particle, scene.material.matProps, scene.material.stateVars,
                                                scene.element.LnID, scene.element.shape_fn, scene.element.dshape_fn, scene.element.node_size)
        
    def update_velocity_gradient_bbar_axisy_2D(self, sims: Simulation, scene: myScene):
        kernel_update_velocity_gradient_bbar_2DAxisy(scene.element.grid_nodes, int(scene.particleNum[0]), sims.dt, scene.node, scene.particle, scene.material.matProps, scene.material.stateVars,
                                                     scene.element.LnID, scene.element.shape_fn, scene.element.shape_fnc, scene.element.dshape_fn, scene.element.dshape_fnc, scene.element.node_size)

    def particle_shifting(self, sims: Simulation, scene: myScene):
        scene.extra_node.fill(0)
        kernel_volume_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), scene.extra_node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)
        kernel_particle_shifting_delta_correction(scene.element.grid_nodes, int(scene.particleNum[0]), scene.element.grid_size, scene.extra_node, scene.particle, scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)

    def compute_pressure_strain(self, sims: Simulation, scene: myScene):
        kernel_find_sound_speed(int(scene.particleNum[0]), scene.particle, scene.material.matProps)
        kernel_compute_stress_strain(scene.element.grid_nodes, sims.dt, int(scene.particleNum[0]), scene.node, scene.particle, scene.material.matProps, scene.material.stateVars,
                                     scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)
        
    def compute_force(self, sims: Simulation, scene: myScene):
        kernel_force_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), sims.gravity, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.dshape_fn, scene.element.node_size)

    def compute_force_2D(self, sims: Simulation, scene: myScene):
        kernel_force_p2g_2D(scene.element.grid_nodes, int(scene.particleNum[0]), sims.gravity, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.dshape_fn, scene.element.node_size)

    def compute_force_2DAxisy(self, sims: Simulation, scene: myScene):
        kernel_force_p2g_2DAxisy(scene.element.grid_nodes, int(scene.particleNum[0]), sims.gravity, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.dshape_fn, scene.element.node_size)

    def compute_force_bbar_2DAxisy(self, sims: Simulation, scene: myScene):
        kernel_force_bbar_p2g_2DAxisy(scene.element.grid_nodes, int(scene.particleNum[0]), scene.element.grid_size, sims.gravity, scene.node, scene.particle, scene.element.LnID, 
                                      scene.element.shape_fn, scene.element.shape_fnc, scene.element.dshape_fn, scene.element.dshape_fnc, scene.element.node_size)
    
    def compute_force_mls(self, sims: Simulation, scene: myScene):
        kernel_force_mls_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), sims.gravity, scene.element.gnum, scene.element.grid_size, scene.element.inertia_tensor, scene.node, scene.particle, 
                                      scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)
        
    def compute_force_mls_2D(self, sims: Simulation, scene: myScene):
        kernel_force_mls_p2g_2D(scene.element.grid_nodes, int(scene.particleNum[0]), sims.gravity, scene.element.gnum, scene.element.grid_size, scene.element.inertia_tensor, scene.node, scene.particle, 
                                      scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)

    def compute_force_gauss(self, sims: Simulation, scene: myScene):
        kernel_external_force_p2g(scene.element.grid_nodes, sims.gravity, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)
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
        kernel_external_force_p2g(scene.element.grid_nodes, sims.gravity, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)
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
        kernel_force_bbar_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), sims.gravity, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.dshape_fn, scene.element.dshape_fnc, scene.element.node_size)

    def compute_force_bbar_2D(self, sims: Simulation, scene: myScene):
        kernel_force_bbar_p2g_2D(scene.element.grid_nodes, int(scene.particleNum[0]), sims.gravity, scene.node,scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.dshape_fn, scene.element.dshape_fnc, scene.element.node_size)

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
        apply_traction_constraint(int(scene.boundary.traction_list[0]), scene.boundary.traction_boundary, scene.node)

    def traction_constraints_2D(self, sims: Simulation, scene: myScene):
        apply_traction_constraint_2D(int(scene.boundary.traction_list[0]), scene.boundary.traction_boundary, scene.node)
    
    def absorbing_constraints(self, sims: Simulation, scene: myScene):
        apply_absorbing_constraint(int(scene.boundary.absorbing_list[0]), scene.boundary.absorbing_boundary, scene.material, scene.node, scene.extra_node)

    def velocity_constraints(self, sims: Simulation, scene: myScene):
        apply_velocity_constraint(scene.mass_cut_off, int(scene.boundary.velocity_list[0]), scene.boundary.velocity_boundary, scene.is_rigid, scene.node)

    def reflection_constraints(self, sims: Simulation, scene: myScene):
        apply_reflection_constraint(scene.mass_cut_off, int(scene.boundary.reflection_list[0]), scene.boundary.reflection_boundary, scene.is_rigid, scene.node)

    def contact_velocity_constraints(self, sims: Simulation, scene: myScene):
        apply_contact_velocity_constraint(scene.mass_cut_off, int(scene.boundary.velocity_list[0]), scene.boundary.velocity_boundary, scene.node)

    def contact_reflection_constraints(self, sims: Simulation, scene: myScene):
        apply_contact_reflection_constraint(scene.mass_cut_off, int(scene.boundary.reflection_list[0]), scene.boundary.reflection_boundary, scene.node)

    def friction_constraints(self, sims: Simulation, scene: myScene):
        apply_friction_constraint(scene.mass_cut_off, int(scene.boundary.friction_list[0]), scene.boundary.friction_boundary, scene.is_rigid, scene.node, sims.dt)
        
    def compute_particle_kinematics(self, sims: Simulation, scene: myScene):
        kernel_kinemaitc_g2p(scene.element.grid_nodes, sims.alphaPIC, sims.dt, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def compute_particle_kinematics_2D(self, sims: Simulation, scene: myScene):
        kernel_kinemaitc_g2p_2D(scene.element.grid_nodes, sims.alphaPIC, sims.dt, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def postmapping_grid_velocity(self, sims: Simulation, scene: myScene):
        kernel_reset_grid_velocity(scene.node)
        kernel_postmapping_kinemaitc(scene.element.grid_nodes, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def update_verlet_table(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        scene.check_in_domain(sims.domain, int(scene.particleNum[0]), scene.particle)
        self.find_free_surface_by_density(sims, scene)
        neighbor.place_particles(scene)
        self.compute_boundary_direction(scene, neighbor)
        self.free_surface_by_geometry(scene, neighbor)
        scene.reset_verlet_disp()

    def board_search(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        if self.is_need_update_verlet_table(scene) == 1:
            self.update_verlet_table(sims, scene, neighbor)
        else:
            self.system_resolve(sims, scene)

    def detection_boundary_direction(self, scene: myScene, neighbor: SpatialHashGrid):
        find_boundary_direction_by_geometry(neighbor.igrid_size, neighbor.cnum, int(scene.particleNum[0]), scene.particle, neighbor.ParticleID, neighbor.current, neighbor.count)

    def detection_free_surface(self, scene: myScene, neighbor: SpatialHashGrid):
        find_free_surface_by_geometry(neighbor.igrid_size, neighbor.cnum, int(scene.particleNum[0]), scene.particle, neighbor.ParticleID, neighbor.current, neighbor.count)

    def find_free_surface_by_density(self, sims, scene: myScene):
        scene.element.calculate(scene.particleNum, scene.particle)
        self.system_resolve(sims, scene)
        kernel_mass_g2p(scene.element.grid_nodes, scene.element.cell_volume, scene.element.node_size, scene.element.LnID, scene.element.shape_fn, scene.node, int(scene.particleNum[0]), scene.particle)
        assign_particle_free_surface(int(scene.particleNum[0]), scene.particle, scene.material.matProps)

    def pre_calculation(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        scene.element.calculate_characteristic_length(sims, int(scene.particleNum[0]), scene.particle, scene.psize)
        if sims.neighbor_detection:
            grid_mass_reset(scene.mass_cut_off, scene.node)
            scene.check_in_domain(sims.domain, int(scene.particleNum[0]), scene.particle)
            self.find_free_surface_by_density(sims, scene)
            neighbor.place_particles(scene)
            self.compute_boundary_direction(scene, neighbor)
            self.free_surface_by_geometry(scene, neighbor)
            grid_mass_reset(scene.mass_cut_off, scene.node)
        self.limit = sims.verlet_distance * sims.verlet_distance
        self.stress = ti.Matrix.field(2,2,float,shape=sims.max_particle_num)

    def calculate_interpolations(self, sims: Simulation, scene: myScene):
        scene.element.calculate(scene.particleNum, scene.particle)

    def usl_updating(self, sims: Simulation, scene: myScene):
        self.calculate_interpolation(sims, scene)
        self.compute_nodal_kinematic(sims, scene)
        self.compute_grid_velcity(sims, scene)
        self.apply_particle_traction_constraints(sims, scene)
        self.compute_forces(sims, scene)
        self.apply_traction_constraints(sims, scene)
        self.apply_absorbing_constraints(sims, scene)
        self.compute_grid_kinematic(sims, scene)
        self.pre_contact_calculate(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.compute_contact_force_(sims, scene)
        self.compute_particle_kinematic(sims, scene)
        self.compute_velocity_gradient(sims, scene)
        self.compute_stress_strains(sims, scene)
        self.pressure_smoothing_(scene)

    def usf_updating(self, sims: Simulation, scene: myScene):
        self.calculate_interpolation(sims, scene)
        self.compute_nodal_kinematic(sims, scene)
        self.compute_grid_velcity(sims, scene)
        self.apply_dirichlet_constraints(sims, scene)
        self.compute_velocity_gradient(sims, scene)
        self.compute_stress_strains(sims, scene)
        self.pressure_smoothing_(scene)
        self.apply_particle_traction_constraints(sims, scene)
        self.compute_forces(sims, scene)
        self.apply_traction_constraints(sims, scene)
        self.apply_absorbing_constraints(sims, scene)
        self.compute_grid_kinematic(sims, scene)
        self.pre_contact_calculate(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.compute_contact_force_(sims, scene)
        self.compute_particle_kinematic(sims, scene)

    def musl_updating(self, sims: Simulation, scene: myScene):
        self.calculate_interpolation(sims, scene)
        self.compute_nodal_kinematic(sims, scene)
        self.compute_grid_velcity(sims, scene)
        self.apply_particle_traction_constraints(sims, scene)
        self.compute_forces(sims, scene)
        self.apply_traction_constraints(sims, scene)
        self.apply_absorbing_constraints(sims, scene)
        self.compute_grid_kinematic(sims, scene)
        self.pre_contact_calculate(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.compute_contact_force_(sims, scene)
        self.compute_particle_kinematic(sims, scene)
        self.postmapping_grid_velocity(sims, scene)
        self.compute_grid_velcity(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.compute_velocity_gradient(sims, scene)
        self.compute_stress_strains(sims, scene)
        self.pressure_smoothing_(scene)

    def velocity_projection_updating(self, sims: Simulation, scene: myScene):
        self.calculate_interpolation(sims, scene)
        self.compute_nodal_kinematic(sims, scene)
        self.compute_grid_velcity(sims, scene)
        self.apply_dirichlet_constraints(sims, scene)
        self.compute_stress_strains(sims, scene)
        self.pressure_smoothing_(scene)
        self.apply_particle_traction_constraints(sims, scene)
        self.compute_forces(sims, scene)
        self.apply_traction_constraints(sims, scene)
        self.apply_absorbing_constraints(sims, scene)
        self.compute_grid_kinematic(sims, scene)
        self.pre_contact_calculate(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.compute_contact_force_(sims, scene)
        self.compute_velocity_gradient(sims, scene)
        self.compute_particle_kinematic(sims, scene)

    def g2p2g(self, sims: Simulation, scene: myScene):
        output_layer = 1 - self.input_layer
        #self.grid[output_layer].deactivate_all()
        #build_pid(self.pid[self.input_layer], self.grid_m[self.input_layer], 0.5)
        #g2p2g(dt, self.pid[self.input_layer], self.grid[self.input_layer], self.grid_v[output_grid], self.grid_m[output_layer])
        #grid_normalization_and_gravity(dt, self.grid[output_layer])
        self.input_layer = output_layer

    def test(self, sims: Simulation, scene: myScene):
        pass

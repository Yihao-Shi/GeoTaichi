import numpy as np

from src.mpm.boundaries.BoundaryCore import *
from src.mpm.BaseKernel import validate_particle_displacement_
from src.mpm.engines.EngineKernel import *
from src.mpm.Simulation import Simulation
from src.mpm.SceneManager import myScene
from src.mpm.SpatialHashGrid import SpatialHashGrid
from src.utils.linalg import no_operation
from src.mpm.engines.FreeSurfaceDetection import *


class Engine(object):
    def __init__(self, sims: Simulation) -> None:
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
        self.manage_function(sims)

        self.reset_grid_messages = self.reset_grid_message
        if sims.sparse_grid:
            self.reset_grid_messages = self.deactivate_grid

        self.limit = 0.
        
    def choose_engine(self, sims: Simulation):
        if sims.mode == "Normal":
            if sims.mapping == "USL":
                self.compute = self.usl_updating
            elif sims.mapping == "USF":
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
        elif sims.mode == "Lightweight":
            self.compute = self.lightweight

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
        if int(scene.boundary.traction_list[0]) > 0:
            if sims.dimension == 2:
                self.apply_traction_constraints = self.traction_constraints_2D
            elif sims.dimension == 3:
                self.apply_traction_constraints = self.traction_constraints
        if int(scene.boundary.ptraction_list[0]) > 0:
            if sims.material_type == "TwoPhaseSingleLayer":
                self.apply_particle_traction_constraints = self.particle_traction_constraints_twophase
            else:
                if sims.mode == "Lightweight":
                    self.apply_particle_traction_constraints = self.lightweight_particle_traction_constraints
                else:
                    self.apply_particle_traction_constraints = self.particle_traction_constraints
        if sims.ptraction_method == "Virtual":
            if sims.dimension == 2:
                self.apply_particle_traction_constraints = self.virtual_stress_constraints_2D
            elif sims.dimension == 3:
                self.apply_particle_traction_constraints = self.virtual_stress_constraints

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
        self.compute_velocity_gradient = no_operation
        self.calculate_velocity_gradient = no_operation
        self.compute_particle_kinematic = no_operation
        self.calculate_interpolation = no_operation
        if sims.mode == "Normal":
            self.calculate_interpolation = self.calculate_interpolations

        if sims.dimension == 3:
            self.compute_particle_kinematic = self.compute_particle_kinematics
            self.compute_stress_strains = self.compute_stress_strain
            if sims.stabilize == "B-Bar Method":
                self.compute_forces = self.compute_force_bbar
                self.compute_velocity_gradient = self.update_velocity_gradient_bbar
                if sims.velocity_projection_scheme == "Affine":
                    self.compute_velocity_gradient = self.update_velocity_gradient_affine
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
                if sims.velocity_projection_scheme == "Affine":
                    self.compute_velocity_gradient = self.update_velocity_gradient_affine

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
        elif sims.dimension == 2:
            self.compute_particle_kinematic = self.compute_particle_kinematics
            if not sims.is_2DAxisy:
                self.compute_stress_strains = self.compute_stress_strain_2D
                if sims.stabilize == "B-Bar Method":
                    self.compute_forces = self.compute_force_bbar_2D
                    self.compute_velocity_gradient = self.update_velocity_gradient_bbar_2D
                    if sims.velocity_projection_scheme == "Affine":
                        self.compute_velocity_gradient = self.update_velocity_gradient_affine_2D
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
                    if sims.velocity_projection_scheme == "Affine":
                        self.compute_velocity_gradient = self.update_velocity_gradient_affine_2D

                if sims.gauss_number > 0:
                    self.compute_forces = self.compute_force_gauss_2D

                if sims.contact_detection:
                    self.pre_contact_calculate = self.calculate_precontact
                    if sims.contact_detection == "MPMContact":
                        self.compute_contact_force_ = self.compute_contact_force
                    elif sims.contact_detection == "GeoContact":
                        self.compute_contact_force_ = self.compute_geocontact_force
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
                    if sims.velocity_projection_scheme == "Affine":
                        raise RuntimeError("")
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
                    if sims.velocity_projection_scheme == "Affine":
                        raise RuntimeError("")

                if sims.gauss_number > 0:
                    self.compute_forces = self.compute_force_gauss_2D

                if sims.contact_detection:
                    self.pre_contact_calculate = self.calculate_precontact_2DAxisy
                    if sims.contact_detection == "MPMContact":
                        self.compute_contact_force_ = self.compute_contact_force
                    elif sims.contact_detection == "GeoContact":
                        self.compute_contact_force_ = self.compute_geocontact_force
                    elif sims.contact_detection == "DEMContact":
                        self.pre_contact_calculate = no_operation
                        self.compute_contact_force_ = self.compute_demcontact_force_2D
                    else:
                        raise RuntimeError("Wrong contact type!")

            if sims.pressure_smoothing:
                self.pressure_smoothing_ = self.pressure_smoothing

            if sims.velocity_projection_scheme == "Affine" or sims.velocity_projection_scheme == "Taylor":
                if sims.is_2DAxisy:
                    self.compute_nodal_kinematic = self.compute_nodal_kinematics_taylor_2DAxisy
                else:
                    self.compute_nodal_kinematic = self.compute_nodal_kinematics_taylor
        
        self.is_verlet_update = self.is_need_update_verlet_table
        if sims.neighbor_detection:
            if sims.coupling == "Lagrangian":
                self.execute_board_serach = self.update_verlet_table
                self.system_resolve = self.compute_nodal_kinematic
            self.bulid_neighbor_list = self.board_search

            self.free_surface_by_geometry = no_operation
            if sims.free_surface_detection:
                self.free_surface_by_geometry = self.detection_free_surface

            self.compute_boundary_direction = no_operation
            if sims.boundary_direction_detection:
                self.compute_boundary_direction = self.detection_boundary_direction
            self.compute_nodal_kinematic = no_operation

    def reset_particle_message(self, scene: myScene):
        contact_force_reset(int(scene.particleNum[0]), scene.particle)

    def reset_grid_message(self, scene: myScene):
        grid_reset(scene.mass_cut_off, scene.node)

    def deactivate_grid(self, scene: myScene):
        scene.parent.deactivate_all()

    def valid_contact(self, sims: Simulation, scene: myScene):
        if sims.contact_detection == "GeoContact":
            if scene.is_rigid[0] == 0 and scene.is_rigid[1] == 0:
                raise RuntimeError("GeoContact is only suitable for soil-structure interaction")

    def calculate_interpolations(self, sims: Simulation, scene: myScene):
        scene.element.calculate(scene.particleNum, scene.particle)

    def pressure_smoothing(self, scene: myScene):
        scene.node.pressure.fill(0)
        kernel_pressure_p2g(int(scene.particleNum[0]), scene.element.grid_nodes, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)
        kernel_grid_pressure(scene.mass_cut_off, scene.is_rigid, scene.node)
        kernel_pressure_g2p(int(scene.particleNum[0]), scene.element.grid_nodes, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def compute_nodal_kinematics(self, sims: Simulation, scene: myScene):
        kernel_mass_momentum_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def compute_nodal_kinematics_taylor(self, sims: Simulation, scene: myScene):
        kernel_mass_momentum_taylor_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), scene.element.gnum, scene.element.grid_size, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def compute_nodal_kinematics_taylor_2DAxisy(self, sims: Simulation, scene: myScene):
        kernel_mass_momentum_taylor_p2g_2DAxisy(scene.element.grid_nodes, int(scene.particleNum[0]), scene.element.gnum, scene.element.grid_size, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def update_verlet_table(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        scene.check_in_domain(sims)
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
        find_boundary_direction_by_geometry(neighbor.igrid_size, neighbor.cnum, int(scene.particleNum[0]), scene.particle, neighbor.sorted.object_list, neighbor.sorted.bin_count)

    def detection_free_surface(self, scene: myScene, neighbor: SpatialHashGrid):
        find_free_surface_by_geometry(neighbor.igrid_size, neighbor.cnum, int(scene.particleNum[0]), scene.particle, neighbor.sorted.object_list, neighbor.sorted.bin_count)

    def find_free_surface_by_density(self, sims, scene: myScene):
        self.calculate_interpolation(sims, scene)
        self.system_resolve(sims, scene)
        kernel_mass_g2p(scene.element.grid_nodes, scene.element.cell_volume, scene.element.node_size, scene.element.LnID, scene.element.shape_fn, scene.node, int(scene.particleNum[0]), scene.particle)

        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            assign_particle_free_surface(start_index, end_index, scene.particle, scene.material.materialID, scene.material.matProps[materialID + 1])

    def calculate_precontact_2DAxisy(self, sims: Simulation, scene: myScene):
        raise NotImplementedError

    def calculate_precontact(self, sims: Simulation, scene: myScene):
        raise NotImplementedError

    def compute_geocontact_force(self, sims: Simulation, scene: myScene):
        raise NotImplementedError

    def compute_demcontact_force_2D(self, sims: Simulation, scene: myScene):
        raise NotImplementedError
    
    def compute_contact_force(self, sims: Simulation, scene: myScene):
        raise NotImplementedError
    
    def compute_force(self, sims: Simulation, scene: myScene):
        raise NotImplementedError

    def compute_force_2D(self, sims: Simulation, scene: myScene):
        raise NotImplementedError

    def compute_force_2DAxisy(self, sims: Simulation, scene: myScene):
        raise NotImplementedError

    def compute_force_bbar_2DAxisy(self, sims: Simulation, scene: myScene):
        raise NotImplementedError
    
    def compute_force_mls(self, sims: Simulation, scene: myScene):
        raise NotImplementedError
        
    def compute_force_mls_2D(self, sims: Simulation, scene: myScene):
        raise NotImplementedError

    def compute_force_gauss(self, sims: Simulation, scene: myScene):
        raise NotImplementedError

    def compute_force_gauss_2D(self, sims: Simulation, scene: myScene):
        raise NotImplementedError

    def compute_force_bbar(self, sims: Simulation, scene: myScene):
        raise NotImplementedError

    def compute_force_bbar_2D(self, sims: Simulation, scene: myScene):
        raise NotImplementedError

    def compute_stress_strain(self, sims, scene):
        raise NotImplementedError
    
    def compute_stress_strain_2D(self, sims, scene):
        raise NotImplementedError
    
    def update_angular_velocity(self, sims: Simulation, scene: myScene):
        update_coupling_quanternion(int(scene.particleNum[0]), scene.particle, sims.dt)
    
    def update_velocity_gradient_2D(self, sims: Simulation, scene: myScene):
        raise NotImplementedError
    
    def update_velocity_gradient(self, sims: Simulation, scene: myScene):
        raise NotImplementedError
        
    def update_velocity_gradient_affine_2D(self, sims: Simulation, scene: myScene):
        raise NotImplementedError
        
    def update_velocity_gradient_affine(self, sims: Simulation, scene: myScene):
        raise NotImplementedError

    def update_velocity_gradient_bbar_2D(self, sims: Simulation, scene: myScene):
        raise NotImplementedError
    
    def update_velocity_gradient_bbar(self, sims: Simulation, scene: myScene):
        raise NotImplementedError
    
    def update_velocity_gradient_axisy_2D(self, sims: Simulation, scene: myScene):
        raise NotImplementedError
        
    def update_velocity_gradient_bbar_axisy_2D(self, sims: Simulation, scene: myScene):
        raise NotImplementedError
    
    def compute_stress_strain_velocity_projection_2D(self, sims: Simulation, scene: myScene):
        raise NotImplementedError

    def compute_stress_strain_velocity_projection(self, sims: Simulation, scene: myScene):
        raise NotImplementedError
    
    def update_velocity_gradient_fbar(self, sims: Simulation, scene: myScene):
        scene.node.jacobian.fill(0)
        self.calculate_velocity_gradient(sims, scene)
        kernel_jacobian_p2g(scene.element.grid_nodes, sims.dt, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)
        kernel_grid_jacobian(scene.volume_cut_off, scene.is_rigid, scene.node)
        kernel_update_velocity_gradient_fbar(sims.fbar_fraction, scene.volume_cut_off, scene.element.grid_nodes, sims.dt, int(scene.particleNum[0]), scene.node, scene.particle, 
                                             scene.element.LnID, scene.element.shape_fn, scene.element.node_size)
        
    def compute_external_force(self, sims: Simulation, scene: myScene):
        kernel_external_force_p2g(scene.element.grid_nodes, sims.gravity, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def compute_internal_force(self, sims, scene):
        raise NotImplementedError
    
    def compute_internal_force_bbar(self, sims, scene):
        raise NotImplementedError

    def compute_grid_kinematic(self, sims, scene):
        raise NotImplementedError

    def apply_kinematic_constraints(self, sims, scene):
        raise NotImplementedError

    def apply_dirichlet_constraints(self, sims, scene):
        raise NotImplementedError
    
    def virtual_stress_constraints(self, sims: Simulation, scene: myScene):
        apply_particle_virtual_traction_constraint(int(scene.particleNum[0]), scene.element.grid_size, scene.element.cnum, scene.boundary.particle_traction.auxiliary_cell, scene.boundary.particle_traction.auxiliary_node, scene.node, scene.particle)
        apply_virtual_traction_field(scene.element.grid_nodes, int(scene.particleNum[0]), scene.boundary.particle_traction.virtual_force, scene.boundary.particle_traction.virtual_stress, 
                                     scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.dshape_fn, scene.element.node_size, scene.boundary.particle_traction.auxiliary_node)
        
    def virtual_stress_constraints_2D(self, sims: Simulation, scene: myScene):
        apply_particle_virtual_traction_constraint_2D(int(scene.particleNum[0]), scene.element.grid_size, scene.element.cnum, scene.boundary.particle_traction.auxiliary_cell, scene.boundary.particle_traction.auxiliary_node, scene.node, scene.particle)
        apply_virtual_traction_field_2D(scene.element.grid_nodes, int(scene.particleNum[0]), scene.boundary.particle_traction.virtual_force, scene.boundary.particle_traction.virtual_stress, 
                                        scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.dshape_fn, scene.element.node_size, scene.boundary.particle_traction.auxiliary_node)

    def particle_traction_constraints(self, sims: Simulation, scene: myScene):
        apply_particle_traction_constraint(int(scene.boundary.ptraction_list[0]), scene.element.grid_nodes, scene.boundary.particle_traction, sims.dt, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def particle_traction_constraints_twophase(self, sims: Simulation, scene: myScene):
        apply_particle_traction_constraint_twophase(int(scene.boundary.ptraction_list[0]), scene.element.grid_nodes, scene.boundary.particle_traction, sims.dt, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def lightweight_particle_traction_constraints(self, sims: Simulation, scene: myScene):
        lightweight_particle_traction_constraint(int(scene.boundary.ptraction_list[0]), scene.element.gnum, scene.element.grid_size, scene.element.igrid_size, scene.boundary.particle_traction, sims.dt, scene.element.calLength, scene.element.boundary_type, scene.node, scene.particle)

    def traction_constraints(self, sims, scene):
        raise NotImplementedError
    
    def traction_constraints_2D(self, sims, scene):
        raise NotImplementedError

    def velocity_constraints(self, sims, scene):
        raise NotImplementedError
    
    def velocity_projection_updating(self, sims, scene):
        raise NotImplementedError
        
    def compute_particle_kinematics(self, sims, scene):
        raise NotImplementedError

    def postmapping_grid_velocity(self, sims, scene):
        raise NotImplementedError

    def is_need_update_verlet_table(self, scene: myScene):
        return validate_particle_displacement_(self.limit, int(scene.couplingNum[0]), scene.particle)

    def pre_calculation(self, sims, scene, neighbor):
        raise NotImplementedError

    def usl_updating(self, sims, scene):
        raise NotImplementedError

    def usf_updating(self, sims, scene):
        raise NotImplementedError

    def musl_updating(self, sims, scene):
        raise NotImplementedError

    def g2p2g(self, sims, scene):
        raise NotImplementedError
    
    def lightweight(self, sims, scene):
        raise NotImplementedError
    
    def test(self, sims, scene):
        raise NotImplementedError

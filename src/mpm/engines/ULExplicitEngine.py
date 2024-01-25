import numpy as np

from src.mpm.BaseKernel import *
from src.mpm.boundaries.BoundaryCore import *
from src.mpm.engines.Engine import Engine
from src.mpm.engines.EngineKernel import *
from src.mpm.SceneManager import myScene
from src.mpm.Simulation import Simulation
from src.mpm.SpatialHashGrid import SpatialHashGrid
from src.utils.FreeSurfaceDetection import *


class ULExplicitEngine(Engine):
    def __init__(self, sims) -> None:
        super().__init__(sims)

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

        self.stress_smoothing_ = None
        self.compute_nodal_kinematic = None
        self.compute_internal_forces = None
        self.execute_board_serach = None
        self.calculate_interpolation = None
        self.free_surface_by_geometry = None
        self.manage_function(sims)

        self.is_verlet_update = np.zeros(1, dtype=np.int32)
        self.limit = 0.
        
    def choose_engine(self, sims: Simulation):
        if sims.mapping == "USL":
            self.compute = self.usl_updating
        elif sims.mapping == "USF":
            self.compute = self.usf_updating
        elif sims.mapping == "MUSL":
            self.compute = self.musl_updating
        elif sims.mapping == "MLSMPM":
            self.compute = self.mls_updating
        else:
            raise ValueError(f"The mapping scheme {sims.mapping} is not supported yet")

    def choose_boundary_constraints(self, sims: Simulation, scene: myScene):
        self.apply_traction_constraints = self.no_operation
        self.apply_absorbing_constraints = self.no_operation
        self.apply_friction_constraints = self.no_operation
        self.apply_velocity_constraints = self.no_operation
        self.apply_reflection_constraints = self.no_operation
        self.apply_contact_velocity_constraints = self.no_operation
        self.apply_contact_reflection_constraints = self.no_operation

        if int(scene.velocity_list[0]) > 0:
            self.apply_velocity_constraints = self.velocity_constraints
            if sims.contact_detection:
                self.apply_contact_velocity_constraints = self.contact_velocity_constraints
        if int(scene.reflection_list[0]) > 0:
            self.apply_reflection_constraints = self.reflection_constraints
            if sims.contact_detection:
                self.apply_contact_reflection_constraints = self.contact_reflection_constraints
        if int(scene.friction_list[0]) > 0:
            self.apply_friction_constraints = self.friction_constraints
        if int(scene.absorbing_list[0]) > 0:
            self.apply_absorbing_constraints = self.absorbing_constraints
        if int(scene.traction_list[0]) > 0:
            self.apply_traction_constraints = self.traction_constraints


    def manage_function(self, sims: Simulation):
        self.pre_contact_calculate = self.no_operation
        self.compute_nodal_kinematic = self.compute_nodal_kinematics
        self.compute_contact_force_ = self.no_operation
        self.stress_smoothing_ = self.no_operation
        self.compute_internal_forces = self.no_operation
        self.compute_stress_strains = self.no_operation
        self.bulid_neighbor_list = self.no_operation_neighbor
        self.execute_board_serach = self.no_operation_neighbor
        self.system_resolve = self.no_operation
        self.calculate_interpolation = self.calculate_interpolations

        if sims.stabilize == "B-Bar Method":
            self.compute_internal_forces = self.compute_internal_force_bbar
            self.compute_stress_strains = self.compute_stress_strain_bbar
        elif sims.stabilize == "F-Bar Method":
            self.compute_internal_forces = self.compute_internal_force
            self.compute_stress_strains = self.compute_stress_strain_fbar
        else:
            self.compute_internal_forces = self.compute_internal_force
            self.compute_stress_strains = self.compute_stress_strain

        if sims.gauss_number > 0:
            self.compute_internal_forces = self.compute_internal_force_gauss

        if sims.contact_detection:
            self.pre_contact_calculate = self.calculate_precontact
            self.compute_contact_force_ = self.compute_contact_force

        if sims.stress_smoothing:
            self.stress_smoothing_ = self.stress_smoothing

        if sims.mapping == "MLSMPM":
            self.compute_nodal_kinematic = self.compute_nodal_kinematics_apic

        if sims.neighbor_detection:
            self.calculate_interpolation = self.no_operation
            self.compute_nodal_kinematic = self.no_operation
            if sims.coupling:
                self.execute_board_serach = self.update_verlet_table
                self.system_resolve = self.compute_nodal_kinematics
            else:
                self.bulid_neighbor_list = self.board_search

            self.free_surface_by_geometry = self.no_operation
            if sims.free_surface_detection:
                self.free_surface_by_geometry = self.detection_free_surface

            self.compute_boundary_direction = self.no_operation
            if sims.boundary_direction_detection:
                self.compute_boundary_direction = self.detection_boundary_direction

    def calculate_precontact(self, sims: Simulation, scene: myScene):
        kernel_calc_contact_normal(scene.element.grid_nodes, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)
        kernel_calc_contact_displacement(scene.element.grid_nodes, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def compute_contact_force(self, sims: Simulation, scene: myScene):
        kernel_calc_friction_contact(scene.mass_cut_off, scene.mu, scene.element.contact_position_offset, sims.dt, scene.is_rigid, scene.node)
        self.apply_contact_velocity_constraints(sims, scene)
        self.apply_contact_reflection_constraints(sims, scene)
        kernel_assemble_contact_force(scene.mass_cut_off, sims.dt, scene.node)

    def stress_smoothing(self, sims, scene: myScene):
        kernel_pressure_p2g(scene.element.gnum, scene.element.igrid_size, int(scene.particleNum[0]), scene.extra_node, scene.particle)
        kernel_grid_pressure(scene.mass_cut_off, scene.is_rigid, scene.node, scene.extra_node)
        kernel_pressure_g2p(scene.element.gnum, scene.element.igrid_size, scene.extra_node, int(scene.particleNum[0]), scene.particle)

    def compute_nodal_kinematics(self, sims: Simulation, scene: myScene):
        kernel_mass_momentum_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def compute_nodal_kinematics_apic(self, sims: Simulation, scene: myScene):
        kernel_momentum_apic_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def compute_grid_velcity(self, sims: Simulation, scene: myScene):
        kernel_compute_grid_velocity(scene.mass_cut_off, scene.node)

    def compute_stress_strain(self, sims: Simulation, scene: myScene):
        kernel_compute_stress_strain(scene.element.grid_nodes, sims.dt, int(scene.particleNum[0]), scene.node, scene.particle, scene.material.matProps, scene.material.stateVars,
                                     scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)
        
    def compute_stress_strain_apic(self, sims: Simulation, scene: myScene):
        kernel_compute_stress_strain_apic(scene.element.grid_nodes, sims.dt, int(scene.particleNum[0]), scene.node, scene.particle, scene.material.matProps, scene.material.stateVars,
                                          scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)    
        
    def compute_stress_strain_bbar(self, sims: Simulation, scene: myScene):
        kernel_compute_stress_strain_bbar(scene.element.grid_nodes, sims.dt, int(scene.particleNum[0]), scene.node, scene.particle, scene.material.matProps, scene.material.stateVars,
                                          scene.element.LnID, scene.element.dshape_fn, scene.element.dshape_fnc, scene.element.node_size)
        
    def compute_stress_strain_fbar(self, sims: Simulation, scene: myScene):
        extra_grid_reset(scene.volume_cut_off, scene.extra_node)
        kernel_calc_increment_velocity(scene.element.grid_nodes, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)
        kernel_volume_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), scene.extra_node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)
        kernel_jacobian_p2g(scene.element.grid_nodes, sims.dt, int(scene.particleNum[0]), scene.extra_node, scene.particle, scene.extra_particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)
        kernel_grid_jacobian(scene.volume_cut_off, scene.is_rigid, scene.extra_node)
        kernel_compute_stress_strain_fbar(scene.element.grid_nodes, sims.dt, int(scene.particleNum[0]), scene.extra_node, scene.particle, scene.extra_particle,
                                          scene.material.matProps, scene.material.stateVars, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def compute_external_force(self, sims: Simulation, scene: myScene):
        kernel_external_force_p2g(scene.element.grid_nodes, sims.gravity, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def compute_internal_force(self, sims: Simulation, scene: myScene):
        kernel_internal_force_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)

    def set_gauss_stress(self, sims: Simulation, scene: myScene):
        gauss_cell_reset(scene.element.cell, scene.element.gauss_cell)
        kernel_volume_p2c(scene.element.cnum, scene.element.igrid_size, int(scene.particleNum[0]), scene.element.cell, scene.particle)
        kernel_find_valid_element(scene.element.cell_volume, scene.element.cell)
        kernel_sum_cell_stress(sims.gauss_number, scene.element.grid_size, scene.element.igrid_size, scene.element.cnum, int(scene.particleNum[0]), scene.particle, scene.element.cell, scene.element.gauss_cell)
        kernel_compute_gauss_average_stress(sims.gauss_number, scene.volume_cut_off, scene.element.cell, scene.element.gauss_cell)
        kernel_average_pressure(sims.gauss_number, scene.element.cell, scene.element.gauss_cell)
    
    def compute_internal_force_gauss(self, sims: Simulation, scene: myScene):
        self.set_gauss_stress(sims, scene)
        kernel_internal_force_on_gauss_point_p2g(sims.gauss_number, scene.element.cnum, scene.element.gnum, scene.element.grid_size, scene.element.igrid_size, scene.node, 
                                                 scene.element.cell, scene.element.gauss_cell, scene.element.gauss_point.gpcoords, scene.element.gauss_point.weight)
        kernel_internal_force_on_material_point_p2g(scene.element.cnum, scene.element.gnum, scene.element.grid_size, scene.element.igrid_size, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.cell)

    def compute_internal_force_bbar(self, sims: Simulation, scene: myScene):
        kernel_internal_force_bbar_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.dshape_fn, scene.element.dshape_fnc, scene.element.node_size)

    def compute_grid_kinematic(self, sims: Simulation, scene: myScene):
        kernel_compute_grid_kinematic(scene.mass_cut_off, sims.background_damping, scene.node, sims.dt)

    def apply_kinematic_constraints(self, sims: Simulation, scene: myScene):
        self.apply_reflection_constraints(sims, scene)
        self.apply_velocity_constraints(sims, scene)

    def apply_dirichlet_constraints(self, sims: Simulation, scene: myScene):
        self.apply_reflection_constraints(sims, scene)
        self.apply_velocity_constraints(sims, scene)

    def traction_constraints(self, sims: Simulation, scene: myScene):
        apply_traction_constraint(int(scene.traction_list[0]), scene.traction_boundary, scene.node)
    
    def absorbing_constraints(self, sims: Simulation, scene: myScene):
        apply_absorbing_constraint(int(scene.absorbing_list[0]), scene.absorbing_boundary, scene.material, scene.node, scene.extra_node)

    def velocity_constraints(self, sims: Simulation, scene: myScene):
        apply_velocity_constraint(scene.mass_cut_off, int(scene.velocity_list[0]), scene.velocity_boundary, scene.is_rigid, scene.node)

    def reflection_constraints(self, sims: Simulation, scene: myScene):
        apply_reflection_constraint(scene.mass_cut_off, int(scene.reflection_list[0]), scene.reflection_boundary, scene.is_rigid, scene.node)

    def contact_velocity_constraints(self, sims: Simulation, scene: myScene):
        apply_contact_velocity_constraint(scene.mass_cut_off, int(scene.velocity_list[0]), scene.velocity_boundary, scene.node)

    def contact_reflection_constraints(self, sims: Simulation, scene: myScene):
        apply_contact_reflection_constraint(scene.mass_cut_off, int(scene.reflection_list[0]), scene.reflection_boundary, scene.node)

    def friction_constraints(self, sims: Simulation, scene: myScene):
        apply_friction_constraint(scene.mass_cut_off, int(scene.friction_list[0]), scene.friction_boundary, scene.is_rigid, scene.node, sims.dt)
        
    def compute_particle_kinematics(self, sims: Simulation, scene: myScene):
        kernel_kinemaitc_g2p(scene.element.grid_nodes, sims.alphaPIC, sims.dt, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def postmapping_grid_velocity(self, sims: Simulation, scene: myScene):
        kernel_reset_grid_velocity(scene.node)
        kernel_postmapping_kinemaitc(scene.element.grid_nodes, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def update_verlet_table(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        check_in_domain(sims.domain, int(scene.particleNum[0]), scene.particle)
        self.find_free_surface_by_density(sims, scene)
        neighbor.place_particles(scene)
        self.compute_boundary_direction(scene, neighbor)
        self.free_surface_by_geometry(scene, neighbor)
        scene.reset_verlet_disp()

    def board_search(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        if self.is_need_update_verlet_table(scene) == 1:
            self.update_verlet_table(sims, scene, neighbor)
        else:
            self.compute_nodal_kinematics(sims, scene)

    def detection_boundary_direction(self, scene: myScene, neighbor: SpatialHashGrid):
        find_boundary_direction_by_geometry(neighbor.igrid_size, neighbor.cnum, int(scene.particleNum[0]), scene.particle, neighbor.ParticleID, neighbor.current, neighbor.count)

    def detection_free_surface(self, scene: myScene, neighbor: SpatialHashGrid):
        find_free_surface_by_geometry(neighbor.igrid_size, neighbor.cnum, int(scene.particleNum[0]), scene.particle, neighbor.ParticleID, neighbor.current, neighbor.count)

    def find_free_surface_by_density(self, sims, scene: myScene):
        scene.element.calculate(scene.particleNum, scene.particle)
        self.compute_nodal_kinematics(sims, scene)
        kernel_mass_g2p(scene.element.grid_nodes, scene.element.cell_volume, scene.element.node_size, scene.element.LnID, scene.element.shape_fn, scene.node, int(scene.particleNum[0]), scene.particle)
        assign_particle_free_surface(int(scene.particleNum[0]), scene.particle, scene.material.matProps)

    def pre_calculation(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        if sims.neighbor_detection:
            grid_mass_reset(scene.mass_cut_off, scene.node)
            check_in_domain(sims.domain, int(scene.particleNum[0]), scene.particle)
            self.find_free_surface_by_density(sims, scene)
            neighbor.place_particles(scene)
            self.compute_boundary_direction(scene, neighbor)
            self.free_surface_by_geometry(scene, neighbor)
            grid_mass_reset(scene.mass_cut_off, scene.node)

        if sims.gauss_number > 0.:
            scene.element.calculate(scene.particleNum, scene.particle)
            self.set_gauss_stress(sims, scene)

        self.limit = sims.verlet_distance * sims.verlet_distance

    def calculate_interpolations(self, sims: Simulation, scene: myScene):
        scene.element.calculate(scene.particleNum, scene.particle)

    def usl_updating(self, sims: Simulation, scene: myScene):
        self.calculate_interpolation(sims, scene)
        self.compute_nodal_kinematic(sims, scene)
        self.compute_grid_velcity(sims, scene)
        self.compute_external_force(sims, scene)
        self.compute_internal_forces(sims, scene)
        self.apply_traction_constraints(sims, scene)
        self.apply_absorbing_constraints(sims, scene)
        self.apply_friction_constraints(sims, scene)
        self.compute_grid_kinematic(sims, scene)
        self.pre_contact_calculate(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.compute_contact_force_(sims, scene)
        self.compute_particle_kinematics(sims, scene)
        self.compute_stress_strains(sims, scene)
        self.stress_smoothing_(sims, scene)

    def usf_updating(self, sims: Simulation, scene: myScene):
        self.calculate_interpolation(sims, scene)
        self.compute_nodal_kinematic(sims, scene)
        self.compute_grid_velcity(sims, scene)
        self.apply_dirichlet_constraints(sims, scene)
        self.compute_stress_strains(sims, scene)
        self.stress_smoothing_(sims, scene)
        self.compute_external_force(sims, scene)
        self.compute_internal_forces(sims, scene)
        self.apply_traction_constraints(sims, scene)
        self.apply_absorbing_constraints(sims, scene)
        self.apply_friction_constraints(sims, scene)
        self.compute_grid_kinematic(sims, scene)
        self.pre_contact_calculate(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.compute_contact_force_(sims, scene)
        self.compute_particle_kinematics(sims, scene)

    def musl_updating(self, sims: Simulation, scene: myScene):
        self.calculate_interpolation(sims, scene)
        self.compute_nodal_kinematic(sims, scene)
        self.compute_grid_velcity(sims, scene)
        self.compute_external_force(sims, scene)
        self.compute_internal_forces(sims, scene)
        self.apply_traction_constraints(sims, scene)
        self.apply_absorbing_constraints(sims, scene)
        self.apply_friction_constraints(sims, scene)
        self.compute_grid_kinematic(sims, scene)
        self.pre_contact_calculate(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.compute_contact_force_(sims, scene)
        self.compute_particle_kinematics(sims, scene)
        self.postmapping_grid_velocity(sims, scene)
        self.apply_friction_constraints(sims, scene)
        self.compute_grid_velcity(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.compute_stress_strains(sims, scene)
        self.stress_smoothing_(sims, scene)

    def mls_updating(self, sims: Simulation, scene: myScene):
        self.calculate_interpolation(sims, scene)
        self.compute_nodal_kinematic(sims, scene)
        self.compute_grid_velcity(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.compute_stress_strain_apic(sims, scene)
        self.stress_smoothing_(sims, scene)
        self.compute_external_force(sims, scene)
        self.compute_internal_forces(sims, scene)
        self.apply_traction_constraints(sims, scene)
        self.apply_absorbing_constraints(sims, scene)
        self.apply_friction_constraints(sims, scene)
        self.compute_grid_kinematic(sims, scene)
        self.pre_contact_calculate(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.compute_contact_force_(sims, scene)
        self.compute_particle_kinematics(sims, scene)
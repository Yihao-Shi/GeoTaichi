from src.mpm.boundaries.BoundaryCore import *
from src.mpm.engines.Engine import Engine
from src.mpm.engines.EngineKernel import *
from src.mpm.SceneManager import myScene
from src.mpm.Simulation import Simulation
from src.mpm.SpatialHashGrid import SpatialHashGrid
from src.utils.FreeSurfaceDetection import *
from src.utils.linalg import no_operation


class TLExplicitEngine(Engine):
    def __init__(self, sims) -> None:
        self.compute = None
        self.compute_stress_strains = None
        self.bulid_neighbor_list = None
        self.apply_traction_constraints = None
        self.apply_absorbing_constraints = None
        self.apply_velocity_constraints = None
        self.apply_friction_constraints = None
        self.apply_reflection_constraints = None
        super().__init__(sims)
        
    def choose_engine(self, sims: Simulation):
        if sims.mapping == "USL":
            self.compute = self.usl_updating
        elif sims.mapping == "USF":
            self.compute = self.usf_updating
        elif sims.mapping == "MUSL":
            self.compute = self.musl_updating
        else:
            raise ValueError(f"The mapping scheme {sims.mapping} is not supported yet")

    def choose_boundary_constraints(self, sims: Simulation, scene: myScene):
        self.apply_traction_constraints = no_operation
        self.apply_absorbing_constraints = no_operation
        self.apply_friction_constraints = no_operation
        self.apply_velocity_constraints = no_operation
        self.apply_reflection_constraints = no_operation
        self.apply_particle_traction_constraints = no_operation

        if int(scene.boundary.velocity_list[0]) > 0:
            self.apply_velocity_constraints = self.velocity_constraints
        if int(scene.boundary.reflection_list[0]) > 0:
            self.apply_reflection_constraints = self.reflection_constraints
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
            self.apply_particle_traction_constraints = self.particle_traction_constraints


    def manage_function(self, sims: Simulation):
        self.compute_forces = no_operation
        self.compute_stress_strains = no_operation
        self.bulid_neighbor_list = no_operation
        self.system_resolve = no_operation
        self.execute_board_serach = no_operation

        if sims.stabilize == "B-Bar Method":
            raise RuntimeError("Total lagrangian material point method do not support B-bar method")
        elif sims.stabilize == "F-Bar Method":
            self.compute_forces = self.compute_force
            self.compute_stress_strains = self.compute_stress_strain_fbar
        else:
            self.compute_forces = self.compute_force
            self.compute_stress_strains = self.compute_stress_strain

        if sims.contact_detection:
            raise RuntimeError("Total lagrangian material point method do not support two body contact")

        if sims.pressure_smoothing:
            raise RuntimeError("Total lagrangian material point method do not support pressure smoothing")
        
        if sims.gauss_number > 0:
            raise RuntimeError("Total lagrangian material point method do not support gauss integration")

        if sims.neighbor_detection:
            raise RuntimeError("Total lagrangian material point method do not support free surface detection/boundary direction detection")
        
    def valid_contact(self, sims, scene):
        pass

    def reset_grid_message(self, scene: myScene):
        tlgrid_reset(scene.mass_cut_off, scene.node)

    def compute_nodal_kinematics(self, sims: Simulation, scene: myScene):
        kernel_momentum_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def compute_nodal_mass(self, sims: Simulation, scene: myScene):
        kernel_mass_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def compute_grid_velcity(self, sims: Simulation, scene: myScene):
        kernel_compute_grid_velocity(scene.mass_cut_off, scene.node)

    def compute_stress_strain(self, sims: Simulation, scene: myScene):
        kernel_compute_reference_stress_strain(scene.element.grid_nodes, sims.dt, int(scene.particleNum[0]), scene.node, scene.particle, scene.material.matProps, scene.material.stateVars,
                                               scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)  

    def compute_force(self, sims: Simulation, scene: myScene):
        kernel_reference_force_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), sims.gravity, scene.node, scene.particle, scene.material.stateVars, scene.element.LnID, scene.element.shape_fn, scene.element.dshape_fn, scene.element.node_size)

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

    def friction_constraints(self, sims: Simulation, scene: myScene):
        apply_friction_constraint(scene.mass_cut_off, int(scene.boundary.friction_list[0]), scene.boundary.friction_boundary, scene.is_rigid, scene.node, sims.dt)
        
    def compute_particle_kinematics(self, sims: Simulation, scene: myScene):
        kernel_kinemaitc_g2p(scene.element.grid_nodes, sims.alphaPIC, sims.dt, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def postmapping_grid_velocity(self, sims: Simulation, scene: myScene):
        kernel_reset_grid_velocity(scene.node)
        kernel_postmapping_kinemaitc(scene.element.grid_nodes, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def update_verlet_table(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        scene.check_in_domain(sims.domain, int(scene.particleNum[0]), scene.particle)
        neighbor.place_particles(scene)
        scene.reset_verlet_disp()

    def board_search(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        if self.is_need_update_verlet_table(scene) == 1:
            self.update_verlet_table(sims, scene, neighbor)
        else:
            self.compute_nodal_kinematics(sims, scene)

    def detection_free_surface(self, scene: myScene, neighbor: SpatialHashGrid):
        find_free_surface_by_geometry(neighbor.igrid_size, neighbor.cnum, int(scene.particleNum[0]), scene.particle, neighbor.ParticleID, neighbor.current, neighbor.count)

    def find_free_surface_by_density(self, sims, scene: myScene):
        scene.element.calculate(scene.particleNum, scene.particle)
        self.compute_nodal_kinematics(sims, scene)
        kernel_mass_g2p(scene.element.grid_nodes, scene.element.cell_volume, scene.element.node_size, scene.element.LnID, scene.element.shape_fn, scene.node, scene.particleNum, scene.particle)
        assign_particle_free_surface(int(scene.particleNum[0]), scene.particle, scene.material.matProps)

    def pre_calculation(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        self.limit = sims.verlet_distance * sims.verlet_distance
        scene.element.calculate_characteristic_length(sims, int(scene.particleNum[0]), scene.particle, scene.psize)
        scene.element.calculate(scene.particleNum, scene.particle)
        self.compute_nodal_mass(sims, scene)

    def usl_updating(self, sims: Simulation, scene: myScene):
        self.compute_nodal_kinematics(sims, scene)
        self.compute_grid_velcity(sims, scene)
        self.apply_particle_traction_constraints(sims, scene)
        self.compute_forces(sims, scene)
        self.apply_traction_constraints(sims, scene)
        self.apply_absorbing_constraints(sims, scene)
        self.compute_grid_kinematic(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.compute_particle_kinematics(sims, scene)
        self.compute_stress_strains(sims, scene)

    def usf_updating(self, sims: Simulation, scene: myScene):
        self.compute_nodal_kinematics(sims, scene)
        self.compute_grid_velcity(sims, scene)
        self.apply_dirichlet_constraints(sims, scene)
        self.compute_stress_strains(sims, scene)
        self.apply_particle_traction_constraints(sims, scene)
        self.compute_forces(sims, scene)
        self.apply_traction_constraints(sims, scene)
        self.apply_absorbing_constraints(sims, scene)
        self.compute_grid_kinematic(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.compute_particle_kinematics(sims, scene)

    def musl_updating(self, sims: Simulation, scene: myScene):
        self.compute_nodal_kinematics(sims, scene)
        self.compute_grid_velcity(sims, scene)
        self.apply_particle_traction_constraints(sims, scene)
        self.compute_forces(sims, scene)
        self.apply_traction_constraints(sims, scene)
        self.apply_absorbing_constraints(sims, scene)
        self.apply_friction_constraints(sims, scene)
        self.compute_grid_kinematic(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.compute_particle_kinematics(sims, scene)
        self.postmapping_grid_velocity(sims, scene)
        self.compute_grid_velcity(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.compute_stress_strains(sims, scene)

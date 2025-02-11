from src.mpm.engines.MatrixFree import MatrixFree
from src.mpm.engines.Engine import Engine
from src.mpm.engines.EngineKernel import *
from src.mpm.SceneManager import myScene
from src.mpm.Simulation import Simulation
from src.mpm.SpatialHashGrid import SpatialHashGrid
from src.utils.FreeSurfaceDetection import *
from src.utils.linalg import no_operation


class ImplicitEngine(Engine):
    def __init__(self, sims) -> None:
        self.update_nodal_displacements = None
        self.compute_internal_forces = None
        super().__init__(sims)

    def manage_function(self, sims: Simulation):
        super().manage_function(sims)
        if sims.dimension == 2:
            self.update_nodal_displacements = self.update_nodal_displacement_2D
            self.compute_internal_forces = self.compute_internal_force_2D
        elif sims.dimension == 3:
            self.update_nodal_displacements = self.update_nodal_displacement
            self.compute_internal_forces = self.compute_internal_force
        
    def choose_engine(self, sims: Simulation):
        if sims.integration_scheme == "Newmark":
            self.compute = self.newmark_integrate
        else:
            raise ValueError(f"The integration scheme {sims.integration_scheme} is not supported yet")

    def choose_boundary_constraints(self, sims: Simulation, scene: myScene):
        self.apply_traction_constraints = no_operation
        self.apply_particle_traction_constraints = no_operation
        if scene.boundary.traction_list[None] > 0:
            self.apply_traction_constraints = self.traction_constraints
        if int(scene.boundary.ptraction_list[0]) > 0:
            self.apply_particle_traction_constraints = self.particle_traction_constraints

    def reset_iterative_grid_message(self, scene: myScene):
        grid_internal_force_reset(scene.mass_cut_off, scene.node)

    def compute_nodal_mass(self, sims: Simulation, scene: myScene):
        kernel_mass_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def compute_nodal_kinematics(self, sims: Simulation, scene: myScene):
        kernel_mass_momentum_acceleration_force_ip2g(scene.element.grid_nodes, int(scene.particleNum[0]), sims.gravity, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def compute_grid_velcity(self, sims: Simulation, scene: myScene):
        kernel_compute_grid_velocity_acceleration(scene.mass_cut_off, scene.node)

    def compute_nodal_kinematics_newmark(self, sims: Simulation, scene: myScene):
        kernel_compute_nodal_kinematics_newmark(sims.newmark_beta, sims.newmark_gamma, scene.mass_cut_off, scene.node, sims.dt)

    def compute_internal_force(self, sims: Simulation, scene: myScene):
        kernel_internal_force_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)

    def compute_internal_force_2D(self, sims: Simulation, scene: myScene):
        kernel_internal_force_p2g_2D(scene.element.grid_nodes, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)

    def update_nodal_displacement(self, scene: myScene):
        kernel_update_nodal_disp(scene.mass_cut_off, scene.node, self.matrix_free.unknow_vector)

    def update_nodal_displacement_2D(self, scene: myScene):
        kernel_update_nodal_disp_2D(scene.mass_cut_off, scene.node, self.matrix_free.unknow_vector)

    def compute_stress_strain(self, sims: Simulation, scene: myScene):
        kernel_compute_stress_strain_newmark(sims.dt, int(scene.particleNum[0]), scene.particle, scene.material.matProps, scene.material.stateVars, scene.material.stiffness_matrix)
        
    def compute_stress_strain_2D(self, sims: Simulation, scene: myScene):
        kernel_compute_stress_strain_newmark_2D(sims.dt, int(scene.particleNum[0]), scene.particle, scene.material.matProps, scene.material.stateVars, scene.material.stiffness_matrix)

    def update_velocity_gradient_2D(self, sims: Simulation, scene: myScene):
        kernel_update_displacement_gradient_2D(scene.element.grid_nodes, int(scene.particleNum[0]), sims.dt, scene.node, scene.particle, scene.material.matProps, scene.material.stateVars,
                                        scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)
    
    def update_velocity_gradient(self, sims: Simulation, scene: myScene):
        kernel_update_displacement_gradient(scene.element.grid_nodes, int(scene.particleNum[0]), sims.dt, scene.node, scene.particle, scene.material.matProps, scene.material.stateVars,
                                        scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)
        
    def update_velocity_gradient_affine_2D(self, sims: Simulation, scene: myScene):
        kernel_update_displacement_gradient_affine_2D(scene.element.grid_nodes, int(scene.particleNum[0]), scene.element.gnum, scene.element.grid_size, sims.dt, scene.node, scene.particle, 
                                                  scene.material.matProps, scene.material.stateVars, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)
        
    def update_velocity_gradient_affine(self, sims: Simulation, scene: myScene):
        kernel_update_displacement_gradient_affine(scene.element.grid_nodes, int(scene.particleNum[0]), scene.element.gnum, scene.element.grid_size, sims.dt, scene.node, scene.particle, 
                                               scene.material.matProps, scene.material.stateVars, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def update_velocity_gradient_bbar_2D(self, sims: Simulation, scene: myScene):
        kernel_update_displacement_gradient_bbar_2D(scene.element.grid_nodes, int(scene.particleNum[0]), sims.dt, scene.node, scene.particle, scene.material.matProps, scene.material.stateVars,
                                                scene.element.LnID, scene.element.dshape_fn, scene.element.dshape_fnc, scene.element.node_size)
    
    def update_velocity_gradient_bbar(self, sims: Simulation, scene: myScene):
        kernel_update_displacement_gradient_bbar(scene.element.grid_nodes, int(scene.particleNum[0]), sims.dt, scene.node, scene.particle, scene.material.matProps, scene.material.stateVars,
                                             scene.element.LnID, scene.element.dshape_fn, scene.element.dshape_fnc, scene.element.node_size)
    
    def compute_particle_kinematics(self, sims: Simulation, scene: myScene):
        kernel_kinemaitc_ig2p(scene.element.grid_nodes, sims.alphaPIC, sims.dt, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def finalize_newton_raphson_iteration(self, sims: Simulation, scene: myScene):
        kernel_update_stress_strain_newmark(int(scene.particleNum[0]), scene.particle, sims.dt)

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
        self.compute_nodal_mass(sims, scene)
        scene.material.compute_elasto_plastic_stiffness(int(scene.particleNum[0]), scene.particle)
        total_dofs = scene.element.initial_estimate_active_dofs(scene.mass_cut_off, scene.node)
        grid_mass_reset(scene.mass_cut_off, scene.node)
        
        self.matrix_free = MatrixFree()
        self.matrix_free.manage_function(sims, scene)
        self.matrix_free.set_matrix_vector(total_dofs, sims, scene)
        self.matrix_free.manage_operator(scene)
        self.matrix_free.operator.update_active_dofs(total_dofs)

    def newmark_integrate(self, sims: Simulation, scene: myScene, neighbor=None):
        self.calculate_interpolation(sims, scene)
        self.compute_nodal_kinematics(sims, scene)
        self.apply_particle_traction_constraints(sims, scene)
        self.compute_grid_velcity(sims, scene)
        total_dofs = scene.element.find_active_nodes(scene.mass_cut_off, scene.node)
        self.matrix_free.operator.update_active_dofs(total_dofs)
        self.matrix_free.assemble_mass_matrix(sims, scene)
        
        iter_num = 0
        convergence = False
        while not convergence and iter_num < sims.iter_max:
            self.reset_iterative_grid_message(scene)
            self.compute_internal_forces(sims, scene)
            self.matrix_free.run(sims, scene)
            self.update_nodal_displacements(scene)
            self.compute_velocity_gradient(sims, scene)
            self.compute_stress_strains(sims, scene)
            convergence = self.matrix_free.compute_residual_error(scene.mass_cut_off, scene.node, self.matrix_free.unknow_vector) < sims.displacement_tolerance
            iter_num += 1
        
        self.finalize_newton_raphson_iteration(sims, scene)
        self.compute_nodal_kinematics_newmark(sims, scene)
        self.compute_particle_kinematic(sims, scene)
        self.matrix_free.calculate_reaction_force(scene)


from src.mpm.boundaries.BoundaryCore import *
from src.mpm.engines.ULExplicitEngine import ULExplicitEngine
from src.mpm.engines.EngineKernel import *
from src.mpm.SceneManager import myScene
from src.mpm.Simulation import Simulation
from src.mpm.SpatialHashGrid import SpatialHashGrid
from src.utils.linalg import no_operation
from src.mpm.engines.FreeSurfaceDetection import *


class ULExplicitTwoPhaseEngine(ULExplicitEngine):
    def __init__(self, sims) -> None:
        super().__init__(sims)

    def compute_particle_kinematics(self, sims: Simulation, scene: myScene):
        kernel_kinemaitc_g2p_twophase(scene.element.grid_nodes, sims.alphaPIC, sims.dt, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def compute_nodal_kinematics(self, sims: Simulation, scene: myScene):
        kernel_mass_momentum_p2g_twophase(scene.element.grid_nodes, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def compute_grid_velcity(self, sims: Simulation, scene: myScene):
        kernel_compute_grid_velocity_twophase(scene.mass_cut_off, scene.node)

    def apply_dirichlet_constraints(self, sims: Simulation, scene: myScene):
        #self.apply_reflection_constraints(sims, scene)
        self.apply_velocity_constraints(sims, scene)
    
    def compute_stress_strain(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_compute_stress_strain_twophase(start_index, end_index, sims.dt, scene.particle, scene.material.materialID, scene.material.matProps[materialID + 1], scene.material.stateVars)
        
    def compute_stress_strain_2D(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_compute_stress_strain_twophase2D(start_index, end_index, sims.dt, scene.particle, scene.material.materialID, scene.material.matProps[materialID + 1], scene.material.stateVars)
        
    def update_velocity_gradient_2D(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_update_velocity_gradient_twophase_2D(scene.element.grid_nodes, start_index, end_index, sims.dt, scene.node, scene.particle, scene.material.materialID, scene.material.matProps[materialID + 1], scene.material.stateVars,
                                                    scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)
        
    def update_velocity_gradient_bbar_2D(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_update_velocity_gradient_bbar_twophase_2D(scene.element.grid_nodes, start_index, end_index, sims.dt, scene.node, scene.particle, scene.material.materialID, scene.material.matProps[materialID + 1], scene.material.stateVars,
                                                         scene.element.LnID, scene.element.dshape_fn, scene.element.dshape_fnc, scene.element.node_size)
    
    def update_velocity_gradient(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_update_velocity_gradient_twophase(scene.element.grid_nodes, start_index, end_index, sims.dt, scene.node, scene.particle, scene.material.materialID, scene.material.matProps[materialID + 1], scene.material.stateVars,
                                                 scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)
        
    def compute_force(self, sims: Simulation, scene: myScene):
        kernel_force_p2g_twophase(scene.element.grid_nodes, int(scene.particleNum[0]), sims.gravity, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.dshape_fn, scene.element.node_size)
        
    def compute_force_2D(self, sims: Simulation, scene: myScene):
        kernel_force_p2g_twophase2D(scene.element.grid_nodes, int(scene.particleNum[0]), sims.gravity, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.dshape_fn, scene.element.node_size)

    def compute_force_bbar_2D(self, sims: Simulation, scene: myScene):
        kernel_force_bbar_p2g_twophase2D(scene.element.grid_nodes, int(scene.particleNum[0]), sims.gravity, scene.node, scene.particle, scene.element.LnID, 
                                         scene.element.shape_fn, scene.element.shape_fnc, scene.element.dshape_fnc, scene.element.node_size)

    def compute_force_2DAxisy(self, sims: Simulation, scene: myScene):
        kernel_force_p2g_twophase_2DAxisy(scene.element.grid_nodes, int(scene.particleNum[0]), sims.gravity, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.dshape_fn, scene.element.node_size)

    def compute_force_bbar_2DAxisy(self, sims: Simulation, scene: myScene):
        kernel_force_bbar_p2g_twophase_2DAxisy(scene.element.grid_nodes, int(scene.particleNum[0]), sims.gravity, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.dshape_fn, scene.element.node_size)
        
    def compute_external_force(self, sims: Simulation, scene: myScene):
        kernel_external_force_p2g_twophase(scene.element.grid_nodes, sims.gravity, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def compute_internal_force_2D(self, sims: Simulation, scene: myScene):
        kernel_internal_force_p2g_twophase2D(scene.element.grid_nodes, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.dshape_fn, scene.element.node_size)

    def compute_internal_force_bbar_2D(self, sims: Simulation, scene: myScene):
        kernel_internal_force_bbar_p2g_twophase2D(scene.element.grid_nodes, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.dshape_fn, scene.element.dshape_fnc, scene.element.node_size)

    def compute_grid_kinematic(self, sims: Simulation, scene: myScene):
        kernel_compute_grid_kinematic_fluid(scene.mass_cut_off, sims.background_damping, scene.node, sims.dt)
        kernel_compute_grid_kinematic_solid(scene.mass_cut_off, sims.background_damping, scene.node, sims.dt)

    def postmapping_grid_velocity(self, sims: Simulation, scene: myScene):
        kernel_reset_grid_velocity_twophase2D(scene.node)
        kernel_postmapping_kinemaitc_twophase2D(scene.element.grid_nodes, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def pre_calculation(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        scene.element.calculate_characteristic_length(sims, int(scene.particleNum[0]), scene.particle, scene.psize)
        if sims.neighbor_detection:
            grid_mass_reset(scene.mass_cut_off, scene.node)
            scene.check_in_domain(sims)
            self.find_free_surface_by_density(sims, scene)
            neighbor.place_particles(scene)
            self.compute_boundary_direction(scene, neighbor)
            self.free_surface_by_geometry(scene, neighbor)
            grid_mass_reset(scene.mass_cut_off, scene.node)
        self.limit = sims.verlet_distance * sims.verlet_distance
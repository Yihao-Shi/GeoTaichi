from src.mpm.engines.ULExplicitEngine import ULExplicitEngine
from src.mpm.engines.EngineKernel import *
from src.mpm.SceneManager import myScene
from src.mpm.Simulation import Simulation
from src.mpm.SpatialHashGrid import SpatialHashGrid


class TLExplicitEngine(ULExplicitEngine):
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

    def choose_boundary_constraints(self, sims: Simulation, scene: myScene):
        super().choose_boundary_constraints(sims, scene)
        if int(scene.boundary.reflection_list[0]) > 0:
            self.apply_reflection_constraints = self.reflection_constraints
        if int(scene.boundary.friction_list[0]) > 0:
            self.apply_friction_constraints = self.friction_constraints
        if int(scene.boundary.absorbing_list[0]) > 0:
            self.apply_absorbing_constraints = self.absorbing_constraints
        
    def valid_contact(self, sims, scene):
        pass

    def reset_grid_message(self, scene: myScene):
        tlgrid_reset(scene.mass_cut_off, scene.node)

    def compute_nodal_kinematics(self, sims: Simulation, scene: myScene):
        kernel_momentum_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def compute_nodal_mass(self, sims: Simulation, scene: myScene):
        kernel_mass_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.node_size)

    def compute_stress_strain(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_compute_reference_stress_strain(start_index, end_index, sims.dt, scene.particle, scene.material.materialID, scene.material.matProps[materialID + 1], scene.material.stateVars)  
        
    def compute_force(self, sims: Simulation, scene: myScene):
        kernel_reference_force_p2g(scene.element.grid_nodes, int(scene.particleNum[0]), sims.gravity, scene.node, scene.particle, scene.element.LnID, scene.element.shape_fn, scene.element.dshape_fn, scene.element.node_size)

    def pre_calculation(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        self.limit = sims.verlet_distance * sims.verlet_distance
        scene.element.calculate_characteristic_length(sims, int(scene.particleNum[0]), scene.particle, scene.psize)
        scene.element.calculate(scene.particleNum, scene.particle)
        self.compute_nodal_mass(sims, scene)

    def usl_updating(self, sims: Simulation, scene: myScene, neighbor=None):
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

    def usf_updating(self, sims: Simulation, scene: myScene, neighbor=None):
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

    def musl_updating(self, sims: Simulation, scene: myScene, neighbor=None):
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

    def velocity_projection_updating(self, sims: Simulation, scene: myScene, neighbor=None):
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

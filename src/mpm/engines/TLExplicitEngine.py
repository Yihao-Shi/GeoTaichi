from src.mpm.engines.ULExplicitEngine import ULExplicitEngine
from src.mpm.engines.EngineKernel import *
from src.mpm.SceneManager import myScene
from src.mpm.Simulation import Simulation
from src.mpm.SpatialHashGrid import SpatialHashGrid
from src.utils.linalg import no_operation


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
        if sims.dimension == 2:
            self.compute_forces = self.compute_force_2D
            self.compute_stress_strains = self.compute_stress_strain_2D
        elif sims.dimension == 3:
            self.compute_forces = self.compute_force
            self.compute_stress_strains = self.compute_stress_strain

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

    def reset_adaptive_grid_message(self, scene: myScene):
        grid_reset(scene.mass_cut_off, scene.node)

    def reset_reference_grid_message(self, scene: myScene):
        tlgrid_reset(scene.mass_cut_off, scene.node)

    def compute_nodal_kinematics(self, sims: Simulation, scene: myScene):
        kernel_momentum_p2g(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.element.gnum,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )

    def compute_adaptive_nodal_kinematics(self, sims: Simulation, scene: myScene):
        kernel_mass_momentum_p2g(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.element.gnum,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )

    def _bind_adaptive_runtime_functions(self, scene: myScene):
        adaptive = bool(getattr(scene.element, "adaptive", False))
        self.update_adaptive_interpolation = self.calculate_interpolation if adaptive else no_operation
        self.compute_nodal_kinematic = (
            self.compute_adaptive_nodal_kinematics if adaptive else self.compute_nodal_kinematics
        )
        selected_reset = self.reset_adaptive_grid_message if adaptive else self.reset_reference_grid_message
        self.reset_grid_message = selected_reset
        if scene.sparse_grid is None:
            self.reset_grid_messages = selected_reset

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

    def compute_stress_strain(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_compute_reference_stress_strain(
                start_index,
                end_index,
                sims.dt,
                scene.particle,
                scene.material.materialID,
                scene.material.matProps[materialID + 1],
                scene.material.stateVars,
            )

    def compute_stress_strain_2D(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_compute_reference_stress_strain_2D(
                start_index,
                end_index,
                sims.dt,
                scene.particle,
                scene.material.materialID,
                scene.material.matProps[materialID + 1],
                scene.material.stateVars,
            )

    def compute_force(self, sims: Simulation, scene: myScene):
        kernel_reference_force_p2g(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.element.gnum,
            sims.gravity,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.dshape_fn,
            scene.element.node_size,
        )

    def compute_force_2D(self, sims: Simulation, scene: myScene):
        kernel_reference_force_p2g_2D(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.element.gnum,
            sims.gravity,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.dshape_fn,
            scene.element.node_size,
        )

    def pre_calculation(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        self._bind_adaptive_runtime_functions(scene)
        self.limit = sims.verlet_distance * sims.verlet_distance
        scene.element.calculate_characteristic_length(sims, int(scene.particleNum[0]), scene.particle, scene.psize)
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

    def usl_updating(self, sims: Simulation, scene: myScene, neighbor=None):
        sims.timer.begin("Shape function")
        self.update_adaptive_interpolation(sims, scene)
        sims.timer.end("Shape function")
        sims.timer.begin("P2G")
        self.compute_nodal_kinematic(sims, scene)
        self.compute_grid_velcity(sims, scene)
        self.compute_forces(sims, scene)
        sims.timer.end("P2G")
        sims.timer.begin("Traction boundary")
        self.apply_particle_traction_constraints(sims, scene)
        self.apply_traction_constraints(sims, scene)
        self.apply_absorbing_constraints(sims, scene)
        sims.timer.end("Traction boundary")
        sims.timer.begin("Grid kinematic")
        self.compute_grid_kinematic(sims, scene)
        sims.timer.end("Grid kinematic")
        sims.timer.begin("Contact & Kinematic boundary")
        self.pre_contact_calculate(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.apply_adaptive_constraints(sims, scene)
        self.apply_dirichlet_constraints(sims, scene)
        self.compute_contact_force_(sims, scene)
        sims.timer.end("Contact & Kinematic boundary")
        sims.timer.begin("G2P")
        self.compute_particle_kinematic(sims, scene)
        sims.timer.end("G2P")
        sims.timer.begin("Stress update")
        self.compute_velocity_gradient(sims, scene)
        self.compute_stress_strains(sims, scene)
        self.compute_affine_velocity_gradient(sims, scene)
        sims.timer.end("Stress update")
        self.update_grid_refinement_profiled(sims, scene)
        sims.timer.begin("Misc")
        self.pressure_smoothing_(sims, scene)
        sims.timer.end("Misc")

    def usf_updating(self, sims: Simulation, scene: myScene, neighbor=None):
        sims.timer.begin("Shape function")
        self.update_adaptive_interpolation(sims, scene)
        sims.timer.end("Shape function")
        sims.timer.begin("P2G")
        self.compute_nodal_kinematic(sims, scene)
        self.compute_grid_velcity(sims, scene)
        self.apply_dirichlet_constraints(sims, scene)
        self.apply_adaptive_velocity_constraints(sims, scene)
        self.apply_dirichlet_constraints(sims, scene)
        sims.timer.end("P2G")
        sims.timer.begin("Stress update")
        self.compute_velocity_gradient(sims, scene)
        self.compute_stress_strains(sims, scene)
        sims.timer.end("Stress update")
        sims.timer.begin("Misc")
        self.pressure_smoothing_(sims, scene)
        sims.timer.end("Misc")
        sims.timer.begin("P2G")
        self.compute_forces(sims, scene)
        sims.timer.end("P2G")
        sims.timer.begin("Traction boundary")
        self.apply_particle_traction_constraints(sims, scene)
        self.apply_traction_constraints(sims, scene)
        self.apply_absorbing_constraints(sims, scene)
        sims.timer.end("Traction boundary")
        sims.timer.begin("Grid kinematic")
        self.compute_grid_kinematic(sims, scene)
        sims.timer.end("Grid kinematic")
        sims.timer.begin("Contact & Kinematic boundary")
        self.pre_contact_calculate(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.apply_adaptive_constraints(sims, scene)
        self.apply_dirichlet_constraints(sims, scene)
        self.compute_contact_force_(sims, scene)
        sims.timer.end("Contact & Kinematic boundary")
        sims.timer.begin("G2P")
        self.compute_particle_kinematic(sims, scene)
        self.compute_affine_velocity_gradient(sims, scene)
        sims.timer.end("G2P")
        self.update_grid_refinement_profiled(sims, scene)

    def musl_updating(self, sims: Simulation, scene: myScene, neighbor=None):
        sims.timer.begin("Shape function")
        self.update_adaptive_interpolation(sims, scene)
        sims.timer.end("Shape function")
        sims.timer.begin("P2G")
        self.compute_nodal_kinematic(sims, scene)
        self.compute_grid_velcity(sims, scene)
        self.compute_forces(sims, scene)
        sims.timer.end("P2G")
        sims.timer.begin("Traction boundary")
        self.apply_particle_traction_constraints(sims, scene)
        self.apply_traction_constraints(sims, scene)
        self.apply_absorbing_constraints(sims, scene)
        sims.timer.end("Traction boundary")
        sims.timer.begin("Grid kinematic")
        self.compute_grid_kinematic(sims, scene)
        sims.timer.end("Grid kinematic")
        sims.timer.begin("Contact & Kinematic boundary")
        self.pre_contact_calculate(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.apply_adaptive_constraints(sims, scene)
        self.apply_dirichlet_constraints(sims, scene)
        self.compute_contact_force_(sims, scene)
        sims.timer.end("Contact & Kinematic boundary")
        sims.timer.begin("G2P")
        self.compute_particle_kinematic(sims, scene)
        sims.timer.end("G2P")
        sims.timer.begin("Velocity remapping")
        self.postmapping_grid_velocity(sims, scene)
        self.compute_grid_velcity(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.apply_adaptive_velocity_constraints(sims, scene)
        self.apply_dirichlet_constraints(sims, scene)
        sims.timer.end("Velocity remapping")
        sims.timer.begin("Stress update")
        self.compute_velocity_gradient(sims, scene)
        self.compute_stress_strains(sims, scene)
        self.compute_affine_velocity_gradient(sims, scene)
        sims.timer.end("Stress update")
        self.update_grid_refinement_profiled(sims, scene)
        sims.timer.begin("Misc")
        self.pressure_smoothing_(sims, scene)
        sims.timer.end("Misc")

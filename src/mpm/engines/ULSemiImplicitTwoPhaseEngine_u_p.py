from src.mpm.boundaries.BoundaryCore import *
from src.mpm.engines.ULSemiImplicitTwoPhaseEngine import ULSemiImplicitTwoPhaseEngine
from src.mpm.engines.EngineKernel import *
from src.mpm.engines.PoissonEquation import MatrixFree
from src.mpm.engines.FreeSurfaceDetection import *
from src.mpm.SceneManager import myScene
from src.mpm.Simulation import Simulation
from src.mpm.SpatialHashGrid import SpatialHashGrid
from src.utils.linalg import no_operation

from src.mpm.engines.ULSemiImplicitTwoPhaseEngineKernel import (
    kernel_recover_darcy_fluid_velocity_2D,
)


class ULSemiImplicitTwoPhaseEngine_u_p(ULSemiImplicitTwoPhaseEngine):
    def __init__(self, sims) -> None:
        super().__init__(sims)

    def choose_engine(self, sims: Simulation):
        if sims.mapping == "USF":
            self.compute = self.usf_updating
        else:
            raise ValueError(f"The mapping scheme {sims.mapping} is not supported yet")

        if sims.coupling != "Lagrangian":
            self.bulid_neighbor_list = no_operation

        self.free_surface_detections = no_operation
        if sims.free_surface_detection:
            self.free_surface_detections = self.free_surface_detection_poisson

        if sims.dimension == 2:
            self.update_nodal_pressure = self.update_nodal_pressure_2D
            self.compute_particle_kinematic = self.compute_particle_kinematics_twophase2D
            if not sims.is_2DAxisy:
                if sims.stabilize == "B-Bar Method":
                    self.compute_internal_forces = self.compute_internal_force_bbar
                    self.compute_stress_strains = self.compute_stresspressure_strain_bbar
                else:
                    self.compute_forces = self.compute_force_2D
                    self.compute_Poisson_equation_implicit = self.compute_Poisson_equation_2D
                    self.compute_stress_strains = self.compute_stress_strain
            else:
                if sims.stabilize == "B-Bar Method":
                    self.compute_internal_forces = self.compute_internal_force_bbar
                    self.compute_stresspressure_strains = self.compute_stresspressure_strain_bbar
                else:
                    self.compute_forces = self.compute_force_2DAxisy
                    self.compute_stresspressure_strains = self.compute_stresspressure_strain

    def compute_nodal_kinematics(self, sims: Simulation, scene: myScene):
        kernel_mass_momentum_p2g_twophase_u_p(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
            sims.current_step,
            scene.is_rigid,
        )

    def compute_particle_kinematics(self, sims: Simulation, scene: myScene):
        kernel_kinemaitc_g2p_twophase(
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

    def compute_particle_kinematics_2D(self, sims: Simulation, scene: myScene):
        kernel_kinemaitc_g2p_twophase2D(
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

    def apply_dirichlet_constraints(self, sims: Simulation, scene: myScene):
        # self.apply_reflection_constraints(sims, scene)
        self.apply_velocity_constraints(sims, scene)

    def compute_stress_strain(self, sims: Simulation, scene: myScene):
        for _, start_index, end_index, mat_prop in self._iter_twophase_materials(scene):
            kernel_compute_stress_strain_twophase2D_u_p(
                scene.element.grid_nodes,
                sims.dt,
                start_index,
                end_index,
                scene.node,
                scene.particle,
                scene.material.materialID,
                mat_prop,
                scene.material.stateVars,
                scene.element.LnID,
                scene.element.dshape_fn,
                scene.element.node_size,
            )

    def compute_force_2D(self, sims: Simulation, scene: myScene):
        kernel_force_p2g_semitwophase2D_u_p(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            sims.gravity,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.dshape_fn,
            scene.element.node_size,
            sims.pressure_beta,
        )

    def compute_force_bbar_2D(self, sims: Simulation, scene: myScene):
        kernel_force_bbar_p2g_twophase2D(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            sims.gravity,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.shape_fnc,
            scene.element.dshape_fnc,
            scene.element.node_size,
        )

    def compute_force_2DAxisy(self, sims: Simulation, scene: myScene):
        kernel_force_p2g_twophase_2DAxisy(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            sims.gravity,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.dshape_fn,
            scene.element.node_size,
        )

    def compute_force_bbar_2DAxisy(self, sims: Simulation, scene: myScene):
        kernel_force_bbar_p2g_twophase_2DAxisy(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            sims.gravity,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.dshape_fn,
            scene.element.node_size,
        )

    def compute_prediction_grid_kinematic(self, sims: Simulation, scene: myScene):
        kernel_compute_grid_kinematic(scene.mass_cut_off, sims.background_damping, scene.node, sims.dt)

    def compute_correction_grid_kinematic(self, sims: Simulation, scene: myScene):
        kernel_correct_grid_kinematic_semitwophase_u_p(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.node_size,
            scene.element.LnID,
            scene.node,
            scene.element.dshape_fn,
            scene.element.shape_fn,
            sims.dt,
            scene.mass_cut_off,
        )

    def compute_particle_kinematics_twophase2D(self, sims: Simulation, scene: myScene):
        kernel_kinemaitc_g2p_semitwophase2D_u_p(
            scene.element.grid_nodes,
            sims.alphaPIC,
            sims.pressure_beta,
            sims.dt,
            int(scene.particleNum[0]),
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )
        self.clamp_particle_pressure(scene)
        for _, start_index, end_index, mat_prop in self._iter_twophase_materials(scene):
            kernel_recover_darcy_fluid_velocity_2D(
                scene.element.grid_nodes,
                start_index,
                end_index,
                scene.material.materialID,
                mat_prop,
                sims.pressure_beta,
                sims.gravity,
                scene.particle,
                scene.node,
                scene.element.LnID,
                scene.element.dshape_fn,
                scene.element.node_size,
            )

    def compute_nodal_mass(self, sims: Simulation, scene: myScene):
        kernel_mass_p2g_twophase(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )

    def find_free_surface_by_density_poisson(self, sims, scene: myScene):
        # scene.element.calculate(scene.particleNum, scene.particle)
        # self.system_resolve(sims, scene)
        kernel_mass_g2p_poisson(
            scene.element.grid_nodes,
            scene.element.cell_volume,
            scene.element.node_size,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.node,
            int(scene.particleNum[0]),
            scene.particle,
        )
        assign_particle_free_surface_poisson(int(scene.particleNum[0]), scene.particle, scene.material.matProps)

    def find_free_surface_by_volume_fraction(self, sims, scene: myScene):
        cell_volume_reset(scene.element.cell_volumefrac)
        calculate_cell_volume(
            scene.element.cell_volumefrac,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.cell_volume,
            scene.element.grid_size,
            scene.element.cnum,
            sims.is_2DAxisy,
        )
        self.assign_particle_free_surface_by_volume_fraction(scene)

    def free_surface_detection_poisson(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        if sims.free_surface_detection:
            scene.check_in_domain_2D(sims)
            self.find_free_surface_by_volume_fraction(sims, scene)

    def pre_calculation(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        scene.element.calculate_characteristic_length(sims, int(scene.particleNum[0]), scene.particle, scene.psize)
        if scene.element.cell_volumefrac is None:
            scene.element.create_element_volume_fraction()

        scene.element.calculate(scene.particleNum, scene.particle)
        self.compute_nodal_mass(sims, scene)
        total_dofs = scene.element.initial_estimate_active_dofs_poisson(scene.mass_cut_off, scene.node)
        # The estimate maps m, ms and mf. Clearing only m leaves phase masses
        # doubled on the first physical P2G and halves its mapped pressure.
        grid_reset(scene.mass_cut_off, scene.node)

        self.matrix_free = MatrixFree()
        self.matrix_free.manage_function_poisson(sims, scene)
        self.matrix_free.set_matrix_vector(total_dofs, sims, scene)
        self.matrix_free.manage_operator(scene)
        self.matrix_free.operator.update_active_dofs(total_dofs)

    def reset_iterative_grid_message(self, scene: myScene):
        grid_internal_force_reset(scene.mass_cut_off, scene.node)

    def compute_Poisson_equation_2D(self, sims: Simulation, scene: myScene):
        cell_volume_reset(scene.element.cell_volumefrac)
        calculate_cell_volume(
            scene.element.cell_volumefrac,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.cell_volume,
            scene.element.grid_size,
            scene.element.cnum,
            sims.is_2DAxisy,
        )
        total_dofs = scene.element.find_active_nodes_poisson(
            scene.mass_cut_off, scene.node, scene.element.cell_volumefrac, scene.element.cnum, scene.is_rigid
        )
        self.matrix_free.operator.update_active_dofs(total_dofs)
        # print(total_dofs)
        self.matrix_free.run_poisson(sims, scene)
        self.update_nodal_pressure(sims, scene)
        self.limit_nodal_pressure_increment(sims, scene)

    def postmapping_grid_velocity(self, sims: Simulation, scene: myScene):
        kernel_reset_grid_velocity_twophase2D(scene.node)
        kernel_postmapping_kinemaitc_twophase2D(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.element.gnum,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )

    def calculate_interpolations(self, sims: Simulation, scene: myScene):
        scene.element.calculate(scene.particleNum, scene.particle)

    # -------------------------------------------------------
    def usf_updating(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        self.calculate_interpolation(sims, scene)
        self.compute_nodal_kinematics(sims, scene)
        self.free_surface_detections(sims, scene, neighbor)
        self.compute_grid_velcity(sims, scene)
        self.apply_dirichlet_constraints(sims, scene)
        self.compute_stress_strains(sims, scene)
        # self.pressure_smoothing_(sims, scene)
        self.compute_forces(sims, scene)
        self.apply_particle_traction_constraints(sims, scene)
        self.apply_traction_constraints(sims, scene)  # node
        self.apply_absorbing_constraints(sims, scene)
        self.compute_prediction_grid_kinematic(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        # self.pre_contact_calculate(sims, scene)
        # self.compute_contact_force_(sims, scene)
        self.compute_Poisson_equation_implicit(sims, scene)
        self.compute_correction_grid_kinematic(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.compute_particle_kinematic(sims, scene)

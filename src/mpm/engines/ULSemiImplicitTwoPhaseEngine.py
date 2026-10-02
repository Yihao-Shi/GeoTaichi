from src.mpm.boundaries.BoundaryCore import *
from src.mpm.engines.ULExplicitEngine import ULExplicitEngine
from src.mpm.engines.EngineKernel import *
from src.mpm.engines.PoissonEquation import MatrixFree
from src.mpm.engines.FreeSurfaceDetection import *
from src.mpm.SceneManager import myScene
from src.mpm.Simulation import Simulation
from src.mpm.SpatialHashGrid import SpatialHashGrid
from src.utils.linalg import no_operation
from src.linear_solver.MultiGridPCG_mixture import MGPCGMixPoissonSolver
from src.linear_solver.MultiGridPCG_mixture_Axi import MGPCGMixPoissonSolver_Axi


class ULSemiImplicitTwoPhaseEngine(ULExplicitEngine):
    def __init__(self, sims) -> None:
        super().__init__(sims)

    def _single_twophase_matprop(self, scene: myScene):
        active_materials = list(self._iter_twophase_materials(scene))
        if len(active_materials) != 1:
            raise RuntimeError(
                "This SemiImplicit operation requires exactly one active "
                f"TwoPhase material, found {len(active_materials)}"
            )
        return active_materials[0][3]

    def _iter_twophase_materials(self, scene: myScene):
        mapping = getattr(scene.material, "mapping", None)
        if mapping is None:
            particle_num = int(scene.particleNum[0])
            for material_id in range(1, scene.material.matProps.size()):
                yield material_id, 0, particle_num, scene.material.matProps[material_id]
            return

        for material_offset in range(mapping.shape[0] - 1):
            material_id = material_offset + 1
            if material_id >= scene.material.matProps.size():
                continue
            start_index = int(mapping[material_offset])
            end_index = int(mapping[material_offset + 1])
            if end_index > start_index:
                yield material_id, start_index, end_index, scene.material.matProps[material_id]

    def clamp_particle_pressure(self, scene: myScene):
        for _, start_index, end_index, mat_prop in self._iter_twophase_materials(scene):
            kernel_clamp_twophase_particle_pressure(
                start_index,
                end_index,
                scene.material.materialID,
                mat_prop.cavitation_pressure,
                scene.particle,
            )

    def limit_nodal_pressure_increment(self, sims: Simulation, scene: myScene):
        minimum_pressure = max(
            mat_prop.cavitation_pressure for _, _, _, mat_prop in self._iter_twophase_materials(scene)
        )
        kernel_limit_twophase_nodal_pressure_increment(
            scene.mass_cut_off,
            sims.pressure_beta,
            minimum_pressure,
            scene.node,
        )

    def limit_cell_pressure_increment(self, sims: Simulation, scene: myScene):
        minimum_pressure = max(
            mat_prop.cavitation_pressure for _, _, _, mat_prop in self._iter_twophase_materials(scene)
        )
        kernel_limit_twophase_cell_pressure_increment(
            sims.pressure_beta,
            minimum_pressure,
            scene.node_type,
            scene.cell_pressure,
            scene.cell_dpressure,
        )

    def choose_engine(self, sims: Simulation):
        if sims.mapping == "USL":
            self.compute = self.usl_updating
        elif sims.mapping == "USF":
            self.compute = self.usf_updating
        else:
            raise ValueError(f"The mapping scheme {sims.mapping} is not supported yet")

        # This formulation detects the pressure free surface from element
        # volume fractions. The generic particle-neighbor/SPH pass is
        # redundant here and its geometry kernels do not accept 2-D particles.
        if sims.coupling != "Lagrangian":
            self.bulid_neighbor_list = no_operation

        self.free_surface_detections = no_operation
        self.compute_contact_force_prediction = no_operation
        self.compute_contact_force_correction = no_operation
        self.compute_affine_matrix = no_operation
        self.map_tpic_pressure = no_operation
        self.update_tpic_pressure_gradient = no_operation
        if sims.use_mgpcg_pressure_solver():
            if sims.is_2DAxisy:
                self.free_surface_detections = self.free_surface_detection_poisson_byMGPCG_2DAxisy
            else:
                self.free_surface_detections = self.free_surface_detection_poisson_byMGPCG
        elif sims.free_surface_detection:
            self.free_surface_detections = self.free_surface_detection_poisson
        else:
            self.free_surface_detections = self.calculate_pressure_domain_volume_fraction

        if sims.dimension == 2:
            self.update_nodal_pressure = self.update_nodal_pressure_2D
            self.compute_particle_kinematic = self.compute_particle_kinematics_twophase2D
            if sims.pressure_smoothing:
                self.pressure_smoothing_ = self.pressure_smoothing_twophase_2D

            if sims.pressure_stabilize == "FIC":
                self.compute_nodal_kinematics = self.compute_nodal_kinematics_FIC
                self.compute_particle_kinematic = self.compute_particle_kinematics_twophase2D_FIC
                # self.compute_grid_velcity = self.compute_grid_velcity_FIC
            if not sims.is_2DAxisy:
                if not sims.use_mgpcg_pressure_solver() and sims.free_surface_detection and sims.pressure_beta > 0.0:
                    self.map_tpic_pressure = self.map_tpic_pressure_2D
                    self.update_tpic_pressure_gradient = self.update_tpic_pressure_gradient_2D
                if sims.stabilize == "B-Bar Method":
                    self.compute_internal_forces = no_operation
                    self.compute_stress_strains = no_operation
                else:
                    self.compute_forces = self.compute_force_2D
                    self.compute_stress_strains = self.compute_stress_strain
                    if sims.use_mgpcg_pressure_solver():
                        self.compute_Poisson_equation_implicit = self.compute_Poisson_equation_2D_byMGPCG
                        self.compute_correction_grid_kinematics = self.compute_correction_grid_kinematic_byMGPCG
                        if sims.pressure_stabilize == "FIC":
                            self.compute_Poisson_equation_implicit = self.compute_Poisson_equation_FIC_2D_byMGPCG
                            self.compute_correction_grid_kinematics = self.compute_correction_grid_kinematic_FIC_byMGPCG
                    else:
                        self.compute_Poisson_equation_implicit = self.compute_Poisson_equation_2D
                        self.compute_correction_grid_kinematics = self.compute_correction_grid_kinematic

                if sims.contact_detection:
                    self.pre_contact_calculate = self.calculate_precontact
                    if sims.contact_detection == "MPMContact":
                        self.compute_contact_force_prediction = self.compute_contact_force_semi_2D
                        self.compute_contact_force_correction = self.compute_contact_force_semi_2D
                    elif sims.contact_detection == "GeoContact":
                        self.compute_contact_force_ = self.compute_geocontact_force_2D
                    elif sims.contact_detection == "DEMContact":
                        pass
                    else:
                        raise RuntimeError("Wrong contact type!")

            elif sims.is_2DAxisy:
                if sims.stabilize == "B-Bar Method":
                    self.compute_internal_forces = no_operation
                    self.compute_stress_strains = no_operation
                else:
                    self.compute_forces = self.compute_force_2DAxisy
                    self.compute_stress_strains = self.compute_stress_strain_2DAxisy
                    if sims.use_mgpcg_pressure_solver():
                        self.compute_Poisson_equation_implicit = self.compute_Poisson_equation_2DAxisy_byMGPCG
                        self.compute_correction_grid_kinematics = self.compute_correction_grid_kinematic_2DAxisy_byMGPCG
                        if sims.pressure_stabilize == "FIC":
                            self.compute_Poisson_equation_implicit = no_operation
                            self.compute_correction_grid_kinematics = no_operation
                    else:
                        self.compute_Poisson_equation_implicit = self.compute_Poisson_equation_2D
                        self.compute_correction_grid_kinematics = self.compute_correction_grid_kinematic_2DAxisy
                        if sims.pressure_stabilize == "FIC":
                            self.compute_Poisson_equation_implicit = no_operation
                            self.compute_correction_grid_kinematics = no_operation

                if sims.contact_detection:
                    self.pre_contact_calculate = self.calculate_precontact_2DAxisy
                    if sims.contact_detection == "MPMContact":
                        self.compute_contact_force_prediction = self.compute_contact_force_semi_2D
                        self.compute_contact_force_correction = self.compute_contact_force_semi_2D
                    elif sims.contact_detection == "GeoContact":
                        self.compute_contact_force_ = no_operation
                    elif sims.contact_detection == "DEMContact":
                        self.pre_contact_calculate = no_operation
                        self.compute_contact_force_prediction = self.compute_ndemcontact_force_semi_2D
                        self.compute_contact_force_correction = no_operation

            if sims.velocity_projection_scheme == "Affine":
                self.compute_nodal_kinematics = self.compute_nodal_kinematics_APIC
                if not sims.is_2DAxisy:
                    self.compute_stress_strains = self.compute_stress_strain_APIC_2D
                    self.compute_affine_matrix = self.compute_affine_matrix_APIC_2D
                elif sims.is_2DAxisy:
                    self.compute_stress_strains = self.compute_stress_strain_APIC_2DAxisy
                    self.compute_affine_matrix = self.compute_affine_matrix_APIC_2DAxisy

        elif sims.dimension == 3:
            if not sims.use_mgpcg_pressure_solver():
                raise RuntimeError("3D SemiImplicit TwoPhaseSingleLayer currently requires pressure_solver='MGPCG'")
            if sims.pressure_stabilize == "FIC":
                raise RuntimeError(
                    "3D SemiImplicit TwoPhaseSingleLayer does not support FIC pressure stabilization yet"
                )
            if sims.velocity_projection_scheme == "Affine":
                raise RuntimeError("3D SemiImplicit TwoPhaseSingleLayer does not support Affine/APIC projection yet")
            self.compute_forces = self.compute_force_3D
            self.compute_stress_strains = self.compute_stress_strain_3D
            self.compute_particle_kinematic = self.compute_particle_kinematics_twophase3D
            self.compute_Poisson_equation_implicit = self.compute_Poisson_equation_3D_byMGPCG
            self.compute_correction_grid_kinematics = self.compute_correction_grid_kinematic_byMGPCG_3D

    def compute_nodal_kinematics(self, sims: Simulation, scene: myScene):
        kernel_mass_momentum_p2g_semitwophase(
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
        self.map_tpic_pressure(scene)

    def compute_nodal_kinematics_APIC(self, sims: Simulation, scene: myScene):
        kernel_mass_momentum_p2g_twophase_APIC(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
            sims.current_step,
            scene.is_rigid,
            scene.element.grid_size,
            scene.element.gnum,
        )
        self.map_tpic_pressure(scene)

    def compute_nodal_kinematics_FIC(self, sims: Simulation, scene: myScene):
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
        self.map_tpic_pressure(scene)

    def map_tpic_pressure_2D(self, scene: myScene):
        kernel_pressure_tpic_p2g_correction_2D(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.element.grid_size,
            scene.element.gnum,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )

    def update_tpic_pressure_gradient_2D(self, sims: Simulation, scene: myScene):
        kernel_update_particle_pressure_gradient_2D(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            sims.pressure_beta,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.dshape_fn,
            scene.element.node_size,
        )

    def compute_grid_velcity(self, sims: Simulation, scene: myScene):
        kernel_compute_grid_velocity_twophase(scene.mass_cut_off, scene.node)

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
            kernel_compute_stress_strain_semitwophase2D(
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

    def compute_stress_strain_3D(self, sims: Simulation, scene: myScene):
        for _, start_index, end_index, mat_prop in self._iter_twophase_materials(scene):
            kernel_compute_stress_strain_semitwophase3D(
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

    def compute_stress_strain_APIC_2D(self, sims: Simulation, scene: myScene):
        for _, start_index, end_index, mat_prop in self._iter_twophase_materials(scene):
            kernel_compute_stress_strain_APIC_twophase2D(
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
                scene.element.shape_fn,
                scene.element.node_size,
                scene.element.grid_size,
                scene.element.gnum,
            )

    def compute_stress_strain_2DAxisy(self, sims: Simulation, scene: myScene):
        for _, start_index, end_index, mat_prop in self._iter_twophase_materials(scene):
            kernel_compute_stress_strain_twophase2DAxisy(
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
                scene.element.shape_fn,
                scene.element.dshape_fn,
                scene.element.node_size,
                sims.axis_offset,
            )

    def compute_stress_strain_APIC_2DAxisy(self, sims: Simulation, scene: myScene):
        for _, start_index, end_index, mat_prop in self._iter_twophase_materials(scene):
            kernel_compute_stress_strain_APIC_twophase2DAxisy(
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
                scene.element.shape_fn,
                scene.element.dshape_fn,
                scene.element.node_size,
                scene.element.grid_size,
                scene.element.gnum,
                sims.axis_offset,
            )

    def compute_affine_matrix_APIC_2D(self, sims: Simulation, scene: myScene):
        kernel_compute_affine_matrix_APIC_twophase2D(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
            scene.element.grid_size,
            scene.element.gnum,
        )

    def compute_affine_matrix_APIC_2DAxisy(self, sims: Simulation, scene: myScene):
        kernel_compute_affine_matrix_APIC_twophase2DAxisy(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
            scene.element.grid_size,
            scene.element.gnum,
            sims.axis_offset,
        )

    def compute_force_2D(self, sims: Simulation, scene: myScene):
        kernel_force_p2g_semitwophase2D(
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
            not sims.use_mgpcg_pressure_solver(),
        )

    def compute_force_3D(self, sims: Simulation, scene: myScene):
        kernel_force_p2g_semitwophase3D(
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
        kernel_force_p2g_semitwophase_2DAxisy(
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
            sims.axis_offset,
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
        kernel_compute_grid_kinematic_semitwophase(
            scene.mass_cut_off, sims.background_damping, scene.node, sims.dt, scene.element.ti_nodal_coords
        )

    def compute_prediction_grid_kinematic_(self, sims: Simulation, scene: myScene):
        kernel_compute_grid_kinematic_semitwophase_(
            scene.mass_cut_off, sims.background_damping, scene.node, sims.dt, scene.element.ti_nodal_coords
        )

    def compute_correction_grid_kinematic(self, sims: Simulation, scene: myScene):
        kernel_correct_grid_kinematic_semitwophase(
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

    def compute_correction_grid_kinematic_2DAxisy(self, sims: Simulation, scene: myScene):
        kernel_correct_grid_kinematic_semitwophase_2DAxisy(
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

    def project_fic_pressure_gradient(self, scene: myScene):
        # Rebuild from the mapped old pressure on the current particle support.
        # This also initializes new nodes and restart runs, without stale grid
        # history or a different first-step stabilization equation.
        scene.node.extra_stabilize.fill(0)
        for _, start_index, end_index, mat_prop in self._iter_twophase_materials(scene):
            kernel_project_fic_pressure_gradient(
                scene.element.grid_nodes,
                start_index,
                end_index,
                scene.material.materialID,
                mat_prop,
                scene.particle,
                scene.element.node_size,
                scene.element.LnID,
                scene.node,
                scene.element.dshape_fn,
                scene.element.shape_fn,
            )
        kernel_normalize_fic_pressure_projection(scene.node)

    def compute_correction_grid_kinematic_byMGPCG(self, sims: Simulation, scene: myScene):
        kernel_correct_grid_kinematic_semitwophase_mg(
            sims.current_step,
            scene.node,
            sims.dt,
            scene.mass_cut_off,
            scene.element.grid_size,
            scene.element.gnum,
            scene.node_type,
            scene.cell_porosity,
            scene.cell_dpressure,
            scene.cell_phi,
            self._single_twophase_matprop(scene),
            scene.is_rigid,
        )

    def compute_correction_grid_kinematic_byMGPCG_3D(self, sims: Simulation, scene: myScene):
        kernel_correct_grid_kinematic_semitwophase_mg_3D(
            sims.current_step,
            scene.node,
            sims.dt,
            scene.mass_cut_off,
            scene.element.grid_size,
            scene.element.gnum,
            scene.node_type,
            scene.cell_porosity,
            scene.cell_dpressure,
            scene.cell_phi,
            self._single_twophase_matprop(scene),
            scene.is_rigid,
        )

    def compute_correction_grid_kinematic_2DAxisy_byMGPCG(self, sims: Simulation, scene: myScene):
        # kernel_correct_grid_kinematic_semitwophase_2DAxi_mg_map(scene.material.matProps, scene.node, scene.element.grid_size, scene.element.extra_function, scene.element.extra_grad_function, scene.element.gnum, sims.dt, scene.node_type, scene.cell_dpressure, scene.cell_free_surface, scene.is_rigid, sims.axis_offset)
        kernel_correct_grid_kinematic_semitwophase_2DAxi_mg_fdm(
            sims.current_step,
            scene.node,
            sims.dt,
            scene.mass_cut_off,
            scene.element.grid_size,
            scene.element.gnum,
            scene.node_type,
            scene.cell_porosity,
            scene.cell_dpressure,
            scene.cell_phi,
            self._single_twophase_matprop(scene),
            scene.is_rigid,
            sims.axis_offset,
        )

    def compute_correction_grid_kinematic_FIC_byMGPCG(self, sims: Simulation, scene: myScene):
        scene.node.extra_stabilize.fill(0)
        # kernel_correct_grid_kinematic_semitwophase_FIC_mg_map(scene.material.matProps, scene.node, scene.element.grid_size, scene.element.extra_function, scene.element.extra_grad_function, scene.element.gnum, sims.dt, scene.node_type, scene.cell_porosity)
        kernel_correct_grid_kinematic_semitwophase_FIC_mg_fdm(
            sims.current_step,
            scene.node,
            sims.dt,
            scene.mass_cut_off,
            scene.element.grid_size,
            scene.element.gnum,
            scene.node_type,
            scene.cell_porosity,
            scene.cell_pressure,
            scene.cell_dpressure,
            scene.cell_phi,
            self._single_twophase_matprop(scene),
        )

    def update_nodal_pressure_2D(self, sims: Simulation, scene: myScene):
        kernel_update_nodal_pressure_2D(
            scene.mass_cut_off,
            sims.pressure_beta,
            0.0,  # Atmospheric gauge pressure, not the material's cavitation limit.
            scene.node,
            self.matrix_free.unknow_vector,
        )

    def compute_particle_kinematics_twophase2D(self, sims: Simulation, scene: myScene):
        kernel_kinemaitc_g2p_semitwophase2D(
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
        self.update_tpic_pressure_gradient(sims, scene)
        self.clamp_particle_pressure(scene)

    def compute_particle_kinematics_twophase3D(self, sims: Simulation, scene: myScene):
        kernel_kinemaitc_g2p_semitwophase3D(
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

    def compute_particle_kinematics_twophase2D_FIC(self, sims: Simulation, scene: myScene):
        kernel_kinemaitc_g2p_semitwophase2D_FIC(
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
        self.update_tpic_pressure_gradient(sims, scene)
        self.clamp_particle_pressure(scene)

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

    def calculate_pressure_domain_volume_fraction(self, sims, scene: myScene, neighbor=None):
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

    def find_free_surface_by_volume_fraction(self, sims, scene: myScene):
        self.calculate_pressure_domain_volume_fraction(sims, scene)
        self.assign_particle_free_surface_by_volume_fraction(scene)

    def assign_particle_free_surface_by_volume_fraction(self, scene: myScene):
        assign_particle_free_surface_by_volume_multilayer(
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.cell_volumefrac,
            scene.element.grid_size,
            scene.element.cnum,
        )

    def record_free_surface_particles(self, sims: Simulation, scene: myScene):
        record_free_surface_particle_2D(
            sims.domain,
            scene.element.grid_size,
            int(scene.particleNum[0]),
            scene.particle,
            scene.free_particleNum,
            scene.free_particle_list,
        )

    def free_surface_detection_poisson(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        scene.check_in_domain_2D(sims)
        self.find_free_surface_by_volume_fraction(sims, scene)
        """self.find_free_surface_by_density_poisson(sims, scene)"""
        """neighbor.place_particles(scene)  # sph -- update Hash Table
        self.compute_boundary_direction(scene, neighbor)  # sph -- compute normal vector
        self.free_surface_by_geometry(scene, neighbor)    # sph -- identify free surface particle ID
        self.record_free_surface_particles(sims, scene)"""

    def free_surface_detection_poisson_byMGPCG(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        cell_volume_reset(scene.element.cell_volumefrac)
        calculate_cell_volume_fraction(
            scene.element.cell_volumefrac,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.cell_volume,
            scene.element.grid_size,
            scene.element.cnum,
            scene.cell_rigid,
        )
        self.assign_particle_free_surface_by_volume_fraction(scene)

    def free_surface_detection_poisson_byMGPCG_2DAxisy(
        self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid
    ):
        cell_volume_reset(scene.element.cell_volumefrac)
        calculate_cell_volume_fraction_2DAxisy_plane(
            scene.element.cell_volumefrac,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.cell_volume,
            scene.element.grid_size,
            scene.element.cnum,
            scene.cell_rigid,
            sims.axis_offset,
        )
        self.assign_particle_free_surface_by_volume_fraction(scene)

    def pre_calculation(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        scene.element.calculate_characteristic_length(sims, int(scene.particleNum[0]), scene.particle, scene.psize)
        if scene.element.cell_volumefrac is None:
            scene.element.create_element_volume_fraction()

        # scene.element.calculate(scene.particleNum, scene.particle)
        # self.compute_nodal_mass(sims, scene)
        # total_dofs = scene.element.initial_estimate_active_dofs_poisson(scene.mass_cut_off, scene.node)
        # grid_mass_reset(scene.mass_cut_off, scene.node)
        total_dofs = scene.node.shape[0]
        print(total_dofs)

        if sims.use_mgpcg_pressure_solver():
            self._single_twophase_matprop(scene)
            self.n_mg_levels = sims.multilevel
            self.iterations = sims.iter_max
            pre_and_post_smoothing = sims.pre_and_post_smoothing
            bottom_smoothing = sims.bottom_smoothing
            print(scene.element.cnum)
            dr = scene.element.grid_size[0]
            if sims.is_2DAxisy:
                self.poisson_solver = MGPCGMixPoissonSolver_Axi(
                    sims.dimension,
                    scene.element.cnum,
                    dr,
                    self.n_mg_levels,
                    pre_and_post_smoothing,
                    bottom_smoothing,
                    sims.axis_offset,
                )
            else:
                self.poisson_solver = MGPCGMixPoissonSolver(
                    sims.dimension, scene.element.cnum, self.n_mg_levels, pre_and_post_smoothing, bottom_smoothing
                )
        else:
            self.matrix_free = MatrixFree()
            self.matrix_free.manage_function_poisson(sims, scene)
            self.matrix_free.set_matrix_vector(total_dofs, sims, scene)  # generate a larger space
            self.matrix_free.manage_operator(scene)
            self.matrix_free.operator.update_active_dofs(total_dofs)
            if sims.assemble_type == "COO":
                self.solve_poisson_system = self.matrix_free.run_poisson_coo
            elif sims.assemble_type == "MatrixFree":
                self.solve_poisson_system = self.matrix_free.run_poisson
            else:
                raise ValueError(f"Unsupported semi-implicit Poisson assemble_type: {sims.assemble_type}")

    def reset_iterative_grid_message(self, scene: myScene):
        grid_internal_force_reset(scene.mass_cut_off, scene.node)

    def solve_pressure_mgpcg(self, sims: Simulation):
        solved = self.poisson_solver.solve(
            self.iterations,
            rel_tol=sims.residual_tolerance,
            abs_tol=1.0e-10,
        )
        if not solved:
            raise RuntimeError(
                "TwoPhaseSingleLayer pressure MGPCG failed: "
                f"{self.poisson_solver.breakdown_reason}={self.poisson_solver.breakdown_value:.6e}, "
                f"initial_rTr={self.poisson_solver.initial_residual:.6e}, "
                f"final_rTr={self.poisson_solver.final_residual:.6e}, "
                f"iterations={self.poisson_solver.last_iterations}"
            )

    def compute_Poisson_equation_2D(self, sims: Simulation, scene: myScene):
        kernel_single_point_porosity_p2g(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.particle,
            scene.node,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )
        if sims.pressure_stabilize == "FIC":
            self.project_fic_pressure_gradient(scene)
        active_dofs = scene.element.find_active_nodes_poisson(
            scene.mass_cut_off,
            scene.node,
            scene.element.cell_volumefrac,
            scene.element.cnum,
            scene.is_rigid,
            scene.element.ti_nodal_coords,
            sims.shape_function,
            scene.particle[0].x[1] - 0.2,
        )
        # active_dofs = scene.element.find_active_nodes_poisson_with_pressure_boundary(scene.mass_cut_off, scene.node, scene.element.cell_volumefrac, scene.element.cnum, scene.is_rigid, sims.npressure, sims.pressure_constraint_list)
        # print(active_dofs)
        self.matrix_free.operator.update_active_dofs(active_dofs)
        # self.matrix_free.assemble_compute_penalty_matrix(sims, scene)
        self.solve_poisson_system(sims, scene)
        self.update_nodal_pressure(sims, scene)
        self.limit_nodal_pressure_increment(sims, scene)

    def compute_Poisson_equation_2D_byMGPCG(self, sims: Simulation, scene: myScene):
        # self.element.set_node_type(scene.node, scene.node_type)
        scene.element.set_cell_type(
            scene.element.cell_volumefrac,
            scene.cell_rigid,
            scene.node_type,
            round(sims.axis_offset / scene.element.grid_size[0]),
            round(scene.particle[0].x[1] / scene.element.grid_size[1]),
        )
        self.update_cell_information(sims, scene)
        self.update_signed_distance_phi(sims, scene)
        self.poisson_solver.reinitialize(scene.node_type)
        self.build_b(sims, scene)
        self.build_A(0, sims, scene)
        for l in range(1, self.n_mg_levels):
            self.poisson_solver.init_gridtype(self.poisson_solver.grid_type[l - 1], self.poisson_solver.grid_type[l])
            self.build_A(l, sims, scene)
        self.solve_pressure_mgpcg(sims)
        scene.cell_dpressure.copy_from(self.poisson_solver.x)
        self.limit_cell_pressure_increment(sims, scene)
        self.update_nodal_dpressure(scene)

    def compute_Poisson_equation_3D_byMGPCG(self, sims: Simulation, scene: myScene):
        scene.element.set_cell_type(scene.element.cell_volumefrac, scene.cell_rigid, scene.node_type)
        self.update_cell_information(sims, scene)
        self.update_signed_distance_phi(sims, scene)
        self.poisson_solver.reinitialize(scene.node_type)
        self.build_b_3D(sims, scene)
        self.build_A(0, sims, scene)
        for l in range(1, self.n_mg_levels):
            self.poisson_solver.init_gridtype(self.poisson_solver.grid_type[l - 1], self.poisson_solver.grid_type[l])
            self.build_A(l, sims, scene)
        self.solve_pressure_mgpcg(sims)
        scene.cell_dpressure.copy_from(self.poisson_solver.x)
        self.limit_cell_pressure_increment(sims, scene)
        self.update_nodal_dpressure_3D(scene)

    def compute_Poisson_equation_2DAxisy_byMGPCG(self, sims: Simulation, scene: myScene):
        scene.element.set_cell_type(
            scene.element.cell_volumefrac,
            scene.cell_rigid,
            scene.node_type,
            round(sims.axis_offset / scene.element.grid_size[0]),
            round(scene.particle[0].x[1] / scene.element.grid_size[1]),
        )
        self.update_cell_information(sims, scene)
        self.update_signed_distance_phi(sims, scene)
        self.poisson_solver.reinitialize(scene.node_type)
        self.build_b_2DAxisy(sims, scene)
        self.build_A_2DAxisy(0, sims, scene)
        for l in range(1, self.n_mg_levels):
            self.poisson_solver.init_gridtype(self.poisson_solver.grid_type[l - 1], self.poisson_solver.grid_type[l])
            self.build_A_2DAxisy(l, sims, scene)
        self.solve_pressure_mgpcg(sims)
        scene.cell_dpressure.copy_from(self.poisson_solver.x)
        self.limit_cell_pressure_increment(sims, scene)
        self.update_nodal_dpressure(scene)

    def compute_Poisson_equation_FIC_2D_byMGPCG(self, sims: Simulation, scene: myScene):
        scene.element.set_cell_type(scene.element.cell_volumefrac, scene.cell_rigid, scene.node_type)
        self.update_cell_information(sims, scene)
        # self.update_signed_distance_phi(sims, scene)
        self.poisson_solver.reinitialize(scene.node_type)
        self.build_b_FIC(sims, scene)
        self.build_A_FIC(0, sims, scene)
        for l in range(1, self.n_mg_levels):
            self.poisson_solver.init_gridtype(self.poisson_solver.grid_type[l - 1], self.poisson_solver.grid_type[l])
            self.build_A_FIC(l, sims, scene)
        self.solve_pressure_mgpcg(sims)
        scene.cell_dpressure.copy_from(self.poisson_solver.x)
        self.limit_cell_pressure_increment(sims, scene)
        self.update_nodal_dpressure(scene)

    def update_nodal_dpressure(self, scene: myScene):
        kernel_update_nodal_dpressure_2D(
            scene.node,
            scene.element.grid_size,
            scene.element.extra_function,
            scene.element.gnum,
            scene.node_type,
            scene.cell_volume,
            scene.cell_dpressure,
            scene.cell_free_surface,
            scene.cell_phi,
            scene.is_rigid,
        )
        kernel_update_nodal_dpressure_2D_(scene.mass_cut_off, scene.node)

    def update_nodal_dpressure_3D(self, scene: myScene):
        kernel_update_nodal_dpressure_3D(
            scene.node,
            scene.element.grid_size,
            scene.element.shape_function,
            scene.element.gnum,
            scene.node_type,
            scene.cell_volume,
            scene.cell_dpressure,
            scene.cell_free_surface,
            scene.cell_phi,
            scene.is_rigid,
        )
        kernel_update_nodal_dpressure_2D_(scene.mass_cut_off, scene.node)

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

    def calculate_precontact(self, sims: Simulation, scene: myScene):
        kernel_calc_contact_normal_twophase(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.dshape_fn,
            scene.element.node_size,
        )

    def calculate_precontact_2DAxisy(self, sims: Simulation, scene: myScene):
        kernel_calc_contact_normal_twophase_2DAxisy(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.dshape_fn,
            scene.element.node_size,
            sims.axis_offset,
        )

    def compute_contact_force_semi_2D(self, sims: Simulation, scene: myScene):
        kernel_calc_friction_contact_semi_2D(
            scene.mass_cut_off,
            scene.contact.friction,
            sims.dt,
            scene.is_rigid,
            scene.node,
            scene.element.ti_nodal_coords,
            sims.axis_offset,
        )

    def compute_ndemcontact_force_semi_2D(self, sims: Simulation, scene: myScene):
        kernel_calc_normal_demcontact_semi_2D(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.particle,
            scene.material.matProps,
            scene.element.grid_size,
            scene.node,
            scene.contact.polygon_vertices,
            scene.contact.velocity,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )
        kernel_calc_tangential_demcontact_semi_2D(
            scene.mass_cut_off,
            self._single_twophase_matprop(scene).friction,
            scene.contact.velocity,
            sims.dt,
            scene.node,
            scene.element.ti_nodal_coords,
            scene.particle,
        )
        kernel_assemble_contact_force_solid(scene.mass_cut_off, sims.dt, scene.node)

    def apply_displacement_constraints(self, sims: Simulation, scene: myScene):
        kernel_apply_displacement_mixture(scene.mass_cut_off, sims.dt, scene.node, scene.element.ti_nodal_coords)

    def apply_node_traction_constraints(self, sims: Simulation, scene: myScene):
        kernel_apply_node_traction(scene.mass_cut_off, sims.dt, scene.node, scene.element.ti_nodal_coords)

    def calculate_interpolations(self, sims: Simulation, scene: myScene):
        scene.element.calculate(scene.particleNum, scene.particle)

    def update_cell_information(self, sims: Simulation, scene: myScene):
        kernel_reset_cell_infor(
            scene.node_type,
            scene.cell_volume,
            scene.cell_phi,
            scene.cell_porosity,
            scene.cell_pressure,
            scene.cell_dpressure,
        )
        kernel_update_cell_infor(
            sims.current_step,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.grid_size,
            scene.cell_volume,
            scene.cell_porosity,
            scene.cell_pressure,
        )
        kernel_porosity_p2g(
            int(scene.particleNum[0]),
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.grid_nodes,
            scene.element.node_size,
        )
        kernel_grid_porosity(scene.mass_cut_off, scene.is_rigid, scene.node)
        kernel_update_cell_porosity(
            sims.current_step,
            scene.node_type,
            scene.cell_porosity,
            scene.cell_pressure,
            scene.node,
            scene.element.gnum,
            scene.is_rigid,
        )

    def update_signed_distance_phi(self, sims: Simulation, scene: myScene):
        kernel_build_fluid_sdf_from_cell_type(
            0,
            scene.element.cnum,
            scene.element.grid_size,
            False,
            scene.node_type,
            scene.cell_phi,
        )
        kernel_pre_update_phi(sims.current_step, scene.element.cnum, scene.node_type, scene.cell_free_surface)

    def build_A(self, level, sims, scene):
        kernel_assemble_A(
            level,
            sims.dt,
            self._single_twophase_matprop(scene),
            scene.element.grid_size,
            scene.cell_porosity,
            scene.cell_phi,
            self.poisson_solver.grid_type[level],
            self.poisson_solver.Adiag[level],
            self.poisson_solver.Ax[level],
        )

    def build_A_2DAxisy(self, level, sims, scene):
        # kernel_assemble_A_2DAxisy(sims.dt, scene.material.matProps, scene.element.grid_size, scene.cell_porosity, scene.cell_phi, self.poisson_solver.grid_type[level], self.poisson_solver.Adiag[level], self.poisson_solver.Ax[level], self.poisson_solver.Ax_neg[level], sims.axis_offset)
        # kernel_assemble_A_2DAxisy_(sims.dt, scene.material.matProps, scene.element.grid_size, scene.cell_porosity, scene.cell_phi, self.poisson_solver.grid_type[level], self.poisson_solver.Adiag[level], self.poisson_solver.Ax[level], self.poisson_solver.Ax_neg[level], sims.axis_offset)
        kernel_assemble_A_2DAxisy_multi(
            sims.dt,
            self._single_twophase_matprop(scene),
            scene.element.grid_size,
            scene.cell_porosity,
            scene.cell_phi,
            self.poisson_solver.grid_type[level],
            self.poisson_solver.Adiag[level],
            self.poisson_solver.Ax[level],
            self.poisson_solver.Ax_neg[level],
            sims.axis_offset,
            level,
        )

    def build_b(self, sims: Simulation, scene: myScene):
        # kernel_assemble_bn(scene.element.grid_nodes, int(scene.particleNum[0]), scene.particle, scene.element.node_size, scene.element.LnID, scene.node,
        #                    scene.element.dshape_fn, scene.element.shape_fn, scene.element.gnum, self.poisson_solver.b)
        kernel_assemble_bc_map(
            scene.node,
            scene.element.grid_size,
            scene.element.extra_function,
            scene.element.extra_grad_function,
            scene.element.gnum,
            self.poisson_solver.b,
            scene.node_type,
            scene.cell_porosity,
            scene.is_rigid,
        )
        # kernel_assemble_bc_fdm(sims.current_step, scene.node, scene.element.grid_size, scene.element.gnum, self.poisson_solver.b, scene.node_type, scene.cell_porosity, sims.dt, scene.material.matProps)

    def build_b_3D(self, sims: Simulation, scene: myScene):
        kernel_assemble_bc_map_3D(
            scene.node,
            scene.element.grid_size,
            scene.element.shape_function,
            scene.element.grad_shape_function,
            scene.element.gnum,
            self.poisson_solver.b,
            scene.node_type,
            scene.cell_porosity,
            scene.is_rigid,
        )

    def build_b_2DAxisy(self, sims: Simulation, scene: myScene):
        # kernel_assemble_bc_2DAxisy_map(scene.node, scene.element.grid_size, scene.element.extra_function, scene.element.extra_grad_function, scene.element.gnum, self.poisson_solver.b, scene.node_type, scene.cell_porosity, scene.is_rigid, sims.axis_offset)
        kernel_assemble_bc_2DAxisy_map_(
            scene.node,
            scene.element.grid_size,
            scene.element.extra_function,
            scene.element.extra_grad_function,
            scene.element.gnum,
            self.poisson_solver.b,
            scene.node_type,
            scene.cell_porosity,
            scene.is_rigid,
            sims.axis_offset,
        )
        # kernel_assemble_bc_2DAxisy_fdm(sims.current_step, scene.node, scene.element.grid_size, scene.element.gnum, self.poisson_solver.b, scene.node_type, scene.cell_porosity, sims.dt, scene.material.matProps, sims.axis_offset)

    def build_A_FIC(self, level, sims, scene):
        kernel_assemble_A_FIC(
            sims.dt,
            self._single_twophase_matprop(scene),
            scene.element.grid_size,
            scene.cell_porosity,
            scene.cell_phi,
            self.poisson_solver.grid_type[level],
            self.poisson_solver.Adiag[level],
            self.poisson_solver.Ax[level],
        )

    def build_b_FIC(self, sims: Simulation, scene: myScene):
        kernel_assemble_FIC_bc_map(
            self._single_twophase_matprop(scene),
            scene.node,
            scene.element.grid_size,
            scene.element.extra_function,
            scene.element.extra_grad_function,
            scene.element.gnum,
            self.poisson_solver.b,
            scene.node_type,
            scene.cell_porosity,
        )
        # kernel_assemble_FIC_bc_fdm(sims.current_step, scene.node, scene.element.grid_size, scene.element.gnum, self.poisson_solver.b, scene.node_type, scene.cell_porosity, sims.dt, scene.material.matProps, scene.cell_pressure, scene.cell_dpressure)

    def pressure_smoothing_twophase_2D(self, sims, scene: myScene):
        scene.extra_node.fill(0)
        # kernel_pressure_p2g_twophase_2D_linear(scene.element.gnum, scene.element.igrid_size, int(scene.particleNum[0]), scene.extra_node, scene.particle)
        # kernel_grid_pressure(scene.mass_cut_off, scene.is_rigid, scene.node, scene.extra_node)
        # kernel_pressure_g2p_twophase_2D_linear(scene.element.gnum, scene.element.igrid_size, scene.extra_node, int(scene.particleNum[0]), scene.particle)
        kernel_pressure_p2g_twophase_2D(
            int(scene.particleNum[0]),
            scene.extra_node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.grid_nodes,
            scene.element.node_size,
        )
        kernel_grid_pressure(scene.mass_cut_off, scene.is_rigid, scene.node, scene.extra_node)
        kernel_pressure_g2p_twophase_2D(
            scene.extra_node,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.grid_nodes,
            scene.element.node_size,
        )

    # -------------------------------------------------------
    def usl_updating(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        self.calculate_interpolation(sims, scene)
        self.compute_nodal_kinematics(sims, scene)
        self.free_surface_detections(sims, scene, neighbor)
        self.compute_grid_velcity(sims, scene)
        self.apply_dirichlet_constraints(sims, scene)
        # self.pressure_smoothing_(sims, scene)
        self.compute_forces(sims, scene)
        self.apply_particle_traction_constraints(sims, scene)
        self.apply_traction_constraints(sims, scene)  # node
        self.apply_absorbing_constraints(sims, scene)
        self.compute_prediction_grid_kinematic(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.pre_contact_calculate(sims, scene)
        self.compute_contact_force_prediction(sims, scene)
        self.compute_Poisson_equation_implicit(sims, scene)
        self.compute_correction_grid_kinematics(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.compute_contact_force_correction(sims, scene)
        self.compute_stress_strains(sims, scene)
        self.compute_particle_kinematic(sims, scene)

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
        # self.apply_node_traction_constraints(sims, scene)
        self.apply_absorbing_constraints(sims, scene)
        self.compute_prediction_grid_kinematic(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.pre_contact_calculate(sims, scene)
        self.compute_contact_force_prediction(sims, scene)
        # self.apply_displacement_constraints(sims, scene)
        self.compute_Poisson_equation_implicit(sims, scene)
        self.compute_correction_grid_kinematics(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.compute_contact_force_correction(sims, scene)
        # self.apply_displacement_constraints(sims, scene)
        self.compute_affine_matrix(sims, scene)
        self.compute_particle_kinematic(sims, scene)

    # drained updating scheme, without Poisson & correction step
    def usf_updating_(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        self.calculate_interpolation(sims, scene)
        self.compute_nodal_kinematics(sims, scene)
        # self.free_surface_detections(sims, scene, neighbor)
        self.compute_grid_velcity(sims, scene)
        self.apply_dirichlet_constraints(sims, scene)
        self.compute_stress_strains(sims, scene)
        # self.pressure_smoothing_(sims, scene)
        self.compute_forces(sims, scene)
        self.apply_particle_traction_constraints(sims, scene)
        self.apply_traction_constraints(sims, scene)  # node
        self.apply_absorbing_constraints(sims, scene)
        self.compute_prediction_grid_kinematic_(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.pre_contact_calculate(sims, scene)
        self.compute_contact_force_prediction(sims, scene)
        # self.compute_Poisson_equation_implicit(sims, scene)
        # self.compute_correction_grid_kinematics(sims, scene)
        # self.apply_kinematic_constraints(sims, scene)
        # self.compute_contact_force_correction(sims, scene)
        self.compute_affine_matrix(sims, scene)
        self.compute_particle_kinematic(sims, scene)

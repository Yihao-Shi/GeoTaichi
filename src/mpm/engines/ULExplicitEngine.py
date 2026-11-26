from src.mpm.boundaries.BoundaryCore import *
from src.mpm.engines.Engine import Engine
from src.mpm.engines.EngineKernel import *
from src.mpm.SceneManager import SceneManager
from src.mpm.Simulation import Simulation
from src.mpm.SpatialHashGrid import SpatialHashGrid
from src.utils.errors import SimulationStateError
from src.utils.linalg import no_operation


@ti.data_oriented
class ULExplicitEngine(Engine):
    def __init__(self, sims) -> None:
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
        self.apply_adaptive_constraints = no_operation
        self.apply_adaptive_velocity_constraints = no_operation
        self.update_adaptive_grid = no_operation
        super().__init__(sims)

    def reset_grid_message(self, scene: SceneManager):
        if self.distributed_mpm is None:
            super().reset_grid_message(scene)
        else:
            # Forces at zero-mass nodes must not enter the next collective.
            scene.node.fill(0)

    def calculate_interpolations(self, sims: Simulation, scene: SceneManager):
        if self.distributed_mpm is not None and sims.is_2DAxisy:
            self.check_axisymmetric_radii(scene)
        super().calculate_interpolations(sims, scene)

    def map_nodal_kinematics(self, sims: Simulation, scene: SceneManager):
        self.compute_nodal_kinematic(sims, scene)
        if self.distributed_mpm is not None:
            if sims.configuration == "TLMPM":
                self.distributed_mpm.sum_field(scene.node.momentum)
            else:
                self.distributed_mpm.reduce_grid(force=False)
        self.compute_grid_velcity(sims, scene)

    def apply_grid_loads(self, sims: Simulation, scene: SceneManager):
        self.apply_particle_traction_constraints(sims, scene)
        if self.distributed_mpm is not None:
            self.distributed_mpm.reduce_grid(force=True)
        self.apply_traction_constraints(sims, scene)
        self.apply_absorbing_constraints(sims, scene)

    def adaptive_timestep(self, sims: Simulation, scene: SceneManager):
        if self.distributed_mpm is None:
            super().adaptive_timestep(sims, scene)
        elif sims.adaptive_timestep > 0 and sims.current_step % sims.adaptive_timestep == 0:
            min_dt = self.distributed_mpm.minimum_timestep(scene.adaptive_timestep(sims))
            sims.set_timestep(min(sims.init_delta, sims.CFL * min_dt))

    def choose_boundary_constraints(self, sims: Simulation, scene: SceneManager):
        super().choose_boundary_constraints(sims, scene)
        if getattr(scene.element, "adaptive", False):
            if hasattr(scene.element, "configure_penalty_scale"):
                scene.element.configure_penalty_scale(scene.material)
            self.apply_adaptive_constraints = self.hanging_node_constraints
            self.apply_adaptive_velocity_constraints = self.hanging_node_velocity_constraints
            self.update_adaptive_grid = self.update_grid_refinement
        if int(scene.boundary.velocity_list[0]) > 0:
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

    def calculate_precontact_2d_axisy(self, sims: Simulation, scene: SceneManager):
        kernel_calc_contact_normal_2d_axisy(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.element.gnum,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.dshape_fn,
            scene.element.node_size,
        )
        if self.distributed_mpm is not None:
            self.distributed_mpm.sum_field(scene.node.grad_domain)

    def calculate_precontact(self, sims: Simulation, scene: SceneManager):
        kernel_calc_contact_normal(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.element.gnum,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.dshape_fn,
            scene.element.node_size,
        )
        if self.distributed_mpm is not None:
            self.distributed_mpm.sum_field(scene.node.grad_domain)

    def compute_geocontact_force(self, sims: Simulation, scene: SceneManager):
        kernel_calc_contact_displacement(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.mass_cut_off,
            scene.element.gnum,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
            scene.element,
        )
        if self.distributed_mpm is not None:
            self.distributed_mpm.sum_field(scene.node.contact_pos)
        kernel_calc_geocontact(
            scene.mass_cut_off,
            scene.contact.friction,
            scene.contact.alpha,
            scene.contact.beta,
            scene.contact.cut_off,
            scene.element.gnum,
            scene.element.grid_size,
            sims.dt,
            scene.is_rigid,
            scene.node,
            scene.element,
            scene.sparse_grid,
        )
        self.apply_contact_velocity_constraints(sims, scene)
        self.apply_contact_reflection_constraints(sims, scene)
        kernel_assemble_contact_force(scene.mass_cut_off, sims.dt, scene.node)

    def compute_demcontact_force_2d(self, sims: Simulation, scene: SceneManager):
        for material_id in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[material_id]
            end_index = scene.material.mapping[material_id + 1]
            kernel_calc_demcontact_2d(
                scene.element.grid_nodes,
                start_index,
                end_index,
                scene.element.gnum,
                scene.element.grid_size,
                scene.contact.velocity,
                scene.particle,
                scene.material.materialID,
                scene.material.matProps[material_id + 1],
                sims.dt,
                scene.contact.polygon_vertices,
                scene.element.LnID,
                scene.element.shape_fn,
                scene.element.node_size,
                scene.node,
                scene.element,
            )
        if self.distributed_mpm is not None:
            self.distributed_mpm.sum_field(scene.node.contact_force)
        kernel_move_contact_polygon(scene.contact.polygon_vertices, scene.contact.velocity, sims.dt)
        self.apply_contact_velocity_constraints(sims, scene)
        self.apply_contact_reflection_constraints(sims, scene)
        kernel_assemble_contact_force(scene.mass_cut_off, sims.dt, scene.node)

    def compute_contact_force(self, sims: Simulation, scene: SceneManager):
        kernel_calc_friction_contact(
            scene.mass_cut_off, scene.contact.friction, scene.element.gnum, sims.dt, scene.is_rigid, scene.node
        )
        self.apply_contact_velocity_constraints(sims, scene)
        self.apply_contact_reflection_constraints(sims, scene)
        kernel_assemble_contact_force(scene.mass_cut_off, sims.dt, scene.node)

    def compute_grid_velcity(self, sims: Simulation, scene: SceneManager):
        kernel_compute_grid_velocity(scene.mass_cut_off, scene.node)

    def compute_stress_strain(self, sims: Simulation, scene: SceneManager):
        for material_id in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[material_id]
            end_index = scene.material.mapping[material_id + 1]
            kernel_compute_stress_strain(
                start_index,
                end_index,
                sims.dt,
                scene.particle,
                scene.material.materialID,
                scene.material.matProps[material_id + 1],
                scene.material.stateVars,
            )

    def compute_stress_strain_2d(self, sims: Simulation, scene: SceneManager):
        for material_id in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[material_id]
            end_index = scene.material.mapping[material_id + 1]
            kernel_compute_stress_strain_2d(
                start_index,
                end_index,
                sims.dt,
                scene.particle,
                scene.material.materialID,
                scene.material.matProps[material_id + 1],
                scene.material.stateVars,
            )

    def compute_stress_strain_velocity_projection_2d(self, sims: Simulation, scene: SceneManager):
        scene.node.vol.fill(0)
        scene.node.jacobian.fill(0)
        scene.node.pressure.fill(0)
        scene.element.calc_shape_fn_spline_lower_order(scene.particleNum, scene.particle)
        kernel_dilatational_velocity_p2g(
            scene.element.grid_nodes_lower_order,
            int(scene.particleNum[0]),
            scene.element.gnum,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )
        if self.distributed_mpm is not None:
            self.distributed_mpm.sum_field(scene.node.jacobian)
            self.distributed_mpm.sum_field(scene.node.vol)
        kernel_grid_jacobian(scene.volume_cut_off, scene.is_rigid, scene.node)
        for material_id in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[material_id]
            end_index = scene.material.mapping[material_id + 1]
            kernel_gradient_velocity_projection_correction_2d(
                scene.element.grid_nodes_lower_order,
                start_index,
                end_index,
                scene.element.gnum,
                scene.node,
                scene.particle,
                scene.material.materialID,
                scene.element.LnID,
                scene.element.shape_fn,
                scene.element.node_size,
                scene.material.matProps[material_id + 1],
                scene.material.stateVars,
                sims.dt,
            )
        if self.distributed_mpm is not None:
            self.distributed_mpm.sum_field(scene.node.pressure)
        kernel_grid_pressure_volume(scene.volume_cut_off, scene.is_rigid, scene.node)
        kernel_pressure_correction(
            scene.element.grid_nodes_lower_order,
            int(scene.particleNum[0]),
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )

    def compute_stress_strain_velocity_projection(self, sims: Simulation, scene: SceneManager):
        scene.node.vol.fill(0)
        scene.node.jacobian.fill(0)
        scene.node.pressure.fill(0)
        scene.element.calc_shape_fn_spline_lower_order(scene.particleNum, scene.particle)
        kernel_dilatational_velocity_p2g(
            scene.element.grid_nodes_lower_order,
            int(scene.particleNum[0]),
            scene.element.gnum,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )
        if self.distributed_mpm is not None:
            self.distributed_mpm.sum_field(scene.node.jacobian)
            self.distributed_mpm.sum_field(scene.node.vol)
        kernel_grid_jacobian(scene.volume_cut_off, scene.is_rigid, scene.node)
        for material_id in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[material_id]
            end_index = scene.material.mapping[material_id + 1]
            kernel_gradient_velocity_projection_correction(
                scene.element.grid_nodes_lower_order,
                start_index,
                end_index,
                scene.element.gnum,
                scene.node,
                scene.particle,
                scene.material.materialID,
                scene.element.LnID,
                scene.element.shape_fn,
                scene.element.node_size,
                scene.material.matProps[material_id + 1],
                scene.material.stateVars,
                sims.dt,
            )
        if self.distributed_mpm is not None:
            self.distributed_mpm.sum_field(scene.node.pressure)
        kernel_grid_pressure_volume(scene.volume_cut_off, scene.is_rigid, scene.node)
        kernel_pressure_correction(
            scene.element.grid_nodes_lower_order,
            int(scene.particleNum[0]),
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )

    def update_velocity_gradient_2d(self, sims: Simulation, scene: SceneManager):
        for material_id in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[material_id]
            end_index = scene.material.mapping[material_id + 1]
            kernel_update_velocity_gradient_2d(
                scene.element.grid_nodes,
                start_index,
                end_index,
                sims.dt,
                scene.node,
                scene.particle,
                scene.material.materialID,
                scene.material.matProps[material_id + 1],
                scene.material.stateVars,
                scene.element.LnID,
                scene.element.dshape_fn,
                scene.element.node_size,
            )

    def update_velocity_gradient(self, sims: Simulation, scene: SceneManager):
        for material_id in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[material_id]
            end_index = scene.material.mapping[material_id + 1]
            kernel_update_velocity_gradient(
                scene.element.grid_nodes,
                start_index,
                end_index,
                sims.dt,
                scene.node,
                scene.particle,
                scene.material.materialID,
                scene.material.matProps[material_id + 1],
                scene.material.stateVars,
                scene.element.LnID,
                scene.element.dshape_fn,
                scene.element.node_size,
            )

    def update_velocity_gradient_bbar_2d(self, sims: Simulation, scene: SceneManager):
        for material_id in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[material_id]
            end_index = scene.material.mapping[material_id + 1]
            kernel_update_velocity_gradient_bbar_2d(
                scene.element.grid_nodes,
                start_index,
                end_index,
                sims.dt,
                scene.node,
                scene.particle,
                scene.material.materialID,
                scene.material.matProps[material_id + 1],
                scene.material.stateVars,
                scene.element.LnID,
                scene.element.dshape_fn,
                scene.element.dshape_fnc,
                scene.element.node_size,
            )

    def update_velocity_gradient_bbar(self, sims: Simulation, scene: SceneManager):
        for material_id in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[material_id]
            end_index = scene.material.mapping[material_id + 1]
            kernel_update_velocity_gradient_bbar(
                scene.element.grid_nodes,
                start_index,
                end_index,
                sims.dt,
                scene.node,
                scene.particle,
                scene.material.materialID,
                scene.material.matProps[material_id + 1],
                scene.material.stateVars,
                scene.element.LnID,
                scene.element.dshape_fn,
                scene.element.dshape_fnc,
                scene.element.node_size,
            )

    def update_velocity_gradient_axisy_2d(self, sims: Simulation, scene: SceneManager):
        for material_id in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[material_id]
            end_index = scene.material.mapping[material_id + 1]
            kernel_update_velocity_gradient_2d_axisy(
                scene.element.grid_nodes,
                start_index,
                end_index,
                sims.dt,
                scene.node,
                scene.particle,
                scene.material.materialID,
                scene.material.matProps[material_id + 1],
                scene.material.stateVars,
                scene.element.LnID,
                scene.element.shape_fn,
                scene.element.dshape_fn,
                scene.element.node_size,
            )

    def update_velocity_gradient_bbar_axisy_2d(self, sims: Simulation, scene: SceneManager):
        for material_id in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[material_id]
            end_index = scene.material.mapping[material_id + 1]
            kernel_update_velocity_gradient_bbar_2d_axisy(
                scene.element.grid_nodes,
                start_index,
                end_index,
                sims.dt,
                scene.node,
                scene.particle,
                scene.material.materialID,
                scene.material.matProps[material_id + 1],
                scene.material.stateVars,
                scene.element.LnID,
                scene.element.shape_fn,
                scene.element.shape_fnc,
                scene.element.dshape_fn,
                scene.element.dshape_fnc,
                scene.element.node_size,
            )

    def particle_shifting(self, sims: Simulation, scene: SceneManager):
        scene.node.vol.fill(0)
        kernel_volume_p2g(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.element.gnum,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )
        kernel_particle_shifting_delta_correction(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.element.grid_size,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.dshape_fn,
            scene.element.node_size,
        )

    def compute_force(self, sims: Simulation, scene: SceneManager):
        kernel_force_p2g(
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

    def compute_force_2d(self, sims: Simulation, scene: SceneManager):
        kernel_force_p2g_2d(
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

    def compute_force_2d_axisy(self, sims: Simulation, scene: SceneManager):
        kernel_force_p2g_2d_axisy(
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

    def compute_force_bbar_2d_axisy(self, sims: Simulation, scene: SceneManager):
        kernel_force_bbar_p2g_2d_axisy(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.element.gnum,
            sims.gravity,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.shape_fnc,
            scene.element.dshape_fn,
            scene.element.dshape_fnc,
            scene.element.node_size,
        )

    def compute_force_mls(self, sims: Simulation, scene: SceneManager):
        kernel_force_mls_p2g(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            sims.gravity,
            scene.element.gnum,
            scene.element.grid_size,
            scene.element.inertia_tensor,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )

    def compute_force_mls_2d(self, sims: Simulation, scene: SceneManager):
        kernel_force_mls_p2g_2d(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            sims.gravity,
            scene.element.gnum,
            scene.element.grid_size,
            scene.element.inertia_tensor,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )

    def compute_force_gauss(self, sims: Simulation, scene: SceneManager):
        kernel_external_force_p2g(
            scene.element.grid_nodes,
            scene.element.gnum,
            sims.gravity,
            int(scene.particleNum[0]),
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )
        gauss_cell_reset(scene.element.cell, scene.element.gauss_cell)
        kernel_volume_p2c(
            scene.element.cnum, scene.element.igrid_size, int(scene.particleNum[0]), scene.element.cell, scene.particle
        )
        if self.distributed_mpm is not None:
            self.distributed_mpm.sum_field(scene.element.cell.volume)
        kernel_find_valid_element(scene.element.cell_volume, scene.element.cell)
        kernel_sum_cell_stress(
            sims.gauss_number,
            scene.element.grid_size,
            scene.element.igrid_size,
            scene.element.cnum,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.cell,
            scene.element.gauss_cell,
        )
        if self.distributed_mpm is not None:
            self.distributed_mpm.sum_field(scene.element.gauss_cell.vol)
            self.distributed_mpm.sum_field(scene.element.gauss_cell.stress)
        kernel_compute_gauss_average_stress(
            sims.gauss_number, scene.volume_cut_off, scene.element.cell, scene.element.gauss_cell
        )
        kernel_average_pressure(sims.gauss_number, scene.element.cell, scene.element.gauss_cell)
        # The global Gauss integral must enter the force sum only once.
        if self.distributed_mpm is None or self.distributed_mpm.dist.get_rank() == 0:
            kernel_internal_force_on_gauss_point_p2g(
                sims.gauss_number,
                scene.element.cnum,
                scene.element.gnum,
                scene.element.grid_size,
                scene.element.igrid_size,
                scene.node,
                scene.element.cell,
                scene.element.gauss_cell,
                scene.element.gauss_point.gpcoords,
                scene.element.gauss_point.weight,
            )
        kernel_internal_force_on_material_point_p2g(
            scene.element.cnum,
            scene.element.gnum,
            scene.element.grid_size,
            scene.element.igrid_size,
            int(scene.particleNum[0]),
            scene.node,
            scene.particle,
            scene.element.cell,
        )

    def compute_force_gauss_2d(self, sims: Simulation, scene: SceneManager):
        kernel_external_force_p2g(
            scene.element.grid_nodes,
            scene.element.gnum,
            sims.gravity,
            int(scene.particleNum[0]),
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )
        gauss_cell_reset(scene.element.cell, scene.element.gauss_cell)
        kernel_volume_p2c_2d(
            scene.element.cnum, scene.element.igrid_size, int(scene.particleNum[0]), scene.element.cell, scene.particle
        )
        if self.distributed_mpm is not None:
            self.distributed_mpm.sum_field(scene.element.cell.volume)
        kernel_find_valid_element(scene.element.cell_volume, scene.element.cell)
        kernel_sum_cell_stress_2d(
            sims.gauss_number,
            scene.element.grid_size,
            scene.element.igrid_size,
            scene.element.cnum,
            int(scene.particleNum[0]),
            scene.particle,
            scene.element.cell,
            scene.element.gauss_cell,
        )
        if self.distributed_mpm is not None:
            self.distributed_mpm.sum_field(scene.element.gauss_cell.vol)
            self.distributed_mpm.sum_field(scene.element.gauss_cell.stress)
        kernel_compute_gauss_average_stress_2d(
            sims.gauss_number, scene.volume_cut_off, scene.element.cell, scene.element.gauss_cell
        )
        kernel_average_pressure_2d(sims.gauss_number, scene.element.cell, scene.element.gauss_cell)
        # The global Gauss integral must enter the force sum only once.
        if self.distributed_mpm is None or self.distributed_mpm.dist.get_rank() == 0:
            kernel_internal_force_on_gauss_point_p2g_2d(
                sims.gauss_number,
                scene.element.cnum,
                scene.element.gnum,
                scene.element.grid_size,
                scene.element.igrid_size,
                scene.node,
                scene.element.cell,
                scene.element.gauss_cell,
                scene.element.gauss_point.gpcoords,
                scene.element.gauss_point.weight,
            )
        kernel_internal_force_on_material_point_p2g_2d(
            scene.element.cnum,
            scene.element.gnum,
            scene.element.grid_size,
            scene.element.igrid_size,
            int(scene.particleNum[0]),
            scene.node,
            scene.particle,
            scene.element.cell,
        )

    def compute_force_bbar(self, sims: Simulation, scene: SceneManager):
        kernel_force_bbar_p2g(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.element.gnum,
            sims.gravity,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.dshape_fn,
            scene.element.dshape_fnc,
            scene.element.node_size,
        )

    def compute_force_bbar_2d(self, sims: Simulation, scene: SceneManager):
        kernel_force_bbar_p2g_2d(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.element.gnum,
            sims.gravity,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.dshape_fn,
            scene.element.dshape_fnc,
            scene.element.node_size,
        )

    def compute_grid_kinematic(self, sims: Simulation, scene: SceneManager):
        kernel_compute_grid_kinematic(scene.mass_cut_off, sims.background_damping, scene.node, sims.dt)

    def apply_kinematic_constraints(self, sims: Simulation, scene: SceneManager):
        self.apply_friction_constraints(sims, scene)
        self.apply_reflection_constraints(sims, scene)
        self.apply_velocity_constraints(sims, scene)

    def apply_dirichlet_constraints(self, sims: Simulation, scene: SceneManager):
        self.apply_reflection_constraints(sims, scene)
        self.apply_velocity_constraints(sims, scene)

    def traction_constraints(self, sims: Simulation, scene: SceneManager):
        if scene.sparse_grid is not None:
            apply_traction_constraint_sparse(
                int(scene.boundary.traction_list[0]),
                scene.boundary.traction_boundary,
                scene.node,
                scene.element.gnum,
                scene.sparse_grid.block_count,
                scene.sparse_grid.block_size,
                scene.sparse_grid.block_volume,
                scene.sparse_grid.block_map,
            )
        else:
            apply_traction_constraint(
                int(scene.boundary.traction_list[0]), scene.boundary.traction_boundary, scene.node
            )

    def traction_constraints_2d(self, sims: Simulation, scene: SceneManager):
        if scene.sparse_grid is not None:
            apply_traction_constraint_sparse(
                int(scene.boundary.traction_list[0]),
                scene.boundary.traction_boundary,
                scene.node,
                scene.element.gnum,
                scene.sparse_grid.block_count,
                scene.sparse_grid.block_size,
                scene.sparse_grid.block_volume,
                scene.sparse_grid.block_map,
            )
        else:
            apply_traction_constraint_2d(
                int(scene.boundary.traction_list[0]), scene.boundary.traction_boundary, scene.node
            )

    def absorbing_constraints(self, sims: Simulation, scene: SceneManager):
        apply_absorbing_constraint(
            int(scene.boundary.absorbing_list[0]), scene.boundary.absorbing_boundary, scene.material, scene.node
        )

    def velocity_constraints(self, sims: Simulation, scene: SceneManager):
        if scene.sparse_grid is not None:
            apply_velocity_constraint_sparse(
                scene.mass_cut_off,
                int(scene.boundary.velocity_list[0]),
                scene.boundary.velocity_boundary,
                scene.is_rigid,
                scene.node,
                scene.element.gnum,
                scene.sparse_grid.block_count,
                scene.sparse_grid.block_size,
                scene.sparse_grid.block_volume,
                scene.sparse_grid.block_map,
            )
        else:
            apply_velocity_constraint(
                scene.mass_cut_off,
                int(scene.boundary.velocity_list[0]),
                scene.boundary.velocity_boundary,
                scene.is_rigid,
                scene.node,
            )

    def reflection_constraints(self, sims: Simulation, scene: SceneManager):
        if scene.sparse_grid is not None:
            apply_reflection_constraint_sparse(
                scene.mass_cut_off,
                int(scene.boundary.reflection_list[0]),
                scene.boundary.reflection_boundary,
                scene.is_rigid,
                scene.node,
                scene.element.gnum,
                scene.sparse_grid.block_count,
                scene.sparse_grid.block_size,
                scene.sparse_grid.block_volume,
                scene.sparse_grid.block_map,
            )
        else:
            apply_reflection_constraint(
                scene.mass_cut_off,
                int(scene.boundary.reflection_list[0]),
                scene.boundary.reflection_boundary,
                scene.is_rigid,
                scene.node,
            )

    def contact_velocity_constraints(self, sims: Simulation, scene: SceneManager):
        apply_contact_velocity_constraint(
            scene.mass_cut_off,
            int(scene.boundary.velocity_list[0]),
            scene.boundary.velocity_boundary,
            scene.node,
            scene.element.gnum,
            scene.sparse_grid,
        )

    def contact_reflection_constraints(self, sims: Simulation, scene: SceneManager):
        apply_contact_reflection_constraint(
            scene.mass_cut_off,
            int(scene.boundary.reflection_list[0]),
            scene.boundary.reflection_boundary,
            scene.node,
            scene.element.gnum,
            scene.sparse_grid,
        )

    def friction_constraints(self, sims: Simulation, scene: SceneManager):
        if scene.sparse_grid is not None:
            apply_friction_constraint_sparse(
                scene.mass_cut_off,
                int(scene.boundary.friction_list[0]),
                scene.boundary.friction_boundary,
                scene.is_rigid,
                scene.node,
                sims.dt,
                scene.element.gnum,
                scene.sparse_grid.block_count,
                scene.sparse_grid.block_size,
                scene.sparse_grid.block_volume,
                scene.sparse_grid.block_map,
            )
        else:
            apply_friction_constraint(
                scene.mass_cut_off,
                int(scene.boundary.friction_list[0]),
                scene.boundary.friction_boundary,
                scene.is_rigid,
                scene.node,
                sims.dt,
            )

    def hanging_node_constraints(self, sims: Simulation, scene: SceneManager):
        scene.element.apply_hanging_constraints(scene.mass_cut_off, sims.dt, scene.node, scene.particle)

    def hanging_node_velocity_constraints(self, sims: Simulation, scene: SceneManager):
        scene.element.apply_hanging_velocity_constraints(scene.mass_cut_off, sims.dt, scene.node, scene.particle)

    def update_grid_refinement(self, sims: Simulation, scene: SceneManager):
        scene.element.update_refinement(sims, scene)

    def update_grid_refinement_profiled(self, sims: Simulation, scene: SceneManager):
        if getattr(scene.element, "adaptive", False) and scene.element.should_update_refinement(
            sims, scene.particleNum
        ):
            sims.timer.begin("Adaptive grid")
            self.update_adaptive_grid(sims, scene)
            sims.timer.end("Adaptive grid")

    def compute_particle_kinematics(self, sims: Simulation, scene: SceneManager):
        kernel_kinemaitc_g2p(
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

    def postmapping_grid_velocity(self, sims: Simulation, scene: SceneManager):
        kernel_reset_grid_velocity(scene.node)
        kernel_postmapping_kinemaitc(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.element.gnum,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )
        if self.distributed_mpm is not None:
            self.distributed_mpm.sum_field(scene.node.momentum)

    def pre_calculation(self, sims: Simulation, scene: SceneManager, neighbor: SpatialHashGrid):
        scene.element.calculate_characteristic_length(sims, int(scene.particleNum[0]), scene.particle, scene.psize)
        if sims.neighbor_detection:
            scene.check_in_domain(sims)
            if sims.free_surface_detection:
                self.reset_grid_messages(scene)
                self.find_free_surface_by_density(sims, scene)
            neighbor.place_particles(scene)
            self.compute_boundary_direction(scene, neighbor)
            self.free_surface_by_geometry(scene, neighbor)
            if sims.free_surface_detection:
                self.reset_grid_messages(scene)
        self.limit = sims.verlet_distance * sims.verlet_distance

    def usl_updating(self, sims: Simulation, scene: SceneManager, neighbor=None):
        sims.timer.begin("Shape function")
        self.calculate_interpolation(sims, scene)
        sims.timer.end("Shape function")
        sims.timer.begin("P2G")
        self.map_nodal_kinematics(sims, scene)
        self.compute_forces(sims, scene)
        sims.timer.end("P2G")
        sims.timer.begin("Traction boundary")
        self.apply_grid_loads(sims, scene)
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
        sims.timer.begin("Misc")
        self.pressure_smoothing_(sims, scene)
        sims.timer.end("Misc")
        self.update_grid_refinement_profiled(sims, scene)

    def affine_updating(self, sims: Simulation, scene: SceneManager, neighbor=None):
        if sims.mode == "Lightweight":
            return self.lightweight(sims, scene, neighbor)
        sims.timer.begin("Shape function")
        self.calculate_interpolation(sims, scene)
        sims.timer.end("Shape function")
        sims.timer.begin("P2G")
        self.map_nodal_kinematics(sims, scene)
        self.compute_forces(sims, scene)
        sims.timer.end("P2G")
        sims.timer.begin("Traction boundary")
        self.apply_grid_loads(sims, scene)
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
        sims.timer.begin("Stress update")
        self.compute_velocity_gradient(sims, scene)
        self.compute_stress_strains(sims, scene)
        # Save the transfer matrix from final grid velocities before G2P moves the particles.
        if sims.stabilize == "Displacement F-Bar Method":
            self.calculate_interpolation(sims, scene)
        self.compute_affine_velocity_gradient(sims, scene)
        sims.timer.end("Stress update")
        sims.timer.begin("G2P")
        self.compute_particle_kinematic(sims, scene)
        sims.timer.end("G2P")
        sims.timer.begin("Misc")
        self.pressure_smoothing_(sims, scene)
        sims.timer.end("Misc")
        self.update_grid_refinement_profiled(sims, scene)

    def usf_updating(self, sims: Simulation, scene: SceneManager, neighbor=None):
        sims.timer.begin("Shape function")
        self.calculate_interpolation(sims, scene)
        sims.timer.end("Shape function")
        sims.timer.begin("P2G-1")
        self.map_nodal_kinematics(sims, scene)
        self.apply_dirichlet_constraints(sims, scene)
        self.apply_adaptive_velocity_constraints(sims, scene)
        self.apply_dirichlet_constraints(sims, scene)
        sims.timer.end("P2G-1")
        sims.timer.begin("Stress update")
        self.compute_velocity_gradient(sims, scene)
        self.compute_stress_strains(sims, scene)
        self.compute_affine_velocity_gradient(sims, scene)
        sims.timer.end("Stress update")
        sims.timer.begin("Misc")
        self.pressure_smoothing_(sims, scene)
        sims.timer.end("Misc")
        sims.timer.begin("P2G-2")
        self.compute_forces(sims, scene)
        sims.timer.end("P2G-2")
        sims.timer.begin("Traction boundary")
        self.apply_grid_loads(sims, scene)
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
        self.update_grid_refinement_profiled(sims, scene)

    def musl_updating(self, sims: Simulation, scene: SceneManager, neighbor=None):
        sims.timer.begin("Shape function")
        self.calculate_interpolation(sims, scene)
        sims.timer.end("Shape function")
        sims.timer.begin("P2G")
        self.map_nodal_kinematics(sims, scene)
        self.compute_forces(sims, scene)
        sims.timer.end("P2G")
        sims.timer.begin("Traction boundary")
        self.apply_grid_loads(sims, scene)
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
        sims.timer.begin("Misc")
        self.pressure_smoothing_(sims, scene)
        sims.timer.end("Misc")
        self.update_grid_refinement_profiled(sims, scene)

    def first_substep_cfd_coupling(self, sims: Simulation, scene: SceneManager):
        self.calculate_interpolation(sims, scene)
        self.compute_nodal_kinematic(sims, scene)
        self.compute_grid_velcity(sims, scene)
        self.apply_dirichlet_constraints(sims, scene)
        self.compute_stress_strains(sims, scene)
        self.pressure_smoothing_(sims, scene)
        self.compute_forces(sims, scene)

    def second_substep_cfd_coupling(self, sims: Simulation, scene: SceneManager):
        self.apply_particle_traction_constraints(sims, scene)
        self.apply_traction_constraints(sims, scene)
        self.apply_absorbing_constraints(sims, scene)
        self.compute_grid_kinematic(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.compute_velocity_gradient(sims, scene)
        self.compute_particle_kinematic(sims, scene)

    def check_axisymmetric_radii(self, scene: SceneManager):
        invalid_radius = lightweight_has_invalid_radius(int(scene.particleNum[0]), scene.particle)
        if self.distributed_mpm is not None:
            invalid_radius = self.distributed_mpm.any_invalid_radius(invalid_radius)
        if invalid_radius:
            raise SimulationStateError("Axisymmetric MPM requires finite positive particle radii.")

    def lightweight(self, sims: Simulation, scene: SceneManager, neighbor: SpatialHashGrid):
        axisymmetric = sims.is_2DAxisy
        particle_num = int(scene.particleNum[0])
        if axisymmetric:
            self.check_axisymmetric_radii(scene)
        if scene.sparse_grid is not None:
            sims.timer.begin("Shape function")
            scene.sparse_grid.rebuild_lightweight(particle_num, scene.element, scene.particle)
            sims.timer.end("Shape function")
        transfer = (
            particle_num,
            scene.element.gnum,
            scene.element.grid_size,
            scene.element.igrid_size,
            sims.gravity,
            scene.element.calLength,
            scene.element.boundary_type,
            scene.node,
            scene.particle,
            scene.sparse_grid,
            axisymmetric,
            axisymmetric and sims.contact_detection == "MPMContact",
        )
        sims.timer.begin("P2G-1")
        lightweight_p2g(*transfer, False)
        if self.distributed_mpm is not None:
            self.distributed_mpm.reduce_grid(force=False)
        self.compute_grid_velcity(sims, scene)
        self.apply_dirichlet_constraints(sims, scene)
        sims.timer.end("P2G-1")
        sims.timer.begin("Stress update")
        lightweight_update_stress(
            particle_num,
            scene.mass_cut_off,
            sims.fbar_fraction,
            scene.element.gnum,
            scene.element.grid_size,
            scene.element.igrid_size,
            sims.dt,
            scene.element.calLength,
            scene.element.boundary_type,
            scene.node,
            scene.particle,
            scene.sparse_grid,
            scene.material.matProps,
            scene.material.stateVars,
            axisymmetric,
        )
        sims.timer.end("Stress update")
        sims.timer.begin("P2G-2")
        lightweight_p2g(*transfer, True)
        sims.timer.end("P2G-2")
        sims.timer.begin("Traction boundary")
        self.apply_grid_loads(sims, scene)
        sims.timer.end("Traction boundary")
        sims.timer.begin("Grid kinematic")
        self.compute_grid_kinematic(sims, scene)
        sims.timer.end("Grid kinematic")
        sims.timer.begin("Contact & Kinematic boundary")
        if not axisymmetric:
            self.pre_contact_calculate(sims, scene)
        self.apply_kinematic_constraints(sims, scene)
        self.apply_dirichlet_constraints(sims, scene)
        self.compute_contact_force_(sims, scene)
        sims.timer.end("Contact & Kinematic boundary")
        sims.timer.begin("G2P")
        lightweight_g2p(
            particle_num,
            sims.alphaPIC,
            scene.element.gnum,
            scene.element.grid_size,
            scene.element.igrid_size,
            sims.dt,
            scene.element.calLength,
            scene.element.boundary_type,
            scene.node,
            scene.particle,
            scene.sparse_grid,
            axisymmetric,
        )
        sims.timer.end("G2P")

    def test(self, sims: Simulation, scene: SceneManager, neighbor: SpatialHashGrid):
        pass

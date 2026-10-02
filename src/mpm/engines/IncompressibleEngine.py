import os

import taichi as ti

from src.mpm.boundaries.BoundaryCore import apply_reflection_constraint, apply_velocity_constraint
from src.mpm.Simulation import Simulation
from src.mpm.SceneManager import myScene
from src.mpm.SpatialHashGrid import SpatialHashGrid
from src.mpm.engines.Engine import Engine
from src.mpm.engines.EngineKernel import *
from src.mpm.engines.AssembleMatrixKernel import *
from src.mpm.engines.FreeSurfaceDetection import *
from src.utils.linalg import no_operation


class IncompressibleEngine(Engine):
    def __init__(self, sims) -> None:
        super().__init__(sims)
        self.poisson_solver = None
        self.operator = None
        self.initial_pressure_weight = None
        self.shifting_node_volume = None
        self.shifting_reference_volume = None
        self.shifting_reference_ready = False
        self.density_projection_cell_density = None
        self.density_projection_cell_fraction = None
        self.density_projection_pressure = None
        self.density_projection_face_displacement = None
        self.density_projection_needs_solve = None
        self.solid_face_open_fraction = None
        self.solid_face_velocity = None
        self.ibm_solid_fraction = None
        self.ibm_solid_density = None
        self.ibm_solid_velocity_cell = None
        self.ibm_force_cell = None
        self.pressure_coupling_mode = 0
        self.pressure_solid_fraction = None
        self.pressure_previous_solid_fraction = None
        self.pressure_solid_density = None
        self.coupled_viscous_delta = None
        self.density_projection_fields_ready = False
        self.fluid_sdf_build_step = -1
        self.trace_precalculation = os.environ.get("GEOTAICHI_TRACE_PRECALC", "0") != "0"
        self.dimension = int(sims.dimension)
        self.use_pressure_free_surface_theta = False
        self.update_external_cut_cell_boundary = no_operation
        self.update_external_ibm_fields = no_operation
        self.update_external_fluid_level_set = no_operation
        self.run_external_fluid_particle_coupling = no_operation
        self.ensure_solid_cut_cell_fields_step = None
        self.reset_solid_face_velocity_cut_cell_bound = None
        self.update_solid_cut_cell_bound = None
        self.create_pcg_solver = None
        self.create_mgpcg_solver = None

    def choose_engine(self, sims: Simulation):
        if sims.discretization == "FDM":
            if sims.linear_solver == "PCG":
                self.compute = self.fdm_discretization_pcg
            elif sims.linear_solver == "MGPCG":
                self.compute = self.fdm_discretization_mgpcg
        elif sims.discretization == "FEM":
            self.compute = self.fem_discretization

    def manage_function(self, sims: Simulation):
        self.is_verlet_update = self.is_need_update_verlet_table
        self.bulid_neighbor_list = no_operation
        self.run_pressure_fluid_level_set = no_operation
        self.run_cut_cell = no_operation
        self.run_external_fluid_particle_coupling = no_operation
        self.run_ibm_source = no_operation
        self.run_coupled_viscosity = no_operation
        self.run_density_projection = no_operation
        self.run_density_projection_level_set = no_operation
        self.run_particle_shifting = no_operation
        self.reset_solid_face_velocity_step = no_operation
        self.update_solid_cut_cell_step = no_operation
        self.compute_density_projection_fluid_fraction_step = no_operation
        self.ensure_solid_cut_cell_fields_step = None
        self.reset_solid_face_velocity_cut_cell_bound = None
        self.update_solid_cut_cell_bound = None
        self.create_pcg_solver = None
        self.create_mgpcg_solver = None
        self.prepare_density_projection_equations_step = None
        self.prepare_density_projection_equations_mgpcg_step = None
        self.prepare_poisson_equations_step = None
        self.prepare_poisson_equations_mgpcg_step = None
        self.apply_pressures_step = None
        self.apply_pressures_mgpcg_step = None
        self.enforce_mac_boundary_step = None
        self.enforce_mac_boundary_after_pressure_step = None
        self.solid_face_fraction2 = None
        self.solid_face_velocity2 = None
        if sims.neighbor_detection:
            self.compute_nodal_kinematic = no_operation
            self.execute_board_serach = self.update_verlet_table
            self.system_resolve = self.compute_nodal_kinematic
            self.bulid_neighbor_list = self.board_search

            self.free_surface_by_geometry = no_operation
            if sims.free_surface_detection:
                self.free_surface_by_geometry = self.detection_free_surface

            self.compute_boundary_direction = no_operation
            if sims.boundary_direction_detection:
                self.compute_boundary_direction = self.detection_boundary_direction

    def choose_boundary_constraints(self, sims: Simulation, scene: myScene):
        pass

    def reset_grid_message(self, scene: myScene):
        scene.node.grid_reset(scene.mass_cut_off)

    def trace_precalculation_stage(self, name):
        if self.trace_precalculation:
            print(f"# Pre-calculate: {name}", flush=True)

    def compute_nodal_kinematics(self, sims: Simulation, scene: myScene):
        particle_num = int(scene.particleNum[0])
        self.reset_fluid_domain_volume(scene)
        kernel_mass_momentum_mac_cell_p2g(
            scene.element.grid_nodes,
            particle_num,
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.grid_size,
            scene.element.igrid_size,
            scene.node,
            scene.particle,
            scene.element.calLength,
            scene.element.cell_volumefrac,
            scene.element.cell_volume,
            sims.is_2DAxisy,
        )
        self.classify_fluid_domain(sims, scene)

    def update_velocity_gradient_2D(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_update_velocity_gradient_2D(
                scene.element.grid_nodes,
                start_index,
                end_index,
                sims.dt,
                scene.node,
                scene.particle,
                scene.material.materialID,
                scene.material.matProps[materialID + 1],
                scene.material.stateVars,
                scene.element.LnID,
                scene.element.dshape_fn,
                scene.element.node_size,
            )

    def update_velocity_gradient(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_update_velocity_gradient(
                scene.element.grid_nodes,
                start_index,
                end_index,
                sims.dt,
                scene.node,
                scene.particle,
                scene.material.materialID,
                scene.material.matProps[materialID + 1],
                scene.material.stateVars,
                scene.element.LnID,
                scene.element.dshape_fn,
                scene.element.node_size,
            )

    def compute_force(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_viscous_force_p2g(
                scene.element.grid_nodes,
                start_index,
                end_index,
                scene.element.gnum,
                sims.dt,
                scene.node,
                scene.particle,
                scene.element.LnID,
                scene.element.dshape_fn,
                scene.element.node_size,
                scene.material.matProps[materialID + 1],
            )

    def compute_force_2D(self, sims: Simulation, scene: myScene):
        for materialID in range(scene.material.mapping.shape[0] - 1):
            start_index = scene.material.mapping[materialID]
            end_index = scene.material.mapping[materialID + 1]
            kernel_viscous_force_p2g_2D(
                scene.element.grid_nodes,
                start_index,
                end_index,
                scene.element.gnum,
                sims.dt,
                scene.node,
                scene.particle,
                scene.element.LnID,
                scene.element.dshape_fn,
                scene.element.node_size,
                scene.material.matProps[materialID + 1],
            )

    def reflection_constraints(self, sims: Simulation, scene: myScene):
        apply_reflection_constraint(
            scene.mass_cut_off,
            int(scene.boundary.reflection_list[0]),
            scene.boundary.reflection_boundary,
            scene.is_rigid,
            scene.node,
        )

    def velocity_constraints(self, sims: Simulation, scene: myScene):
        apply_velocity_constraint(
            scene.mass_cut_off,
            int(scene.boundary.velocity_list[0]),
            scene.boundary.velocity_boundary,
            scene.is_rigid,
            scene.node,
        )

    def apply_boundary_constraints(self, sims: Simulation, scene: myScene):
        self.apply_reflection_constraints(sims, scene)
        self.apply_velocity_constraints(sims, scene)

    def compute_grid_velcity(self, sims: Simulation, scene: myScene):
        kernel_compute_mac_grid_velocity(scene.mass_cut_off, scene.node)

    def compute_grid_velcity_gravity(self, sims: Simulation, scene: myScene):
        kernel_compute_mac_grid_velocity_gravity(scene.mass_cut_off, sims.gravity, sims.dt, scene.node)

    def has_ibm_source(self):
        return (
            self.ibm_solid_fraction is not None
            and self.ibm_solid_density is not None
            and self.ibm_solid_velocity_cell is not None
            and self.ibm_force_cell is not None
        )

    def ensure_ibm_source_fields(self, scene: myScene):
        if self.ibm_solid_fraction is None:
            offset = 0 * scene.element.cnum - scene.element.ghost_cell
            self.ibm_solid_fraction = ti.field(dtype=float, shape=scene.element.cnum, offset=offset)
        if self.ibm_solid_density is None:
            offset = 0 * scene.element.cnum - scene.element.ghost_cell
            self.ibm_solid_density = ti.field(dtype=float, shape=scene.element.cnum, offset=offset)
        if self.ibm_solid_velocity_cell is None:
            offset = 0 * scene.element.cnum - scene.element.ghost_cell
            self.ibm_solid_velocity_cell = ti.Vector.field(
                self.dimension, dtype=float, shape=scene.element.cnum, offset=offset
            )
        if self.ibm_force_cell is None:
            offset = 0 * scene.element.cnum - scene.element.ghost_cell
            self.ibm_force_cell = ti.Vector.field(self.dimension, dtype=float, shape=scene.element.cnum, offset=offset)

    def apply_ibm_source(self, sims: Simulation, scene: myScene):
        if not self.has_ibm_source():
            return
        mat_props = self.get_single_fluid_mat_props(scene)
        kernel_apply_incompressible_ibm_mac_source(
            scene.element.ghost_cell,
            scene.element.cnum,
            sims.dt,
            mat_props,
            scene.element.cell.type,
            self.ibm_solid_fraction,
            self.ibm_solid_density,
            self.ibm_solid_velocity_cell,
            self.ibm_force_cell,
            scene.node,
        )

    def add_grid_gravity(self, sims: Simulation, scene: myScene):
        kernel_add_mac_grid_gravity(sims.gravity, sims.dt, scene.node)

    def reset_fluid_domain_volume(self, scene: myScene):
        if scene.element.cell_volumefrac is None:
            scene.element.create_element_volume_fraction()
        cell_volume_reset(scene.element.cell_volumefrac)

    def classify_fluid_domain(self, sims: Simulation, scene: myScene):
        kernel_find_fluid_domain_by_volume(
            int(scene.particleNum[0]),
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.igrid_size,
            sims.fluid_domain_volume_fraction,
            scene.element.cell_volumefrac,
            scene.element.cell.type,
            scene.particle,
        )
        kernel_fill_enclosed_fluid_cells(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.cell.type,
        )

    def identify_fluid_domain(self, sims: Simulation, scene: myScene):
        self.reset_fluid_domain_volume(scene)
        calculate_cell_volume_weighted(
            scene.element.cell_volumefrac,
            int(scene.particleNum[0]),
            scene.element.ghost_cell,
            scene.particle,
            scene.element.cell_volume,
            scene.element.grid_size,
            scene.element.cnum,
            getattr(sims, "is_2DAxisy", False),
        )
        self.classify_fluid_domain(sims, scene)

    def needs_pressure_fluid_level_set(self, sims: Simulation, scene: myScene):
        mat_props = self.get_single_fluid_mat_props(scene)
        return bool(sims.fluid_level_set or getattr(mat_props, "surface_tension", 0.0) > 0.0)

    def needs_density_projection_fluid_level_set(self, sims: Simulation, scene: myScene):
        return bool(sims.density_projection and not sims.density_projection_interior_only)

    def needs_particle_fluid_level_set(self, sims: Simulation, scene: myScene):
        return bool(
            self.needs_pressure_fluid_level_set(sims, scene)
            or self.needs_density_projection_fluid_level_set(sims, scene)
        )

    def configure_runtime_functions(self, sims: Simulation, scene: myScene = None):
        self.run_pressure_fluid_level_set = no_operation
        needs_pressure_level_set = sims.discretization == "FDM"
        if scene is not None:
            needs_pressure_level_set = self.needs_pressure_fluid_level_set(sims, scene)
        self.use_pressure_free_surface_theta = bool(needs_pressure_level_set)
        if needs_pressure_level_set:
            self.run_pressure_fluid_level_set = self.timed_pressure_fluid_level_set

        if sims.dimension == 3:
            self.solid_face_fraction2 = self.solid_face_fraction_z
            self.solid_face_velocity2 = self.solid_face_velocity_z
            self.ensure_solid_cut_cell_fields_step = self.ensure_solid_cut_cell_fields3d
            self.reset_solid_face_velocity_cut_cell_bound = self.reset_solid_face_velocity_cut_cell3d
            self.update_solid_cut_cell_bound = self.update_solid_cut_cell3d
            self.create_pcg_solver = self.create_matrix_free_poisson_solver3d
            self.create_mgpcg_solver = self.create_mgpcg_poisson_solver3d
        else:
            self.solid_face_fraction2 = self.solid_face_fraction_y
            self.solid_face_velocity2 = self.solid_face_velocity_y
            self.ensure_solid_cut_cell_fields_step = self.ensure_solid_cut_cell_fields2d
            self.reset_solid_face_velocity_cut_cell_bound = self.reset_solid_face_velocity_cut_cell2d
            self.update_solid_cut_cell_bound = self.update_solid_cut_cell2d
            self.create_pcg_solver = self.create_matrix_free_poisson_solver2d
            self.create_mgpcg_solver = self.create_mgpcg_poisson_solver2d

        self.reset_solid_face_velocity_step = no_operation
        self.update_solid_cut_cell_step = no_operation
        self.run_cut_cell = no_operation
        self.prepare_poisson_equations_step = self.prepare_poisson_equations_regular
        self.prepare_poisson_equations_mgpcg_step = self.prepare_poisson_equations_mgpcg_regular
        self.apply_pressures_step = self.apply_pressures_regular
        self.apply_pressures_mgpcg_step = self.apply_pressures_mgpcg_regular
        self.enforce_mac_boundary_step = self.enforce_mac_boundary_regular
        self.enforce_mac_boundary_after_pressure_step = self.enforce_mac_boundary_after_pressure_regular
        self.prepare_density_projection_equations_step = self.prepare_density_projection_equations_regular
        self.prepare_density_projection_equations_mgpcg_step = self.prepare_density_projection_equations_mgpcg_regular
        if sims.solid_sdf_cut_cell:
            self.run_cut_cell = self.timed_update_solid_cut_cell
            self.reset_solid_face_velocity_step = self.reset_solid_face_velocity_cut_cell_bound
            self.update_solid_cut_cell_step = self.update_solid_cut_cell_bound
            self.prepare_poisson_equations_step = self.prepare_poisson_equations_cut_cell
            self.prepare_poisson_equations_mgpcg_step = self.prepare_poisson_equations_mgpcg_cut_cell
            self.apply_pressures_step = self.apply_pressures_cut_cell
            self.apply_pressures_mgpcg_step = self.apply_pressures_mgpcg_cut_cell
            self.enforce_mac_boundary_step = self.enforce_mac_boundary_cut_cell
            self.enforce_mac_boundary_after_pressure_step = self.enforce_mac_boundary_after_pressure_cut_cell
            self.prepare_density_projection_equations_step = self.prepare_density_projection_equations_cut_cell
            self.prepare_density_projection_equations_mgpcg_step = (
                self.prepare_density_projection_equations_mgpcg_cut_cell
            )

        # Newtonian viscosity also belongs to uncoupled incompressible flow.
        if scene is not None:
            self.ensure_coupled_viscosity_fields(scene)
        self.run_coupled_viscosity = self.apply_coupled_viscosity
        if self.pressure_coupling_mode > 0:
            self.prepare_poisson_equations_step = self.prepare_poisson_equations_coupled
            self.prepare_poisson_equations_mgpcg_step = self.prepare_poisson_equations_mgpcg_coupled
            self.apply_pressures_step = self.apply_pressures_coupled
            self.apply_pressures_mgpcg_step = self.apply_pressures_mgpcg_coupled

        self.run_ibm_source = (
            self.timed_apply_ibm_source
            if self.has_ibm_source() or self.update_external_ibm_fields is not no_operation
            else no_operation
        )

        self.run_density_projection = no_operation
        self.run_density_projection_level_set = no_operation
        self.compute_density_projection_fluid_fraction_step = no_operation
        if sims.density_projection:
            if sims.linear_solver == "PCG":
                self.run_density_projection = self.timed_density_projection_pcg
            elif sims.linear_solver == "MGPCG":
                self.run_density_projection = self.timed_density_projection_mgpcg
            if not sims.density_projection_interior_only:
                self.run_density_projection_level_set = self.timed_density_projection_level_set
                self.compute_density_projection_fluid_fraction_step = self.compute_density_projection_fluid_fraction

        self.run_particle_shifting = self.timed_particle_shifting if sims.particle_shifting else no_operation
        self.enforce_particle_boundary = (
            self.enforce_particle_domain_and_sdf_boundary
            if sims.solid_sdf_cut_cell
            else self.enforce_particle_domain_boundary
        )

    def timed_pressure_fluid_level_set(self, sims: Simulation, scene: myScene):
        sims.timer.begin("Fluid level set")
        self.update_fluid_level_set(sims, scene)
        sims.timer.end("Fluid level set")

    def timed_density_projection_level_set(self, sims: Simulation, scene: myScene):
        sims.timer.begin("Density projection level set")
        self.update_fluid_level_set(sims, scene, force_rebuild=True)
        sims.timer.end("Density projection level set")

    def timed_update_solid_cut_cell(self, sims: Simulation, scene: myScene):
        sims.timer.begin("Cut cell")
        self.update_solid_cut_cell_step(sims, scene)
        sims.timer.end("Cut cell")

    def timed_apply_ibm_source(self, sims: Simulation, scene: myScene):
        sims.timer.begin("IBM source")
        self.apply_ibm_source(sims, scene)
        sims.timer.end("IBM source")

    def ensure_coupled_viscosity_fields(self, scene: myScene):
        if self.coupled_viscous_delta is None:
            offset = [-scene.element.ghost_cell for _ in range(self.dimension)]
            self.coupled_viscous_delta = [
                ti.field(dtype=float, shape=scene.node.velocity[d].shape, offset=offset) for d in range(self.dimension)
            ]

    def apply_coupled_viscosity(self, sims: Simulation, scene: myScene):
        for field in self.coupled_viscous_delta:
            field.fill(0.0)
        # Mode zero does not read the coupled material fields.
        solid_fraction = self.pressure_solid_fraction if self.pressure_coupling_mode else scene.element.cell.type
        solid_density = self.pressure_solid_density if self.pressure_coupling_mode else scene.element.cell.type
        if sims.fluid_wall_no_slip:
            self.enforce_mac_boundary_step(sims, scene)
        kernel_compute_mac_viscous_delta(
            scene.mass_cut_off,
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.grid_size,
            sims.dt,
            self.get_single_fluid_mat_props(scene),
            scene.element.cell.type,
            solid_fraction,
            solid_density,
            self.pressure_coupling_mode,
            self.coupled_viscous_delta[0],
            self.coupled_viscous_delta[1],
            self.coupled_viscous_delta[-1],
            scene.node,
            sims.fluid_wall_no_slip,
        )
        kernel_apply_mac_viscous_delta(
            self.coupled_viscous_delta[0],
            self.coupled_viscous_delta[1],
            self.coupled_viscous_delta[-1],
            scene.node,
        )

    def timed_density_projection_pcg(self, sims: Simulation, scene: myScene):
        sims.timer.begin("Density projection")
        self.density_projection_pcg(sims, scene)
        sims.timer.end("Density projection")

    def timed_density_projection_mgpcg(self, sims: Simulation, scene: myScene):
        sims.timer.begin("Density projection")
        self.density_projection_mgpcg(sims, scene)
        sims.timer.end("Density projection")

    def timed_particle_shifting(self, sims: Simulation, scene: myScene):
        sims.timer.begin("Particle shifting")
        self.particle_shifting(sims, scene)
        sims.timer.end("Particle shifting")
        sims.timer.begin("Particle boundary")
        self.enforce_particle_boundary(scene)
        sims.timer.end("Particle boundary")

    def update_fluid_level_set(self, sims: Simulation, scene: myScene, force_rebuild=False):
        current_step = int(sims.current_step)
        use_particle_sdf = self.needs_particle_fluid_level_set(sims, scene)
        if force_rebuild or self.fluid_sdf_build_step != current_step:
            if use_particle_sdf:
                kernel_build_fluid_sdf_from_volume_fraction(
                    scene.element.ghost_cell,
                    scene.element.cnum,
                    scene.element.grid_size,
                    scene.element.cell_volumefrac,
                    scene.element.cell.type,
                    scene.element.cell.fluid_sdf,
                )
            else:
                kernel_build_fluid_sdf_from_cell_type(
                    scene.element.ghost_cell,
                    scene.element.cnum,
                    scene.element.grid_size,
                    False,
                    scene.element.cell.type,
                    scene.element.cell.fluid_sdf,
                )
            self.fluid_sdf_build_step = current_step
        self.update_external_fluid_level_set(scene)
        mat_props = self.get_single_fluid_mat_props(scene)
        if getattr(mat_props, "surface_tension", 0.0) > 0.0:
            kernel_compute_fluid_surface_tension(
                scene.element.ghost_cell,
                scene.element.cnum,
                scene.element.grid_size,
                mat_props,
                scene.element.cell.type,
                scene.element.cell.fluid_sdf,
                scene.element.cell.surface_tension,
            )

    def apply_solid_cell_boundaries(self, sims: Simulation, scene: myScene):
        if scene.boundary is None:
            kernel_reset_solid_sdf(scene.element.grid_size, scene.element.cell.solid_sdf)
            return
        kernel_reset_solid_sdf(scene.element.grid_size, scene.element.cell.solid_sdf)
        for start_point, end_point in scene.boundary.solid_cell_regions:
            mark_solid_cell_region(
                scene.element.ghost_cell,
                scene.element.cnum,
                scene.element.grid_size,
                ti.Vector(start_point.tolist()),
                ti.Vector(end_point.tolist()),
                scene.element.cell.type,
            )
            kernel_update_solid_sdf_from_box(
                scene.element.grid_size,
                ti.Vector(start_point.tolist()),
                ti.Vector(end_point.tolist()),
                scene.element.cell.solid_sdf,
            )
        kernel_close_solid_cell_boundary_corners(scene.element.ghost_cell, scene.element.cnum, scene.element.cell.type)
        kernel_finalize_solid_sdf_from_cell_type(
            scene.element.grid_size, scene.element.cell.type, scene.element.cell.solid_sdf
        )

    def ensure_solid_cut_cell_fields(self, sims: Simulation, scene: myScene):
        if self.ensure_solid_cut_cell_fields_step is None:
            self.configure_runtime_functions(sims, scene)
        self.ensure_solid_cut_cell_fields_step(scene)

    def ensure_solid_cut_cell_fields2d(self, scene: myScene):
        if self.solid_face_open_fraction is None:
            self.solid_face_open_fraction = [
                ti.field(
                    dtype=float,
                    shape=scene.node.velocity[0].shape,
                    offset=[-scene.element.ghost_cell, -scene.element.ghost_cell],
                ),
                ti.field(
                    dtype=float,
                    shape=scene.node.velocity[1].shape,
                    offset=[-scene.element.ghost_cell, -scene.element.ghost_cell],
                ),
            ]
        if self.solid_face_velocity is None:
            self.solid_face_velocity = [
                ti.field(
                    dtype=float,
                    shape=scene.node.velocity[0].shape,
                    offset=[-scene.element.ghost_cell, -scene.element.ghost_cell],
                ),
                ti.field(
                    dtype=float,
                    shape=scene.node.velocity[1].shape,
                    offset=[-scene.element.ghost_cell, -scene.element.ghost_cell],
                ),
            ]

    def ensure_solid_cut_cell_fields3d(self, scene: myScene):
        if self.solid_face_open_fraction is None:
            self.solid_face_open_fraction = [
                ti.field(
                    dtype=float,
                    shape=scene.node.velocity[0].shape,
                    offset=[-scene.element.ghost_cell, -scene.element.ghost_cell, -scene.element.ghost_cell],
                ),
                ti.field(
                    dtype=float,
                    shape=scene.node.velocity[1].shape,
                    offset=[-scene.element.ghost_cell, -scene.element.ghost_cell, -scene.element.ghost_cell],
                ),
                ti.field(
                    dtype=float,
                    shape=scene.node.velocity[2].shape,
                    offset=[-scene.element.ghost_cell, -scene.element.ghost_cell, -scene.element.ghost_cell],
                ),
            ]
        if self.solid_face_velocity is None:
            self.solid_face_velocity = [
                ti.field(
                    dtype=float,
                    shape=scene.node.velocity[0].shape,
                    offset=[-scene.element.ghost_cell, -scene.element.ghost_cell, -scene.element.ghost_cell],
                ),
                ti.field(
                    dtype=float,
                    shape=scene.node.velocity[1].shape,
                    offset=[-scene.element.ghost_cell, -scene.element.ghost_cell, -scene.element.ghost_cell],
                ),
                ti.field(
                    dtype=float,
                    shape=scene.node.velocity[2].shape,
                    offset=[-scene.element.ghost_cell, -scene.element.ghost_cell, -scene.element.ghost_cell],
                ),
            ]

    def solid_face_fraction_y(self):
        return self.solid_face_open_fraction[1]

    def solid_face_fraction_z(self):
        return self.solid_face_open_fraction[2]

    def solid_face_velocity_y(self):
        return self.solid_face_velocity[1]

    def solid_face_velocity_z(self):
        return self.solid_face_velocity[2]

    def reset_solid_face_velocity(self, sims: Simulation, scene: myScene):
        if not sims.solid_sdf_cut_cell:
            return
        self.reset_solid_face_velocity_step(sims, scene)

    def reset_solid_face_velocity_cut_cell2d(self, sims: Simulation, scene: myScene):
        self.ensure_solid_cut_cell_fields2d(scene)
        self.solid_face_velocity[0].fill(0.0)
        self.solid_face_velocity[1].fill(0.0)

    def reset_solid_face_velocity_cut_cell3d(self, sims: Simulation, scene: myScene):
        self.ensure_solid_cut_cell_fields3d(scene)
        self.solid_face_velocity[0].fill(0.0)
        self.solid_face_velocity[1].fill(0.0)
        self.solid_face_velocity[2].fill(0.0)

    def update_solid_cut_cell(self, sims: Simulation, scene: myScene):
        if not sims.solid_sdf_cut_cell:
            return
        self.update_solid_cut_cell_step(sims, scene)

    def update_solid_cut_cell2d(self, sims: Simulation, scene: myScene):
        self.ensure_solid_cut_cell_fields2d(scene)
        kernel_build_solid_face_open_fraction_direction(
            0,
            scene.element.ghost_cell,
            scene.element.cnum,
            sims.solid_cut_cell_min_fraction,
            scene.element.cell.type,
            scene.element.cell.solid_sdf,
            self.solid_face_open_fraction[0],
        )
        kernel_build_solid_face_open_fraction_direction(
            1,
            scene.element.ghost_cell,
            scene.element.cnum,
            sims.solid_cut_cell_min_fraction,
            scene.element.cell.type,
            scene.element.cell.solid_sdf,
            self.solid_face_open_fraction[1],
        )

    def update_solid_cut_cell3d(self, sims: Simulation, scene: myScene):
        self.ensure_solid_cut_cell_fields3d(scene)
        kernel_build_solid_face_open_fraction_direction(
            0,
            scene.element.ghost_cell,
            scene.element.cnum,
            sims.solid_cut_cell_min_fraction,
            scene.element.cell.type,
            scene.element.cell.solid_sdf,
            self.solid_face_open_fraction[0],
        )
        kernel_build_solid_face_open_fraction_direction(
            1,
            scene.element.ghost_cell,
            scene.element.cnum,
            sims.solid_cut_cell_min_fraction,
            scene.element.cell.type,
            scene.element.cell.solid_sdf,
            self.solid_face_open_fraction[1],
        )
        kernel_build_solid_face_open_fraction_direction(
            2,
            scene.element.ghost_cell,
            scene.element.cnum,
            sims.solid_cut_cell_min_fraction,
            scene.element.cell.type,
            scene.element.cell.solid_sdf,
            self.solid_face_open_fraction[2],
        )

    def remove_particles_inside_solid_cells(self, scene: myScene):
        deactivated = kernel_deactivate_particles_in_solid_cells(
            int(scene.particleNum[0]),
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.igrid_size,
            scene.element.cell.type,
            scene.particle,
        )
        if scene.element.cell.solid_sdf is not None:
            deactivated += kernel_deactivate_particles_in_solid_sdf(
                int(scene.particleNum[0]),
                scene.element.ghost_cell,
                scene.element.cnum,
                scene.element.grid_size,
                scene.element.cell.solid_sdf,
                scene.particle,
            )
        if deactivated > 0:
            scene.particleNum[0] = kernel_update_incompressible_particle_storage(
                int(scene.particleNum[0]), scene.particle
            )

    def initialize_cell_pressure_from_particles(self, scene: myScene):
        if self.initial_pressure_weight is None:
            self.initial_pressure_weight = ti.field(
                dtype=float, shape=scene.element.cnum, offset=0 * scene.element.cnum - scene.element.ghost_cell
            )
        kernel_initialize_fdm_cell_pressure_from_particles(
            int(scene.particleNum[0]),
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.igrid_size,
            scene.element.cell.type,
            scene.element.cell.pressure,
            self.initial_pressure_weight,
            scene.particle,
        )

    def enforce_particle_domain_boundary(self, scene: myScene):
        enforce_particle_domain_collision(
            int(scene.particleNum[0]),
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.grid_size,
            scene.element.cell.type,
            scene.particle,
        )

    def enforce_particle_domain_and_sdf_boundary(self, scene: myScene):
        self.enforce_particle_domain_boundary(scene)
        enforce_particle_solid_sdf_collision(
            int(scene.particleNum[0]),
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.grid_size,
            scene.element.cell.solid_sdf,
            self.solid_face_velocity[0],
            self.solid_face_velocity[1],
            self.solid_face_velocity2(),
            scene.particle,
        )

    def particle_shifting(self, sims: Simulation, scene: myScene):
        if not sims.particle_shifting:
            return
        if self.shifting_node_volume is None:
            self.shifting_node_volume = ti.field(dtype=float, shape=(scene.element.gridSum, scene.element.grid_level))
            self.shifting_reference_volume = ti.field(dtype=float, shape=self.shifting_node_volume.shape)
        if not self.shifting_reference_ready:
            kernel_fdm_shifting_reference_volume(
                scene.element.ghost_cell,
                scene.element.gnum,
                scene.element.grid_size,
                scene.element.calLength,
                scene.element.boundary_type,
                self.shifting_reference_volume,
            )
            self.shifting_reference_ready = True
        if scene.element.LnID is not None:
            scene.element.calculate(scene.particleNum, scene.particle)
            kernel_volume_p2g_fdm_mac_shifting(
                scene.element.grid_nodes,
                int(scene.particleNum[0]),
                self.shifting_node_volume,
                scene.particle,
                scene.element.LnID,
                scene.element.shape_fn,
                scene.element.node_size,
            )
            reference_volume = self.shifting_reference_volume
            E2 = kernel_compute_particle_shifting_energy(self.shifting_node_volume, reference_volume)
            den = kernel_compute_particle_shifting_gradient_fdm_mac(
                scene.element.grid_nodes,
                int(scene.particleNum[0]),
                scene.element.ghost_cell,
                scene.element.cnum,
                scene.element.grid_size,
                reference_volume,
                self.shifting_node_volume,
                scene.element.cell.type,
                scene.particle,
                scene.element.LnID,
                scene.element.dshape_fn,
                scene.element.node_size,
            )
            if den > 0.0:
                kernel_apply_particle_shifting_gradient(
                    int(scene.particleNum[0]),
                    scene.element.ghost_cell,
                    scene.element.cnum,
                    scene.element.grid_size,
                    0.05,
                    E2 / den,
                    scene.element.cell.type,
                    scene.particle,
                )
        elif sims.dimension == 2:
            kernel_volume_p2g_fdm_shifting_on_the_fly_2d(
                scene.element.influenced_node,
                int(scene.particleNum[0]),
                scene.element.ghost_cell,
                scene.element.grid_size,
                scene.element.igrid_size,
                scene.element.gnum,
                self.shifting_node_volume,
                scene.particle,
                scene.element.calLength,
                scene.element.boundary_type,
            )
            kernel_particle_shifting_delta_correction_fdm_on_the_fly_2d(
                scene.element.influenced_node,
                int(scene.particleNum[0]),
                scene.element.ghost_cell,
                scene.element.cnum,
                scene.element.grid_size,
                scene.element.igrid_size,
                scene.element.gnum,
                0.05,
                self.shifting_node_volume,
                self.shifting_reference_volume,
                scene.element.cell.type,
                scene.particle,
                scene.element.calLength,
                scene.element.boundary_type,
            )
        else:
            kernel_volume_p2g_fdm_shifting_on_the_fly_3d(
                scene.element.influenced_node,
                int(scene.particleNum[0]),
                scene.element.ghost_cell,
                scene.element.grid_size,
                scene.element.igrid_size,
                scene.element.gnum,
                self.shifting_node_volume,
                scene.particle,
                scene.element.calLength,
                scene.element.boundary_type,
            )
            reference_volume = self.shifting_reference_volume
            E2 = kernel_compute_particle_shifting_energy(self.shifting_node_volume, reference_volume)
            den = kernel_compute_particle_shifting_gradient_fdm_on_the_fly_3d(
                scene.element.influenced_node,
                int(scene.particleNum[0]),
                scene.element.ghost_cell,
                scene.element.cnum,
                scene.element.grid_size,
                scene.element.igrid_size,
                scene.element.gnum,
                reference_volume,
                self.shifting_node_volume,
                scene.element.cell.type,
                scene.particle,
                scene.element.calLength,
            )
            if den > 0.0:
                kernel_apply_particle_shifting_gradient(
                    int(scene.particleNum[0]),
                    scene.element.ghost_cell,
                    scene.element.cnum,
                    scene.element.grid_size,
                    0.05,
                    E2 / den,
                    scene.element.cell.type,
                    scene.particle,
                )

    def ensure_density_projection_fields(self, scene: myScene):
        if self.density_projection_fields_ready:
            return
        if self.density_projection_cell_density is None:
            self.density_projection_cell_density = ti.field(
                dtype=float, shape=scene.element.cnum, offset=0 * scene.element.cnum - scene.element.ghost_cell
            )
        if self.density_projection_cell_fraction is None:
            self.density_projection_cell_fraction = ti.field(
                dtype=float, shape=scene.element.cnum, offset=0 * scene.element.cnum - scene.element.ghost_cell
            )
        if self.density_projection_pressure is None:
            self.density_projection_pressure = ti.field(
                dtype=float, shape=scene.element.cnum, offset=0 * scene.element.cnum - scene.element.ghost_cell
            )
        if self.density_projection_face_displacement is None:
            offset = [-scene.element.ghost_cell for _ in range(self.dimension)]
            self.density_projection_face_displacement = [
                ti.field(dtype=float, shape=scene.node.velocity[d].shape, offset=offset) for d in range(self.dimension)
            ]
        if self.density_projection_needs_solve is None:
            self.density_projection_needs_solve = ti.field(dtype=int, shape=())
        self.density_projection_fields_ready = True

    def compute_density_projection_cell_density(self, scene: myScene):
        mat_props = self.get_single_fluid_mat_props(scene)
        kernel_compute_fdm_cell_density_from_volume_fraction(
            scene.element.ghost_cell,
            scene.element.cnum,
            mat_props.density,
            scene.element.cell.type,
            scene.element.cell_volumefrac,
            self.density_projection_cell_density,
        )

    def compute_density_projection_fluid_fraction(self, scene: myScene):
        kernel_compute_density_projection_fluid_fraction(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.grid_size,
            scene.element.cell.type,
            scene.element.cell.fluid_sdf,
            self.density_projection_cell_fraction,
        )

    def find_density_projection_active_nodes(self, sims: Simulation, scene: myScene):
        find_active_density_projection_cell(
            scene.element.ghost_cell,
            scene.element.cellSum,
            scene.element.cnum,
            scene.element.cell.type,
            sims.density_projection_interior_only,
            scene.element.flag,
        )
        scene.element.pse.run(scene.element.flag)
        return set_active_density_projection_cell_dofs(
            scene.element.ghost_cell,
            scene.element.cellSum,
            scene.element.cnum,
            scene.element.cell.type,
            sims.density_projection_interior_only,
            scene.element.flag,
        )

    def prepare_density_projection_equations_regular(self, sims: Simulation, scene: myScene):
        self.unknow_vector.fill(0)
        mat_props = self.get_single_fluid_mat_props(scene)
        use_fluid_fraction = not sims.density_projection_interior_only
        kernel_assemble_density_projection_rhs(
            scene.element.ghost_cell,
            scene.element.cnum,
            sims.dt,
            scene.element.flag,
            scene.element.cell.type,
            self.density_projection_cell_density,
            self.density_projection_cell_fraction,
            mat_props,
            sims.density_projection_tolerance,
            sims.density_projection_error_clamp,
            sims.density_projection_interior_only,
            use_fluid_fraction,
            self.right_hand_vector,
        )
        kernel_preconditioning_poisson_equation_matrix(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.grid_size,
            scene.element.igrid_size,
            scene.element.flag,
            scene.element.cell.type,
            scene.element.cell.fluid_sdf,
            use_fluid_fraction,
            self.diag_A,
        )

    def prepare_density_projection_equations_cut_cell(self, sims: Simulation, scene: myScene):
        self.unknow_vector.fill(0)
        mat_props = self.get_single_fluid_mat_props(scene)
        use_fluid_fraction = not sims.density_projection_interior_only
        kernel_assemble_density_projection_rhs(
            scene.element.ghost_cell,
            scene.element.cnum,
            sims.dt,
            scene.element.flag,
            scene.element.cell.type,
            self.density_projection_cell_density,
            self.density_projection_cell_fraction,
            mat_props,
            sims.density_projection_tolerance,
            sims.density_projection_error_clamp,
            sims.density_projection_interior_only,
            use_fluid_fraction,
            self.right_hand_vector,
        )
        kernel_preconditioning_poisson_equation_matrix_cut_cell(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.grid_size,
            scene.element.igrid_size,
            scene.element.flag,
            scene.element.cell.type,
            scene.element.cell.fluid_sdf,
            scene.element.cell.solid_sdf,
            self.solid_face_open_fraction[0],
            self.solid_face_open_fraction[1],
            self.solid_face_fraction2(),
            use_fluid_fraction,
            self.diag_A,
        )

    def prepare_density_projection_equations_mgpcg_regular(self, sims: Simulation, scene: myScene):
        self.poisson_solver.initialize()
        mat_props = self.get_single_fluid_mat_props(scene)
        use_fluid_fraction = not sims.density_projection_interior_only
        kernel_copy_density_projection_mg_cell_type(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.cell.type,
            sims.density_projection_interior_only,
            self.poisson_solver.grid_type[0],
        )
        kernel_assemble_density_projection_mg_b(
            scene.element.ghost_cell,
            scene.element.cnum,
            sims.dt,
            self.poisson_solver.grid_type[0],
            scene.element.cell.type,
            self.density_projection_cell_density,
            self.density_projection_cell_fraction,
            mat_props,
            sims.density_projection_tolerance,
            sims.density_projection_error_clamp,
            sims.density_projection_interior_only,
            use_fluid_fraction,
            self.poisson_solver.b,
        )
        kernel_assemble_incompressible_mg_A_level0(
            scene.element.ghost_cell,
            scene.element.cnum,
            sims.dt,
            scene.element.grid_size,
            scene.element.igrid_size,
            mat_props,
            self.poisson_solver.grid_type[0],
            scene.element.cell.type,
            scene.element.cell.fluid_sdf,
            not sims.density_projection_interior_only,
            self.poisson_solver.Adiag[0],
            self.poisson_solver.Ax[0],
        )
        for level in range(1, self.poisson_solver.n_mg_levels):
            self.poisson_solver.init_gridtype(
                self.poisson_solver.grid_type[level - 1], self.poisson_solver.grid_type[level]
            )
            kernel_assemble_incompressible_mg_A(
                sims.dt,
                scene.element.grid_size,
                scene.element.igrid_size,
                mat_props,
                self.poisson_solver.grid_type[level],
                self.poisson_solver.Adiag[level],
                self.poisson_solver.Ax[level],
            )

    def prepare_density_projection_equations_mgpcg_cut_cell(self, sims: Simulation, scene: myScene):
        self.poisson_solver.initialize()
        mat_props = self.get_single_fluid_mat_props(scene)
        use_fluid_fraction = not sims.density_projection_interior_only
        kernel_copy_density_projection_mg_cell_type(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.cell.type,
            sims.density_projection_interior_only,
            self.poisson_solver.grid_type[0],
        )
        kernel_assemble_density_projection_mg_b(
            scene.element.ghost_cell,
            scene.element.cnum,
            sims.dt,
            self.poisson_solver.grid_type[0],
            scene.element.cell.type,
            self.density_projection_cell_density,
            self.density_projection_cell_fraction,
            mat_props,
            sims.density_projection_tolerance,
            sims.density_projection_error_clamp,
            sims.density_projection_interior_only,
            use_fluid_fraction,
            self.poisson_solver.b,
        )
        kernel_assemble_incompressible_mg_A_level0_cut_cell(
            scene.element.ghost_cell,
            scene.element.cnum,
            sims.dt,
            scene.element.grid_size,
            scene.element.igrid_size,
            mat_props,
            self.poisson_solver.grid_type[0],
            scene.element.cell.type,
            scene.element.cell.fluid_sdf,
            scene.element.cell.solid_sdf,
            not sims.density_projection_interior_only,
            self.solid_face_open_fraction[0],
            self.solid_face_open_fraction[1],
            self.solid_face_fraction2(),
            self.poisson_solver.Adiag[0],
            self.poisson_solver.Ax[0],
        )
        for level in range(1, self.poisson_solver.n_mg_levels):
            self.poisson_solver.init_gridtype(
                self.poisson_solver.grid_type[level - 1], self.poisson_solver.grid_type[level]
            )
            kernel_assemble_incompressible_mg_A(
                sims.dt,
                scene.element.grid_size,
                scene.element.igrid_size,
                mat_props,
                self.poisson_solver.grid_type[level],
                self.poisson_solver.Adiag[level],
                self.poisson_solver.Ax[level],
            )

    def apply_density_projection_position_correction(self, sims: Simulation, scene: myScene):
        mat_props = self.get_single_fluid_mat_props(scene)
        face2 = (
            self.density_projection_face_displacement[2]
            if self.dimension == 3
            else self.density_projection_face_displacement[1]
        )
        kernel_compute_density_projection_face_displacement(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.grid_size,
            sims.dt,
            mat_props,
            self.density_projection_pressure,
            scene.element.cell.type,
            scene.element.cell.fluid_sdf,
            self.needs_density_projection_fluid_level_set(sims, scene),
            self.density_projection_face_displacement[0],
            self.density_projection_face_displacement[1],
            face2,
        )
        kernel_apply_density_projection_position_correction_cached(
            int(scene.particleNum[0]),
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.grid_size,
            scene.element.igrid_size,
            mat_props,
            sims.density_projection_max_shift_ratio,
            scene.element.cell.type,
            sims.density_projection_interior_only,
            scene.particle,
            scene.element.calLength,
            self.density_projection_face_displacement[0],
            self.density_projection_face_displacement[1],
            face2,
        )

    def density_projection(self, sims: Simulation, scene: myScene):
        if not sims.density_projection:
            return
        if sims.linear_solver == "PCG":
            self.density_projection_pcg(sims, scene)
        elif sims.linear_solver == "MGPCG":
            self.density_projection_mgpcg(sims, scene)

    def prepare_density_projection_state(self, sims: Simulation, scene: myScene):
        self.ensure_density_projection_fields(scene)
        self.identify_fluid_domain(sims, scene)
        self.run_density_projection_level_set(sims, scene)
        self.compute_density_projection_cell_density(scene)
        self.compute_density_projection_fluid_fraction_step(scene)

    def density_projection_is_required(self, sims: Simulation, scene: myScene):
        mat_props = self.get_single_fluid_mat_props(scene)
        kernel_density_projection_needs_solve(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.cell.type,
            self.density_projection_cell_density,
            self.density_projection_cell_fraction,
            mat_props.density,
            sims.density_projection_tolerance,
            sims.density_projection_interior_only,
            not sims.density_projection_interior_only,
            self.density_projection_needs_solve,
        )
        return bool(self.density_projection_needs_solve[None])

    def density_projection_pcg(self, sims: Simulation, scene: myScene):
        self.prepare_density_projection_state(sims, scene)
        if not self.density_projection_is_required(sims, scene):
            return
        total_dofs = self.find_density_projection_active_nodes(sims, scene)
        self.prepare_density_projection_equations_step(sims, scene)
        self.operator.use_free_surface_theta = self.needs_density_projection_fluid_level_set(sims, scene)
        solved = self.poisson_solver.solve(
            self.operator,
            self.right_hand_vector,
            self.unknow_vector,
            self.diag_A,
            total_dofs,
            maxiter=total_dofs,
            tol=sims.residual_tolerance,
        )
        self.require_pressure_solver_success(solved, "density-projection PCG")
        kernel_update_cell_pressure(
            scene.element.ghost_cell,
            scene.element.cnum,
            self.density_projection_pressure,
            scene.element.flag,
            scene.element.cell.type,
            self.unknow_vector,
        )
        self.apply_density_projection_position_correction(sims, scene)

    def density_projection_mgpcg(self, sims: Simulation, scene: myScene):
        self.prepare_density_projection_state(sims, scene)
        if not self.density_projection_is_required(sims, scene):
            return
        self.prepare_density_projection_equations_mgpcg_step(sims, scene)
        solved = self.poisson_solver.solve(max_iters=max(1, sims.iter_max), rel_tol=sims.residual_tolerance)
        self.require_pressure_solver_success(solved, "density-projection MGPCG")
        kernel_update_cell_pressure_from_mg(
            self.density_projection_pressure, self.poisson_solver.grid_type[0], self.poisson_solver.x
        )
        self.apply_density_projection_position_correction(sims, scene)

    def get_single_fluid_mat_props(self, scene: myScene):
        material_count = scene.material.mapping.shape[0] - 1
        if material_count != 1:
            raise RuntimeError(f"Incompressible MGPCG currently expects one fluid material, got {material_count}")
        return scene.material.matProps[1]

    def require_pressure_solver_success(self, solved, stage):
        if solved:
            return
        solver = self.poisson_solver
        reason = getattr(solver, "last_breakdown_reason", getattr(solver, "breakdown_reason", "unknown"))
        initial = getattr(solver, "last_initial_residual", getattr(solver, "initial_residual", float("nan")))
        final = getattr(solver, "last_residual", getattr(solver, "final_residual", float("nan")))
        iterations = getattr(solver, "last_iterations", -1)
        raise RuntimeError(
            f"Incompressible {stage} failed to converge after {iterations} iterations "
            f"(initial residual={initial:.6e}, final residual={final:.6e}, reason={reason})"
        )

    def create_matrix_free_poisson_solver2d(self, sims: Simulation, scene: myScene):
        total_dofs = estimate_active_fdm_cell_dofs(
            scene.element.ghost_cell, scene.element.cnum, scene.element.cell.type
        )
        allocated_dofs = max(1, int(sims.dof_multiplier * max(1, total_dofs)))

        from src.linear_solver.MatrixFreePCG import MatrixFreePCG
        from src.mpm.engines.Operator import PoissonEquationOperator

        self.poisson_solver = MatrixFreePCG(allocated_dofs)
        self.operator = PoissonEquationOperator(2)
        self.unknow_vector = ti.field(dtype=float)
        self.right_hand_vector = ti.field(dtype=float)
        self.diag_A = ti.field(dtype=float)
        ti.root.dense(ti.i, allocated_dofs).place(self.unknow_vector, self.right_hand_vector, self.diag_A)
        self.operator.link_ptrs(
            scene,
            self.solid_face_open_fraction if sims.solid_sdf_cut_cell else None,
            self.use_pressure_free_surface_theta,
            self.pressure_coupling_mode,
            self.pressure_solid_fraction,
            self.pressure_previous_solid_fraction,
            self.pressure_solid_density,
            self.get_single_fluid_mat_props(scene),
        )
        self.operator.update_active_dofs(total_dofs)

    def create_matrix_free_poisson_solver3d(self, sims: Simulation, scene: myScene):
        total_dofs = estimate_active_fdm_cell_dofs(
            scene.element.ghost_cell, scene.element.cnum, scene.element.cell.type
        )
        allocated_dofs = max(1, int(sims.dof_multiplier * max(1, total_dofs)))

        from src.linear_solver.MatrixFreePCG import MatrixFreePCG
        from src.mpm.engines.Operator import PoissonEquationOperator

        self.poisson_solver = MatrixFreePCG(allocated_dofs)
        self.operator = PoissonEquationOperator(3)
        self.unknow_vector = ti.field(dtype=float)
        self.right_hand_vector = ti.field(dtype=float)
        self.diag_A = ti.field(dtype=float)
        ti.root.dense(ti.i, allocated_dofs).place(self.unknow_vector, self.right_hand_vector, self.diag_A)
        self.operator.link_ptrs(
            scene,
            self.solid_face_open_fraction if sims.solid_sdf_cut_cell else None,
            self.use_pressure_free_surface_theta,
            self.pressure_coupling_mode,
            self.pressure_solid_fraction,
            self.pressure_previous_solid_fraction,
            self.pressure_solid_density,
            self.get_single_fluid_mat_props(scene),
        )
        self.operator.update_active_dofs(total_dofs)

    def create_mgpcg_poisson_solver2d(self, sims: Simulation, scene: myScene):
        from src.linear_solver.MultiGridPCG import MGPCGPoissonSolver

        interior_cnum = [
            max(1, int(scene.element.cnum[0]) - 2 * scene.element.ghost_cell),
            max(1, int(scene.element.cnum[1]) - 2 * scene.element.ghost_cell),
        ]
        self.poisson_solver = MGPCGPoissonSolver(
            2, interior_cnum, sims.multilevel, sims.pre_and_post_smoothing, sims.bottom_smoothing
        )

    def create_mgpcg_poisson_solver3d(self, sims: Simulation, scene: myScene):
        from src.linear_solver.MultiGridPCG import MGPCGPoissonSolver

        interior_cnum = [
            max(1, int(scene.element.cnum[0]) - 2 * scene.element.ghost_cell),
            max(1, int(scene.element.cnum[1]) - 2 * scene.element.ghost_cell),
            max(1, int(scene.element.cnum[2]) - 2 * scene.element.ghost_cell),
        ]
        self.poisson_solver = MGPCGPoissonSolver(
            3, interior_cnum, sims.multilevel, sims.pre_and_post_smoothing, sims.bottom_smoothing
        )

    def prepare_poisson_equations_regular(self, sims: Simulation, scene: myScene):
        self.unknow_vector.fill(0)
        if self.operator is not None:
            self.operator.use_free_surface_theta = self.use_pressure_free_surface_theta
        mat_props = self.get_single_fluid_mat_props(scene)
        kernel_assemble_poisson_equation_dynamic(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.gnum,
            scene.element.grid_size,
            scene.element.igrid_size,
            sims.dt,
            scene.node,
            scene.element.flag,
            scene.element.cell.surface_tension,
            scene.element.cell.type,
            scene.element.cell.fluid_sdf,
            mat_props,
            self.use_pressure_free_surface_theta,
            self.right_hand_vector,
        )
        kernel_preconditioning_poisson_equation_matrix(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.grid_size,
            scene.element.igrid_size,
            scene.element.flag,
            scene.element.cell.type,
            scene.element.cell.fluid_sdf,
            self.use_pressure_free_surface_theta,
            self.diag_A,
        )

    def prepare_poisson_equations_cut_cell(self, sims: Simulation, scene: myScene):
        self.unknow_vector.fill(0)
        if self.operator is not None:
            self.operator.use_free_surface_theta = self.use_pressure_free_surface_theta
        mat_props = self.get_single_fluid_mat_props(scene)
        face_fraction2 = self.solid_face_fraction2()
        solid_velocity2 = self.solid_face_velocity2()
        kernel_assemble_poisson_equation_dynamic_cut_cell(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.gnum,
            scene.element.grid_size,
            scene.element.igrid_size,
            sims.dt,
            scene.node,
            scene.element.flag,
            scene.element.cell.surface_tension,
            scene.element.cell.type,
            scene.element.cell.fluid_sdf,
            scene.element.cell.solid_sdf,
            self.solid_face_open_fraction[0],
            self.solid_face_open_fraction[1],
            face_fraction2,
            self.solid_face_velocity[0],
            self.solid_face_velocity[1],
            solid_velocity2,
            mat_props,
            self.use_pressure_free_surface_theta,
            self.right_hand_vector,
        )
        kernel_preconditioning_poisson_equation_matrix_cut_cell(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.grid_size,
            scene.element.igrid_size,
            scene.element.flag,
            scene.element.cell.type,
            scene.element.cell.fluid_sdf,
            scene.element.cell.solid_sdf,
            self.solid_face_open_fraction[0],
            self.solid_face_open_fraction[1],
            face_fraction2,
            self.use_pressure_free_surface_theta,
            self.diag_A,
        )

    def prepare_poisson_equations_coupled(self, sims: Simulation, scene: myScene):
        self.unknow_vector.fill(0)
        if self.operator is not None:
            self.operator.use_free_surface_theta = self.use_pressure_free_surface_theta
        mat_props = self.get_single_fluid_mat_props(scene)
        if sims.solid_sdf_cut_cell:
            face_fraction = self.solid_face_open_fraction
            solid_velocity = self.solid_face_velocity
        else:
            face_fraction = [self.pressure_solid_fraction] * 3
            solid_velocity = [self.pressure_solid_fraction] * 3
        kernel_assemble_poisson_equation_coupled_3d(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.igrid_size,
            sims.dt,
            scene.node,
            scene.element.flag,
            scene.element.cell.surface_tension,
            scene.element.cell.type,
            scene.element.cell.fluid_sdf,
            face_fraction[0],
            face_fraction[1],
            face_fraction[2],
            solid_velocity[0],
            solid_velocity[1],
            solid_velocity[2],
            self.pressure_solid_fraction,
            self.pressure_previous_solid_fraction,
            self.pressure_solid_density,
            mat_props,
            self.pressure_coupling_mode,
            sims.solid_sdf_cut_cell,
            self.use_pressure_free_surface_theta,
            self.right_hand_vector,
        )
        kernel_preconditioning_poisson_equation_coupled_3d(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.igrid_size,
            scene.element.flag,
            scene.element.cell.type,
            scene.element.cell.fluid_sdf,
            face_fraction[0],
            face_fraction[1],
            face_fraction[2],
            self.pressure_solid_fraction,
            self.pressure_solid_density,
            mat_props,
            self.pressure_coupling_mode,
            sims.solid_sdf_cut_cell,
            self.use_pressure_free_surface_theta,
            self.diag_A,
        )

    def prepare_poisson_equations_mgpcg_regular(self, sims: Simulation, scene: myScene):
        self.poisson_solver.initialize()
        mat_props = self.get_single_fluid_mat_props(scene)
        kernel_prepare_incompressible_mg_level0(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.igrid_size,
            sims.dt,
            scene.node,
            scene.element.cell.surface_tension,
            scene.element.cell.type,
            scene.element.cell.fluid_sdf,
            mat_props,
            self.use_pressure_free_surface_theta,
            self.poisson_solver.grid_type[0],
            self.poisson_solver.b,
        )
        kernel_assemble_incompressible_mg_A_level0(
            scene.element.ghost_cell,
            scene.element.cnum,
            sims.dt,
            scene.element.grid_size,
            scene.element.igrid_size,
            mat_props,
            self.poisson_solver.grid_type[0],
            scene.element.cell.type,
            scene.element.cell.fluid_sdf,
            self.use_pressure_free_surface_theta,
            self.poisson_solver.Adiag[0],
            self.poisson_solver.Ax[0],
        )
        for level in range(1, self.poisson_solver.n_mg_levels):
            self.poisson_solver.init_gridtype(
                self.poisson_solver.grid_type[level - 1], self.poisson_solver.grid_type[level]
            )
            kernel_assemble_incompressible_mg_A(
                sims.dt,
                scene.element.grid_size,
                scene.element.igrid_size,
                mat_props,
                self.poisson_solver.grid_type[level],
                self.poisson_solver.Adiag[level],
                self.poisson_solver.Ax[level],
            )

    def prepare_poisson_equations_mgpcg_cut_cell(self, sims: Simulation, scene: myScene):
        self.poisson_solver.initialize()
        mat_props = self.get_single_fluid_mat_props(scene)
        face_fraction2 = self.solid_face_fraction2()
        solid_velocity2 = self.solid_face_velocity2()
        kernel_prepare_incompressible_mg_level0_cut_cell(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.igrid_size,
            sims.dt,
            scene.node,
            scene.element.cell.surface_tension,
            scene.element.cell.type,
            scene.element.cell.fluid_sdf,
            scene.element.cell.solid_sdf,
            self.solid_face_open_fraction[0],
            self.solid_face_open_fraction[1],
            face_fraction2,
            self.solid_face_velocity[0],
            self.solid_face_velocity[1],
            solid_velocity2,
            mat_props,
            self.use_pressure_free_surface_theta,
            self.poisson_solver.grid_type[0],
            self.poisson_solver.b,
        )
        kernel_assemble_incompressible_mg_A_level0_cut_cell(
            scene.element.ghost_cell,
            scene.element.cnum,
            sims.dt,
            scene.element.grid_size,
            scene.element.igrid_size,
            mat_props,
            self.poisson_solver.grid_type[0],
            scene.element.cell.type,
            scene.element.cell.fluid_sdf,
            scene.element.cell.solid_sdf,
            self.use_pressure_free_surface_theta,
            self.solid_face_open_fraction[0],
            self.solid_face_open_fraction[1],
            face_fraction2,
            self.poisson_solver.Adiag[0],
            self.poisson_solver.Ax[0],
        )
        for level in range(1, self.poisson_solver.n_mg_levels):
            self.poisson_solver.init_gridtype(
                self.poisson_solver.grid_type[level - 1], self.poisson_solver.grid_type[level]
            )
            kernel_assemble_incompressible_mg_A(
                sims.dt,
                scene.element.grid_size,
                scene.element.igrid_size,
                mat_props,
                self.poisson_solver.grid_type[level],
                self.poisson_solver.Adiag[level],
                self.poisson_solver.Ax[level],
            )

    def prepare_poisson_equations_mgpcg_coupled(self, sims: Simulation, scene: myScene):
        self.poisson_solver.initialize()
        mat_props = self.get_single_fluid_mat_props(scene)
        if sims.solid_sdf_cut_cell:
            face_fraction = self.solid_face_open_fraction
            solid_velocity = self.solid_face_velocity
        else:
            face_fraction = [self.pressure_solid_fraction] * 3
            solid_velocity = [self.pressure_solid_fraction] * 3
        kernel_prepare_incompressible_mg_level0_coupled_3d(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.igrid_size,
            sims.dt,
            scene.node,
            scene.element.cell.surface_tension,
            scene.element.cell.type,
            scene.element.cell.fluid_sdf,
            face_fraction[0],
            face_fraction[1],
            face_fraction[2],
            solid_velocity[0],
            solid_velocity[1],
            solid_velocity[2],
            self.pressure_solid_fraction,
            self.pressure_previous_solid_fraction,
            self.pressure_solid_density,
            mat_props,
            self.pressure_coupling_mode,
            sims.solid_sdf_cut_cell,
            self.use_pressure_free_surface_theta,
            self.poisson_solver.grid_type[0],
            self.poisson_solver.b,
        )
        kernel_assemble_incompressible_mg_A_level0_coupled_3d(
            sims.dt,
            scene.element.igrid_size,
            mat_props,
            self.poisson_solver.grid_type[0],
            scene.element.cell.type,
            scene.element.cell.fluid_sdf,
            face_fraction[0],
            face_fraction[1],
            face_fraction[2],
            self.pressure_solid_fraction,
            self.pressure_solid_density,
            self.pressure_coupling_mode,
            sims.solid_sdf_cut_cell,
            self.use_pressure_free_surface_theta,
            self.poisson_solver.Adiag[0],
            self.poisson_solver.Ax[0],
        )
        for level in range(1, self.poisson_solver.n_mg_levels):
            self.poisson_solver.init_gridtype(
                self.poisson_solver.grid_type[level - 1], self.poisson_solver.grid_type[level]
            )
            kernel_assemble_incompressible_mg_A(
                sims.dt,
                scene.element.grid_size,
                scene.element.igrid_size,
                mat_props,
                self.poisson_solver.grid_type[level],
                self.poisson_solver.Adiag[level],
                self.poisson_solver.Ax[level],
            )

    def apply_pressures_regular(self, sims: Simulation, scene: myScene):
        kernel_update_cell_pressure(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.cell.pressure,
            scene.element.flag,
            scene.element.cell.type,
            self.unknow_vector,
        )
        mat_props = self.get_single_fluid_mat_props(scene)
        kernel_correct_velocity(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.grid_size,
            sims.dt,
            mat_props,
            scene.element.cell.pressure,
            scene.element.cell.surface_tension,
            scene.element.cell.type,
            scene.element.cell.fluid_sdf,
            self.use_pressure_free_surface_theta,
            scene.node,
        )
        kernel_compute_mac_grid_acceleration(scene.mass_cut_off, scene.node)

    def apply_pressures_cut_cell(self, sims: Simulation, scene: myScene):
        kernel_update_cell_pressure(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.cell.pressure,
            scene.element.flag,
            scene.element.cell.type,
            self.unknow_vector,
        )
        mat_props = self.get_single_fluid_mat_props(scene)
        face_fraction2 = self.solid_face_fraction2()
        solid_velocity2 = self.solid_face_velocity2()
        kernel_correct_velocity_cut_cell(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.grid_size,
            sims.dt,
            mat_props,
            scene.element.cell.pressure,
            scene.element.cell.surface_tension,
            scene.element.cell.type,
            scene.element.cell.fluid_sdf,
            scene.element.cell.solid_sdf,
            self.solid_face_open_fraction[0],
            self.solid_face_open_fraction[1],
            face_fraction2,
            self.solid_face_velocity[0],
            self.solid_face_velocity[1],
            solid_velocity2,
            self.use_pressure_free_surface_theta,
            scene.node,
        )
        kernel_compute_mac_grid_acceleration(scene.mass_cut_off, scene.node)

    def apply_pressures_coupled(self, sims: Simulation, scene: myScene):
        kernel_update_cell_pressure(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.cell.pressure,
            scene.element.flag,
            scene.element.cell.type,
            self.unknow_vector,
        )
        self.correct_velocity_coupled(sims, scene)

    def apply_pressures_mgpcg_regular(self, sims: Simulation, scene: myScene):
        mat_props = self.get_single_fluid_mat_props(scene)
        kernel_update_pressure_and_correct_velocity_from_mg(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.grid_size,
            sims.dt,
            mat_props,
            self.poisson_solver.x,
            self.poisson_solver.grid_type[0],
            scene.element.cell.pressure,
            scene.element.cell.surface_tension,
            scene.element.cell.type,
            scene.element.cell.fluid_sdf,
            self.use_pressure_free_surface_theta,
            scene.node,
        )
        kernel_compute_mac_grid_acceleration(scene.mass_cut_off, scene.node)

    def apply_pressures_mgpcg_cut_cell(self, sims: Simulation, scene: myScene):
        mat_props = self.get_single_fluid_mat_props(scene)
        face_fraction2 = self.solid_face_fraction2()
        solid_velocity2 = self.solid_face_velocity2()
        kernel_update_pressure_and_correct_velocity_from_mg_cut_cell(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.grid_size,
            sims.dt,
            mat_props,
            self.poisson_solver.x,
            self.poisson_solver.grid_type[0],
            scene.element.cell.pressure,
            scene.element.cell.surface_tension,
            scene.element.cell.type,
            scene.element.cell.fluid_sdf,
            scene.element.cell.solid_sdf,
            self.solid_face_open_fraction[0],
            self.solid_face_open_fraction[1],
            face_fraction2,
            self.solid_face_velocity[0],
            self.solid_face_velocity[1],
            solid_velocity2,
            self.use_pressure_free_surface_theta,
            scene.node,
        )
        kernel_compute_mac_grid_acceleration(scene.mass_cut_off, scene.node)

    def apply_pressures_mgpcg_coupled(self, sims: Simulation, scene: myScene):
        kernel_update_cell_pressure_from_mg(
            scene.element.cell.pressure, self.poisson_solver.grid_type[0], self.poisson_solver.x
        )
        self.correct_velocity_coupled(sims, scene)

    def correct_velocity_coupled(self, sims: Simulation, scene: myScene):
        mat_props = self.get_single_fluid_mat_props(scene)
        if sims.solid_sdf_cut_cell:
            face_fraction = self.solid_face_open_fraction
            solid_velocity = self.solid_face_velocity
        else:
            face_fraction = [self.pressure_solid_fraction] * 3
            solid_velocity = [self.pressure_solid_fraction] * 3
        kernel_correct_velocity_coupled_3d(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.grid_size,
            sims.dt,
            mat_props,
            scene.element.cell.pressure,
            scene.element.cell.surface_tension,
            scene.element.cell.type,
            scene.element.cell.fluid_sdf,
            face_fraction[0],
            face_fraction[1],
            face_fraction[2],
            solid_velocity[0],
            solid_velocity[1],
            solid_velocity[2],
            self.pressure_solid_fraction,
            self.pressure_solid_density,
            self.pressure_coupling_mode,
            sims.solid_sdf_cut_cell,
            self.use_pressure_free_surface_theta,
            scene.node,
        )
        kernel_compute_mac_grid_acceleration(scene.mass_cut_off, scene.node)

    def compute_particle_kinematics(self, sims: Simulation, scene: myScene):
        kernel_kinemaitc_mac_cell_g2p(
            scene.element.grid_nodes,
            sims.alphaPIC,
            sims.dt,
            int(scene.particleNum[0]),
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.grid_size,
            scene.element.igrid_size,
            scene.node,
            scene.particle,
            scene.element.calLength,
        )

    def enforce_mac_boundary_regular(self, sims: Simulation, scene: myScene):
        enforce_boundary(scene.element.ghost_cell, scene.element.cnum, scene.element.cell.type, scene.node)

    def enforce_mac_boundary_cut_cell(self, sims: Simulation, scene: myScene):
        enforce_boundary_cut_cell(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.cell.type,
            self.solid_face_velocity[0],
            self.solid_face_velocity[1],
            self.solid_face_velocity2(),
            scene.node,
            sims.fluid_wall_no_slip,
        )

    def enforce_mac_boundary_after_pressure_regular(self, sims: Simulation, scene: myScene):
        enforce_boundary_and_extrapolate_mac_tangent_velocity(
            scene.element.ghost_cell, scene.element.cnum, scene.element.cell.type, scene.node
        )

    def enforce_mac_boundary_after_pressure_cut_cell(self, sims: Simulation, scene: myScene):
        enforce_boundary_cut_cell(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.cell.type,
            self.solid_face_velocity[0],
            self.solid_face_velocity[1],
            self.solid_face_velocity2(),
            scene.node,
            sims.fluid_wall_no_slip,
        )

    def find_active_nodes(self, scene: myScene):
        find_active_fdm_cell(
            scene.element.ghost_cell,
            scene.element.cellSum,
            scene.element.cnum,
            scene.element.cell.type,
            scene.element.flag,
        )
        scene.element.pse.run(scene.element.flag)
        return set_active_fdm_cell_dofs(
            scene.element.ghost_cell,
            scene.element.cellSum,
            scene.element.cnum,
            scene.element.cell.type,
            scene.element.flag,
        )

    def pre_calculation(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        # This is also the characteristic-length refresh point for new/restarted runs.
        # The basis integral is independent of the moving particles and cell types.
        self.shifting_reference_ready = False
        self.trace_precalculation_stage("Characteristic length begin")
        sims.timer.begin("Characteristic length")
        scene.element.calculate_characteristic_length(sims, int(scene.particleNum[0]), scene.particle, scene.psize)
        sims.timer.end("Characteristic length")
        self.trace_precalculation_stage("Characteristic length end")
        if sims.neighbor_detection:
            self.trace_precalculation_stage("Neighbor setup begin")
            sims.timer.begin("Neighbor setup")
            grid_mass_reset(scene.mass_cut_off, scene.node)
            scene.check_in_domain(sims)
            self.find_free_surface_by_density(sims, scene)
            neighbor.place_particles(scene)
            self.compute_boundary_direction(scene, neighbor)
            self.free_surface_by_geometry(scene, neighbor)
            grid_mass_reset(scene.mass_cut_off, scene.node)
            sims.timer.end("Neighbor setup")
            self.trace_precalculation_stage("Neighbor setup end")
        if sims.discretization == "FDM":
            self.trace_precalculation_stage("FDM setup begin")
            sims.timer.begin("FDM setup")
            scene.element.cell.set_ptr(pressure=None, cell_type=None)
            self.configure_runtime_functions(sims, scene)
            if sims.density_projection:
                self.ensure_density_projection_fields(scene)
            sims.timer.end("FDM setup")
            self.trace_precalculation_stage("FDM setup end")
            self.trace_precalculation_stage("Solid boundary begin")
            sims.timer.begin("Solid boundary")
            init_boundary(scene.element.ghost_cell, scene.element.cnum, scene.element.cell.type)
            self.apply_solid_cell_boundaries(sims, scene)
            self.reset_solid_face_velocity_step(sims, scene)
            self.update_external_cut_cell_boundary(sims, scene)
            self.update_external_ibm_fields(sims, scene)
            self.update_solid_cut_cell_step(sims, scene)
            self.remove_particles_inside_solid_cells(scene)
            sims.timer.end("Solid boundary")
            self.trace_precalculation_stage("Solid boundary end")
            self.trace_precalculation_stage("Fluid domain begin")
            sims.timer.begin("Fluid domain")
            self.identify_fluid_domain(sims, scene)
            sims.timer.end("Fluid domain")
            self.trace_precalculation_stage("Fluid domain end")
            if self.needs_particle_fluid_level_set(sims, scene):
                self.trace_precalculation_stage("Fluid level set begin")
                sims.timer.begin("Fluid level set")
                self.update_fluid_level_set(sims, scene, force_rebuild=True)
                sims.timer.end("Fluid level set")
                self.trace_precalculation_stage("Fluid level set end")
            self.trace_precalculation_stage("Cell pressure begin")
            sims.timer.begin("Cell pressure")
            self.initialize_cell_pressure_from_particles(scene)
            sims.timer.end("Cell pressure")
            self.trace_precalculation_stage("Cell pressure end")
            self.trace_precalculation_stage("Pressure solver setup begin")
            sims.timer.begin("Pressure solver setup")
            if sims.linear_solver == "PCG":
                self.create_pcg_solver(sims, scene)
            elif sims.linear_solver == "MGPCG":
                self.create_mgpcg_solver(sims, scene)
            sims.timer.end("Pressure solver setup")
            self.trace_precalculation_stage("Pressure solver setup end")
        elif sims.discretization == "FEM":
            from src.linear_solver.CompressedSparseRow import CompressedSparseRow

            self.poisson_solver = CompressedSparseRow()

    def fdm_discretization_pcg(self, sims: Simulation, scene: myScene, neighbor=None):
        sims.timer.begin("Solid boundary")
        self.reset_solid_face_velocity_step(sims, scene)
        self.update_external_cut_cell_boundary(sims, scene)
        self.update_external_ibm_fields(sims, scene)
        sims.timer.end("Solid boundary")
        sims.timer.begin("P2G and fluid domain")
        self.compute_nodal_kinematics(sims, scene)
        sims.timer.end("P2G and fluid domain")
        self.run_pressure_fluid_level_set(sims, scene)
        self.run_cut_cell(sims, scene)
        sims.timer.begin("Grid kinematic")
        self.compute_grid_velcity_gravity(sims, scene)
        self.run_coupled_viscosity(sims, scene)
        sims.timer.end("Grid kinematic")
        sims.timer.begin("Fluid-particle coupling")
        self.run_external_fluid_particle_coupling(sims, scene)
        sims.timer.end("Fluid-particle coupling")
        self.run_ibm_source(sims, scene)
        sims.timer.begin("Boundary")
        self.enforce_mac_boundary_step(sims, scene)
        sims.timer.end("Boundary")
        sims.timer.begin("Active cells")
        total_dofs = self.find_active_nodes(scene)
        sims.timer.end("Active cells")
        sims.timer.begin("Poisson assembly")
        self.prepare_poisson_equations_step(sims, scene)
        sims.timer.end("Poisson assembly")
        sims.timer.begin("Poisson solve")
        solved = self.poisson_solver.solve(
            self.operator,
            self.right_hand_vector,
            self.unknow_vector,
            self.diag_A,
            total_dofs,
            maxiter=total_dofs,
            tol=sims.residual_tolerance,
        )
        self.require_pressure_solver_success(solved, "pressure PCG")
        sims.timer.end("Poisson solve")
        sims.timer.begin("Pressure correction")
        self.apply_pressures_step(sims, scene)
        sims.timer.end("Pressure correction")
        sims.timer.begin("Boundary")
        self.enforce_mac_boundary_after_pressure_step(sims, scene)
        sims.timer.end("Boundary")
        sims.timer.begin("G2P")
        self.compute_particle_kinematics(sims, scene)
        sims.timer.end("G2P")
        self.run_density_projection(sims, scene)
        self.run_particle_shifting(sims, scene)
        sims.timer.begin("Particle boundary")
        self.enforce_particle_boundary(scene)
        sims.timer.end("Particle boundary")

    def fdm_discretization_mgpcg(self, sims: Simulation, scene: myScene, neighbor=None):
        sims.timer.begin("Solid boundary")
        self.reset_solid_face_velocity_step(sims, scene)
        self.update_external_cut_cell_boundary(sims, scene)
        self.update_external_ibm_fields(sims, scene)
        sims.timer.end("Solid boundary")
        sims.timer.begin("P2G and fluid domain")
        self.compute_nodal_kinematics(sims, scene)
        sims.timer.end("P2G and fluid domain")
        self.run_pressure_fluid_level_set(sims, scene)
        self.run_cut_cell(sims, scene)
        sims.timer.begin("Grid kinematic")
        self.compute_grid_velcity_gravity(sims, scene)
        self.run_coupled_viscosity(sims, scene)
        sims.timer.end("Grid kinematic")
        sims.timer.begin("Fluid-particle coupling")
        self.run_external_fluid_particle_coupling(sims, scene)
        sims.timer.end("Fluid-particle coupling")
        self.run_ibm_source(sims, scene)
        sims.timer.begin("Boundary")
        self.enforce_mac_boundary_step(sims, scene)
        sims.timer.end("Boundary")
        sims.timer.begin("Poisson assembly")
        self.prepare_poisson_equations_mgpcg_step(sims, scene)
        sims.timer.end("Poisson assembly")
        sims.timer.begin("Poisson solve")
        solved = self.poisson_solver.solve(max_iters=max(1, sims.iter_max), rel_tol=sims.residual_tolerance)
        self.require_pressure_solver_success(solved, "pressure MGPCG")
        sims.timer.end("Poisson solve")
        sims.timer.begin("Pressure correction")
        self.apply_pressures_mgpcg_step(sims, scene)
        sims.timer.end("Pressure correction")
        sims.timer.begin("Boundary")
        self.enforce_mac_boundary_after_pressure_step(sims, scene)
        sims.timer.end("Boundary")
        sims.timer.begin("G2P")
        self.compute_particle_kinematics(sims, scene)
        sims.timer.end("G2P")
        self.run_density_projection(sims, scene)
        self.run_particle_shifting(sims, scene)
        sims.timer.begin("Particle boundary")
        self.enforce_particle_boundary(scene)
        sims.timer.end("Particle boundary")

    def fem_discretization(self, sims, scene: myScene, neighbor=None):
        pass

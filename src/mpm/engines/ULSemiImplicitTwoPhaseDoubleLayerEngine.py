import taichi as ti

from src.linear_solver.MultiGridPCG import MGPCGPoissonSolver
from src.mpm.engines.ULExplicitEngine import ULExplicitEngine
from src.mpm.engines.TwoPhaseDoubleLayerKernel import (
    kernel_accumulate_double_layer_material_fields2d,
    kernel_accumulate_double_layer_material_fields3d,
    kernel_reset_double_layer_grid,
    MAC_SHAPE_LINEAR,
    MAC_SHAPE_GIMP,
    MAC_SHAPE_QUAD_BSPLINE,
    MAC_SHAPE_CUBIC_BSPLINE,
    kernel_assemble_double_layer_pressure_A,
    kernel_assemble_double_layer_pressure_A3d,
    kernel_assemble_double_layer_pressure_mg_A,
    kernel_assemble_double_layer_pressure_mg_A3d,
    kernel_assemble_double_layer_pressure_rhs,
    kernel_assemble_double_layer_pressure_rhs3d,
    kernel_build_double_layer_fluid_sdf2d,
    kernel_build_double_layer_fluid_sdf3d,
    kernel_classify_double_layer_fluid_cells2d,
    kernel_classify_double_layer_fluid_cells3d,
    kernel_correct_double_layer_velocity2d,
    kernel_correct_double_layer_velocity3d,
    kernel_coarsen_double_layer_grid_type2d,
    kernel_coarsen_double_layer_grid_type3d,
    kernel_correct_double_layer_solid_velocity_paper2d,
    kernel_correct_double_layer_solid_velocity_paper3d,
    kernel_advect_double_layer_fluid_particles2d,
    kernel_advect_double_layer_fluid_particles3d,
    kernel_delta_correct_double_layer_fluid2d,
    kernel_delta_correct_double_layer_fluid3d,
    kernel_constrain_double_layer_particles_to_solid_region2d,
    kernel_constrain_double_layer_particles_to_solid_region3d,
    kernel_g2p_double_layer2d,
    kernel_g2p_double_layer3d,
    kernel_enforce_double_layer_solid_cell_faces2d,
    kernel_enforce_double_layer_solid_cell_faces3d,
    kernel_mac_p2g_double_layer2d,
    kernel_mac_p2g_double_layer3d,
    kernel_mark_double_layer_solid_cell_region2d,
    kernel_mark_double_layer_solid_cell_region3d,
    kernel_mark_double_layer_solid_plane_region2d,
    kernel_mark_double_layer_solid_plane_region3d,
    kernel_normalize_double_layer_material_fields,
    kernel_normalize_double_layer_material_fields3d,
    kernel_normalize_double_layer_nodes2d,
    kernel_normalize_double_layer_nodes3d,
    kernel_normalize_double_layer_mac_fields2d,
    kernel_normalize_double_layer_mac_fields3d,
    kernel_p2g_double_layer_mass2d,
    kernel_p2g_double_layer_mass3d,
    kernel_update_double_layer_solid_state2d,
    kernel_update_double_layer_solid_state3d,
    kernel_update_double_layer_fluid_volume2d,
    kernel_update_double_layer_fluid_volume3d,
    kernel_volume_p2g_double_layer_fluid2d,
    kernel_volume_p2g_double_layer_fluid3d,
    kernel_force_p2g_double_layer2d,
    kernel_force_p2g_double_layer3d,
    kernel_predict_double_layer2d,
    kernel_predict_double_layer3d,
    kernel_project_solid_grid_velocity_to_mac2d,
    kernel_project_solid_grid_velocity_to_mac3d,
    kernel_reset_double_layer_fields,
    kernel_reset_double_layer_fields3d,
    kernel_reset_double_layer_material_fields,
    kernel_reset_double_layer_material_fields3d,
    kernel_project_double_layer_pressure_to_solid_nodes2d,
    kernel_project_double_layer_pressure_to_solid_nodes3d,
    kernel_sample_double_layer_solid_pressure_from_nodes2d,
    kernel_sample_double_layer_solid_pressure_from_nodes3d,
    kernel_constrain_double_layer_particles_to_solid_plane_region2d,
    kernel_constrain_double_layer_particles_to_solid_plane_region3d,
    kernel_enforce_double_layer_solid_plane_nodes2d,
    kernel_enforce_double_layer_solid_plane_nodes3d,
)
from src.mpm.SceneManager import myScene
from src.mpm.Simulation import Simulation
from src.mpm.SpatialHashGrid import SpatialHashGrid
from src.utils.linalg import no_operation


class _DoubleLayerCellOutput:
    def __init__(self, cell_type, pressure, fluid_sdf):
        self.type = cell_type
        self.pressure = pressure
        self.fluid_sdf = fluid_sdf
        self.surface_tension = None
        self.solid_sdf = None


@ti.data_oriented
class ULSemiImplicitTwoPhaseDoubleLayerEngine(ULExplicitEngine):
    """Semi-implicit two-phase two-point MPM.

    This path is intentionally separate from ULSemiImplicitTwoPhaseEngine, which
    keeps the existing single-layer material point formulation unchanged.
    """

    def __init__(self, sims) -> None:
        super().__init__(sims)
        self.allocated = False
        self.poisson_solver = None
        self.poisson_iterations = 100
        self.mac_shape_type = 0
        self.mac_influenced_node = 0
        self.use_affine_projection = 0
        self.delayed_fluid_advection_flag = 0
        self.fluid_volume_initialized = False
        self.allocate_mac_fields = None
        self.advect_double_layer_fluid_particles = no_operation
        self.reset_double_layer_fields = None
        self.accumulate_material_fields = None
        self.assemble_pressure_mg_levels = None
        self.solve_pressure = None
        self.build_fluid_sdf = None
        self.sample_pressure_to_solid_particles = None
        self.correct_solid_velocity_paper = None
        self.mark_solid_cell_boundaries = None
        self.enforce_solid_cell_faces = None
        self.constrain_particles_to_solid_cell_regions = no_operation
        self.shift_double_layer_fluid_particles = no_operation
        self.update_external_mac_boundary = no_operation
        self.enforce_external_mac_boundary_after_pressure = no_operation

    def _mac_shape_settings(self, sims: Simulation):
        if sims.shape_function in ("Linear", "SmoothLinear"):
            return MAC_SHAPE_LINEAR, 2
        if sims.shape_function == "GIMP":
            return MAC_SHAPE_GIMP, 3
        if sims.shape_function == "QuadBSpline":
            return MAC_SHAPE_QUAD_BSPLINE, 3
        if sims.shape_function == "CubicBSpline":
            return MAC_SHAPE_CUBIC_BSPLINE, 4
        raise RuntimeError(
            f"TwoPhaseDoubleLayer does not support MAC transfer for shape function {sims.shape_function}"
        )

    def _use_affine_projection(self, sims: Simulation):
        if sims.shape_function in ("Linear", "SmoothLinear"):
            return 0
        return int(sims.velocity_projection_scheme == "Affine")

    def _single_twophase_matprop(self, scene: myScene):
        for material_id in range(1, scene.material.matProps.size()):
            return scene.material.matProps[material_id]
        return scene.material.matProps[0]

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

    def choose_engine(self, sims: Simulation):
        if sims.is_2DAxisy:
            raise RuntimeError("TwoPhaseDoubleLayer SemiImplicit does not support 2D axisymmetric mode")
        if sims.dimension not in (2, 3):
            raise RuntimeError("TwoPhaseDoubleLayer SemiImplicit currently supports 2D and 3D only")
        if sims.mapping not in ["USL", "USF", "MUSL"]:
            raise ValueError(f"The mapping scheme {sims.mapping} is not supported for TwoPhaseDoubleLayer SemiImplicit")
        self.mac_shape_type, self.mac_influenced_node = self._mac_shape_settings(sims)
        self.use_affine_projection = self._use_affine_projection(sims)
        self.delayed_fluid_advection_flag = int(sims.delayed_fluid_advection)
        self.advect_double_layer_fluid_particles = no_operation
        if sims.dimension == 3:
            self.compute = self._semi_implicit_update_3d
            self.allocate_mac_fields = self._allocate_mac_fields3d
            self.reset_double_layer_fields = self._reset_double_layer_fields3d
            self.accumulate_material_fields = self._accumulate_material_fields3d
            self.assemble_pressure_mg_levels = self._assemble_pressure_mg_levels3d
            self.solve_pressure = self._solve_pressure3d
            self.build_fluid_sdf = self._build_fluid_sdf3d
            self.sample_pressure_to_solid_particles = self._sample_pressure_to_solid_particles3d
            self.correct_solid_velocity_paper = self._correct_solid_velocity_paper3d
            self.mark_solid_cell_boundaries = self._mark_solid_cell_boundaries3d
            self.enforce_solid_cell_faces = self._enforce_solid_cell_faces3d
            self.constrain_particles_to_solid_cell_regions = self._constrain_particles_to_solid_cell_regions3d
            if sims.delayed_fluid_advection:
                self.advect_double_layer_fluid_particles = self._advect_double_layer_fluid_particles3d
            if sims.particle_shifting:
                self.shift_double_layer_fluid_particles = self._shift_double_layer_fluid_particles3d
        else:
            self.compute = self._semi_implicit_update_2d
            self.allocate_mac_fields = self._allocate_mac_fields2d
            self.reset_double_layer_fields = self._reset_double_layer_fields2d
            self.accumulate_material_fields = self._accumulate_material_fields2d
            self.assemble_pressure_mg_levels = self._assemble_pressure_mg_levels2d
            self.solve_pressure = self._solve_pressure2d
            self.build_fluid_sdf = self._build_fluid_sdf2d
            self.sample_pressure_to_solid_particles = self._sample_pressure_to_solid_particles2d
            self.correct_solid_velocity_paper = self._correct_solid_velocity_paper2d
            self.mark_solid_cell_boundaries = self._mark_solid_cell_boundaries2d
            self.enforce_solid_cell_faces = self._enforce_solid_cell_faces2d
            self.constrain_particles_to_solid_cell_regions = self._constrain_particles_to_solid_cell_regions2d
            if sims.delayed_fluid_advection:
                self.advect_double_layer_fluid_particles = self._advect_double_layer_fluid_particles2d
            if sims.particle_shifting:
                self.shift_double_layer_fluid_particles = self._shift_double_layer_fluid_particles2d

    def reset_grid_message(self, scene: myScene):
        kernel_reset_double_layer_grid(scene.node)

    def _allocate_common_mac_fields(self, sims: Simulation, scene: myScene, dimension, grid_shape):
        node_shape = (int(scene.element.gridSum), int(scene.grid_level))
        self.node_material_weight = ti.field(float, shape=node_shape)
        self.node_solid_density = ti.field(float, shape=node_shape)
        self.node_fluid_density = ti.field(float, shape=node_shape)
        self.node_fluid_viscosity = ti.field(float, shape=node_shape)
        self.node_grain_diameter = ti.field(float, shape=node_shape)
        self.node_permeability = ti.field(float, shape=node_shape)
        self.node_fluid_unit_weight = ti.field(float, shape=node_shape)
        self.node_drag_model = ti.field(float, shape=node_shape)

        self.poisson_solver = MGPCGPoissonSolver(
            dimension,
            grid_shape,
            n_mg_levels=max(1, int(sims.multilevel)),
            pre_and_post_smoothing=sims.pre_and_post_smoothing,
            bottom_smoothing=max(10, sims.bottom_smoothing),
            smoother="rbgs",
        )
        self.poisson_iterations = max(50, int(sims.iter_max))
        self._bind_cell_output_fields(scene)
        self.allocated = True

    def _allocate_mac_fields2d(self, sims: Simulation, scene: myScene):
        if self.allocated:
            return

        nx = int(scene.element.cnum[0])
        ny = int(scene.element.cnum[1])
        self.fluid_mass_x = ti.field(float, shape=(nx + 1, ny))
        self.fluid_mass_y = ti.field(float, shape=(nx, ny + 1))
        self.fluid_velocity_x = ti.field(float, shape=(nx + 1, ny))
        self.fluid_velocity_y = ti.field(float, shape=(nx, ny + 1))
        self.fluid_velocity0_x = ti.field(float, shape=(nx + 1, ny))
        self.fluid_velocity0_y = ti.field(float, shape=(nx, ny + 1))
        self.fluid_acceleration_x = ti.field(float, shape=(nx + 1, ny))
        self.fluid_acceleration_y = ti.field(float, shape=(nx, ny + 1))

        self.solid_mass_x = ti.field(float, shape=(nx + 1, ny))
        self.solid_mass_y = ti.field(float, shape=(nx, ny + 1))
        self.solid_velocity_x = ti.field(float, shape=(nx + 1, ny))
        self.solid_velocity_y = ti.field(float, shape=(nx, ny + 1))
        self.face_porosity_x = ti.field(float, shape=(nx + 1, ny))
        self.face_porosity_y = ti.field(float, shape=(nx, ny + 1))
        self.face_material_weight_x = ti.field(float, shape=(nx + 1, ny))
        self.face_material_weight_y = ti.field(float, shape=(nx, ny + 1))
        self.face_solid_density_x = ti.field(float, shape=(nx + 1, ny))
        self.face_solid_density_y = ti.field(float, shape=(nx, ny + 1))
        self.face_fluid_density_x = ti.field(float, shape=(nx + 1, ny))
        self.face_fluid_density_y = ti.field(float, shape=(nx, ny + 1))
        self.face_fluid_viscosity_x = ti.field(float, shape=(nx + 1, ny))
        self.face_fluid_viscosity_y = ti.field(float, shape=(nx, ny + 1))
        self.face_grain_diameter_x = ti.field(float, shape=(nx + 1, ny))
        self.face_grain_diameter_y = ti.field(float, shape=(nx, ny + 1))
        self.face_permeability_x = ti.field(float, shape=(nx + 1, ny))
        self.face_permeability_y = ti.field(float, shape=(nx, ny + 1))
        self.face_fluid_unit_weight_x = ti.field(float, shape=(nx + 1, ny))
        self.face_fluid_unit_weight_y = ti.field(float, shape=(nx, ny + 1))
        self.face_drag_model_x = ti.field(float, shape=(nx + 1, ny))
        self.face_drag_model_y = ti.field(float, shape=(nx, ny + 1))

        self.cell_type = ti.field(int, shape=(nx, ny))
        self.cell_fluid_mass = ti.field(float, shape=(nx, ny))
        self.cell_solid_mass = ti.field(float, shape=(nx, ny))
        self.cell_porosity = ti.field(float, shape=(nx, ny))
        self.cell_solid_velocity = ti.Vector.field(2, float, shape=(nx, ny))
        self.cell_fluid_velocity = ti.Vector.field(2, float, shape=(nx, ny))
        self.cell_pressure = ti.field(float, shape=(nx, ny))
        self.cell_fluid_sdf = ti.field(float, shape=(nx, ny))
        self.cell_material_weight = ti.field(float, shape=(nx, ny))
        self.cell_solid_density = ti.field(float, shape=(nx, ny))
        self.cell_fluid_density = ti.field(float, shape=(nx, ny))
        self.cell_fluid_viscosity = ti.field(float, shape=(nx, ny))
        self.cell_grain_diameter = ti.field(float, shape=(nx, ny))
        self._allocate_common_mac_fields(sims, scene, 2, [nx, ny])

    def _allocate_mac_fields3d(self, sims: Simulation, scene: myScene):
        if self.allocated:
            return

        nx = int(scene.element.cnum[0])
        ny = int(scene.element.cnum[1])
        nz = int(scene.element.cnum[2])
        self.fluid_mass_x = ti.field(float, shape=(nx + 1, ny, nz))
        self.fluid_mass_y = ti.field(float, shape=(nx, ny + 1, nz))
        self.fluid_mass_z = ti.field(float, shape=(nx, ny, nz + 1))
        self.fluid_velocity_x = ti.field(float, shape=(nx + 1, ny, nz))
        self.fluid_velocity_y = ti.field(float, shape=(nx, ny + 1, nz))
        self.fluid_velocity_z = ti.field(float, shape=(nx, ny, nz + 1))
        self.fluid_velocity0_x = ti.field(float, shape=(nx + 1, ny, nz))
        self.fluid_velocity0_y = ti.field(float, shape=(nx, ny + 1, nz))
        self.fluid_velocity0_z = ti.field(float, shape=(nx, ny, nz + 1))
        self.fluid_acceleration_x = ti.field(float, shape=(nx + 1, ny, nz))
        self.fluid_acceleration_y = ti.field(float, shape=(nx, ny + 1, nz))
        self.fluid_acceleration_z = ti.field(float, shape=(nx, ny, nz + 1))

        self.solid_mass_x = ti.field(float, shape=(nx + 1, ny, nz))
        self.solid_mass_y = ti.field(float, shape=(nx, ny + 1, nz))
        self.solid_mass_z = ti.field(float, shape=(nx, ny, nz + 1))
        self.solid_velocity_x = ti.field(float, shape=(nx + 1, ny, nz))
        self.solid_velocity_y = ti.field(float, shape=(nx, ny + 1, nz))
        self.solid_velocity_z = ti.field(float, shape=(nx, ny, nz + 1))
        self.face_porosity_x = ti.field(float, shape=(nx + 1, ny, nz))
        self.face_porosity_y = ti.field(float, shape=(nx, ny + 1, nz))
        self.face_porosity_z = ti.field(float, shape=(nx, ny, nz + 1))
        self.face_material_weight_x = ti.field(float, shape=(nx + 1, ny, nz))
        self.face_material_weight_y = ti.field(float, shape=(nx, ny + 1, nz))
        self.face_material_weight_z = ti.field(float, shape=(nx, ny, nz + 1))
        self.face_solid_density_x = ti.field(float, shape=(nx + 1, ny, nz))
        self.face_solid_density_y = ti.field(float, shape=(nx, ny + 1, nz))
        self.face_solid_density_z = ti.field(float, shape=(nx, ny, nz + 1))
        self.face_fluid_density_x = ti.field(float, shape=(nx + 1, ny, nz))
        self.face_fluid_density_y = ti.field(float, shape=(nx, ny + 1, nz))
        self.face_fluid_density_z = ti.field(float, shape=(nx, ny, nz + 1))
        self.face_fluid_viscosity_x = ti.field(float, shape=(nx + 1, ny, nz))
        self.face_fluid_viscosity_y = ti.field(float, shape=(nx, ny + 1, nz))
        self.face_fluid_viscosity_z = ti.field(float, shape=(nx, ny, nz + 1))
        self.face_grain_diameter_x = ti.field(float, shape=(nx + 1, ny, nz))
        self.face_grain_diameter_y = ti.field(float, shape=(nx, ny + 1, nz))
        self.face_grain_diameter_z = ti.field(float, shape=(nx, ny, nz + 1))
        self.face_permeability_x = ti.field(float, shape=(nx + 1, ny, nz))
        self.face_permeability_y = ti.field(float, shape=(nx, ny + 1, nz))
        self.face_permeability_z = ti.field(float, shape=(nx, ny, nz + 1))
        self.face_fluid_unit_weight_x = ti.field(float, shape=(nx + 1, ny, nz))
        self.face_fluid_unit_weight_y = ti.field(float, shape=(nx, ny + 1, nz))
        self.face_fluid_unit_weight_z = ti.field(float, shape=(nx, ny, nz + 1))
        self.face_drag_model_x = ti.field(float, shape=(nx + 1, ny, nz))
        self.face_drag_model_y = ti.field(float, shape=(nx, ny + 1, nz))
        self.face_drag_model_z = ti.field(float, shape=(nx, ny, nz + 1))

        self.cell_type = ti.field(int, shape=(nx, ny, nz))
        self.cell_fluid_mass = ti.field(float, shape=(nx, ny, nz))
        self.cell_solid_mass = ti.field(float, shape=(nx, ny, nz))
        self.cell_porosity = ti.field(float, shape=(nx, ny, nz))
        self.cell_solid_velocity = ti.Vector.field(3, float, shape=(nx, ny, nz))
        self.cell_fluid_velocity = ti.Vector.field(3, float, shape=(nx, ny, nz))
        self.cell_pressure = ti.field(float, shape=(nx, ny, nz))
        self.cell_fluid_sdf = ti.field(float, shape=(nx, ny, nz))
        self.cell_material_weight = ti.field(float, shape=(nx, ny, nz))
        self.cell_solid_density = ti.field(float, shape=(nx, ny, nz))
        self.cell_fluid_density = ti.field(float, shape=(nx, ny, nz))
        self.cell_fluid_viscosity = ti.field(float, shape=(nx, ny, nz))
        self.cell_grain_diameter = ti.field(float, shape=(nx, ny, nz))
        self._allocate_common_mac_fields(sims, scene, 3, [nx, ny, nz])

    def _bind_cell_output_fields(self, scene: myScene):
        cell = getattr(scene.element, "cell", None)
        if cell is not None and hasattr(cell, "set_ptr"):
            cell.set_ptr(
                pressure=self.cell_pressure,
                cell_type=self.cell_type,
                fluid_sdf=self.cell_fluid_sdf,
            )
        else:
            scene.element.cell = _DoubleLayerCellOutput(
                self.cell_type,
                self.cell_pressure,
                self.cell_fluid_sdf,
            )

    def pre_calculation(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        scene.element.calculate_characteristic_length(sims, int(scene.particleNum[0]), scene.particle, scene.psize)
        self.allocate_mac_fields(sims, scene)
        self.limit = sims.verlet_distance * sims.verlet_distance

    def _advect_double_layer_fluid_particles2d(self, sims: Simulation, scene: myScene):
        kernel_advect_double_layer_fluid_particles2d(
            int(scene.particleNum[0]),
            sims.domain,
            sims.dt,
            scene.particle,
        )

    def _advect_double_layer_fluid_particles3d(self, sims: Simulation, scene: myScene):
        kernel_advect_double_layer_fluid_particles3d(
            int(scene.particleNum[0]),
            sims.domain,
            sims.dt,
            scene.particle,
        )

    def _update_double_layer_fluid_volume2d(self, scene: myScene):
        initialize_mass = int(not self.fluid_volume_initialized)
        for _, start_index, end_index, mat_prop in self._iter_twophase_materials(scene):
            kernel_update_double_layer_fluid_volume2d(
                scene.element.grid_nodes,
                start_index,
                end_index,
                initialize_mass,
                mat_prop.fluid_density,
                scene.node,
                scene.particle,
                scene.material.materialID,
                scene.element.LnID,
                scene.element.shape_fn,
                scene.element.node_size,
            )
        self.fluid_volume_initialized = True
        return bool(initialize_mass)

    def _update_double_layer_fluid_volume3d(self, scene: myScene):
        initialize_mass = int(not self.fluid_volume_initialized)
        for _, start_index, end_index, mat_prop in self._iter_twophase_materials(scene):
            kernel_update_double_layer_fluid_volume3d(
                scene.element.grid_nodes,
                start_index,
                end_index,
                initialize_mass,
                mat_prop.fluid_density,
                scene.node,
                scene.particle,
                scene.material.materialID,
                scene.element.LnID,
                scene.element.shape_fn,
                scene.element.node_size,
            )
        self.fluid_volume_initialized = True
        return bool(initialize_mass)

    def _shift_double_layer_fluid_particles2d(self, sims: Simulation, scene: myScene):
        self.calculate_interpolation(sims, scene)
        scene.node.vol.fill(0.0)
        kernel_volume_p2g_double_layer_fluid2d(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )
        kernel_delta_correct_double_layer_fluid2d(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            sims.domain,
            scene.element.grid_size,
            scene.element.gnum,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.dshape_fn,
            scene.element.node_size,
        )

    def _shift_double_layer_fluid_particles3d(self, sims: Simulation, scene: myScene):
        self.calculate_interpolation(sims, scene)
        scene.node.vol.fill(0.0)
        kernel_volume_p2g_double_layer_fluid3d(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )
        kernel_delta_correct_double_layer_fluid3d(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            sims.domain,
            scene.element.grid_size,
            scene.element.gnum,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.dshape_fn,
            scene.element.node_size,
        )

    def _constrain_particles_to_solid_cell_regions2d(self, sims: Simulation, scene: myScene):
        if scene.boundary is None:
            return
        for start_point, end_point in scene.boundary.solid_cell_regions:
            kernel_constrain_double_layer_particles_to_solid_region2d(
                int(scene.particleNum[0]),
                sims.domain,
                ti.Vector(start_point.tolist()),
                ti.Vector(end_point.tolist()),
                scene.particle,
            )
        for start_point, end_point, plane_point, plane_normal in getattr(
            scene.boundary, "solid_cell_plane_regions", []
        ):
            kernel_constrain_double_layer_particles_to_solid_plane_region2d(
                int(scene.particleNum[0]),
                sims.domain,
                ti.Vector(start_point.tolist()),
                ti.Vector(end_point.tolist()),
                ti.Vector(plane_point.tolist()),
                ti.Vector(plane_normal.tolist()),
                scene.particle,
            )

    def _apply_solid_plane_node_boundaries2d(self, scene: myScene):
        if scene.boundary is None:
            return
        for start_point, end_point, plane_point, plane_normal in getattr(
            scene.boundary, "solid_cell_plane_regions", []
        ):
            kernel_enforce_double_layer_solid_plane_nodes2d(
                scene.mass_cut_off,
                scene.element.grid_size,
                scene.element.gnum,
                ti.Vector(start_point.tolist()),
                ti.Vector(end_point.tolist()),
                ti.Vector(plane_point.tolist()),
                ti.Vector(plane_normal.tolist()),
                scene.node,
            )

    def _constrain_particles_to_solid_cell_regions3d(self, sims: Simulation, scene: myScene):
        if scene.boundary is None:
            return
        for start_point, end_point in scene.boundary.solid_cell_regions:
            kernel_constrain_double_layer_particles_to_solid_region3d(
                int(scene.particleNum[0]),
                sims.domain,
                ti.Vector(start_point.tolist()),
                ti.Vector(end_point.tolist()),
                scene.particle,
            )
        for start_point, end_point, plane_point, plane_normal in getattr(
            scene.boundary, "solid_cell_plane_regions", []
        ):
            kernel_constrain_double_layer_particles_to_solid_plane_region3d(
                int(scene.particleNum[0]),
                sims.domain,
                ti.Vector(start_point.tolist()),
                ti.Vector(end_point.tolist()),
                ti.Vector(plane_point.tolist()),
                ti.Vector(plane_normal.tolist()),
                scene.particle,
            )

    def _apply_solid_plane_node_boundaries3d(self, scene: myScene):
        if scene.boundary is None:
            return
        for start_point, end_point, plane_point, plane_normal in getattr(
            scene.boundary, "solid_cell_plane_regions", []
        ):
            kernel_enforce_double_layer_solid_plane_nodes3d(
                scene.mass_cut_off,
                scene.element.grid_size,
                scene.element.gnum,
                ti.Vector(start_point.tolist()),
                ti.Vector(end_point.tolist()),
                ti.Vector(plane_point.tolist()),
                ti.Vector(plane_normal.tolist()),
                scene.node,
            )

    def _reset_double_layer_fields2d(self, sims: Simulation):
        kernel_reset_double_layer_fields(
            self.fluid_mass_x,
            self.fluid_mass_y,
            self.fluid_velocity_x,
            self.fluid_velocity_y,
            self.fluid_velocity0_x,
            self.fluid_velocity0_y,
            self.fluid_acceleration_x,
            self.fluid_acceleration_y,
            self.solid_mass_x,
            self.solid_mass_y,
            self.solid_velocity_x,
            self.solid_velocity_y,
            self.face_porosity_x,
            self.face_porosity_y,
            self.cell_type,
            self.cell_fluid_mass,
            self.cell_solid_mass,
            self.cell_porosity,
            self.cell_solid_velocity,
            self.cell_fluid_velocity,
            self.cell_pressure,
        )
        kernel_reset_double_layer_material_fields(
            self.face_material_weight_x,
            self.face_material_weight_y,
            self.face_solid_density_x,
            self.face_solid_density_y,
            self.face_fluid_density_x,
            self.face_fluid_density_y,
            self.face_fluid_viscosity_x,
            self.face_fluid_viscosity_y,
            self.face_grain_diameter_x,
            self.face_grain_diameter_y,
            self.face_permeability_x,
            self.face_permeability_y,
            self.face_fluid_unit_weight_x,
            self.face_fluid_unit_weight_y,
            self.face_drag_model_x,
            self.face_drag_model_y,
            self.cell_material_weight,
            self.cell_solid_density,
            self.cell_fluid_density,
            self.cell_fluid_viscosity,
            self.cell_grain_diameter,
            self.node_material_weight,
            self.node_solid_density,
            self.node_fluid_density,
            self.node_fluid_viscosity,
            self.node_grain_diameter,
            self.node_permeability,
            self.node_fluid_unit_weight,
            self.node_drag_model,
        )

    def _reset_double_layer_fields3d(self, sims: Simulation):
        kernel_reset_double_layer_fields3d(
            self.fluid_mass_x,
            self.fluid_mass_y,
            self.fluid_mass_z,
            self.fluid_velocity_x,
            self.fluid_velocity_y,
            self.fluid_velocity_z,
            self.fluid_velocity0_x,
            self.fluid_velocity0_y,
            self.fluid_velocity0_z,
            self.fluid_acceleration_x,
            self.fluid_acceleration_y,
            self.fluid_acceleration_z,
            self.solid_mass_x,
            self.solid_mass_y,
            self.solid_mass_z,
            self.solid_velocity_x,
            self.solid_velocity_y,
            self.solid_velocity_z,
            self.face_porosity_x,
            self.face_porosity_y,
            self.face_porosity_z,
            self.cell_type,
            self.cell_fluid_mass,
            self.cell_solid_mass,
            self.cell_porosity,
            self.cell_solid_velocity,
            self.cell_fluid_velocity,
            self.cell_pressure,
        )
        kernel_reset_double_layer_material_fields3d(
            self.face_material_weight_x,
            self.face_material_weight_y,
            self.face_material_weight_z,
            self.face_solid_density_x,
            self.face_solid_density_y,
            self.face_solid_density_z,
            self.face_fluid_density_x,
            self.face_fluid_density_y,
            self.face_fluid_density_z,
            self.face_fluid_viscosity_x,
            self.face_fluid_viscosity_y,
            self.face_fluid_viscosity_z,
            self.face_grain_diameter_x,
            self.face_grain_diameter_y,
            self.face_grain_diameter_z,
            self.face_permeability_x,
            self.face_permeability_y,
            self.face_permeability_z,
            self.face_fluid_unit_weight_x,
            self.face_fluid_unit_weight_y,
            self.face_fluid_unit_weight_z,
            self.face_drag_model_x,
            self.face_drag_model_y,
            self.face_drag_model_z,
            self.cell_material_weight,
            self.cell_solid_density,
            self.cell_fluid_density,
            self.cell_fluid_viscosity,
            self.cell_grain_diameter,
            self.node_material_weight,
            self.node_solid_density,
            self.node_fluid_density,
            self.node_fluid_viscosity,
            self.node_grain_diameter,
            self.node_permeability,
            self.node_fluid_unit_weight,
            self.node_drag_model,
        )

    def _accumulate_material_fields2d(self, sims: Simulation, scene: myScene):
        mac_shape_type = self.mac_shape_type
        mac_influenced_node = self.mac_influenced_node
        for _, start_index, end_index, mat_prop in self._iter_twophase_materials(scene):
            kernel_accumulate_double_layer_material_fields2d(
                scene.element.grid_nodes,
                start_index,
                end_index,
                scene.element.grid_size,
                mac_shape_type,
                mac_influenced_node,
                scene.particle,
                scene.material.materialID,
                mat_prop,
                scene.element.calLength,
                scene.element.LnID,
                scene.element.shape_fn,
                scene.element.node_size,
                self.face_material_weight_x,
                self.face_material_weight_y,
                self.face_solid_density_x,
                self.face_solid_density_y,
                self.face_fluid_density_x,
                self.face_fluid_density_y,
                self.face_fluid_viscosity_x,
                self.face_fluid_viscosity_y,
                self.face_grain_diameter_x,
                self.face_grain_diameter_y,
                self.face_permeability_x,
                self.face_permeability_y,
                self.face_fluid_unit_weight_x,
                self.face_fluid_unit_weight_y,
                self.face_drag_model_x,
                self.face_drag_model_y,
                self.cell_material_weight,
                self.cell_solid_density,
                self.cell_fluid_density,
                self.cell_fluid_viscosity,
                self.cell_grain_diameter,
                self.node_material_weight,
                self.node_solid_density,
                self.node_fluid_density,
                self.node_fluid_viscosity,
                self.node_grain_diameter,
                self.node_permeability,
                self.node_fluid_unit_weight,
                self.node_drag_model,
            )
        kernel_normalize_double_layer_material_fields(
            self._single_twophase_matprop(scene),
            self.face_material_weight_x,
            self.face_material_weight_y,
            self.face_solid_density_x,
            self.face_solid_density_y,
            self.face_fluid_density_x,
            self.face_fluid_density_y,
            self.face_fluid_viscosity_x,
            self.face_fluid_viscosity_y,
            self.face_grain_diameter_x,
            self.face_grain_diameter_y,
            self.face_permeability_x,
            self.face_permeability_y,
            self.face_fluid_unit_weight_x,
            self.face_fluid_unit_weight_y,
            self.face_drag_model_x,
            self.face_drag_model_y,
            self.cell_material_weight,
            self.cell_solid_density,
            self.cell_fluid_density,
            self.cell_fluid_viscosity,
            self.cell_grain_diameter,
            self.node_material_weight,
            self.node_solid_density,
            self.node_fluid_density,
            self.node_fluid_viscosity,
            self.node_grain_diameter,
            self.node_permeability,
            self.node_fluid_unit_weight,
            self.node_drag_model,
        )

    def _accumulate_material_fields3d(self, sims: Simulation, scene: myScene):
        mac_shape_type = self.mac_shape_type
        mac_influenced_node = self.mac_influenced_node
        for _, start_index, end_index, mat_prop in self._iter_twophase_materials(scene):
            kernel_accumulate_double_layer_material_fields3d(
                scene.element.grid_nodes,
                start_index,
                end_index,
                scene.element.grid_size,
                mac_shape_type,
                mac_influenced_node,
                scene.particle,
                scene.material.materialID,
                mat_prop,
                scene.element.calLength,
                scene.element.LnID,
                scene.element.shape_fn,
                scene.element.node_size,
                self.face_material_weight_x,
                self.face_material_weight_y,
                self.face_material_weight_z,
                self.face_solid_density_x,
                self.face_solid_density_y,
                self.face_solid_density_z,
                self.face_fluid_density_x,
                self.face_fluid_density_y,
                self.face_fluid_density_z,
                self.face_fluid_viscosity_x,
                self.face_fluid_viscosity_y,
                self.face_fluid_viscosity_z,
                self.face_grain_diameter_x,
                self.face_grain_diameter_y,
                self.face_grain_diameter_z,
                self.face_permeability_x,
                self.face_permeability_y,
                self.face_permeability_z,
                self.face_fluid_unit_weight_x,
                self.face_fluid_unit_weight_y,
                self.face_fluid_unit_weight_z,
                self.face_drag_model_x,
                self.face_drag_model_y,
                self.face_drag_model_z,
                self.cell_material_weight,
                self.cell_solid_density,
                self.cell_fluid_density,
                self.cell_fluid_viscosity,
                self.cell_grain_diameter,
                self.node_material_weight,
                self.node_solid_density,
                self.node_fluid_density,
                self.node_fluid_viscosity,
                self.node_grain_diameter,
                self.node_permeability,
                self.node_fluid_unit_weight,
                self.node_drag_model,
            )
        kernel_normalize_double_layer_material_fields3d(
            self._single_twophase_matprop(scene),
            self.face_material_weight_x,
            self.face_material_weight_y,
            self.face_material_weight_z,
            self.face_solid_density_x,
            self.face_solid_density_y,
            self.face_solid_density_z,
            self.face_fluid_density_x,
            self.face_fluid_density_y,
            self.face_fluid_density_z,
            self.face_fluid_viscosity_x,
            self.face_fluid_viscosity_y,
            self.face_fluid_viscosity_z,
            self.face_grain_diameter_x,
            self.face_grain_diameter_y,
            self.face_grain_diameter_z,
            self.face_permeability_x,
            self.face_permeability_y,
            self.face_permeability_z,
            self.face_fluid_unit_weight_x,
            self.face_fluid_unit_weight_y,
            self.face_fluid_unit_weight_z,
            self.face_drag_model_x,
            self.face_drag_model_y,
            self.face_drag_model_z,
            self.cell_material_weight,
            self.cell_solid_density,
            self.cell_fluid_density,
            self.cell_fluid_viscosity,
            self.cell_grain_diameter,
            self.node_material_weight,
            self.node_solid_density,
            self.node_fluid_density,
            self.node_fluid_viscosity,
            self.node_grain_diameter,
            self.node_permeability,
            self.node_fluid_unit_weight,
            self.node_drag_model,
        )

    def _assemble_pressure_mg_levels2d(self, sims: Simulation, scene: myScene):
        for level in range(1, self.poisson_solver.n_mg_levels):
            factor = 2**level
            kernel_coarsen_double_layer_grid_type2d(
                self.poisson_solver.grid_type[level - 1],
                self.poisson_solver.grid_type[level],
            )
            kernel_assemble_double_layer_pressure_mg_A(
                scene.element.grid_size,
                sims.dt,
                factor,
                self.poisson_solver.grid_type[level],
                self.face_porosity_x,
                self.face_porosity_y,
                self.cell_solid_density,
                self.cell_fluid_density,
                self.cell_fluid_sdf,
                self.poisson_solver.Adiag[level],
                self.poisson_solver.Ax[level],
            )

    def _assemble_pressure_mg_levels3d(self, sims: Simulation, scene: myScene):
        for level in range(1, self.poisson_solver.n_mg_levels):
            factor = 2**level
            kernel_coarsen_double_layer_grid_type3d(
                self.poisson_solver.grid_type[level - 1],
                self.poisson_solver.grid_type[level],
            )
            kernel_assemble_double_layer_pressure_mg_A3d(
                scene.element.grid_size,
                sims.dt,
                factor,
                self.poisson_solver.grid_type[level],
                self.face_porosity_x,
                self.face_porosity_y,
                self.face_porosity_z,
                self.cell_solid_density,
                self.cell_fluid_density,
                self.cell_fluid_sdf,
                self.poisson_solver.Adiag[level],
                self.poisson_solver.Ax[level],
            )

    def _solve_pressure2d(self, sims: Simulation, scene: myScene):
        self._build_fluid_sdf2d(sims, scene)
        self.poisson_solver.reinitialize(self.cell_type)
        kernel_assemble_double_layer_pressure_rhs(
            scene.element.grid_size,
            self.cell_type,
            self.face_porosity_x,
            self.face_porosity_y,
            self.fluid_velocity_x,
            self.fluid_velocity_y,
            self.solid_velocity_x,
            self.solid_velocity_y,
            self.poisson_solver.b,
        )
        kernel_assemble_double_layer_pressure_A(
            scene.element.grid_size,
            sims.dt,
            self.poisson_solver.grid_type[0],
            self.face_porosity_x,
            self.face_porosity_y,
            self.cell_solid_density,
            self.cell_fluid_density,
            self.cell_fluid_sdf,
            self.poisson_solver.Adiag[0],
            self.poisson_solver.Ax[0],
        )
        self._assemble_pressure_mg_levels2d(sims, scene)
        solved = self.poisson_solver.solve(self.poisson_iterations, rel_tol=sims.residual_tolerance, abs_tol=1.0e-10)
        if not solved:
            raise RuntimeError(
                "TwoPhaseDoubleLayer pressure solve failed: "
                f"{self.poisson_solver.breakdown_reason}={self.poisson_solver.breakdown_value:.6e}, "
                f"initial_rTr={self.poisson_solver.initial_residual:.6e}, "
                f"final_rTr={self.poisson_solver.final_residual:.6e}, "
                f"iterations={self.poisson_solver.last_iterations}"
            )
        self.cell_pressure.copy_from(self.poisson_solver.x)

    def _solve_pressure3d(self, sims: Simulation, scene: myScene):
        self._build_fluid_sdf3d(sims, scene)
        self.poisson_solver.reinitialize(self.cell_type)
        kernel_assemble_double_layer_pressure_rhs3d(
            scene.element.grid_size,
            self.cell_type,
            self.face_porosity_x,
            self.face_porosity_y,
            self.face_porosity_z,
            self.fluid_velocity_x,
            self.fluid_velocity_y,
            self.fluid_velocity_z,
            self.solid_velocity_x,
            self.solid_velocity_y,
            self.solid_velocity_z,
            self.poisson_solver.b,
        )
        kernel_assemble_double_layer_pressure_A3d(
            scene.element.grid_size,
            sims.dt,
            self.poisson_solver.grid_type[0],
            self.face_porosity_x,
            self.face_porosity_y,
            self.face_porosity_z,
            self.cell_solid_density,
            self.cell_fluid_density,
            self.cell_fluid_sdf,
            self.poisson_solver.Adiag[0],
            self.poisson_solver.Ax[0],
        )
        self._assemble_pressure_mg_levels3d(sims, scene)
        solved = self.poisson_solver.solve(self.poisson_iterations, rel_tol=sims.residual_tolerance, abs_tol=1.0e-10)
        if not solved:
            raise RuntimeError(
                "TwoPhaseDoubleLayer pressure solve failed: "
                f"{self.poisson_solver.breakdown_reason}={self.poisson_solver.breakdown_value:.6e}, "
                f"initial_rTr={self.poisson_solver.initial_residual:.6e}, "
                f"final_rTr={self.poisson_solver.final_residual:.6e}, "
                f"iterations={self.poisson_solver.last_iterations}"
            )
        self.cell_pressure.copy_from(self.poisson_solver.x)

    def _sample_pressure_to_solid_particles2d(self, sims: Simulation, scene: myScene):
        kernel_project_double_layer_pressure_to_solid_nodes2d(
            scene.mass_cut_off,
            scene.element.gnum,
            scene.element.grid_size,
            self.mac_shape_type,
            self.mac_influenced_node,
            self.cell_type,
            self.cell_pressure,
            self.cell_fluid_sdf,
            scene.node,
            scene.element.calLength,
        )
        for _, start_index, end_index, _ in self._iter_twophase_materials(scene):
            kernel_sample_double_layer_solid_pressure_from_nodes2d(
                scene.element.grid_nodes,
                start_index,
                end_index,
                scene.mass_cut_off,
                scene.particle,
                scene.material.materialID,
                scene.node,
                scene.element.LnID,
                scene.element.shape_fn,
                scene.element.node_size,
            )

    def _sample_pressure_to_solid_particles3d(self, sims: Simulation, scene: myScene):
        kernel_project_double_layer_pressure_to_solid_nodes3d(
            scene.mass_cut_off,
            scene.element.gnum,
            scene.element.grid_size,
            self.mac_shape_type,
            self.mac_influenced_node,
            self.cell_type,
            self.cell_pressure,
            self.cell_fluid_sdf,
            scene.node,
            scene.element.calLength,
        )
        for _, start_index, end_index, _ in self._iter_twophase_materials(scene):
            kernel_sample_double_layer_solid_pressure_from_nodes3d(
                scene.element.grid_nodes,
                start_index,
                end_index,
                scene.mass_cut_off,
                scene.particle,
                scene.material.materialID,
                scene.node,
                scene.element.LnID,
                scene.element.shape_fn,
                scene.element.node_size,
            )

    def _correct_solid_velocity_paper2d(self, sims: Simulation, scene: myScene):
        for _, start_index, end_index, _ in self._iter_twophase_materials(scene):
            kernel_correct_double_layer_solid_velocity_paper2d(
                scene.element.grid_nodes,
                start_index,
                end_index,
                scene.mass_cut_off,
                sims.dt,
                scene.node,
                scene.particle,
                scene.material.materialID,
                scene.element.LnID,
                scene.element.shape_fn,
                scene.element.dshape_fn,
                scene.element.node_size,
            )

    def _correct_solid_velocity_paper3d(self, sims: Simulation, scene: myScene):
        for _, start_index, end_index, _ in self._iter_twophase_materials(scene):
            kernel_correct_double_layer_solid_velocity_paper3d(
                scene.element.grid_nodes,
                start_index,
                end_index,
                scene.mass_cut_off,
                sims.dt,
                scene.node,
                scene.particle,
                scene.material.materialID,
                scene.element.LnID,
                scene.element.shape_fn,
                scene.element.dshape_fn,
                scene.element.node_size,
            )

    def _build_fluid_sdf2d(self, sims: Simulation, scene: myScene):
        kernel_build_double_layer_fluid_sdf2d(
            scene.element.grid_size,
            self.cell_fluid_mass,
            self.cell_fluid_density,
            self.cell_porosity,
            self.cell_type,
            self.cell_fluid_sdf,
        )

    def _build_fluid_sdf3d(self, sims: Simulation, scene: myScene):
        kernel_build_double_layer_fluid_sdf3d(
            scene.element.grid_size,
            self.cell_fluid_mass,
            self.cell_fluid_density,
            self.cell_porosity,
            self.cell_type,
            self.cell_fluid_sdf,
        )

    def _mark_solid_cell_boundaries2d(self, sims: Simulation, scene: myScene):
        if scene.boundary is None:
            return
        for start_point, end_point in scene.boundary.solid_cell_regions:
            kernel_mark_double_layer_solid_cell_region2d(
                scene.element.grid_size,
                ti.Vector(start_point.tolist()),
                ti.Vector(end_point.tolist()),
                self.cell_type,
            )
        for start_point, end_point, plane_point, plane_normal in getattr(
            scene.boundary, "solid_cell_plane_regions", []
        ):
            kernel_mark_double_layer_solid_plane_region2d(
                scene.element.grid_size,
                ti.Vector(start_point.tolist()),
                ti.Vector(end_point.tolist()),
                ti.Vector(plane_point.tolist()),
                ti.Vector(plane_normal.tolist()),
                self.cell_type,
            )

    def _mark_solid_cell_boundaries3d(self, sims: Simulation, scene: myScene):
        if scene.boundary is None:
            return
        for start_point, end_point in scene.boundary.solid_cell_regions:
            kernel_mark_double_layer_solid_cell_region3d(
                scene.element.grid_size,
                ti.Vector(start_point.tolist()),
                ti.Vector(end_point.tolist()),
                self.cell_type,
            )
        for start_point, end_point, plane_point, plane_normal in getattr(
            scene.boundary, "solid_cell_plane_regions", []
        ):
            kernel_mark_double_layer_solid_plane_region3d(
                scene.element.grid_size,
                ti.Vector(start_point.tolist()),
                ti.Vector(end_point.tolist()),
                ti.Vector(plane_point.tolist()),
                ti.Vector(plane_normal.tolist()),
                self.cell_type,
            )

    def _enforce_solid_cell_faces2d(self, sims: Simulation):
        kernel_enforce_double_layer_solid_cell_faces2d(
            self.cell_type,
            self.fluid_velocity_x,
            self.fluid_velocity_y,
            self.fluid_acceleration_x,
            self.fluid_acceleration_y,
            self.solid_velocity_x,
            self.solid_velocity_y,
        )

    def _enforce_solid_cell_faces3d(self, sims: Simulation):
        kernel_enforce_double_layer_solid_cell_faces3d(
            self.cell_type,
            self.fluid_velocity_x,
            self.fluid_velocity_y,
            self.fluid_velocity_z,
            self.fluid_acceleration_x,
            self.fluid_acceleration_y,
            self.fluid_acceleration_z,
            self.solid_velocity_x,
            self.solid_velocity_y,
            self.solid_velocity_z,
        )

    def _apply_solid_cell_boundaries(self, sims: Simulation, scene: myScene):
        self.mark_solid_cell_boundaries(sims, scene)
        self.enforce_solid_cell_faces(sims)

    def _semi_implicit_update_2d(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid):
        default_mat_prop = self._single_twophase_matprop(scene)
        mac_shape_type = self.mac_shape_type
        mac_influenced_node = self.mac_influenced_node
        use_affine = self.use_affine_projection
        self.advect_double_layer_fluid_particles(sims, scene)
        self.constrain_particles_to_solid_cell_regions(sims, scene)
        self.calculate_interpolation(sims, scene)
        self.reset_double_layer_fields(sims)

        kernel_p2g_double_layer_mass2d(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.element.grid_size,
            scene.element.gnum,
            use_affine,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )
        kernel_normalize_double_layer_nodes2d(scene.mass_cut_off, scene.node)
        if sims.particle_shifting:
            if self._update_double_layer_fluid_volume2d(scene):
                kernel_reset_double_layer_grid(scene.node)
                kernel_p2g_double_layer_mass2d(
                    scene.element.grid_nodes,
                    int(scene.particleNum[0]),
                    scene.element.grid_size,
                    scene.element.gnum,
                    use_affine,
                    scene.node,
                    scene.particle,
                    scene.element.LnID,
                    scene.element.shape_fn,
                    scene.element.node_size,
                )
                kernel_normalize_double_layer_nodes2d(scene.mass_cut_off, scene.node)
        self.apply_velocity_constraints(sims, scene)
        self._apply_solid_plane_node_boundaries2d(scene)
        for _, start_index, end_index, mat_prop in self._iter_twophase_materials(scene):
            kernel_update_double_layer_solid_state2d(
                scene.element.grid_nodes,
                start_index,
                end_index,
                sims.dt,
                mat_prop,
                scene.material.stateVars,
                scene.node,
                scene.particle,
                scene.material.materialID,
                scene.element.LnID,
                scene.element.dshape_fn,
                scene.element.node_size,
            )
        kernel_force_p2g_double_layer2d(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.dshape_fn,
            scene.element.node_size,
        )
        self.apply_particle_traction_constraints(sims, scene)
        self.apply_traction_constraints(sims, scene)
        kernel_mac_p2g_double_layer2d(
            int(scene.particleNum[0]),
            scene.element.grid_size,
            mac_shape_type,
            mac_influenced_node,
            use_affine,
            self.fluid_mass_x,
            self.fluid_mass_y,
            self.fluid_velocity_x,
            self.fluid_velocity_y,
            self.solid_mass_x,
            self.solid_mass_y,
            self.solid_velocity_x,
            self.solid_velocity_y,
            self.face_porosity_x,
            self.face_porosity_y,
            self.cell_fluid_mass,
            self.cell_solid_mass,
            self.cell_porosity,
            self.cell_solid_velocity,
            self.cell_fluid_velocity,
            scene.particle,
            scene.element.calLength,
        )
        self.accumulate_material_fields(sims, scene)
        kernel_normalize_double_layer_mac_fields2d(
            scene.mass_cut_off,
            self.fluid_mass_x,
            self.fluid_mass_y,
            self.fluid_velocity_x,
            self.fluid_velocity_y,
            self.fluid_velocity0_x,
            self.fluid_velocity0_y,
            self.solid_mass_x,
            self.solid_mass_y,
            self.solid_velocity_x,
            self.solid_velocity_y,
            self.face_porosity_x,
            self.face_porosity_y,
            self.cell_type,
            self.cell_fluid_mass,
            self.cell_solid_mass,
            self.cell_porosity,
            self.cell_solid_velocity,
            self.cell_fluid_velocity,
        )
        kernel_classify_double_layer_fluid_cells2d(
            int(scene.particleNum[0]),
            scene.element.grid_size,
            scene.particle,
            self.cell_fluid_mass,
            self.cell_fluid_density,
            self.cell_porosity,
            self.cell_type,
        )
        self._apply_solid_cell_boundaries(sims, scene)

        self.apply_velocity_constraints(sims, scene)
        self._apply_solid_plane_node_boundaries2d(scene)
        kernel_predict_double_layer2d(
            scene.mass_cut_off,
            sims.gravity,
            scene.element.grid_size,
            sims.dt,
            scene.node,
            self.fluid_mass_x,
            self.fluid_mass_y,
            self.fluid_velocity_x,
            self.fluid_velocity_y,
            self.fluid_acceleration_x,
            self.fluid_acceleration_y,
            self.solid_velocity_x,
            self.solid_velocity_y,
            self.face_porosity_x,
            self.face_porosity_y,
            self.face_solid_density_x,
            self.face_solid_density_y,
            self.face_fluid_density_x,
            self.face_fluid_density_y,
            self.face_fluid_viscosity_x,
            self.face_fluid_viscosity_y,
            self.face_grain_diameter_x,
            self.face_grain_diameter_y,
            self.face_permeability_x,
            self.face_permeability_y,
            self.face_fluid_unit_weight_x,
            self.face_fluid_unit_weight_y,
            self.face_drag_model_x,
            self.face_drag_model_y,
            self.node_solid_density,
            self.node_fluid_density,
            self.node_fluid_viscosity,
            self.node_grain_diameter,
            self.node_permeability,
            self.node_fluid_unit_weight,
            self.node_drag_model,
        )
        self.apply_velocity_constraints(sims, scene)
        self._apply_solid_plane_node_boundaries2d(scene)
        kernel_project_solid_grid_velocity_to_mac2d(
            scene.mass_cut_off,
            scene.element.grid_size,
            scene.element.gnum,
            mac_shape_type,
            mac_influenced_node,
            scene.node,
            scene.element.calLength,
            self.solid_mass_x,
            self.solid_mass_y,
            self.solid_velocity_x,
            self.solid_velocity_y,
        )
        self.update_external_mac_boundary(sims, scene)
        self.enforce_solid_cell_faces(sims)

        self.solve_pressure(sims, scene)
        kernel_correct_double_layer_velocity2d(
            scene.mass_cut_off,
            scene.element.grid_size,
            sims.dt,
            self.cell_type,
            self.cell_pressure,
            self.cell_fluid_sdf,
            self.fluid_mass_x,
            self.fluid_mass_y,
            self.fluid_velocity_x,
            self.fluid_velocity_y,
            self.fluid_acceleration_x,
            self.fluid_acceleration_y,
            self.face_fluid_density_x,
            self.face_fluid_density_y,
        )
        self.sample_pressure_to_solid_particles(sims, scene)
        self.correct_solid_velocity_paper(sims, scene)
        self.enforce_solid_cell_faces(sims)
        self.apply_velocity_constraints(sims, scene)
        self._apply_solid_plane_node_boundaries2d(scene)

        for _, start_index, end_index, mat_prop in self._iter_twophase_materials(scene):
            kernel_g2p_double_layer2d(
                scene.element.grid_nodes,
                start_index,
                end_index,
                sims.alphaPIC,
                sims.domain,
                scene.element.grid_size,
                mac_shape_type,
                mac_influenced_node,
                use_affine,
                sims.dt,
                mat_prop,
                scene.material.stateVars,
                scene.node,
                scene.particle,
                scene.material.materialID,
                scene.element.calLength,
                scene.element.LnID,
                scene.element.shape_fn,
                scene.element.dshape_fn,
                scene.element.node_size,
                self.fluid_velocity_x,
                self.fluid_velocity_y,
                self.fluid_acceleration_x,
                self.fluid_acceleration_y,
                self.cell_type,
                self.cell_pressure,
                self.cell_fluid_sdf,
                1,
                0,
                self.delayed_fluid_advection_flag,
            )
        self.shift_double_layer_fluid_particles(sims, scene)
        self.constrain_particles_to_solid_cell_regions(sims, scene)

    def _semi_implicit_update_3d(self, sims: Simulation, scene: myScene, neighbor: SpatialHashGrid = None):
        default_mat_prop = self._single_twophase_matprop(scene)
        mac_shape_type = self.mac_shape_type
        mac_influenced_node = self.mac_influenced_node
        use_affine = self.use_affine_projection
        self.advect_double_layer_fluid_particles(sims, scene)
        self.constrain_particles_to_solid_cell_regions(sims, scene)
        self.calculate_interpolation(sims, scene)
        self.reset_double_layer_fields(sims)

        kernel_p2g_double_layer_mass3d(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.element.grid_size,
            scene.element.gnum,
            use_affine,
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.node_size,
        )
        kernel_normalize_double_layer_nodes3d(scene.mass_cut_off, scene.node)
        if sims.particle_shifting:
            if self._update_double_layer_fluid_volume3d(scene):
                kernel_reset_double_layer_grid(scene.node)
                kernel_p2g_double_layer_mass3d(
                    scene.element.grid_nodes,
                    int(scene.particleNum[0]),
                    scene.element.grid_size,
                    scene.element.gnum,
                    use_affine,
                    scene.node,
                    scene.particle,
                    scene.element.LnID,
                    scene.element.shape_fn,
                    scene.element.node_size,
                )
                kernel_normalize_double_layer_nodes3d(scene.mass_cut_off, scene.node)
        self.apply_velocity_constraints(sims, scene)
        self._apply_solid_plane_node_boundaries3d(scene)
        for _, start_index, end_index, mat_prop in self._iter_twophase_materials(scene):
            kernel_update_double_layer_solid_state3d(
                scene.element.grid_nodes,
                start_index,
                end_index,
                sims.dt,
                mat_prop,
                scene.material.stateVars,
                scene.node,
                scene.particle,
                scene.material.materialID,
                scene.element.LnID,
                scene.element.dshape_fn,
                scene.element.node_size,
            )
        kernel_force_p2g_double_layer3d(
            scene.element.grid_nodes,
            int(scene.particleNum[0]),
            scene.node,
            scene.particle,
            scene.element.LnID,
            scene.element.shape_fn,
            scene.element.dshape_fn,
            scene.element.node_size,
        )
        self.apply_particle_traction_constraints(sims, scene)
        self.apply_traction_constraints(sims, scene)
        kernel_mac_p2g_double_layer3d(
            int(scene.particleNum[0]),
            scene.element.grid_size,
            mac_shape_type,
            mac_influenced_node,
            use_affine,
            self.fluid_mass_x,
            self.fluid_mass_y,
            self.fluid_mass_z,
            self.fluid_velocity_x,
            self.fluid_velocity_y,
            self.fluid_velocity_z,
            self.solid_mass_x,
            self.solid_mass_y,
            self.solid_mass_z,
            self.solid_velocity_x,
            self.solid_velocity_y,
            self.solid_velocity_z,
            self.face_porosity_x,
            self.face_porosity_y,
            self.face_porosity_z,
            self.cell_fluid_mass,
            self.cell_solid_mass,
            self.cell_porosity,
            self.cell_solid_velocity,
            self.cell_fluid_velocity,
            scene.particle,
            scene.element.calLength,
        )
        self.accumulate_material_fields(sims, scene)
        kernel_normalize_double_layer_mac_fields3d(
            scene.mass_cut_off,
            self.fluid_mass_x,
            self.fluid_mass_y,
            self.fluid_mass_z,
            self.fluid_velocity_x,
            self.fluid_velocity_y,
            self.fluid_velocity_z,
            self.fluid_velocity0_x,
            self.fluid_velocity0_y,
            self.fluid_velocity0_z,
            self.solid_mass_x,
            self.solid_mass_y,
            self.solid_mass_z,
            self.solid_velocity_x,
            self.solid_velocity_y,
            self.solid_velocity_z,
            self.face_porosity_x,
            self.face_porosity_y,
            self.face_porosity_z,
            self.cell_type,
            self.cell_fluid_mass,
            self.cell_solid_mass,
            self.cell_porosity,
            self.cell_solid_velocity,
            self.cell_fluid_velocity,
        )
        kernel_classify_double_layer_fluid_cells3d(
            int(scene.particleNum[0]),
            scene.element.grid_size,
            scene.particle,
            self.cell_fluid_mass,
            self.cell_fluid_density,
            self.cell_porosity,
            self.cell_type,
        )
        self._apply_solid_cell_boundaries(sims, scene)

        self.apply_velocity_constraints(sims, scene)
        self._apply_solid_plane_node_boundaries3d(scene)
        kernel_predict_double_layer3d(
            scene.mass_cut_off,
            sims.gravity,
            scene.element.grid_size,
            sims.dt,
            scene.node,
            self.fluid_mass_x,
            self.fluid_mass_y,
            self.fluid_mass_z,
            self.fluid_velocity_x,
            self.fluid_velocity_y,
            self.fluid_velocity_z,
            self.fluid_acceleration_x,
            self.fluid_acceleration_y,
            self.fluid_acceleration_z,
            self.solid_velocity_x,
            self.solid_velocity_y,
            self.solid_velocity_z,
            self.face_porosity_x,
            self.face_porosity_y,
            self.face_porosity_z,
            self.face_solid_density_x,
            self.face_solid_density_y,
            self.face_solid_density_z,
            self.face_fluid_density_x,
            self.face_fluid_density_y,
            self.face_fluid_density_z,
            self.face_fluid_viscosity_x,
            self.face_fluid_viscosity_y,
            self.face_fluid_viscosity_z,
            self.face_grain_diameter_x,
            self.face_grain_diameter_y,
            self.face_grain_diameter_z,
            self.face_permeability_x,
            self.face_permeability_y,
            self.face_permeability_z,
            self.face_fluid_unit_weight_x,
            self.face_fluid_unit_weight_y,
            self.face_fluid_unit_weight_z,
            self.face_drag_model_x,
            self.face_drag_model_y,
            self.face_drag_model_z,
            self.node_solid_density,
            self.node_fluid_density,
            self.node_fluid_viscosity,
            self.node_grain_diameter,
            self.node_permeability,
            self.node_fluid_unit_weight,
            self.node_drag_model,
        )
        self.apply_velocity_constraints(sims, scene)
        self._apply_solid_plane_node_boundaries3d(scene)
        kernel_project_solid_grid_velocity_to_mac3d(
            scene.mass_cut_off,
            scene.element.grid_size,
            scene.element.gnum,
            mac_shape_type,
            mac_influenced_node,
            scene.node,
            scene.element.calLength,
            self.solid_mass_x,
            self.solid_mass_y,
            self.solid_mass_z,
            self.solid_velocity_x,
            self.solid_velocity_y,
            self.solid_velocity_z,
        )
        self.update_external_mac_boundary(sims, scene)
        self.enforce_solid_cell_faces(sims)

        self.solve_pressure(sims, scene)
        kernel_correct_double_layer_velocity3d(
            scene.mass_cut_off,
            scene.element.grid_size,
            sims.dt,
            self.cell_type,
            self.cell_pressure,
            self.cell_fluid_sdf,
            self.fluid_mass_x,
            self.fluid_mass_y,
            self.fluid_mass_z,
            self.fluid_velocity_x,
            self.fluid_velocity_y,
            self.fluid_velocity_z,
            self.fluid_acceleration_x,
            self.fluid_acceleration_y,
            self.fluid_acceleration_z,
            self.face_fluid_density_x,
            self.face_fluid_density_y,
            self.face_fluid_density_z,
        )
        self.enforce_external_mac_boundary_after_pressure(sims, scene)
        self.sample_pressure_to_solid_particles(sims, scene)
        self.correct_solid_velocity_paper(sims, scene)
        self.enforce_solid_cell_faces(sims)
        self.apply_velocity_constraints(sims, scene)
        self._apply_solid_plane_node_boundaries3d(scene)

        for _, start_index, end_index, mat_prop in self._iter_twophase_materials(scene):
            kernel_g2p_double_layer3d(
                scene.element.grid_nodes,
                start_index,
                end_index,
                sims.alphaPIC,
                sims.domain,
                scene.element.grid_size,
                mac_shape_type,
                mac_influenced_node,
                use_affine,
                sims.dt,
                mat_prop,
                scene.material.stateVars,
                scene.node,
                scene.particle,
                scene.material.materialID,
                scene.element.calLength,
                scene.element.LnID,
                scene.element.shape_fn,
                scene.element.dshape_fn,
                scene.element.node_size,
                self.fluid_velocity_x,
                self.fluid_velocity_y,
                self.fluid_velocity_z,
                self.fluid_acceleration_x,
                self.fluid_acceleration_y,
                self.fluid_acceleration_z,
                self.cell_type,
                self.cell_pressure,
                self.cell_fluid_sdf,
                1,
                0,
                self.delayed_fluid_advection_flag,
            )
        self.shift_double_layer_fluid_particles(sims, scene)
        self.constrain_particles_to_solid_cell_regions(sims, scene)

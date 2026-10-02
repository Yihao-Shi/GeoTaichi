import taichi as ti

from src.mpdem.fluid_dynamics.IncompressibleCouplingKernel import (
    cell_center_velocity_3d,
    cell_pressure_gradient_3d,
    cell_velocity_laplacian_3d,
    clamp_cell_index_3d,
    estimate_cell_solid_fraction_3d,
    kernel_accumulate_double_layer_lsdem_ibm_force,
    kernel_accumulate_lsdem_volume_fraction_ibm_force,
    kernel_apply_double_layer_lsdem_ibm,
    kernel_update_lsdem_cell_ibm_fields,
    lsdem_volume_fraction_ibm_force_density,
    min_grid_spacing_3d,
    partitioned_body_share,
    sample_cell_center_velocity_3d,
)
from src.utils.constants import Threshold


class IncompressibleLSDEMCoupling:
    def __init__(self, msims, dsims, mscene, dscene, mengine):
        if msims.dimension != 3:
            raise RuntimeError("Incompressible LSDEM-MPM coupling currently supports 3D FDM grids only")
        self.msims = msims
        self.dsims = dsims
        self.mscene = mscene
        self.dscene = dscene
        self.mengine = mengine
        self.solid_fraction = None
        self.solid_fraction_sum = None
        self.solid_density = None
        self.solid_velocity_cell = None
        self.ibm_force_cell = None

    def attach(self):
        self.msims.set_fluid_level_set(True)
        if self.solid_fraction is None:
            self.solid_fraction = ti.field(
                dtype=float,
                shape=self.mscene.element.cnum,
                offset=0 * self.mscene.element.cnum - self.mscene.element.ghost_cell,
            )
        if self.solid_fraction_sum is None:
            self.solid_fraction_sum = ti.field(
                dtype=float,
                shape=self.mscene.element.cnum,
                offset=0 * self.mscene.element.cnum - self.mscene.element.ghost_cell,
            )
        if self.solid_density is None:
            self.solid_density = ti.field(
                dtype=float,
                shape=self.mscene.element.cnum,
                offset=0 * self.mscene.element.cnum - self.mscene.element.ghost_cell,
            )
        if self.solid_velocity_cell is None:
            self.solid_velocity_cell = ti.Vector.field(
                3,
                dtype=float,
                shape=self.mscene.element.cnum,
                offset=0 * self.mscene.element.cnum - self.mscene.element.ghost_cell,
            )
        if self.ibm_force_cell is None:
            self.ibm_force_cell = ti.Vector.field(
                3,
                dtype=float,
                shape=self.mscene.element.cnum,
                offset=0 * self.mscene.element.cnum - self.mscene.element.ghost_cell,
            )
        self.mengine.ibm_solid_fraction = self.solid_fraction
        self.mengine.ibm_solid_density = self.solid_density
        self.mengine.ibm_solid_velocity_cell = self.solid_velocity_cell
        self.mengine.ibm_force_cell = self.ibm_force_cell
        self.mengine.pressure_coupling_mode = 2
        self.mengine.pressure_solid_fraction = self.solid_fraction
        self.mengine.pressure_previous_solid_fraction = self.solid_fraction
        self.mengine.pressure_solid_density = self.solid_density
        self.mengine.update_external_ibm_fields = self.update_solid_cells
        if hasattr(self.mengine, "configure_runtime_functions"):
            self.mengine.configure_runtime_functions(self.msims, self.mscene)

    def update_solid_cells(self, sims, scene):
        rigid_num = int(self.dscene.rigidNum[0])
        if rigid_num <= 0:
            return
        if (
            self.dscene.rigid is None
            or self.dscene.box is None
            or self.dscene.rigid_grid is None
            or self.dscene.particle is None
        ):
            raise RuntimeError(
                "LSDEM incompressible coupling requires rigid, bounding box, bounding sphere and level-set grid fields"
            )

        kernel_update_lsdem_cell_ibm_fields(
            rigid_num,
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.grid_size,
            scene.element.igrid_size,
            self.dscene.particle,
            self.dscene.rigid,
            self.dscene.box,
            self.dscene.rigid_grid,
            self.solid_fraction,
            self.solid_fraction_sum,
            self.solid_density,
            self.solid_velocity_cell,
        )

    def accumulate_pressure_force(self, sims, scene):
        rigid_num = int(self.dscene.rigidNum[0])
        if rigid_num <= 0:
            return
        mat_props = self.mengine.get_single_fluid_mat_props(scene)
        kernel_accumulate_lsdem_volume_fraction_ibm_force(
            rigid_num,
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.grid_size,
            scene.element.igrid_size,
            self.dscene.particle,
            self.dscene.rigid,
            self.dscene.box,
            self.dscene.rigid_grid,
            scene.node,
            mat_props,
            self.solid_fraction,
            self.solid_fraction_sum,
            self.solid_density,
            self.ibm_force_cell,
            scene.element.cell.pressure,
            scene.element.cell.surface_tension,
            scene.element.cell.type,
            scene.element.cell.fluid_sdf,
            self.mengine.use_pressure_free_surface_theta,
        )


class TwoPhaseDoubleLayerLSDEMCoupling:
    """Hybrid LSDEM coupling: fluid IBM plus solid material-point contact."""

    def __init__(self, msims, dsims, mscene, dscene, mengine):
        if msims.dimension != 3:
            raise RuntimeError("TwoPhaseDoubleLayer LSDEM coupling currently supports 3D only")
        self.msims = msims
        self.dsims = dsims
        self.mscene = mscene
        self.dscene = dscene
        self.mengine = mengine
        self.external_mac_boundary = mengine.update_external_mac_boundary
        shape = mscene.element.cnum
        self.solid_fraction = ti.field(float, shape=shape)
        self.solid_fraction_sum = ti.field(float, shape=shape)
        self.solid_density = ti.field(float, shape=shape)
        self.solid_velocity_cell = ti.Vector.field(3, float, shape=shape)
        self.ibm_reaction_cell = ti.Vector.field(3, float, shape=shape)
        self.zero_surface_tension = ti.field(float, shape=shape)

    def attach(self):
        self.mengine.update_external_mac_boundary = self.apply_before_pressure
        self.mengine.enforce_external_mac_boundary_after_pressure = self.apply_after_pressure

    def update_solid_cells(self, scene):
        rigid_num = int(self.dscene.rigidNum[0])
        if rigid_num <= 0:
            return False
        if self.dscene.rigid is None or self.dscene.box is None or self.dscene.rigid_grid is None:
            raise RuntimeError("TwoPhaseDoubleLayer IBM requires LSDEM rigid, box and level-set grid fields")
        kernel_update_lsdem_cell_ibm_fields(
            rigid_num,
            0,
            scene.element.cnum,
            scene.element.grid_size,
            scene.element.igrid_size,
            self.dscene.particle,
            self.dscene.rigid,
            self.dscene.box,
            self.dscene.rigid_grid,
            self.solid_fraction,
            self.solid_fraction_sum,
            self.solid_density,
            self.solid_velocity_cell,
        )
        return True

    def apply_before_pressure(self, sims, scene):
        self.external_mac_boundary(sims, scene)
        if not self.update_solid_cells(scene):
            return
        self.ibm_reaction_cell.fill(0.0)
        self._apply_ibm(sims, scene)

    def apply_after_pressure(self, sims, scene):
        if int(self.dscene.rigidNum[0]) > 0:
            self._apply_ibm(sims, scene)

    def _apply_ibm(self, sims, scene):
        kernel_apply_double_layer_lsdem_ibm(
            scene.mass_cut_off,
            sims.dt,
            self.solid_fraction,
            self.solid_velocity_cell,
            self.mengine.fluid_mass_x,
            self.mengine.fluid_mass_y,
            self.mengine.fluid_mass_z,
            self.mengine.fluid_velocity_x,
            self.mengine.fluid_velocity_y,
            self.mengine.fluid_velocity_z,
            self.mengine.fluid_acceleration_x,
            self.mengine.fluid_acceleration_y,
            self.mengine.fluid_acceleration_z,
            self.ibm_reaction_cell,
        )

    def accumulate_pressure_force(self, sims, scene):
        rigid_num = int(self.dscene.rigidNum[0])
        if rigid_num <= 0:
            return
        kernel_accumulate_double_layer_lsdem_ibm_force(
            rigid_num,
            scene.element.cnum,
            scene.element.grid_size,
            scene.element.igrid_size,
            self.dscene.particle,
            self.dscene.rigid,
            self.dscene.box,
            self.dscene.rigid_grid,
            self.solid_fraction,
            self.solid_fraction_sum,
            self.solid_density,
            self.ibm_reaction_cell,
            self.mengine.cell_pressure,
            self.zero_surface_tension,
            self.mengine.cell_type,
            self.mengine.cell_fluid_sdf,
            self.mengine.cell_fluid_density,
            self.mengine.cell_fluid_viscosity,
            self.mengine.fluid_velocity_x,
            self.mengine.fluid_velocity_y,
            self.mengine.fluid_velocity_z,
        )


@ti.data_oriented
class IncompressibleAffineBodyCoupling:
    """Partitioned volume-fraction IBM coupling for LevelSet affine bodies."""

    def __init__(self, msims, mscene, mengine, affine_engine):
        if msims.dimension != 3:
            raise RuntimeError("Incompressible MPM--AffineBody coupling currently supports 3D FDM grids only")
        self.msims = msims
        self.mscene = mscene
        self.mengine = mengine
        self.affine_engine = affine_engine
        self.affine = affine_engine.operator
        if self.affine is None or not self.affine.levelset_contact:
            raise RuntimeError("Incompressible MPM--AffineBody coupling requires LevelSet affine-body templates")
        self.body_num = int(self.affine.body_num)
        self.body_bbox_min = ti.Vector.field(3, dtype=float, shape=max(self.body_num, 1))
        self.body_bbox_max = ti.Vector.field(3, dtype=float, shape=max(self.body_num, 1))
        self.solid_fraction = None
        self.solid_fraction_sum = None
        self.solid_density = None
        self.solid_velocity_cell = None
        self.ibm_force_cell = None

    def attach(self):
        self.msims.set_fluid_level_set(True)
        offset = 0 * self.mscene.element.cnum - self.mscene.element.ghost_cell
        if self.solid_fraction is None:
            self.solid_fraction = ti.field(dtype=float, shape=self.mscene.element.cnum, offset=offset)
            self.solid_fraction_sum = ti.field(dtype=float, shape=self.mscene.element.cnum, offset=offset)
            self.solid_density = ti.field(dtype=float, shape=self.mscene.element.cnum, offset=offset)
            self.solid_velocity_cell = ti.Vector.field(3, dtype=float, shape=self.mscene.element.cnum, offset=offset)
            self.ibm_force_cell = ti.Vector.field(3, dtype=float, shape=self.mscene.element.cnum, offset=offset)
        self.mengine.ibm_solid_fraction = self.solid_fraction
        self.mengine.ibm_solid_density = self.solid_density
        self.mengine.ibm_solid_velocity_cell = self.solid_velocity_cell
        self.mengine.ibm_force_cell = self.ibm_force_cell
        self.mengine.pressure_coupling_mode = 2
        self.mengine.pressure_solid_fraction = self.solid_fraction
        self.mengine.pressure_previous_solid_fraction = self.solid_fraction
        self.mengine.pressure_solid_density = self.solid_density
        self.mengine.update_external_ibm_fields = self.update_solid_cells
        self.mengine.configure_runtime_functions(self.msims, self.mscene)

    @ti.func
    def _material_coordinate(self, body, position):
        y0 = self.affine.y[4 * body]
        frame = self.affine._affine_levelset_matrix(body)
        return frame.inverse() @ (position - y0)

    @ti.func
    def _cell_intersects_body(self, cell, grid_size, body):
        intersects = True
        lower = cell.cast(float) * grid_size
        upper = (cell.cast(float) + 1.0) * grid_size
        margin = min_grid_spacing_3d(grid_size)
        for d in ti.static(range(3)):
            intersects = (
                intersects
                and upper[d] >= self.body_bbox_min[body][d] - margin
                and lower[d] <= self.body_bbox_max[body][d] + margin
            )
        return intersects

    @ti.func
    def _cell_body_fraction(self, cell, grid_size, body):
        negative = 0.0
        magnitude = 0.0
        offset = ti.Vector([0.25, 0.25, 0.25])
        scale = ti.max(self.affine.body_scale[body], 1.0e-12)
        for ox, oy, oz in ti.static(ti.ndrange(2, 2, 2)):
            position = (cell.cast(float) + ti.Vector([ox, oy, oz]).cast(float)) * grid_size
            material = self._material_coordinate(body, position)
            template_point = (material - offset) / scale
            phi, unused_gradient, unused_hessian, inside = self.affine._sample_affine_levelset(body, template_point)
            distance = scale * phi
            if not inside:
                distance = 1.0e6 * min_grid_spacing_3d(grid_size)
            magnitude += ti.abs(distance)
            if distance < 0.0:
                negative += -distance
        fraction = 0.0
        if magnitude > 1.0e-12:
            fraction = ti.min(1.0, ti.max(0.0, negative / magnitude))
        return fraction

    @ti.kernel
    def _reset_body_bounds(self):
        for body in range(self.body_num):
            self.body_bbox_min[body] = ti.Vector([1.0e30, 1.0e30, 1.0e30])
            self.body_bbox_max[body] = ti.Vector([-1.0e30, -1.0e30, -1.0e30])

    @ti.kernel
    def _build_body_bounds(self):
        for vertex in range(self.affine.vertex_num):
            body = self.affine.node2body[vertex]
            for d in ti.static(range(3)):
                ti.atomic_min(self.body_bbox_min[body][d], self.affine.x[vertex][d])
                ti.atomic_max(self.body_bbox_max[body][d], self.affine.x[vertex][d])

    @ti.kernel
    def _update_cell_fields(self, grid_size: ti.types.vector(3, float)):
        # ponytail: the inner scan targets small robot/link counts; add a
        # device body-to-cell bin only when many-body IBM becomes a workload.
        for I in ti.grouped(self.solid_fraction):
            fraction_sum = 0.0
            density_sum = 0.0
            velocity_sum = ti.Vector.zero(float, 3)
            position = (I.cast(float) + 0.5) * grid_size
            for body in range(self.body_num):
                if self._cell_intersects_body(I, grid_size, body):
                    fraction = self._cell_body_fraction(I, grid_size, body)
                    if fraction > 1.0e-12:
                        density = self.affine.body_mass[body] / ti.max(self.affine.volume[body], 1.0e-12)
                        material = self._material_coordinate(body, position)
                        weights = ti.Vector([1.0 - material.sum(), material[0], material[1], material[2]])
                        velocity = ti.Vector.zero(float, 3)
                        for control in ti.static(range(4)):
                            velocity += weights[control] * self.affine.velocity_y[4 * body + control]
                        fraction_sum += fraction
                        density_sum += fraction * density
                        velocity_sum += fraction * velocity
            if fraction_sum > 1.0e-12:
                self.solid_fraction[I] = ti.min(1.0, fraction_sum)
                self.solid_fraction_sum[I] = fraction_sum
                self.solid_density[I] = density_sum / fraction_sum
                self.solid_velocity_cell[I] = velocity_sum / fraction_sum
            else:
                self.solid_fraction[I] = 0.0
                self.solid_fraction_sum[I] = 0.0
                self.solid_density[I] = 0.0
                self.solid_velocity_cell[I] = ti.Vector.zero(float, 3)

    @ti.kernel
    def _accumulate_generalized_force(
        self,
        ghost_cell: int,
        cnum: ti.types.vector(3, int),
        grid_size: ti.types.vector(3, float),
        node: ti.template(),
        mat_props: ti.template(),
        pressure: ti.template(),
        surface_tension: ti.template(),
        cell_type: ti.template(),
        fluid_sdf: ti.template(),
        use_free_surface_theta: ti.template(),
    ):
        active_cnum = cnum - 2 * ghost_cell
        cell_volume = grid_size[0] * grid_size[1] * grid_size[2]
        for I in ti.grouped(ti.ndrange(active_cnum[0], active_cnum[1], active_cnum[2])):
            total_fraction = ti.min(1.0, ti.max(0.0, self.solid_fraction[I]))
            if total_fraction > Threshold and int(cell_type[I]) == 1:
                density = self.solid_density[I]
                if density <= Threshold:
                    density = mat_props.density
                stress_divergence = -cell_pressure_gradient_3d(
                    I,
                    active_cnum,
                    grid_size,
                    mat_props.atmospheric_pressure,
                    pressure,
                    surface_tension,
                    cell_type,
                    fluid_sdf,
                    use_free_surface_theta,
                ) + mat_props.viscosity * cell_velocity_laplacian_3d(I, active_cnum, grid_size, node)
                force_density = lsdem_volume_fraction_ibm_force_density(
                    total_fraction,
                    mat_props.density,
                    density,
                    stress_divergence,
                    self.ibm_force_cell[I],
                )
                position = (I.cast(float) + 0.5) * grid_size
                for body in range(self.body_num):
                    if self._cell_intersects_body(I, grid_size, body):
                        fraction = self._cell_body_fraction(I, grid_size, body)
                        if fraction > 1.0e-6:
                            force = (
                                partitioned_body_share(fraction, self.solid_fraction_sum[I])
                                * force_density
                                * cell_volume
                            )
                            material = self._material_coordinate(body, position)
                            weights = ti.Vector([1.0 - material.sum(), material[0], material[1], material[2]])
                            for control in ti.static(range(4)):
                                for d in ti.static(range(3)):
                                    ti.atomic_add(
                                        self.affine.external_generalized_force[4 * body + control][d],
                                        weights[control] * force[d],
                                    )

    def update_solid_cells(self, sims, scene):
        del sims
        self.affine._reconstruct_vertices()
        self._reset_body_bounds()
        self._build_body_bounds()
        self._update_cell_fields(scene.element.grid_size)

    def accumulate_pressure_force(self, sims, scene):
        del sims
        self.affine.device_clear_external_generalized_force()
        self._accumulate_generalized_force(
            scene.element.ghost_cell,
            scene.element.cnum,
            scene.element.grid_size,
            scene.node,
            self.mengine.get_single_fluid_mat_props(scene),
            scene.element.cell.pressure,
            scene.element.cell.surface_tension,
            scene.element.cell.type,
            scene.element.cell.fluid_sdf,
            self.mengine.use_pressure_free_surface_theta,
        )

from src.utils.errors import SimulationStateError

import taichi as ti
from src.mpdem.fluid_dynamics.IncompressibleCouplingKernel import (
    kernel_accumulate_double_layer_lsdem_ibm_force,
    kernel_apply_double_layer_lsdem_ibm,
    kernel_apply_double_layer_lsdem_ibm_pressure_predictor,
    kernel_initialize_double_layer_ibm_pressure,
    kernel_update_lsdem_cell_ibm_fields,
)


class TwoPhaseDoubleLayerLSDEMCoupling:
    """Hybrid LSDEM coupling: fluid IBM plus solid material-point contact."""

    def __init__(self, msims, dsims, mscene, dscene, mengine):
        if msims.dimension != 3:
            raise SimulationStateError("TwoPhaseDoubleLayer LSDEM coupling currently supports 3D only")
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
        self.previous_pressure = ti.field(float, shape=shape)
        self.initial_pressure_weight = ti.field(float, shape=shape)
        self.pressure_initialized = False

    def attach(self):
        self.mengine.update_external_mac_boundary = self.apply_before_pressure
        self.mengine.enforce_external_mac_boundary_after_pressure = self.apply_after_pressure

    def update_solid_cells(self, scene):
        rigid_num = int(self.dscene.rigidNum[0])
        if rigid_num <= 0:
            return False
        if self.dscene.rigid is None or self.dscene.box is None or self.dscene.rigid_grid is None:
            raise SimulationStateError("TwoPhaseDoubleLayer IBM requires LSDEM rigid, box and level-set grid fields")
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
        if not self.pressure_initialized:
            if scene.restart_cell_pressure is not None:
                self.previous_pressure.from_numpy(scene.restart_cell_pressure)
                scene.restart_cell_pressure = None
            else:
                kernel_initialize_double_layer_ibm_pressure(
                    int(scene.particleNum[0]),
                    scene.element.grid_size,
                    scene.particle,
                    self.previous_pressure,
                    self.initial_pressure_weight,
                )
            self.pressure_initialized = True
        # Use the same current free-surface geometry as the pressure projection.
        self.mengine._build_fluid_sdf3d(sims, scene)
        self.ibm_reaction_cell.fill(0.0)
        kernel_apply_double_layer_lsdem_ibm_pressure_predictor(
            scene.mass_cut_off,
            sims.dt,
            scene.element.grid_size,
            self.solid_fraction,
            self.solid_velocity_cell,
            (self.mengine.fluid_mass_x, self.mengine.fluid_mass_y, self.mengine.fluid_mass_z),
            (self.mengine.fluid_velocity_x, self.mengine.fluid_velocity_y, self.mengine.fluid_velocity_z),
            (self.mengine.fluid_acceleration_x, self.mengine.fluid_acceleration_y, self.mengine.fluid_acceleration_z),
            self.ibm_reaction_cell,
            self.previous_pressure,
            self.mengine.cell_type,
            self.mengine.cell_fluid_sdf,
            (self.mengine.face_fluid_density_x, self.mengine.face_fluid_density_y, self.mengine.face_fluid_density_z),
            self.mengine.fluid_pressure_mobility,
        )

    def apply_after_pressure(self, sims, scene):
        if int(self.dscene.rigidNum[0]) > 0:
            self._apply_ibm(sims, scene)
            self.previous_pressure.copy_from(self.mengine.cell_pressure)

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
            self.dscene.particle,
            self.dscene.rigid,
            self.dscene.box,
            self.dscene.rigid_grid,
            self.solid_fraction,
            self.solid_fraction_sum,
            self.ibm_reaction_cell,
            scene.mass_cut_off,
            sims.gravity,
            (self.mengine.fluid_mass_x, self.mengine.fluid_mass_y, self.mengine.fluid_mass_z),
            (self.mengine.fluid_acceleration_x, self.mengine.fluid_acceleration_y, self.mengine.fluid_acceleration_z),
        )

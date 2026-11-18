from src.utils.errors import SimulationStateError

import taichi as ti
from src.mpdem.fluid_dynamics.IncompressibleCouplingKernel import (
    kernel_accumulate_lsdem_volume_fraction_ibm_force,
    kernel_update_lsdem_cell_ibm_fields,
)


class IncompressibleLSDEMCoupling:
    def __init__(self, msims, dsims, mscene, dscene, mengine):
        if msims.dimension != 3:
            raise SimulationStateError("Incompressible LSDEM-MPM coupling currently supports 3D FDM grids only")
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
            raise SimulationStateError(
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
            float(sims.dt[None]),
            self.mengine.coupled_viscous_delta[0],
            self.mengine.coupled_viscous_delta[1],
            self.mengine.coupled_viscous_delta[2],
        )

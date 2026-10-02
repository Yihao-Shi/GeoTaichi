import math

import taichi as ti

from src.mpdem.fluid_dynamics.DragForceModel import DragForce
from src.mpdem.fluid_dynamics.IncompressibleCoupling import cell_center_velocity_3d
from src.utils.ObjectIO import DictIO
from src.utils.ShapeFunctions import Guassian
from src.utils.constants import PI, Threshold


from src.mpdem.fluid_dynamics.IncompressibleSemiResolvedKernel import (
    expanded_domain_gaussian,
    clamp_index_3d,
    clamp_active_cell_3d,
    sample_cell_solid_fraction_3d,
    face_fluid_fraction_3d,
    sample_pressure_3d,
    pressure_gradient_3d,
    cell_center_3d,
    sphere_support_bounds,
    kernel_update_sphere_cell_solid_fraction,
    kernel_compute_incompressible_sphere_drag,
    kernel_apply_cell_drag_force_to_mac_velocity,
    kernel_accumulate_incompressible_sphere_pressure_force,
    kernel_apply_integrated_added_mass,
    kernel_apply_sphere_plane_wall_lubrication,
)


class IncompressibleDEMSphereCoupling:
    def __init__(self, sims, msims, dsims, mscene, dscene, mengine, drag_model):
        if msims.dimension != 3:
            raise RuntimeError("Incompressible DEM-sphere semi-resolved coupling currently supports 3D FDM grids only")
        if msims.discretization != "FDM":
            raise RuntimeError("Incompressible DEM-sphere semi-resolved coupling requires FDM incompressible MPM")
        self.sims = sims
        self.msims = msims
        self.dsims = dsims
        self.mscene = mscene
        self.dscene = dscene
        self.mengine = mengine
        self.drag_model = DragForce(drag_model)
        self.added_mass_coefficient = float(DictIO.GetAlternative(drag_model, "AddedMassCoefficient", 0.0))
        self.wall_lubrication_cutoff = float(DictIO.GetAlternative(drag_model, "WallLubricationCutoff", 0.0))
        self.wall_lubrication_minimum_gap = float(DictIO.GetAlternative(drag_model, "WallLubricationMinimumGap", 0.0))
        if not math.isfinite(self.added_mass_coefficient) or self.added_mass_coefficient < 0.0:
            raise ValueError("AddedMassCoefficient must be non-negative")
        if not math.isfinite(self.wall_lubrication_cutoff) or self.wall_lubrication_cutoff < 0.0:
            raise ValueError("WallLubricationCutoff must be finite and non-negative")
        if self.wall_lubrication_cutoff > 0.0:
            if (
                not math.isfinite(self.wall_lubrication_minimum_gap)
                or self.wall_lubrication_minimum_gap <= 0.0
                or self.wall_lubrication_minimum_gap >= self.wall_lubrication_cutoff
            ):
                raise ValueError("WallLubricationMinimumGap must lie strictly between zero and WallLubricationCutoff")
            if self.dsims.wall_type != 0:
                raise RuntimeError("Wall lubrication currently requires DEM Plane walls")
        self.cell_solid_fraction = None
        self.previous_cell_solid_fraction = None
        self.cell_drag_force = None
        self.fluid_velocity = None
        self.fluid_fraction = None
        self.fluid_volume = None

    def build_essential_field(self):
        cell_shape = self.mscene.element.cnum
        cell_offset = 0 * self.mscene.element.cnum - self.mscene.element.ghost_cell
        self.cell_solid_fraction = ti.field(dtype=float, shape=cell_shape, offset=cell_offset)
        self.previous_cell_solid_fraction = ti.field(dtype=float, shape=cell_shape, offset=cell_offset)
        self.cell_drag_force = ti.Vector.field(3, dtype=float, shape=cell_shape, offset=cell_offset)
        self.fluid_velocity = ti.Vector.field(3, dtype=float, shape=self.dsims.max_particle_num)
        self.fluid_fraction = ti.field(dtype=float, shape=self.dsims.max_particle_num)
        self.fluid_volume = ti.field(dtype=float, shape=self.dsims.max_particle_num)

    def attach(self):
        if self.cell_solid_fraction is None:
            self.build_essential_field()
        self.mengine.pressure_coupling_mode = 1
        self.mengine.pressure_solid_fraction = self.cell_solid_fraction
        self.mengine.pressure_previous_solid_fraction = self.previous_cell_solid_fraction
        self.mengine.pressure_solid_density = self.cell_solid_fraction
        self.mengine.run_external_fluid_particle_coupling = self.apply_drag_source
        self.mengine.configure_runtime_functions(self.msims, self.mscene)

    def pre_compute(self):
        active_cnum = self.mscene.element.cnum - 2 * self.mscene.element.ghost_cell
        kernel_update_sphere_cell_solid_fraction(
            int(self.dscene.sphereNum[0]),
            self.sims.infludence_domain,
            active_cnum,
            self.mscene.element.grid_size,
            self.mscene.element.igrid_size,
            self.dscene.particle,
            self.dscene.sphere,
            self.cell_solid_fraction,
            self.fluid_volume,
        )
        self.previous_cell_solid_fraction.copy_from(self.cell_solid_fraction)

    def apply_drag_source(self, sims, scene):
        active_cnum = scene.element.cnum - 2 * scene.element.ghost_cell
        self.previous_cell_solid_fraction.copy_from(self.cell_solid_fraction)
        kernel_update_sphere_cell_solid_fraction(
            int(self.dscene.sphereNum[0]),
            self.sims.infludence_domain,
            active_cnum,
            scene.element.grid_size,
            scene.element.igrid_size,
            self.dscene.particle,
            self.dscene.sphere,
            self.cell_solid_fraction,
            self.fluid_volume,
        )
        kernel_compute_incompressible_sphere_drag(
            int(self.dscene.sphereNum[0]),
            self.sims.dependent_domain,
            self.sims.infludence_domain,
            active_cnum,
            scene.element.grid_size,
            scene.element.igrid_size,
            scene.material.matProps[1],
            self.dscene.particle,
            self.dscene.sphere,
            scene.node,
            self.cell_solid_fraction,
            self.cell_drag_force,
            self.fluid_velocity,
            self.fluid_fraction,
            self.fluid_volume,
            self.drag_model,
        )
        kernel_apply_cell_drag_force_to_mac_velocity(
            scene.mass_cut_off,
            sims.dt,
            active_cnum,
            scene.node,
            self.cell_solid_fraction,
            self.cell_drag_force,
        )

    def accumulate_pressure_force(self, sims, scene):
        active_cnum = scene.element.cnum - 2 * scene.element.ghost_cell
        kernel_accumulate_incompressible_sphere_pressure_force(
            int(self.dscene.sphereNum[0]),
            self.sims.dependent_domain,
            active_cnum,
            scene.element.grid_size,
            scene.element.igrid_size,
            self.dscene.particle,
            self.dscene.sphere,
            scene.element.cell.pressure,
            scene.element.cell.type,
            scene.material.matProps[1].atmospheric_pressure,
        )

    def transform_translational_load(self):
        if self.wall_lubrication_cutoff > 0.0:
            wall_num = int(self.dscene.wallNum[0])
            if wall_num <= 0:
                raise RuntimeError("Wall lubrication requires at least one active DEM Plane wall")
            kernel_apply_sphere_plane_wall_lubrication(
                int(self.dscene.sphereNum[0]),
                wall_num,
                self.mscene.material.matProps[1].viscosity,
                self.wall_lubrication_cutoff,
                self.wall_lubrication_minimum_gap,
                self.dscene.particle,
                self.dscene.sphere,
                self.dscene.wall,
            )
        if self.added_mass_coefficient == 0.0:
            return
        kernel_apply_integrated_added_mass(
            int(self.dscene.sphereNum[0]),
            self.added_mass_coefficient,
            self.mscene.material.matProps[1].density,
            self.dsims.gravity,
            self.dscene.particle,
            self.dscene.sphere,
        )

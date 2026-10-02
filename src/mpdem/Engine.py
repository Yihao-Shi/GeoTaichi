import math

import taichi as ti

from src.dem.BaseKernel import kernel_adaptive_timestep
from src.dem.neighbor.LinkedCell import LinkedCell
from src.dem.engines.ExplicitEngine import ExplicitEngine as DEMExplicitEngine
from src.dem.engines.EngineKernel import (
    get_contact_stiffness_,
    get_wall_contact_force_,
    cache_dem_external_load,
    restore_dem_external_load,
)
from src.dem.SceneManager import myScene as DEMScene
from src.dem.Simulation import Simulation as DEMSimulation
from src.mpdem.contact.ContactModelBase import ContactModelBase
from src.mpdem.contact.NeighborBase import NeighborBase
from src.mpdem.Simulation import Simulation
from src.mpm.SpatialHashGrid import SpatialHashGrid
from src.mpm.engines.ULExplicitEngine import ULExplicitEngine as MPMExplicitEngine
from src.mpm.SceneManager import myScene as MPMScene
from src.mpm.Simulation import Simulation as MPMSimulation
from src.utils.linalg import no_operation


def dem_substep_count(coupled_timestep, dem_timestep):
    return max(1, int(math.ceil(coupled_timestep / dem_timestep - 1.0e-12)))


class Engine(object):
    sims: Simulation
    msims: MPMSimulation
    dsims: DEMSimulation
    mscene: MPMScene
    dscene: DEMScene
    mengine: MPMExplicitEngine
    dengine: DEMExplicitEngine
    neighbor: NeighborBase
    mneighbor: SpatialHashGrid
    dneighbor: LinkedCell
    physpp: ContactModelBase
    physpw: ContactModelBase

    def __init__(
        self, sims, msims, dsims, mscene, dscene, mengine, dengine, neighbor, mneighbor, dneighbor, physpp, physpw
    ) -> None:
        self.sims = sims
        self.msims = msims
        self.dsims = dsims
        self.mscene = mscene
        self.dscene = dscene
        self.mengine = mengine
        self.dengine = dengine
        self.neighbor = neighbor
        self.mneighbor = mneighbor
        self.dneighbor = dneighbor
        self.physpp = physpp
        self.physpw = physpw
        self.compute = None
        self.reset_contact_energy = self.reset_contact_energies
        self.incompressible_coupler = None
        self.precompute_incompressible = no_operation
        self.accumulate_pressure_force = no_operation
        self.resolve_cross_contact = no_operation
        self.dem_external_force = None
        self.dem_external_torque = None

    def choose_incompressible_lsdem_coupling(self):
        from src.mpdem.fluid_dynamics.IncompressibleCoupling import IncompressibleLSDEMCoupling

        self.incompressible_coupler = IncompressibleLSDEMCoupling(
            self.msims, self.dsims, self.mscene, self.dscene, self.mengine
        )
        self.incompressible_coupler.attach()
        self._bind_incompressible_coupler()
        self.compute = self.incompressible_lsdem_integration

    def choose_two_phase_double_layer_lsdem_coupling(self):
        from src.mpdem.fluid_dynamics.IncompressibleCoupling import TwoPhaseDoubleLayerLSDEMCoupling

        self.incompressible_coupler = TwoPhaseDoubleLayerLSDEMCoupling(
            self.msims, self.dsims, self.mscene, self.dscene, self.mengine
        )
        self.incompressible_coupler.attach()
        self._bind_incompressible_coupler()
        # This path needs the ordinary MPM--LSDEM Verlet/contact update in
        # addition to the fluid IBM load assembled by the coupler.
        self.compute = self.lsintegration

    def choose_incompressible_dem_sphere_coupling(self, drag_model):
        if int(self.dscene.clumpNum[0]) > 0:
            raise RuntimeError(
                "SemiResolved incompressible DEM-MPM coupling supports DEM spheres only; "
                "represent clumps and irregular particles as LSDEM SDF bodies in FullyResolved coupling"
            )
        from src.mpdem.fluid_dynamics.IncompressibleSemiResolved import IncompressibleDEMSphereCoupling

        self.incompressible_coupler = IncompressibleDEMSphereCoupling(
            self.sims, self.msims, self.dsims, self.mscene, self.dscene, self.mengine, drag_model
        )
        self.incompressible_coupler.attach()
        self._bind_incompressible_coupler()
        self.compute = self.incompressible_dem_sphere_integration

    def _bind_incompressible_coupler(self):
        self.precompute_incompressible = getattr(self.incompressible_coupler, "pre_compute", no_operation)
        self.accumulate_pressure_force = self.incompressible_coupler.accumulate_pressure_force
        self.dengine.transform_translational_load = getattr(
            self.incompressible_coupler, "transform_translational_load", no_operation
        )
        self.resolve_cross_contact = (
            self.system_resolve
            if self.sims.coupling_scheme != "CFDEM" and self.sims.particle_interaction
            else no_operation
        )

    def choose_engine(self, drag_model):
        valid_list = [
            "DEM: The material points are serves as rigid wall",
            "MPM: The discrete element particles are serves as rigid wall",
            "MPDEM: Two-way coupling scheme",
            "DEMPM: Two-way coupling scheme",
            "CFDEM: Considering computational fluid dynamics method",
        ]

        if self.sims.coupling_scheme == "MPDEM" or self.sims.coupling_scheme == "DEMPM":
            if self.dsims.scheme == "DEM":
                if "Implicit" in self.msims.solver_type:
                    if self.msims.material_type == "Solid":
                        pass
                    elif self.msims.material_type == "Fluid":
                        if self.msims.discretization == "FDM":
                            self.choose_incompressible_dem_sphere_coupling(drag_model)
                    elif self.msims.material_type == "TwoPhaseSingleLayer":
                        pass
                    elif self.msims.material_type == "TwoPhaseDoubleLayer":
                        pass
                else:
                    self.compute = self.integration
            elif self.dsims.scheme == "LSDEM":
                if "Implicit" in self.msims.solver_type:
                    if self.msims.material_type == "Solid":
                        pass
                    elif self.msims.material_type == "Fluid":
                        if self.msims.discretization == "FDM":
                            self.choose_incompressible_lsdem_coupling()
                    elif self.msims.material_type == "TwoPhaseSingleLayer":
                        pass
                    elif self.msims.material_type == "TwoPhaseDoubleLayer":
                        self.choose_two_phase_double_layer_lsdem_coupling()
                else:
                    self.compute = self.lsintegration
        elif self.sims.coupling_scheme == "DEM":
            if self.dsims.scheme == "DEM":
                self.compute = self.dem_integration
            elif self.dsims.scheme == "LSDEM":
                self.compute = self.lsdem_integration
        elif self.sims.coupling_scheme == "MPM":
            self.compute = self.mpm_integration
        elif self.sims.coupling_scheme == "CFDEM":
            self.reset_contact_energy = no_operation
            if self.dsims.scheme == "DEM":
                self.choose_incompressible_dem_sphere_coupling(drag_model)
            elif self.dsims.scheme == "LSDEM":
                self.choose_incompressible_lsdem_coupling()
        else:
            raise RuntimeError(
                f"Keyword:: /coupling_scheme: {self.sims.coupling_scheme}/ is invalid. Only the following is valid: \n{valid_list}"
            )

        if self.compute is None:
            raise RuntimeError(
                "Unsupported DEM-MPM coupling combination: "
                f"coupling_scheme={self.sims.coupling_scheme}, "
                f"DEM scheme={self.dsims.scheme}, "
                f"MPM solver_type={self.msims.solver_type}, "
                f"MPM material_type={self.msims.material_type}, "
                f"MPM discretization={self.msims.discretization}"
            )

        self.get_wall_contact_forces = no_operation
        if self.dsims.max_wall_num > 0:
            if (
                "wall" in self.dsims.monitor_type
                and self.msims.max_particle_num > 0
                and self.sims.wall_interaction
                and self.dsims.wall_type < 2
            ):
                self.get_wall_contact_forces = self.get_wall_contact_force

    def set_servo_mechanism(self):
        self.update_servo_wall = no_operation
        if self.sims.wall_interaction:
            if self.dsims.max_servo_wall_num > 0 and self.dsims.servo_status == "On":
                if self.dsims.servo_type == "StiffnessControl":
                    self.update_servo_wall = self.update_servo_stiffness_control
                elif self.dsims.servo_type == "GainControl":
                    self.update_servo_wall = self.update_servo_gain_control
            else:
                self.update_servo_wall = no_operation

    def pre_calculate(self):
        self.mengine.pre_calculation(self.msims, self.mscene, self.mneighbor)
        self.dengine.pre_calculation(self.dsims, self.dscene, self.dneighbor)
        if self.sims.coupling_scheme != "CFDEM" and (self.sims.particle_interaction or self.sims.wall_interaction):
            self.neighbor.pre_neighbor(self.mscene, self.dscene)
            self.physpp.update_contact_table(self.sims, self.mscene, self.dscene, self.neighbor)
            self.physpw.update_contact_table(self.sims, self.mscene, self.dscene, self.neighbor)
            self.neighbor.update_particle_particle_auxiliary_lists()
            self.neighbor.update_particle_wall_auxiliary_lists()
            self.physpp.resolve(self.sims, self.mscene, self.dscene, self.neighbor)
            self.physpw.resolve(self.sims, self.mscene, self.dscene, self.neighbor)
        self.precompute_incompressible()

    def reset_contact_energies(self):
        self.physpp.reset()
        self.physpw.reset()

    def reset_message(self):
        self.sims.timer.begin("Reset")
        self.mengine.reset_grid_message(self.mscene)
        if self.sims.coupling_scheme != "CFDEM" and (self.sims.particle_interaction or self.sims.wall_interaction):
            self.mengine.reset_particle_message(self.mscene)
        self.dengine.reset_wall_message(self.dscene)
        self.dengine.reset_particle_message(self.dscene)
        self.dengine.reset_contact_energy()
        self.reset_contact_energy()
        self.sims.timer.end("Reset")

    def apply_contact_model(self):
        self.sims.timer.begin("Coupling narrow search")
        self.physpp.update_contact_table(self.sims, self.mscene, self.dscene, self.neighbor)
        self.physpw.update_contact_table(self.sims, self.mscene, self.dscene, self.neighbor)
        self.neighbor.update_particle_particle_auxiliary_lists()
        self.neighbor.update_particle_wall_auxiliary_lists()
        self.sims.timer.end("Coupling narrow search")

    def system_resolve(self):
        self.sims.timer.begin("Coupling force calculate")
        self.physpp.resolve(self.sims, self.mscene, self.dscene, self.neighbor)
        self.physpw.resolve(self.sims, self.mscene, self.dscene, self.neighbor)
        self.sims.timer.end("Coupling force calculate")

    def update_verlet_table(self):
        self.sims.timer.begin("Coupling broad search")
        self.neighbor.update_verlet_table(self.mscene, self.dscene)
        self.sims.timer.end("Coupling broad search")
        self.apply_contact_model()

    def update_servo_stiffness_control(self):
        self.sims.timer.begin("Coupling servo wall")
        get_contact_stiffness_(
            self.sims.max_material_num,
            int(self.mscene.couplingNum[0]),
            self.mscene.particle,
            self.dscene.wall,
            self.physpw.surfaceProps,
            self.physpw.cplist,
            self.neighbor.particle_wall,
        )
        self.sims.timer.end("Coupling servo wall")

    def update_servo_gain_control(self):
        self.sims.timer.begin("Coupling servo wall")
        get_wall_contact_force_(
            int(self.mscene.couplingNum[0]), self.dscene.wall, self.physpw.cplist, self.neighbor.particle_wall
        )
        self.sims.timer.end("Coupling servo wall")

    def get_wall_contact_force(self):
        get_wall_contact_force_(
            int(self.mscene.couplingNum[0]), self.dscene.wall, self.physpw.cplist, self.neighbor.particle_wall
        )

    def dem_integration(self):
        if self.dengine.is_verlet_update(self.dengine.limit1) == 1:
            self.dengine.update_verlet_table(self.dsims, self.dscene, self.dneighbor)
            self.update_verlet_table()

        self.dengine.system_resolve(self.dsims, self.dscene, self.dneighbor)
        self.system_resolve()

        self.update_servo_wall()
        self.dengine.integration(self.dsims, self.dscene, self.dneighbor)

    def lsdem_integration(self):
        if self.dengine.is_verlet_update(self.dengine.limit1) == 1:
            self.dengine.update_LSDEM_verlet_table1(self.dsims, self.dscene, self.dneighbor)
            self.dengine.update_LSDEM_verlet_table2(self.dsims, self.dscene, self.dneighbor)
            self.update_verlet_table()
        elif self.dengine.is_verlet_update_point(self.dengine.limit2) == 1:
            self.dengine.update_LSDEM_verlet_table2(self.dsims, self.dscene, self.dneighbor)
        self.dengine.system_resolve(self.dsims, self.dscene, self.dneighbor)

        self.update_servo_wall()
        self.dengine.integration(self.dsims, self.dscene, self.dneighbor)

    def mpm_integration(self):
        if self.mengine.is_verlet_update(self.mscene) == 1:
            self.mengine.execute_board_serach(self.msims, self.mscene, self.mneighbor)
            self.update_verlet_table()
        else:
            self.mengine.system_resolve(self.msims, self.mscene)
        self.system_resolve()

        self.update_servo_wall()
        self.get_wall_contact_forces()
        self.mengine.compute(self.msims, self.mscene, self.mneighbor)

    def integration(self):
        if self.mengine.is_verlet_update(self.mscene) == 1 or self.dengine.is_verlet_update(self.dengine.limit1) == 1:
            self.dengine.update_verlet_table(self.dsims, self.dscene, self.dneighbor)
            self.mengine.execute_board_serach(self.msims, self.mscene, self.mneighbor)
            self.update_verlet_table()
        else:
            self.mengine.system_resolve(self.msims, self.mscene)

        self.dengine.system_resolve(self.dsims, self.dscene, self.dneighbor)
        self.system_resolve()

        self.update_servo_wall()
        self.get_wall_contact_forces()
        self.mengine.compute(self.msims, self.mscene, self.mneighbor)
        self.accumulate_pressure_force(self.msims, self.mscene)
        self.dengine.integration(self.dsims, self.dscene, self.dneighbor)

    def lsintegration(self):
        if self.mengine.is_verlet_update(self.mscene) == 1 or self.dengine.is_verlet_update(self.dengine.limit1) == 1:
            self.dengine.update_LSDEM_verlet_table1(self.dsims, self.dscene, self.dneighbor)
            self.dengine.update_LSDEM_verlet_table2(self.dsims, self.dscene, self.dneighbor)
            self.mengine.execute_board_serach(self.msims, self.mscene, self.mneighbor)
            self.update_verlet_table()
        elif self.dengine.is_verlet_update_point(self.dengine.limit2) == 1:
            self.dengine.update_LSDEM_verlet_table2(self.dsims, self.dscene, self.dneighbor)
        else:
            self.mengine.system_resolve(self.msims, self.mscene)

        self.dengine.system_resolve(self.dsims, self.dscene, self.dneighbor)
        self.system_resolve()

        self.update_servo_wall()
        self.get_wall_contact_forces()
        self.mengine.compute(self.msims, self.mscene, self.mneighbor)
        self.accumulate_pressure_force(self.msims, self.mscene)
        self.dengine.integration(self.dsims, self.dscene, self.dneighbor)

    def incompressible_dem_sphere_integration(self):
        self._incompressible_dem_integration(self.dscene.particle, self._sphere_contact_search)

    def _sphere_contact_search(self):
        if self.dengine.is_verlet_update(self.dengine.limit1) == 1:
            self.dengine.update_verlet_table(self.dsims, self.dscene, self.dneighbor)

    def _lsdem_contact_search(self):
        if self.dengine.is_verlet_update(self.dengine.limit1) == 1:
            self.dengine.update_LSDEM_verlet_table1(self.dsims, self.dscene, self.dneighbor)
            self.dengine.update_LSDEM_verlet_table2(self.dsims, self.dscene, self.dneighbor)
        elif self.dengine.is_verlet_update_point(self.dengine.limit2) == 1:
            self.dengine.update_LSDEM_verlet_table2(self.dsims, self.dscene, self.dneighbor)

    def _ensure_dem_external_load_cache(self):
        if self.dem_external_force is None:
            capacity = int(self.dsims.max_particle_num)
            self.dem_external_force = ti.Vector.field(3, dtype=float, shape=capacity)
            self.dem_external_torque = ti.Vector.field(3, dtype=float, shape=capacity)

    def _subcycle_incompressible_dem_contact(self, coupled_timestep, body, contact_search):
        body_num = int(self.dscene.particleNum[0])
        self._ensure_dem_external_load_cache()
        cache_dem_external_load(body_num, body, self.dem_external_force, self.dem_external_torque)
        substeps = dem_substep_count(coupled_timestep, self.sims.dem_timestep)
        self.dsims.set_timestep(coupled_timestep / substeps)
        try:
            for _ in range(substeps):
                self.dengine.reset_wall_message(self.dscene)
                self.dengine.reset_particle_message(self.dscene)
                restore_dem_external_load(body_num, body, self.dem_external_force, self.dem_external_torque)
                contact_search()
                self.dengine.system_resolve(self.dsims, self.dscene, self.dneighbor)
                self.resolve_cross_contact()
                self.update_servo_wall()
                self.get_wall_contact_forces()
                self.dengine.integration(self.dsims, self.dscene, self.dneighbor)
        finally:
            self.dsims.set_timestep(coupled_timestep)

    def incompressible_lsdem_integration(self):
        self._incompressible_dem_integration(self.dscene.rigid, self._lsdem_contact_search)

    def _incompressible_dem_integration(self, body, contact_search):
        contact_search()
        # Honor the DEM stability limit even before contact enters the neighbor
        # list: a collision can start within this coupled step (also with zero skin).
        use_contact_substeps = self.sims.dem_timestep < self.sims.delta
        if not use_contact_substeps:
            self.dengine.system_resolve(self.dsims, self.dscene, self.dneighbor)
            self.resolve_cross_contact()
            self.update_servo_wall()
            self.get_wall_contact_forces()
        self.mengine.compute(self.msims, self.mscene, self.mneighbor)
        self.accumulate_pressure_force(self.msims, self.mscene)
        if use_contact_substeps:
            self._subcycle_incompressible_dem_contact(self.sims.delta, body, contact_search)
        else:
            self.dengine.integration(self.dsims, self.dscene, self.dneighbor)

    def enforce_reset_dempm_contact_list(self):
        self.update_verlet_table()

    def enforce_reset_contact_list(self):
        self.dengine.update_verlet_table(self.dsims, self.dscene, self.dneighbor)
        self.update_verlet_table()

    def adaptive_timestep(self):
        self.sims.timer.begin("Adaptive time step")
        if self.sims.adaptive_timestep > 0 and self.sims.current_step % self.sims.adaptive_timestep == 0:
            min_dt1 = self.mscene.adaptive_timestep(self.msims)
            min_dt2 = self.dscene.adaptive_timestep(self.dsims)
            min_dt3 = kernel_adaptive_timestep(int(self.mscene.particleNum[0]), self.mscene.particle)
            min_dt = min(self.sims.init_delta, self.sims.CFL * min(min_dt1, min_dt2, min_dt3))
            self.dsims.set_timestep(min_dt)
            self.msims.set_timestep(min_dt)
            self.sims.set_timestep(min_dt)
        self.sims.timer.end("Adaptive time step")

import numpy as np

from src.dem.neighbor.LinkedCell import LinkedCell 
from src.dem.engines.ExplicitEngine import ExplicitEngine as DEMExplicitEngine
from src.dem.engines.EngineKernel import get_contact_stiffness
from src.dem.SceneManager import myScene as DEMScene
from src.dem.Simulation import Simulation as DEMSimulation
from src.mpdem.contact.ContactModelBase import ContactModelBase
from src.mpdem.contact.MultiLinkedCell import MultiLinkedCell
from src.mpdem.Simulation import Simulation 
from src.mpm.SpatialHashGrid import SpatialHashGrid
from src.mpm.engines.ULExplicitEngine import ULExplicitEngine as MPMExplicitEngine
from src.mpm.SceneManager import myScene as MPMScene
from src.mpm.Simulation import Simulation as MPMSimulation
from src.utils.linalg import no_operation


class Engine(object):
    sims: Simulation
    msims: MPMSimulation
    dsims: DEMSimulation
    mscene: MPMScene
    dscene: DEMScene
    mengine: MPMExplicitEngine
    dengine: DEMExplicitEngine
    neighbor: MultiLinkedCell
    mneighbor: SpatialHashGrid
    dneighbor: LinkedCell
    physpp: ContactModelBase
    physpw: ContactModelBase

    def __init__(self, sims, msims, dsims, mscene, dscene, mengine, dengine, neighbor, mneighbor, dneighbor, physpp, physpw) -> None:
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

    def manage_function(self, drag_model):
        valid_list = ["DEM: The material points are serves as rigid wall", 
                      "MPM: The discrete element particles are serves as rigid wall", 
                      "MPDEM: Two-way coupling scheme", 
                      "DEMPM: Two-way coupling scheme",
                      "CFDEM: Considering computational fluid dynamics method"]
        
        if self.sims.coupling_scheme == 'MPDEM' or self.sims.coupling_scheme == 'DEMPM':
            if self.dsims.scheme == "DEM":
                if "Implicit" in self.msims.solver_type:
                    if self.msims.material_type == "Solid":
                        pass
                    elif self.msims.material_type == "Fluid":
                        self.compute = self.CFDEMintegration
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
                        self.compute = self.CFDEMlsintegration
                    elif self.msims.material_type == "TwoPhaseSingleLayer":
                        pass
                    elif self.msims.material_type == "TwoPhaseDoubleLayer":
                        pass
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
            from src.mpdem.fluid_dynamics.CFDpart import CFDMPM
            if self.dsims.scheme == "DEM":
                self.compute = self.CFDEMintegration
            elif self.dsims.scheme == "LSDEM":
                self.compute = self.CFDEMlsintegration
            self.fluid_particle_model = CFDMPM(self.sims, self.msims, self.dsims, self.mscene, self.dscene, self.dneighbor)
            self.fluid_particle_model.build_essential_field(drag_model)
        else:
            raise RuntimeError(f"Keyword:: /coupling_scheme: {self.sims.coupling_scheme}/ is invalid. Only the following is valid: \n{valid_list}")
        
        self.update_servo_wall = no_operation
        if self.dsims.max_servo_wall_num > 0 and self.dsims.servo_status == "On" and self.sims.wall_interaction:
            self.update_servo_wall = self.update_servo_motion

    def pre_calculate(self):
        self.mengine.pre_calculation(self.msims, self.mscene, self.mneighbor)
        self.dengine.pre_calculation(self.dsims, self.dscene, self.dneighbor)
        if self.sims.coupling_scheme != "CFDEM":
            self.neighbor.pre_neighbor(self.mscene, self.dscene)
            self.physpp.update_contact_table(self.sims, self.mscene, self.dscene, self.neighbor)
            self.physpw.update_contact_table(self.sims, self.mscene, self.dscene, self.neighbor)
            self.neighbor.update_particle_particle_auxiliary_lists()
            self.neighbor.update_particle_wall_auxiliary_lists()
            self.physpp.resolve(self.sims, self.mscene, self.dscene, self.neighbor)
            self.physpw.resolve(self.sims, self.mscene, self.dscene, self.neighbor)
        else:
            self.fluid_particle_model.pre_compute()
    
    def reset_message(self):
        self.mengine.reset_grid_message(self.mscene)
        self.mengine.reset_particle_message(self.mscene)
        self.dengine.reset_wall_message(self.dscene)
        self.dengine.reset_particle_message(self.dscene)

    def apply_contact_model(self):
        self.physpp.update_contact_table(self.sims, self.mscene, self.dscene, self.neighbor)
        self.physpw.update_contact_table(self.sims, self.mscene, self.dscene, self.neighbor)
        self.neighbor.update_particle_particle_auxiliary_lists()
        self.neighbor.update_particle_wall_auxiliary_lists()

    def system_resolve(self):
        self.physpp.resolve(self.sims, self.mscene, self.dscene, self.neighbor)
        self.physpw.resolve(self.sims, self.mscene, self.dscene, self.neighbor)

    def update_verlet_table(self):
        self.neighbor.update_verlet_table(self.mscene, self.dscene)
        self.apply_contact_model()

    def update_servo_motion(self):
        get_contact_stiffness(self.sims.max_material_num, int(self.mscene.particleNum[0]), self.mscene.particle, self.dscene.wall, 
                              self.physpw.surfaceProps, self.physpw.cplist, self.neighbor.particle_wall)

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
        self.dengine.integration(self.dsims, self.dscene, self.dneighbor)
        self.mengine.compute(self.msims, self.mscene, self.mneighbor)

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
        self.dengine.integration(self.dsims, self.dscene, self.dneighbor)
        self.mengine.compute(self.msims, self.mscene, self.mneighbor)

    def CFDEMintegration(self):
        if self.dengine.is_verlet_update(self.dengine.limit1) == 1:
            self.dengine.update_verlet_table(self.dsims, self.dscene, self.dneighbor)

        self.dengine.system_resolve(self.dsims, self.dscene, self.dneighbor)
        self.mengine.first_substep_cfd_coupling(self.msims, self.mscene)
        self.fluid_particle_model.coupling()
        self.mengine.second_substep_cfd_coupling(self.msims, self.mscene)
        self.dengine.integration(self.dsims, self.dscene, self.dneighbor)

    def CFDEMlsintegration(self):
        if self.dengine.is_verlet_update(self.dengine.limit1) == 1:
            self.dengine.update_LSDEM_verlet_table1(self.dsims, self.dscene, self.dneighbor)
            self.dengine.update_LSDEM_verlet_table2(self.dsims, self.dscene, self.dneighbor)
        elif self.dengine.is_verlet_update_point(self.dengine.limit2) == 1:
            self.dengine.update_LSDEM_verlet_table2(self.dsims, self.dscene, self.dneighbor)

        self.dengine.system_resolve(self.dsims, self.dscene, self.dneighbor)
        self.mengine.first_substep_cfd_coupling(self.msims, self.mscene)
        self.fluid_particle_model.coupling()
        self.mengine.second_substep_cfd_coupling(self.msims, self.mscene)
        self.dengine.integration(self.dsims, self.dscene, self.dneighbor)

    def enforce_reset_dempm_contact_list(self):
        self.update_verlet_table()

    def enforce_reset_contact_list(self):
        self.dengine.update_verlet_table(self.dsims, self.dscene, self.dneighbor)
        self.update_verlet_table()
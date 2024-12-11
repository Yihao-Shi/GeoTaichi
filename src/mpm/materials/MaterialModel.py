import numpy as np
import taichi as ti

from src.consititutive_model.MaterialModel import *
from src.mpm.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.mpm.Simulation import Simulation


class UserDefined(ConstitutiveModelBase):
    def __init__(self, sims: Simulation):
        super().__init__()
        self.add_material(sims.max_material_num, sims.material_type, sims.contact_detection, MaterialModel)
        if sims.configuration == "ULMPM":
            self.stateVars = ULStateVariable.field(shape=sims.max_particle_num) 
        elif sims.configuration == "TLMPM":
            self.stateVars = TLStateVariable.field(shape=sims.max_particle_num) 

        if sims.solver_type == "Implicit":
            self.stiffness_matrix = ti.Matrix.field(6, 6, float, shape=sims.max_particle_num)

    def model_initialize(self, material):
        pass

    def reload_state_variables(self, state_vars):
        pass

    def get_state_vars_dict(self, start_particle, end_particle):
        pass

import numpy as np
import taichi as ti

from src.consititutive_model.RigidBody import *
from src.mpm.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.mpm.Simulation import Simulation
from src.utils.ObjectIO import DictIO


class RigidBody(ConstitutiveModelBase):
    def __init__(self, sims: Simulation):
        super().__init__()
        self.add_material(sims.max_material_num, sims.material_type, sims.contact_detection, Rigid)
        if sims.configuration == "ULMPM":
            self.stateVars = ULStateVariable.field(shape=sims.max_particle_num) 
        elif sims.configuration == "TLMPM":
            self.stateVars = TLStateVariable.field(shape=sims.max_particle_num) 

        if sims.solver_type == "Implicit":
            self.stiffness_matrix = None

    def get_state_vars_dict(self, start_particle, end_particle):
        estress = np.ascontiguousarray(self.stateVars.estress.to_numpy()[start_particle:end_particle])
        return {'estress': estress}

    def reload_state_variables(self, state_vars):
        estress = state_vars.item()['estress']
        kernel_reload_state_variables(estress, self.stateVars)
    
    def model_initialize(self, material):
        materialID = DictIO.GetAlternative(material, 'MaterialID', 0)
        if materialID > 0:
            materialID = 0
            
        if self.matProps[materialID].density > 0.:
            print("Previous Material Property will be overwritten!")
        density = DictIO.GetAlternative(material, 'Density', 2650)
        self.matProps[materialID].add_material(density)
        self.matProps[materialID].print_message(materialID)


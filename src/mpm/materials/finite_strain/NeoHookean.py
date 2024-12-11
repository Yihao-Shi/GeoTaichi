import numpy as np

from src.consititutive_model.finite_strain.NeoHookean import *
from src.mpm.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.mpm.Simulation import Simulation
from src.utils.ObjectIO import DictIO


class NeoHookean(ConstitutiveModelBase):
    def __init__(self, sims: Simulation):
        super().__init__()
        self.is_elastic = True
        self.add_material(sims.max_material_num, sims.material_type, sims.contact_detection, NeoHookeanModel)
        self.stateVars = StateVariable.field(shape=sims.max_particle_num)

    def get_state_vars_dict(self, start_particle, end_particle):
        estress = np.ascontiguousarray(self.stateVars.estress.to_numpy()[start_particle:end_particle])
        deformation_gradient = np.ascontiguousarray(self.stateVars.deformation_gradient.to_numpy()[start_particle:end_particle])
        return {'estress': estress, 'deformation_gradient': deformation_gradient}
    
    def reload_state_variables(self, state_vars):
        estress = state_vars.item()['estress']
        deformation_gradient = state_vars.item()['deformation_gradient']
        kernel_reload_state_variables(estress, deformation_gradient, self.stateVars)

    def model_initialize(self, material):
        materialID = DictIO.GetEssential(material, 'MaterialID') 
        self.check_materialID(materialID, self.matProps.shape[0])

        if self.matProps[materialID].density > 0.:
            print("Previous Material Property will be overwritten!")
        density = DictIO.GetAlternative(material, 'Density', 2650)
        young = DictIO.GetEssential(material, 'YoungModulus')
        possion = DictIO.GetAlternative(material, 'PossionRatio', 0.3)

        self.matProps[materialID].add_material(density, young, possion)
        self.contact_initialize(material)
        self.matProps[materialID].print_message(materialID)

    def get_lateral_coefficient(self, materialID):
        mu = self.matProps[materialID].possion
        return 0.3#mu / (1. - mu)
import numpy as np
import taichi as ti

from src.consititutive_model.infinitesimal_strain.SoilStructureInteraction import *
from src.mpm.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.mpm.Simulation import Simulation
from src.utils.constants import PI
from src.utils.ObjectIO import DictIO


class SoilStructureInteraction(ConstitutiveModelBase):
    def __init__(self, sims: Simulation):
        super().__init__()
        self.add_material(sims.max_material_num, sims.material_type, sims.contact_detection, SSIModel)
        if sims.configuration == "ULMPM":
            self.stateVars = ULStateVariable.field(shape=sims.max_particle_num) 
        elif sims.configuration == "TLMPM":
            self.stateVars = TLStateVariable.field(shape=sims.max_particle_num) 

        if sims.solver_type == "Implicit":
            self.stiffness_matrix = ti.Matrix.field(6, 6, float, shape=sims.max_particle_num)

    def get_state_vars_dict(self, start_particle, end_particle):
        epstrain = np.ascontiguousarray(self.stateVars.epstrain.to_numpy()[start_particle:end_particle])
        estress = np.ascontiguousarray(self.stateVars.estress.to_numpy()[start_particle:end_particle])
        return {'epstrain': epstrain, 'estress': estress}
    
    def reload_state_variables(self, state_vars):
        estress = state_vars.item()['estress']
        epstrain = state_vars.item()['epstrain']
        kernel_reload_state_variables(estress, epstrain, self.stateVars)

    def model_initialize(self, material):
        materialID = DictIO.GetEssential(material, 'MaterialID') 
        self.check_materialID(materialID, self.matProps.shape[0])
        
        if self.matProps[materialID].density > 0.:
            print("Previous Material Property will be overwritten!")
        is_structure = DictIO.GetEssential(material, 'IsStructure')
        density = DictIO.GetAlternative(material, 'Density', 2650)
        young = DictIO.GetEssential(material, 'YoungModulus')
        possion = DictIO.GetAlternative(material, 'PossionRatio', 0.3)
        if is_structure is False:
            c = DictIO.GetAlternative(material, 'Cohesion', 0.)
            fai = DictIO.GetAlternative(material, 'Friction', 0.) * PI / 180.
            psi = DictIO.GetAlternative(material, 'Dilation', 0.) * PI / 180.
            tensile = DictIO.GetAlternative(material, 'Tensile', 0.)
            dpType = DictIO.GetAlternative(material, 'dpType', "Inscribed")
            self.matProps[materialID].add_soil_material(density, young, possion, c, fai, psi, tensile, dpType)
        else:
            self.matProps[materialID].add_structure_material(density, young, possion)

        self.contact_initialize(material)
        self.matProps[materialID].print_message(materialID)
    
    def get_lateral_coefficient(self, materialID):
        mu = self.matProps[materialID].possion
        return mu / (1. - mu)

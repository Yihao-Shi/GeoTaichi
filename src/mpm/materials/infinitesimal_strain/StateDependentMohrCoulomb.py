import numpy as np
import taichi as ti

from src.consititutive_model.infinitesimal_strain.StateDependentMohrCoulomb import *
from src.mpm.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.mpm.Simulation import Simulation
from src.utils.ObjectIO import DictIO


class StateDependentMohrCoulomb(ConstitutiveModelBase):
    def __init__(self, sims: Simulation):
        super().__init__()
        self.add_material(sims.max_material_num, sims.material_type, sims.contact_detection, StateDependentMohrCoulombModel)
        if sims.configuration == "ULMPM":
            self.stateVars = ULStateVariable.field(shape=sims.max_particle_num) 
        elif sims.configuration == "TLMPM":
            self.stateVars = TLStateVariable.field(shape=sims.max_particle_num) 

        if sims.solver_type == "Implicit":
            self.stiffness_matrix = ti.Matrix.field(6, 6, float, shape=sims.max_particle_num)

    def get_state_vars_dict(self, start_particle, end_particle):
        epstrain = np.ascontiguousarray(self.stateVars.epstrain.to_numpy()[start_particle:end_particle])
        estress = np.ascontiguousarray(self.stateVars.estress.to_numpy()[start_particle:end_particle])
        void_ratio = np.ascontiguousarray(self.stateVars.void_ratio.to_numpy()[start_particle:end_particle])
        # eps0 = np.ascontiguousarray(self.stateVars.eps0.to_numpy()[start_particle:end_particle])
        # eps1 = np.ascontiguousarray(self.stateVars.eps1.to_numpy()[start_particle:end_particle])
        # eps2 = np.ascontiguousarray(self.stateVars.eps2.to_numpy()[start_particle:end_particle])
        # Matpara1 = np.ascontiguousarray(self.stateVars.Matpara1.to_numpy()[start_particle:end_particle])
        # Matpara2 = np.ascontiguousarray(self.stateVars.Matpara2.to_numpy()[start_particle:end_particle])
        return {'epstrain': epstrain, 'estress': estress, 'void_ratio':void_ratio}
    
    def reload_state_variables(self, state_vars):
        estress = state_vars.item()['estress']
        epstrain = state_vars.item()['epstrain']
        kernel_reload_state_variables(estress, epstrain, self.stateVars)

    def model_initialize(self, material):
        materialID = DictIO.GetEssential(material, 'MaterialID')
        self.check_materialID(materialID, self.matProps.shape[0])
        
        if self.matProps[materialID].density > 0.:
            print("Previous Material Property will be overwritten!")
        density = DictIO.GetAlternative(material, 'Density', 2650)
        young = DictIO.GetEssential(material, 'YoungModulus')
        possion = DictIO.GetAlternative(material, 'PossionRatio', 0.3)
        e0 = DictIO.GetAlternative(material, 'e0', 0.3)
        e_Tao = DictIO.GetAlternative(material, 'e_Tao', 0.3)
        lambda_c = DictIO.GetAlternative(material, 'lambda_c', 0.3)
        ksi = DictIO.GetAlternative(material, 'ksi', 0.3)
        nd = DictIO.GetAlternative(material, 'nd', 0.3)
        nf = DictIO.GetAlternative(material, 'nf', 0.3)
        fai_c = DictIO.GetAlternative(material, 'fai_c', 0.3) * np.pi / 180.
        c = DictIO.GetAlternative(material, 'Cohesion', 0.)
        tensile = DictIO.GetAlternative(material, 'Tensile', 0.)
        self.matProps[materialID].add_material(density, young, possion, e0, e_Tao, lambda_c, ksi, nd, nf, fai_c, c, tensile)
        self.contact_initialize(material)
        self.matProps[materialID].print_message(materialID)
    
    def get_lateral_coefficient(self, materialID):
        mu = self.matProps[materialID].possion
        return mu / (1. - mu)


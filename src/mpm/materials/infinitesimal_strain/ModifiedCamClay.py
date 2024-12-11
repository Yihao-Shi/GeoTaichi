import numpy as np
import taichi as ti

from src.consititutive_model.infinitesimal_strain.ModifiedCamClay import *
from src.mpm.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.mpm.Simulation import Simulation
from src.utils.ObjectIO import DictIO


class ModifiedCamClay(ConstitutiveModelBase):
    def __init__(self, sims: Simulation):
        super().__init__()
        self.add_material(sims.max_material_num, sims.material_type, sims.contact_detection, ModifiedCamClayModel)
        if sims.configuration == "ULMPM":
            self.stateVars = ULStateVariable.field(shape=sims.max_particle_num) 
        elif sims.configuration == "TLMPM":
            self.stateVars = TLStateVariable.field(shape=sims.max_particle_num) 

        if sims.solver_type == "Implicit":
            self.stiffness_matrix = ti.Matrix.field(6, 6, float, shape=sims.max_particle_num)

    def model_initialize(self, material):
        materialID = DictIO.GetEssential(material, 'MaterialID')
        self.check_materialID(materialID, self.matProps.shape[0])
        
        if self.matProps[materialID].density > 0.:
            print("Previous Material Property will be overwritten!")
        density = DictIO.GetAlternative(material, 'Density', 2650)
        possion = DictIO.GetAlternative(material, 'PossionRatio', 0.3)
        m_theta = DictIO.GetEssential(material, 'StressRatio')
        lambda_ = DictIO.GetEssential(material, 'lambda')
        kappa = DictIO.GetEssential(material, 'kappa')
        ocr = DictIO.GetEssential(material, 'OverConsolidationRatio', 0)
        e_ref = DictIO.GetEssential(material, 'void_ratio_ref')
        p_ref = DictIO.GetAlternative(material, 'pressure_ref', 1000.)
        
        self.matProps[materialID].add_material(density, possion, m_theta, kappa, lambda_, e_ref, p_ref, ocr)
        self.contact_initialize(material)
        self.matProps[materialID].print_message(materialID)

    def get_state_vars_dict(self, start_particle, end_particle):
        pc = np.ascontiguousarray(self.stateVars.pc.to_numpy()[start_particle:end_particle])
        epstrain = np.ascontiguousarray(self.stateVars.epstrain.to_numpy()[start_particle:end_particle])
        estress = np.ascontiguousarray(self.stateVars.estress.to_numpy()[start_particle:end_particle])
        void_ratio = np.ascontiguousarray(self.stateVars.void_ratio.to_numpy()[start_particle:end_particle])
        return {'pc': pc, 'epstrain': epstrain, 'estress': estress, 'void_ratio': void_ratio}
    
    def reload_state_variables(self, state_vars):
        estress = state_vars.item()['estress']
        epstrain = state_vars.item()['epstrain']
        pc = state_vars.item()['pc']
        void_ratio = state_vars.item()['void_ratio']
        kernel_reload_state_variables(estress, epstrain, pc, void_ratio, self.stateVars)
    
    def get_lateral_coefficient(self, materialID):
        return 0.9

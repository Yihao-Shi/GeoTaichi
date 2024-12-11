import numpy as np
import taichi as ti

from src.consititutive_model.infinitesimal_strain.CohesiveModifiedCamClay import *
from src.mpm.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.mpm.Simulation import Simulation
from src.utils.constants import PI
from src.utils.ObjectIO import DictIO


class CohesiveModifiedCamClay(ConstitutiveModelBase):
    def __init__(self, sims: Simulation):
        super().__init__()
        self.add_material(sims.max_material_num, sims.material_type, sims.contact_detection, CohesiveModifiedCamClayModel)
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
        lambda_ = DictIO.GetEssential(material, 'lambda')
        kappa = DictIO.GetEssential(material, 'kappa')
        
        if 'StressRatio' in material:
            m_theta = DictIO.GetEssential(material, 'StressRatio')
        elif 'CSFriction' in material:
            fric = DictIO.GetEssential(material, 'CSFriction') * PI / 180.
            sinc_friction_cs = ti.sin(fric)
            m_theta = (6. * sinc_friction_cs) / (3. - sinc_friction_cs)
        else:
            raise RuntimeError("Critical state ratio is not defined")
        
        e_ref = DictIO.GetEssential(material, 'void_ratio_ref')
        p_ref = DictIO.GetAlternative(material, 'pressure_ref', 1000.)
        ocr = DictIO.GetEssential(material, 'OverConsolidationRatio', 0)

        subloading = DictIO.GetAlternative(material, 'Subloading', False)
        subloading_u = DictIO.GetAlternative(material, 'SubloadingSurfaceRatio', 0.)

        bonding = DictIO.GetAlternative(material, 'Bonding', False)
        s_h = DictIO.GetAlternative(material, 'HydrateSaturation', 0.)
        mc_a = DictIO.GetAlternative(material, 'Coefficient1', 0.)
        mc_b = DictIO.GetAlternative(material, 'Coefficient2', 0.)
        mc_c = DictIO.GetAlternative(material, 'Coefficient3', 0.)
        mc_d = DictIO.GetAlternative(material, 'Coefficient4', 0.)
        m_degradation = DictIO.GetAlternative(material, 'Degradation', 0.)
        m_shear = DictIO.GetAlternative(material, 'ModulusIncrement', 0.)

        self.matProps[materialID].add_material(density, possion, m_theta, kappa, lambda_, e_ref, p_ref, ocr, subloading, subloading_u,
                                               bonding, s_h, mc_a, mc_b, mc_c, mc_d, m_degradation, m_shear)
        self.contact_initialize(material)
        self.matProps[materialID].print_message(materialID)

    def get_state_vars_dict(self, start_particle, end_particle):
        pc = np.ascontiguousarray(self.stateVars.pc.to_numpy()[start_particle:end_particle])
        pcc = np.ascontiguousarray(self.stateVars.pcc.to_numpy()[start_particle:end_particle])
        pcd = np.ascontiguousarray(self.stateVars.pcd.to_numpy()[start_particle:end_particle])
        epstrain = np.ascontiguousarray(self.stateVars.epstrain.to_numpy()[start_particle:end_particle])
        estress = np.ascontiguousarray(self.stateVars.estress.to_numpy()[start_particle:end_particle])
        void_ratio0 = np.ascontiguousarray(self.stateVars.void_ratio0.to_numpy()[start_particle:end_particle])
        void_ratio = np.ascontiguousarray(self.stateVars.void_ratio.to_numpy()[start_particle:end_particle])
        chi = np.ascontiguousarray(self.stateVars.chi.to_numpy()[start_particle:end_particle])
        subloading = np.ascontiguousarray(self.stateVars.subloading.to_numpy()[start_particle:end_particle])
        return {'pc': pc, 'pcc': pcc, 'pcd': pcd, 'epstrain': epstrain, 'estress': estress, 'void_ratio': void_ratio, 'void_ratio0': void_ratio0, 'chi': chi, 'subloading': subloading}
    
    def reload_state_variables(self, state_vars):
        estress = state_vars.item()['estress']
        epstrain = state_vars.item()['epstrain']
        void_ratio0 = state_vars.item()['void_ratio0']
        void_ratio = state_vars.item()['void_ratio']
        chi = state_vars.item()['chi']
        subloading = state_vars.item()['csubloadinghi']
        pc = state_vars.item()['pc']
        pcc = state_vars.item()['pcc']
        pcd = state_vars.item()['pcd']
        kernel_reload_state_variables(estress, epstrain, void_ratio0, void_ratio, pc, pcc, pcd, chi, subloading, self.stateVars)
    
    def get_lateral_coefficient(self, materialID):
        return 0.9

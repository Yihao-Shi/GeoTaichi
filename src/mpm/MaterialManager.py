from src.mpm.materials.RigidBody import RigidBody
from src.mpm.materials.finite_strain.HencyElastic import HencyElastic
from src.mpm.materials.finite_strain.NeoHookean import NeoHookean
from src.mpm.materials.infinitesimal_strain.LinearElastic import LinearElastic
from src.mpm.materials.infinitesimal_strain.ElasticPerfectlyPlastic import ElasticPerfectlyPlastic
from src.mpm.materials.infinitesimal_strain.IsotropicHardeningPlastic import IsotropicHardeningPlastic
from src.mpm.materials.infinitesimal_strain.WillianMohrCoulomb import WillianMohrCoulomb
from src.mpm.materials.infinitesimal_strain.MohrCoulomb import MohrCoulomb
from src.mpm.materials.infinitesimal_strain.DruckerPrager import DruckerPrager
from src.mpm.materials.infinitesimal_strain.ModifiedCamClay import ModifiedCamClay
from src.mpm.materials.infinitesimal_strain.CohesiveModifiedCamClay import CohesiveModifiedCamClay
from src.mpm.materials.infinitesimal_strain.SoilStructureInteraction import SoilStructureInteraction
from src.mpm.materials.strain_rate.Newtonian import Newtonian
from src.mpm.materials.strain_rate.Bingham import Bingham
from src.mpm.materials.strain_rate.FluidStructureInteraction import FluidStructureInteraction
from src.mpm.materials.MaterialModel import UserDefined

class ConstitutiveModel:
    @classmethod
    def initialize(cls, material_type, stabilize, constitutive_model, max_material_num, max_particle_num, configuration, solver_type):
        if material_type == "Solid":
            model_type = ["None", "LinearElastic", "HencyElastic", "NeoHookean", "ElasticPerfectlyPlastic", "IsotropicHardeningPlastic",
                          "MohrCoulomb", "SoftenMohrCoulomb", "DruckerPrager", "ModifiedCamClay", "CohesiveModifiedCamClay", "SoilStructureInteraction", "UserDefined"]
            if constitutive_model == "None":
                return RigidBody(max_material_num, max_particle_num, configuration, solver_type)
            elif constitutive_model == "HencyElastic":
                if stabilize == "B-Bar Method":
                    raise RuntimeError("B bar method is unsupported in HencyElastic material")
                return HencyElastic(max_material_num, max_particle_num, configuration, solver_type)
            elif constitutive_model == "NeoHookean":
                if stabilize == "B-Bar Method":
                    raise RuntimeError("B bar method is unsupported in NeoHookean material")
                return NeoHookean(max_material_num, max_particle_num, configuration, solver_type)
            elif constitutive_model == "LinearElastic":
                return LinearElastic(max_material_num, max_particle_num, configuration, solver_type)
            elif constitutive_model == "ElasticPerfectlyPlastic":
                return ElasticPerfectlyPlastic(max_material_num, max_particle_num, configuration, solver_type)
            elif constitutive_model == "IsotropicHardeningPlastic":
                return IsotropicHardeningPlastic(max_material_num, max_particle_num, configuration, solver_type)
            elif constitutive_model == "MohrCoulomb":
                return WillianMohrCoulomb(max_material_num, max_particle_num, configuration, solver_type)
            elif constitutive_model == "SoftenMohrCoulomb":
                return MohrCoulomb(max_material_num, max_particle_num, configuration, solver_type)
            elif constitutive_model == "DruckerPrager":
                return DruckerPrager(max_material_num, max_particle_num, configuration, solver_type)
            elif constitutive_model == "ModifiedCamClay":
                return ModifiedCamClay(max_material_num, max_particle_num, configuration, solver_type)
            elif constitutive_model == "CohesiveModifiedCamClay":
                return CohesiveModifiedCamClay(max_material_num, max_particle_num, configuration, solver_type)
            elif constitutive_model == "SoilStructureInteraction":
                if solver_type == "Implicit":
                    raise RuntimeError("Only /Explicit/ /ULMPM/ supports soil-structure interaction model")
                return SoilStructureInteraction(max_material_num, max_particle_num, configuration, solver_type)
            elif constitutive_model == "UserDefined":
                return UserDefined(max_material_num, max_particle_num, configuration, solver_type)
            else:
                raise ValueError(f'Constitutive Model: {constitutive_model} error! Only the following is aviliable:\n{model_type}')
        elif material_type == "Fluid":
            if configuration =="TLMPM":
                raise RuntimeError("Only /Explicit/ /ULMPM/ supports fluid model")
            if solver_type == "Implicit":
                raise RuntimeError("Only /Explicit/ /ULMPM/ supports fluid model")
            
            model_type = ["Newtonian", "Bingham", "FluidStructureInteraction", "UserDefined"]
            if constitutive_model == "Newtonian":
                return Newtonian(max_material_num, max_particle_num)
            elif constitutive_model == "Bingham":
                return Bingham(max_material_num, max_particle_num)
            elif constitutive_model == "FluidStructureInteraction":
                return FluidStructureInteraction(max_material_num, max_particle_num)
            else:
                raise ValueError(f'Constitutive Model: {constitutive_model} error! Only the following is aviliable:\n{model_type}')
        



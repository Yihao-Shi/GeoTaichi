from geotaichi.mpm.materials.RigidBody import RigidBody
from geotaichi.mpm.materials.finite_strain.HencyElastic import HencyElastic
from geotaichi.mpm.materials.finite_strain.NeoHookean import NeoHookean
from geotaichi.mpm.materials.infinitesimal_strain.LinearElastic import LinearElastic
from geotaichi.mpm.materials.infinitesimal_strain.ElasticPerfectlyPlastic import ElasticPerfectlyPlastic
from geotaichi.mpm.materials.infinitesimal_strain.IsotropicHardeningPlastic import IsotropicHardeningPlastic
from geotaichi.mpm.materials.infinitesimal_strain.WillianMohrCoulomb import WillianMohrCoulomb
from geotaichi.mpm.materials.infinitesimal_strain.MohrCoulomb import MohrCoulomb
from geotaichi.mpm.materials.infinitesimal_strain.DruckerPrager import DruckerPrager
from geotaichi.mpm.materials.infinitesimal_strain.ModifiedCamClay import ModifiedCamClay
from geotaichi.mpm.materials.infinitesimal_strain.CohesiveModifiedCamClay import CohesiveModifiedCamClay
from geotaichi.mpm.materials.infinitesimal_strain.SoilStructureInteraction import SoilStructureInteraction
from geotaichi.mpm.materials.strain_rate.Newtonian import Newtonian
from geotaichi.mpm.materials.strain_rate.Bingham import Bingham
from geotaichi.mpm.materials.strain_rate.FluidStructureInteraction import FluidStructureInteraction
from geotaichi.mpm.materials.MaterialModel import UserDefined

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
        



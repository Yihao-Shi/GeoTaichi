from src.mpm.Simulation import Simulation
from src.mpm.materials.RigidBody import RigidBody
from src.mpm.materials.finite_strain.HenckyElastic import HenckyElastic
from src.mpm.materials.finite_strain.NeoHookean import NeoHookean
from src.mpm.materials.infinitesimal_strain.LinearElastic import LinearElastic
from src.mpm.materials.infinitesimal_strain.ElasticPerfectlyPlastic import ElasticPerfectlyPlastic
from src.mpm.materials.infinitesimal_strain.IsotropicHardeningPlastic import IsotropicHardeningPlastic
from src.mpm.materials.infinitesimal_strain.WillianMohrCoulomb import WillianMohrCoulomb
from src.mpm.materials.infinitesimal_strain.MohrCoulomb import MohrCoulomb
from src.mpm.materials.infinitesimal_strain.StateDependentMohrCoulomb import StateDependentMohrCoulomb
from src.mpm.materials.infinitesimal_strain.DruckerPrager import DruckerPrager
from src.mpm.materials.infinitesimal_strain.ModifiedCamClay import ModifiedCamClay
from src.mpm.materials.infinitesimal_strain.CohesiveModifiedCamClay import CohesiveModifiedCamClay
from src.mpm.materials.infinitesimal_strain.SoilStructureInteraction import SoilStructureInteraction
from src.mpm.materials.strain_rate.Newtonian import Newtonian
from src.mpm.materials.strain_rate.Bingham import Bingham
from src.mpm.materials.strain_rate.FluidStructureInteraction import FluidStructureInteraction
from src.mpm.materials.MaterialModel import UserDefined

class ConstitutiveModel:
    def __init__(self) -> None:
        self.constitutive_model = None
        self.material = None

    def save_material(self, model, material):
        self.constitutive_model = model
        self.material = material

    def initialize(self, sims: Simulation):
        if sims.material_type == "Solid" or sims.material_type == "TwoPhaseDoubleLayer":
            model_type = ["None", "RigidBoyd", "LinearElastic", "HenckyElastic", "NeoHookean", "ElasticPerfectlyPlastic", "IsotropicHardeningPlastic",
                          "MohrCoulomb", "SoftenMohrCoulomb", "DruckerPrager", "ModifiedCamClay", "CohesiveModifiedCamClay", "SoilStructureInteraction", "UserDefined"]
            if self.constitutive_model == "None" or self.constitutive_model == "RigidBody":
                return RigidBody(sims)
            elif self.constitutive_model == "HenckyElastic":
                if sims.stabilize == "B-Bar Method":
                    raise RuntimeError("B bar method is unsupported in HenckyElastic material")
                return HenckyElastic(sims)
            elif self.constitutive_model == "NeoHookean":
                if sims.stabilize == "B-Bar Method":
                    raise RuntimeError("B bar method is unsupported in NeoHookean material")
                return NeoHookean(sims)
            elif self.constitutive_model == "LinearElastic":
                return LinearElastic(sims)
            elif self.constitutive_model == "ElasticPerfectlyPlastic":
                return ElasticPerfectlyPlastic(sims)
            elif self.constitutive_model == "IsotropicHardeningPlastic":
                return IsotropicHardeningPlastic(sims)
            elif self.constitutive_model == "MohrCoulomb":
                return WillianMohrCoulomb(sims)
            elif self.constitutive_model == "SoftenMohrCoulomb":
                return MohrCoulomb(sims)
            elif self.constitutive_model == "StateDependentMohrCoulomb":
                return StateDependentMohrCoulomb(sims)
            elif self.constitutive_model == "DruckerPrager":
                return DruckerPrager(sims)
            elif self.constitutive_model == "ModifiedCamClay":
                return ModifiedCamClay(sims)
            elif self.constitutive_model == "CohesiveModifiedCamClay":
                return CohesiveModifiedCamClay(sims)
            elif self.constitutive_model == "SoilStructureInteraction":
                if sims.solver_type == "Implicit":
                    raise RuntimeError("Only /Explicit/ /ULMPM/ supports soil-structure interaction model")
                return SoilStructureInteraction(sims)
            elif self.constitutive_model == "UserDefined":
                return UserDefined(sims)
            else:
                raise ValueError(f'Constitutive Model: {self.constitutive_model} error! Only the following is aviliable:\n{model_type}')
        elif sims.material_type == "Fluid":
            if sims.configuration =="TLMPM":
                raise RuntimeError("Only /Explicit/ /ULMPM/ supports fluid model")
            
            model_type = ["Newtonian", "Bingham", "FluidStructureInteraction", "UserDefined"]
            if self.constitutive_model == "Newtonian":
                return Newtonian(sims)
            elif self.constitutive_model == "Bingham":
                return Bingham(sims)
            elif self.constitutive_model == "FluidStructureInteraction":
                return FluidStructureInteraction(sims)
            else:
                raise ValueError(f'Constitutive Model: {self.constitutive_model} error! Only the following is aviliable:\n{model_type}')
        elif sims.material_type == "TwoPhaseSingleLayer":
            if sims.configuration =="TLMPM":
                raise RuntimeError("Only /Explicit/ /ULMPM/ supports fluid model")
            if sims.solver_type == "Implicit":
                raise RuntimeError("Only /Explicit/ /ULMPM/ supports fluid model")
            
            model_type = ["LinearElastic"]
            if self.constitutive_model == "LinearElastic":
                return LinearElastic(sims)
            else:
                raise ValueError(f'Constitutive Model: {self.constitutive_model} error! Only the following is aviliable:\n{model_type}')



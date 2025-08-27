import taichi as ti
import numpy as np

from src.mpm.Contact import ContactBase
from src.mpm.Simulation import Simulation
from src.physics_model.consititutive_model.ConstitutiveModelBase import ConstitutiveBase
from src.utils.linalg import read_dict_list
from src.utils.ObjectIO import DictIO


class MaterialHandle(ConstitutiveBase):
    def __init__(self, sims: Simulation):
        super().__init__()
        self.material_parameters = []
        self.mapping = None
        self.materialID_numpy = None
        self.materialID = ti.field(int, shape=sims.max_particle_num)

    def initialize(self, parameter, sims: Simulation, material_model):
        materialID = DictIO.GetEssential(parameter, 'MaterialID')
        self.check_materialID(materialID)
        material_struct = self.material_handle(sims, material_model)
        material_struct.initialize_coupling()
        if sims.random_field:
            material_struct.random_field_initialize(parameter)
        else:
            material_struct.model_initialize(parameter)
        self.stateDict.update(material_struct.get_state_vars())
        material_struct.print_message(materialID)
        if materialID >= self.matProps.size():
            self.matProps += [material_struct]
        else:
            self.matProps[materialID] = material_struct
        self.material_parameters.append(parameter)

    def contact_parameter_initialize(self, parameter, contact: ContactBase):
        materialID = DictIO.GetEssential(parameter, 'materialID')
        self.matProps[materialID].members_update(**contact.get_parameters(parameter))

    def setup_contact(self, contact: ContactBase):
        if contact is not None and contact.name == "DEMContact":
            read_dict_list(contact.contact_phys, self.contact_parameter_initialize, contact=contact)

    def setup(self, sims: Simulation, contact: ContactBase, material_model, parameters):
        if self.matProps.size() == 0:
            temp_mat = self.material_handle(sims, constitutive_model="RigidBody")
            temp_mat.model_initialize({"Density": 2650})
            self.matProps += [temp_mat]
        read_dict_list(parameters, self.initialize, sims=sims, material_model=material_model)
        self.setup_contact(contact)
        if self.stiffness_matrix is None:
            if sims.solver_type == "Implicit" and sims.material_type == "Solid":
                self.stiffness_matrix = ti.Matrix.field(6, 6, float, shape=sims.max_particle_num)

    def get_unified_configuration(self, input_string):
        return next((keyword for keyword in ["UL", "TL"] if keyword in input_string), None)

    def material_handle(self, sims: Simulation, constitutive_model):
        from src.physics_model.consititutive_model.infinitesimal_strain.LinearElastic import LinearElasticModel
        from src.physics_model.consititutive_model.infinitesimal_strain.ElasticPerfectlyPlastic import ElasticPerfectlyPlasticModel
        from src.physics_model.consititutive_model.infinitesimal_strain.MohrCoulomb import MohrCoulombModel
        from src.physics_model.consititutive_model.infinitesimal_strain.StateDependentMohrCoulomb import StateDependentMohrCoulombModel
        from src.physics_model.consititutive_model.infinitesimal_strain.DruckerPrager import DruckerPragerModel
        from src.physics_model.consititutive_model.infinitesimal_strain.ModifiedCamClay import ModifiedCamClayModel
        from src.physics_model.consititutive_model.strain_rate.Newtonian import NewtonianModel
        from src.physics_model.consititutive_model.strain_rate.Bingham import BinghamModel
        from src.physics_model.consititutive_model.UserDefined import UserDefined
        if constitutive_model == "None" or constitutive_model == "RigidBody":
            from src.physics_model.consititutive_model.RigidBody import RigidModel
            return RigidModel(material_type=sims.material_type, configuration=self.get_unified_configuration(sims.configuration), solver_type=sims.solver_type)
        
        if sims.material_type == "Solid" or sims.material_type == "TwoPhaseSingleLayer" or sims.material_type == "TwoPhaseDoubleLayer":
            model_type = ["LinearElastic", "HenckyElastic", "NeoHookean", "ElasticPerfectlyPlastic", "MohrCoulomb", "DruckerPrager", "ModifiedCamClay", "CohesiveModifiedCamClay", "UserDefined"]
            if sims.material_type == "TwoPhaseSingleLayer" or sims.material_type == "TwoPhaseDoubleLayer":
                if sims.configuration =="TLMPM":
                    raise RuntimeError("Only /Explicit/ /ULMPM/ supports two phase model")
                if sims.solver_type == "Implicit":
                    raise RuntimeError("Only /Explicit/ /ULMPM/ supports two phase model")
                
            if constitutive_model == "HenckyElastic":
                if sims.stabilize == "B-Bar Method":
                    raise RuntimeError("B bar method is unsupported in HenckyElastic material")
                from src.physics_model.consititutive_model.finite_strain.HenckyElastic import HenckyElasticModel
                return HenckyElasticModel(material_type=sims.material_type, configuration=self.get_unified_configuration(sims.configuration), solver_type=sims.solver_type)
            elif constitutive_model == "NeoHookean":
                if sims.stabilize == "B-Bar Method":
                    raise RuntimeError("B bar method is unsupported in NeoHookean material")
                from src.physics_model.consititutive_model.finite_strain.NeoHookean import NeoHookeanModel
                return NeoHookeanModel(material_type=sims.material_type, configuration=self.get_unified_configuration(sims.configuration), solver_type=sims.solver_type)
            elif constitutive_model == "MooneyRivlin":
                if sims.stabilize == "B-Bar Method":
                    raise RuntimeError("B bar method is unsupported in NeoHookean material")
                from src.physics_model.consititutive_model.finite_strain.MooneyRivlin import MooneyRivlin
                return MooneyRivlin(material_type=sims.material_type, configuration=self.get_unified_configuration(sims.configuration), solver_type=sims.solver_type)
            elif constitutive_model == "Gent":
                if sims.stabilize == "B-Bar Method":
                    raise RuntimeError("B bar method is unsupported in NeoHookean material")
                from src.physics_model.consititutive_model.finite_strain.Gent import Gent
                return Gent(material_type=sims.material_type, configuration=self.get_unified_configuration(sims.configuration), solver_type=sims.solver_type)
            elif constitutive_model == "Hydrogel":
                if sims.stabilize == "B-Bar Method":
                    raise RuntimeError("B bar method is unsupported in NeoHookean material")
                from src.physics_model.consititutive_model.finite_strain.Hydrogel import Hydrogel
                return Hydrogel(material_type=sims.material_type, configuration=self.get_unified_configuration(sims.configuration), solver_type=sims.solver_type)
            elif constitutive_model == "LinearElastic":
                return LinearElasticModel(material_type=sims.material_type, configuration=self.get_unified_configuration(sims.configuration), solver_type=sims.solver_type)
            elif constitutive_model == "ElasticPerfectlyPlastic":
                return ElasticPerfectlyPlasticModel(material_type=sims.material_type, configuration=self.get_unified_configuration(sims.configuration), solver_type=sims.solver_type, stress_integration=sims.stress_integration)
            elif constitutive_model == "MohrCoulomb":
                return MohrCoulombModel(material_type=sims.material_type, configuration=self.get_unified_configuration(sims.configuration), solver_type=sims.solver_type, stress_integration=sims.stress_integration)
            elif constitutive_model == "StateDependentMohrCoulomb":
                return StateDependentMohrCoulombModel(material_type=sims.material_type, configuration=self.get_unified_configuration(sims.configuration), solver_type=sims.solver_type, stress_integration=sims.stress_integration)
            elif constitutive_model == "DruckerPrager":
                return DruckerPragerModel(material_type=sims.material_type, configuration=self.get_unified_configuration(sims.configuration), solver_type=sims.solver_type, stress_integration=sims.stress_integration)
            elif constitutive_model == "ModifiedCamClay":
                return ModifiedCamClayModel(material_type=sims.material_type, configuration=self.get_unified_configuration(sims.configuration), solver_type=sims.solver_type, stress_integration=sims.stress_integration)
            elif constitutive_model == "UserDefined":
                return UserDefined(material_type=sims.material_type, configuration=self.get_unified_configuration(sims.configuration), solver_type=sims.solver_type)
            else:
                raise ValueError(f'Constitutive Model: {constitutive_model} error! Only the following is aviliable:\n{model_type}')
        elif sims.material_type == "Fluid":
            if sims.configuration =="TLMPM":
                raise RuntimeError("Only /Explicit/ /ULMPM/ supports fluid model")
            
            model_type = ["Newtonian", "Bingham", "UserDefined"]
            if constitutive_model == "Newtonian":
                return NewtonianModel(material_type=sims.material_type, configuration=self.get_unified_configuration(sims.configuration), solver_type=sims.solver_type)
            elif constitutive_model == "Bingham":
                return BinghamModel(material_type=sims.material_type, configuration=self.get_unified_configuration(sims.configuration), solver_type=sims.solver_type)
            else:
                raise ValueError(f'Constitutive Model: {constitutive_model} error! Only the following is aviliable:\n{model_type}')
            
    def activate_state_variables(self, sims: Simulation):
        if self.stateDict and self.stateVars is None:
            self.stateVars = ti.Struct.field(self.stateDict, shape=sims.max_particle_num)

    def update_material_mapping(self, particle, particleNum):
        self.materialID_numpy = np.ascontiguousarray(particle.materialID.to_numpy()[0:particleNum], dtype=np.int32)
        self.mapping = np.bincount(self.materialID_numpy)
        np.cumsum(self.mapping, out=self.mapping)
        new_positions = np.pad(np.argsort(self.materialID_numpy, kind='stable'), (0, self.materialID.shape[0] - particleNum), mode='constant', constant_values=0)
        self.materialID.from_numpy(new_positions)

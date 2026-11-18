from src.utils.ObjectIO import DictIO
from src.utils.errors import ConfigurationError

import taichi as ti
import numpy as np
from src.mpm.BaseKernel import rebuild_material_mapping
from src.mpm.Contact import ContactBase
from src.mpm.Simulation import Simulation
from src.physics_model.constitutive_model.ConstitutiveModelBase import ConstitutiveBase
from src.utils.FieldIO import field_to_numpy_prefix


class MaterialHandle(ConstitutiveBase):
    def __init__(self, sims: Simulation):
        super().__init__()
        self.material_parameters = []
        self.mapping = None
        self._materialID_numpy = None
        self._material_id_particle = None
        self._material_id_count = 0
        self.materialID = ti.field(int, shape=sims.max_particle_num)
        self._mapping_capacity = max(1, getattr(sims, "max_material_num", 0))
        self.material_counts = None
        self.material_cursor = None
        self.material_ends = None
        self.material_block_counts = None
        self.material_block_cursor = None
        self.material_mapping_block_size = 256

    @property
    def materialID_numpy(self):
        if self._materialID_numpy is None and self._material_id_particle is not None:
            self._materialID_numpy = np.ascontiguousarray(
                field_to_numpy_prefix(self._material_id_particle.materialID, self._material_id_count), dtype=np.int32
            )
        return self._materialID_numpy

    @materialID_numpy.setter
    def materialID_numpy(self, value):
        self._materialID_numpy = value

    def initialize(self, parameter, sims: Simulation, material_model):
        material_id = DictIO.get_essential(parameter, "MaterialID")
        self.check_material_id(material_id)
        material_struct = self.material_handle(sims, material_model)
        material_struct.initialize_coupling()
        if sims.random_field:
            material_struct.random_field_initialize(parameter)
        else:
            material_struct.model_initialize(parameter)
        self.stateDict.update(material_struct.get_state_vars())
        material_struct.console_solver_name = "MPM"
        material_struct.print_message(material_id)
        if material_id >= self.matProps.size():
            self.matProps += [material_struct]
        else:
            self.matProps[material_id] = material_struct
        self.material_parameters.append(parameter)

    def contact_parameter_initialize(self, parameter, contact: ContactBase):
        material_id = DictIO.get_essential(parameter, "materialID")
        if material_id <= 0:
            raise ValueError("DEMContact materialID must identify a deformable material")
        # Materials may be registered in separate add_material calls.
        if material_id < self.matProps.size():
            self.matProps[material_id].members_update(**contact.get_parameters(parameter))

    def setup_contact(self, contact: ContactBase):
        if contact is not None and contact.name == "DEMContact":
            DictIO.read_dict_list(contact.contact_phys, self.contact_parameter_initialize, contact=contact)

    def setup(self, sims: Simulation, contact: ContactBase, material_model, parameters):
        if self.matProps.size() == 0:
            temp_mat = self.material_handle(sims, constitutive_model="RigidBody")
            temp_mat.model_initialize({"Density": 2650})
            self.matProps += [temp_mat]
        DictIO.read_dict_list(parameters, self.initialize, sims=sims, material_model=material_model)
        self.setup_contact(contact)
        if self.stiffness_matrix is None:
            if sims.solver_type == "Implicit" and sims.material_type == "Solid":
                self.stiffness_matrix = ti.Matrix.field(6, 6, float, shape=sims.max_particle_num)

    def get_unified_configuration(self, input_string):
        return next((keyword for keyword in ["UL", "TL"] if keyword in input_string), None)

    def material_handle(self, sims: Simulation, constitutive_model):
        from src.physics_model.constitutive_model.infinitesimal_strain.LinearElastic import LinearElasticModel
        from src.physics_model.constitutive_model.infinitesimal_strain.ElasticPerfectlyPlastic import (
            ElasticPerfectlyPlasticModel,
        )
        from src.physics_model.constitutive_model.infinitesimal_strain.MohrCoulomb import MohrCoulombModel
        from src.physics_model.constitutive_model.infinitesimal_strain.StateDependentMohrCoulomb import (
            StateDependentMohrCoulombModel,
        )
        from src.physics_model.constitutive_model.infinitesimal_strain.DruckerPrager import DruckerPragerModel
        from src.physics_model.constitutive_model.infinitesimal_strain.ModifiedCamClay import ModifiedCamClayModel
        from src.physics_model.constitutive_model.infinitesimal_strain.GranularMaterial import GranularMaterial
        from src.physics_model.constitutive_model.infinitesimal_strain.SanisandMS import SanisandMSModel
        from src.physics_model.constitutive_model.infinitesimal_strain.NorSandModel import NorSandModel
        from src.physics_model.constitutive_model.strain_rate.Newtonian import NewtonianModel
        from src.physics_model.constitutive_model.strain_rate.Bingham import BinghamModel
        from src.physics_model.constitutive_model.UserDefined import UserDefined

        if constitutive_model == "None" or constitutive_model == "RigidBody":
            from src.physics_model.constitutive_model.RigidBody import RigidModel

            return RigidModel(
                material_type=sims.material_type,
                configuration=self.get_unified_configuration(sims.configuration),
                solver_type=sims.solver_type,
            )

        if (
            sims.material_type == "Solid"
            or sims.material_type == "TwoPhaseSingleLayer"
            or sims.material_type == "TwoPhaseDoubleLayer"
        ):
            model_type = [
                "LinearElastic",
                "HenckyElastic",
                "NeoHookean",
                "ElasticPerfectlyPlastic",
                "MohrCoulomb",
                "DruckerPrager",
                "ModifiedCamClay",
                "GranularMaterial",
                "SanisandMS",
                "NorSand",
                "UserDefined",
            ]
            if sims.material_type == "TwoPhaseSingleLayer" or sims.material_type == "TwoPhaseDoubleLayer":
                if sims.configuration == "TLMPM":
                    raise ConfigurationError("Only /Explicit/ /ULMPM/ supports two phase model")
                if sims.solver_type == "Implicit":
                    raise ConfigurationError("Only /Explicit/ /ULMPM/ supports two phase model")

            if constitutive_model == "HenckyElastic":
                if sims.stabilize == "B-Bar Method":
                    raise ConfigurationError("B bar method is unsupported in HenckyElastic material")
                from src.physics_model.constitutive_model.finite_strain.HenckyElastic import HenckyElasticModel

                return HenckyElasticModel(
                    material_type=sims.material_type,
                    configuration=self.get_unified_configuration(sims.configuration),
                    solver_type=sims.solver_type,
                )
            elif constitutive_model == "NeoHookean":
                if sims.stabilize == "B-Bar Method":
                    raise ConfigurationError("B bar method is unsupported in NeoHookean material")
                from src.physics_model.constitutive_model.finite_strain.NeoHookean import NeoHookeanModel

                return NeoHookeanModel(
                    material_type=sims.material_type,
                    configuration=self.get_unified_configuration(sims.configuration),
                    solver_type=sims.solver_type,
                )
            elif constitutive_model == "MooneyRivlin":
                if sims.stabilize == "B-Bar Method":
                    raise ConfigurationError("B bar method is unsupported in NeoHookean material")
                from src.physics_model.constitutive_model.finite_strain.MooneyRivlin import MooneyRivlin

                return MooneyRivlin(
                    material_type=sims.material_type,
                    configuration=self.get_unified_configuration(sims.configuration),
                    solver_type=sims.solver_type,
                )
            elif constitutive_model == "Gent":
                if sims.stabilize == "B-Bar Method":
                    raise ConfigurationError("B bar method is unsupported in NeoHookean material")
                from src.physics_model.constitutive_model.finite_strain.Gent import Gent

                return Gent(
                    material_type=sims.material_type,
                    configuration=self.get_unified_configuration(sims.configuration),
                    solver_type=sims.solver_type,
                )
            elif constitutive_model == "Hydrogel":
                if sims.stabilize == "B-Bar Method":
                    raise ConfigurationError("B bar method is unsupported in NeoHookean material")
                from src.physics_model.constitutive_model.finite_strain.Hydrogel import Hydrogel

                return Hydrogel(
                    material_type=sims.material_type,
                    configuration=self.get_unified_configuration(sims.configuration),
                    solver_type=sims.solver_type,
                )
            elif constitutive_model == "LinearElastic":
                return LinearElasticModel(
                    material_type=sims.material_type,
                    configuration=self.get_unified_configuration(sims.configuration),
                    solver_type=sims.solver_type,
                )
            elif constitutive_model == "ElasticPerfectlyPlastic":
                return ElasticPerfectlyPlasticModel(
                    material_type=sims.material_type,
                    configuration=self.get_unified_configuration(sims.configuration),
                    solver_type=sims.solver_type,
                    stress_integration=sims.stress_integration,
                )
            elif constitutive_model == "MohrCoulomb":
                return MohrCoulombModel(
                    material_type=sims.material_type,
                    configuration=self.get_unified_configuration(sims.configuration),
                    solver_type=sims.solver_type,
                    stress_integration=sims.stress_integration,
                )
            elif constitutive_model == "StateDependentMohrCoulomb":
                return StateDependentMohrCoulombModel(
                    material_type=sims.material_type,
                    configuration=self.get_unified_configuration(sims.configuration),
                    solver_type=sims.solver_type,
                    stress_integration=sims.stress_integration,
                )
            elif constitutive_model == "DruckerPrager":
                return DruckerPragerModel(
                    material_type=sims.material_type,
                    configuration=self.get_unified_configuration(sims.configuration),
                    solver_type=sims.solver_type,
                    stress_integration=sims.stress_integration,
                )
            elif constitutive_model == "ModifiedCamClay":
                return ModifiedCamClayModel(
                    material_type=sims.material_type,
                    configuration=self.get_unified_configuration(sims.configuration),
                    solver_type=sims.solver_type,
                    stress_integration=sims.stress_integration,
                )
            elif constitutive_model == "GranularMaterial":
                return GranularMaterial(
                    material_type=sims.material_type,
                    configuration=self.get_unified_configuration(sims.configuration),
                    solver_type=sims.solver_type,
                    stress_integration=sims.stress_integration,
                )
            elif constitutive_model == "SanisandMS":
                return SanisandMSModel(
                    material_type=sims.material_type,
                    configuration=self.get_unified_configuration(sims.configuration),
                    solver_type=sims.solver_type,
                    stress_integration=sims.stress_integration,
                )
            elif constitutive_model == "NorSand":
                return NorSandModel(
                    material_type=sims.material_type,
                    configuration=self.get_unified_configuration(sims.configuration),
                    solver_type=sims.solver_type,
                    stress_integration=sims.stress_integration,
                )
            elif constitutive_model == "UserDefined":
                return UserDefined(
                    material_type=sims.material_type,
                    configuration=self.get_unified_configuration(sims.configuration),
                    solver_type=sims.solver_type,
                )
            else:
                raise ValueError(
                    f"Constitutive Model: {constitutive_model} error! Only the following is aviliable:\n{model_type}"
                )
        elif sims.material_type == "Fluid":
            if sims.configuration == "TLMPM":
                raise ConfigurationError("Only /Explicit/ /ULMPM/ supports fluid model")

            model_type = ["Newtonian", "Bingham", "UserDefined"]
            if constitutive_model == "Newtonian":
                return NewtonianModel(
                    material_type=sims.material_type,
                    configuration=self.get_unified_configuration(sims.configuration),
                    solver_type=sims.solver_type,
                )
            elif constitutive_model == "Bingham":
                return BinghamModel(
                    material_type=sims.material_type,
                    configuration=self.get_unified_configuration(sims.configuration),
                    solver_type=sims.solver_type,
                )
            else:
                raise ValueError(
                    f"Constitutive Model: {constitutive_model} error! Only the following is aviliable:\n{model_type}"
                )

    def activate_state_variables(self, sims: Simulation):
        if self.stateDict and self.stateVars is None:
            self.stateVars = ti.Struct.field(self.stateDict, shape=sims.max_particle_num)

    def update_material_mapping(self, particle, particleNum):
        self.materialID_numpy = np.ascontiguousarray(
            field_to_numpy_prefix(particle.materialID, particleNum), dtype=np.int32
        )
        material_count = np.bincount(self.materialID_numpy)
        if self.material_counts is None or self.material_counts.shape[0] < material_count.size:
            capacity = max(self._mapping_capacity, material_count.size)
            self.material_counts = ti.field(int, shape=capacity)
            self.material_cursor = ti.field(int, shape=capacity)
            self.material_ends = ti.field(int, shape=capacity)
            blocks = (
                self.materialID.shape[0] + self.material_mapping_block_size - 1
            ) // self.material_mapping_block_size
            self.material_block_counts = ti.field(int, shape=(blocks, capacity))
            self.material_block_cursor = ti.field(int, shape=(blocks, capacity))
        non_rigid_materials = np.flatnonzero(material_count[1:]) + 1

        if non_rigid_materials.size == 1:
            material_id = int(non_rigid_materials[0])
            if material_count[0] == 0:
                non_rigid_positions = np.arange(particleNum, dtype=np.int32)
            else:
                non_rigid_positions = np.flatnonzero(self.materialID_numpy == material_id).astype(np.int32)

            self.mapping = np.zeros(material_id + 1, dtype=material_count.dtype)
            self.mapping[material_id] = non_rigid_positions.size
            new_positions = np.zeros(self.materialID.shape[0], dtype=np.int32)
            new_positions[: non_rigid_positions.size] = non_rigid_positions
            self.materialID.from_numpy(new_positions)
            return

        self.mapping = np.cumsum(material_count)
        new_positions = np.pad(
            np.argsort(self.materialID_numpy, kind="stable"),
            (0, self.materialID.shape[0] - particleNum),
            mode="constant",
            constant_values=0,
        )
        self.materialID.from_numpy(new_positions)

    def update_material_mapping_device(self, particle, particleNum):
        rebuild_material_mapping(
            particleNum,
            particle,
            self.material_counts,
            self.material_cursor,
            self.material_ends,
            self.materialID,
            self.material_block_counts,
            self.material_block_cursor,
            self.material_mapping_block_size,
        )
        # Only the per-material launch bounds return to Python; particle IDs remain on device.
        for material_id in range(self.mapping.shape[0]):
            self.mapping[material_id] = self.material_ends[material_id]
        self._material_id_particle = particle
        self._material_id_count = particleNum
        self._materialID_numpy = None

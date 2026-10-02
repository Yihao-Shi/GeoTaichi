import taichi as ti
import numpy as np

from src.mpm.Contact import ContactBase
from src.mpm.Simulation import Simulation
from src.physics_model.consititutive_model.ConstitutiveModelBase import ConstitutiveBase
from src.utils.FieldIO import field_to_numpy_prefix
from src.utils.linalg import read_dict_list
from src.utils.MatrixFunction import flatten_matrix
from src.utils.ObjectIO import DictIO
from src.utils.constants import Threshold


@ti.data_oriented
class SoftParticleSingleMaterialAdapter(object):
    """Give legacy soft constitutive models the material-indexed interface."""

    def __init__(self, model):
        self.model = model
        self.is_finite_strain_plastic = bool(getattr(model, "is_finite_strain_plastic", False))

    @ti.func
    def soft_particle_pk1(self, material_id, deformation_gradient):
        return self.model.soft_particle_pk1(deformation_gradient)

    @ti.func
    def Psi(self, material_id, deformation_gradient):
        return self.model.Psi(deformation_gradient)

    @ti.func
    def dPsi_div_dF(self, material_id, deformation_gradient):
        return self.model.dPsi_div_dF(deformation_gradient)

    @ti.func
    def d2Psi_div_d2F(self, material_id, deformation_gradient):
        return self.model.d2Psi_div_d2F(deformation_gradient)

    @ti.func
    def update_soft_particle_state(self, np, material_id, deformation_gradient, pk1_stress, state_vars):
        self.model.update_soft_particle_state(np, deformation_gradient, pk1_stress, state_vars)

    @ti.func
    def trial_deformation_gradient(self, particle_id, material_id, deformation_gradient, displacement_gradient):
        return deformation_gradient + displacement_gradient

    @ti.func
    def Psi_at(self, particle_id, material_id, deformation_gradient):
        value = 0.0
        if ti.static(self.is_finite_strain_plastic):
            value = self.model.total_strain_energy_density_at(particle_id, deformation_gradient)
        else:
            value = self.model.Psi(deformation_gradient)
        return value

    @ti.func
    def dPsi_div_dF_at(self, particle_id, material_id, deformation_gradient):
        value = ti.Vector.zero(float, 9)
        if ti.static(self.is_finite_strain_plastic):
            value = self.model.total_dPsi_div_dF_at(particle_id, deformation_gradient)
        else:
            value = self.model.dPsi_div_dF(deformation_gradient)
        return value

    @ti.func
    def d2Psi_div_d2F_at(self, particle_id, material_id, deformation_gradient):
        value = ti.Matrix.zero(float, 9, 9)
        if ti.static(self.is_finite_strain_plastic):
            value = self.model.total_d2Psi_div_d2F_at(particle_id, deformation_gradient)
        else:
            value = self.model.d2Psi_div_d2F(deformation_gradient)
        return value

    @ti.func
    def commit_soft_particle_state(self, particle_id, material_id, deformation_gradient, state_vars):
        committed = ti.Matrix.zero(float, 3, 3)
        stress = ti.Matrix.zero(float, 3, 3)
        if ti.static(self.is_finite_strain_plastic):
            committed = self.model.commit_total_state(particle_id, deformation_gradient)
            stress = self.model.total_first_piola_stress_at(particle_id, committed)
        else:
            committed = deformation_gradient
            stress = self.model.soft_particle_pk1(deformation_gradient)
        self.model.update_soft_particle_state(particle_id, committed, stress, state_vars)
        return committed, stress


@ti.data_oriented
class SoftParticleNeoHookeanMaterialTable(object):
    """Material-indexed Neo-Hookean parameters for mixed soft bodies."""

    def __init__(self, capacity, models):
        self.shear = ti.field(float, shape=capacity)
        self.lame = ti.field(float, shape=capacity)
        shear = np.zeros(capacity, dtype=np.float64)
        lame = np.zeros(capacity, dtype=np.float64)
        for material_id, model in models.items():
            shear[material_id] = model.shear
            lame[material_id] = model.lame_lambda
        self.shear.from_numpy(shear)
        self.lame.from_numpy(lame)

    @ti.func
    def soft_particle_pk1(self, material_id, deformation_gradient):
        mu = self.shear[material_id]
        la = self.lame[material_id]
        det_f = ti.max(deformation_gradient.determinant(), Threshold)
        inverse_transpose = deformation_gradient.inverse().transpose()
        return mu * (deformation_gradient - inverse_transpose) + la * ti.log(det_f) * inverse_transpose

    @ti.func
    def Psi(self, material_id, deformation_gradient):
        mu = self.shear[material_id]
        la = self.lame[material_id]
        det_f = ti.max(deformation_gradient.determinant(), Threshold)
        log_j = ti.log(det_f)
        i1 = 0.0
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                i1 += deformation_gradient[i, j] * deformation_gradient[i, j]
        return 0.5 * mu * (i1 - 3.0) - mu * log_j + 0.5 * la * log_j * log_j

    @ti.func
    def dPsi_div_dF(self, material_id, deformation_gradient):
        return flatten_matrix(self.soft_particle_pk1(material_id, deformation_gradient))

    @ti.func
    def d2Psi_div_d2F(self, material_id, deformation_gradient):
        mu = self.shear[material_id]
        la = self.lame[material_id]
        det_f = ti.max(deformation_gradient.determinant(), Threshold)
        log_j = ti.log(det_f)
        inverse_transpose = deformation_gradient.inverse().transpose()
        tangent = ti.Matrix.zero(float, 9, 9)
        for row in range(9):
            for column in range(9):
                a = row // 3
                i = row - 3 * a
                b = column // 3
                j = column - 3 * b
                diagonal = 0.0
                if i == j and a == b:
                    diagonal = mu
                tangent[row, column] = (
                    diagonal
                    + la * inverse_transpose[i, a] * inverse_transpose[j, b]
                    - (la * log_j - mu) * inverse_transpose[i, b] * inverse_transpose[j, a]
                )
        return tangent

    @ti.func
    def update_soft_particle_state(self, np, material_id, deformation_gradient, pk1_stress, state_vars):
        det_f = ti.max(deformation_gradient.determinant(), Threshold)
        cauchy = pk1_stress @ deformation_gradient.transpose() / det_f
        state_vars[np].estress = ti.sqrt(
            ti.max(
                0.5
                * (
                    (cauchy[0, 0] - cauchy[1, 1]) ** 2
                    + (cauchy[1, 1] - cauchy[2, 2]) ** 2
                    + (cauchy[0, 0] - cauchy[2, 2]) ** 2
                )
                + 3.0 * (cauchy[0, 1] ** 2 + cauchy[1, 2] ** 2 + cauchy[0, 2] ** 2),
                0.0,
            )
        )

    @ti.func
    def trial_deformation_gradient(self, particle_id, material_id, deformation_gradient, displacement_gradient):
        return deformation_gradient + displacement_gradient

    @ti.func
    def Psi_at(self, particle_id, material_id, deformation_gradient):
        return self.Psi(material_id, deformation_gradient)

    @ti.func
    def dPsi_div_dF_at(self, particle_id, material_id, deformation_gradient):
        return self.dPsi_div_dF(material_id, deformation_gradient)

    @ti.func
    def d2Psi_div_d2F_at(self, particle_id, material_id, deformation_gradient):
        return self.d2Psi_div_d2F(material_id, deformation_gradient)

    @ti.func
    def commit_soft_particle_state(self, particle_id, material_id, deformation_gradient, state_vars):
        stress = self.soft_particle_pk1(material_id, deformation_gradient)
        self.update_soft_particle_state(particle_id, material_id, deformation_gradient, stress, state_vars)
        return deformation_gradient, stress


class MaterialHandle(ConstitutiveBase):
    def __init__(self, sims: Simulation):
        super().__init__()
        self.material_parameters = []
        self.mapping = None
        self.materialID_numpy = None
        self.materialID = ti.field(int, shape=sims.max_particle_num)

    def initialize(self, parameter, sims: Simulation, material_model):
        materialID = DictIO.GetEssential(parameter, "MaterialID")
        self.check_materialID(materialID)
        material_struct = self.material_handle(sims, material_model)
        material_struct.initialize_coupling()
        if sims.random_field:
            material_struct.random_field_initialize(parameter)
        else:
            material_struct.model_initialize(parameter)
        self.stateDict.update(material_struct.get_state_vars())
        material_struct.console_solver_name = "MPM"
        material_struct.print_message(materialID)
        if materialID >= self.matProps.size():
            self.matProps += [material_struct]
        else:
            self.matProps[materialID] = material_struct
        self.material_parameters.append(parameter)

    def contact_parameter_initialize(self, parameter, contact: ContactBase):
        materialID = DictIO.GetEssential(parameter, "materialID")
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
        from src.physics_model.consititutive_model.infinitesimal_strain.ElasticPerfectlyPlastic import (
            ElasticPerfectlyPlasticModel,
        )
        from src.physics_model.consititutive_model.infinitesimal_strain.MohrCoulomb import MohrCoulombModel
        from src.physics_model.consititutive_model.infinitesimal_strain.StateDependentMohrCoulomb import (
            StateDependentMohrCoulombModel,
        )
        from src.physics_model.consititutive_model.infinitesimal_strain.DruckerPrager import DruckerPragerModel
        from src.physics_model.consititutive_model.infinitesimal_strain.ModifiedCamClay import ModifiedCamClayModel
        from src.physics_model.consititutive_model.infinitesimal_strain.GranularMaterial import GranularMaterial
        from src.physics_model.consititutive_model.infinitesimal_strain.SanisandMS import SanisandMSModel
        from src.physics_model.consititutive_model.infinitesimal_strain.NorSandModel import NorSandModel
        from src.physics_model.consititutive_model.strain_rate.Newtonian import NewtonianModel
        from src.physics_model.consititutive_model.strain_rate.Bingham import BinghamModel
        from src.physics_model.consititutive_model.UserDefined import UserDefined

        if constitutive_model == "None" or constitutive_model == "RigidBody":
            from src.physics_model.consititutive_model.RigidBody import RigidModel

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
                    raise RuntimeError("Only /Explicit/ /ULMPM/ supports two phase model")
                if sims.solver_type == "Implicit":
                    raise RuntimeError("Only /Explicit/ /ULMPM/ supports two phase model")

            if constitutive_model == "HenckyElastic":
                if sims.stabilize == "B-Bar Method":
                    raise RuntimeError("B bar method is unsupported in HenckyElastic material")
                from src.physics_model.consititutive_model.finite_strain.HenckyElastic import HenckyElasticModel

                return HenckyElasticModel(
                    material_type=sims.material_type,
                    configuration=self.get_unified_configuration(sims.configuration),
                    solver_type=sims.solver_type,
                )
            elif constitutive_model == "NeoHookean":
                if sims.stabilize == "B-Bar Method":
                    raise RuntimeError("B bar method is unsupported in NeoHookean material")
                from src.physics_model.consititutive_model.finite_strain.NeoHookean import NeoHookeanModel

                return NeoHookeanModel(
                    material_type=sims.material_type,
                    configuration=self.get_unified_configuration(sims.configuration),
                    solver_type=sims.solver_type,
                )
            elif constitutive_model == "MooneyRivlin":
                if sims.stabilize == "B-Bar Method":
                    raise RuntimeError("B bar method is unsupported in NeoHookean material")
                from src.physics_model.consititutive_model.finite_strain.MooneyRivlin import MooneyRivlin

                return MooneyRivlin(
                    material_type=sims.material_type,
                    configuration=self.get_unified_configuration(sims.configuration),
                    solver_type=sims.solver_type,
                )
            elif constitutive_model == "Gent":
                if sims.stabilize == "B-Bar Method":
                    raise RuntimeError("B bar method is unsupported in NeoHookean material")
                from src.physics_model.consititutive_model.finite_strain.Gent import Gent

                return Gent(
                    material_type=sims.material_type,
                    configuration=self.get_unified_configuration(sims.configuration),
                    solver_type=sims.solver_type,
                )
            elif constitutive_model == "Hydrogel":
                if sims.stabilize == "B-Bar Method":
                    raise RuntimeError("B bar method is unsupported in NeoHookean material")
                from src.physics_model.consititutive_model.finite_strain.Hydrogel import Hydrogel

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
                raise RuntimeError("Only /Explicit/ /ULMPM/ supports fluid model")

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


class SoftParticleMaterialManager(object):
    def __init__(self):
        self.materialID = None
        self.matProps = None
        self.stateVars = None

    def setup(self, scene, sims, allow_plastic=False):
        if allow_plastic:
            raise ValueError(
                "Soft-particle materials are hyperelastic-only; "
                "Drucker-Prager/Von-Mises plasticity is available only on the "
                "ordinary Direct MPM path."
            )
        soft_material_ids = self._get_soft_material_ids(scene)
        if len(soft_material_ids) == 0:
            raise RuntimeError("Soft particle material setup requires at least one soft body.")
        material_parameters = getattr(scene, "material_parameters", {})
        models = {}
        model_names = {}
        for materialID in soft_material_ids:
            if materialID not in material_parameters:
                raise RuntimeError(
                    f"Soft particle material {materialID} was not initialized. "
                    "Call DEM.add_attribute(materialID, ...) before creating soft particles."
                )
            parameter = dict(material_parameters[materialID])
            parameter["MaterialID"] = materialID
            model_name = self._get_model_name(parameter)
            self._normalize_parameter(parameter, model_name)
            model = self._material_handle(model_name)
            model.model_initialize(parameter)
            if getattr(model, "is_finite_strain_plastic", False):
                raise ValueError(
                    "Soft-particle materials are hyperelastic-only; "
                    f"material {materialID} uses plastic model {model_name}."
                )
            model.print_message(materialID)
            models[materialID] = model
            model_names[materialID] = model_name

        self.materialID = tuple(soft_material_ids)
        if len(models) == 1:
            model = next(iter(models.values()))
            self.matProps = SoftParticleSingleMaterialAdapter(model)
            state_dict = model.define_soft_particle_state_vars()
        else:
            unsupported = {
                material_id: model_name for material_id, model_name in model_names.items() if model_name != "NeoHookean"
            }
            if unsupported:
                raise RuntimeError(
                    "Mixed LSMPM soft materials currently require all soft "
                    f"materials to be Neo-Hookean; received {unsupported}"
                )
            self.matProps = SoftParticleNeoHookeanMaterialTable(
                max(int(sims.max_material_num), max(soft_material_ids) + 1),
                models,
            )
            state_dict = {"estress": float}
        self.stateVars = ti.Struct.field(state_dict, shape=sims.max_material_point_num)

    def _get_soft_material_ids(self, scene):
        soft_num = int(scene.softNum[0])
        if soft_num <= 0:
            return []
        material_ids = scene.soft.materialID.to_numpy()[:soft_num]
        unique_ids = []
        for materialID in material_ids:
            materialID = int(materialID)
            if materialID not in unique_ids:
                unique_ids.append(materialID)
        return unique_ids

    def _get_model_name(self, parameter, allow_plastic=False):
        model = DictIO.GetOptional(parameter, "ConstitutiveModel")
        if model is None:
            model = DictIO.GetOptional(parameter, "MaterialModel")
        if model is None:
            model = DictIO.GetOptional(parameter, "SoftConstitutiveModel")
        if model is None:
            model = "NeoHookean"
        return self._normalize_model_name(model, allow_plastic)

    def _normalize_model_name(self, model, allow_plastic=False):
        model_key = str(model).replace("-", "").replace("_", "").replace(" ", "").lower()
        model_map = {
            "neohookean": "NeoHookean",
            "neohookeanmodel": "NeoHookean",
            "hencky": "HenckyElastic",
            "henckyelastic": "HenckyElastic",
            "henckyelasticmodel": "HenckyElastic",
            "mooneyrivlin": "MooneyRivlin",
            "mooneyrivlinmodel": "MooneyRivlin",
            "gent": "Gent",
            "gentmodel": "Gent",
            "hydrogel": "Hydrogel",
            "hydrogelmodel": "Hydrogel",
        }
        if model_key in {
            "druckerprager",
            "druckerpragermodel",
            "dp",
            "vonmises",
            "vonmisesmodel",
            "vm",
        }:
            raise ValueError(
                "Soft-particle materials are hyperelastic-only; "
                "Drucker-Prager/Von-Mises plasticity is available only on the "
                "ordinary Direct MPM path."
            )
        if model_key not in model_map:
            raise ValueError(f"Unsupported LSMPM soft finite-strain constitutive model: {model}")
        return model_map[model_key]

    def _normalize_parameter(self, parameter, model_name):
        if DictIO.GetOptional(parameter, "YoungModulus") is None:
            elastic_modulus = DictIO.GetOptional(parameter, "ElasticModulus")
            if elastic_modulus is not None:
                parameter["YoungModulus"] = elastic_modulus

        if model_name == "MooneyRivlin":
            coeff = DictIO.GetOptional(parameter, "Coefficient")
            if coeff is None:
                young = DictIO.GetEssential(parameter, "YoungModulus")
                poisson = DictIO.GetAlternative(parameter, "PoissonRatio", 0.3)
                shear = 0.5 * young / (1.0 + poisson)
                parameter["Coefficient"] = [[0.0, 0.0], [0.5 * shear, 0.0]]
            elif isinstance(coeff, (float, int)):
                parameter["Coefficient"] = [[0.0, 0.0], [float(coeff), 0.0]]
            elif len(coeff) == 2 and not hasattr(coeff[0], "__len__"):
                parameter["Coefficient"] = [[0.0, float(coeff[1])], [float(coeff[0]), 0.0]]

        if model_name == "Gent" and DictIO.GetOptional(parameter, "Tensile1") is None:
            tensile = DictIO.GetOptional(parameter, "Tensile")
            if tensile is not None:
                parameter["Tensile1"] = tensile

        if model_name == "Hydrogel" and DictIO.GetOptional(parameter, "Tensile") is None:
            tensile = DictIO.GetOptional(parameter, "Tensile1")
            if tensile is not None:
                parameter["Tensile"] = tensile

    def _material_handle(self, model_name):
        if model_name == "NeoHookean":
            from src.physics_model.consititutive_model.finite_strain.NeoHookean import NeoHookeanModel

            return NeoHookeanModel()
        if model_name == "HenckyElastic":
            from src.physics_model.consititutive_model.finite_strain.HenckyElastic import HenckyElasticModel

            return HenckyElasticModel()
        if model_name == "MooneyRivlin":
            from src.physics_model.consititutive_model.finite_strain.MooneyRivlin import MooneyRivlin

            return MooneyRivlin()
        if model_name == "Gent":
            from src.physics_model.consititutive_model.finite_strain.Gent import Gent

            return Gent()
        if model_name == "Hydrogel":
            from src.physics_model.consititutive_model.finite_strain.Hydrogel import Hydrogel

            return Hydrogel()
        if model_name == "DruckerPrager":
            from src.physics_model.consititutive_model.finite_strain.DruckerPrager import (
                DruckerPragerModel,
            )

            return DruckerPragerModel()
        raise ValueError(f"Unsupported LSMPM soft finite-strain constitutive model: {model_name}")

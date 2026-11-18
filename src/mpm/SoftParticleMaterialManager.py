from src.utils.errors import ConfigurationError, SimulationStateError
from src.mpm.MaterialManager import SoftParticleSingleMaterialAdapter
from src.mpm.SoftParticleNeoHookeanMaterialTable import SoftParticleNeoHookeanMaterialTable

import taichi as ti
from src.utils.ObjectIO import DictIO


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
            raise SimulationStateError("Soft particle material setup requires at least one soft body.")
        material_parameters = getattr(scene, "material_parameters", {})
        models = {}
        model_names = {}
        for material_id in soft_material_ids:
            if material_id not in material_parameters:
                raise SimulationStateError(
                    f"Soft particle material {material_id} was not initialized. "
                    "Call DEM.add_attribute(materialID, ...) before creating soft particles."
                )
            parameter = dict(material_parameters[material_id])
            parameter["MaterialID"] = material_id
            model_name = self._get_model_name(parameter)
            self._normalize_parameter(parameter, model_name)
            model = self._material_handle(model_name)
            model.model_initialize(parameter)
            if getattr(model, "is_finite_strain_plastic", False):
                raise ValueError(
                    "Soft-particle materials are hyperelastic-only; "
                    f"material {material_id} uses plastic model {model_name}."
                )
            model.print_message(material_id)
            models[material_id] = model
            model_names[material_id] = model_name

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
                raise ConfigurationError(
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
        for material_id in material_ids:
            material_id = int(material_id)
            if material_id not in unique_ids:
                unique_ids.append(material_id)
        return unique_ids

    def _get_model_name(self, parameter, allow_plastic=False):
        model = DictIO.get_optional(parameter, "ConstitutiveModel")
        if model is None:
            model = DictIO.get_optional(parameter, "MaterialModel")
        if model is None:
            model = DictIO.get_optional(parameter, "SoftConstitutiveModel")
        if model is None:
            model = "NeoHookean"
        return self._normalize_model_name(model, allow_plastic)

    def _normalize_model_name(self, model, allow_plastic=False):
        if model in ("DruckerPrager", "VonMises"):
            raise ValueError(
                "Soft-particle materials are hyperelastic-only; Drucker-Prager/Von-Mises plasticity is available only on the ordinary Direct MPM path."
            )
        if model not in ("NeoHookean", "HenckyElastic", "MooneyRivlin", "Gent", "Hydrogel"):
            raise ValueError(f"Unsupported LSMPM soft finite-strain constitutive model: {model}")
        return model

    def _normalize_parameter(self, parameter, model_name):
        if DictIO.get_optional(parameter, "YoungModulus") is None:
            elastic_modulus = DictIO.get_optional(parameter, "ElasticModulus")
            if elastic_modulus is not None:
                parameter["YoungModulus"] = elastic_modulus

        if model_name == "MooneyRivlin":
            coeff = DictIO.get_optional(parameter, "Coefficient")
            if coeff is None:
                young = DictIO.get_essential(parameter, "YoungModulus")
                poisson = DictIO.get_alternative(parameter, "PoissonRatio", 0.3)
                shear = 0.5 * young / (1.0 + poisson)
                parameter["Coefficient"] = [[0.0, 0.0], [0.5 * shear, 0.0]]
            elif isinstance(coeff, (float, int)):
                parameter["Coefficient"] = [[0.0, 0.0], [float(coeff), 0.0]]
            elif len(coeff) == 2 and not hasattr(coeff[0], "__len__"):
                parameter["Coefficient"] = [[0.0, float(coeff[1])], [float(coeff[0]), 0.0]]

        if model_name == "Gent" and DictIO.get_optional(parameter, "Tensile1") is None:
            tensile = DictIO.get_optional(parameter, "Tensile")
            if tensile is not None:
                parameter["Tensile1"] = tensile

        if model_name == "Hydrogel" and DictIO.get_optional(parameter, "Tensile") is None:
            tensile = DictIO.get_optional(parameter, "Tensile1")
            if tensile is not None:
                parameter["Tensile"] = tensile

    def _material_handle(self, model_name):
        if model_name == "NeoHookean":
            from src.physics_model.constitutive_model.finite_strain.NeoHookean import NeoHookeanModel

            return NeoHookeanModel()
        if model_name == "HenckyElastic":
            from src.physics_model.constitutive_model.finite_strain.HenckyElastic import HenckyElasticModel

            return HenckyElasticModel()
        if model_name == "MooneyRivlin":
            from src.physics_model.constitutive_model.finite_strain.MooneyRivlin import MooneyRivlin

            return MooneyRivlin()
        if model_name == "Gent":
            from src.physics_model.constitutive_model.finite_strain.Gent import Gent

            return Gent()
        if model_name == "Hydrogel":
            from src.physics_model.constitutive_model.finite_strain.Hydrogel import Hydrogel

            return Hydrogel()
        if model_name == "DruckerPrager":
            from src.physics_model.constitutive_model.finite_strain.DruckerPrager import (
                DruckerPragerModel,
            )

            return DruckerPragerModel()
        raise ValueError(f"Unsupported LSMPM soft finite-strain constitutive model: {model_name}")

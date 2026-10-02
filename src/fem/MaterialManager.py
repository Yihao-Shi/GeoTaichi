"""Finite-element material selection following GeoTaichi manager APIs."""


def _canonical_parameters(parameters):
    aliases = {
        "youngsmodulus": "young_modulus",
        "youngmodulus": "young_modulus",
        "elasticmodulus": "young_modulus",
        "poissonsratio": "poisson_ratio",
        "poissonratio": "poisson_ratio",
        "minimumjacobian": "minimum_jacobian",
        "stretchstiffness": "stretch_stiffness",
        "compressionstiffness": "compression_stiffness",
        "compressstiffness": "compression_stiffness",
        "bendingstiffness": "bending_stiffness",
        "bendingmodulus": "bending_stiffness",
        "bendingpoissonratio": "bending_poisson_ratio",
        "bendingmodel": "bending_model",
        "initialstress": "initial_stress",
    }
    canonical = {}
    for name, value in parameters.items():
        key = str(name).replace("_", "").replace("-", "").replace(" ", "").lower()
        canonical[aliases.get(key, name)] = value
    return canonical


def _incremental_material_parameters(parameters):
    """Translate FEM-style keywords to the shared incremental-solid API."""
    aliases = {
        "density": "Density",
        "youngmodulus": "YoungModulus",
        "poissonratio": "PoissonRatio",
        "yieldstress": "YieldStress",
        "residualyieldstress": "ResidualYieldStress",
        "plasticdevstrain": "PlasticDevStrain",
        "residualplasticdevstrain": "ResidualPlasticDevStrain",
        "cohesion": "Cohesion",
        "residualcohesion": "ResidualCohesion",
        "friction": "Friction",
        "staticfriction": "StaticFriction",
        "residualfriction": "ResidualFriction",
        "dilation": "Dilation",
        "residualdilation": "ResidualDilation",
        "tensile": "Tensile",
        "dptype": "dpType",
        "softtype": "SoftType",
        "softenparameter": "SoftenParameter",
        "ratedependent": "RateDependent",
        "stressratio": "StressRatio",
        "lambda": "lambda",
        "overconsolidationratio": "OverConsolidationRatio",
        "consolidationpressure": "ConsolidationPressure",
        "averagediameter": "AverageDiameter",
        "inertialnumber": "InertialNumber",
        "graindensity": "GrainDensity",
        "dynamicfriction": "DynamicFriction",
        "criticalstateratio": "CriticalStateRatio",
        "loderatio": "LodeRatio",
        "referencevoidratio": "ReferenceVoidRatio",
        "yieldsurfacesize": "YieldSurfaceSize",
        "initialvoidratio": "InitialVoidRatio",
        "maximumvoidratio": "MaximumVoidRatio",
        "minimumvoidratio": "MinimumVoidRatio",
        "dilatancybeta": "DilatancyBeta",
        "criticalspecificvolume": "CriticalSpecificVolume",
        "initialspecificvolume": "InitialSpecificVolume",
        "hardeningmodulus": "HardeningModulus",
    }
    translated = {}
    for name, value in parameters.items():
        key = str(name).replace("_", "").replace("-", "").replace(" ", "").lower()
        translated[aliases.get(key, name)] = value
    return translated


class FEMMaterialManager:
    def __init__(self):
        self.material = None

    def material_handle(self, model="StVK", **parameters):
        key = str(model).strip().replace("_", "").replace("-", "").replace(" ", "").lower()
        parameters = _canonical_parameters(parameters)
        if key in (
            "stvk",
            "saintvenantkirchhoff",
            "linear",
            "linearelastic",
            "elastic",
        ):
            from src.physics_model.consititutive_model.finite_strain.StVenantKirchhoff import (
                StVenantKirchhoffModel,
            )

            parameters.setdefault("density", 1.0)
            parameters.setdefault("thickness", 1.0)
            return StVenantKirchhoffModel().initialize_from_kwargs(**parameters)
        if key in ("neohookean", "compressibleneohookean", "nh"):
            from src.physics_model.consititutive_model.finite_strain.NeoHookean import (
                NeoHookeanModel,
            )

            parameters.setdefault("density", 1.0)
            parameters.setdefault("thickness", 1.0)
            return NeoHookeanModel().initialize_from_kwargs(**parameters)
        if key in ("cloth", "clotharap", "arap"):
            from src.physics_model.consititutive_model.finite_strain.Cloth import (
                ClothARAP,
            )

            if "young_modulus" in parameters and "stretch_stiffness" not in parameters:
                parameters["stretch_stiffness"] = parameters.pop("young_modulus")
            parameters.pop("poisson_ratio", None)
            parameters.pop("shear_stiffness", None)
            return ClothARAP(**parameters)
        if key in ("clothneohookean", "clothnh"):
            from src.physics_model.consititutive_model.finite_strain.Cloth import (
                ClothNeoHookean,
            )

            parameters.pop("shear_stiffness", None)
            return ClothNeoHookean(**parameters)
        incremental_model = None
        if key in (
            "elasticperfectlyplastic",
            "vonmises",
            "j2plastic",
            "j2plasticity",
        ):
            from src.physics_model.consititutive_model.infinitesimal_strain.ElasticPerfectlyPlastic import (
                ElasticPerfectlyPlasticModel,
            )

            incremental_model = ElasticPerfectlyPlasticModel
        elif key in ("mohrcoulomb", "mc"):
            from src.physics_model.consititutive_model.infinitesimal_strain.MohrCoulomb import (
                MohrCoulombModel,
            )

            incremental_model = MohrCoulombModel
        elif key in ("druckerprager", "dp"):
            from src.physics_model.consititutive_model.infinitesimal_strain.DruckerPrager import (
                DruckerPragerModel,
            )

            incremental_model = DruckerPragerModel
        elif key in ("statedependentmohrcoulomb", "sdmc"):
            from src.physics_model.consititutive_model.infinitesimal_strain.StateDependentMohrCoulomb import (
                StateDependentMohrCoulombModel,
            )

            incremental_model = StateDependentMohrCoulombModel
        elif key in ("modifiedcamclay", "mcc"):
            from src.physics_model.consititutive_model.infinitesimal_strain.ModifiedCamClay import (
                ModifiedCamClayModel,
            )

            incremental_model = ModifiedCamClayModel
        elif key in ("granularmaterial", "granular"):
            from src.physics_model.consititutive_model.infinitesimal_strain.GranularMaterial import (
                GranularMaterial,
            )

            incremental_model = GranularMaterial
        elif key in ("sanisandms", "sanisand"):
            from src.physics_model.consititutive_model.infinitesimal_strain.SanisandMS import (
                SanisandMSModel,
            )

            incremental_model = SanisandMSModel
        elif key in ("norsand", "norsandmodel"):
            from src.physics_model.consititutive_model.infinitesimal_strain.NorSandModel import (
                NorSandModel,
            )

            incremental_model = NorSandModel
        if incremental_model is not None:
            initial_stress = parameters.pop("initial_stress", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            stress_integration = parameters.pop("stress_integration", None)
            arguments = {
                "material_type": "Solid",
                "configuration": "UL",
                "solver_type": "Explicit",
            }
            if stress_integration is not None:
                arguments["stress_integration"] = stress_integration
            created = incremental_model(**arguments)
            created.model_initialize(_incremental_material_parameters(parameters))
            created.fem_initial_stress = tuple(initial_stress)
            if len(created.fem_initial_stress) != 6:
                raise ValueError("HEX8 elastoplastic initial_stress must contain six Voigt components")
            return self._mark_incremental_hex8(created)
        raise ValueError(
            f"Unsupported FEM material model {model!r}; expected StVK, "
            "NeoHookean, ClothARAP, ClothNeoHookean, "
            "ElasticPerfectlyPlastic, MohrCoulomb, DruckerPrager, "
            "StateDependentMohrCoulomb, ModifiedCamClay, "
            "GranularMaterial, SanisandMS, or NorSand"
        )

    @staticmethod
    def _mark_incremental_hex8(model):
        model.is_fem_elastoplastic = True
        model.thickness = 1.0
        bulk = float(getattr(model, "bulk", 0.0))
        shear = float(getattr(model, "shear", 0.0))
        model.lame_parameters = (
            bulk - 2.0 * shear / 3.0,
            shear,
        )
        return model

    def add_material(self, model="StVK", material=None, **parameters):
        self.material = material if material is not None else self.material_handle(model, **parameters)
        return self.material


__all__ = ["FEMMaterialManager"]

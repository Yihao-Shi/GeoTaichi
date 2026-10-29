"""Valid deterministic material inputs shared by solver adapter unit tests."""

from copy import deepcopy

COMMON = {"Density": 1800.0, "YoungModulus": 2.0e5, "PoissonRatio": 0.3}
FINITE_ELASTIC = ("NeoHookean", "HenckyElastic", "MooneyRivlin", "Gent", "Hydrogel")
INCREMENTAL = (
    "ElasticPerfectlyPlastic",
    "MohrCoulomb",
    "DruckerPrager",
    "StateDependentMohrCoulomb",
    "ModifiedCamClay",
    "GranularMaterial",
    "SanisandMS",
    "NorSand",
)
FLUIDS = ("Newtonian", "Bingham")
NATIVE_MPM = ("LinearElastic", *FINITE_ELASTIC, *INCREMENTAL, *FLUIDS)
DIRECT_EXPLICIT_MPM = tuple(name for name in NATIVE_MPM if name not in INCREMENTAL[5:])

_PARAMETERS = {
    "ElasticPerfectlyPlastic": {"YieldStress": 1000.0},
    "MohrCoulomb": {"Cohesion": 1000.0, "Friction": 30.0, "Dilation": 0.0},
    "DruckerPrager": {"Cohesion": 1000.0, "Friction": 30.0, "Dilation": 0.0},
    "StateDependentMohrCoulomb": {
        "e0": 0.7,
        "e_Tao": 0.9,
        "lambda_c": 0.02,
        "ksi": 0.7,
        "nd": 1.0,
        "nf": 1.0,
        "fai_c": 30.0,
    },
    "ModifiedCamClay": {
        "StressRatio": 1.2,
        "lambda": 0.2,
        "kappa": 0.04,
        "pc": 2e5,
        "void_ratio_ref": 0.8,
        "void_ratio_initial": 0.7,
    },
    "GranularMaterial": {"StaticFriction": 30.0, "AverageDiameter": 1e-3, "InertialNumber": 0.3},
    "SanisandMS": {
        "G0": 125.0,
        "Mc": 1.25,
        "c": 0.7,
        "lambda_c": 0.02,
        "e0": 0.8,
        "ksi": 0.7,
        "m": 0.05,
        "h0": 7.0,
        "ch": 0.9,
        "nb": 1.1,
        "A0": 0.5,
        "nd": 1.0,
        "zeta": 0.0,
        "mu0": 0.0,
        "beta": 1.0,
        "e_init": 0.7,
        "emax": 1.0,
        "emin": 0.5,
    },
    "NorSand": {
        "G0": 1e5,
        "kappa": 0.04,
        "lambda": 0.2,
        "M": 1.2,
        "N": 0.3,
        "beta": 1.0,
        "vc0": 1.9,
        "v0": 1.7,
        "h": 100.0,
    },
    "MooneyRivlin": {"Coefficient": [[0.0, 0.0], [COMMON["YoungModulus"] / (4 * 1.3), 0.0]]},
    "Gent": {"Tensile1": 30.0},
    "Hydrogel": {"Tensile": 30.0},
    "Newtonian": {"Modulus": 2e5, "Viscosity": 1.0, "cL": 0.0, "cQ": 0.0},
    "Bingham": {"Modulus": 2e5, "Viscosity": 1.0, "YieldStress": 0.1, "CriticalStrainRate": 1e-8},
}


def material_parameters(model):
    return deepcopy({**COMMON, **_PARAMETERS.get(model, {})})


def initial_stress(model):
    pressure = 1e5 if model in INCREMENTAL[3:] else 0.0
    return [-pressure, -pressure, -pressure, 0.0, 0.0, 0.0]

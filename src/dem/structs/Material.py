import taichi as ti
from src.utils.ObjectIO import DictIO

SOFT_CONSTITUTIVE_MODEL_NAMES = {
    0: "Neo-Hookean",
    1: "Hencky Elastic",
    2: "Mooney-Rivlin",
    3: "Gent",
    4: "Hydrogel",
    5: "Drucker-Prager",
}


@ti.dataclass
class Material:

    density: float
    fdamp: float
    tdamp: float
    young: float
    poisson: float
    constitutive_model: int
    coeff00: float
    coeff01: float
    coeff10: float
    coeff11: float
    tensile1: float
    tensile2: float

    def add_attribute(self, attribute):
        density = DictIO.get_essential(attribute, "Density")
        fdamp = DictIO.get_alternative(attribute, "ForceLocalDamping", 0.7)
        tdamp = DictIO.get_alternative(attribute, "TorqueLocalDamping", 0.7)
        young = DictIO.get_alternative(
            attribute, "YoungModulus", DictIO.get_alternative(attribute, "ElasticModulus", 1.0e5)
        )
        poisson = DictIO.get_alternative(attribute, "PoissonRatio", 0.3)
        model = DictIO.get_optional(attribute, "ConstitutiveModel")
        if model is None:
            model = DictIO.get_optional(attribute, "MaterialModel")
        if model is None:
            model = DictIO.get_optional(attribute, "SoftConstitutiveModel")
        if model is None:
            model = "NeoHookean"
        coeff = DictIO.get_alternative(attribute, "Coefficient", None)
        tensile = DictIO.get_alternative(attribute, "Tensile", 0.0)
        tensile1 = DictIO.get_alternative(attribute, "Tensile1", tensile)
        tensile2 = DictIO.get_alternative(attribute, "Tensile2", 0.0)
        self.add_density(density)
        self.add_local_damping_coefficient(fdamp, tdamp)
        self.add_elasticity(young, poisson)
        self.add_constitutive_model(model)
        self.add_finite_strain_parameters(young, poisson, coeff, tensile1, tensile2)

    def add_density(self, density):
        self.density = density

    def add_local_damping_coefficient(self, fdamp, tdamp):
        self.fdamp = fdamp
        self.tdamp = tdamp

    def add_elasticity(self, young, poisson):
        self.young = young
        self.poisson = poisson

    def add_constitutive_model(self, model):
        models = ("NeoHookean", "HenckyElastic", "MooneyRivlin", "Gent", "Hydrogel", "DruckerPrager")
        if model not in models:
            raise ValueError(f"Unsupported LSMPM soft finite-strain constitutive model: {model}")
        self.constitutive_model = models.index(model)

    def add_finite_strain_parameters(self, young, poisson, coeff, tensile1, tensile2):
        shear = 0.5 * young / (1.0 + poisson)
        self.coeff00 = 0.0
        self.coeff01 = 0.0
        self.coeff10 = 0.5 * shear
        self.coeff11 = 0.0
        if coeff is not None:
            if isinstance(coeff, (float, int)):
                self.coeff10 = float(coeff)
            elif len(coeff) == 2 and not hasattr(coeff[0], "__len__"):
                self.coeff10 = float(coeff[0])
                self.coeff01 = float(coeff[1])
            elif len(coeff) == 2 and hasattr(coeff[0], "__len__") and len(coeff[0]) == 2 and len(coeff[1]) == 2:
                self.coeff00 = float(coeff[0][0])
                self.coeff01 = float(coeff[0][1])
                self.coeff10 = float(coeff[1][0])
                self.coeff11 = float(coeff[1][1])
            else:
                raise ValueError("Keyword:: /Coefficient/ should be scalar, [C10, C01], or [[C00, C01], [C10, C11]].")
        self.tensile1 = float(tensile1)
        self.tensile2 = float(tensile2)

    @ti.func
    def _get_density(self):
        return self.density

    @ti.func
    def _get_force_damping_coefficient(self):
        return self.fdamp

    @ti.func
    def _get_torque_damping_coefficient(self):
        return self.tdamp

    @ti.func
    def _get_young_modulus(self):
        return self.young

    @ti.func
    def _get_poisson_ratio(self):
        return self.poisson

    @ti.func
    def _get_constitutive_model(self):
        return self.constitutive_model

    def print_info(self, matID, model_name=None):
        if model_name is None:
            model_id = int(self.constitutive_model)
            model_name = SOFT_CONSTITUTIVE_MODEL_NAMES.get(model_id, f"Unknown ({model_id})")
        print("Constitutive model: ", model_name)
        print("Material ID: ", matID)
        print("Density = ", self.density)
        print("Local Damping (Force) = ", self.fdamp)
        print("Local Damping (Torque) = ", self.tdamp)
        print("Young Modulus = ", self.young)
        print("Poisson Ratio = ", self.poisson, "\n")

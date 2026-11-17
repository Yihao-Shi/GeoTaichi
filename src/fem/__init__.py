from importlib import import_module

__all__ = [
    "FEM",
    "FEMSimulation",
    "FEMScene",
    "FEMGenerateManager",
    "FEMMaterialManager",
    "FEMMesh",
    "DirichletBoundary",
    "NeumannBoundary",
    "FEMContact",
    "StVenantKirchhoffModel",
    "NeoHookeanModel",
    "ClothARAP",
    "ClothNeoHookean",
    "ExplicitFEM",
    "ImplicitFEM",
    "DifferentiableFEM",
    "ArmijoLineSearch",
]

_LAZY_EXPORTS = {
    "FEM": ("src.fem.mainFEM", "FEM"),
    "FEMSimulation": ("src.fem.Simulation", "FEMSimulation"),
    "FEMScene": ("src.fem.SceneManager", "FEMScene"),
    "FEMGenerateManager": ("src.fem.generator.MeshGenerator", "FEMGenerateManager"),
    "FEMMaterialManager": ("src.fem.MaterialManager", "FEMMaterialManager"),
    "FEMMesh": ("src.fem.generator.Mesh", "FEMMesh"),
    "DirichletBoundary": ("src.fem.boundaries.DirichletBoundary", "DirichletBoundary"),
    "NeumannBoundary": ("src.fem.boundaries.NeumannBoundary", "NeumannBoundary"),
    "FEMContact": ("src.fem.contact.ContactModel", "FEMContact"),
    "StVenantKirchhoffModel": (
        "src.physics_model.constitutive_model.finite_strain.StVenantKirchhoff",
        "StVenantKirchhoffModel",
    ),
    "NeoHookeanModel": ("src.physics_model.constitutive_model.finite_strain.NeoHookean", "NeoHookeanModel"),
    "ClothARAP": (
        "src.physics_model.constitutive_model.cloth_model.ClothARAP",
        "ClothARAP",
    ),
    "ClothNeoHookean": (
        "src.physics_model.constitutive_model.cloth_model.ClothNeoHookean",
        "ClothNeoHookean",
    ),
    "ExplicitFEM": ("src.fem.engines.ExplicitFEM", "ExplicitFEM"),
    "ImplicitFEM": ("src.fem.engines.ImplicitFEM", "ImplicitFEM"),
    "DifferentiableFEM": ("src.fem.engines.DifferentiableFEM", "DifferentiableFEM"),
    "ArmijoLineSearch": ("src.fem.engines.LineSearch", "ArmijoLineSearch"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attribute_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attribute_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

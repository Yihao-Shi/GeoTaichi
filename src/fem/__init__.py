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
    "FEMGenerateManager": ("src.fem.generator", "FEMGenerateManager"),
    "FEMMaterialManager": ("src.fem.MaterialManager", "FEMMaterialManager"),
    "FEMMesh": ("src.fem.generator", "FEMMesh"),
    "DirichletBoundary": ("src.fem.boundaries", "DirichletBoundary"),
    "NeumannBoundary": ("src.fem.boundaries", "NeumannBoundary"),
    "FEMContact": ("src.fem.contact", "FEMContact"),
    "StVenantKirchhoffModel": (
        "src.physics_model.consititutive_model.finite_strain",
        "StVenantKirchhoffModel",
    ),
    "NeoHookeanModel": (
        "src.physics_model.consititutive_model.finite_strain",
        "NeoHookeanModel",
    ),
    "ClothARAP": (
        "src.physics_model.consititutive_model.finite_strain",
        "ClothARAP",
    ),
    "ClothNeoHookean": (
        "src.physics_model.consititutive_model.finite_strain",
        "ClothNeoHookean",
    ),
    "ExplicitFEM": ("src.fem.engines", "ExplicitFEM"),
    "ImplicitFEM": ("src.fem.engines", "ImplicitFEM"),
    "DifferentiableFEM": ("src.fem.engines.DifferentiableFEM", "DifferentiableFEM"),
    "ArmijoLineSearch": ("src.fem.engines", "ArmijoLineSearch"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attribute_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attribute_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

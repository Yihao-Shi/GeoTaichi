from importlib import import_module

from src.iga.config import get_dimension, set_dimension

__all__ = [
    "IGA",
    "IGASimulation",
    "IGAScene",
    "IGAGenerateManager",
    "IGAMaterialManager",
    "ExplicitIGA",
    "ImplicitIGA",
    "Primitives",
    "Cube",
    "Tube",
    "Cylinder",
    "Rectangle",
    "Ring",
    "DirichletBoundary",
    "NeumannBoundary",
    "set_dimension",
    "get_dimension",
]

_LAZY_EXPORTS = {
    "IGA": ("src.iga.mainIGA", "IGA"),
    "IGASimulation": ("src.iga.Simulation", "IGASimulation"),
    "IGAScene": ("src.iga.SceneManager", "IGAScene"),
    "IGAGenerateManager": ("src.iga.GenerateManager", "IGAGenerateManager"),
    "IGAMaterialManager": ("src.iga.MaterialManager", "IGAMaterialManager"),
    "ExplicitIGA": ("src.iga.engines.ExplicitIGA", "ExplicitIGA"),
    "ImplicitIGA": ("src.iga.engines.ImplicitIGA", "ImplicitIGA"),
    "Primitives": ("src.iga.generator.Primitives", "Primitives"),
    "Cube": ("src.nurbs.Cube", "Cube"),
    "Tube": ("src.nurbs.Tube", "Tube"),
    "Cylinder": ("src.nurbs.Cylinder", "Cylinder"),
    "Rectangle": ("src.nurbs.Rectangle", "Rectangle"),
    "Ring": ("src.nurbs.Ring", "Ring"),
    "DirichletBoundary": ("src.iga.boundaries.BoundaryCondition", "DirichletBoundary"),
    "NeumannBoundary": ("src.iga.boundaries.BoundaryCondition", "NeumannBoundary"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

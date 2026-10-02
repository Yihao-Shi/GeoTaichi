from importlib import import_module

__all__ = [
    "Engine",
    "IncompressibleEngine",
    "TLExplicitEngine",
    "ULExplicitEngine",
    "ULExplicitTwoPhaseEngine",
    "ImplicitEngine",
    "ULSemiImplicitTwoPhaseDoubleLayerEngine",
    "ULSemiImplicitTwoPhaseEngine",
    "ULSemiImplicitTwoPhaseEngine_u_p",
]

_LAZY_EXPORTS = {
    "Engine": ("src.mpm.engines.Engine", "Engine"),
    "IncompressibleEngine": ("src.mpm.engines.IncompressibleEngine", "IncompressibleEngine"),
    "TLExplicitEngine": ("src.mpm.engines.TLExplicitEngine", "TLExplicitEngine"),
    "ULExplicitEngine": ("src.mpm.engines.ULExplicitEngine", "ULExplicitEngine"),
    "ULExplicitTwoPhaseEngine": ("src.mpm.engines.ULExplicitTwoPhaseEngine", "ULExplicitTwoPhaseEngine"),
    "ImplicitEngine": ("src.mpm.engines.ULImplicitEngine", "ImplicitEngine"),
    "ULSemiImplicitTwoPhaseDoubleLayerEngine": (
        "src.mpm.engines.ULSemiImplicitTwoPhaseDoubleLayerEngine",
        "ULSemiImplicitTwoPhaseDoubleLayerEngine",
    ),
    "ULSemiImplicitTwoPhaseEngine": ("src.mpm.engines.ULSemiImplicitTwoPhaseEngine", "ULSemiImplicitTwoPhaseEngine"),
    "ULSemiImplicitTwoPhaseEngine_u_p": (
        "src.mpm.engines.ULSemiImplicitTwoPhaseEngine_u_p",
        "ULSemiImplicitTwoPhaseEngine_u_p",
    ),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

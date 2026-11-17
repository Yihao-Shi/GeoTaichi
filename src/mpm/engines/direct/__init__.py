from importlib import import_module

__all__ = [
    "ExplicitMPM",
    "ExplicitTLMPM",
    "ExplicitULMPM",
    "ImplicitMPM",
    "ImplicitTLMPM",
    "ImplicitULMPM",
    "StaticTwoPhaseULMPM",
]

_LAZY_EXPORTS = {
    "ExplicitMPM": ("src.mpm.engines.direct.ExplicitMPM", "ExplicitMPM"),
    "ExplicitTLMPM": ("src.mpm.engines.direct.ExplicitTLMPM", "ExplicitTLMPM"),
    "ExplicitULMPM": ("src.mpm.engines.direct.ExplicitULMPM", "ExplicitULMPM"),
    "ImplicitMPM": ("src.mpm.engines.direct.ImplicitMPM", "ImplicitMPM"),
    "ImplicitTLMPM": ("src.mpm.engines.direct.ImplicitTLMPM", "ImplicitTLMPM"),
    "ImplicitULMPM": ("src.mpm.engines.direct.ImplicitULMPM", "ImplicitULMPM"),
    "StaticTwoPhaseULMPM": ("src.mpm.engines.direct.StaticTwoPhaseULMPM", "StaticTwoPhaseULMPM"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

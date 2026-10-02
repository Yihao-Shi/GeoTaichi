from importlib import import_module

from src.igampm.config import get_dimension, set_dimension

__all__ = [
    "IGAMPM",
    "CouplingContactSurface",
    "get_dimension",
    "set_dimension",
]

_LAZY_EXPORTS = {
    "IGAMPM": ("src.igampm.mainIGAMPM", "IGAMPM"),
    "CouplingContactSurface": ("src.igampm.contact.ContactSurface", "CouplingContactSurface"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

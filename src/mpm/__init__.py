from importlib import import_module


def MPM(*args, **kwargs):
    from src.mpm.mainMPM import MPM as _MPM

    return _MPM(*args, **kwargs)


__all__ = ["MPM", "DifferentiableMPM"]


def __getattr__(name):
    if name == "DifferentiableMPM":
        value = getattr(
            import_module("src.mpm.soft_particle.DifferentiableMPM"),
            name,
        )
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

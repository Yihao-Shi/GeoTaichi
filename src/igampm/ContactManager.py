import math

import src.igampm.config as config
from src.physics_model.contact_model.ipc import Barrier, Friction
from src.physics_model.contact_model.ipc.IPC import normalize_ipc_model
from src.igampm.contact.DEMContact import (
    HertzMindlinDEMContactModel,
    LinearDEMContactModel,
)


def _finite_nonnegative(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _finite_positive(value, name):
    result = _finite_nonnegative(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _friction_law_parameters(kwargs):
    """Normalize the public friction aliases exactly as shared IPC does."""
    legacy_mu = _finite_nonnegative(kwargs.get("mu", kwargs.get("dynamic_friction", 0.0)), "mu")
    mu_dynamic = _finite_nonnegative(
        kwargs.get("dynamic_friction", kwargs.get("mu_dynamic", legacy_mu)),
        "dynamic_friction",
    )
    mu_static = _finite_nonnegative(
        kwargs.get("static_friction", kwargs.get("mu_static", mu_dynamic)),
        "static_friction",
    )
    mu_viscous = _finite_nonnegative(
        kwargs.get("viscous_friction", kwargs.get("mu_viscous", 0.0)),
        "viscous_friction",
    )
    epsv = _finite_positive(kwargs.get("epsv", 1.0e-3), "epsv")
    stribeck_velocity = _finite_nonnegative(
        kwargs.get("stribeck_velocity", 10.0 * epsv),
        "stribeck_velocity",
    )
    profile_key = str(kwargs.get("friction_profile", "quadratic")).strip().replace("-", "_").replace(" ", "_").lower()
    profile_aliases = {
        "quadratic": 0,
        "c1": 0,
        "ipc": 0,
        "stabilized": 1,
        "stabilised": 1,
        "cinfinity": 1,
        "c_infinity": 1,
    }
    if profile_key not in profile_aliases:
        raise ValueError("friction_profile must be 'quadratic' or 'stabilized'")
    return {
        "mu": legacy_mu if "mu" in kwargs else mu_dynamic,
        "mu_dynamic": mu_dynamic,
        "mu_static": mu_static,
        "mu_viscous": mu_viscous,
        "stribeck_velocity": stribeck_velocity,
        "profile_id": profile_aliases[profile_key],
    }


def _positive_integer(value, name):
    try:
        numeric = float(value)
        result = int(numeric)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not math.isfinite(numeric) or numeric != result or result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _validate_iga_friction_configuration(kwargs):
    friction_mode = str(kwargs.get("friction_mode", "lagged")).strip().replace("-", "_").lower()
    aliases = {
        "lag": "lagged",
        "lagged": "lagged",
        "fullyimplicit": "fully_implicit",
        "fully_implicit": "fully_implicit",
    }
    if friction_mode not in aliases:
        raise ValueError("IGA-MPM IPC friction_mode must be 'lagged' or 'fully_implicit'")
    friction_mode = aliases[friction_mode]

    raw_iterations = kwargs.get(
        "friction_iterations",
        kwargs.get("friction_fixed_point_iterations", 1),
    )
    try:
        iterations = int(raw_iterations)
        exact_integer = float(raw_iterations) == iterations
    except (TypeError, ValueError, OverflowError):
        iterations = 0
        exact_integer = False
    if not exact_integer:
        raise ValueError("IGA-MPM IPC friction_iterations must be an integer")
    iterations = -1 if iterations <= 0 else iterations
    law = _friction_law_parameters(kwargs)
    if friction_mode == "lagged":
        if (
            law["mu"] != law["mu_dynamic"]
            or law["mu_dynamic"] != law["mu_static"]
            or law["mu_viscous"] != 0.0
            or law["profile_id"] != 0
        ):
            raise RuntimeError(
                "lagged IGA-MPM IPC friction supports only equal static/"
                "dynamic quadratic Coulomb friction; Stribeck, viscous, or "
                "stabilized-profile friction requires friction_mode="
                "'fully_implicit'"
            )
    else:
        if iterations != 1:
            raise RuntimeError(
                "friction_iterations is only defined for lagged friction " "and must equal 1 in fully implicit mode"
            )
        if law["mu_static"] != law["mu_dynamic"] and law["stribeck_velocity"] <= 0.0:
            raise RuntimeError(
                "fully implicit IGA-MPM IPC friction requires positive "
                "stribeck_velocity when static_friction differs from "
                "dynamic_friction"
            )
    # Match original IPC fricIterAmt in lagged mode: any non-positive value
    # means iterate until convergence (with GeoTaichi's explicit safety cap).
    return friction_mode, iterations


class ContactManager:
    def __init__(self, **kwargs):
        self._configuration_frozen = False
        self.contact_model = None
        self.barrier = None
        self.friction = None
        self.phys = None
        self.activate_barrier = False
        self.activate_friction = False
        self.friction_mode = "lagged"
        self.friction_iterations = 1
        self.friction_tolerance = 1.0e-7
        self.friction_max_iterations = 50
        self.choose_contact_model(**kwargs)

    def freeze_configuration(self):
        """Freeze mode and field ownership once an Engine has been built."""
        self._configuration_frozen = True

    def choose_contact_model(self, contact_model="IPC", **kwargs):
        if self._configuration_frozen:
            raise RuntimeError(
                "IGA-MPM contact configuration cannot be changed after build(); "
                "construct a new coupling engine instead"
            )
        contact_key = config.normalize_contact_model(contact_model)
        if contact_key in ("Linear", "HertzMindlin"):
            self.contact_model = contact_key
            self.barrier = None
            self.friction = None
            self.phys = LinearDEMContactModel() if contact_key == "Linear" else HertzMindlinDEMContactModel()
            self.activate_barrier = False
            self.activate_friction = False
            return
        if contact_key == "IPC":
            ipc_model = normalize_ipc_model(contact_model)
            friction_mode, friction_iterations = _validate_iga_friction_configuration(kwargs)
            self.contact_model = "IPC"
            self.friction_mode = friction_mode
            self.friction_iterations = friction_iterations
            self.friction_tolerance = _finite_nonnegative(
                kwargs.get(
                    "friction_tolerance",
                    kwargs.get("friction_fixed_point_tolerance", 1.0e-7),
                ),
                "friction_tolerance",
            )
            self.friction_max_iterations = _positive_integer(
                kwargs.get("friction_max_iterations", 50),
                "friction_max_iterations",
            )
            self.barrier = Barrier(ipc_model=ipc_model, **kwargs)
            self.friction = Friction(**kwargs)
            self.phys = None
            self.activate_barrier = True
            default_friction_active = getattr(
                self.friction,
                "has_friction",
                any(
                    float(kwargs.get(name, 0.0)) != 0.0
                    for name in (
                        "mu",
                        "dynamic_friction",
                        "static_friction",
                        "viscous_friction",
                    )
                ),
            )
            self.activate_friction = bool(kwargs.get("activate_friction", default_friction_active))
            return

        raise ValueError("IGA-MPM contact_model must be IPC, Linear, or HertzMindlin")

    def add_property(self, mpm_material, iga_body, parameters):
        if self._configuration_frozen:
            raise RuntimeError("IGA-MPM contact properties cannot be changed after build()")
        if self.phys is None:
            raise RuntimeError("add_property is available for explicit Linear/HertzMindlin " "IGA-MPM contact")
        self.phys.add_property(mpm_material, iga_body, parameters)

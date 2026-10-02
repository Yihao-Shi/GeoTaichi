"""IPC configuration for implicit FEM--MPM surface coupling."""

from __future__ import annotations

import math

import numpy as np

from src.fem.contact.ContactModel import FEMContact, _integer


def _canonical_property(parameters):
    aliases = {
        "activationdistance": "dhat",
        "barrierdistance": "dhat",
        "minimumdistance": "dmin",
        "contactdistance": "dmin",
        "barrierstiffness": "kappa",
        "contactstiffness": "kappa",
        "friction": "friction_coefficient",
        "mu": "friction_coefficient",
        "frictioncoefficient": "friction_coefficient",
        "velocityepsilon": "epsv",
        "projectpd": "project_pd",
        "ccdsafety": "ccd_safety",
        "ccdmaxiterations": "ccd_max_iterations",
    }
    result = {}
    for name, value in dict(parameters).items():
        key = str(name).replace("_", "").replace("-", "").replace(" ", "").lower()
        result[aliases.get(key, name)] = value
    return result


class IPCModel:
    """Validated IPC barrier and lagged-friction parameters.

    The first index of a pair property is an MPM body id for the Direct
    backend.  The historical ``MPMmaterial`` spelling remains accepted by the
    public coupling facade so explicit scripts do not need a second method.
    """

    model_name = "IPC"
    coupling_name = "FEMPM"
    first_body_name = "MPM"

    def __init__(self, simulation, **kwargs):
        self.simulation = simulation
        parameters = dict(kwargs)
        ipc_model = parameters.pop("ipc_model", parameters.pop("normal_contact_model", "BarrierIPC"))
        friction_mode_value = parameters.pop("friction_mode", "lagged")
        friction_iterations_value = parameters.pop("friction_iterations", 1)
        friction_max_iterations_value = parameters.pop("friction_max_iterations", 50)
        friction_tolerance_value = parameters.pop("friction_tolerance", 1.0e-7)
        parameters.pop("activate_friction", None)
        parameters.setdefault("broad_phase", simulation.search)
        parameters.setdefault("self_contact", False)
        self.contact = FEMContact.create(ipc_model, **parameters)
        if not self.contact.project_pd:
            raise ValueError(f"{self.coupling_name} implicit IPC requires " "project_pd=True")
        mode = str(friction_mode_value).replace("_", "").replace("-", "").replace(" ", "").lower()
        if mode not in ("lagged", "lag"):
            raise ValueError(
                f"{self.coupling_name} IPC currently supports "
                "friction_mode='lagged'; "
                "fully implicit friction is not yet available for the "
                "point--triangle/grid pullback"
            )
        self.friction_mode = "lagged"
        self.friction_iterations = _integer("friction_iterations", friction_iterations_value)
        if self.friction_iterations != -1 and self.friction_iterations <= 0:
            raise ValueError(f"{self.coupling_name} IPC friction_iterations must be -1 or positive")
        self.friction_max_iterations = _integer("friction_max_iterations", friction_max_iterations_value)
        if self.friction_max_iterations <= 0:
            raise ValueError(f"{self.coupling_name} IPC friction_max_iterations must be positive")
        self.friction_tolerance = float(friction_tolerance_value)
        if not math.isfinite(self.friction_tolerance) or self.friction_tolerance < 0.0:
            raise ValueError(f"{self.coupling_name} IPC friction_tolerance must be finite and non-negative")
        self.properties = {}

    @property
    def activate_friction(self):
        return any(entry.friction_coefficient > 0.0 for entry in self.properties.values()) or (
            not self.properties and self.contact.friction_coefficient > 0.0
        )

    def add_property(self, mpm_body, fem_body, parameters):
        values = {
            "model": self.contact.model,
            "broad_phase": self.contact.broad_phase,
            "dhat": self.contact.dhat,
            "dmin": self.contact.dmin,
            "kappa": self.contact.kappa,
            "penalty": self.contact.penalty,
            "self_contact": False,
            "project_pd": self.contact.project_pd,
            "ccd_safety": self.contact.ccd_safety,
            "ccd_max_iterations": self.contact.ccd_max_iterations,
            "friction_coefficient": self.contact.friction_coefficient,
            "epsv": self.contact.epsv,
        }
        values.update(_canonical_property(parameters))
        created = FEMContact.create(values.pop("model"), **values)
        if not created.project_pd:
            raise ValueError(f"{self.coupling_name} implicit IPC pair properties require " "project_pd=True")
        key = (int(mpm_body), int(fem_body))
        if min(key) < 0:
            raise ValueError(f"{self.coupling_name} IPC body ids must be non-negative")
        self.properties[key] = created
        return created

    def parameter_arrays(self, mpm_body_count, fem_body_count, characteristic):
        mpm_body_count = max(int(mpm_body_count), 1)
        fem_body_count = max(int(fem_body_count), 1)
        characteristic = float(characteristic)
        default_dhat = float(self.contact.dhat) if self.contact.dhat is not None else 1.0e-3 * max(characteristic, 1.0)
        default_dmin = 0.0 if self.contact.dmin is None else float(self.contact.dmin)
        shape = (mpm_body_count, fem_body_count)
        active = np.ones(shape, dtype=np.int32)
        kappa = np.full(shape, self.contact.kappa, dtype=np.float64)
        dhat = np.full(shape, default_dhat, dtype=np.float64)
        dmin = np.full(shape, default_dmin, dtype=np.float64)
        friction = np.full(shape, self.contact.friction_coefficient, dtype=np.float64)
        epsv = np.full(shape, self.contact.epsv, dtype=np.float64)
        penalty = np.full(shape, self.contact.penalty, dtype=np.float64)
        if self.properties:
            active.fill(0)
            for (mpm_body, fem_body), entry in self.properties.items():
                if mpm_body >= mpm_body_count or fem_body >= fem_body_count:
                    raise ValueError(
                        f"{self.coupling_name} IPC property body pair "
                        f"({mpm_body}, {fem_body}) exceeds available bodies "
                        f"({mpm_body_count}, {fem_body_count})"
                    )
                active[mpm_body, fem_body] = 1
                kappa[mpm_body, fem_body] = entry.kappa
                dhat[mpm_body, fem_body] = default_dhat if entry.dhat is None else entry.dhat
                dmin[mpm_body, fem_body] = default_dmin if entry.dmin is None else entry.dmin
                friction[mpm_body, fem_body] = entry.friction_coefficient
                epsv[mpm_body, fem_body] = entry.epsv
                penalty[mpm_body, fem_body] = entry.penalty
        if not np.any(active):
            raise ValueError(f"{self.coupling_name} IPC has no active " f"{self.first_body_name}-body/FEM-body pair")
        for name, values in (("dhat", dhat), ("kappa", kappa), ("epsv", epsv)):
            if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError(f"{self.coupling_name} IPC {name} values must be " "finite and positive")
        if np.any(~np.isfinite(dmin)) or np.any(dmin < 0.0):
            raise ValueError(f"{self.coupling_name} IPC dmin values must be finite and " "non-negative")
        search_distance = float(np.max((dmin + dhat)[active != 0]))
        if not math.isfinite(search_distance) or search_distance <= 0.0:
            raise ValueError(f"{self.coupling_name} IPC search distance must be positive")
        return {
            "active": active,
            "kappa": kappa,
            "dhat": dhat,
            "dmin": dmin,
            "friction": friction,
            "epsv": epsv,
            "penalty": penalty,
            "search_distance": search_distance,
            "ccd_safety": min([self.contact.ccd_safety] + [entry.ccd_safety for entry in self.properties.values()]),
            "ccd_max_iterations": max(
                [self.contact.ccd_max_iterations] + [entry.ccd_max_iterations for entry in self.properties.values()]
            ),
        }

    def critical_timestep(self, *_):
        return math.inf


__all__ = ["IPCModel"]

"""Configuration objects for FEM contact."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _positive(name, value, *, allow_zero=False):
    value = float(value)
    valid = value >= 0.0 if allow_zero else value > 0.0
    if not np.isfinite(value) or not valid:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {relation}")
    return value


def _integer(name, value):
    try:
        numeric = float(value)
        integer = int(numeric)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not np.isfinite(numeric) or numeric != integer:
        raise ValueError(f"{name} must be an integer")
    return integer


def _normalize_plane(entry):
    if isinstance(entry, dict):
        point = entry.get("point", entry.get("origin", (0.0, 0.0, 0.0)))
        normal = entry.get("normal")
    elif isinstance(entry, (tuple, list)) and len(entry) == 2:
        point, normal = entry
    else:
        raise TypeError("a contact plane must be {'point': ..., 'normal': ...} or (point, normal)")
    point = np.asarray(point, dtype=np.float64).reshape(-1)
    normal = np.asarray(normal, dtype=np.float64).reshape(-1)
    if point.size != 3 or normal.size != 3 or not np.all(np.isfinite(point)) or not np.all(np.isfinite(normal)):
        raise ValueError("contact plane point and normal must be finite three-vectors")
    length = float(np.linalg.norm(normal))
    if length <= 0.0:
        raise ValueError("contact plane normal cannot be zero")
    return point, normal / length


@dataclass
class FEMContact:
    """Validated FEM contact settings shared by IPC and augmented Lagrangian."""

    model: str = "IPC"
    broad_phase: str = "LinkedCell"
    dhat: float | None = None
    dmin: float | None = None
    kappa: float = 1.0e5
    self_contact: bool = True
    planes: tuple = field(default_factory=tuple)
    project_pd: bool = True
    ccd_safety: float = 0.9
    ccd_max_iterations: int = 64
    friction_coefficient: float = 0.0
    epsv: float = 1.0e-3
    friction_iterations: int = 1
    friction_max_iterations: int = 50
    friction_tolerance: float = 1.0e-7
    penalty: float = 1.0e5
    max_penalty: float = 1.0e12
    penalty_growth: float = 2.0
    penalty_update_interval: int = 50
    sufficient_reduction: float = 0.75
    constraint_tolerance: float = 1.0e-6
    point_triangle_coordination_number: float = 32.0
    edge_edge_coordination_number: float = 128.0
    max_point_triangle_pairs: int | None = None
    max_edge_edge_pairs: int | None = None
    body_pair: tuple | None = None
    pair_contacts: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        key = str(self.model).strip().replace("_", "").replace("-", "").replace(" ", "").lower()
        aliases = {
            "ipc": "IPC",
            "barrier": "IPC",
            "barrieripc": "IPC",
            "incrementalpotentialcontact": "IPC",
            "al": "AugmentedLagrangian",
            "semi": "AugmentedLagrangian",
            "semiipc": "AugmentedLagrangian",
            "augmentlagrangian": "AugmentedLagrangian",
            "augmentedlagrangian": "AugmentedLagrangian",
            "nonbarrier": "AugmentedLagrangian",
            "nonbarrieraugmentlagrangian": "AugmentedLagrangian",
            "nonbarrieraugmentedlagrangian": "AugmentedLagrangian",
        }
        if key not in aliases:
            raise ValueError("FEM contact model must be 'BarrierIPC' or 'SemiIPC'")
        self.model = aliases[key]
        broad_phase_key = str(self.broad_phase).strip().replace("_", "").replace("-", "").replace(" ", "").lower()
        broad_phase_aliases = {
            "linkedcell": "LinkedCell",
            "cell": "LinkedCell",
            "dynamiclinkedcell": "LinkedCell",
            "bvh": "BVH",
            "boundingvolumehierarchy": "BVH",
        }
        if broad_phase_key not in broad_phase_aliases:
            raise ValueError("FEM broad_phase must be 'LinkedCell' or 'BVH'")
        self.broad_phase = broad_phase_aliases[broad_phase_key]
        if self.dhat is not None:
            self.dhat = _positive("dhat", self.dhat)
        if self.dmin is not None:
            self.dmin = _positive("dmin", self.dmin, allow_zero=True)
        self.kappa = _positive("kappa", self.kappa)
        self.penalty = _positive("penalty", self.penalty)
        self.max_penalty = _positive("max_penalty", self.max_penalty)
        if self.max_penalty < self.penalty:
            raise ValueError("max_penalty must be at least penalty")
        self.penalty_growth = _positive("penalty_growth", self.penalty_growth)
        if self.penalty_growth <= 1.0:
            raise ValueError("penalty_growth must be greater than one")
        self.sufficient_reduction = _positive("sufficient_reduction", self.sufficient_reduction)
        if self.sufficient_reduction >= 1.0:
            raise ValueError("sufficient_reduction must be less than one")
        self.constraint_tolerance = _positive("constraint_tolerance", self.constraint_tolerance, allow_zero=True)
        self.point_triangle_coordination_number = _positive(
            "point_triangle_coordination_number",
            self.point_triangle_coordination_number,
        )
        self.edge_edge_coordination_number = _positive(
            "edge_edge_coordination_number",
            self.edge_edge_coordination_number,
        )
        for name in ("max_point_triangle_pairs", "max_edge_edge_pairs"):
            value = getattr(self, name)
            if value is not None:
                value = int(value)
                if value <= 0:
                    raise ValueError(f"{name} must be positive")
                setattr(self, name, value)
        self.friction_coefficient = _positive("friction_coefficient", self.friction_coefficient, allow_zero=True)
        self.epsv = _positive("epsv", self.epsv)
        self.friction_iterations = _integer("friction_iterations", self.friction_iterations)
        if self.friction_iterations != -1 and self.friction_iterations <= 0:
            raise ValueError("friction_iterations must be -1 or a positive integer")
        self.friction_max_iterations = _integer("friction_max_iterations", self.friction_max_iterations)
        if self.friction_max_iterations <= 0:
            raise ValueError("friction_max_iterations must be positive")
        self.friction_tolerance = _positive("friction_tolerance", self.friction_tolerance, allow_zero=True)
        self.ccd_safety = _positive("ccd_safety", self.ccd_safety)
        if self.ccd_safety > 1.0:
            raise ValueError("ccd_safety must not exceed one")
        self.ccd_max_iterations = int(self.ccd_max_iterations)
        self.penalty_update_interval = int(self.penalty_update_interval)
        if self.ccd_max_iterations <= 0 or self.penalty_update_interval <= 0:
            raise ValueError("contact iteration counts must be positive")
        self.planes = tuple(_normalize_plane(entry) for entry in (self.planes or ()))
        if self.body_pair is not None:
            if len(self.body_pair) != 2:
                raise ValueError("FEM contact body_pair must contain two IDs")
            first, second = (int(value) for value in self.body_pair)
            if first < 0 or second < 0:
                raise ValueError("FEM contact body IDs must be non-negative")
            self.body_pair = tuple(sorted((first, second)))
        if self.model not in ("IPC", "AugmentedLagrangian") and self.pair_contacts:
            raise ValueError("per-body contact parameters require FEM IPC contact")

    @classmethod
    def create(cls, model="IPC", **kwargs):
        if isinstance(model, cls):
            if kwargs:
                raise TypeError("contact keyword arguments cannot accompany an FEMContact object")
            return model
        if isinstance(model, dict):
            parameters = dict(model)
            parameters.update(kwargs)
            model = "IPC"
            for name in list(parameters):
                key = str(name).replace("_", "").replace("-", "").replace(" ", "").lower()
                if key in ("model", "type"):
                    model = parameters.pop(name)
        else:
            parameters = dict(kwargs)
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
            "frictioniterations": "friction_iterations",
            "frictionmaxiterations": "friction_max_iterations",
            "frictiontolerance": "friction_tolerance",
            "initialpenalty": "penalty",
            "maxpenalty": "max_penalty",
            "penaltygrowth": "penalty_growth",
            "penaltyupdateinterval": "penalty_update_interval",
            "constrainttolerance": "constraint_tolerance",
            "selfcontact": "self_contact",
            "ccdsafety": "ccd_safety",
            "ccdmaxiterations": "ccd_max_iterations",
            "projectpd": "project_pd",
            "broadphase": "broad_phase",
            "contactsearch": "broad_phase",
            "pointtrianglecoordinationnumber": "point_triangle_coordination_number",
            "edgeedgecoordinationnumber": "edge_edge_coordination_number",
            "maxpointtrianglepairs": "max_point_triangle_pairs",
            "maxedgeedgepairs": "max_edge_edge_pairs",
        }
        canonical = {}
        for name, value in parameters.items():
            key = str(name).replace("_", "").replace("-", "").replace(" ", "").lower()
            canonical[aliases.get(key, name)] = value
        return cls(model=model, **canonical)

    def add_property(self, body_id1, body_id2, property=None, **kwargs):
        """Add one IPC parameter set for an unordered FEM body pair."""
        if self.model not in ("IPC", "AugmentedLagrangian"):
            raise ValueError("per-body contact parameters require FEM IPC contact")
        parameters = dict(property or {})
        parameters.update(kwargs)
        inherited = {
            "broad_phase": self.broad_phase,
            "dhat": self.dhat,
            "dmin": self.dmin,
            "kappa": self.kappa,
            "self_contact": self.self_contact,
            "project_pd": self.project_pd,
            "ccd_safety": self.ccd_safety,
            "ccd_max_iterations": self.ccd_max_iterations,
            "friction_coefficient": self.friction_coefficient,
            "epsv": self.epsv,
            "friction_iterations": self.friction_iterations,
            "friction_max_iterations": self.friction_max_iterations,
            "friction_tolerance": self.friction_tolerance,
            "penalty": self.penalty,
            "max_penalty": self.max_penalty,
            "penalty_growth": self.penalty_growth,
            "penalty_update_interval": self.penalty_update_interval,
            "sufficient_reduction": self.sufficient_reduction,
            "constraint_tolerance": self.constraint_tolerance,
            "point_triangle_coordination_number": self.point_triangle_coordination_number,
            "edge_edge_coordination_number": self.edge_edge_coordination_number,
            "max_point_triangle_pairs": self.max_point_triangle_pairs,
            "max_edge_edge_pairs": self.max_edge_edge_pairs,
        }
        inherited.update(parameters)
        created = type(self).create(self.model, **inherited)
        created.body_pair = tuple(sorted((int(body_id1), int(body_id2))))
        if created.body_pair[0] < 0:
            raise ValueError("FEM contact body IDs must be non-negative")
        created.planes = ()
        # This flag activates PT/EE assembly inside one pair-specific
        # assembler. Exact same/cross-body selection is performed by culling.
        created.self_contact = True
        for index, existing in enumerate(self.pair_contacts):
            if existing.body_pair == created.body_pair:
                self.pair_contacts[index] = created
                return created
        self.pair_contacts.append(created)
        return created

    def contacts_for_mesh(self, mesh):
        """Return validated pair-specific contacts for assembler creation."""
        if not self.pair_contacts:
            return (self,)
        available = set(map(int, mesh.body_ids))
        selected = []
        for contact in self.pair_contacts:
            if not set(contact.body_pair).issubset(available):
                raise ValueError(
                    f"FEM contact body pair {contact.body_pair} is not in " f"mesh body IDs {sorted(available)}"
                )
            selected.append(contact)
        if not selected and not self.planes:
            raise ValueError("FEM IPC has no active body pair or plane")
        if self.planes:
            selected.append(
                type(self).create(
                    "IPC",
                    broad_phase=self.broad_phase,
                    dhat=self.dhat,
                    dmin=self.dmin,
                    kappa=self.kappa,
                    self_contact=False,
                    planes=self.planes,
                    project_pd=self.project_pd,
                    ccd_safety=self.ccd_safety,
                    ccd_max_iterations=self.ccd_max_iterations,
                    friction_coefficient=self.friction_coefficient,
                    epsv=self.epsv,
                    friction_iterations=self.friction_iterations,
                    friction_max_iterations=self.friction_max_iterations,
                    friction_tolerance=self.friction_tolerance,
                    point_triangle_coordination_number=self.point_triangle_coordination_number,
                    edge_edge_coordination_number=self.edge_edge_coordination_number,
                    max_point_triangle_pairs=self.max_point_triangle_pairs,
                    max_edge_edge_pairs=self.max_edge_edge_pairs,
                )
            )
        return tuple(selected)


__all__ = ["FEMContact"]

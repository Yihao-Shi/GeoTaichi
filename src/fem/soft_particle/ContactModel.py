"""Explicit DEM-style contact parameters for FEM soft particles."""

import math

import taichi as ti


def property_value(mapping, name, *aliases, default=None, required=False):
    normalized = {str(key).replace("_", "").replace("-", "").lower(): value for key, value in mapping.items()}
    for candidate in (name, *aliases):
        key = str(candidate).replace("_", "").replace("-", "").lower()
        if key in normalized:
            return normalized[key]
    if required:
        raise KeyError(f"FEM soft-particle contact property {name!r} is required")
    return default


@ti.dataclass
class FEMSoftParticleProperty:
    active: ti.i32
    thickness: float
    friction: float
    kn: float
    ks: float
    normal_damping: float
    tangential_damping: float
    effective_young: float
    effective_shear: float
    restitution_damping: float
    barrier_kappa: float
    barrier_cutoff: float
    barrier_stiffness_ratio: float


def _pair(first, second):
    first, second = int(first), int(second)
    if first < 0 or second < 0:
        raise ValueError("FEM soft-particle body IDs must be non-negative")
    if first == second:
        raise ValueError("FEM soft-particle contact properties require two different bodies")
    return (first, second) if first < second else (second, first)


def _model_name(model):
    key = str(model).replace("_", "").replace("-", "").replace(" ", "").lower()
    if key in ("linear", "linearspring", "linearmodel"):
        return "Linear"
    if key in ("hertz", "hertzmindlin", "hertzmindlinmodel"):
        return "HertzMindlin"
    if key in ("barrier", "barriermodel"):
        return "Barrier"
    raise ValueError("FEM soft-particle contact model must be Linear, HertzMindlin, or Barrier")


def _search_name(search):
    key = str(search).replace("_", "").replace("-", "").replace(" ", "").lower()
    if key in ("linkedcell", "cell", "dynamiclinkedcell"):
        return "LinkedCell"
    if key in ("bvh", "lbvh", "boundingvolumehierarchy"):
        return "BVH"
    raise ValueError("FEM soft-particle search must be LinkedCell or BVH")


class FEMSoftParticleContactModel:
    """Host configuration compiled into a device pair table at FEM build."""

    def __init__(self, model="Linear", **kwargs):
        self.model_name = _model_name(model)
        self.model_type = {
            "Linear": 0,
            "HertzMindlin": 1,
            "Barrier": 2,
        }[self.model_name]
        self.search = _search_name(kwargs.pop("search", kwargs.pop("broad_phase", "LinkedCell")))
        self.verlet_distance = kwargs.pop("verlet_distance", None)
        self.verlet_distance_multiplier = float(kwargs.pop("verlet_distance_multiplier", 0.1))
        self.maximum_penetration_fraction = float(
            kwargs.pop(
                "maximum_penetration_fraction",
                kwargs.pop("MaximumPenetrationFraction", 0.5),
            )
        )
        self.max_point_triangle_pairs = kwargs.pop(
            "max_point_triangle_pairs",
            kwargs.pop("MaxPointTrianglePairs", None),
        )
        self.max_edge_edge_pairs = kwargs.pop(
            "max_edge_edge_pairs",
            kwargs.pop("MaxEdgeEdgePairs", None),
        )
        self.max_filtered_point_triangle_pairs = kwargs.pop(
            "max_filtered_point_triangle_pairs",
            kwargs.pop("MaxFilteredPointTrianglePairs", None),
        )
        self.max_filtered_edge_edge_pairs = kwargs.pop(
            "max_filtered_edge_edge_pairs",
            kwargs.pop("MaxFilteredEdgeEdgePairs", None),
        )
        self.raw_point_triangle_coordination_number = kwargs.pop(
            "raw_point_triangle_coordination_number",
            kwargs.pop("RawPointTriangleCoordinationNumber", None),
        )
        self.raw_edge_edge_coordination_number = kwargs.pop(
            "raw_edge_edge_coordination_number",
            kwargs.pop("RawEdgeEdgeCoordinationNumber", None),
        )
        self.point_triangle_coordination_number = kwargs.pop(
            "point_triangle_coordination_number",
            kwargs.pop("PointTriangleCoordinationNumber", None),
        )
        self.edge_edge_coordination_number = kwargs.pop(
            "edge_edge_coordination_number",
            kwargs.pop("EdgeEdgeCoordinationNumber", None),
        )
        self.contact_history_capacity = kwargs.pop(
            "contact_history_capacity",
            kwargs.pop("ContactHistoryCapacity", None),
        )
        self.contact_history_load_factor = float(
            kwargs.pop(
                "contact_history_load_factor",
                kwargs.pop("ContactHistoryLoadFactor", 0.5),
            )
        )
        for name in (
            "max_point_triangle_pairs",
            "max_edge_edge_pairs",
            "max_filtered_point_triangle_pairs",
            "max_filtered_edge_edge_pairs",
            "contact_history_capacity",
        ):
            value = getattr(self, name)
            if value is not None:
                value = int(value)
                if value <= 0:
                    raise ValueError(f"FEM soft-particle {name} must be positive")
                setattr(self, name, value)
        for name in (
            "raw_point_triangle_coordination_number",
            "raw_edge_edge_coordination_number",
            "point_triangle_coordination_number",
            "edge_edge_coordination_number",
        ):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(f"FEM soft-particle {name} must be positive")
                setattr(self, name, value)
        if self.verlet_distance is not None and float(self.verlet_distance) <= 0.0:
            raise ValueError("FEM soft-particle verlet_distance must be positive")
        if self.verlet_distance_multiplier <= 0.0:
            raise ValueError("FEM soft-particle verlet_distance_multiplier must be positive")
        if self.maximum_penetration_fraction <= 0.0:
            raise ValueError("FEM soft-particle maximum_penetration_fraction must be positive")
        if not 0.0 < self.contact_history_load_factor < 1.0:
            raise ValueError("FEM soft-particle contact_history_load_factor must be in (0, 1)")
        self.default_property = dict(kwargs)
        self.properties = {}

    def pair_capacities(self, surface_triangle_count):
        """Resolve raw and Verlet PT/EE capacities from one surface measure."""

        triangle_count = int(surface_triangle_count)
        if triangle_count <= 0:
            raise ValueError("FEM soft-particle surface_triangle_count must be positive")

        def resolve(absolute, coordination):
            if absolute is not None:
                return int(absolute)
            if coordination is None:
                return None
            return max(1, int(math.ceil(triangle_count * coordination)))

        return {
            "raw_point_triangle": resolve(
                self.max_point_triangle_pairs,
                self.raw_point_triangle_coordination_number,
            ),
            "raw_edge_edge": resolve(
                self.max_edge_edge_pairs,
                self.raw_edge_edge_coordination_number,
            ),
            "verlet_point_triangle": resolve(
                self.max_filtered_point_triangle_pairs,
                self.point_triangle_coordination_number,
            ),
            "verlet_edge_edge": resolve(
                self.max_filtered_edge_edge_pairs,
                self.edge_edge_coordination_number,
            ),
        }

    def add_property(self, first_body, second_body, property=None, **kwargs):
        parameters = dict(property or {})
        parameters.update(kwargs)
        self.properties[_pair(first_body, second_body)] = parameters
        return parameters

    def _linear_property(self, parameters):
        kn = float(property_value(parameters, "NormalStiffness", "kn", required=True))
        ks = float(property_value(parameters, "TangentialStiffness", "ks", required=True))
        if kn <= 0.0 or ks <= 0.0:
            raise ValueError("FEM soft-particle linear stiffnesses must be positive")
        normal_damping = float(property_value(parameters, "NormalViscousDamping", "ndratio", default=0.0))
        tangential_damping = float(property_value(parameters, "TangentialViscousDamping", "sdratio", default=0.0))
        if normal_damping < 0.0 or tangential_damping < 0.0:
            raise ValueError("FEM soft-particle damping ratios must be non-negative")
        return {
            "kn": kn,
            "ks": ks,
            "normal_damping": normal_damping,
            "tangential_damping": tangential_damping,
            "effective_young": 0.0,
            "effective_shear": 0.0,
            "restitution_damping": 0.0,
        }

    def _hertz_property(self, parameters):
        modulus = float(property_value(parameters, "ShearModulus", "Modulus", required=True))
        poisson = float(property_value(parameters, "Poisson", "PoissonRatio", required=True))
        restitution = float(property_value(parameters, "Restitution", required=True))
        if modulus <= 0.0 or not -1.0 < poisson < 0.5:
            raise ValueError("invalid FEM soft-particle Hertz modulus/Poisson ratio")
        if not 0.0 <= restitution <= 1.0:
            raise ValueError("FEM soft-particle restitution must be in [0, 1]")
        effective_shear = 0.5 * modulus / (2.0 - poisson)
        effective_young = (4.0 * effective_shear - 2.0 * effective_shear * poisson) / (1.0 - poisson)
        damping = 0.0
        if restitution >= 1.0e-16:
            logarithm = math.log(restitution)
            damping = -logarithm / math.sqrt(math.pi * math.pi + logarithm * logarithm)
        return {
            "kn": 0.0,
            "ks": 0.0,
            "normal_damping": 0.0,
            "tangential_damping": 0.0,
            "effective_young": effective_young,
            "effective_shear": effective_shear,
            "restitution_damping": damping,
            "barrier_kappa": 0.0,
            "barrier_cutoff": 0.0,
            "barrier_stiffness_ratio": 0.0,
        }

    def _barrier_property(self, parameters):
        kappa = float(property_value(parameters, "Stiffness", "kappa", required=True))
        cutoff = float(
            property_value(
                parameters,
                "NormalCutOff",
                "NormalCutoff",
                "ncut",
                required=True,
            )
        )
        stiffness_ratio = float(
            property_value(
                parameters,
                "StiffnessRatio",
                "ratio",
                default=1.0,
            )
        )
        normal_damping = float(property_value(parameters, "NormalViscousDamping", "ndratio", default=0.0))
        tangential_damping = float(
            property_value(
                parameters,
                "TangentialViscousDamping",
                "sdratio",
                default=0.0,
            )
        )
        if kappa <= 0.0 or cutoff <= 0.0 or stiffness_ratio <= 0.0:
            raise ValueError("FEM soft-particle barrier stiffness, cutoff, and stiffness ratio must be positive")
        if normal_damping < 0.0 or tangential_damping < 0.0:
            raise ValueError("FEM soft-particle damping ratios must be non-negative")
        return {
            "kn": 0.0,
            "ks": 0.0,
            "normal_damping": normal_damping,
            "tangential_damping": tangential_damping,
            "effective_young": 0.0,
            "effective_shear": 0.0,
            "restitution_damping": 0.0,
            "barrier_kappa": kappa,
            "barrier_cutoff": cutoff,
            "barrier_stiffness_ratio": stiffness_ratio,
        }

    def canonical_property(self, parameters):
        parameters = dict(parameters)
        thickness = float(
            property_value(
                parameters,
                "ContactThickness",
                "Thickness",
                "dmin",
                default=0.0,
            )
        )
        friction = float(property_value(parameters, "Friction", "mu", default=0.0))
        if thickness < 0.0 or friction < 0.0:
            raise ValueError("FEM soft-particle thickness/friction must be non-negative")
        if self.model_type == 0:
            values = self._linear_property(parameters)
            values.update(
                {
                    "barrier_kappa": 0.0,
                    "barrier_cutoff": 0.0,
                    "barrier_stiffness_ratio": 0.0,
                }
            )
        elif self.model_type == 1:
            values = self._hertz_property(parameters)
        else:
            values = self._barrier_property(parameters)
        values.update({"active": 1, "thickness": thickness, "friction": friction})
        return values

    def pair_values(self, body_count):
        default, overrides = self.compact_pair_values(body_count)
        values = {}
        if default is not None:
            for first in range(int(body_count)):
                for second in range(first + 1, int(body_count)):
                    values[(first, second)] = dict(default)
        values.update(overrides)
        return values

    def compact_pair_values(self, body_count):
        """Return one default law plus explicit pair overrides.

        Keeping the default compact avoids materializing and subsequently
        uploading an O(n_body^2) Python dictionary for large soft-particle
        assemblies.  ``pair_values`` remains available for callers that need
        the fully expanded mapping.
        """
        body_count = int(body_count)
        if body_count <= 0:
            raise ValueError("FEM soft-particle contact requires at least one body")
        default = self.canonical_property(self.default_property) if self.default_property else None
        overrides = {}
        for pair, parameters in self.properties.items():
            if pair[1] >= body_count:
                raise ValueError(f"FEM soft-particle property pair {pair} exceeds {body_count} bodies")
            overrides[pair] = self.canonical_property(parameters)
        if body_count > 1 and default is None and not overrides:
            raise ValueError(
                "FEM soft-particle contact has no active body pair; provide default law parameters "
                "or call add_soft_particle_property"
            )
        return default, overrides


__all__ = ["FEMSoftParticleContactModel", "FEMSoftParticleProperty"]

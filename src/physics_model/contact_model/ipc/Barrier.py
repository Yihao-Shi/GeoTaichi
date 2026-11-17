from src.physics_model.contact_model.ipc.Utilities import ipc_barrier_distance_offset_terms
from src.physics_model.contact_model.ipc.Utilities import _single_finite_parameter

import taichi as ti


@ti.data_oriented
class Barrier:
    """Field-backed BarrierIPC parameters used by coupled engines."""

    @staticmethod
    def normalize_ipc_model(value):
        """Validate the one supported normal-contact model name."""
        if value != "BarrierIPC":
            raise ValueError("IPC contact model must be 'BarrierIPC'")
        return value

    def __init__(self, **kwargs):
        removed = {
            "penalty",
            "max_penalty",
            "penalty_growth",
            "penalty_update_interval",
            "sufficient_reduction",
            "constraint_tolerance",
        }.intersection(kwargs)
        if removed:
            raise ValueError(f"removed SemiIPC parameters: {', '.join(sorted(removed))}")
        requested_model = kwargs.get(
            "ipc_model",
            kwargs.get("normal_contact_model", kwargs.get("contact_form", "BarrierIPC")),
        )
        self.model = Barrier.normalize_ipc_model(requested_model)
        kappa = _single_finite_parameter("kappa", kwargs.get("kappa", 1.0e5), positive=True)
        dhat = _single_finite_parameter("dhat", kwargs.get("dhat", 1.0e-3), positive=True)
        dmin = _single_finite_parameter("dmin", kwargs.get("dmin", 0.0), nonnegative=True)
        self.kappa = ti.field(ti.f64, shape=1)
        self.dhat = ti.field(ti.f64, shape=1)
        self.dmin = ti.field(ti.f64, shape=1)
        self.kappa[0] = kappa
        self.dhat[0] = dhat
        self.dmin[0] = dmin
        self.use_physical_barrier = bool(kwargs.get("use_physical_barrier", kwargs.get("physical_barrier", False)))
        self._use_physical_barrier_id = int(self.use_physical_barrier)
        if "barrier_form" in kwargs:
            raise ValueError(
                "barrier_form is no longer configurable; GeoTaichi contact "
                "always uses the finite-clearance BarrierIPC law"
            )

    @ti.func
    def _terms(self, distance):
        return ipc_barrier_distance_offset_terms(
            distance,
            self.dhat[0],
            self.dmin[0],
            self.kappa[0],
            ti.static(self._use_physical_barrier_id),
        )

    @property
    def stiffness(self):
        return self.kappa[0]

    @stiffness.setter
    def stiffness(self, stiffness):
        self.kappa[0] = _single_finite_parameter("stiffness", stiffness, positive=True)

    @property
    def threshold(self):
        return self.dmin[0] + self.dhat[0]

    @threshold.setter
    def threshold(self, threshold):
        threshold = _single_finite_parameter("threshold", threshold, positive=True)
        if threshold <= self.dmin[0]:
            raise ValueError("threshold must be greater than dmin")
        self.dhat[0] = threshold - self.dmin[0]

    @property
    def activation_gap(self):
        return self.dhat[0]

    @activation_gap.setter
    def activation_gap(self, dhat):
        self.dhat[0] = _single_finite_parameter("dhat", dhat, positive=True)

    @property
    def minimum_distance(self):
        return self.dmin[0]

    @minimum_distance.setter
    def minimum_distance(self, dmin):
        dmin = _single_finite_parameter("dmin", dmin, nonnegative=True)
        self.dmin[0] = dmin

    @ti.func
    def activation_distance_term(self):
        return self.dmin[0] + self.dhat[0]

    @ti.func
    def energy_term(self, distance):
        energy, _, _ = self._terms(distance)
        return energy

    @ti.func
    def grad_term(self, distance):
        _, gradient, _ = self._terms(distance)
        return gradient

    @ti.func
    def hess_term(self, distance):
        _, _, hessian = self._terms(distance)
        return hessian

    @ti.func
    def energy(self, distance, area=1.0):
        return self.energy_term(distance) * area

    @ti.func
    def gradient(self, distance, ddistance2_dvertex, area=1.0):
        return 0.5 * self.grad_term(distance) / distance * ddistance2_dvertex * area

    @ti.func
    def hessian(
        self,
        distance,
        ddistance2_dvertex_1,
        ddistance2_dvertex_2,
        d2distance2_d2vertex,
        area=1.0,
    ):
        inv_distance = 1.0 / distance
        inv_distance2 = inv_distance * inv_distance
        inv_distance3 = inv_distance2 * inv_distance
        barrier_gradient = self.grad_term(distance)
        barrier_hessian = self.hess_term(distance)
        hessian = (
            0.25
            * (barrier_hessian * inv_distance2 - barrier_gradient * inv_distance3)
            * ddistance2_dvertex_1.outer_product(ddistance2_dvertex_2)
        )
        hessian += 0.5 * barrier_gradient * inv_distance * d2distance2_d2vertex
        return hessian * area

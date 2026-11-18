from src.physics_model.contact_model.ipc.Utilities import ipc_friction_f0
from src.physics_model.contact_model.ipc.Utilities import ipc_friction_f1_over_speed
from src.physics_model.contact_model.ipc.Utilities import ipc_friction_hessian_term
from src.physics_model.contact_model.ipc.Utilities import _single_finite_parameter

import taichi as ti


@ti.data_oriented
class Friction:
    """Field-backed parameters for smooth IPC friction laws.

    The legacy ``mu`` interface remains the equal-static/dynamic Coulomb case.
    Fully implicit contact additionally accepts:
    ``dynamic_friction``, ``static_friction``, ``viscous_friction``,
    ``stribeck_velocity`` and ``friction_profile``.
    """

    @staticmethod
    def _normalize_friction_profile(value):
        """Return the friction-profile id.

        ``quadratic`` is the compact C1 profile in Eq. (5) of Larionov et al. and
        is also the profile used by IPC. ``stabilized`` is a C-infinity
        alternative. Keeping this choice explicit avoids silently mixing the two
        regularizations when comparing results.
        """
        key = str(value).strip().replace("-", "_").replace(" ", "_").lower()
        aliases = {
            "quadratic": 0,
            "c1": 0,
            "ipc": 0,
            "stabilized": 1,
            "stabilised": 1,
            "cinfinity": 1,
            "c_infinity": 1,
        }
        if key not in aliases:
            raise ValueError("friction_profile must be 'quadratic' (paper/IPC C1) or " "'stabilized' (C-infinity)")
        return aliases[key]

    def __init__(self, **kwargs):
        legacy_mu = _single_finite_parameter(
            "mu",
            kwargs.get("mu", kwargs.get("dynamic_friction", 0.0)),
            nonnegative=True,
        )
        epsv = _single_finite_parameter("epsv", kwargs.get("epsv", 1.0e-3), positive=True)
        mu_dynamic = _single_finite_parameter(
            "dynamic_friction",
            kwargs.get("dynamic_friction", kwargs.get("mu_dynamic", legacy_mu)),
            nonnegative=True,
        )
        mu_static = _single_finite_parameter(
            "static_friction",
            kwargs.get("static_friction", kwargs.get("mu_static", mu_dynamic)),
            nonnegative=True,
        )
        mu_viscous = _single_finite_parameter(
            "viscous_friction",
            kwargs.get("viscous_friction", kwargs.get("mu_viscous", 0.0)),
            nonnegative=True,
        )
        stribeck_velocity = _single_finite_parameter(
            "stribeck_velocity",
            kwargs.get("stribeck_velocity", 10.0 * epsv),
            nonnegative=True,
        )
        if mu_static != mu_dynamic and stribeck_velocity <= 0.0:
            raise ValueError("stribeck_velocity must be positive when static_friction " "differs from dynamic_friction")
        self.mu = ti.field(ti.f64, shape=1)
        self.epsv = ti.field(ti.f64, shape=1)
        self.mu_dynamic = ti.field(ti.f64, shape=1)
        self.mu_static = ti.field(ti.f64, shape=1)
        self.mu_viscous = ti.field(ti.f64, shape=1)
        self.stribeck_velocity = ti.field(ti.f64, shape=1)
        # Lagged IPC has one Coulomb coefficient. If the caller uses only the
        # Use dynamic friction for the backward-compatible scalar field.
        self.mu[0] = mu_dynamic if "mu" not in kwargs else legacy_mu
        self.epsv[0] = epsv
        self.mu_dynamic[0] = mu_dynamic
        self.mu_static[0] = mu_static
        self.mu_viscous[0] = mu_viscous
        self.stribeck_velocity[0] = stribeck_velocity
        self.profile_id = Friction._normalize_friction_profile(kwargs.get("friction_profile", "quadratic"))

    @property
    def has_friction(self):
        return (
            float(self.mu[0]) > 0.0
            or float(self.mu_dynamic[0]) > 0.0
            or float(self.mu_static[0]) > 0.0
            or float(self.mu_viscous[0]) > 0.0
        )

    @property
    def friction(self):
        return self.mu[0]

    @friction.setter
    def friction(self, friction):
        value = _single_finite_parameter("friction", friction, nonnegative=True)
        self.mu[0] = value
        self.mu_dynamic[0] = value
        self.mu_static[0] = value

    @property
    def dynamic_friction(self):
        return self.mu_dynamic[0]

    @dynamic_friction.setter
    def dynamic_friction(self, friction):
        value = _single_finite_parameter("dynamic_friction", friction, nonnegative=True)
        self.mu_dynamic[0] = value
        self.mu[0] = value

    @property
    def static_friction(self):
        return self.mu_static[0]

    @static_friction.setter
    def static_friction(self, friction):
        self.mu_static[0] = _single_finite_parameter("static_friction", friction, nonnegative=True)

    @property
    def threshold(self):
        return self.epsv[0]

    @threshold.setter
    def threshold(self, threshold):
        self.epsv[0] = _single_finite_parameter("epsv", threshold, positive=True)

    @ti.func
    def energy_term(self, speed, timestep):
        return ipc_friction_f0(speed, self.epsv[0], timestep)

    @ti.func
    def grad_term(self, speed):
        return ipc_friction_f1_over_speed(speed, self.epsv[0])

    @ti.func
    def hess_term(self, speed):
        return ipc_friction_hessian_term(speed, self.epsv[0])

    @ti.func
    def energy(self, relative_displacement, normal, mu_lambda, timestep, area=1.0):
        tangent = ti.Matrix.identity(ti.f64, normal.n) - normal.outer_product(normal)
        # FrictionPotential is a potential in relative *velocity*.  GeoTaichi's
        # nonlinear unknown is displacement, hence v = T^T du / h and the
        # incremental potential is h D(v).  ``energy_term`` contains that h.
        tangential_velocity = tangent.transpose() @ relative_displacement / timestep
        return mu_lambda * self.energy_term(tangential_velocity.norm(), timestep) * area

    @ti.func
    def gradient(
        self,
        relative_displacement,
        normal,
        mu_lambda,
        timestep,
        dvelocity_dvertex,
        area=1.0,
    ):
        tangent = ti.Matrix.identity(ti.f64, normal.n) - normal.outer_product(normal)
        tangential_velocity = tangent.transpose() @ relative_displacement / timestep
        speed = tangential_velocity.norm()
        # Gradient of h D(v) with respect to v.  The caller-provided Jacobian
        # maps its vertex/generalized variable to relative velocity.
        gradient_velocity = mu_lambda * timestep * self.grad_term(speed) * tangent @ tangential_velocity
        return gradient_velocity @ dvelocity_dvertex * area

    @ti.func
    def hessian(
        self,
        relative_displacement,
        normal,
        mu_lambda,
        timestep,
        dvelocity_dvertex_1,
        dvelocity_dvertex_2,
        area=1.0,
    ):
        tangent = ti.Matrix.identity(ti.f64, normal.n) - normal.outer_product(normal)
        tangential_velocity = tangent.transpose() @ relative_displacement / timestep
        speed = tangential_velocity.norm()
        f1_over_speed = self.grad_term(speed)
        inner = f1_over_speed * ti.Matrix.identity(ti.f64, normal.n)
        if speed > 0.0:
            inner += (self.hess_term(speed) / speed) * tangential_velocity.outer_product(tangential_velocity)
        hessian_velocity = mu_lambda * timestep * tangent @ inner @ tangent.transpose()
        return dvelocity_dvertex_1.transpose() @ hessian_velocity @ dvelocity_dvertex_2 * area

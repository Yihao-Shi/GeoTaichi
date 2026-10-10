"""Finite-strain DP with the state law of StateDependentMohrCoulomb."""

import math

import taichi as ti

from src.physics_model.consititutive_model.finite_strain.DruckerPrager import FiniteStrainDruckerPragerModel
from src.physics_model.consititutive_model.finite_strain.HenckyPlasticity import HenckyAssociatedPlasticityModel


@ti.data_oriented
class StateDependentDruckerPragerModel(FiniteStrainDruckerPragerModel):
    """Nonassociated physical return; previous pressure is frozen per step.

    Void ratio follows the total volume increment, including the hoop stretch
    in axisymmetry. Only accepted commits change history. This model has no
    incremental energy potential and requires the coupled residual-Newton path.
    """

    def __init__(self, material_type="Solid", configuration="UL", solver_type="Implicit"):
        super().__init__(material_type, configuration, solver_type)
        self.is_state_dependent = True
        self.void_ratio = None
        self.committed_jacobian = None
        self.state_pressure = None
        self.history_state_size = 14

    def model_initialize(self, material):
        parameters = dict(material)
        self.e0 = float(self._parameter(material, ("e0",), 0.3))
        self.e_Tao = float(self._parameter(material, ("e_Tao",), 0.3))
        self.lambda_c = float(self._parameter(material, ("lambda_c",), 0.3))
        self.ksi = float(self._parameter(material, ("ksi",), 0.3))
        self.nd = float(self._parameter(material, ("nd",), 0.3))
        self.nf = float(self._parameter(material, ("nf",), 0.3))
        critical_angle = float(self._parameter(material, ("fai_c", "FrictionAngle", "friction_angle"), required=True))
        for name in ("e0", "e_Tao", "lambda_c", "ksi", "nd", "nf"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"StateDependentDruckerPrager {name} must be finite")
        if not 0.1 <= self.e0 <= 1.5 or self.e_Tao <= 0.0 or self.ksi <= 0.0:
            raise ValueError("StateDependentDruckerPrager requires 0.1 <= e0 <= 1.5, e_Tao > 0 and ksi > 0")
        if min(self.lambda_c, self.nd, self.nf) < 0.0:
            raise ValueError("StateDependentDruckerPrager lambda_c, nd and nf must be nonnegative")
        parameters["FrictionAngle"] = critical_angle
        parameters["DilationAngle"] = 0.0
        parameters["dpType"] = self._parameter(
            material, ("dpType", "DPType", "yield_surface_type"), "MiddleCircumscribed"
        )
        super().model_initialize(parameters)
        # Evolving nonassociated coefficients require the physical Jacobian
        # even if the current state happens to have zero dilatancy.
        self.is_nonassociated = True
        self.has_symmetric_tangent = False
        self.has_incremental_potential = False
        self.has_physical_incremental_potential = False
        self.requires_lagged_incremental_potential = False
        self.use_direct_nonassociated_solve = True

    def allocate_state(self, particle_count):
        HenckyAssociatedPlasticityModel.allocate_state(self, particle_count)
        self.void_ratio = ti.field(ti.f64, shape=int(particle_count))
        self.committed_jacobian = ti.field(ti.f64, shape=int(particle_count))
        self.state_pressure = ti.field(ti.f64, shape=int(particle_count))
        self.void_ratio.fill(self.e0)
        self.committed_jacobian.fill(1.0)
        self.state_pressure.fill(1000.0)

    @ti.kernel
    def prepare_step_state(self, total_deformation: ti.template(), particle_count: ti.i32):
        for particle_id in range(particle_count):
            total = total_deformation[particle_id]
            elastic = total @ self.plastic_deformation_inverse[particle_id]
            elastic_jacobian = elastic.determinant()
            assert elastic_jacobian > 0.0, "state-dependent DP requires det(Fe) > 0"
            self.committed_jacobian[particle_id] = total.determinant()
            self.state_pressure[particle_id] = ti.max(-self.bulk * ti.log(elastic_jacobian) / elastic_jacobian, 1000.0)

    @ti.func
    def get_history_state(self, particle_id):
        base = HenckyAssociatedPlasticityModel.get_history_state(self, particle_id)
        state = ti.Vector.zero(ti.f64, 14)
        for i in ti.static(range(11)):
            state[i] = base[i]
        state[11] = self.void_ratio[particle_id]
        state[12] = self.committed_jacobian[particle_id]
        state[13] = self.state_pressure[particle_id]
        return state

    @ti.func
    def set_history_state(self, particle_id, state):
        base = ti.Vector.zero(ti.f64, 11)
        for i in ti.static(range(11)):
            base[i] = state[i]
        HenckyAssociatedPlasticityModel.set_history_state(self, particle_id, base)
        self.void_ratio[particle_id] = state[11]
        self.committed_jacobian[particle_id] = state[12]
        self.state_pressure[particle_id] = state[13]

    @ti.func
    def _trial_void_ratio(self, particle_id, total_jacobian):
        volume = (1.0 + self.void_ratio[particle_id]) * total_jacobian / self.committed_jacobian[particle_id]
        derivative = 0.0
        if volume > 1.1 and volume < 2.5:
            derivative = volume
        return ti.min(1.5, ti.max(0.1, volume - 1.0)), derivative

    @ti.func
    def _cone_coefficients(self, angle):
        sine, cosine = ti.sin(angle), ti.cos(angle)
        q, k, dq, dk = 0.0, 0.0, 0.0, 0.0
        if ti.static(self.dp_type_code in (0, 1)):
            sign = -1.0 if ti.static(self.dp_type_code == 0) else 1.0
            denominator = 3.0 + sign * sine
            q = 6.0 * sine / (ti.sqrt(3.0) * denominator)
            k = 6.0 * self.cohesion * cosine / (ti.sqrt(3.0) * denominator)
            dq = 18.0 * cosine / (ti.sqrt(3.0) * denominator**2)
            dk = -6.0 * self.cohesion * (3.0 * sine + sign) / (ti.sqrt(3.0) * denominator**2)
        else:
            tangent = ti.tan(angle)
            denominator = ti.sqrt(9.0 + 12.0 * tangent * tangent)
            q = 3.0 * tangent / denominator
            k = 3.0 * self.cohesion / denominator
            dq = 27.0 * (1.0 + tangent * tangent) / denominator**3
            dk = -36.0 * self.cohesion * tangent * (1.0 + tangent * tangent) / denominator**3
        return ti.sqrt(2.0) * q / 3.0, ti.sqrt(2.0) * k, ti.sqrt(2.0) * dq / 3.0, ti.sqrt(2.0) * dk

    @ti.func
    def state_parameters(self, particle_id, elastic_trace):
        total_jacobian = ti.exp(elastic_trace) / self.plastic_deformation_inverse[particle_id].determinant()
        void, dvoid = self._trial_void_ratio(particle_id, total_jacobian)
        critical_void = self.e_Tao - self.lambda_c * (self.state_pressure[particle_id] / 101000.0) ** self.ksi
        state = void - critical_void
        friction, dilation = self.friction_angle, 0.0
        dfriction, ddilation = 0.0, 0.0
        if state < 0.0:
            friction = ti.atan2(ti.tan(self.friction_angle) * ti.exp(-self.nf * state), 1.0)
            dilation = ti.atan2(-self.nd * state, 1.0)
            dfriction = -self.nf * ti.sin(friction) * ti.cos(friction) * dvoid
            ddilation = -self.nd * dvoid / (1.0 + (self.nd * state) ** 2)
        alpha, intercept, dalpha, dintercept = self._cone_coefficients(friction)
        beta, _, dbeta, _ = self._cone_coefficients(dilation)
        return alpha, beta, intercept, dalpha * dfriction, dbeta * ddilation, dintercept * dfriction

    @ti.func
    def _associated_principal_return(self, particle_id, strain, trace, dev, norm):
        alpha, beta, intercept, _, _, _ = self.state_parameters(particle_id, trace)
        projected, multiplier, volume, region = strain, 0.0, 0.0, 0
        trial_yield = 2.0 * self.shear * norm + 3.0 * self.bulk * alpha * trace - intercept
        if trial_yield > 0.0:
            multiplier = trial_yield / (2.0 * self.shear + 9.0 * self.bulk * alpha * beta)
            if alpha > 1.0e-14 and multiplier >= norm:
                apex = intercept / (3.0 * self.bulk * alpha)
                region = 2
                multiplier = norm
                if beta > 1.0e-14:
                    multiplier = (trace - apex) / (3.0 * beta)
                projected = ti.Vector.one(ti.f64, 3) * (apex / 3.0)
                volume = trace - apex
            elif norm > 1.0e-14:
                region = 1
                projected = strain - multiplier * (dev / norm + beta)
                volume = 3.0 * beta * multiplier
        return projected, multiplier, volume, region

    @ti.func
    def _potential_principal_return(self, particle_id, strain, trace, dev, norm):
        return self._associated_principal_return(particle_id, strain, trace, dev, norm)

    @ti.func
    def _projected_strain_jacobian(self, particle_id, strain, trace, dev, norm, multiplier, region):
        alpha, beta, intercept, dalpha, dbeta, dintercept = self.state_parameters(particle_id, trace)
        derivative = ti.Matrix.identity(ti.f64, 3)
        if region == 1 and norm > 1.0e-14:
            direction = dev / norm
            denominator = 2.0 * self.shear + 9.0 * self.bulk * alpha * beta
            ddenominator = 9.0 * self.bulk * (dalpha * beta + alpha * dbeta)
            for i, j in ti.static(ti.ndrange(3, 3)):
                identity = 1.0 if ti.static(i == j) else 0.0
                dmultiplier = (
                    2.0 * self.shear * direction[j]
                    + 3.0 * self.bulk * (alpha + dalpha * trace)
                    - dintercept
                    - multiplier * ddenominator
                ) / denominator
                derivative[i, j] = (
                    identity
                    - (direction[i] + beta) * dmultiplier
                    - multiplier * dbeta
                    - multiplier / norm * (identity - 1.0 / 3.0 - direction[i] * direction[j])
                )
        elif region == 2:
            dapex = (dintercept * alpha - intercept * dalpha) / (3.0 * self.bulk * alpha**2)
            derivative = ti.Matrix.one(ti.f64, 3, 3) * (dapex / 3.0)
        return derivative

    @ti.func
    def _potential_strain_jacobian(self, particle_id, strain, trace, dev, norm, multiplier, region):
        return self._projected_strain_jacobian(particle_id, strain, trace, dev, norm, multiplier, region)

    @ti.func
    def commit_total_state(self, particle_id, total_deformation_gradient):
        jacobian = total_deformation_gradient.determinant()
        void, _ = self._trial_void_ratio(particle_id, jacobian)
        result = HenckyAssociatedPlasticityModel.commit_total_state(self, particle_id, total_deformation_gradient)
        self.void_ratio[particle_id] = void
        self.committed_jacobian[particle_id] = jacobian
        return result

    @ti.func
    def strain_energy_density_at(self, particle_id, deformation_gradient):
        raise NotImplementedError(
            "StateDependentDruckerPrager requires physical residual Newton, not energy minimization"
        )

    @ti.func
    def total_first_piola_parameter_vjp_at(self, particle_id, total_deformation_gradient, stress_vjp):
        raise NotImplementedError("StateDependentDruckerPrager parameter/history adjoints are not implemented")


__all__ = ["StateDependentDruckerPragerModel"]

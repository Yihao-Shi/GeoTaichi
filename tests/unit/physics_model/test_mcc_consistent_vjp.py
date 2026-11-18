"""Physical MCC derivatives against an independent coupled-hardening oracle."""

import numpy as np
import pytest
import taichi as ti
from scipy.optimize import least_squares

from src.physics_model.constitutive_model.finite_strain.ModifiedCamClay import FiniteStrainModifiedCamClayModel

pytestmark = [pytest.mark.materials, pytest.mark.cpu]


def reference(F, B, pc, parameters):
    nu, M, lam, kap, p0 = parameters
    G0 = 3 * p0 / kap * (1 - 2 * nu) / (2 * (1 + nu))
    U, sigma, Vh = np.linalg.svd(F @ B)
    trace = np.log(np.linalg.det(F @ B))
    dev = np.log(sigma) - trace / 3
    norm = np.linalg.norm(dev)

    def elastic(t, r):
        scale = np.exp(-t / kap + G0 * r * r / (kap * p0))
        return p0 * scale, G0 * scale, np.sqrt(6) * G0 * scale * r

    p, G, q = elastic(trace, norm)
    t, r, gamma, pc_new = trace, norm, 0.0, pc
    if q * q / M**2 + p * (p - pc) > 1e-10 * pc**2:

        def residual(z):
            t, r, multiplier, logpc = z
            gamma, current_pc = multiplier / pc, pc * np.exp(logpc)
            p, G, q = elastic(t, r)
            return [
                t - trace - gamma * (2 * p - current_pc),
                r * (1 + 6 * G * gamma / M**2) - norm,
                (q * q / M**2 + p * (p - current_pc)) / pc**2,
                logpc - (t - trace) / (lam - kap),
            ]

        result = least_squares(
            residual,
            [trace, max(norm, 1e-14), 1e-8, 0.0],
            bounds=([-np.inf, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
            max_nfev=1000,
        )
        assert np.linalg.norm(residual(result.x), np.inf) < 1e-9
        t, r, multiplier, logpc = result.x
        gamma, pc_new = multiplier / pc, pc * np.exp(logpc)
        p, G, q = elastic(t, r)
    projected = np.full(3, t / 3)
    if norm > 1e-14:
        projected += r * dev / norm
    Fe = U @ np.diag(np.exp(projected)) @ Vh
    stress = np.linalg.det(F) / np.exp(t) * (U @ np.diag((2 * G * (projected - t / 3) - p) / sigma) @ Vh) @ B.T
    return stress, np.linalg.solve(F, Fe), 2 * gamma * q / M**2, t - trace, pc_new


@pytest.mark.slow
def test_mcc_physical_stress_commit_and_parameter_vjps(taichi_runtime):
    parameters = np.array([0.3, 1.2, 0.20 / 1.8, 0.05 / 1.8, 1e5])
    model = FiniteStrainModifiedCamClayModel().initialize_from_kwargs(
        density=1800,
        PoissonRatio=parameters[0],
        StressRatio=parameters[1],
        lambda_star=parameters[2],
        kappa_star=parameters[3],
        void_ratio_ref=0.8,
        pc0=2e5,
        InitialPressure=parameters[4],
    )
    model.use_direct_consistent_solve = True
    model.allocate_state(1)
    values = ti.Matrix.field(3, 3, ti.f64, shape=3)
    state_seed = ti.Vector.field(3, ti.f64, shape=())
    input_gradients = ti.Matrix.field(3, 3, ti.f64, shape=2)
    pressure_gradient = ti.field(ti.f64, shape=())
    parameter_gradient = ti.Vector.field(5, ti.f64, shape=())
    output = ti.Matrix.field(3, 3, ti.f64, shape=())
    tangent = ti.Matrix.field(9, 9, ti.f64, shape=())

    @ti.kernel
    def evaluate(equilibrium: ti.template()):
        F = values[0]
        zero = ti.Matrix.zero(ti.f64, 3, 3)
        gF, gB, _, _, gpc, gparam = model.physical_return_vjp(
            0,
            F,
            zero,
            values[2] if not equilibrium else zero,
            state_seed[None][0] if not equilibrium else 0.0,
            state_seed[None][1] if not equilibrium else 0.0,
            state_seed[None][2] if not equilibrium else 0.0,
            values[1] if equilibrium else zero,
            equilibrium,
        )
        input_gradients[0], input_gradients[1] = gF, gB
        pressure_gradient[None], parameter_gradient[None] = gpc, gparam
        output[None] = model.total_first_piola_stress_at(0, F)
        tangent[None] = model.total_first_piola_tangent_at(0, F)

    rng = np.random.default_rng(61)
    stress_seed, inverse_seed = rng.normal(size=(3, 3)) * 1e-5, rng.normal(size=(3, 3))
    state_weights = np.array([0.3, -0.2, 1e-6])
    state_seed[None] = state_weights
    B = np.eye(3)
    pc = 2e5
    for F in (np.diag([1.001, 0.999, 1.0]), np.diag(np.exp([0.01, -0.02, -0.04])), np.eye(3) * 0.99):
        values.from_numpy(np.array([F, stress_seed, inverse_seed]))
        for equilibrium in (True, False):
            evaluate(equilibrium)
            baseline = reference(F, B, pc, parameters)
            np.testing.assert_allclose(output.to_numpy()[()], baseline[0], rtol=1e-7, atol=1e-4)
            gF, gB = input_gradients.to_numpy()
            gpc, gparam = pressure_gradient[None], np.array(parameter_gradient[None])

            def objective(f, b, pressure, params):
                P, inverse, eq, vol, hardening = reference(f, b, pressure, params)
                return (
                    np.sum(stress_seed * P)
                    if equilibrium
                    else np.sum(inverse_seed * inverse) + np.dot(state_weights, [eq, vol, hardening])
                )

            if equilibrium:
                numerical_tangent = np.zeros((9, 9))
                for column in range(9):
                    delta = np.zeros((3, 3))
                    delta[column % 3, column // 3] = 2e-7
                    numerical_tangent[:, column] = (
                        (reference(F + delta, B, pc, parameters)[0] - reference(F - delta, B, pc, parameters)[0]) / 4e-7
                    ).T.ravel()
                np.testing.assert_allclose(tangent.to_numpy()[()], numerical_tangent, rtol=5e-4, atol=1.0)
            for kind in ("F", "B", "pc", 0, 1, 2, 3, 4):
                direction = (
                    rng.normal(size=(3, 3)) * 0.01
                    if kind in ("F", "B")
                    else (1000.0 if kind == "pc" else np.eye(5)[kind] * np.array([0.02, 0.1, 0.01, 0.003, 1000.0]))
                )
                analytic = (
                    np.sum((gF if kind == "F" else gB) * direction)
                    if kind in ("F", "B")
                    else (gpc * direction if kind == "pc" else gparam @ direction)
                )
                for step in (1e-5, 2e-6):
                    probes = []
                    for sign in (1, -1):
                        probes.append(
                            objective(
                                F + sign * step * direction if kind == "F" else F,
                                B + sign * step * direction if kind == "B" else B,
                                pc + sign * step * direction if kind == "pc" else pc,
                                parameters + sign * step * direction if isinstance(kind, int) else parameters,
                            )
                        )
                    numerical = (probes[0] - probes[1]) / (2 * step)
                    assert analytic == pytest.approx(numerical, rel=5e-4, abs=2e-7), (
                        equilibrium,
                        kind,
                        analytic,
                        numerical,
                    )


def test_mcc_clipped_local_jacobian(taichi_runtime):
    model = FiniteStrainModifiedCamClayModel().initialize_from_kwargs(
        density=1800,
        PoissonRatio=0.3,
        StressRatio=1.2,
        lambda_star=0.12,
        kappa_star=0.03,
        void_ratio_ref=0.8,
        pc0=200.0,
        InitialPressure=100.0,
    )
    model.use_direct_consistent_solve = True
    jacobian = ti.Matrix.field(4, 4, ti.f64, shape=())

    @ti.kernel
    def evaluate(state: ti.types.vector(4, ti.f64), trial_trace: ti.f64) -> ti.types.vector(4, ti.f64):
        residual, J = model._consistent_local_system(state, trial_trace, 0.1, 200.0)
        jacobian[None] = J
        return residual

    for values, trace in (
        ([-2.0, 0.1, 1e-4, 300.0], -8.0),
        ([2.0, 0.1, 1e-4, 300.0], 8.0),
        ([0.01, 0.1, 1e-4, 300.0], 0.0),
    ):
        values = np.array(values)
        evaluate(values, trace)
        analytic = jacobian.to_numpy()[()]
        for column in range(4):
            delta = np.zeros(4)
            delta[column] = 1e-6 * max(abs(values[column]), 1e-3)
            numerical = (np.array(evaluate(values + delta, trace)) - np.array(evaluate(values - delta, trace))) / (
                2 * delta[column]
            )
            scale = np.maximum(1.0, np.max(np.abs(analytic), axis=1))
            np.testing.assert_allclose(analytic[:, column] / scale, numerical / scale, rtol=1e-5, atol=1e-7)


def test_mcc_dry_side_softening_return_derivatives(taichi_runtime):
    parameters = np.array([0.3, 1.2, 0.20 / 1.8, 0.05 / 1.8, 1e5])
    model = FiniteStrainModifiedCamClayModel().initialize_from_kwargs(
        density=1800,
        PoissonRatio=parameters[0],
        StressRatio=parameters[1],
        lambda_star=parameters[2],
        kappa_star=parameters[3],
        void_ratio_ref=0.8,
        pc0=2e5,
        InitialPressure=parameters[4],
    )
    model.use_direct_consistent_solve = True
    model.allocate_state(1)
    derivative = ti.Matrix.field(4, 8, ti.f64, shape=())

    @ti.kernel
    def evaluate(trace: ti.f64, norm: ti.f64) -> ti.types.vector(4, ti.f64):
        state = model._solve_consistent_return(trace, norm, model.preconsolidation_pressure[0])
        derivative[None] = model._consistent_return_derivatives(0, trace, norm, state)
        return ti.Vector([state[0], state[1], state[3], state[2]])

    def oracle(trace, norm, pc, params):
        strains = trace / 3 + np.array([norm / np.sqrt(2), -norm / np.sqrt(2), 0])
        F = np.diag(np.exp(strains))
        _, B, eq, volume, pressure = reference(F, np.eye(3), pc, params)
        elastic = F @ B
        principal = np.log(np.linalg.svd(elastic, compute_uv=False))
        t = principal.sum()
        r = np.linalg.norm(principal - t / 3)
        G0 = 3 * params[4] / params[3] * (1 - 2 * params[0]) / (2 * (1 + params[0]))
        G = G0 * np.exp(-t / params[3] + G0 * r * r / (params[3] * params[4]))
        q = np.sqrt(6) * G * r
        return np.array([t, r, eq * params[1] ** 2 / (2 * q), pressure])

    trace, norm, pc = 0.09, 0.07, 2e5
    state = np.array(evaluate(trace, norm))
    assert state[0] < trace and state[3] < pc
    np.testing.assert_allclose(state, oracle(trace, norm, pc, parameters), rtol=1e-8, atol=1e-8)
    J = derivative.to_numpy()[()]
    input_scale = np.array([0.02, 0.02, 1000, 0.02, 0.1, 0.01, 0.003, 1000])
    output_scale = np.array([1, 1, pc, 1 / pc])
    base = np.r_[trace, norm, pc, parameters]
    for column in range(8):
        for step in (1e-5, 2e-6):
            delta = np.eye(8)[column] * input_scale[column] * step
            plus, minus = base + delta, base - delta
            numerical = (
                oracle(plus[0], plus[1], plus[2], plus[3:]) - oracle(minus[0], minus[1], minus[2], minus[3:])
            ) / (2 * step)
            np.testing.assert_allclose(
                J[:, column] * input_scale[column] * output_scale, numerical * output_scale, rtol=5e-4, atol=2e-7
            )

    gradients = ti.Matrix.field(3, 3, ti.f64, shape=2)
    pressure_gradient = ti.field(ti.f64, shape=())
    parameter_gradient = ti.Vector.field(5, ti.f64, shape=())
    F = np.diag(np.exp(trace / 3 + np.array([norm / np.sqrt(2), -norm / np.sqrt(2), 0])))

    @ti.kernel
    def pressure_vjp():
        zero = ti.Matrix.zero(ti.f64, 3, 3)
        deformation = ti.Matrix([[F[i, j] for j in range(3)] for i in range(3)])
        gF, gB, _, _, gpc, gparam = model.physical_return_vjp(
            0, deformation, zero, zero, 0.0, 0.0, 1.0 / pc, zero, False
        )
        gradients[0], gradients[1] = gF, gB
        pressure_gradient[None], parameter_gradient[None] = gpc, gparam

    pressure_vjp()
    rng = np.random.default_rng(94)
    dF, dB = 0.01 * rng.normal(size=(2, 3, 3))
    dparam = np.array([0.02, 0.1, 0.01, 0.003, 1000])
    gF, gB = gradients.to_numpy()
    analytic = (
        np.sum(gF * dF) + np.sum(gB * dB) + 1000 * pressure_gradient[None] + np.array(parameter_gradient[None]) @ dparam
    )
    for step in (1e-5, 2e-6):
        plus = reference(F + step * dF, np.eye(3) + step * dB, pc + step * 1000, parameters + step * dparam)[4]
        minus = reference(F - step * dF, np.eye(3) - step * dB, pc - step * 1000, parameters - step * dparam)[4]
        assert analytic == pytest.approx((plus - minus) / (2 * step * pc), rel=5e-4, abs=2e-7)

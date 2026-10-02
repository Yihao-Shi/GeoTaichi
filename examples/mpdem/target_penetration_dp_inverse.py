"""Invert finite-strain DP parameters from a target penetration depth.

This is the small, deterministic regression used by the MPM--ABD inverse
driver: the device return-map supplies both the reaction and ``dR/dp``; the
one-dimensional equilibrium solve then applies ``dd/dp = -R_p/R_d``.  The
same routine is useful before launching a large coupled IPC scene because it
checks material sensitivities without contact-search noise.
"""

import argparse
import json
import math

import numpy as np
import taichi as ti

from src.physics_model.consititutive_model.finite_strain.DruckerPrager import (
    FiniteStrainDruckerPragerModel,
)


def _model(cohesion, friction_angle):
    model = FiniteStrainDruckerPragerModel().initialize_from_kwargs(
        density=1800.0,
        young_modulus=2.0e4,
        poisson_ratio=0.30,
        Cohesion=float(cohesion),
        FrictionAngle=float(friction_angle),
        DilationAngle=float(friction_angle),
    )
    model.allocate_state(1)
    return model


@ti.data_oriented
class _Response:
    """One compiled device response kernel per constitutive model."""

    def __init__(self, model):
        self.model = model
        self.reaction = ti.field(ti.f64, shape=())
        self.depth_derivative = ti.field(ti.f64, shape=())
        self.parameter_derivative = ti.Vector.field(4, ti.f64, shape=())

    @ti.kernel
    def evaluate(self, depth: ti.f64):
        F = ti.Matrix.identity(ti.f64, 3)
        F[0, 0] = 1.0 + 0.25 * depth
        F[1, 1] = 1.0 + 0.10 * depth
        F[2, 2] = 1.0 - depth
        P = self.model.first_piola_stress_at(0, F)
        tangent = self.model.first_piola_tangent_at(0, F)
        # Positive reaction is resistance to downward penetration.
        self.reaction[None] = -P[2, 2]
        self.depth_derivative[None] = -(tangent[8, 0] * 0.25 + tangent[8, 4] * 0.10 - tangent[8, 8])
        all_derivatives = self.model.first_piola_parameter_derivatives_at(0, F)
        for parameter in ti.static(range(4)):
            self.parameter_derivative[None][parameter] = -all_derivatives[8, parameter]

    def __call__(self, depth):
        self.evaluate(float(depth))
        return (
            float(self.reaction[None]),
            float(self.depth_derivative[None]),
            np.asarray(self.parameter_derivative[None], dtype=np.float64),
        )


def solve_depth(response, target_reaction, initial_depth=0.2):
    depth = float(initial_depth)
    for _ in range(30):
        reaction, derivative, _ = response(depth)
        residual = reaction - target_reaction
        if abs(residual) < 1.0e-10:
            break
        if not math.isfinite(derivative) or abs(derivative) < 1.0e-10:
            raise RuntimeError("DP penetration equilibrium has a singular depth tangent")
        depth = min(0.75, max(1.0e-5, depth - residual / derivative))
    return depth


def invert(args):
    truth_depth = float(args.target_depth)
    truth_model = _model(args.truth_cohesion, args.truth_friction)
    target_response = _Response(truth_model)
    target_reaction = target_response(truth_depth)[0]
    cohesion = float(args.initial_cohesion)
    friction = float(args.initial_friction)
    history = []
    for iteration in range(int(args.iterations)):
        model = _model(cohesion, friction)
        response = _Response(model)
        depth = solve_depth(response, target_reaction)
        reaction, depth_tangent, reaction_parameter = response(depth)
        depth_parameter = -reaction_parameter / depth_tangent
        residual = depth - truth_depth
        gradient = residual * depth_parameter[[2, 3]]
        cohesion -= args.learning_rate * gradient[0]
        friction -= args.learning_rate * gradient[1]
        cohesion = max(1.0e-6, cohesion)
        friction = min(85.0, max(0.01, friction))
        history.append(
            {
                "iteration": iteration,
                "depth": depth,
                "target_depth": truth_depth,
                "cohesion": cohesion,
                "friction_angle_degrees": friction,
                "loss": 0.5 * residual * residual,
            }
        )

    # Directional FD check for the two inverted parameters.
    model = _model(cohesion, friction)
    response = _Response(model)
    depth = solve_depth(response, target_reaction)
    _, depth_tangent, reaction_parameter = response(depth)
    analytic = (depth - truth_depth) * (-reaction_parameter / depth_tangent)
    epsilon = np.array([1.0e-3, 1.0e-3])
    plus_response = _Response(_model(cohesion + epsilon[0], friction + epsilon[1]))
    minus_response = _Response(_model(cohesion - epsilon[0], friction - epsilon[1]))
    plus = solve_depth(plus_response, target_reaction)
    minus = solve_depth(minus_response, target_reaction)
    finite_difference_direction = (0.5 * (plus - truth_depth) ** 2 - 0.5 * (minus - truth_depth) ** 2) / 2.0
    analytic_directional_loss = float(analytic[2] * epsilon[0] + analytic[3] * epsilon[1])
    if not np.isclose(
        finite_difference_direction,
        analytic_directional_loss,
        rtol=5.0e-3,
        atol=1.0e-8,
    ):
        raise AssertionError("DP penetration parameter VJP failed the directional finite-difference check")
    return {
        "target_reaction": target_reaction,
        "estimate": {"cohesion": cohesion, "friction_angle_degrees": friction},
        "history": history,
        "fd_directional_loss": float(finite_difference_direction),
        "analytic_directional_loss": analytic_directional_loss,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-depth", type=float, default=0.22)
    parser.add_argument("--truth-cohesion", type=float, default=140.0)
    parser.add_argument("--truth-friction", type=float, default=30.0)
    parser.add_argument("--initial-cohesion", type=float, default=90.0)
    parser.add_argument("--initial-friction", type=float, default=22.0)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--iterations", type=int, default=8)
    args = parser.parse_args()
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1)
    result = invert(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

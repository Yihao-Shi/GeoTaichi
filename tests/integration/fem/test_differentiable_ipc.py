"""Check the analytic adjoint against independently perturbed forward solves."""

import numpy as np
import pytest
import taichi as ti

from examples.fem.differentiable_ipc import build_problem

pytestmark = [pytest.mark.integration, pytest.mark.fem, pytest.mark.ipc, pytest.mark.cpu, pytest.mark.serial]


def test_plane_elastic_trajectory_adjoint():
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1)
    try:
        simulation = build_problem()
        for _ in range(3):
            simulation.step()
        positions = simulation.solver.positions
        solver = simulation.solver
        seed = np.arange(positions.size, dtype=float).reshape(positions.shape) / positions.size
        velocity_seed = seed * 0.03
        acceleration_seed = seed * 0.0002
        gradient = simulation.backward(seed, velocity_seed, acceleration_seed)

        def objective(**kwargs):
            perturbed = build_problem(**kwargs)
            for _ in range(3):
                perturbed.step()
            final = perturbed.solver.positions
            return np.sum(
                final * seed
                + perturbed.solver.velocity * velocity_seed
                + perturbed.solver.acceleration * acceleration_seed
            )

        for argument, key, value, delta in (
            ("young", "young_modulus", 100.0, 0.1),
            ("kappa", "kappa", 20.0, 0.01),
        ):
            fd = (objective(**{argument: value + delta}) - objective(**{argument: value - delta})) / (2 * delta)
            np.testing.assert_allclose(gradient[key], fd, rtol=3e-4, atol=1e-9)
        delta = 1e-4
        fd = (objective(velocity=(0.2 + delta, 0, 0)) - objective(velocity=(0.2 - delta, 0, 0))) / (2 * delta)
        np.testing.assert_allclose(gradient["initial_velocity"][:, 0].sum(), fd, rtol=3e-4, atol=1e-9)
        fd = (objective(gravity=(0, 0, -9.8 + delta)) - objective(gravity=(0, 0, -9.8 - delta))) / (2 * delta)
        np.testing.assert_allclose(gradient["gravity"][2], fd, rtol=3e-4, atol=1e-9)
    finally:
        ti.reset()

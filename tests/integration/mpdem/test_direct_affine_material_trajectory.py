"""Rebuild compile-time material/gravity constants for complete-forward FD."""

import os

import numpy as np
import pytest
import taichi as ti

from tests.integration.mpdem.test_direct_affine_nonassociated import _case, runtime
from tests.integration.mpdem.test_direct_affine_trajectory_gradients import _fields, _signature

pytestmark = [
    pytest.mark.integration,
    pytest.mark.mpdem,
    pytest.mark.ipc,
    pytest.mark.serial,
    pytest.mark.isolated_dimension(3),
]


@pytest.mark.parametrize(
    "material,dilation",
    [("VonMises", 15.0), ("DruckerPrager", 15.0), ("DruckerPrager", None), ("ModifiedCamClay", 15.0)],
)
def test_material_and_gravity_three_step_vjp_matches_rebuilt_forward(tmp_path, material, dilation):
    names = (
        ("young_modulus", "poisson_ratio", "YieldStress", "HardeningModulus")
        if material == "VonMises"
        else ("young_modulus", "poisson_ratio", "Cohesion", "FrictionAngle")
    )
    values = np.array([1e4, 0.3, 20.0, 100.0] if material == "VonMises" else [1e4, 0.3, 1.0, 30.0])
    direction = np.array([1000.0, 0.02, 0.1, 1.0])
    if material == "ModifiedCamClay":
        names = ("PoissonRatio", "StressRatio", "lambda_star", "kappa_star", "InitialPressure")
        values = np.array([0.3, 1.2, 0.12, 0.03, 100.0])
        direction = np.array([0.02, 0.1, 0.01, 0.003, 10.0])
    gravity = np.array([0.1, -0.2, -0.3])
    gravity_direction = np.array([0.7, -0.3, 0.2])

    def build(shift, suffix):
        coupling = _case(
            tmp_path / suffix,
            material=material,
            dilation=dilation,
            material_parameters=dict(zip(names, values + shift * direction)),
            gravity=gravity + shift * gravity_direction,
        )
        mpm = coupling.enginer.mpm
        deformation = (
            np.diag(np.exp([0.01, -0.02, -0.04])) if material == "ModifiedCamClay" else np.diag([1.08, 0.94, 0.98])
        )
        mpm.F0.from_numpy(np.array([deformation]))
        mpm.material.equivalent_plastic_strain[0] = 0.01
        return coupling.differentiable(steps=3)

    trajectory = build(0.0, "baseline")
    fields = _fields(trajectory)
    rng = np.random.default_rng(29)
    seeds = {name: 0.1 * rng.normal(size=field.to_numpy().shape) for name, field in fields.items()}
    seeds["acceleration"] *= 1e-6
    if material == "ModifiedCamClay":
        seeds["preconsolidation_pressure"] *= 1e-3
    signatures = []
    for _ in range(3):
        forward = trajectory.step(verbose=False)
        assert forward["friction_converged"]
        signatures.append(_signature(trajectory, forward))
    gradient = trajectory.backward(seeds)
    analytic = np.dot(gradient["material_parameters"], direction) + np.dot(gradient["gravity"], gravity_direction)
    assert abs(analytic) > 1e-9
    # Rebuilding is necessary: these model constants are specialized by Taichi.
    for step in (1e-3, 2e-4):
        objectives = []
        for sign in (1, -1):
            ti.reset()
            ti.init(
                arch=ti.cuda if os.environ.get("GEOTAICHI_TEST_ARCH") == "cuda" else ti.cpu,
                default_fp=ti.f64,
                cpu_max_num_threads=1,
                offline_cache=False,
                device_memory_GB=0.25,
            )
            probe = build(sign * step, f"probe_{step}_{sign}")
            for record in range(3):
                forward = probe._advance(False)
                assert forward["friction_converged"]
                assert _signature(probe, forward) == signatures[record]
            objectives.append(sum(np.sum(seeds[name] * field.to_numpy()) for name, field in _fields(probe).items()))
        numerical = (objectives[0] - objectives[1]) / (2 * step)
        assert analytic == pytest.approx(numerical, rel=3e-3, abs=2e-7), (material, step, analytic, numerical)

"""Complete-forward VJPs on the continuum Direct MPM--ABD path, CPU/f64.

Every probe reruns transfers, plastic commits and all friction cache refreshes.
Topology and outer-iteration counts must agree on both sides of each difference.
"""

import numpy as np
import pytest
import taichi as ti

from tests.integration.mpdem.test_direct_affine_nonassociated import _case, runtime

pytestmark = [
    pytest.mark.integration,
    pytest.mark.mpdem,
    pytest.mark.ipc,
    pytest.mark.serial,
    pytest.mark.isolated_dimension(3),
]


def _fields(trajectory):
    mpm, affine = trajectory.mpm, trajectory.affine
    fields = {
        "position": mpm.particle.x,
        "velocity": mpm.particle.v,
        "acceleration": mpm.particle.a,
        "deformation_gradient": mpm.F0,
        "plastic_deformation_inverse": mpm.material.plastic_deformation_inverse,
        "equivalent_plastic_strain": mpm.material.equivalent_plastic_strain,
        "volumetric_plastic_strain": mpm.material.volumetric_plastic_strain,
        "affine_position": affine.y,
        "affine_velocity": affine.velocity_y,
    }
    if hasattr(mpm.material, "preconsolidation_pressure"):
        fields["preconsolidation_pressure"] = mpm.material.preconsolidation_pressure
    return fields


def _signature(trajectory, result):
    mixed, ipc, affine = trajectory.mixed, trajectory.ipc, trajectory.affine
    assert mixed.diagnostics()["minimum_distance"] > 0.001
    if ipc.ground.num:
        gap = np.dot(ipc.mpm.p_temp.to_numpy()[0] - ipc.ground.pos.to_numpy()[0], ipc.ground.norm.to_numpy()[0])
        assert gap > ipc.barrier.dmin[0]

    def rows(field, count):
        return sorted(tuple(row) for row in field.to_numpy()[:count])

    mixed_features = sorted(
        (tuple(candidate), tuple(abs(weight) > 1e-10), bool(load > 0.0))
        for candidate, weight, load in zip(
            mixed.friction_candidate.to_numpy()[: mixed.friction_count],
            mixed.friction_weight.to_numpy()[: mixed.friction_count],
            mixed.friction_normal_force.to_numpy()[: mixed.friction_count],
        )
    )
    return (
        result["friction_iterations"],
        rows(mixed.friction_candidate, mixed.friction_count),
        rows(affine.friction_contact_vertices, int(affine.friction_contact_count[0])),
        int(ipc.gfrictionNum[0]),
        mixed_features,
    )


@pytest.mark.parametrize(
    "material,epsv,ground,bodies",
    [
        ("VonMises", 1e-3, False, 1),
        ("DruckerPrager", 1.0, False, 1),
        ("DruckerPrager", 1e-3, False, 1),
        ("VonMises", 1e-3, True, 2),
    ],
)
def test_three_step_friction_state_and_parameter_vjps(tmp_path, material, epsv, ground, bodies):
    _check_three_step_vjps(tmp_path, material, epsv, ground, bodies)


def test_shared_grid_two_particle_friction_vjp(tmp_path):
    _check_three_step_vjps(tmp_path, "VonMises", 1e-3, False, 1, particle_count=2)


def _check_three_step_vjps(tmp_path, material, epsv, ground, bodies, particle_count=1):
    coupling = _case(
        tmp_path,
        material=material,
        dilation=None if material == "DruckerPrager" and epsv == 1e-3 else 15.0,
        ground=ground,
        affine_bodies=bodies,
        particle_count=particle_count,
    )
    system, mpm = coupling.enginer, coupling.enginer.mpm
    system.mixed.pair_epsv.fill(epsv)
    system.ipc.friction.epsv[0] = epsv
    system.affine.epsv = epsv
    mpm.F0.from_numpy(np.tile(np.diag([1.08, 0.94, 0.98]), (particle_count, 1, 1)))
    if particle_count == 2:
        supports = mpm.LnID.to_numpy()
        assert set(supports[0, : mpm.offset[0]]) & set(supports[1, : mpm.offset[1]])
    mpm.material.equivalent_plastic_strain[0] = 0.01
    trajectory = coupling.differentiable(steps=3)
    assert ti.lang.impl.current_cfg().default_fp == ti.f64
    fields = _fields(trajectory)
    initial = {name: field.to_numpy().copy() for name, field in fields.items()}
    rng = np.random.default_rng(921)
    seeds = {name: 0.1 * rng.normal(size=value.shape) for name, value in initial.items()}
    seeds["acceleration"] *= 1e-6
    parameters = {
        "mixed_friction_coefficient": system.mixed.pair_friction,
        "affine_young_modulus": system.affine.young,
        "affine_gravity": system.affine.gravity,
    }
    if ground:
        parameters["friction_coefficient"] = system.ipc.friction.mu
    if bodies > 1:
        parameters["friction_scale"] = system.affine.friction_scale
    initial_parameters = {name: field.to_numpy().copy() for name, field in parameters.items()}
    signatures = []
    slip_ratios = []
    for _ in range(3):
        forward = trajectory.step(verbose=False)
        assert forward["friction_converged"]
        assert system.mixed.friction_count > 0
        if ground:
            assert system.ipc.gfrictionNum[0] > 0
        if bodies > 1:
            assert system.affine.friction_contact_count[0] > 0
        signatures.append(_signature(trajectory, forward))
        count = system.mixed.friction_count
        positions = system.mixed.positions.to_numpy()
        references = system.mixed.friction_hat.to_numpy()
        for stencil, weights, normal, load in zip(
            system.mixed.friction_candidate.to_numpy()[:count],
            system.mixed.friction_weight.to_numpy()[:count],
            system.mixed.friction_normal.to_numpy()[:count],
            system.mixed.friction_normal_force.to_numpy()[:count],
        ):
            if load > 0.0:
                relative = weights @ (positions[stencil] - references[stencil]) / mpm.dt
                tangent = relative - normal * np.dot(normal, relative)
                slip_ratios.append(np.linalg.norm(tangent) / epsv)
    assert slip_ratios
    assert min(slip_ratios) < 1.0 if epsv == 1.0 else max(slip_ratios) > 1.0
    assert mpm.material.equivalent_plastic_strain[0] > 0.010001
    terminal = {name: field.to_numpy().copy() for name, field in fields.items()}
    gradient = trajectory.backward(seeds)
    repeated = trajectory.backward(seeds)
    for key in gradient:
        np.testing.assert_allclose(gradient[key], repeated[key], rtol=1e-12, atol=1e-12)
    for name, field in fields.items():
        np.testing.assert_array_equal(field.to_numpy(), terminal[name])

    def objective(name, direction, shift):
        for key, field in fields.items():
            field.from_numpy(initial[key] + (shift * direction if key == name else 0.0))
        for key, field in parameters.items():
            field.from_numpy(initial_parameters[key] + (shift * direction if key == name else 0.0))
        mpm.time, mpm.step_count = 0.0, 0
        for record in range(3):
            forward = trajectory._advance(False)
            assert forward["friction_converged"]
            assert _signature(trajectory, forward) == signatures[record]
        return sum(np.sum(seeds[key] * field.to_numpy()) for key, field in fields.items())

    for name, values in {**initial, **initial_parameters}.items():
        direction = rng.normal(size=values.shape)
        direction *= {
            "affine_young_modulus": 2000.0,
            "affine_gravity": 1000.0,
            "acceleration": 1000.0,
            "friction_scale": 1000.0,
        }.get(name, 0.02)
        key = (
            ("affine_initial_" + name[7:] if name.startswith("affine_") else "initial_" + name)
            if name in fields
            else name
        )
        analytic = np.sum(gradient[key] * direction)
        for step in (1e-5, 2e-6):
            numerical = (objective(name, direction, step) - objective(name, direction, -step)) / (2 * step)
            assert analytic == pytest.approx(numerical, rel=3e-3, abs=2e-7), (
                material,
                ground,
                name,
                step,
                analytic,
                numerical,
            )

    if ground:
        # Keep right probes on one branch with converged ground/self caches.
        coupling.dem.sims.affine_friction_iterations = 8
        # At zero coefficient, FD differences are small relative to the primal loss.
        coupling.dem.sims.affine_newton_tolerance = 1e-10
        coupling.dem.sims.affine_friction_tolerance = 1e-9
        mpm.linear_solver_tolerance = 1e-11
        mpm.linear_solver_relative_tolerance = 1e-13
        for name in ("mixed_friction_coefficient", "friction_coefficient", "friction_scale"):
            saved = initial_parameters[name]
            initial_parameters[name] = np.zeros_like(saved)
            for key, field in fields.items():
                field.from_numpy(initial[key])
            for key, field in parameters.items():
                field.from_numpy(initial_parameters[key])
            mpm.time, mpm.step_count, trajectory.record_count = 0.0, 0, 0
            signatures = []
            for _ in range(3):
                forward = trajectory.step(verbose=False)
                assert forward["friction_converged"], (name, forward)
                signatures.append(_signature(trajectory, forward))
            baseline = sum(np.sum(seeds[key] * field.to_numpy()) for key, field in fields.items())
            zero_gradient = trajectory.backward(seeds)
            direction = np.ones_like(saved)
            analytic = np.sum(zero_gradient[name] * direction)
            assert abs(analytic) > 1e-10
            # Coefficients are nonnegative: use a second-order right difference at zero.
            for step in (1e-4, 2e-5):
                numerical = (
                    -3 * baseline + 4 * objective(name, direction, step) - objective(name, direction, 2 * step)
                ) / (2 * step)
                assert analytic == pytest.approx(numerical, rel=3e-3, abs=2e-7), (name, step, analytic, numerical)
            initial_parameters[name] = saved

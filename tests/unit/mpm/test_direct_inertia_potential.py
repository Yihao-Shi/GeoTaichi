"""The implicit nodal potential differentiates to its assembled physical forces."""

import numpy as np
import pytest
import taichi as ti

pytestmark = [pytest.mark.unit, pytest.mark.cpu, pytest.mark.serial, pytest.mark.isolated_dimension(2)]


@pytest.mark.parametrize("integration", [(1.0, 0.5, 1.0), (0.5, 0.25, 0.5), (0.7, 0.25, 0.6), (0.7, 0.25, 0.0)])
@pytest.mark.parametrize("timestep", [0.0005, 0.0025])
@pytest.mark.parametrize("damping", [0.0, 0.05])
def test_nodal_inertia_and_damping_potential_matches_force_and_tangent(
    taichi_runtime, monkeypatch, integration, timestep, damping
):
    import src.mpm.config as config
    from src.mpm.engines.direct.ImplicitMPM import ImplicitMPM

    config.set_dimension(2)
    monkeypatch.setattr(config, "DYNAMIC", True)
    engine = object.__new__(ImplicitMPM)
    engine.damping = damping
    engine.val_lim = 1e-12
    node = ti.types.struct(m=ti.f64, v=ti.types.vector(2, ti.f64), a=ti.types.vector(2, ti.f64))
    engine.grid = node.field(shape=3)
    engine.grid.m.from_numpy(np.array([2.0, 0.0, 3.0]))
    engine.grid.v.from_numpy(np.array([[0.2, -0.1], [17.0, 19.0], [-0.3, 0.15]]))
    engine.grid.a.from_numpy(np.array([[0.4, -0.3], [23.0, 29.0], [-0.2, 0.5]]))
    engine.node2dof = ti.field(ti.i32, shape=3)
    engine.node2dof.from_numpy(np.array([1, 1, 2], dtype=np.int32))
    engine.TIdt = ti.field(ti.f64, shape=())
    engine.TIdt[None] = timestep
    engine.energy = ti.field(ti.f64, shape=())
    engine.grid_disp = ti.field(ti.f64, shape=4)
    engine.rhs = ti.field(ti.f64, shape=4)
    engine.mass_vec = ti.field(ti.f64, shape=4)
    engine.volume_force = ti.field(ti.f64, shape=4)
    engine.volume_force.from_numpy(np.array([1.5, -2.0, -0.5, 0.75]))
    gravity = [0.1, -9.81]
    displacement = np.array([2e-5, -3e-5, -4e-5, 1e-5])
    step = 1e-9

    def potential(values):
        engine.grid_disp.from_numpy(values)
        engine.energy.fill(0)
        engine.get_inertia_energy(engine.damping, integration, gravity, engine.grid_disp)
        return float(engine.energy[None])

    def gradient(values):
        engine.grid_disp.from_numpy(values)
        engine.assemble_inertia_force(4, engine.damping, gravity, integration, engine.grid_disp)
        return -engine.rhs.to_numpy()

    analytic = gradient(displacement)
    numerical = np.zeros(4)
    numerical_tangent = np.zeros((4, 4))
    for dof in range(4):
        perturbation = np.eye(4)[dof] * step
        numerical[dof] = (potential(displacement + perturbation) - potential(displacement - perturbation)) / (2 * step)
        numerical_tangent[:, dof] = (gradient(displacement + perturbation) - gradient(displacement - perturbation)) / (
            2 * step
        )
    np.testing.assert_allclose(numerical, analytic, rtol=1e-7, atol=1e-7)
    engine.compute_mass_list(integration)
    np.testing.assert_allclose(numerical_tangent, np.diag(engine.mass_vec.to_numpy()), rtol=1e-7, atol=1e-7)

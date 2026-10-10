"""Convergence follows represented particle motion, strain, and force balance."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from src.igampm.engines.ImplicitEngine import ImplicitEngineMixin
from src.igampm.engines import Engine
from src.igampm import config


class _MPMKinematics(SimpleNamespace):
    __hash__ = object.__hash__


def _kinematic_engine(axisymmetric=False):
    engine = object.__new__(Engine)
    dim = config.DIM
    engine.iga = SimpleNamespace(degree_of_freedom=dim)
    particle = ti.Struct.field({"x": ti.types.vector(dim, ti.f64), "bodyID": ti.i32}, shape=1)
    body = ti.Struct.field({"grid_size": ti.f64}, shape=1)
    body.grid_size[0] = 0.1
    particle.x[0] = [0.01] + [0.0] * (dim - 1)
    engine.mpm = _MPMKinematics(
        particle=particle,
        body=body,
        particleNum=ti.field(ti.i32, shape=1),
        offset=ti.field(ti.i32, shape=1),
        LnID=ti.field(ti.i32, shape=(1, 2)),
        node2dof=ti.field(ti.i32, shape=2),
        shape=ti.field(ti.f64, shape=(1, 2)),
        dshape=ti.Vector.field(dim, ti.f64, shape=(1, 2)),
        is_axisymmetric=axisymmetric,
        axis_offset=0.0,
    )
    engine.mpm.particleNum[0] = 1
    engine.mpm.offset[0] = 2
    engine.mpm.LnID.from_numpy(np.array([[0, 1]], dtype=np.int32))
    engine.mpm.node2dof.from_numpy(np.array([1, 2], dtype=np.int32))
    engine.monolithic_correction = ti.field(ti.f64, shape=3 * dim)
    return engine


def test_weak_grid_mode_is_measured_through_particles_and_gradient(taichi_runtime):
    engine = _kinematic_engine()
    engine.mpm.shape.from_numpy(np.array([[1.0 - 1e-8, 1e-8]]))
    engine.mpm.dshape.from_numpy(np.array([[[-1e-5, 0.0], [1e-5, 0.0]]]))
    engine.monolithic_correction.from_numpy(np.array([0.0, 0.0, 0.0, 0.0, 1e-8, 0.0]))
    # Raw grid velocity is 1e-5; the represented strain controls this norm.
    assert engine._device_monolithic_correction_residual(4, 0.001, 0.001) == pytest.approx(1e-11)


def test_zero_particle_translation_cannot_hide_nonzero_strain(taichi_runtime):
    engine = _kinematic_engine()
    engine.mpm.shape.from_numpy(np.array([[0.5, 0.5]]))
    engine.mpm.dshape.from_numpy(np.array([[[-10.0, 0.0], [10.0, 0.0]]]))
    engine.monolithic_correction.from_numpy(np.array([0.0, 0.0, -1e-4, 0.0, 1e-4, 0.0]))
    assert engine._device_monolithic_correction_residual(4, 0.001, 0.001) == pytest.approx(0.2)


def test_axisymmetric_norm_includes_hoop_strain(taichi_runtime):
    engine = _kinematic_engine(axisymmetric=True)
    engine.mpm.shape.from_numpy(np.array([[0.5, 0.5]]))
    engine.mpm.dshape.from_numpy(np.array([[[-10.0, 0.0], [10.0, 0.0]]]))
    engine.monolithic_correction.from_numpy(np.array([0.0, 0.0, 1e-4, 0.0, 1e-4, 0.0]))
    assert engine._device_monolithic_correction_residual(4, 0.001, 0.001) == pytest.approx(1.0)


def test_fempm_embedded_grid_uses_represented_correction(taichi_runtime):
    from src.fempm.ImplicitEngine import FEMPMImplicitEngine

    kinematics = _kinematic_engine()
    engine = object.__new__(FEMPMImplicitEngine)
    engine.mpm = kinematics.mpm
    engine.mpm.active_dof = 4
    engine.mpm_dimension = 2
    engine.fem_nodes = 1
    engine.mpm.shape.from_numpy(np.array([[1.0 - 1e-8, 1e-8]]))
    engine.mpm.dshape.from_numpy(np.array([[[-1e-5, 0.0], [1e-5, 0.0]]]))
    engine.correction = ti.field(ti.f64, shape=9)
    engine.correction[6] = 1e-8
    engine.rhs = ti.field(ti.f64, shape=9)
    engine.fixed = ti.field(ti.i32, shape=9)
    engine.fixed_correction = ti.field(ti.f64, shape=9)
    for name in ("residual_squared", "directional_derivative", "correction_inf_norm", "constraint_inf_norm"):
        setattr(engine, name, ti.field(ti.f64, shape=()))
    engine._reduce_system_metrics(9)
    assert engine.correction_inf_norm[None] == pytest.approx(1e-14, abs=1e-20)


@pytest.mark.parametrize("axisymmetric", [False, True])
def test_standalone_mpm_lagged_probe_uses_particle_measure(taichi_runtime, axisymmetric):
    from src.mpm.soft_particle.IPCMPM import IPCMPM

    engine = _kinematic_engine(axisymmetric=axisymmetric)
    engine.mpm.shape.from_numpy(np.array([[0.5, 0.5]]))
    engine.mpm.dshape.from_numpy(np.array([[[-10.0, 0.0], [10.0, 0.0]]]))
    engine.mpm.active_dof = 4
    engine.mpm.dt = 0.001
    engine.mpm.incre_resolution = ti.field(ti.f64, shape=4)
    engine.mpm.incre_resolution.from_numpy(np.array([1e-4, 0.0, 1e-4, 0.0]))
    contact = object.__new__(IPCMPM)
    contact.mpm = engine.mpm
    contact.friction_mode = "lagged"
    contact.solve_current_system = lambda: {"solution_inf_norm": 1e-4}
    before = engine.mpm.incre_resolution.to_numpy()
    assert contact.updated_friction_system_residual() == pytest.approx(1.0 if axisymmetric else 0.1)
    np.testing.assert_array_equal(engine.mpm.incre_resolution.to_numpy(), before)


@pytest.mark.isolated_dimension(3)
def test_three_dimensional_norm_detects_transverse_strain(taichi_runtime):
    engine = _kinematic_engine()
    engine.mpm.shape.from_numpy(np.array([[0.5, 0.5]]))
    engine.mpm.dshape.from_numpy(np.array([[[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]]))
    engine.monolithic_correction.from_numpy(np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1e-4, 0.0, 0.0, 1e-4]))
    assert engine._device_monolithic_correction_residual(6, 0.001, 0.001) == pytest.approx(0.2)
    from src.mpdem.engines.DirectAffineIPCSystem import DirectAffineIPCSystem

    system = object.__new__(DirectAffineIPCSystem)
    system.mpm = engine.mpm
    system.mpm.incre_resolution = ti.field(ti.f64, shape=6)
    system.mpm.incre_resolution.from_numpy(np.array([0.0, 0.0, -1e-4, 0.0, 0.0, 1e-4]))
    system.affine = SimpleNamespace(device_surface_direction_inf_norm=lambda: 0.0)
    assert system.direction_inf_norm(6) == pytest.approx(2e-4)


def _newton_engine(forces, corrections, inexact=False):
    engine = object.__new__(ImplicitEngineMixin)
    engine.nonassociated_newton = True
    engine.monolithic_inexact_newton = inexact
    engine.monolithic_force_atol = 1e-10
    engine.monolithic_force_rtol = 1e-8
    engine.monolithic_dirichlet_tolerance = 1e-12
    engine.monolithic_linear_solver_relative_tolerance = 1e-7
    engine.monolithic_solver_name = "BiCGSTAB"
    engine.is_semi = False
    engine.semi_contact_converged = lambda: True
    engine.mpm_has_plastic_history = True
    engine.iga = SimpleNamespace(dt=0.001)
    engine.mpm = SimpleNamespace(dt=0.001, material=SimpleNamespace(has_incremental_potential=False))
    engine.assemble_monolithic_newton_system = lambda **kwargs: {"active_dof": 6, "active_mpm_dof": 4}
    force_values = iter(forces)
    correction_values = iter(corrections)
    engine._device_monolithic_free_rhs_norm = lambda n: next(force_values)
    engine._device_dirichlet_residual = lambda n: 0.0
    engine._device_monolithic_correction_residual = lambda *args: next(correction_values)
    engine._split_device_monolithic_correction = lambda n: None
    engine._solve_monolithic_linear_system = lambda *args, **kwargs: {"converged": True}
    engine._newton_linear_tolerance = lambda *args: 0.01
    engine._material_feasible_step_device = lambda: 1.0
    engine._assemble_device_physical_tangent_product = lambda **kwargs: -1.0
    engine.fully_implicit_residual_armijo_device = lambda *args, **kwargs: {"accepted": False}
    return engine


def _solve(engine, iterations=1):
    return engine._solve_monolithic_newton_device(
        include_friction=True, max_iterations=iterations, tolerance=1e-8, energy_function=None, verbose=False
    )


def test_later_friction_inner_solve_preserves_force_scale_but_requires_motion_convergence():
    engine = _newton_engine([1e-3, 1e-7], [0.1, 1e-9])
    engine._monolithic_force_reference = 1e6
    engine.fully_implicit_residual_armijo_device = lambda *args, **kwargs: {"accepted": True, "step": 1.0}
    result = _solve(engine, iterations=2)
    assert result["converged"]
    assert result["iterations"] == 1
    assert result["force_tolerance"] == pytest.approx(0.0100000001)
    assert result["residual"] == pytest.approx(1e-9)


def test_small_correction_alone_cannot_accept_unbalanced_force():
    engine = _newton_engine([0.1], [1e-12])
    result = _solve(engine)
    assert not result["converged"]
    assert result["force_residual"] > result["force_tolerance"]


def test_inexact_terminal_correction_is_rechecked_with_configured_accuracy():
    engine = _newton_engine([1e-9], [1e-12, 0.1], inexact=True)
    solves = []

    def linear(system, **kwargs):
        solves.append(system["linear_relative_tolerance"])
        return {"converged": True}

    engine._solve_monolithic_linear_system = linear
    result = _solve(engine)
    assert solves == [0.01, 1e-7]
    assert not result["converged"]

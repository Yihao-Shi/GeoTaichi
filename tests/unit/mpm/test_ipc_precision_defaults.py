"""Public IPC construction preserves explicit precision and supplies defaults."""

import importlib
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("configuration", ["ULMPM", "TLMPM"])
@pytest.mark.parametrize("override", [None, "direct", "implicit"])
def test_public_ipc_precision_reaches_both_solver_layers(taichi_runtime, monkeypatch, configuration, override):
    from src.mpm.mainMPM import MPM

    wrapper = importlib.import_module("src.mpm.soft_particle.IPC" + configuration)
    captured = {}

    def inner(*args, **kwargs):
        captured["inner"] = kwargs
        return SimpleNamespace(tol=kwargs["residual"])

    def contact(*args, **kwargs):
        captured["contact"] = kwargs
        return object()

    monkeypatch.setattr(wrapper, "Implicit" + configuration, inner)
    monkeypatch.setattr(wrapper, "IPCMPM", contact)
    mpm = MPM(log=False)
    mpm.set_configuration(
        mpm_backend="Direct",
        solver_type="Implicit",
        configuration=configuration,
        ipc=True,
        domain=[1.0, 1.0, 1.0],
        log=False,
    )
    mpm.direct_bodies = SimpleNamespace(bodies={"one": object()})
    mpm.direct_ground = object()
    expected = 1e-8
    if override == "direct":
        mpm.direct_solver.update(
            residual=2e-5,
            linear_solver_tolerance=3e-10,
            linear_solver_relative_tolerance=2e-9,
            friction_tolerance=4e-6,
        )
        expected = 2e-5
    elif override == "implicit":
        mpm.set_implicit_solver_parameters(residual_tolerance=9e-6, linear_solver_relative_tolerance=2e-9)
        mpm.set_implicit_solver_parameters(residual_tolerance=3e-6)
        expected = 3e-6
    mpm._build_direct_engine()
    assert captured["inner"]["residual"] == expected
    assert captured["contact"]["residual"] == expected
    assert captured["inner"]["linear_solver_tolerance"] == (3e-10 if override == "direct" else 1e-12)
    if override is not None:
        assert captured["inner"]["linear_solver_relative_tolerance"] == 2e-9
    if override == "direct":
        assert captured["contact"]["friction_tolerance"] == 4e-6

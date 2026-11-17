"""A saved two-phase trajectory must resume like the uninterrupted computation."""

import importlib

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.mpm, pytest.mark.cpu, pytest.mark.slow]


@pytest.mark.parametrize("dimension", [2, pytest.param(3, marks=pytest.mark.isolated_dimension(3))])
def test_two_phase_consolidation_restart_matches_uninterrupted(tmp_path, monkeypatch, taichi_runtime, dimension):
    name = "DoublePointConsolidation" + ("3D" if dimension == 3 else "")
    case = importlib.import_module(f"examples.mmpm.Consolidation.{name}.{name}")
    monkeypatch.setattr(case, "init", lambda **_: None)
    changes = dict(
        CELL_SIZE=0.1,
        COLUMN_WIDTH=0.2,
        COLUMN_HEIGHT=0.4,
        BOUNDARY_CELL_THICKNESS=2,
        COLUMN_ORIGIN=np.full(dimension, 0.2),
        DT=1e-5,
        SIMULATION_TIME=8e-5,
        SAVE_INTERVAL=4e-5,
        MG_LEVELS=2,
        TOP_LOAD_THICKNESS=0.1,
    )
    if dimension == 2:
        changes.update(DOMAIN_WIDTH=0.6, DOMAIN_HEIGHT=0.8)
    else:
        changes.update(COLUMN_DEPTH=0.2, DOMAIN=[0.6, 0.6, 0.8], MAX_PARTICLE_NUMBER=2048)
    for key, value in changes.items():
        monkeypatch.setattr(case, key, value)
    continuous_path, resumed_path = tmp_path / "continuous", tmp_path / "resumed"
    monkeypatch.setattr(case, "SAVE_PATH", str(continuous_path))
    continuous = case.create_mpm()
    continuous.run()
    monkeypatch.setattr(case, "SAVE_PATH", str(resumed_path))
    resumed = case.create_mpm()
    resumed.read_restart(1, str(continuous_path))
    assert resumed.sims.current_step == 4
    resumed.run()
    assert resumed.sims.current_step == continuous.sims.current_step == 8
    assert resumed.scene.two_phase_fluid_mass_initialized
    assert resumed.enginer.fluid_volume_initialized
    assert resumed.sims.current_time == continuous.sims.current_time == 8e-5
    for name in (
        "x",
        "v",
        "vs",
        "vf",
        "m",
        "ms",
        "mf",
        "vol",
        "porosity",
        "pressure",
        "stress",
        "solid_velocity_gradient",
        "fluid_velocity_gradient",
    ):
        left = getattr(continuous.scene.particle, name).to_numpy()
        right = getattr(resumed.scene.particle, name).to_numpy()
        count = int(continuous.scene.particleNum[0])
        np.testing.assert_allclose(right[:count], left[:count], rtol=1e-9, atol=1e-10, err_msg=name)
    assert int(resumed.scene.particleNum[0]) == int(continuous.scene.particleNum[0])
    np.testing.assert_array_equal(resumed.scene.psize, continuous.scene.psize)


def test_single_layer_restart_matches_uninterrupted(tmp_path, monkeypatch, taichi_runtime):
    import geotaichi as gt
    from tests.integration.mpm.test_twophase_single_layer_semiimplicit import run_case

    monkeypatch.setattr(gt, "init", lambda **_: None)
    original_factory, models = gt.MPM, []

    def factory(*args, **kwargs):
        model = original_factory(*args, **kwargs)
        original_run = model.run

        def run():
            model.sims.time, model.sims.save_interval = 4e-4, 2e-4
            if models:
                model.read_restart(1, str(tmp_path / "continuous"))
                assert model.sims.current_step == 2
                # The existing case helper checks frame 1 in its separate output directory.
                model.sims.current_print = 0
            original_run()
            models.append(model)

        model.run = run
        return model

    monkeypatch.setattr(gt, "MPM", factory)
    run_case("loaded_pcg", tmp_path / "continuous")
    run_case("loaded_pcg", tmp_path / "resumed")
    left, right = models
    assert left.sims.current_step == right.sims.current_step == 4
    count = int(left.scene.particleNum[0])
    for name in (
        "x",
        "v",
        "vs",
        "vf",
        "ms",
        "mf",
        "vol",
        "porosity",
        "pressure",
        "stress",
        "solid_velocity_gradient",
        "fluid_velocity_gradient",
    ):
        np.testing.assert_allclose(
            getattr(right.scene.particle, name).to_numpy()[:count],
            getattr(left.scene.particle, name).to_numpy()[:count],
            rtol=1e-9,
            atol=1e-10,
            err_msg=name,
        )

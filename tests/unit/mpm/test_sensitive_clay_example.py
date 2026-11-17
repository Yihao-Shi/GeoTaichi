"""Independent saved-state checks must detect corrupted adaptive histories."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def saved_run(tmp_path):
    (tmp_path / "particles").mkdir()
    (tmp_path / "vtks").mkdir()
    manifest = {
        "status": "completed",
        "duration": 0.05,
        "save_interval": 0.05,
        "dt": 1e-4,
        "particle_capacity": 8,
        "command": ["synthetic test"],
        "backend": "test",
        "precision": "float64",
    }
    (tmp_path / "run.json").write_text(json.dumps(manifest))
    for index, number in enumerate((1, 4)):
        np.savez(
            tmp_path / "particles" / f"MPMParticle{index:06d}.npz",
            position=np.tile([2.0, 0.5], (number, 1)),
            velocity=np.zeros((number, 2)),
            stress=np.zeros((number, 6)),
            volume=np.full(number, 1 / number),
            mass=np.full(number, 2000 / number),
            active=np.ones(number),
            grid_level=np.full(number, index),
            refined_cell=np.array([index]),
            t_current=index * 0.05,
            state_vars={"epdstrain": np.full(number, index * 0.02)},
        )
        (tmp_path / "vtks" / f"GraphicMPMParticle{index:06d}.vtu").touch()
    return tmp_path


def evaluator():
    path = Path(__file__).resolve().parents[3] / "examples/mpm/sensitive_clay/draw/evaluate_sensitive_clay.py"
    spec = importlib.util.spec_from_file_location("evaluate_sensitive_clay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate


def test_split_mass_and_physical_times(saved_run):
    result = evaluator()(saved_run)
    assert result["passed"]
    assert result["max_relative_mass_error"] == 0
    assert 'timestep="0.05"' in (saved_run / "sensitive_clay.pvd").read_text()


@pytest.mark.parametrize(
    "key, value, check",
    [
        ("mass", 501.0, "mass_balance"),
        ("stress", np.nan, "finite_state"),
        ("active", 0, "positive_mass_volume_active_particles"),
        ("volume", -1, "positive_mass_volume_active_particles"),
        ("t_current", 0.04, "completed"),
    ],
)
def test_rejects_invalid_saved_state(saved_run, key, value, check):
    path = saved_run / "particles/MPMParticle000001.npz"
    with np.load(path, allow_pickle=True) as data:
        changed = dict(data)
    changed[key] = np.full_like(changed[key], value)
    np.savez(path, **changed)
    result = evaluator()(saved_run)
    assert not result["passed"]
    assert not result["checks"][check]


def test_rejects_missing_reference_plateau(saved_run):
    path = saved_run / "particles/MPMParticle000000.npz"
    with np.load(path, allow_pickle=True) as data:
        changed = dict(data)
    changed["position"][:, 0] = 0.5
    np.savez(path, **changed)
    with pytest.raises(ValueError, match="no reference-plateau particles"):
        evaluator()(saved_run)

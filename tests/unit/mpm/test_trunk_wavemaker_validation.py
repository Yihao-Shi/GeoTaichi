import json
from types import SimpleNamespace

import numpy as np
import pytest

from examples.mmpm.TwoPhaseLSDEMCoupling.wavemaker_lsdem_particles_3d.wavemaker_lsdem_particles_3d import (
    embedded_trunk_shift,
)
from examples.mmpm.TwoPhaseLSDEMCoupling.wavemaker_lsdem_particles_3d.draw import (
    evaluate_wavemaker_lsdem_particles_3d as evaluator,
)


def test_tree_is_embedded_using_actual_bounds_not_equivalent_sphere_radius():
    for bottom in (-0.186, 0.083):
        vertices = np.array([[0.7, 0.1, bottom], [0.8, 0.15, bottom + 0.24]])
        vertices[:, 2] += embedded_trunk_shift(vertices, 0.02)
        assert vertices[:, 2].min() == pytest.approx(0.05)
        assert vertices[:, 2].max() < 0.34
    with pytest.raises(ValueError):
        embedded_trunk_shift(np.empty((0, 3)), 0.02)


def test_lost_tree_writes_failed_metrics_instead_of_broadcasting_error(tmp_path, monkeypatch):
    (tmp_path / "particles").mkdir()
    (tmp_path / "grids").mkdir()
    monkeypatch.setattr(evaluator, "sdf_surface_gauge", lambda *args: 0.24)
    for step, (time, count) in enumerate(((0.0, 5), (4.0, 4))):
        np.savez(
            tmp_path / "particles" / f"MPMParticle{step:06d}.npz",
            t_current=time,
            active=[1, 1],
            phase=[2, 1],
            position=[[0.2, 0.1, 0.1], [0.8, 0.1, 0.05]],
            fluid_velocity=np.zeros((2, 3)),
            solid_velocity=np.zeros((2, 3)),
            pressure=[0, 0],
        )
        np.savez(
            tmp_path / "particles" / f"LSDEMRigid{step:06d}.npz",
            t_current=time,
            mass_center=np.zeros((count, 3)),
            contact_force=np.zeros((count, 3)),
        )
        np.savez(
            tmp_path / "grids" / f"MPMGrid{step:06d}.npz",
            t_current=time,
            cell_type=np.ones((2, 2, 2)),
            cell_fluid_sdf=np.ones((2, 2, 2)),
        )
    args = SimpleNamespace(
        dx=0.02,
        dt=5e-5,
        time=4.0,
        frequency=0.8,
        wave_velocity=0.1,
        soil_young_modulus=12e6,
        soil_cohesion=300,
        soil_friction=26,
        strict=False,
    )
    evaluator.write_metrics(tmp_path, 1, 1, args)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert not metrics["passed"]
    assert metrics["minimum_lsdem_bodies"] == 4
    assert not metrics["lsdem_body_count_constant"]

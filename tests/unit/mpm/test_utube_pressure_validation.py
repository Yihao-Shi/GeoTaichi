import numpy as np
import pytest
import json
import runpy
from pathlib import Path

from examples.mmpm.UTubeFlow2D.draw.evaluate_u_tube_flow_2d import hydrostatic_gradient_error
from examples.mmpm.UTubeFlow2D.draw import evaluate_u_tube_flow_2d as evaluator


def test_utube_gradient_check_rejects_real_pressure_oscillation_but_not_gauge_offsets():
    position = np.array([[x, y] for x in (0.5, 2.5) for y in np.linspace(1.2, 2.4, 20)])
    pressure = 9810 * (2.5 - position[:, 1])
    assert hydrostatic_gradient_error(position, pressure) == pytest.approx(0, abs=1e-12)
    assert hydrostatic_gradient_error(position, pressure + 14000) == pytest.approx(0, abs=1e-12)
    assert hydrostatic_gradient_error(position, pressure * 1.3) == pytest.approx(0.3)


def test_utube_default_step_is_stable_without_changing_ppc(monkeypatch):
    monkeypatch.delenv("GT_UTUBE_2D_DT", raising=False)
    monkeypatch.delenv("GT_UTUBE_2D_PPC", raising=False)
    namespace = runpy.run_path(str(Path(evaluator.__file__).parents[1] / "u_tube_flow_2d_parameters.py"))
    assert namespace["DT"] == 0.001 and namespace["PPC"] == 2


def test_utube_strict_queue_rejects_bad_final_velocity_after_writing_metrics(tmp_path, monkeypatch):
    (tmp_path / "particles").mkdir()
    monkeypatch.setattr(evaluator, "EXPECTED_FLUID_PARTICLES", 40)
    monkeypatch.setattr(evaluator, "EXPECTED_SOLID_PARTICLES", 1)
    monkeypatch.setenv("GT_UTUBE_2D_STRICT", "1")
    for step, time in enumerate((0.0, 100.0)):
        left = np.column_stack([np.full(20, 0.5), np.linspace(1.2, 2.0, 20) + (1.0 if step == 0 else 0.0)])
        right = np.column_stack([np.full(20, 2.5), np.linspace(1.2, 2.0, 20)])
        position = np.vstack([left, right, [1.5, 0.5]])
        velocity = np.zeros((41, 2))
        if step == 1:
            velocity[:40, 0] = 1.0
        np.savez(
            tmp_path / "particles" / f"MPMParticle{step:06d}.npz",
            t_current=time,
            active=np.ones(41),
            phase=[2] * 40 + [1],
            position=position,
            solid_velocity=np.zeros((41, 2)),
            fluid_velocity=velocity,
            pressure=9810 * (3.0 - position[:, 1]),
        )
    with pytest.raises(RuntimeError, match="U-tube strict validation failed"):
        evaluator.postprocess(tmp_path)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert not metrics["passed"] and metrics["duration_complete"]
    assert metrics["late_fluid_speed_p99_max_mps"] == 1.0

from pathlib import Path

import numpy as np

from examples.cfdem.SemiResolved.SphereFallingOil.sphere import case_parameters
from examples.cfdem.SemiResolved.SphereFallingOil.draw.evaluate_sphere import evaluate, load_velocity_experiment


def test_ten_cate_experiment_is_dependency_free_and_unchanged():
    path = Path("examples/cfdem/SemiResolved/SphereFallingOil/experiment.csv")
    data = load_velocity_experiment(path)

    assert data.shape == (40, 2)
    assert np.isfinite(data).all()
    assert np.isclose(data[0, 0], 0.070167064439140808)
    assert np.isclose(data[:, 1].min(), -0.12292682926829268)
    assert np.isclose(data[-1, 0], 1.2095465393794749)


def test_ten_cate_acceptance_uses_the_physically_reachable_pre_wall_history():
    experiment = load_velocity_experiment(Path("examples/cfdem/SemiResolved/SphereFallingOil/experiment.csv"))
    acceleration = 9.81 * (1120.0 - 960.0) / (1120.0 + 2.0 * 960.0)
    times = np.unique(np.r_[0.0, 0.00025, experiment[:, 0], 1.25])
    velocity = np.interp(times, np.r_[0.0, experiment[:, 0]], np.r_[0.0, experiment[:, 1]])
    velocity[1] = -acceleration * times[1]
    velocities = np.zeros((len(times), 1, 3))
    velocities[:, 0, 2] = velocity
    centers = np.tile(np.array([[[0.05, 0.05, 0.1275]]]), (len(times), 1, 1))
    config = case_parameters()

    metrics = evaluate(config, times, centers, velocities, 0.0, 0.0, experiment=experiment)
    assert metrics["passed"]

    inconsistent_late_history = experiment.copy()
    inconsistent_late_history[inconsistent_late_history[:, 0] > 1.0, 1] = -10.0
    late_metrics = evaluate(
        config,
        times,
        centers,
        velocities,
        0.0,
        0.0,
        experiment=inconsistent_late_history,
    )
    assert late_metrics["passed"]
    assert late_metrics["experiment_velocity_rmse_over_peak"] > metrics["experiment_velocity_rmse_over_peak"]

"""Check force comparison thresholds and historical thickness normalization."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "footing_evaluation",
    Path(__file__).resolve().parents[3] / "examples/mpm/Footing/strip_footing_adaptive/draw/evaluate.py",
)
EVALUATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATION)


def test_force_comparison_accepts_five_percent_and_rejects_fifteen_percent():
    settlement = np.linspace(0, 1, 1001)
    reference = np.column_stack([settlement, 5000 * (1 - np.exp(-20 * settlement))])
    for scale, passed in [(1.05, True), (1.15, False)]:
        candidate = reference.copy()
        candidate[:, 1] *= scale
        metrics, _ = EVALUATION.curve_errors(reference, candidate)
        assert metrics["passed"] is passed
        np.testing.assert_allclose(metrics["relative_rms"], abs(scale - 1), rtol=1e-12)
        np.testing.assert_allclose(metrics["peak_relative_error"], abs(scale - 1), rtol=1e-12)
    with pytest.raises(ValueError, match="common settlement"):
        EVALUATION.curve_errors(reference, reference + [2, 0])


def test_historical_force_is_per_thickness_and_uses_measured_settlement(tmp_path):
    (tmp_path / "grids").mkdir()
    (tmp_path / "particles").mkdir()
    for i in range(3):
        position = np.array([[0.5, 0.125, 10.5 - i * 0.03], [0.5, 0.175, 10.5 - i * 0.03]])
        np.savez(
            tmp_path / "particles" / f"MPMParticle{i:06d}.npz",
            t_current=i,
            bodyID=[1, 1],
            position=position,
            mass=[1, 1],
            psize=np.full((2, 3), 0.025),
        )
        force = np.zeros((1, 2, 3))
        force[0, 1, 2] = i * 100
        np.savez(tmp_path / "grids" / f"MPMGrid{i:06d}.npz", t_current=i, contact_force=force)
    output = tmp_path / "reference.csv"
    EVALUATION.extract_reference(tmp_path, output)
    values = np.genfromtxt(output, delimiter=",", names=True)
    np.testing.assert_allclose(values["footing_fy_N_per_m"], [0, 1000, 2000])
    np.testing.assert_allclose(values["settlement_m"], [0, 0.03, 0.06], atol=1e-12)


def test_force_comparison_rejects_a_spike_hidden_by_bin_means():
    settlement = np.linspace(0, 1, 10001)
    reference = np.column_stack([settlement, np.full_like(settlement, 5000)])
    candidate = reference.copy()
    candidate[5000, 1] = 15000
    metrics, _ = EVALUATION.curve_errors(reference, candidate)
    assert metrics["relative_rms"] < 0.01
    assert metrics["raw_peak_relative_error"] == pytest.approx(2.0)
    assert not metrics["passed"]


def test_raw_peak_interpolates_the_boundaries_of_a_sparse_common_interval():
    reference = np.array([[0, 5000], [0.5, 5000], [1, 5000]])
    candidate = np.array([[0.1, 5000], [0.2, 5000], [0.3, 5000]])
    metrics, _ = EVALUATION.curve_errors(reference, candidate)
    assert metrics["passed"]
    assert metrics["raw_peak_relative_error"] == 0


@pytest.mark.parametrize("spike_time,passed", [(6.95, True), (7.0, False)])
def test_historical_sampling_gates_observed_peaks_and_retains_unfiltered_diagnostics(spike_time, passed):
    times = np.arange(17501) / 1000
    dtype = [("time_s", float), ("settlement_m", float), ("footing_fy_N_per_m", float)]
    candidate = np.zeros(len(times), dtype=dtype)
    candidate["time_s"], candidate["settlement_m"] = times, 0.1 * times
    candidate["footing_fy_N_per_m"] = 7000 + 500 * np.sin(2 * np.pi * times)
    reference = candidate[::250].copy()
    candidate["footing_fy_N_per_m"][round(spike_time * 1000)] = 50000
    metrics, samples = EVALUATION.sampled_curve_errors(reference, candidate, reference["time_s"])
    assert metrics["passed"] is passed
    assert not metrics["unfiltered_diagnostics"]["passed"]
    assert metrics["sampling_interval_s"] == pytest.approx(0.25)
    np.testing.assert_allclose(samples[:, 0], reference["time_s"])
    if passed:
        assert metrics["relative_rms"] == pytest.approx(0)
        assert metrics["oscillation_amplitude"]["candidate_N_per_m"] == pytest.approx(1000)


def test_sampled_oscillation_amplitude_is_reported_independently_of_mean_load():
    times = np.arange(71) / 4
    reference = np.zeros(len(times), dtype=[("time_s", float), ("settlement_m", float), ("footing_fy_N_per_m", float)])
    reference["time_s"], reference["settlement_m"] = times, times / 10
    reference["footing_fy_N_per_m"] = 7000 + 500 * np.sin(2 * np.pi * times)
    candidate = reference.copy()
    candidate["footing_fy_N_per_m"] = 7000 + 200 * np.sin(2 * np.pi * times)
    metrics, _ = EVALUATION.sampled_curve_errors(reference, candidate, times)
    amplitude = metrics["oscillation_amplitude"]
    assert amplitude["cycles"] == 12
    assert amplitude["reference_N_per_m"] == pytest.approx(1000)
    assert amplitude["candidate_N_per_m"] == pytest.approx(400)
    assert amplitude["relative_difference"] == pytest.approx(0.6)


@pytest.mark.parametrize("volume_ratio,speed,passed", [(1, 0.02, True), (1e-20, 0.02, False), (1, 100, False)])
def test_preflight_rejects_finite_but_collapsed_or_explosive_state(tmp_path, volume_ratio, speed, passed):
    import json

    (tmp_path / "particles").mkdir()
    run = dict(
        status="completed",
        duration=0.1,
        dt=1e-5,
        effective_dt=1e-5,
        particle_count=2,
        level=0,
        command=[],
        backend="cpu",
        precision="float64",
        smoke=False,
    )
    (tmp_path / "run.json").write_text(json.dumps(run))
    history = np.zeros((3, 9))
    history[:, 0] = [0.02, 0.06, 0.1]
    history[:, 2:4] = [1, -1]
    np.savetxt(
        tmp_path / "contact_history.csv",
        history,
        delimiter=",",
        header="time_s,fx,footing_fy_N_per_m,soil_fy_N_per_m,settlement_m,refined,l1_coarse_cells,l2_coarse_cells,active",
        comments="",
    )
    for i, ratio in enumerate([1, volume_ratio]):
        np.savez(
            tmp_path / "particles" / f"MPMParticle{i:06d}.npz",
            active=[1, 1],
            bodyID=[0, 1],
            position=[[0.5, 0.5], [0.5, 1.5]],
            velocity=[[0, speed if i else 0], [0, -0.02]],
            stress=np.zeros((2, 6)),
            mass=[1, 1],
            volume=[ratio, 1],
            psize=np.full((2, 2), 0.025),
        )
    report = EVALUATION.preflight(tmp_path)
    assert report["finite_state"] and report["positive_mass_volume"]
    assert report["response_bounded"] is passed
    assert report["passed"] is passed


@pytest.mark.parametrize("volume_ratio,passed", [(1.0, True), (0.01, False)])
def test_split_volume_check_uses_mass_not_original_particle_indices(tmp_path, volume_ratio, passed):
    import json

    (tmp_path / "particles").mkdir()
    run = dict(
        status="completed",
        duration=0.1,
        dt=1e-5,
        effective_dt=1e-5,
        particle_count=2,
        level=1,
        refine_particles=True,
        command=[],
        backend="cpu",
        precision="float64",
        smoke=True,
    )
    (tmp_path / "run.json").write_text(json.dumps(run))
    for index in (0, 1):
        # The rigid particle stays at index 1 while three soil children are appended.
        body = np.array([0, 1] if index == 0 else [0, 1, 0, 0, 0])
        mass = np.array([16.0, 1.5] if index == 0 else [4.0, 1.5, 4.0, 4.0, 4.0])
        volume = mass / np.where(body == 0, 1000.0, 1500.0)
        if index:
            volume[body == 0] *= volume_ratio
        pos = np.array([[0.5, 0.5] if b == 0 else [0.5, 1.5] for b in body])
        np.savez(
            tmp_path / "particles" / f"MPMParticle{index:06d}.npz",
            active=np.ones(len(body)),
            bodyID=body,
            position=pos,
            mass=mass,
            volume=volume,
            velocity=np.zeros((len(body), 2)),
            stress=np.zeros((len(body), 6)),
            psize=np.full((len(body), 2), 0.025),
            grid_level=np.ones(len(body)),
            particle_refined=np.where(body == 0, index, 0),
        )
    np.savetxt(
        tmp_path / "contact_history.csv",
        [[0.05, 1, -1, 0, 1, 0, 2], [0.1, 1, -1, 0, 1, 0, 5]],
        delimiter=",",
        comments="",
        header="time_s,footing_fy_N_per_m,soil_fy_N_per_m,settlement_m,l1_coarse_cells,l2_coarse_cells,particle_count",
    )
    report, _ = EVALUATION.evaluate_run(tmp_path)
    assert report["particle_splitting_observed"] and report["particle_count_matches_history"]
    assert report["rigid_particle_count_preserved"] and report["relative_mass_error"] == 0
    assert report["minimum_volume_ratio"] == pytest.approx(volume_ratio)
    assert report["passed"] is passed


def test_joint_refinement_initial_state_allows_coarse_soil_but_checks_ppc_and_rigid_geometry(tmp_path):
    import json

    for level, name in enumerate(("uniform", "adaptive_l1", "adaptive_l2")):
        target = tmp_path / name
        (target / "particles").mkdir(parents=True)
        h = 0.1 * 2**level
        spacing = h / 4
        axis = np.arange(spacing / 2, 1, spacing)
        xy = np.array(np.meshgrid(axis, axis, indexing="ij")).reshape(2, -1).T
        body = np.r_[np.zeros(len(xy), dtype=int), 1]
        volume = np.r_[np.full(len(xy), spacing**2), 0.025**2]
        np.savez(
            target / "particles/MPMParticle000000.npz",
            bodyID=body,
            position=np.vstack([xy, [0.0125, 1.0125]]),
            volume=volume,
            mass=volume * np.where(body == 0, 1000.0, 1500.0),
            psize=np.vstack([np.full((len(xy), 2), spacing / 2), [0.0125, 0.0125]]),
        )
        (target / "run.json").write_text(json.dumps(dict(level=level, coarse_h=h, refine_particles=level > 0)))
    assert all(EVALUATION.initial_discretization_checks(tmp_path).values())
    path = tmp_path / "adaptive_l2/particles/MPMParticle000000.npz"
    with np.load(path) as data:
        bad = {key: data[key].copy() for key in data.files}
    bad["psize"][0] *= 1.1
    np.savez(path, **bad)
    assert not EVALUATION.initial_discretization_checks(tmp_path)["adaptive_l2"]

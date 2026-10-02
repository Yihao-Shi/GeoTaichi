from types import SimpleNamespace
import importlib

import numpy as np
import pytest

from examples.cfdem.SemiResolved.SphereFallingOil.sphere import (
    case_parameters as oil_case_parameters,
    evaluate as evaluate_oil,
)
from examples.cfdem.FullyResolved.IBMResolvedSphereSettling.sphere_settling import (
    case_parameters as resolved_sphere_case_parameters,
    evaluate as evaluate_resolved_sphere,
)
from examples.cfdem.FullyResolved.IBMDraftingKissingTumbling.drafting_kissing_tumbling import (
    adaptive_dem_timestep,
    case_parameters as dkt_case_parameters,
    contact_substep_required,
    evaluate as evaluate_dkt,
    validate_description as validate_dkt_description,
)
from src.mpdem.mainDEMPM import DEMPM
from src.mpdem.Engine import Engine, dem_substep_count
from src.mpdem.Simulation import Simulation


def test_dem_substep_count_covers_coupled_step_without_roundoff_extra_step():
    assert dem_substep_count(5.0e-5, 1.0e-7) == 500
    assert dem_substep_count(5.01e-5, 1.0e-7) == 501


@pytest.mark.parametrize("scheme", ["DEM", "LSDEM"])
def test_contact_substeps_are_limited_to_incompressible_cfdem(scheme):
    sims = object.__new__(Simulation)
    sims.enhanced_coupling = False
    sims.coupling_scheme, sims.cfdem_resolution = "CFDEM", "Auto"
    sims.dem_timestep, sims.delta = 0.1, 1.0
    fluid = SimpleNamespace(
        solver_type="Implicit", material_type="Fluid", discretization="FDM", dimension=3, sparse_grid=False
    )
    dem = SimpleNamespace(scheme=scheme)
    sims.validate_coupling_configuration(fluid, dem)
    fluid.solver_type = "Explicit"
    with pytest.raises(RuntimeError, match="subcycling requires incompressible CFDEM"):
        sims.validate_coupling_configuration(fluid, dem)


def test_oil_workbook_case_parameters_are_explicit():
    e4 = oil_case_parameters()
    assert e4["fluid_density"] == 960.0 and e4["viscosity"] == 0.058
    assert e4["reference_speed"] == pytest.approx(0.12224)
    assert e4["reference_statistic"] == "peak"
    assert e4["tank_contact"] and e4["dem_dt"] < e4["dt"]
    assert e4["centers"][0][2] - e4["diameter"] / 2 == pytest.approx(0.120)


def test_settling_speed_match_cannot_pass_with_a_sphere_through_the_floor():
    config = resolved_sphere_case_parameters()
    config = {**config, "maximum_sample_interval": 0.2}
    times = np.linspace(0.0, config["time"], 5)
    centers = np.tile(config["centers"], (5, 1, 1))
    centers[-1, 0] = [0.05, 0.05, 0.06]
    velocities = np.zeros((5, 1, 3))
    velocities[1:, 0, 2] = -config["reference_speed"]
    assert evaluate_resolved_sphere(config, times, centers, velocities, 0.0, 0.0)["passed"]
    centers[-1, 0, 2] = 0.005
    assert not evaluate_resolved_sphere(config, times, centers, velocities, 0.0, 0.0)["passed"]


def test_settling_peak_match_cannot_hide_unfinished_acceleration():
    config = resolved_sphere_case_parameters()
    times = np.linspace(0.0, config["time"], 21)
    centers = np.tile(config["centers"], (len(times), 1, 1))
    velocities = np.zeros_like(centers)
    velocities[:, 0, 2] = -np.linspace(0.0, config["reference_speed"], len(times))

    metrics = evaluate_resolved_sphere(config, times, centers, velocities, 0.0, 0.0)
    assert metrics["maximum_speed_relative_error"] == pytest.approx(0.0)
    assert not metrics["velocity_plateau_sampled"]
    assert not metrics["passed"]

    velocities[:, 0, 2] = 0.0
    velocities[len(times) // 2, 0, 2] = -config["reference_speed"]
    assert not evaluate_resolved_sphere(config, times, centers, velocities, 0.0, 0.0)["passed"]


def test_oil_peak_match_cannot_hide_transient_error_or_missing_samples():
    config = oil_case_parameters()
    times = np.array([0.0, 0.01, 0.5, 1.25])
    velocities = np.zeros((4, 1, 3))
    initial_acceleration = (
        9.81
        * (config["particle_density"] - config["fluid_density"])
        / (config["particle_density"] + config["added_mass_coefficient"] * config["fluid_density"])
    )
    velocities[:, 0, 2] = [0.0, -initial_acceleration * 0.01, -config["reference_speed"], 0.0]
    centers = np.tile(config["centers"], (4, 1, 1))
    experiment = np.column_stack([times, velocities[:, 0, 2]])
    assert evaluate_oil(config, times, centers, velocities, 0, 0, experiment)["passed"]
    contact_centers = centers.copy()
    contact_centers[:, 0, 2] = 0.5 * config["diameter"] - 1.0e-8
    contact_metrics = evaluate_oil(config, times, contact_centers, velocities, 0, 0, experiment)
    assert contact_metrics["passed"]
    contact_centers[:, 0, 2] -= 2.0 * contact_metrics["elastic_contact_overlap_tolerance"]
    assert not evaluate_oil(config, times, contact_centers, velocities, 0, 0, experiment)["passed"]
    mismatched = experiment.copy()
    mismatched[2, 1] *= 0.5
    assert not evaluate_oil(config, times, centers, velocities, 0, 0, mismatched)["passed"]
    truncated = times.copy()
    truncated[-1] = 1.20  # Old 95%-duration check alone accepts this.
    assert not evaluate_oil(config, truncated, centers, velocities, 0, 0, experiment)["passed"]


def test_dkt_validation_requires_tumbling_and_wall_clearance():
    config = validate_dkt_description(dkt_case_parameters())
    times = np.array([0.0, 0.20, 0.34, 0.37, 0.60])
    centers = np.array(
        [
            config["centers"],
            [[0.005, 0.005, 0.025], [0.0050167, 0.005, 0.028]],
            [[0.005, 0.005, 0.021], [0.0050167, 0.005, 0.0229]],
            [[0.005, 0.005, 0.020], [0.0050167, 0.005, 0.021655]],
            [[0.0045, 0.005, 0.010], [0.0065, 0.005, 0.008]],
        ]
    )
    velocities = np.zeros((5, 2, 3))
    velocities[1, :, 2] = [-0.05, -0.06]
    velocities[2, 1, 2] = -config["reference_collision_speed"]
    velocities[3, 1, 2] = -0.06

    config = {**config, "maximum_sample_interval": 0.30}
    assert evaluate_dkt(config, times, centers, velocities, 0.01, 0.0)["passed"]
    loose_speed = velocities.copy()
    loose_speed[2, 1, 2] = -0.94 * config["reference_collision_speed"]
    assert not evaluate_dkt(config, times, centers, loose_speed, 0.01, 0.0)["passed"]
    late_peak = velocities.copy()
    late_peak[2, 1, 2] = -0.06
    late_peak[3, 1, 2] = -config["reference_collision_speed"]
    assert not evaluate_dkt(config, times, centers, late_peak, 0.01, 0.0)["passed"]
    barely_separated = centers.copy()
    barely_separated[-1] = [[0.0059, 0.005, 0.0100], [0.00405, 0.005, 0.0097]]
    assert not evaluate_dkt(config, times, barely_separated, velocities, 0.01, 0.0)["passed"]
    excessive_overlap = centers.copy()
    excessive_overlap[3, 1, 2] = excessive_overlap[3, 0, 2] + 0.98 * config["diameter"]
    assert not evaluate_dkt(config, times, excessive_overlap, velocities, 0.01, 0.0)["passed"]
    centers[-1, 1, 2] = 0.0007
    assert not evaluate_dkt(config, times, centers, velocities, 0.01, 0.0)["passed"]


def test_dkt_old_wide_acceptance_is_rejected_by_quantitative_checks():
    config = validate_dkt_description(dkt_case_parameters())
    config = {**config, "maximum_sample_interval": 0.30}
    times = np.array([0.0, 0.20, 0.34, 0.37, 0.60])
    centers = np.array(
        [
            config["centers"],
            [[0.005, 0.005, 0.025], [0.0050167, 0.005, 0.028]],
            [[0.005, 0.005, 0.021], [0.0050167, 0.005, 0.0229]],
            [[0.005, 0.005, 0.020], [0.0050167, 0.005, 0.021655]],
            [[0.0045, 0.005, 0.010], [0.0065, 0.005, 0.008]],
        ]
    )
    velocities = np.zeros((5, 2, 3))
    velocities[1, :, 2] = [-0.05, -0.06]
    velocities[2, 1, 2] = -config["reference_collision_speed"]
    velocities[3, 1, 2] = -0.06

    assert not evaluate_dkt(config, times, centers, velocities, 0.10, 0.02)["passed"]


def test_dkt_uses_small_dem_steps_before_contact_only():
    config = dkt_case_parameters()
    diameter = config["diameter"]
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0 * diameter]])
    velocities = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -0.1]])
    assert not contact_substep_required(config, config["dt"], positions, velocities)
    shortened_dt = np.nextafter(config["dt"], 0.0)
    assert adaptive_dem_timestep(config, config["dem_dt"], shortened_dt, positions, velocities) == shortened_dt
    positions[1, 2] = diameter + 1.5 * 0.1 * config["dt"]
    assert contact_substep_required(config, config["dt"], positions, velocities)
    assert adaptive_dem_timestep(config, config["dem_dt"], config["dt"], positions, velocities) == config["dem_dt"]


def test_dkt_default_acceptance_cannot_be_loosened_below_paper_scale():
    config = dkt_case_parameters()
    assert config["time"] == pytest.approx(0.60)
    assert config["contact_time_relative_tolerance"] <= 0.05
    assert config["collision_speed_relative_tolerance"] <= 0.05
    assert config["solid_volume_relative_tolerance"] <= 0.04
    assert config["ibm_velocity_relative_tolerance"] <= 0.01
    assert config["maximum_overlap_ratio"] <= 0.015
    assert config["minimum_final_distance_ratio"] >= 1.10
    assert config["minimum_post_kissing_separation_ratio"] >= 0.15
    assert config["minimum_lateral_growth_ratio"] >= 1.0


@pytest.mark.parametrize("scheme", ["sphere", "lsdem"])
def test_default_dem_path_keeps_original_call_order(scheme):
    engine = object.__new__(Engine)
    calls = []
    engine.sims = SimpleNamespace(dem_timestep=1.0, delta=1.0)
    engine.msims = engine.mscene = engine.mneighbor = object()
    engine.dsims = engine.dneighbor = object()
    engine.dscene = SimpleNamespace(particle=object(), rigid=object())
    engine._lsdem_contact_search = lambda: calls.append("search")
    engine._sphere_contact_search = lambda: calls.append("search")
    engine.resolve_cross_contact = lambda: calls.append("cross-contact")
    engine.update_servo_wall = lambda: calls.append("servo")
    engine.get_wall_contact_forces = lambda: calls.append("wall-force")
    engine.mengine = SimpleNamespace(compute=lambda *_: calls.append("fluid"))
    engine.accumulate_pressure_force = lambda *_: calls.append("fluid-force")
    engine.dengine = SimpleNamespace(
        system_resolve=lambda *_: calls.append("dem-contact"),
        integration=lambda *_: calls.append("dem-integrate"),
    )

    if scheme == "sphere":
        engine.incompressible_dem_sphere_integration()
    else:
        engine.incompressible_lsdem_integration()

    assert calls == [
        "search",
        "dem-contact",
        "cross-contact",
        "servo",
        "wall-force",
        "fluid",
        "fluid-force",
        "dem-integrate",
    ]


@pytest.mark.parametrize("scheme", ["sphere", "lsdem"])
def test_contact_free_dem_honors_requested_substeps(scheme):
    engine = object.__new__(Engine)
    timesteps = []
    engine.sims = SimpleNamespace(dem_timestep=0.1, delta=1.0)
    engine.msims = engine.mscene = engine.mneighbor = object()
    engine.dsims = SimpleNamespace(set_timestep=timesteps.append)
    engine.dneighbor = object()
    engine.dscene = SimpleNamespace(particle=object(), rigid=object())
    engine._lsdem_contact_search = lambda: None
    engine._sphere_contact_search = lambda: None
    engine._subcycle_incompressible_dem_contact = lambda dt, *_: timesteps.append(dt)
    engine.resolve_cross_contact = engine.update_servo_wall = engine.get_wall_contact_forces = lambda: None
    engine.mengine = SimpleNamespace(compute=lambda *_: None)
    engine.accumulate_pressure_force = lambda *_: None
    engine.dengine = SimpleNamespace(system_resolve=lambda *_: None, integration=lambda *_: None)

    if scheme == "sphere":
        engine.incompressible_dem_sphere_integration()
    else:
        engine.incompressible_lsdem_integration()

    assert timesteps == [1.0]


@pytest.mark.parametrize("scheme", ["sphere", "lsdem"])
@pytest.mark.parametrize("fail_contact", [False, True])
def test_dem_substeps_hold_fluid_load_without_accumulating_contact(monkeypatch, scheme, fail_contact):
    module = importlib.import_module("src.mpdem.Engine")
    body = np.zeros(1)
    monkeypatch.setattr(module, "cache_dem_external_load", lambda n, b, f, t: np.copyto(f, b))
    monkeypatch.setattr(module, "restore_dem_external_load", lambda n, b, f, t: np.copyto(b, f))
    engine = object.__new__(Engine)
    timesteps, loads, fluid_calls, wall_resets = [], [], [], []
    engine.sims = SimpleNamespace(dem_timestep=0.3, delta=1.0)
    engine.dsims = SimpleNamespace(set_timestep=timesteps.append)
    engine.dscene = SimpleNamespace(particle=body, rigid=body, particleNum=[1])
    engine.msims = engine.mscene = engine.mneighbor = engine.dneighbor = object()
    engine._sphere_contact_search = engine._lsdem_contact_search = lambda: None
    engine._ensure_dem_external_load_cache = lambda: None
    engine.dem_external_force = np.zeros(1)
    engine.dem_external_torque = None
    engine.resolve_cross_contact = engine.update_servo_wall = engine.get_wall_contact_forces = lambda: None

    def fluid(*_):
        fluid_calls.append(1)
        body[:] += 7.0

    def contact(*_):
        if fail_contact:
            raise RuntimeError("contact failure")
        body[:] += 2.0

    engine.mengine = SimpleNamespace(compute=fluid)
    engine.accumulate_pressure_force = lambda *_: None
    engine.dengine = SimpleNamespace(
        reset_particle_message=lambda *_: body.fill(0.0),
        reset_wall_message=lambda *_: wall_resets.append(1),
        system_resolve=contact,
        integration=lambda *_: loads.append(float(body[0])),
    )
    integrate = (
        engine.incompressible_dem_sphere_integration if scheme == "sphere" else engine.incompressible_lsdem_integration
    )
    if fail_contact:
        with pytest.raises(RuntimeError, match="contact failure"):
            integrate()
    else:
        integrate()
        assert loads == [9.0] * 4
    assert fluid_calls == [1]
    assert timesteps == [0.25, 1.0]
    assert len(wall_resets) == (1 if fail_contact else 4)


def test_implicit_cfdem_does_not_apply_explicit_mpm_wave_speed_limit():
    coupling = object.__new__(DEMPM)
    coupling.dem = SimpleNamespace(sims=object(), get_critical_timestep=lambda: 2.5e-4)
    coupling.mpm = SimpleNamespace(
        sims=SimpleNamespace(solver_type="Implicit"),
        scene=SimpleNamespace(
            get_critical_timestep=lambda: pytest.fail("implicit MPM must not query the explicit acoustic CFL limit")
        ),
    )
    observed = []
    coupling.sims = SimpleNamespace(
        coupling_scheme="CFDEM",
        dem_timestep=1.0,
        delta=1.0,
        update_critical_timestep=lambda msims, dsims, dt: observed.append(dt),
    )

    coupling.check_critical_timestep()

    assert observed == [pytest.approx(2.5e-4)]

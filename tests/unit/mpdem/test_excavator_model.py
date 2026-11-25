"""Geometry and prescribed phases of the minimal stationary excavation model."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def model():
    folder = Path(__file__).resolve().parents[3] / "examples/mpdem/AffineBody/ExcavatorDig"
    spec = importlib.util.spec_from_file_location("excavator_dig", folder / "excavator_dig.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, json.loads((folder / "model-contract.json").read_text())


def test_bucket_is_a_closed_solid_with_an_open_cavity():
    module, contract = model()
    mesh = module.bucket_mesh(contract["geometry"]["dimensions"])
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.volume == pytest.approx(0.26 * 0.24 * 0.20 - 0.24 * 0.20 * 0.18, abs=1e-15)
    assert np.all(np.linalg.eigvalsh(mesh.moment_inertia) > 0)


@pytest.mark.parametrize(
    "time,expected",
    [
        (0.0, [0.0, 0.0]),
        (0.25, [0.0, 54.46232220802562]),
        (0.8, [33.0, 52.48546406361587]),
        (1.05, [30.0, -20.0]),
        (1.4, [-20.0, -70.0]),
        (1.6, [-20.0, -70.0]),
    ],
)
def test_motion_curls_the_bucket_and_preserves_its_pitch_during_lift(time, expected):
    module, contract = model()
    np.testing.assert_allclose(
        module.targets(time, contract["boundary_conditions"][2], contract["geometry"]["dimensions"]),
        expected,
        atol=1e-12,
    )


def test_soil_sides_constrain_only_the_outward_normal_component():
    module, _ = model()
    nx, ny, nz = 4, 5, 3
    fixed = module.soil_side_dofs([nx, ny, nz])
    assert len(fixed[0]) == 2 * ny * nz and len(fixed[1]) == 2 * nx * nz
    for component, dofs in enumerate(fixed):
        for dof in dofs:
            assert dof % 3 == component
            node = dof // 3
            coordinate = node % nx if component == 0 else (node // nx) % ny
            assert coordinate in (0, (nx if component == 0 else ny) - 1)


def test_nominal_bucket_cuts_near_ground_and_clears_it_throughout_curl():
    module, contract = model()
    geometry, motion = contract["geometry"]["dimensions"], contract["boundary_conditions"][2]
    pivot = np.array(geometry["pivot"])
    arm = np.array(geometry["bucket_hinge"]) - pivot
    mesh = module.bucket_mesh(geometry)
    clearances = []
    for time in np.linspace(0, motion["finish"], 1001):
        boom, relative = np.radians(module.targets(time, motion, geometry))
        pitch = boom - relative
        hinge_z = pivot[2] - np.sin(boom) * arm[0] + np.cos(boom) * arm[2]
        z = hinge_z - np.sin(pitch) * mesh.vertices[:, 0] + np.cos(pitch) * mesh.vertices[:, 2]
        clearances.append(z.min() - geometry["ground_z"])
    assert min(clearances) > 0
    assert (
        clearances[round(motion["dig_until"] / motion["finish"] * 1000)]
        < 0.5 * contract["discretization"]["grid_spacing"]
    )


def test_cutting_tip_keeps_its_height_and_starts_without_an_angle_velocity_jump():
    module, contract = model()
    geometry, motion = contract["geometry"]["dimensions"], contract["boundary_conditions"][2]
    pivot = np.array(geometry["pivot"])
    link = np.array(geometry["bucket_hinge"]) - pivot
    for time in np.linspace(motion["insert_until"], motion["dig_until"], 101):
        boom, relative = np.radians(module.targets(time, motion, geometry))
        pitch = boom - relative
        hinge_z = pivot[2] - np.sin(boom) * link[0] + np.cos(boom) * link[2]
        tip_z = hinge_z - np.sin(pitch) * geometry["bucket_x"][0] + np.cos(pitch) * geometry["bucket_z"][0]
        assert tip_z - geometry["ground_z"] == pytest.approx(motion["cutting_height"], abs=1e-12)
    time, epsilon = motion["insert_until"], 1e-4
    velocity = (
        np.array(module.targets(time + epsilon, motion, geometry)) - module.targets(time - epsilon, motion, geometry)
    ) / (2 * epsilon)
    assert abs(velocity).max() < 1.0  # degrees/s; the unsquared ramp jumps by tens of degrees/s.

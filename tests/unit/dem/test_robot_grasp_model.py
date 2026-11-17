"""The minimal grasp example imports safely and keeps its declared motion."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize(
    "time,expected",
    [
        (0.0, [0, 0, 0]),
        (0.1, [0, 0, 0]),
        (0.3, [0, -16, 16]),
        (0.7, [-20, -16, 16]),
        (1.0, [-20, -16, 16]),
        (1.2, [-20, 0, 0]),
        (1.6, [-20, 0, 0]),
    ],
)
def test_grasp_protocol_targets_match_declared_phases(time, expected):
    folder = Path(__file__).resolve().parents[3] / "examples/dem/AffineBody/robot_grasp"
    spec = importlib.util.spec_from_file_location("robot_grasp", folder / "robot_grasp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contract = json.loads((folder / "model-contract.json").read_text())
    np.testing.assert_allclose(module.joint_targets(time, contract["boundary_conditions"][2]), expected, atol=1e-12)

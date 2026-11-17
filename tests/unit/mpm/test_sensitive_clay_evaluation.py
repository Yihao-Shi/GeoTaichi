"""Paper invariant and head-scarp measurement on saved states."""

import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "clay_evaluation",
    Path(__file__).resolve().parents[3] / "examples/mpm/sensitive_clay/draw/evaluate_sensitive_clay.py",
)
EVALUATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATION)


def test_paper_invariant_matches_tensor_and_existing_scalar_exports():
    tensor = np.array([[[0.1, 0.04, 0.02], [0.04, -0.03, 0], [0.02, 0, -0.07]]])
    voigt = tensor[:, [0, 1, 2, 0, 1, 0], [0, 1, 2, 1, 2, 2]]
    expected = np.sqrt(2 / 3 * np.sum(tensor * tensor, axis=(1, 2)))
    for value in [tensor, voigt, expected]:
        np.testing.assert_allclose(EVALUATION.equivalent_plastic_strain({"strain": value}), expected)


def test_head_scarp_uses_crest_reference_and_reports_threshold_sensitivity():
    x = np.arange(0.05, 20, 0.1)
    y = np.where(x < 6.4, 4.8, 2.0)
    measurements = EVALUATION.head_scarp(np.column_stack([x, y]), 4.8, 0.2)
    np.testing.assert_allclose([m["retrogression_m"] for m in measurements], 13.6)

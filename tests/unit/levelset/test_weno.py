"""Deterministic accuracy contracts for the NumPy ENO/WENO operators."""

import numpy as np
import pytest

from src.levelset.WENO import (
    essentially_non_oscillatory,
    weighted_essentially_non_oscillatory,
)


pytestmark = [
    pytest.mark.unit,
    pytest.mark.required,
    pytest.mark.lsm,
    pytest.mark.cpu,
]


def _periodic(values, width):
    return np.pad(values, width, mode="wrap")


@pytest.mark.parametrize(
    "operator",
    [
        weighted_essentially_non_oscillatory,
        essentially_non_oscillatory,
    ],
)
def test_eno_operators_validate_order(operator):
    with pytest.raises(ValueError, match="at least 1"):
        operator(0, np.ones(8), 1.0, _periodic)


@pytest.mark.parametrize(
    "operator",
    [
        weighted_essentially_non_oscillatory,
        essentially_non_oscillatory,
    ],
)
@pytest.mark.parametrize("order", [1, 2, 3])
def test_eno_derivative_of_constant_is_zero(operator, order):
    left, right = operator(
        order, np.full(32, 3.25), 0.125, _periodic
    )
    np.testing.assert_array_equal(left, 0.0)
    np.testing.assert_array_equal(right, 0.0)


@pytest.mark.parametrize(
    ("operator", "order", "minimum_rate"),
    [
        (essentially_non_oscillatory, 2, 3.5),
        (essentially_non_oscillatory, 3, 7.0),
        (weighted_essentially_non_oscillatory, 2, 3.5),
        (weighted_essentially_non_oscillatory, 3, 20.0),
    ],
    ids=("eno2", "eno3", "weno3", "weno5"),
)
def test_periodic_sine_derivative_converges(
    operator, order, minimum_rate
):
    errors = []
    for count in (64, 128):
        spacing = 2.0 * np.pi / count
        coordinates = spacing * np.arange(count)
        values = np.sin(coordinates)
        left, right = operator(
            order, values, spacing, _periodic
        )
        error = max(
            np.max(np.abs(left - np.cos(coordinates))),
            np.max(np.abs(right - np.cos(coordinates))),
        )
        errors.append(error)

    assert errors[0] / errors[1] > minimum_rate


def test_weno5_step_data_remains_finite():
    values = np.zeros(128)
    values[32:96] = 1.0
    left, right = weighted_essentially_non_oscillatory(
        3, values, 1.0, _periodic
    )

    assert left.shape == values.shape
    assert right.shape == values.shape
    assert np.isfinite(left).all()
    assert np.isfinite(right).all()

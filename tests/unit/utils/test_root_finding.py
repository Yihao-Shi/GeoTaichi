"""Deterministic contracts for the scalar root solvers used by SDF and IPC."""

import math

import numpy as np
import pytest

from src.utils.Root import newton


pytestmark = [pytest.mark.cpu]


def _quadratic(value):
    return value * value - 2.0


def _quadratic_derivative(value):
    return 2.0 * value


def _quadratic_second_derivative(_value):
    return 2.0


@pytest.mark.parametrize("method", ("newton", "halley", "secant"))
def test_scalar_root_methods_match_analytic_square_root(method):
    kwargs = {}
    if method in ("newton", "halley"):
        kwargs["fprime"] = _quadratic_derivative
    if method == "halley":
        kwargs["fprime2"] = _quadratic_second_derivative
    if method == "secant":
        kwargs["x1"] = 2.0

    root, result = newton(
        _quadratic,
        x0=1.0,
        tol=1.0e-13,
        full_output=True,
        **kwargs,
    )

    assert root == pytest.approx(math.sqrt(2.0), rel=0.0, abs=1.0e-12)
    assert result.converged
    assert result.flag == "converged"
    assert result.iterations > 0
    assert result.function_calls >= result.iterations


def test_vectorized_newton_solves_each_component_without_mutating_guess():
    guess = np.asarray([1.0, 2.0, 4.0], dtype=np.float64)
    original = guess.copy()

    result = newton(
        lambda value: value * value - np.asarray([2.0, 3.0, 5.0]),
        guess,
        fprime=lambda value: 2.0 * value,
        tol=1.0e-13,
    )

    np.testing.assert_array_equal(guess, original)
    np.testing.assert_allclose(
        result,
        np.sqrt(np.asarray([2.0, 3.0, 5.0])),
        rtol=0.0,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tol": 0.0}, "tol too small"),
        ({"maxiter": 0}, "maxiter must be greater than 0"),
        ({"x1": 1.0}, "x1 and x0 must be different"),
    ],
)
def test_scalar_root_rejects_invalid_iteration_controls(kwargs, message):
    with pytest.raises(ValueError, match=message):
        newton(_quadratic, x0=1.0, **kwargs)

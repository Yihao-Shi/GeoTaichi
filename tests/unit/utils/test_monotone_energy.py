import math

import pytest

from src.utils.SolverRuntime import accepts_monotone_energy


@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
def test_roundoff_acceptance_is_energy_scale_invariant(scale):
    assert accepts_monotone_energy(scale, scale * (1 + 2e-14), 1.0, -1e-15 * scale)
    assert not accepts_monotone_energy(scale, scale * (1 + 2e-14), 1.0, -1e-8 * scale)
    assert not accepts_monotone_energy(scale, scale * (1 + 2e-10), 1.0, -1e-15 * scale)
    assert accepts_monotone_energy(scale, scale * 0.9, 1.0, -scale)


@pytest.mark.parametrize(
    "initial,trial,step,slope",
    [
        (1.0, math.inf, 1.0, -1.0),
        (math.nan, 1.0, 1.0, -1.0),
        (1.0, 1.0, 0.0, -1.0),
        (1.0, 1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0, math.nan),
        (0.0, 1e-30, 1.0, -1e-30),
    ],
)
def test_roundoff_does_not_bypass_feasibility_descent_or_invent_an_energy_floor(initial, trial, step, slope):
    assert not accepts_monotone_energy(initial, trial, step, slope)

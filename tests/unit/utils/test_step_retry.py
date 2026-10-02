import pytest

from src.utils.StepRetry import StepRetryPolicy, is_recoverable_nonlinear_failure, nonlinear_failure_kind


def test_step_retry_policy_is_bounded_and_honors_minimum():
    policy = StepRetryPolicy(
        enabled=True,
        maximum_retries=3,
        reduction=0.25,
        minimum_timestep=0.1,
    )

    assert policy.next_timestep(1.0, 0) == pytest.approx(0.25)
    assert policy.next_timestep(0.25, 1) == pytest.approx(0.1)
    assert policy.next_timestep(0.1, 2) is None
    assert policy.next_timestep(1.0, 3) is None
    assert StepRetryPolicy(enabled=False).next_timestep(1.0, 0) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"maximum_retries": True},
        {"enabled": "yes"},
        {"maximum_retries": "two"},
        {"maximum_retries": -1},
        {"reduction": 0.0},
        {"reduction": 1.0},
        {"minimum_timestep": -1.0},
    ],
)
def test_step_retry_policy_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        StepRetryPolicy(**kwargs)


def test_nonlinear_failure_categories_are_stable():
    assert nonlinear_failure_kind(RuntimeError("PCG linear solve failed")) == "linear_solver_nonconvergence"
    assert nonlinear_failure_kind(RuntimeError("Armijo line search failed")) == "line_search_failure"
    assert nonlinear_failure_kind(RuntimeError("direction is not descending")) == "non_descent_direction"
    assert nonlinear_failure_kind(RuntimeError("Newton did not converge")) == "newton_nonconvergence"
    ccd_failure = RuntimeError("CCD produced no strictly feasible step")
    assert nonlinear_failure_kind(ccd_failure) == "line_search_failure"
    assert is_recoverable_nonlinear_failure(ccd_failure)

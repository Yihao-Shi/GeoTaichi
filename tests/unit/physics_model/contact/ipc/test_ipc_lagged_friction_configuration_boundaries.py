from importlib import import_module
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ipc, pytest.mark.contact]

from src.igampm.ContactManager import (
    ContactManager,
    _validate_iga_friction_configuration,
)
from src.mpdem.engines.SoftAffineIPCOperator import (
    _validate_soft_affine_lagged_friction_configuration,
)


@pytest.mark.parametrize(
    "attribute",
    (
        "affine_friction_iterations",
        "friction_iterations",
        "friction_fixed_point_iterations",
    ),
)
def test_soft_affine_accepts_lagged_iteration_aliases(attribute):
    sims = SimpleNamespace(affine_friction_mode="lag")
    setattr(sims, attribute, 1)

    assert _validate_soft_affine_lagged_friction_configuration(sims) == ("lagged", 1)


@pytest.mark.parametrize(
    ("iterations", "expected"),
    ((-7, -1), (-1, -1), (0, -1), (2, 2), (7, 7)),
)
def test_soft_affine_accepts_outer_fixed_point_iterations(iterations, expected):
    sims = SimpleNamespace(
        affine_friction_mode="lagged",
        affine_friction_iterations=iterations,
    )
    assert _validate_soft_affine_lagged_friction_configuration(sims) == (
        "lagged",
        expected,
    )


@pytest.mark.parametrize("iterations", (1.5, "many"))
def test_soft_affine_rejects_invalid_outer_fixed_point_iterations(iterations):
    sims = SimpleNamespace(
        affine_friction_mode="lagged",
        affine_friction_iterations=iterations,
    )
    with pytest.raises((RuntimeError, ValueError), match="must be an integer"):
        _validate_soft_affine_lagged_friction_configuration(sims)


@pytest.mark.parametrize("mode", ("fully_implicit", "fullyimplicit", "fully-implicit"))
def test_soft_affine_accepts_fully_implicit_mode_aliases(mode):
    sims = SimpleNamespace(
        affine_friction_mode=mode,
        affine_friction_iterations=1,
    )
    assert _validate_soft_affine_lagged_friction_configuration(sims) == (
        "fully_implicit",
        0,
    )


@pytest.mark.parametrize("mode", ("implicit", "fully", "newton_friction"))
def test_soft_affine_rejects_unknown_friction_mode(mode):
    with pytest.raises(ValueError, match="must be 'lagged' or 'fully_implicit'"):
        _validate_soft_affine_lagged_friction_configuration(
            SimpleNamespace(
                affine_friction_mode=mode,
                affine_friction_iterations=1,
            )
        )


@pytest.mark.parametrize(
    "key",
    ("friction_iterations", "friction_fixed_point_iterations"),
)
@pytest.mark.parametrize(
    ("iterations", "expected"),
    ((1, 1), (2, 2), (-1, -1), (0, -1), (-9, -1)),
)
def test_iga_accepts_iteration_aliases(monkeypatch, key, iterations, expected):
    contact_module = import_module("src.igampm.ContactManager")
    monkeypatch.setattr(contact_module, "Barrier", lambda **kwargs: ("barrier", kwargs))
    monkeypatch.setattr(contact_module, "Friction", lambda **kwargs: ("friction", kwargs))

    manager = ContactManager(
        contact_model="IPC",
        friction_mode="lag",
        **{key: iterations},
    )

    assert manager.friction_mode == "lagged"
    assert manager.friction_iterations == expected


@pytest.mark.parametrize("iterations", (1.5, "many"))
def test_iga_rejects_invalid_outer_fixed_point_iterations(iterations):
    with pytest.raises(ValueError, match="must be an integer"):
        ContactManager(contact_model="IPC", friction_iterations=iterations)


@pytest.mark.parametrize("mode", ("fully_implicit", "fullyimplicit", "fully-implicit"))
def test_iga_accepts_fully_implicit_friction_mode(monkeypatch, mode):
    contact_module = import_module("src.igampm.ContactManager")
    monkeypatch.setattr(contact_module, "Barrier", lambda **kwargs: ("barrier", kwargs))
    monkeypatch.setattr(contact_module, "Friction", lambda **kwargs: ("friction", kwargs))
    manager = ContactManager(
        contact_model="IPC",
        friction_mode=mode,
        dynamic_friction=0.31,
        static_friction=0.57,
        viscous_friction=0.02,
        stribeck_velocity=0.12,
        epsv=0.01,
        friction_profile="quadratic",
    )
    assert manager.friction_mode == "fully_implicit"
    forwarded = manager.friction[1]
    assert forwarded["dynamic_friction"] == pytest.approx(0.31)
    assert forwarded["static_friction"] == pytest.approx(0.57)
    assert forwarded["viscous_friction"] == pytest.approx(0.02)
    assert forwarded["stribeck_velocity"] == pytest.approx(0.12)


def test_iga_lagged_iteration_default_is_one():
    assert _validate_iga_friction_configuration({}) == ("lagged", 1)


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), -1.0])
def test_iga_rejects_nonfinite_or_negative_friction_tolerance(tolerance):
    with pytest.raises(ValueError, match="friction_tolerance"):
        ContactManager(contact_model="IPC", friction_tolerance=tolerance)


@pytest.mark.parametrize("maximum", [0, -1, 2.5, float("nan"), float("inf")])
def test_iga_rejects_invalid_friction_iteration_cap(maximum):
    with pytest.raises(ValueError, match="friction_max_iterations"):
        ContactManager(contact_model="IPC", friction_max_iterations=maximum)

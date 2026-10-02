"""Unit checks for immutable IGA-MPM configuration."""

from types import SimpleNamespace

import pytest

import src.igampm.config as config
from src.igampm.ContactManager import ContactManager
from src.igampm.engines import Engine
from src.igampm.Simulation import Simulation
from src.igampm.mainIGAMPM import IGAMPM


@pytest.mark.parametrize(
    "alias", ("DEM", "LINEAR-DEM")
)
def test_igampm_rejects_ambiguous_dem_contact_aliases(alias):
    with pytest.raises(ValueError, match="IPC, Linear, or HertzMindlin"):
        config.normalize_contact_model(alias)


@pytest.mark.parametrize(
    ("alias", "expected"),
    (
        ("linear", "Linear"),
        ("LinearModel", "Linear"),
        ("hertz", "HertzMindlin"),
        ("Hertz-Mindlin", "HertzMindlin"),
    ),
)
def test_igampm_accepts_explicit_dem_law_aliases(alias, expected):
    assert config.normalize_contact_model(alias) == expected


def test_direct_contact_manager_reconfiguration_is_rejected_after_freeze():
    manager = object.__new__(ContactManager)
    manager._configuration_frozen = False
    manager.freeze_configuration()

    with pytest.raises(RuntimeError, match="cannot be changed after build"):
        manager.choose_contact_model(contact_model="IPC")


def test_direct_simulation_reconfiguration_is_rejected_without_mutation():
    sims = Simulation()
    original = dict(vars(sims))
    sims.freeze_configuration()

    with pytest.raises(RuntimeError, match="cannot be changed after build"):
        sims.set_configuration(
            dimension=3 if sims.dimension == 2 else 2,
            contact_model="DEM",
            activate_friction=True,
        )

    expected = dict(original)
    expected["_configuration_frozen"] = True
    assert vars(sims) == expected


def test_wrapper_rejects_same_mode_contact_parameter_replacement_after_build():
    coupling = object.__new__(IGAMPM)
    coupling.engine = SimpleNamespace(implicit_step_in_progress=False)

    with pytest.raises(RuntimeError, match="all contact parameters are frozen"):
        coupling.choose_contact_model(
            contact_model="IPC", friction_mode="fully_implicit", mu=0.7
        )


def test_wrapper_rejects_general_configuration_replacement_after_build():
    coupling = object.__new__(IGAMPM)
    coupling.engine = object()
    coupling.sims = SimpleNamespace(dimension=2)

    with pytest.raises(RuntimeError, match="configuration cannot be changed"):
        coupling.set_configuration(dimension=None, contact_model="IPC")


def test_wrapper_freezes_both_configuration_owners_after_build():
    frozen = []
    coupling = object.__new__(IGAMPM)
    coupling.contactor = SimpleNamespace(
        freeze_configuration=lambda: frozen.append("contact")
    )
    coupling.sims = SimpleNamespace(
        freeze_configuration=lambda: frozen.append("simulation")
    )

    coupling._freeze_built_configuration()

    assert frozen == ["contact", "simulation"]


def test_conservative_newton_rejects_fully_implicit_friction():
    engine = object.__new__(Engine)
    engine.activate_fric = True
    engine.friction_mode = "fully_implicit"

    with pytest.raises(
        RuntimeError, match="solve_fully_implicit_friction_newton"
    ):
        engine.solve_monolithic_newton(include_friction=True)

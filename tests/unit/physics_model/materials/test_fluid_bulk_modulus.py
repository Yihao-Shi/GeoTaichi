"""Only formulations with fluid compressibility consume a bulk modulus."""

import numpy as np
import pytest

from src.physics_model.consititutive_model.infinitesimal_strain.LinearElastic import LinearElasticModel
from src.physics_model.consititutive_model.strain_rate.Bingham import BinghamModel
from src.physics_model.consititutive_model.strain_rate.Newtonian import NewtonianModel

pytestmark = [pytest.mark.materials, pytest.mark.cpu]


def coupled_material():
    return {"YoungModulus": 1.0e6, "Porosity": 0.4, "Permeability": 1.0e-4}


@pytest.mark.parametrize("material_type", ["TwoPhaseSingleLayer", "TwoPhaseDoubleLayer"])
@pytest.mark.parametrize("legacy_bulk", [None, 2.2e9])
def test_incompressible_two_phase_material_needs_no_fluid_bulk(material_type, legacy_bulk):
    model = LinearElasticModel(material_type=material_type, solver_type="SemiImplicit")
    material = coupled_material()
    if legacy_bulk is not None:
        material["FluidBulkModulus"] = legacy_bulk
    model.initialize_coupling()
    model.model_initialize(material)
    assert model.fluid_bulk is None
    assert model.bulk == pytest.approx(1.0e6 / (3 * (1 - 2 * 0.3)))


@pytest.mark.parametrize("solver_type", ["Explicit", "SemiImplicit_u_p"])
def test_compressible_two_phase_material_requires_fluid_bulk(solver_type):
    model = LinearElasticModel(material_type="TwoPhaseSingleLayer", solver_type=solver_type)
    with pytest.raises(KeyError, match="FluidBulkModulus"):
        model.model_initialize(coupled_material())
    model.model_initialize(dict(coupled_material(), FluidBulkModulus=2.2e9))
    assert model.fluid_bulk == 2.2e9


@pytest.mark.parametrize("solver_type", ["Explicit", "SemiImplicit_u_p"])
@pytest.mark.parametrize("bulk", [0, -1, np.nan, np.inf])
def test_compressible_two_phase_material_rejects_invalid_bulk(solver_type, bulk):
    model = LinearElasticModel(material_type="TwoPhaseSingleLayer", solver_type=solver_type)
    with pytest.raises(ValueError, match="FluidBulkModulus must be finite and positive"):
        model.model_initialize(dict(coupled_material(), FluidBulkModulus=bulk))


@pytest.mark.parametrize("model_class", [NewtonianModel, BinghamModel])
@pytest.mark.parametrize("legacy_bulk", [None, 2.2e9])
def test_incompressible_fluid_has_no_bulk_or_acoustic_bound(model_class, legacy_bulk, capsys):
    model = model_class(solver_type="Implicit")
    material = {"Viscosity": 1.0e-3, "YieldStress": 1.0, "CriticalStrainRate": 0.01}
    if legacy_bulk is not None:
        material["Modulus"] = legacy_bulk
    model.model_initialize(material)
    assert model.modulus is None
    assert model.max_sound_speed == 0.0
    assert model.get_state_vars() == {}
    model.print_message(1)
    output = capsys.readouterr().out
    assert "pressure projection" in output
    assert "Bulk Modulus" not in output


@pytest.mark.parametrize("model_class", [NewtonianModel, BinghamModel])
@pytest.mark.parametrize("bulk", [0, -1, np.nan, np.inf])
def test_explicit_fluid_rejects_invalid_bulk(model_class, bulk):
    model = model_class(solver_type="Explicit")
    with pytest.raises(ValueError, match="Modulus must be finite and positive"):
        model.model_initialize({"Modulus": bulk})

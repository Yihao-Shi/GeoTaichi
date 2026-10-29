"""Explicit absolute thresholds cannot inherit a looser relative setting."""

import pytest


@pytest.mark.parametrize(
    "owner",
    [
        "FEM",
        "IGA",
        "DirectMPM",
        "FEDEM",
        "FEMPM",
        "IGAMPM",
        "FEM/configuration",
        "IGA/configuration",
        "FEM/element",
        "DirectMPM/contact",
        "IGAMPM/contact",
    ],
)
def test_public_absolute_override_replaces_relative_but_explicit_pair_is_preserved(taichi_runtime, owner):
    from src.dem.mainDEM import DEM
    from src.fem.mainFEM import FEM
    from src.iga.mainIGA import IGA
    from src.mpm.mainMPM import MPM
    from src.fedem.mainFEDEM import FEDEM
    from src.fempm.mainFEMPM import FEMPM
    from src.igampm.mainIGAMPM import IGAMPM

    owner, _, entry = owner.partition("/")
    absolute, relative = "linear_solver_tolerance", "linear_solver_relative_tolerance"
    if owner == "FEM":
        model = FEM(log=False)
        settings = lambda: model.solver_kwargs
        configure = lambda values: model.set_solver(log=False, **values)
    elif owner == "IGA":
        model = IGA(log=False)
        settings = lambda: model.solver_kwargs
        configure = lambda values: model.set_solver(log=False, **values)
    elif owner == "DirectMPM":
        model = MPM(log=False)
        model.set_configuration(mpm_backend="Direct", solver_type="Implicit", log=False)
        settings = lambda: model.direct_solver
        configure = lambda values: model.set_solver(values, log=False)
    elif owner == "FEDEM":
        model = FEDEM(DEM(log=False), FEM(log=False), log=False)
        settings = lambda: model.solver_parameters
        configure = lambda values: model.set_solver(Timestep=1e-3, SimulationTime=0.0, log=False, **values)
    elif owner == "FEMPM":
        model = FEMPM(FEM(log=False), MPM(log=False), log=False)
        settings = lambda: model.solver_parameters
        configure = lambda values: model.set_solver(Timestep=1e-3, SimulationTime=0.0, log=False, **values)
    else:
        model = IGAMPM(log=False)
        settings = lambda: model.contact_kwargs
        configure = lambda values: model.set_solver(coupling=values, log=False)
        absolute, relative = "monolithic_" + absolute, "monolithic_" + relative

    if entry == "configuration":
        configure = lambda values: model.set_configuration(log=False, **values)
    elif entry == "element":
        configure = lambda values: model.add_element(**values)
    elif entry == "contact":
        configure = (
            (lambda values: model.add_contact("IPC", **values))
            if owner == "DirectMPM"
            else (lambda values: model.choose_contact_model(**values))
        )

    configure({relative: 0.1})
    configure({absolute: 1e-10})
    assert settings()[absolute] == 1e-10
    assert settings()[relative] == 0.0
    configure({absolute: 2e-10, relative: 2e-8})
    configure({})
    assert settings()[absolute] == 2e-10
    assert settings()[relative] == 2e-8
    configure({relative: 3e-8})
    assert settings()[absolute] == 2e-10
    assert settings()[relative] == 3e-8


def test_native_mpm_absolute_override_disables_only_implicit_relative_default(taichi_runtime):
    from src.mpm.mainMPM import MPM

    model = MPM(log=False)
    model.set_configuration(domain=[1.0, 1.0, 1.0], solver_type="Implicit", log=False)
    model.set_implicit_solver_parameters(linear_solver_relative_tolerance=0.1)
    model.set_implicit_solver_parameters(residual_tolerance=1e-10)
    assert model.sims.residual_tolerance == 1e-10
    assert model.sims.linear_solver_relative_tolerance == 0.0
    model.set_implicit_solver_parameters(residual_tolerance=2e-10, linear_solver_relative_tolerance=2e-8)
    assert model.sims.residual_tolerance == 2e-10
    assert model.sims.linear_solver_relative_tolerance == 2e-8


@pytest.mark.parametrize("material", ["StVK", "ClothARAP"])
@pytest.mark.parametrize(
    "options, expected",
    [
        ({}, 1e-12),
        ({"linear_solver_tolerance": 1e-8}, 0.0),
        ({"linear_solver_tolerance": 1e-8, "linear_solver_relative_tolerance": 1e-9}, 1e-9),
    ],
)
def test_fem_constructor_and_matrix_receive_selected_policy(taichi_runtime, material, options, expected):
    from src.fem.mainFEM import FEM

    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Implicit")
    fem.add_mesh(geometry="rectangle", size=(1.0, 0.5), divisions=(1, 1))
    if material == "ClothARAP":
        fem.add_material(material, stretch_stiffness=1e3, thickness=0.1, bending_stiffness=1e2)
    else:
        fem.add_material(material, young_modulus=1e3, poisson_ratio=0.3, thickness=0.1)
    fem.set_solver(step=0, **options)
    engine = fem.build()
    assembler = engine.cloth_assembler if material == "ClothARAP" else engine.classical_assembler
    assert engine.linear_solver_relative_tolerance == expected
    assert assembler.linear_solver_relative_tolerance == expected

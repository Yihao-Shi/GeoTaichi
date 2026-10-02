import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from src.fem import FEM


pytestmark = [
    pytest.mark.unit,
    pytest.mark.fem,
    pytest.mark.materials,
    pytest.mark.cpu,
    pytest.mark.serial,
    pytest.mark.isolated_dimension(3),
]


@pytest.fixture(autouse=True)
def taichi_cpu_runtime():
    ti.reset()
    ti.init(
        arch=ti.cpu,
        default_fp=ti.f64,
        cpu_max_num_threads=1,
        offline_cache=False,
    )
    yield
    ti.reset()


def _hex8_solver(model, **parameters):
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Explicit")
    fem.add_mesh(
        fem.create_mesh(
            "box",
            size=(1.0, 1.0, 1.0),
            divisions=(1, 1, 1),
            element_type="HEX8",
        )
    )
    material = fem.add_material(
        model,
        density=1000.0,
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        **parameters,
    )
    fem.set_solver(dt=1.0e-5, steps=1)
    return fem, material, fem.build()


@pytest.mark.parametrize(
    ("model", "parameters"),
    [
        ("ElasticPerfectlyPlastic", {"yield_stress": 1.0e3}),
        (
            "MohrCoulomb",
            {"cohesion": 1.0e3, "friction": 30.0, "dilation": 0.0},
        ),
        (
            "DruckerPrager",
            {"cohesion": 1.0e3, "friction": 30.0, "dilation": 0.0},
        ),
    ],
)
def test_hex8_shared_elastoplastic_models_advance_eight_gauss_points(
    model, parameters
):
    _, material, engine = _hex8_solver(model, **parameters)

    engine.substep()

    assembler = engine.classical_assembler
    assert material.is_fem_elastoplastic
    assert assembler.integration_point_count == 8
    assert np.allclose(
        assembler.deformation_gradient.to_numpy(), np.eye(3)[None, None]
    )
    assert engine.history[-1]["minimum_jacobian"] == pytest.approx(1.0)


def test_hex8_elastoplastic_rejects_implicit_and_ipc():
    fem, _, _ = _hex8_solver(
        "ElasticPerfectlyPlastic", yield_stress=1.0e3
    )
    fem.set_configuration(dimension=3, solver_type="Implicit")
    with pytest.raises(ValueError, match="Explicit"):
        fem.build()

    fem.set_configuration(dimension=3, solver_type="Explicit")
    fem.add_contact("IPC", dhat=0.1)
    with pytest.raises(ValueError, match="does not support IPC"):
        fem.build()


def test_hex8_elastoplastic_uses_rest_to_current_deformation_increment():
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Explicit")
    mesh = fem.create_mesh(
        "box",
        size=(1.0, 1.0, 1.0),
        divisions=(1, 1, 1),
        element_type="HEX8",
    )
    mesh.points[:, 0] *= 1.01
    fem.add_mesh(mesh)
    fem.add_material(
        "ElasticPerfectlyPlastic",
        density=1000.0,
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        yield_stress=1.0e3,
    )
    fem.set_solver(dt=1.0e-5, steps=1)
    engine = fem.build()

    preprocessed_deformation = (
        engine.classical_assembler.deformation_gradient.to_numpy()
    )
    preprocessed_stress = engine.classical_assembler.cauchy_stress.to_numpy()
    assert np.allclose(preprocessed_deformation[:, :, 0, 0], 1.01)
    assert np.linalg.norm(preprocessed_stress) > 0.0

    engine.substep()

    deformation = engine.classical_assembler.deformation_gradient.to_numpy()
    stress = engine.classical_assembler.cauchy_stress.to_numpy()
    assert np.allclose(deformation[:, :, 0, 0], 1.01)
    assert np.linalg.norm(stress) > 0.0


def test_state_dependent_hex8_requires_explicit_finite_timestep():
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Explicit")
    fem.add_mesh(
        fem.create_mesh(
            "box",
            size=(1.0, 1.0, 1.0),
            divisions=(1, 1, 1),
            element_type="HEX8",
        )
    )
    fem.add_material(
        "ModifiedCamClay",
        density=1000.0,
        poisson_ratio=0.3,
        stress_ratio=1.2,
        lambda_=0.2,
        kappa=0.04,
        pc=2.0e5,
        void_ratio_ref=0.8,
        initial_stress=(-1.0e5, -1.0e5, -1.0e5, 0, 0, 0),
    )
    fem.set_solver(dt="auto", steps=1)

    with pytest.raises(ValueError, match="specify a finite positive dt"):
        fem.build()


@pytest.mark.parametrize(
    ("model", "parameters"),
    [
        (
            "StateDependentMohrCoulomb",
            {
                "initial_stress": (-1.0e5, -1.0e5, -1.0e5, 0, 0, 0),
            },
        ),
        (
            "ModifiedCamClay",
            {
                "stress_ratio": 1.2,
                "lambda": 0.2,
                "kappa": 0.04,
                "pc": 2.0e5,
                "void_ratio_ref": 0.8,
                "initial_stress": (-1.0e5, -1.0e5, -1.0e5, 0, 0, 0),
            },
        ),
        (
            "GranularMaterial",
            {
                "static_friction": 30.0,
                "average_diameter": 1.0e-3,
                "inertial_number": 0.3,
                "initial_stress": (-1.0e5, -1.0e5, -1.0e5, 0, 0, 0),
            },
        ),
        (
            "SanisandMS",
            {
                "G0": 125.0,
                "Mc": 1.25,
                "c": 0.7,
                "lambda_c": 0.02,
                "e0": 0.8,
                "ksi": 0.7,
                "m": 0.05,
                "h0": 7.0,
                "ch": 0.9,
                "nb": 1.1,
                "A0": 0.5,
                "nd": 1.0,
                "zeta": 0.0,
                "mu0": 0.0,
                "beta": 1.0,
                "e_init": 0.7,
                "emax": 1.0,
                "emin": 0.5,
                "initial_stress": (-1.0e5, -1.0e5, -1.0e5, 0, 0, 0),
            },
        ),
        (
            "NorSand",
            {
                "G0": 1.0e5,
                "kappa": 0.04,
                "lambda": 0.2,
                "M": 1.2,
                "N": 0.3,
                "beta": 1.0,
                "vc0": 1.9,
                "v0": 1.7,
                "h": 100.0,
                "initial_stress": (-1.0e5, -1.0e5, -1.0e5, 0, 0, 0),
            },
        ),
    ],
)
def test_hex8_advanced_elastoplastic_models_initialize_and_advance(
    model, parameters
):
    _, material, engine = _hex8_solver(model, **parameters)

    engine.substep()

    stress = engine.classical_assembler.cauchy_stress.to_numpy()
    assert material.is_fem_elastoplastic
    assert stress.shape == (8, 6)
    assert np.all(np.isfinite(stress))

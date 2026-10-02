import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from src.fem import DirichletBoundary, FEM

pytestmark = [pytest.mark.integration, pytest.mark.fem, pytest.mark.contact, pytest.mark.cpu, pytest.mark.serial]


@pytest.fixture(autouse=True)
def taichi_cpu_runtime():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1, offline_cache=False)
    yield
    ti.reset()


def contact_sheet(model, height, **contact_parameters):
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Implicit")
    mesh = fem.add_mesh(geometry="rectangle", size=(1.0, 1.0), divisions=(1, 1))
    mesh.points[:, 2] = height
    fem.add_material("ClothARAP", stretch_stiffness=1.0e3, density=1.0, thickness=0.01)
    fem.add_boundary_condition(dirichlet=DirichletBoundary().add("all", "xy", 0.0))
    fem.add_contact(
        model,
        self_contact=False,
        planes=[((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))],
        dhat=0.05,
        dmin=0.01,
        **contact_parameters,
    )
    return fem, mesh


def test_ipc_plane_runs_through_newton_ccd_and_line_search():
    fem, _ = contact_sheet(
        "BarrierIPC",
        0.04,
        kappa=10.0,
        friction_coefficient=0.3,
        friction_iterations=-1,
        friction_tolerance=1.0e-7,
    )
    fem.set_solver(
        quasi_static=False,
        dt=0.01,
        step=1,
        gravity=(0.0, 0.0, -1.0),
        max_iterations=30,
        residual_tolerance=1.0e-7,
        line_search=True,
        assemble_type="HashTriplet",
        linear_solver="PCG",
        project_pd=True,
    )
    engine = fem.build()

    def reject_stale_host_contact(*_args, **_kwargs):
        raise AssertionError("FEM output must not rebuild device contact through the host adapter")

    engine.contact_assembler.assemble_output = reject_stale_host_contact
    result = fem.run(verbose=False)

    assert result.converged
    assert np.min(result.positions[:, 2]) > 0.01
    assert result.history[0]["contact"]["model"] == "IPC"
    assert result.history[0]["friction_converged"]
    assert result.history[0]["friction_residual"] <= 1.0e-7
    assert all(0.0 <= item["line_search_step"] <= 1.0 for item in result.history[0]["iterations"])


def test_nonbarrier_al_recovers_a_small_initial_penetration():
    fem, _ = contact_sheet(
        "SemiIPC",
        0.005,
        penalty=1.0e4,
        max_penalty=1.0e8,
        penalty_growth=4.0,
        penalty_update_interval=2,
        constraint_tolerance=1.0e-5,
    )
    fem.set_solver(
        quasi_static=False,
        dt=0.01,
        step=1,
        max_iterations=40,
        residual_tolerance=1.0e-7,
        line_search=True,
    )
    result = fem.run(verbose=False)

    contact = result.history[0]["contact"]
    assert result.converged
    assert np.min(result.positions[:, 2]) >= 0.01 - 1.0e-5
    assert contact["model"] == "AugmentedLagrangian"
    assert contact["constraint_violation"] <= 1.0e-4

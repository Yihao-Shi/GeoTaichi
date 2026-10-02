import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from src.fem import FEM

pytestmark = [pytest.mark.integration, pytest.mark.fem, pytest.mark.cpu, pytest.mark.serial]


@pytest.fixture(autouse=True)
def taichi_cpu_runtime():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1, offline_cache=False)
    yield
    ti.reset()


def test_differentiable_cloth_records_and_replays():
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Implicit")
    mesh = fem.add_mesh(geometry="rectangle", size=(1.0, 1.0), divisions=(1, 1))
    fem.add_material(
        "ClothARAP",
        stretch_stiffness=1.0e3,
        compression_stiffness=1.0e3,
        density=1.0,
        thickness=0.01,
        bending_stiffness=0.0,
        bending_model="None",
    )
    fem.add_boundary_condition({"type": "Dirichlet", "nodes": [0, 1, 2, 3], "components": "all", "value": 0.0})
    fem.set_solver(
        dt=1.0e-2,
        step=2,
        gravity=(0.0, 0.0, 0.0),
        max_iterations=8,
        residual_tolerance=1.0e-9,
        linear_solver="PCG",
        project_pd=True,
    )
    tape = fem.differentiable(steps=2)
    for _ in range(2):
        position_field = tape.step()
    assert position_field is tape.solver.state.position
    gradient = tape.backward(np.ones((mesh.number_of_nodes, 3)))
    assert gradient["stretch_stiffness"] == 0.0
    assert np.all(np.isfinite(gradient["initial_position"]))


def test_differentiable_cloth_honors_explicit_host_linear_solver():
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Implicit")
    mesh = fem.add_mesh(geometry="rectangle", size=(1.0, 1.0), divisions=(1, 1))
    mesh.points[:, 2] += 0.01
    mesh.rest_shape[:, 2] += 0.01
    fem.add_material(
        "ClothARAP",
        stretch_stiffness=1.0e3,
        compression_stiffness=1.0e3,
        density=1.0,
        thickness=0.01,
        bending_stiffness=0.0,
        bending_model="None",
    )
    fem.add_boundary_condition(
        {"type": "Dirichlet", "nodes": mesh.node_sets["ymax"], "components": "all", "value": 0.0}
    )
    fem.add_contact(
        "IPC",
        self_contact=False,
        planes=[((0, 0, 0), (0, 0, 1))],
        dhat=0.02,
        kappa=1.0e3,
        friction_coefficient=0.0,
    )
    fem.set_solver(dt=1.0e-2, step=1, linear_solver="Scipy")

    tape = fem.differentiable(steps=1)
    assert tape.assembler.linear_solver == "Scipy"

    def reject_host_contact(*_args, **_kwargs):
        raise AssertionError("explicit SciPy solve must not move contact work to the host")

    contact = tape.solver.contact_assembler
    contact.prepare_iteration = reject_host_contact
    contact.assemble_output = reject_host_contact
    contact.maximum_admissible_step = reject_host_contact
    contact.begin_step = reject_host_contact
    contact.accept_update = reject_host_contact
    tape.step()
    gradient = tape.backward(np.ones((mesh.number_of_nodes, 3)))
    assert np.all(np.isfinite(gradient["initial_position"]))


def test_differentiable_cloth_dihedral_bending_vjp_matches_finite_difference():
    def evaluate(bending_stiffness, differentiate=False):
        fem = FEM(log=False)
        fem.set_configuration(dimension=3, solver_type="Implicit")
        mesh = fem.add_mesh(geometry="rectangle", size=(1.0, 1.0), divisions=(1, 1))
        mesh.points[-1, 2] = 0.1
        fem.add_material(
            "ClothARAP",
            stretch_stiffness=1.0e3,
            compression_stiffness=1.0e3,
            density=1.0,
            thickness=0.1,
            bending_stiffness=bending_stiffness,
            bending_poisson_ratio=0.3,
            bending_model="Dihedral",
        )
        fem.add_boundary_condition(
            {"type": "Dirichlet", "nodes": mesh.node_sets["ymax"], "components": "all", "value": 0.0}
        )
        fem.set_solver(
            dt=2.0e-3,
            step=2,
            gravity=(0.0, 0.0, -9.8),
            max_iterations=40,
            residual_tolerance=1.0e-10,
            correction_velocity_tolerance=1.0e-9,
            linear_solver="PCG",
            assemble_type="HashTriplet",
            project_pd=True,
            project_bending_pd=True,
        )
        tape = fem.differentiable(steps=2)
        for _ in range(2):
            tape.step()
        positions = tape.solver.positions
        difference = positions - mesh.rest_shape
        loss = 0.5 * np.sum(difference * difference)
        if differentiate:
            return loss, tape.backward(difference)["bending_stiffness"]
        return loss

    stiffness = 1.0e4
    epsilon = 1.0
    loss, adjoint = evaluate(stiffness, differentiate=True)
    finite_difference = (evaluate(stiffness + epsilon) - evaluate(stiffness - epsilon)) / (2.0 * epsilon)
    assert np.isfinite(loss)
    assert adjoint == pytest.approx(finite_difference, rel=2.0e-3, abs=1.0e-10)

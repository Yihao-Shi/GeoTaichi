import inspect

import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from src.fem import DirichletBoundary, FEM, NeumannBoundary
from src.fem.engines import ArmijoLineSearch

pytestmark = [pytest.mark.integration, pytest.mark.fem, pytest.mark.cpu]


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


def test_implicit_hex_volume_converges_with_newton_line_search():
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Implicit")
    mesh = fem.add_mesh(
        {
            "Geometry": "Box",
            "Size": (1.0, 0.2, 0.2),
            "Divisions": (2, 1, 1),
            "ElementType": "HEX8",
        }
    )
    fem.add_material("StVK", young_modulus=1.0e5, poisson_ratio=0.3, density=1000.0)
    fixed = DirichletBoundary().add("xmin", "all", 0.0)
    force = NeumannBoundary().add_nodal_force("xmax", (100.0, 0.0, 0.0), total=True)
    fem.add_boundary_condition(dirichlet=fixed, neumann=force)
    fem.set_solver(
        quasi_static=True,
        dt=1.0,
        step=1,
        residual_tolerance=1.0e-9,
        max_iterations=20,
        line_search=True,
    )

    result = fem.run(verbose=False)

    assert result.converged
    assert np.all(result.displacement[mesh.node_sets["xmax"], 0] > 0.0)
    assert len(result.history[0]["iterations"]) >= 2
    np.testing.assert_allclose(np.sum(result.reaction[:, 0]), -100.0, rtol=2.0e-8)


def test_implicit_tet4_reads_state_directly_during_newton_assembly():
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Implicit")
    mesh = fem.add_mesh(
        {
            "Geometry": "Box",
            "Size": (1.0, 0.2, 0.2),
            "Divisions": (1, 1, 1),
            "ElementType": "TET4",
        }
    )
    fem.add_material(
        "StVK",
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        density=1000.0,
    )
    fem.add_boundary_condition(
        dirichlet=DirichletBoundary().add("xmin", "all", 0.0),
        neumann=NeumannBoundary().add_nodal_force("xmax", (10.0, 0.0, 0.0), total=True),
    )
    fem.set_solver(
        quasi_static=True,
        dt=1.0,
        step=1,
        residual_tolerance=1.0e-8,
        max_iterations=20,
        line_search=True,
    )

    result = fem.run(verbose=False)

    assert result.converged
    assert np.all(result.displacement[mesh.node_sets["xmax"], 0] > 0.0)
    np.testing.assert_allclose(np.sum(result.reaction[:, 0]), -10.0, rtol=2.0e-7)


def test_implicit_tri3_membrane_and_explicit_gravity_paths():
    implicit = FEM(log=False)
    implicit.set_configuration(dimension=3, solver_type="Implicit")
    mesh = implicit.add_mesh(geometry="rectangle", size=(1.0, 0.5), divisions=(4, 2))
    implicit.add_material("StVK", young_modulus=1.0e4, poisson_ratio=0.3, density=1.0, thickness=0.01)
    fixed = DirichletBoundary().add("xmin", "xy", 0.0).add("all", "z", 0.0)
    force = NeumannBoundary().add_nodal_force("xmax", (1.0, 0.0, 0.0), total=True)
    implicit.add_boundary_condition(dirichlet=fixed, neumann=force)
    implicit.set_solver(quasi_static=True, dt=1.0, step=1, residual_tolerance=1.0e-8)
    implicit_result = implicit.run(verbose=False)

    explicit = FEM(log=False)
    explicit.set_configuration(dimension=3, solver_type="Explicit")
    explicit.add_mesh(mesh)
    explicit.add_material("StVK", young_modulus=1.0e4, poisson_ratio=0.3, density=1.0, thickness=0.01)
    explicit.add_boundary_condition(dirichlet=DirichletBoundary().add("xmin", "all", 0.0))
    explicit.set_solver(dt=1.0e-4, step=2, gravity=(0.0, 0.0, -9.8))
    explicit_result = explicit.run(verbose=False)

    assert implicit_result.converged
    assert np.mean(implicit_result.displacement[mesh.node_sets["xmax"], 0]) > 0.0
    assert explicit_result.time == pytest.approx(2.0e-4)
    assert np.min(explicit_result.displacement[:, 2]) < 0.0


def test_static_explicit_boundary_data_remain_device_resident(monkeypatch):
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Explicit")
    mesh = fem.add_mesh(
        {
            "Geometry": "Box",
            "Size": (0.2, 0.2, 0.2),
            "Divisions": (1, 1, 1),
            "ElementType": "TET4",
        }
    )
    fem.add_material("StVK", young_modulus=1.0e4, poisson_ratio=0.3, density=1000.0)
    fixed = DirichletBoundary().add(np.arange(mesh.number_of_nodes), "all", 0.0)
    loads = NeumannBoundary()
    fem.add_boundary_condition(dirichlet=fixed, neumann=loads)
    fem.set_solver(dt=1.0e-5, step=1)
    fem.build()
    engine = fem.engine

    assert engine.is_fully_constrained_static

    def unexpected_dirichlet_evaluation(*_args, **_kwargs):
        raise AssertionError("static Dirichlet values were reevaluated")

    monkeypatch.setattr(engine.dirichlet, "values", unexpected_dirichlet_evaluation)
    assert engine.set_boundary_data_step(1.0) == engine.degree_of_freedom

    calls = 0
    original_force = engine.neumann.force

    def count_force(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_force(*args, **kwargs)

    monkeypatch.setattr(engine.neumann, "force", count_force)
    engine.update_external_force_step(0.0)
    engine.update_external_force_step(1.0)
    assert calls == 0
    np.testing.assert_allclose(engine.state.boundary_force.to_numpy(), 0.0)


def test_transient_explicit_boundaries_use_preallocated_device_frames(monkeypatch):
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Explicit")
    mesh = fem.add_mesh(
        {
            "Geometry": "Box",
            "Size": (0.2, 0.2, 0.2),
            "Divisions": (1, 1, 1),
            "ElementType": "TET4",
        }
    )
    fem.add_material("StVK", young_modulus=1.0e4, poisson_ratio=0.3, density=1000.0)

    def displacement(time, coordinates):
        values = np.zeros_like(coordinates)
        values[:, 2] = time
        return values

    def nodal_force(time, coordinates):
        values = np.zeros_like(coordinates)
        values[:, 0] = time
        return values

    fem.add_boundary_condition(
        dirichlet=DirichletBoundary().add("all", "all", displacement),
        neumann=NeumannBoundary().add_nodal_force("all", nodal_force),
    )
    dt = 1.0e-5
    fem.set_solver(dt=dt, step=2)
    fem.build()
    engine = fem.engine

    assert engine.boundary_data.dirichlet_dynamic
    assert engine.boundary_data.force_dynamic
    assert engine.boundary_data.dirichlet_frame_count == 3
    assert engine.boundary_data.force_frame_count == 3

    # Coupled drivers may lower the stable timestep after FEM construction.
    # Rebuilding before the first physical step must still allocate the whole
    # replacement timeline once, rather than fall back to per-step uploads.
    dt *= 0.5
    engine.dt = dt
    engine.reconfigure_device_boundary_timeline(dt, total_step=4)
    assert engine.boundary_data.dirichlet_frame_count == 5
    assert engine.boundary_data.force_frame_count == 5

    def unexpected_host_boundary_evaluation(*_args, **_kwargs):
        raise AssertionError("substep evaluated a Python/NumPy boundary")

    monkeypatch.setattr(engine.dirichlet, "values", unexpected_host_boundary_evaluation)
    monkeypatch.setattr(engine.neumann, "force", unexpected_host_boundary_evaluation)
    engine.substep()

    positions = engine.state.position.to_numpy()
    np.testing.assert_allclose(positions[:, 2], engine.reference_positions[:, 2] + dt)
    np.testing.assert_allclose(engine.state.boundary_force.to_numpy()[:, 0], dt)


def test_dynamic_implicit_dirichlet_keeps_piecewise_linear_velocity():
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Implicit")
    fem.add_mesh(
        {
            "Geometry": "Box",
            "Size": (0.2, 0.2, 0.2),
            "Divisions": (1, 1, 1),
            "ElementType": "TET4",
        }
    )
    fem.add_material("StVK", young_modulus=1.0e4, poisson_ratio=0.3, density=1.0)

    def displacement(time, coordinates):
        values = np.zeros_like(coordinates)
        values[:, 0] = time
        return values

    fem.add_boundary_condition(dirichlet=DirichletBoundary().add("all", "all", displacement))
    fem.set_solver(dt=0.1, step=2)

    result = fem.run(verbose=False)

    np.testing.assert_allclose(result.velocity[:, 0], 1.0)
    np.testing.assert_allclose(result.velocity[:, 1:], 0.0)
    np.testing.assert_allclose(result.acceleration, 0.0)


def test_armijo_policy_is_scalar_only_for_device_line_search():
    search = ArmijoLineSearch(reduction=0.5, sufficient_decrease=1.0e-4)
    assert search.reduction == pytest.approx(0.5)
    assert search.sufficient_decrease == pytest.approx(1.0e-4)
    assert search.accepts(4.681234598761334e-3, 4.681234598776585e-3, 1.0, -7.501284660578497e-16, 1.0e-12)
    assert not search.accepts(1.0, 1.0 + 2.0e-12, 1.0, -1.0e-15, 1.0e-12)
    assert not search.accepts(1.0, float("inf"), 1.0, -1.0, float("inf"))
    assert not hasattr(search, "search")


def test_implicit_final_reaction_reuses_last_newton_force():
    from src.fem.engines.ImplicitFEM import ImplicitFEM

    source = inspect.getsource(ImplicitFEM._substep_once)
    finalization = source.split("self.state.finalize_newmark", 1)[1]

    assert "final_internal = internal_force" in finalization
    assert "_assemble_internal_device" not in finalization
    assert "prepared_converged_device" in inspect.getsource(ImplicitFEM._contact_converged_device)

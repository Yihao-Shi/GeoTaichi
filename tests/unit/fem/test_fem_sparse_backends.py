import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from src.fem import DirichletBoundary, FEM, NeumannBoundary
from src.fem.engines.SparseMatrix import FEMSparseMatrix

pytestmark = [
    pytest.mark.unit,
    pytest.mark.fem,
    pytest.mark.assembly,
    pytest.mark.linear_solver,
    pytest.mark.cpu,
    pytest.mark.serial,
]


@pytest.fixture(autouse=True)
def taichi_cpu_runtime():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1, offline_cache=False)
    yield
    ti.reset()


@pytest.mark.parametrize(
    "assemble_type,linear_solver",
    [
        ("COO", "Scipy"),
        ("HashTriplet", "Scipy"),
        ("COO", "PCG"),
        ("HashTriplet", "BiCGSTAB"),
    ],
)
def test_fem_sparse_assembly_and_solver_choices_match_dense_reference(assemble_type, linear_solver):
    rng = np.random.default_rng(19)
    factor = rng.normal(size=(6, 6))
    dense = factor.T @ factor + 2.0 * np.eye(6)
    rows, columns = np.indices(dense.shape)
    matrix = FEMSparseMatrix(
        6,
        assemble_type=assemble_type,
        linear_solver=linear_solver,
        linear_solver_tolerance=1.0e-11,
        linear_solver_relative_tolerance=1.0e-9,
        linear_solver_max_iters=300,
    )
    matrix.add_triplets(rows.reshape(-1), columns.reshape(-1), dense.reshape(-1))

    np.testing.assert_allclose(matrix.toarray(), dense, rtol=2.0e-15, atol=2.0e-15)

    constrained = np.asarray((0, 4), dtype=np.int32)
    rhs = rng.normal(size=6)
    solution = matrix.solve(rhs, constrained_dofs=constrained)
    assert matrix.linear_solver_relative_tolerance == pytest.approx(1.0e-9)
    free = np.setdiff1d(np.arange(6), constrained)
    expected = np.zeros(6)
    expected[free] = np.linalg.solve(dense[np.ix_(free, free)], rhs[free])

    np.testing.assert_allclose(solution, expected, rtol=2.0e-9, atol=2.0e-10)


def test_fem_hash_matrix_reuses_device_solver_fields():
    matrix = FEMSparseMatrix(6, assemble_type="HashTriplet", linear_solver="PCG")
    matrix.add_triplets(
        np.asarray((0, 1, 2), dtype=np.int32),
        np.asarray((3, 4, 5), dtype=np.int32),
        np.ones(3),
    )

    first, _ = matrix._materialize()
    matrix.add_triplets((3,), (0,), (1.0,))
    second, _ = matrix._materialize()

    assert second is first


def test_fem_coo_matrix_reuses_device_solver_fields():
    matrix = FEMSparseMatrix(6, assemble_type="COO", linear_solver="PCG")
    matrix.add_triplets(
        np.arange(6, dtype=np.int32),
        np.arange(6, dtype=np.int32),
        np.ones(6),
    )

    first, _ = matrix._materialize()
    matrix.add_triplets(
        np.arange(10, dtype=np.int32) % 6,
        np.arange(10, dtype=np.int32) % 6,
        np.zeros(10),
    )
    second, _ = matrix._materialize()

    assert second is first

    matrix.additional_diagonal.fill(3.0)
    constrained, _ = matrix._materialize(constrained_dofs=(0,))
    dense = constrained._to_scipy().toarray()
    assert dense[0, 0] == pytest.approx(1.0)
    np.testing.assert_allclose(dense[0, 1:], 0.0)
    np.testing.assert_allclose(dense[1:, 0], 0.0)
    np.testing.assert_allclose(np.diag(dense)[1:], 4.0)


def test_fem_coo_exact_symmetric_solve_falls_back_to_device_bicgstab():
    matrix = FEMSparseMatrix(3, assemble_type="COO", linear_solver="PCG")
    matrix.add_triplets(
        np.arange(3, dtype=np.int32),
        np.arange(3, dtype=np.int32),
        np.asarray([-2.0, 1.0, 1.0]),
    )
    residual = ti.Vector.field(3, ti.f64, shape=1)
    direction = ti.Vector.field(3, ti.f64, shape=1)
    constrained = ti.field(ti.i32, shape=3)
    residual[0] = [1.0, 0.0, 0.0]

    result = matrix.solve_device(
        residual,
        direction,
        constrained,
        fallback_to_bicgstab=True,
    )

    assert result["fallback_from"] == "PCG"
    np.testing.assert_allclose(direction.to_numpy()[0], [0.5, 0.0, 0.0], rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize(
    "assemble_type,linear_solver,project_pd",
    [("COO", "PCG", True), ("HashTriplet", "BiCGSTAB", False)],
)
def test_implicit_fem_uses_selected_taichi_matrix_and_solver(assemble_type, linear_solver, project_pd):
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Implicit", backend="taichi")
    mesh = fem.add_mesh(geometry="rectangle", size=(1.0, 0.5), divisions=(2, 1))
    fem.add_material(
        "StVK",
        young_modulus=1.0e4,
        poisson_ratio=0.3,
        density=1.0,
        thickness=0.01,
    )
    fem.add_boundary_condition(
        dirichlet=DirichletBoundary().add("xmin", "xy", 0.0).add("all", "z", 0.0),
        neumann=NeumannBoundary().add_nodal_force("xmax", (1.0, 0.0, 0.0), total=True),
    )
    fem.set_solver(
        quasi_static=True,
        dt=1.0,
        step=1,
        residual_tolerance=1.0e-7,
        assemble_type=assemble_type,
        linear_solver=linear_solver,
        linear_solver_tolerance=1.0e-10,
        linear_solver_relative_tolerance=1.0e-9,
        linear_solver_max_iters=500,
        project_pd=project_pd,
    )

    engine = fem.build()
    first_stiffness = engine.classical_assembler.assemble_device(need_stiffness=True)[1]
    second_stiffness = engine.classical_assembler.assemble_device(need_stiffness=True)[1]
    result = fem.run(verbose=False)

    assert second_stiffness is first_stiffness
    assert engine.assemble_type == ("COO" if assemble_type == "COO" else "Hash")
    assert engine.linear_solver == linear_solver
    assert engine.classical_assembler.linear_solver_relative_tolerance == pytest.approx(1.0e-9)
    assert result.converged
    assert np.mean(result.displacement[mesh.node_sets["xmax"], 0]) > 0.0


def test_pcg_requires_projected_element_tangents():
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Implicit", backend="taichi")
    fem.add_mesh(geometry="rectangle")
    fem.add_material("StVK", young_modulus=1.0e4, poisson_ratio=0.3, thickness=0.01)
    fem.set_solver(linear_solver="PCG", project_pd=False, step=0)

    with pytest.raises(ValueError, match="project_pd=True"):
        fem.build()


def test_pcg_enables_projection_by_default():
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Implicit", backend="taichi")
    fem.add_mesh(geometry="rectangle")
    fem.add_material("StVK", young_modulus=1.0e4, poisson_ratio=0.3, thickness=0.01)
    fem.set_solver(linear_solver="PCG", step=0)

    engine = fem.build()

    assert engine.project_pd
    assert engine.classical_assembler.project_pd


def test_pcg_rejects_unprojected_ipc_contact_tangents():
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Implicit", backend="taichi")
    fem.add_mesh(geometry="rectangle")
    fem.add_material("StVK", young_modulus=1.0e4, poisson_ratio=0.3, thickness=0.01)
    fem.add_contact(
        "IPC",
        self_contact=False,
        planes=[((0.0, 0.0, -0.01), (0.0, 0.0, 1.0))],
        project_pd=False,
    )
    fem.set_solver(linear_solver="PCG", step=0)

    with pytest.raises(ValueError, match="contact project_pd=True"):
        fem.build()

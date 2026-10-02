"""Small deterministic checks for the IGA solver backends."""

from types import SimpleNamespace

import numpy as np
import pytest


pytestmark = [pytest.mark.cpu, pytest.mark.serial]


def build_primitives():
    from src.iga import Primitives, Rectangle

    body = Rectangle()
    body.set_parameters(size=[1.0, 0.2])
    body.generate_knot_u(degree=2, num_ctrlpts=3)
    body.generate_knot_v(degree=2, num_ctrlpts=3)
    body.generate_ctrlpts()
    body.generate_weights()
    body.activate_boundary()
    body.gather_boundary_ctrlpts()

    primitives = Primitives()
    primitives.append(body, "beam")
    primitives.finialize()
    return primitives


def test_material_ccd_traverses_each_patch_element_count():
    from src.iga.engines import ImplicitIGA

    engine = object.__new__(ImplicitIGA)
    engine.patch = SimpleNamespace(
        primitive=SimpleNamespace(num_primitives=2),
        prefix_num_knot=[(0, 0), (1, 1)],
        prefix_num_element=[(0, 0), (1, 1)],
        prefix_total_num_ctrlpts=[0, 4],
        total_num_element=[0, 3, 5],
        num_knot=[(0, 0), (4, 4), (5, 5)],
        num_element=[(0, 0), (1, 3), (1, 5)],
        num_ctrlpts=[(0, 0), (2, 2), (2, 3)],
    )
    visited = []

    def material_ccd(total_num_element, *_args):
        visited.append(int(total_num_element))
        return 1.0

    engine.material_ccd = material_ccd

    assert engine.ccd() == pytest.approx(1.0)
    assert visited == [3, 5]


def test_iga_backend_builds_implicit_solver(taichi_runtime, tmp_path):
    from src.iga import IGA, DirichletBoundary, get_dimension
    from src.iga.engines import ImplicitIGA

    iga = IGA(log=False)
    iga.set_configuration(dimension=2, solver_type="Implicit")
    primitives = build_primitives()
    index = np.where(primitives.body["beam"]["primitive"].control_points[:, 0] == 0.0)[0]
    dirichlet = DirichletBoundary()
    dirichlet.append([list(2 * index), list(2 * index + 1)], [0.0] * (2 * len(index)))

    iga.add_primitives(primitives)
    iga.add_boundary_condition(dirichlet=dirichlet)
    iga.add_element(degree=[2, 2])
    iga.add_material(young_modulus=1.0e5, poisson_ratio=0.3, density=1000.0)
    iga.set_solver(
        step=0,
        interval=1,
        residual=1.0e-4,
        path=str(tmp_path / "implicit"),
    )
    engine = iga.build()

    assert get_dimension() == 2
    assert isinstance(engine, ImplicitIGA)
    assert engine.degree_of_freedom == 18


def test_explicit_iga_energy_is_sampled_only_when_enabled(taichi_runtime, tmp_path):
    from src.iga import IGA
    from src.iga.engines import ExplicitIGA

    iga = IGA(log=False)
    iga.set_configuration(dimension=2, solver_type="Explicit")
    iga.add_primitives(build_primitives())
    iga.add_element(degree=[2, 2])
    iga.add_material(
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        density=1000.0,
    )
    iga.set_solver(
        dt=1.0e-5,
        step=1,
        interval=1,
        gravity=[0.0, -9.8],
        damping=0.1,
        track_energy=True,
        path=str(tmp_path / "explicit_energy"),
    )
    engine = iga.build()
    assert isinstance(engine, ExplicitIGA)

    engine.precompute()
    engine.substep(record_history=True)

    record = engine.history[-1]
    for name in (
        "material_energy",
        "external_potential_energy",
        "kinetic_energy",
        "damping_dissipation",
    ):
        assert np.isfinite(record[name])


@pytest.mark.parametrize("use_rest_shape", [False, True])
def test_iga_rest_shape_is_precomputed_without_moving_current_shape(taichi_runtime, tmp_path, use_rest_shape):
    from src.iga import IGA

    iga = IGA(log=False)
    iga.set_configuration(dimension=2, solver_type="Implicit")
    primitives = build_primitives()
    current_shape = primitives.body["beam"]["primitive"].control_points.copy()
    rest_shape = current_shape.copy()
    if use_rest_shape:
        rest_shape[:, 0] *= 0.5
        iga.add_primitives(primitives, rest_shape={"beam": rest_shape})
    else:
        iga.add_primitives(primitives)
    iga.add_element(degree=[2, 2])
    iga.add_material(
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        density=2.0,
    )
    iga.set_solver(step=0, path=str(tmp_path / "rest_shape"))

    engine = iga.build()
    engine.precompute()

    expected = np.diag((2.0, 1.0)) if use_rest_shape else np.eye(2)
    np.testing.assert_allclose(engine.patch.initial_control_points.to_numpy(), current_shape)
    np.testing.assert_allclose(engine.patch.control_points.to_numpy(), current_shape)
    np.testing.assert_allclose(
        engine.patch.rest_control_points.to_numpy(),
        rest_shape if use_rest_shape else current_shape,
    )
    np.testing.assert_allclose(
        engine.initial_deformation_gradients.to_numpy(),
        np.broadcast_to(
            expected,
            engine.initial_deformation_gradients.to_numpy().shape,
        ),
        atol=2.0e-12,
    )
    expected_area = 0.1 if use_rest_shape else 0.2
    assert np.sum(engine.patch.volume.to_numpy()) == pytest.approx(expected_area, rel=2.0e-12)


def test_iga_backend_solves_with_coo_taichi_pcg(taichi_runtime, tmp_path):
    from src.iga import IGA, DirichletBoundary

    iga = IGA(log=False)
    iga.set_configuration(dimension=2, solver_type="Implicit")
    primitives = build_primitives()
    index = np.where(primitives.body["beam"]["primitive"].control_points[:, 0] == 0.0)[0]
    dirichlet = DirichletBoundary()
    dirichlet.append([list(2 * index), list(2 * index + 1)], [0.0] * (2 * len(index)))

    iga.add_primitives(primitives)
    iga.add_boundary_condition(dirichlet=dirichlet)
    iga.add_element(degree=[2, 2])
    iga.add_material(young_modulus=1.0e5, poisson_ratio=0.3, density=1000.0)
    iga.set_solver(
        step=0,
        interval=1,
        residual=1.0e-4,
        assemble_type="COO",
        linear_solver="PCG",
        linear_solver_max_iters=200,
        path=str(tmp_path / "coo_pcg"),
    )
    engine = iga.build()
    engine.precompute()
    engine.grid_disp.fill(0)
    engine.rhs.fill(0)
    engine.incre_resolution.fill(0)
    engine.reset_linear_system()
    engine.assemble_body_matrix()
    fixed_dof = int(2 * index[0])
    engine.grid_disp[fixed_dof] = 0.125
    engine.dirichlet.value[fixed_dof] = 0.375
    engine.apply_dirichlet()
    assert np.isclose(engine.rhs[fixed_dof], 0.25)
    result = engine.solve_system()

    assert engine.assemble_type == "COO"
    assert engine.linear_solver == "PCG"
    assert result.shape[0] == engine.degree_of_freedom
    assert np.all(np.isfinite(result))


def test_iga_hash_bicgstab_stays_in_taichi_solver(taichi_runtime, tmp_path):
    """The Hash production path must not convert its matrix to SciPy."""
    from src.iga import IGA, DirichletBoundary

    iga = IGA(log=False)
    iga.set_configuration(dimension=2, solver_type="Implicit")
    primitives = build_primitives()
    index = np.where(primitives.body["beam"]["primitive"].control_points[:, 0] == 0.0)[0]
    dirichlet = DirichletBoundary()
    dirichlet.append([list(2 * index), list(2 * index + 1)], [0.0] * (2 * len(index)))

    iga.add_primitives(primitives)
    iga.add_boundary_condition(dirichlet=dirichlet)
    iga.add_element(degree=[2, 2])
    iga.add_material(young_modulus=1.0e5, poisson_ratio=0.3, density=1000.0)
    iga.set_solver(
        dt=1.0e-4,
        step=0,
        interval=1,
        residual=1.0e-8,
        gravity=[0.0, -9.8],
        assemble_type="Hash",
        linear_solver="BiCGSTAB",
        linear_solver_tolerance=1.0e-10,
        linear_solver_max_iters=500,
        path=str(tmp_path / "hash_bicgstab"),
    )
    engine = iga.build()
    engine.precompute()
    engine.grid_disp.fill(0)
    engine.rhs.fill(0)
    engine.incre_resolution.fill(0)
    engine.reset_linear_system()
    engine.assemble_body_matrix()
    expected_raw_pairs = engine.stiffness_nnz // (2 * 2) * engine.element.gauss_number
    assert int(engine.hash_matrix.raw_non_diag_count[0]) == expected_raw_pairs
    assert engine.hash_matrix.non_diag.max_pairs_num == expected_raw_pairs
    raw_i = engine.hash_matrix.non_diag.blockI.to_numpy()[:expected_raw_pairs]
    raw_j = engine.hash_matrix.non_diag.blockJ.to_numpy()[:expected_raw_pairs]
    engine.hash_matrix.finalize_taichi_assembly()

    # Reassembly must preserve every raw slot so the GPU reduction can reuse
    # its raw-slot-to-reduced-block map without hashing again.
    engine.rhs.fill(0)
    engine.reset_linear_system()
    engine.assemble_body_matrix()
    np.testing.assert_array_equal(
        engine.hash_matrix.non_diag.blockI.to_numpy()[:expected_raw_pairs],
        raw_i,
    )
    np.testing.assert_array_equal(
        engine.hash_matrix.non_diag.blockJ.to_numpy()[:expected_raw_pairs],
        raw_j,
    )

    fixed_dof = int(2 * index[0])
    engine.grid_disp[fixed_dof] = 0.125
    engine.dirichlet.value[fixed_dof] = 0.375
    engine.apply_dirichlet()
    assert np.isclose(engine.rhs[fixed_dof], 0.25)

    def scipy_fallback_forbidden(*_args, **_kwargs):
        raise AssertionError("Hash+BiCGSTAB must not call BuildTriplet.to_scipy")

    engine.hash_matrix.to_scipy = scipy_fallback_forbidden
    result = engine.solve_system()

    cache_stats = engine.hash_matrix.non_diag.pattern_cache_statistics()
    assert cache_stats["pattern_hits"] >= 1

    assert result.shape[0] == engine.degree_of_freedom
    assert np.all(np.isfinite(result))
    assert np.linalg.norm(result) > 0.0


def test_iga_hash_pcg_uses_exact_structural_symmetry(
    taichi_runtime,
    tmp_path,
):
    """Projected IGA copies its full source into one mirrored PCG triangle."""
    from src.iga import IGA, DirichletBoundary

    iga = IGA(log=False)
    iga.set_configuration(dimension=2, solver_type="Implicit")
    primitives = build_primitives()
    index = np.where(primitives.body["beam"]["primitive"].control_points[:, 0] == 0.0)[0]
    dirichlet = DirichletBoundary()
    dirichlet.append(
        [list(2 * index), list(2 * index + 1)],
        [0.0] * (2 * len(index)),
    )

    iga.add_primitives(primitives)
    iga.add_boundary_condition(dirichlet=dirichlet)
    iga.add_element(degree=[2, 2])
    iga.add_material(
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        density=1000.0,
    )
    iga.set_solver(
        dt=1.0e-4,
        step=0,
        interval=1,
        residual=1.0e-8,
        gravity=[0.0, -9.8],
        assemble_type="Hash",
        linear_solver="PCG",
        project_hessian_to_psd=True,
        linear_solver_tolerance=1.0e-10,
        linear_solver_max_iters=500,
        path=str(tmp_path / "hash_pcg"),
    )
    engine = iga.build()
    engine.precompute()
    engine.grid_disp.fill(0.0)
    engine.rhs.fill(0.0)
    engine.incre_resolution.fill(0.0)
    engine.reset_linear_system()
    engine.assemble_body_matrix()

    # The source remains complete for a possible IGA-MPM fully implicit
    # Jacobian product; only the standalone PCG destination is canonicalized.
    raw_i = engine.hash_matrix.non_diag.blockI.to_numpy()[: int(engine.hash_matrix.raw_non_diag_count[0])]
    raw_j = engine.hash_matrix.non_diag.blockJ.to_numpy()[: int(engine.hash_matrix.raw_non_diag_count[0])]
    assert np.any(raw_i < raw_j)
    assert np.any(raw_i > raw_j)

    fixed_dof = int(2 * index[0])
    engine.grid_disp[fixed_dof] = 0.125
    engine.dirichlet.value[fixed_dof] = 0.375
    engine.apply_dirichlet()
    assert np.isclose(engine.rhs[fixed_dof], 0.25)
    assert engine.pcg_hash_matrix is not None
    assert engine.pcg_hash_matrix.matrix_symmetric
    assert engine.pcg_hash_matrix.full_symmetric_input

    engine.pcg_hash_matrix.finalize_taichi_assembly()
    dense = engine.pcg_hash_matrix.to_scipy(engine.degree_of_freedom // 2).toarray()
    np.testing.assert_array_equal(dense, dense.T)
    assert float(np.linalg.eigvalsh(dense)[0]) > 0.0

    def scipy_fallback_forbidden(*_args, **_kwargs):
        raise AssertionError("Hash+PCG must stay in the Taichi solver")

    engine.hash_matrix.to_scipy = scipy_fallback_forbidden
    engine.pcg_hash_matrix.to_scipy = scipy_fallback_forbidden
    result = engine.solve_system()
    assert result.shape[0] == engine.degree_of_freedom
    assert np.all(np.isfinite(result))
    assert np.linalg.norm(result) > 0.0


def test_iga_backend_builds_and_steps_explicit_solver(
    taichi_runtime,
    tmp_path,
):
    from src.iga import IGA, DirichletBoundary
    from src.iga.engines import ExplicitIGA

    iga = IGA(log=False)
    iga.set_configuration(dimension=2, solver_type="Explicit")
    primitives = build_primitives()
    index = np.where(primitives.body["beam"]["primitive"].control_points[:, 0] == 0.0)[0]
    dirichlet = DirichletBoundary()
    dirichlet.append([list(2 * index), list(2 * index + 1)], [0.0] * (2 * len(index)))

    iga.add_primitives(primitives)
    iga.add_boundary_condition(dirichlet=dirichlet)
    iga.add_element(degree=[2, 2])
    iga.add_material(young_modulus=1.0e5, poisson_ratio=0.3, density=1000.0)
    iga.set_solver(
        dt=1.0e-5,
        step=0,
        interval=1,
        gravity=[0.0, -9.8],
        path=str(tmp_path / "explicit"),
    )
    engine = iga.build()
    engine.precompute()
    engine.substep()

    assert isinstance(engine, ExplicitIGA)
    assert engine.rhs.shape[0] == engine.degree_of_freedom
    assert np.all(np.isfinite(engine.patch.control_points.to_numpy()))

"""Compare cached/fixed IGA assembly with independent scalar COO quadrature."""

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.iga, pytest.mark.assembly, pytest.mark.isolated_dimension(3)]


def test_multi_patch_fixed_assembly_matches_coo(taichi_runtime, tmp_path):
    import taichi as ti
    import src.iga.config as config

    config.set_dimension(3)
    from src.iga import Cube, ImplicitIGA, Primitives
    from src.linear_solver.BuildTriplet import BuildTriplet

    primitives = Primitives()
    for index in range(2):
        body = Cube()
        body.set_parameters(start_point=[2.0 * index, 0.0, 0.0], size=[1.0, 0.5, 0.2])
        body.generate_knot_u(degree=1, num_ctrlpts=3)
        body.generate_knot_v(degree=1, num_ctrlpts=2)
        body.generate_knot_w(degree=1, num_ctrlpts=2)
        body.generate_ctrlpts()
        body.generate_weights()
        primitives.append(body, f"solid{index}")
    primitives.finialize()
    engines = [
        ImplicitIGA(
            primitives=primitives,
            degree=[1, 1, 1],
            young_modulus=1e4,
            poisson_ratio=0.3,
            density=1000.0,
            gravity=[0.0, 0.0, -9.8],
            dt=1e-3,
            step=0,
            assemble_type=backend,
            path=str(tmp_path / backend),
        )
        for backend in ("Hash", "COO")
    ]
    hashed, coo = engines
    coordinates, slots = hashed.fixed_block_coordinates(upper_triangle=True)
    scatter = ti.field(ti.i32, shape=slots.shape)
    scatter.from_numpy(slots)
    fixed = BuildTriplet(
        dim=3,
        max_pairs_num=1,
        max_nonzeros=len(coordinates),
        max_active_nodes=hashed.degree_of_freedom // 3,
        symmetric=False,
        matrix_symmetric=True,
    )
    fixed.install_fixed_pattern(coordinates)
    for engine in engines:
        engine.precompute()
        assert np.sum(engine.patch.volume.to_numpy()) == pytest.approx(0.2, rel=1e-12)
    for stretch in (0.001, -0.002):
        for engine in engines:
            engine.grid_disp.from_numpy(np.linspace(-stretch, stretch, engine.degree_of_freedom))
            engine.rhs.fill(0.0)
            engine.reset_linear_system()
            engine.assemble_body_matrix(project_spd=False)
        hashed.hash_matrix.finalize_taichi_assembly()
        expected = coo.coo_matrix._to_scipy().toarray()
        np.testing.assert_allclose(hashed.hash_matrix.to_scipy().toarray(), expected, rtol=1e-12, atol=1e-8)
        np.testing.assert_allclose(hashed.rhs.to_numpy(), coo.rhs.to_numpy(), rtol=1e-12, atol=1e-8)
        fixed.reset_system()
        source = hashed if stretch > 0 else coo
        source.assemble_body_matrix(project_spd=False, need_force=False, matrix=fixed, fixed_slots=scatter)
        fixed.finalize_taichi_assembly()
        np.testing.assert_allclose(fixed.to_scipy().toarray(), expected, rtol=1e-12, atol=1e-8)
        assert fixed.raw_non_diag_count[0] == 0
        # Reinitializing reference data must invalidate every patch cache.
        hashed.patch.volume.fill(0.0)
        hashed.precompute()
        assert not hashed._reference_ready_offsets

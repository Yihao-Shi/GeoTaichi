import os

import numpy as np
import pytest
import taichi as ti

pytestmark = [pytest.mark.slow, pytest.mark.serial]


def _make_body(volume=0.01):
    from src.mpm.generator.Body import Body

    body = Body()
    points = np.array(
        [
            [0.25, 0.25],
            [0.35, 0.25],
            [0.25, 0.35],
            [0.35, 0.35],
        ],
        dtype=np.float64,
    )
    body.add_particles(points, volume=volume, init_v=[0.0, 0.0], name="block")
    return body


def _configure_direct_mpm(output_path, volume=0.01):
    from src.mpm import MPM

    mpm = MPM(log=False)
    mpm.set_configuration(dimension=2, mpm_backend="Direct", solver_type="Implicit", configuration="ULMPM")
    mpm.add_body(_make_body(volume))
    mpm.add_material(young_modulus=1.0e5, poisson_ratio=0.3, density=1000.0, material="neoHookean")
    mpm.set_solver(
        {
            "domain": [1.0, 1.0],
            "dx": 0.1,
            "dt": 1.0e-3,
            "gravity": [0.0, -9.8],
            "step": 0,
            "interval": 1,
            "residual": 1.0e-4,
            "shape_function": "linear",
            "path": os.fspath(output_path),
        }
    )
    return mpm


def test_direct_backend_builds_implicit_ulmpm_from_main_mpm_flow(tmp_path):
    from src.mpm.engines.direct.ImplicitULMPM import ImplicitULMPM

    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)
    mpm = _configure_direct_mpm(tmp_path / "implicit-ul")
    mpm.sims.set_alpha(0.125)
    mpm.add_engine()

    assert isinstance(mpm.enginer, ImplicitULMPM)
    assert mpm.enginer.n_particles == 4
    assert mpm.enginer.total_background_grid_num > 0
    assert mpm.enginer.coeffPIC == pytest.approx(0.125)


def test_direct_backend_rejects_particles_outside_the_background_grid(tmp_path):
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)
    mpm = _configure_direct_mpm(tmp_path / "domain-guard")
    mpm.add_engine()
    mpm.enginer.particle[0].x = [-0.01, 0.25]

    with pytest.raises(RuntimeError, match="outside its background grid"):
        mpm.enginer.compute_shapefn()


def test_direct_backend_preserves_per_particle_volume(tmp_path):
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)
    volume = np.asarray([0.004, 0.008, 0.012, 0.016], dtype=np.float64)
    mpm = _configure_direct_mpm(tmp_path / "particle-volume", volume=volume)
    mpm.add_engine()

    np.testing.assert_allclose(mpm.enginer.particle.vol0.to_numpy()[:4], volume)
    np.testing.assert_allclose(mpm.enginer.particle.m.to_numpy()[:4], 1000.0 * volume)


def test_direct_backend_keeps_backend_name(tmp_path):
    mpm = _configure_direct_mpm(tmp_path / "backend-name")
    assert mpm.sims.mpm_backend == "Direct"


def test_direct_plane_strain_solver_option_requires_boolean():
    from src.mpm import MPM

    runtime = ti.lang.impl.get_runtime()
    if runtime.prog is None:
        ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)

    mpm = MPM(log=False)
    mpm.set_configuration(
        dimension=2,
        mpm_backend="Direct",
        solver_type="Implicit",
        configuration="ULMPM",
    )
    with pytest.raises(TypeError, match="plane_strain must be a boolean"):
        mpm.set_solver({"plane_strain": "False"}, log=False)
    mpm.set_solver({"plane_strain": True}, log=False)
    assert mpm.direct_solver["plane_strain"] is True


def test_direct_implicit_mpm_hash_solve_stays_in_taichi_fields():
    import src.mpm.config as config

    from src.mpm.engines.direct.ImplicitMPM import ImplicitMPM

    config.set_dimension(2)

    class DeviceHashMatrix:
        def __init__(self):
            self.finalized = False
            self.solve_call = None

        def finalize_taichi_assembly(self):
            self.finalized = True

        def solve_flat_system(self, rhs, solution, **kwargs):
            self.solve_call = (rhs, solution, kwargs)
            return {
                "converged": True,
                "iterations": 7,
                "residual": 2.5e-12,
                "solution_inf_norm": 0.125,
            }

        def to_scipy(self, _active_nodes):
            raise AssertionError("direct MPM must not build a SciPy matrix")

    engine = ImplicitMPM.__new__(ImplicitMPM)
    engine.hash_matrix = DeviceHashMatrix()
    engine.rhs = object()
    engine.incre_resolution = object()
    engine.linear_solver_tolerance = 3.0e-11
    engine.linear_solver_max_iters = 321

    result = engine.solve_hash_system(active_dof=6)

    assert engine.hash_matrix.finalized
    rhs, solution, kwargs = engine.hash_matrix.solve_call
    assert rhs is engine.rhs
    assert solution is engine.incre_resolution
    assert kwargs == {
        "active_nodes": 3,
        "tol": 3.0e-11,
        "maxiter": 321,
        "return_solution": False,
    }
    assert result["backend"] == "taichi_bicgstab"
    assert result["solution_inf_norm"] == 0.125


def test_direct_implicit_mpm_has_no_non_cuda_scipy_fallback(monkeypatch):
    import src.mpm.config as config
    import src.mpm.engines.direct.ImplicitMPM as implicit_mpm_module

    from src.mpm.engines.direct.ImplicitMPM import ImplicitMPM

    config.set_dimension(2)

    class DeviceHashMatrix:
        def __init__(self):
            self.finalized = False
            self.solve_call = None

        def finalize_taichi_assembly(self):
            self.finalized = True

        def solve_flat_system(self, rhs, solution, **kwargs):
            self.solve_call = (rhs, solution, kwargs)
            return {
                "converged": True,
                "iterations": 3,
                "residual": 1.0e-13,
                "solution_inf_norm": 0.5,
            }

        def to_scipy(self, _active_nodes):
            raise AssertionError("CPU/Metal direct MPM must stay in Taichi")

    engine = ImplicitMPM.__new__(ImplicitMPM)
    engine.hash_matrix = DeviceHashMatrix()
    engine.rhs = object()
    engine.incre_resolution = object()
    engine.linear_solver_tolerance = 1.0e-10
    engine.linear_solver_max_iters = 100
    monkeypatch.setattr(
        implicit_mpm_module,
        "solve_csr_system",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("SciPy fallback was invoked")),
    )

    result = engine.solve_hash_system(active_dof=6)

    assert engine.hash_matrix.finalized
    assert engine.hash_matrix.solve_call[0] is engine.rhs
    assert engine.hash_matrix.solve_call[1] is engine.incre_resolution
    assert result["backend"] == "taichi_bicgstab"
    assert result["solution_inf_norm"] == 0.5


def _make_direct_engine(engine_type, output_path, *, scale=1.0):
    body = _make_body()
    # The high-level MPM facade normally finalizes this metadata before it
    # constructs a direct engine; this test instantiates the engine itself.
    body.body_counter = max(int(body.body_counter), len(body.bodies))
    return engine_type(
        domain=[1.0, 1.0],
        dx=0.1,
        dt=1.0e-3,
        bodies=body,
        newmark=[1.0, 0.5, 1.0],
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        density=1000.0,
        gravity=[0.0, -9.8],
        residual=1.0e-8,
        interval=1,
        step=1,
        scale=scale,
        line_search=False,
        shape_function="linear",
        visualize=False,
        device_reduction=True,
        path=os.fspath(output_path),
    )


def _prepare_direct_stiffness(engine, *, updated_lagrangian):
    engine.init_F0()
    engine.grid_disp.fill(0.0)
    if updated_lagrangian:
        engine.mass_vec.fill(0.0)
        engine.grid_reset()
        engine.compute_shapefn()
        engine.mass_vel_acc_p2g()
    # TL computes its fixed shape functions and nodal mass while determining
    # the exact degree of freedom in its constructor.
    engine.find_active_node()
    engine.prefix_sum_executor.run(engine.node2dof)
    return int(engine.set_active_dof())


def _dense_fixed_stencil_reference(engine, active_nodes, raw_count):
    dim = 2
    dense = np.zeros((dim * active_nodes, dim * active_nodes), dtype=np.float64)
    diagonal = engine.hash_matrix.diag.to_numpy()[:active_nodes]
    for block, values in enumerate(diagonal):
        begin = dim * block
        dense[begin : begin + dim, begin : begin + dim] += values.reshape(dim, dim)

    block_i = engine.hash_matrix.non_diag.blockI.to_numpy()[:raw_count]
    block_j = engine.hash_matrix.non_diag.blockJ.to_numpy()[:raw_count]
    block_h = engine.hash_matrix.non_diag.blockH.to_numpy()[:raw_count]
    valid = (block_i >= 0) & (block_j >= 0)
    for row, col, values in zip(block_i[valid], block_j[valid], block_h[valid]):
        r0 = dim * int(row)
        c0 = dim * int(col)
        dense[r0 : r0 + dim, c0 : c0 + dim] += values.reshape(dim, dim)
    return dense, block_i, block_j, int(np.count_nonzero(valid))


def _assert_fixed_particle_local_pair_layout(engine, block_i, block_j):
    offsets = engine.offset.to_numpy()
    local_nodes = engine.LnID.to_numpy()
    node_to_dof = engine.node2dof.to_numpy()
    width = engine.stiffness_stencil_width
    stride = engine.stiffness_stencil_stride
    for particle in range(int(engine.particleNum[0])):
        for local_j in range(width):
            for local_k in range(width):
                raw = particle * stride + local_j * width + local_k
                if local_j >= offsets[particle] or local_k >= offsets[particle]:
                    assert block_i[raw] == -1 and block_j[raw] == -1
                    continue
                row = int(node_to_dof[local_nodes[particle, local_j]]) - 1
                col = int(node_to_dof[local_nodes[particle, local_k]]) - 1
                if row < 0 or col < 0 or row == col:
                    assert block_i[raw] == -1 and block_j[raw] == -1
                else:
                    assert block_i[raw] == row
                    assert block_j[raw] == col


def test_direct_ul_tl_fixed_stiffness_stencil_is_exact_and_reuses_mapping(
    tmp_path,
):
    """Fixed particle/local-pair slots preserve the old triplet matrix."""
    import src.mpm.config as config

    from src.mpm.engines.direct.ImplicitTLMPM import ImplicitTLMPM
    from src.mpm.engines.direct.ImplicitULMPM import ImplicitULMPM

    config.set_dimension(2)
    runtime = ti.lang.impl.get_runtime()
    if runtime.prog is None:
        ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)

    for engine_type, updated_lagrangian in (
        (ImplicitULMPM, True),
        (ImplicitTLMPM, False),
    ):
        engine = _make_direct_engine(engine_type, tmp_path / engine_type.__name__)
        active_dof = _prepare_direct_stiffness(engine, updated_lagrangian=updated_lagrangian)
        active_nodes = active_dof // config.DIM
        if engine_type is ImplicitTLMPM:
            # TL allocates the exact fixed active-node count. Equality is a
            # valid capacity state and must not be rejected.
            assert active_dof == engine.degree_of_freedom
        expected_raw_count = int(engine.particleNum[0]) * engine.stiffness_stencil_stride

        engine.hash_matrix.reset_system()
        engine.assemble_stiffness_matrix_hash(active_dof, engine.grid_disp)
        raw_count = int(engine.hash_matrix.raw_non_diag_count[0])
        assert raw_count == expected_raw_count
        reference, first_i, first_j, valid_count = _dense_fixed_stencil_reference(engine, active_nodes, raw_count)
        _assert_fixed_particle_local_pair_layout(engine, first_i, first_j)
        engine.hash_matrix.finalize_taichi_assembly()
        first_matrix = engine.hash_matrix.to_scipy(active_nodes).toarray()
        np.testing.assert_allclose(first_matrix, reference, rtol=1.0e-12, atol=1.0e-12)
        first_stats = engine.hash_matrix.non_diag.pattern_cache_statistics()
        assert first_stats["last_mapping_misses"] == valid_count

        engine.hash_matrix.reset_system()
        engine.assemble_stiffness_matrix_hash(active_dof, engine.grid_disp)
        second_count = int(engine.hash_matrix.raw_non_diag_count[0])
        second_i = engine.hash_matrix.non_diag.blockI.to_numpy()[:second_count]
        second_j = engine.hash_matrix.non_diag.blockJ.to_numpy()[:second_count]
        np.testing.assert_array_equal(second_i, first_i)
        np.testing.assert_array_equal(second_j, first_j)
        engine.hash_matrix.finalize_taichi_assembly()
        second_matrix = engine.hash_matrix.to_scipy(active_nodes).toarray()
        np.testing.assert_allclose(second_matrix, reference, rtol=1.0e-12, atol=1.0e-12)
        assert engine.hash_matrix.non_diag.pattern_cache_statistics()["last_mapping_misses"] == 0


def test_direct_ul_active_dof_capacity_is_checked_before_compact_map_write(
    tmp_path,
):
    """UL accepts an exact allocation and safely rejects one node too few."""
    import src.mpm.config as config

    from src.mpm.engines.direct.ImplicitULMPM import ImplicitULMPM

    config.set_dimension(2)
    runtime = ti.lang.impl.get_runtime()
    if runtime.prog is None:
        ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)

    probe = _make_direct_engine(ImplicitULMPM, tmp_path / "capacity-probe", scale=1.0)
    required_dof = _prepare_direct_stiffness(probe, updated_lagrangian=True)
    assert required_dof > config.DIM
    denominator = config.DIM * probe.total_background_grid_num

    # The small fractional offset keeps int(scale * denominator) exactly at
    # the requested integer even if the quotient rounds slightly downward.
    exact_scale = (required_dof + 0.25) / denominator
    exact = _make_direct_engine(ImplicitULMPM, tmp_path / "capacity-exact", scale=exact_scale)
    assert exact.degree_of_freedom == required_dof
    assert _prepare_direct_stiffness(exact, updated_lagrangian=True) == exact.degree_of_freedom

    undersized_dof = required_dof - config.DIM
    undersized_scale = (undersized_dof + 0.25) / denominator
    undersized = _make_direct_engine(
        ImplicitULMPM,
        tmp_path / "capacity-undersized",
        scale=undersized_scale,
    )
    assert undersized.degree_of_freedom == undersized_dof
    with pytest.raises(RuntimeError, match="active-DOF capacity exceeded"):
        _prepare_direct_stiffness(undersized, updated_lagrangian=True)


def test_direct_affine_projection_transfers_an_affine_velocity(tmp_path):
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)
    mpm = _configure_direct_mpm(tmp_path / "affine")
    mpm.sims.set_velocity_projection_scheme("Affine")
    mpm.direct_solver["shape_function"] = "bspline"
    mpm.add_engine()
    engine = mpm.enginer
    assert engine.velocity_proj is True
    gradient = np.array([[0.7, 0.2], [-0.1, 0.3]])
    positions = engine.particle.x.to_numpy()[:4]
    intercept = np.array([0.05, -0.02])
    engine.particle.v.from_numpy(positions @ gradient.T + intercept)
    engine.gradv.from_numpy(np.tile(gradient, (4, 1, 1)))
    engine.compute_shapefn()
    engine.grid_reset()
    engine.mass_vel_acc_p2g()
    mass, momentum = engine.grid.m.to_numpy(), engine.grid.v.to_numpy()
    active = mass > engine.val_lim
    grid = engine.body[0]
    n = np.asarray(grid.grid_num)
    ids = np.arange(mass.size)[active] - int(grid.goffset)
    nodes = np.column_stack((ids % n[0], ids // n[0])) * float(grid.grid_size) + np.asarray(grid.xmin)
    np.testing.assert_allclose(
        momentum[active] / mass[active, None], nodes @ gradient.T + intercept, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(mass.sum(), engine.particle.m.to_numpy().sum())
    engine.particle[0].x = [0.0015, 0.25]
    engine.compute_shapefn()
    np.testing.assert_allclose(engine.shape.to_numpy()[0].sum(), 1.0, atol=1e-14)
    np.testing.assert_allclose(engine.dshape.to_numpy()[0].sum(axis=0), 0.0, atol=1e-13)
    engine.particle[0].x = [-0.0015, 0.25]
    with pytest.raises(RuntimeError, match="outside its background grid"):
        engine.compute_shapefn()

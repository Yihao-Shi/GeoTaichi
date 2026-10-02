import os

import numpy as np
import pytest
import taichi as ti

pytestmark = [pytest.mark.slow, pytest.mark.serial]


def _build_soft_soft_contact(
    tmp_path,
    positions=((0.45, 0.50), (0.50, 0.50)),
    dhat=0.08,
    expect_active=True,
    barrier_set=(16, 4),
    friction_set=(16, 4),
    ground_specs=None,
):
    import src.mpm.config as config
    from src.mpm.engines.direct.ImplicitULMPM import ImplicitULMPM
    from src.mpm.generator.Body import Body
    from src.mpm.generator.Ground import Ground
    from src.mpm.soft_particle.IPCMPM import IPCMPM

    config.set_dimension(2)
    bodies = Body()
    bounds = dict(xmin=[0.0, 0.0], xmax=[1.0, 1.0])
    for body_positions in positions:
        body_positions = np.asarray(body_positions, dtype=np.float64)
        if body_positions.ndim == 1:
            body_positions = body_positions.reshape((1, 2))
        bodies.add_particles(
            body_positions,
            volume=1.0e-3,
            boundary_ids=[0] * len(body_positions),
            **bounds,
        )
    mpm = ImplicitULMPM(
        domain=[1.0, 1.0],
        dx=0.1,
        dt=1.0e-3,
        bodies=bodies,
        newmark=[1.0, 0.5, 1.0],
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        density=1000.0,
        gravity=[0.0, 0.0],
        residual=1.0e-8,
        interval=1,
        step=1,
        scale=1.0,
        line_search=False,
        shape_function="linear",
        visualize=False,
        path=str(tmp_path / "mpm"),
    )
    mpm.init_F0()
    mpm.mass_vec.fill(0.0)
    mpm.grid_reset()
    mpm.compute_shapefn()
    mpm.mass_vel_acc_p2g()
    mpm.find_active_node()
    mpm.prefix_sum_executor.run(mpm.node2dof)
    mpm.active_dof = mpm.set_active_dof()
    mpm.compute_nodal_vel_acc()

    ground = Ground()
    if ground_specs is None:
        ground_specs = (([0.0, -10.0], [0.0, 1.0]),)
    for position, normal in ground_specs:
        ground.append(position, normal)
    contact = IPCMPM(
        mpm,
        ground,
        kappa=1.0e4,
        dhat=dhat,
        mu=0.5,
        epsv=1.0e-3,
        activate_friction=True,
        coordination_number=[4, 1],
        friction_set=list(friction_set),
        barrier_set=list(barrier_set),
    )
    contact.initial_surface_temp()
    contact.initialize_hat_x()
    contact.particle_friction_initialize()
    assert int(contact.pfrictionNum[0]) == int(expect_active)
    return config, mpm, contact


def _body_translation_direction(config, mpm):
    direction = np.zeros(mpm.active_dof, dtype=np.float64)
    offsets = mpm.offset.to_numpy()
    node_ids = mpm.LnID.to_numpy()
    node2dof = mpm.node2dof.to_numpy()
    for particle_id, sign in ((0, 0.5), (1, -0.5)):
        for local in range(int(offsets[particle_id])):
            node_id = int(node_ids[particle_id, local])
            active_node = int(node2dof[node_id]) - 1
            assert active_node >= 0
            direction[config.DIM * active_node + 1] = sign
    return direction


def _body_translation_displacement(config, mpm, translations):
    displacement = np.zeros(mpm.degree_of_freedom, dtype=np.float64)
    offsets = mpm.offset.to_numpy()
    node_ids = mpm.LnID.to_numpy()
    node2dof = mpm.node2dof.to_numpy()
    assigned_nodes = {}
    for particle_id, translation in enumerate(translations):
        translation = np.asarray(translation, dtype=np.float64)
        for local in range(int(offsets[particle_id])):
            node_id = int(node_ids[particle_id, local])
            active_node = int(node2dof[node_id]) - 1
            assert active_node >= 0
            if node_id in assigned_nodes:
                np.testing.assert_allclose(assigned_nodes[node_id], translation)
            assigned_nodes[node_id] = translation
            start = config.DIM * active_node
            displacement[start : start + config.DIM] = translation
    return displacement


def _evaluate(config, mpm, contact, displacement):
    full = np.zeros(mpm.degree_of_freedom, dtype=np.float64)
    full[: mpm.active_dof] = displacement
    mpm.grid_disp.from_numpy(full)
    contact.update_particle_pos(mpm.grid_disp)
    contact.friction_hash_matrix.reset_system()
    contact.friction_grad.fill(0.0)
    contact.assemble_particle_friction_matrix()
    matrix = contact.friction_matrix(mpm.active_dof).toarray()
    force = contact.friction_grad.to_numpy()[: mpm.active_dof].copy()
    mpm.energy[None] = 0.0
    contact.get_friction_energy()
    return float(mpm.energy[None]), force, matrix


def _fd_gradient(energy, x, step):
    result = np.zeros_like(x)
    for column in range(x.size):
        perturbation = np.zeros_like(x)
        perturbation[column] = step
        result[column] = (energy(x + perturbation) - energy(x - perturbation)) / (2.0 * step)
    return result


def _fd_jacobian(force, x, step):
    result = np.zeros((x.size, x.size), dtype=np.float64)
    for column in range(x.size):
        perturbation = np.zeros_like(x)
        perturbation[column] = step
        result[:, column] = (force(x + perturbation) - force(x - perturbation)) / (2.0 * step)
    return result


def _relative_error(actual, expected):
    return np.linalg.norm(actual - expected, ord=np.inf) / max(
        np.linalg.norm(actual, ord=np.inf),
        np.linalg.norm(expected, ord=np.inf),
        1.0,
    )


def test_ipc_soft_soft_friction_production_assembly(tmp_path):
    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    ti.init(arch=ti.cpu, default_fp=ti.f64, debug=False, offline_cache=False)
    config, mpm, contact = _build_soft_soft_contact(tmp_path)
    translation = _body_translation_direction(config, mpm)

    for label, amplitude, fd_step in (
        ("dynamic", 2.0e-4, 2.0e-8),
        ("smoothed", 2.0e-7, 2.0e-10),
    ):
        displacement = amplitude * translation
        energy, force, matrix = _evaluate(config, mpm, contact, displacement)
        energy_gradient = _fd_gradient(
            lambda value: _evaluate(config, mpm, contact, value)[0],
            displacement,
            fd_step,
        )
        force_jacobian = _fd_jacobian(
            lambda value: _evaluate(config, mpm, contact, value)[1],
            displacement,
            fd_step,
        )

        assert energy > 0.0, label
        assert _relative_error(-force, energy_gradient) < 2.0e-5, label
        assert _relative_error(force_jacobian, -matrix) < 5.0e-5, label
        assert np.allclose(matrix, matrix.T, rtol=1.0e-10, atol=1.0e-10), label
        eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        assert eigenvalues.min() >= -1.0e-12 * max(eigenvalues.max(), 1.0)
        assert np.allclose(
            force.reshape((-1, config.DIM)).sum(axis=0),
            0.0,
            rtol=1.0e-10,
            atol=1.0e-10,
        ), label
        assert float(np.dot(force, displacement)) < 0.0, label


def test_direct_mpm_surface_body_tasks_are_compact_and_stable(tmp_path):
    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    ti.init(arch=ti.cpu, default_fp=ti.f64, debug=False, offline_cache=False)
    _, _, contact = _build_soft_soft_contact(
        tmp_path,
        positions=((0.47, 0.50), (0.50, 0.50), (0.53, 0.50)),
        expect_active=3,
        barrier_set=(24, 4),
        friction_set=(24, 4),
    )

    expected = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    for _ in range(2):
        contact.update_body_pair_table()
        contact.build_surface_body_contact_table()
        count = int(contact.surface_body_contact_num[0])
        table = contact.surface_body_contact.to_numpy()
        actual = list(zip(table["surfaceID"][:count], table["bodyID"][:count]))
        assert actual == expected
        assert contact.last_particle_search_backend == "LinkedCell"


def test_direct_mpm_particle_full_ccd_covers_nonactive_body_pair(tmp_path):
    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    ti.init(arch=ti.cpu, default_fp=ti.f64, debug=False, offline_cache=False)
    config, mpm, contact = _build_soft_soft_contact(
        tmp_path,
        positions=((0.25, 0.50), (0.75, 0.50)),
        dhat=0.05,
        expect_active=False,
    )
    contact.barrier.dmin[0] = 0.01
    mpm.grid_disp.fill(0.0)
    contact.update_particle_pos(mpm.grid_disp)

    # The activation AABBs are expanded only by dmin + dhat.  They are
    # disjoint at the current iterate, so the ordinary active set is empty.
    contact.update_body_pair_table()
    body_min = contact.body_min.to_numpy()
    body_max = contact.body_max.to_numpy()
    assert body_max[0, 0] < body_min[1, 0]
    assert int(contact.body_pair_num[0]) == 0
    contact.point_point_distance()
    assert int(contact.surface_body_contact_num[0]) == 0
    assert int(contact.pbarrierNum[0]) == 0

    # A 0.35-unit displacement over dt=1e-3 corresponds to a 350-unit/s
    # head-on step.  The complete swept test must find this otherwise-missed
    # body pair and stop before the two points reach dmin.
    sweep = _body_translation_displacement(config, mpm, ((0.35, 0.0), (-0.35, 0.0)))
    mpm.incre_resolution.from_numpy(sweep)
    toc = float(contact.particle_full_ccd(0.9, mpm.incre_resolution))
    expected_toc = 0.9 * (0.50 - 0.01) / 0.70

    assert int(contact.body_pair_num[0]) == 1
    assert 0.0 < toc < 1.0
    assert toc == pytest.approx(expected_toc, rel=1.0e-12, abs=1.0e-12)


def test_direct_mpm_particle_barrier_projection_is_lagged_only(tmp_path):
    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    ti.init(arch=ti.cpu, default_fp=ti.f64, debug=False, offline_cache=False)
    _, mpm, contact = _build_soft_soft_contact(tmp_path)
    mpm.grid_disp.fill(0.0)
    contact.update_particle_pos(mpm.grid_disp)
    contact.point_point_distance()
    assert int(contact.pbarrierNum[0]) == 1

    def assembled_barrier(mode):
        # Assigning the mode directly isolates the common particle-barrier
        # assembly.  Production fully implicit SoftParticle currently limits
        # its supported geometry separately, in configuration validation.
        contact.friction_mode = mode
        mpm.rhs.fill(0.0)
        contact.barrier_hash_matrix.reset_system()
        contact.assemble_particle_barrier_matrix()
        return contact.barrier_matrix(mpm.active_dof).toarray()

    exact_hessian = assembled_barrier("fully_implicit")
    lagged_hessian = assembled_barrier("lagged")
    exact_eigenvalues = np.linalg.eigvalsh(0.5 * (exact_hessian + exact_hessian.T))
    lagged_eigenvalues = np.linalg.eigvalsh(0.5 * (lagged_hessian + lagged_hessian.T))
    exact_scale = max(float(np.max(np.abs(exact_eigenvalues))), 1.0)
    lagged_scale = max(float(np.max(np.abs(lagged_eigenvalues))), 1.0)

    assert np.all(np.isfinite(exact_hessian))
    assert np.all(np.isfinite(lagged_hessian))
    assert np.linalg.norm(exact_hessian, ord=np.inf) > 0.0
    assert np.allclose(exact_hessian, exact_hessian.T, rtol=1.0e-11, atol=1.0e-10)
    assert np.allclose(lagged_hessian, lagged_hessian.T, rtol=1.0e-11, atol=1.0e-10)
    assert exact_eigenvalues.min() < -1.0e-6 * exact_scale
    assert lagged_eigenvalues.min() >= -1.0e-11 * lagged_scale


def test_soft_particle_contact_compaction_has_stable_device_raw_slots(
    tmp_path,
):
    """Repeated active sets must hit the raw->reduced map without hashing."""
    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    ti.init(arch=ti.cpu, default_fp=ti.f64, debug=False, offline_cache=False)
    from src.linear_solver.BuildTriplet import BuildTriplet

    _, mpm, contact = _build_soft_soft_contact(
        tmp_path,
        positions=(
            ((0.40, 0.50), (0.43, 0.50)),
            ((0.49, 0.50), (0.52, 0.50)),
        ),
        dhat=0.15,
        expect_active=4,
    )
    mpm.grid_disp.fill(0.0)
    contact.update_particle_pos(mpm.grid_disp)

    # Exercise the production device reduction kernels on CPU-only CI.  Only
    # the backend differs; raw contact slots and the persistent map are the
    # same fields/kernels used by CUDA.
    def device_matrix_like(matrix):
        return BuildTriplet(
            dim=matrix.dim,
            max_pairs_num=matrix.non_diag.max_pairs_num,
            max_nonzeros=matrix.max_nonzeros,
            max_active_nodes=matrix.max_active_nodes,
            symmetric=matrix.symmetric,
            matrix_symmetric=matrix.matrix_symmetric,
            device_reduction=True,
        )

    contact.barrier_hash_matrix = device_matrix_like(contact.barrier_hash_matrix)
    contact.friction_hash_matrix = device_matrix_like(contact.friction_hash_matrix)

    def contact_pairs(field, count):
        return [(int(field[index].masterID), int(field[index].slaveID)) for index in range(int(count[0]))]

    def assemble_twice(matrix, rebuild, assemble, field, count):
        snapshots = []
        statistics = []
        pairs = []
        for _ in range(2):
            rebuild()
            pairs.append(contact_pairs(field, count))
            matrix.reset_system()
            contact.friction_grad.fill(0.0)
            mpm.rhs.fill(0.0)
            assemble()
            matrix.finalize_taichi_assembly()
            raw = int(matrix.raw_non_diag_count[0])
            snapshots.append(
                (
                    matrix.non_diag.blockI.to_numpy()[:raw].copy(),
                    matrix.non_diag.blockJ.to_numpy()[:raw].copy(),
                )
            )
            statistics.append(matrix.non_diag.pattern_cache_statistics())
        assert pairs[0] == pairs[1]
        assert pairs[0] == sorted(pairs[0])
        assert np.array_equal(snapshots[0][0], snapshots[1][0])
        assert np.array_equal(snapshots[0][1], snapshots[1][1])
        assert statistics[1]["pattern_version"] == statistics[0]["pattern_version"]
        assert statistics[1]["last_mapping_misses"] == 0

    assemble_twice(
        contact.barrier_hash_matrix,
        contact.point_point_distance,
        contact.assemble_particle_barrier_matrix,
        contact.pbarrier,
        contact.pbarrierNum,
    )
    assemble_twice(
        contact.friction_hash_matrix,
        contact.particle_friction_initialize,
        contact.assemble_particle_friction_matrix,
        contact.pfriction,
        contact.pfrictionNum,
    )


def test_stable_contact_compaction_reports_required_capacity_before_fill(
    tmp_path,
):
    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    ti.init(arch=ti.cpu, default_fp=ti.f64, debug=False, offline_cache=False)
    _, mpm, contact = _build_soft_soft_contact(
        tmp_path,
        positions=(
            ((0.40, 0.50), (0.43, 0.50)),
            ((0.49, 0.50), (0.52, 0.50)),
        ),
        dhat=0.15,
        expect_active=4,
        barrier_set=(1, 4),
    )
    mpm.grid_disp.fill(0.0)
    contact.update_particle_pos(mpm.grid_disp)
    with pytest.raises(RuntimeError, match="particle barrier contact capacity"):
        contact.point_point_distance()

    # The prefix sum retains the true requirement instead of silently
    # clamping it to the one-entry storage buffer.  The fill kernel is not
    # launched after this fail-fast check.
    assert int(contact.pbarrierNum[0]) == 4
    assert int(contact.pbarrier_overflow[0]) == 1


def test_ground_contact_compaction_is_surface_wall_deterministic(tmp_path):
    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"
    ti.init(arch=ti.cpu, default_fp=ti.f64, debug=False, offline_cache=False)
    from src.linear_solver.BuildTriplet import BuildTriplet

    _, mpm, contact = _build_soft_soft_contact(
        tmp_path,
        positions=(
            ((0.30, 0.04), (0.35, 0.04)),
            ((0.70, 0.04), (0.75, 0.04)),
        ),
        dhat=0.15,
        expect_active=0,
        barrier_set=(16, 16),
        friction_set=(16, 16),
        ground_specs=(
            ([0.0, 0.00], [0.0, 1.0]),
            ([0.0, -0.02], [0.0, 1.0]),
        ),
    )
    mpm.grid_disp.fill(0.0)
    contact.update_particle_pos(mpm.grid_disp)
    contact.initialize_hat_x()

    def device_matrix_like(matrix):
        return BuildTriplet(
            dim=matrix.dim,
            max_pairs_num=matrix.non_diag.max_pairs_num,
            max_nonzeros=matrix.max_nonzeros,
            max_active_nodes=matrix.max_active_nodes,
            symmetric=matrix.symmetric,
            matrix_symmetric=matrix.matrix_symmetric,
            device_reduction=True,
        )

    contact.barrier_hash_matrix = device_matrix_like(contact.barrier_hash_matrix)
    contact.friction_hash_matrix = device_matrix_like(contact.friction_hash_matrix)
    expected_pairs = [(surface, wall) for surface in range(mpm.total_surface_num) for wall in range(contact.ground.num)]

    def run_twice(matrix, rebuild, assemble, field, count):
        reference_raw = None
        first_statistics = None
        for iteration in range(2):
            rebuild()
            pairs = [
                (
                    int(field[index].surfaceID),
                    int(field[index].wallID),
                )
                for index in range(int(count[0]))
            ]
            assert pairs == expected_pairs
            matrix.reset_system()
            contact.friction_grad.fill(0.0)
            mpm.rhs.fill(0.0)
            assemble()
            matrix.finalize_taichi_assembly()
            raw = int(matrix.raw_non_diag_count[0])
            coordinates = (
                matrix.non_diag.blockI.to_numpy()[:raw].copy(),
                matrix.non_diag.blockJ.to_numpy()[:raw].copy(),
            )
            statistics = matrix.non_diag.pattern_cache_statistics()
            if iteration == 0:
                reference_raw = coordinates
                first_statistics = statistics
            else:
                assert np.array_equal(reference_raw[0], coordinates[0])
                assert np.array_equal(reference_raw[1], coordinates[1])
                assert statistics["pattern_version"] == first_statistics["pattern_version"]
                assert statistics["last_mapping_misses"] == 0

    run_twice(
        contact.barrier_hash_matrix,
        contact.point_ground_distance,
        contact.assemble_ground_barrier_matrix,
        contact.gbarrier,
        contact.gbarrierNum,
    )
    run_twice(
        contact.friction_hash_matrix,
        contact.ground_friction_initialize,
        contact.assemble_ground_friction_matrix,
        contact.gfriction,
        contact.gfrictionNum,
    )


if __name__ == "__main__":
    test_ipc_soft_soft_friction_production_assembly()
    test_direct_mpm_particle_full_ccd_covers_nonactive_body_pair()
    test_direct_mpm_particle_barrier_projection_is_lagged_only()
    test_soft_particle_contact_compaction_has_stable_device_raw_slots()
    test_stable_contact_compaction_reports_required_capacity_before_fill()
    test_ground_contact_compaction_is_surface_wall_deterministic()

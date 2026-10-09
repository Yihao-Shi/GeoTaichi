"""Three-dimensional IGA-MPM friction integration check."""

import numpy as np
import pytest
import taichi as ti

import src.igampm.config as config

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.serial]


@ti.kernel
def _set_uniform_x_disp(node2dof: ti.template(), grid_disp: ti.template(), value: ti.f64):
    for node in node2dof:
        offset = node2dof[node] - 1
        if offset >= 0:
            grid_disp[config.DIM * offset] = value
            for d in ti.static(range(1, config.DIM)):
                grid_disp[config.DIM * offset + d] = 0.0


def _build_iga_cube(output_path):
    from src.iga import Cube, ImplicitIGA, Primitives

    cube = Cube()
    cube.set_parameters(start_point=[0.0, 0.0, 0.0], size=[1.0, 1.0, 0.2])
    cube.generate_knot_u(degree=2, num_ctrlpts=3)
    cube.generate_knot_v(degree=2, num_ctrlpts=3)
    cube.generate_knot_w(degree=2, num_ctrlpts=3)
    cube.generate_ctrlpts()
    cube.generate_weights()

    primitives = Primitives()
    primitives.append(cube, "cube")
    primitives.finialize()
    return ImplicitIGA(
        primitives=primitives,
        newmark=[1.0, 0.5, 1.0],
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        density=1000.0,
        gravity=[0.0, 0.0, 0.0],
        residual=1.0e-8,
        interval=1,
        step=1,
        degree=[2, 2, 2],
        path=str(output_path),
    )


def _build_mpm_particle(output_path, points=None):
    from src.mpm.generator.Body import Body
    from src.mpm.engines.direct.ImplicitULMPM import ImplicitULMPM

    if points is None:
        points = [[0.5, 0.5, -0.031]]
    body = Body()
    body.add_particles(
        points,
        volume=1.0e-3,
        xmin=[-0.1, -0.1, -0.1],
        xmax=[1.1, 1.1, 0.3],
        boundary_ids=list(range(len(points))),
    )
    mpm = ImplicitULMPM(
        domain=[1.2, 1.2, 0.5],
        dx=0.1,
        dt=1.0e-3,
        bodies=body,
        newmark=[1.0, 0.5, 1.0],
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        density=1000.0,
        gravity=[0.0, 0.0, 0.0],
        residual=1.0e-8,
        interval=1,
        step=1,
        scale=1.0,
        line_search=False,
        shape_function="linear",
        visualize=False,
        path=str(output_path),
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
    assert mpm.active_dof > 0
    return mpm


@pytest.mark.isolated_dimension(3)
def test_swept_nurbs_bvh_catches_surface_crossing_inactive_point(taichi_runtime, tmp_path):
    from src.igampm import IGAMPM

    iga = _build_iga_cube(tmp_path / "iga")
    mpm = _build_mpm_particle(tmp_path / "mpm", [[0.5, 0.5, -0.081]])
    engine = IGAMPM(iga, mpm, kappa=1.0e4, dhat=0.02, dmin=0.005).build()
    engine.initialize_barrier()
    assert engine.curr_barrier_contact_num == 0
    direction = np.zeros(iga.degree_of_freedom)
    direction.reshape((-1, 3))[:, 2] = -0.2
    iga.incre_resolution.from_numpy(direction)
    mpm.incre_resolution.fill(0.0)
    alpha = engine.conservative_contact_step_device(safety=0.9)
    # The bottom face translates through the stationary particle.
    expected = 0.9 * (0.081 - 0.005 - engine.strict_feasibility_tolerance) / 0.2
    assert alpha == pytest.approx(expected, rel=1.0e-8, abs=1.0e-10)


@pytest.mark.isolated_dimension(3)
def test_igampm_coupling_friction_3d_opposes_tangential_motion(
    taichi_runtime,
    tmp_path,
):
    config.set_dimension(3)

    from src.igampm import IGAMPM

    iga = _build_iga_cube(tmp_path / "iga")
    mpm = _build_mpm_particle(tmp_path / "mpm")
    coupling = IGAMPM(
        iga, mpm, kappa=1.0e4, dhat=0.08, mu=0.5, epsv=1.0e-3, activate_friction=True, friction_nnz=50_000
    )

    coupling.initialize_friction()
    print(f"3D IGA-MPM friction contacts: {coupling.curr_friction_contact_num}")
    assert coupling.curr_friction_contact_num > 0

    _set_uniform_x_disp(mpm.node2dof, mpm.grid_disp, 1.0e-4)
    coupling.update_particle_pos(mpm.grid_disp)
    coupling.assemble_friction_system()

    K = coupling.friction_matrix()
    grad = coupling.friction_grad.to_numpy()
    forces = coupling.friction_contact_forces()
    sym_diff = (K - K.T).tocoo()
    max_sym = float(np.max(np.abs(sym_diff.data))) if sym_diff.nnz else 0.0
    mpm_force_x = float(np.sum(forces["mpm"][0 :: config.DIM]))
    print(
        "3D IGA-MPM friction assembly:",
        f"contacts={coupling.curr_friction_contact_num}",
        f"raw_triplets={int(coupling.friction_nnz_count[0])}",
        f"reduced_nnz={K.nnz}",
        f"grad_norm={np.linalg.norm(grad):.3e}",
        f"sym_max={max_sym:.3e}",
        f"mpm_force_x={mpm_force_x:.3e}",
    )
    assert int(coupling.friction_nnz_count[0]) > 0
    assert K.nnz > 0
    assert np.all(np.isfinite(K.data))
    assert np.all(np.isfinite(grad))
    assert np.linalg.norm(grad) > 0.0
    assert max_sym < 1.0e-7
    assert mpm_force_x < 0.0


@pytest.mark.isolated_dimension(3)
def test_compact_contact_slots_preserve_normal_and_friction(taichi_runtime, tmp_path):
    from src.igampm import IGAMPM

    points = [[0.5, 0.5, -0.081], [0.5, 0.5, -0.031]]
    engines = []
    for compact in (False, True):
        iga = _build_iga_cube(tmp_path / str(compact) / "iga")
        mpm = _build_mpm_particle(tmp_path / str(compact) / "mpm", points)
        coupling = IGAMPM(
            iga,
            mpm,
            kappa=1.0e4,
            dhat=0.08,
            mu=0.5,
            epsv=1.0e-3,
            activate_friction=True,
            compact_contact_slots=compact,
            **({"barrier_nnz": 289, "friction_nnz": 289} if compact else {}),
        )
        engine = coupling.build()
        engine.initialize_friction()
        assert engine.curr_barrier_contact_num == 1
        _set_uniform_x_disp(mpm.node2dof, mpm.grid_disp, 1.0e-4)
        engine.update_particle_pos(mpm.grid_disp)
        engine.assemble_barrier_system()
        engine.assemble_friction_system()
        engines.append((engine, engine.barrier_matrix(), engine.friction_matrix()))

    full, compact = engines
    for index in (1, 2):
        difference = (full[index] - compact[index]).tocoo()
        assert np.max(np.abs(difference.data), initial=0.0) < 1.0e-7
    np.testing.assert_allclose(full[0].barrier_grad.to_numpy(), compact[0].barrier_grad.to_numpy())
    np.testing.assert_allclose(full[0].friction_grad.to_numpy(), compact[0].friction_grad.to_numpy())
    assert int(compact[0].barrier_hash_matrix.raw_non_diag_count[0]) == 289
    assert int(compact[0].friction_hash_matrix.raw_non_diag_count[0]) == 289
    assert int(full[0].barrier_hash_matrix.raw_non_diag_count[0]) == 12 * 289
    # A second active pair exceeds the one-pair buffer instead of being dropped.
    active = compact[0].contacts.active.to_numpy()
    active[0] = 1
    compact[0].contacts.active.from_numpy(active)
    compact[0].prepare_barrier_matrix_slots()
    assert int(compact[0].barrier_hash_matrix.overflow[0]) == 1

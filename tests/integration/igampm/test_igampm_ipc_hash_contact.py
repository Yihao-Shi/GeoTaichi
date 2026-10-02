"""Integration checks for IPC contact with hash assembly."""

import numpy as np
import pytest

import src.mpm.config as config


pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.serial]


def _run_ipc_contact_case(solver_cls, label, output_path):
    from src.mpm.generator.Body import Body
    from src.mpm.generator.Ground import Ground

    ground = Ground()
    ground.append([0.0, 0.0], [0.0, 1.0])

    body = Body()
    body.add_rectangle([0.05, 0.0002], [0.15, 0.0502], 0.05, ppc=1, init_v=[0.0, 0.0])

    solver = solver_cls(
        domain=[0.25, 0.15],
        dx=0.05,
        dt=1.0e-4,
        bodies=body,
        ground=ground,
        newmark=[1.0, 0.5, 1.0],
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        density=1000.0,
        kappa=1.0e4,
        dhat=0.03,
        mu=0.0,
        epsv=1.0e-3,
        residual=1.0e-8,
        friction_residual=1.0e-8,
        activate_friction=False,
        gravity=[0.0, 0.0],
        step=1,
        interval=1,
        max_iters=1,
        line_search=False,
        shape_function="linear",
        visualize=False,
        path=str(output_path),
    )
    solver.initial_simulation()

    mpm = solver.mpm
    ipc = solver.ipc

    mpm.mass_vec.fill(0.0)
    mpm.grid_reset()
    if label == "UL":
        mpm.compute_shapefn()
        mpm.mass_vel_acc_p2g()
        mpm.find_active_node()
        mpm.prefix_sum_executor.run(mpm.node2dof)
        mpm.active_dof = mpm.set_active_dof()
    else:
        mpm.vel_acc_p2g()
    mpm.compute_nodal_vel_acc()
    if config.DYNAMIC:
        mpm.compute_mass_list(mpm.integration)

    mpm.grid_disp.fill(0.0)
    mpm.hash_matrix.reset_system()
    ipc.barrier_hash_matrix.reset_system()
    ipc.update_particle_pos(mpm.grid_disp)
    ipc.point_ground_distance()
    ipc.point_point_distance()
    ipc.curr_barrier_contact_num = int(ipc.pbarrierNum[0] + ipc.gbarrierNum[0])
    assert ipc.curr_barrier_contact_num > 0

    ipc.assemble_ground_barrier_matrix()
    ipc.assemble_particle_barrier_matrix()
    K = ipc.barrier_matrix(mpm.active_dof)

    diff = (K - K.T).tocoo()
    sym_max = float(np.max(np.abs(diff.data))) if diff.nnz else 0.0
    data_max = float(np.max(np.abs(K.data))) if K.nnz else 0.0
    print(
        f"IPC{label}MPM barrier hash: contacts={ipc.curr_barrier_contact_num} "
        f"raw={int(ipc.barrier_hash_matrix.raw_non_diag_count[0])} nnz={K.nnz} "
        f"max={data_max:.3e} sym={sym_max:.3e}"
    )
    assert K.shape == (mpm.active_dof, mpm.active_dof)
    assert K.nnz > 0
    assert np.all(np.isfinite(K.data))
    assert sym_max <= 1.0e-8


def test_igampm_ipc_hash_contact_is_symmetric_for_ul_and_tl(
    taichi_runtime,
    tmp_path,
):
    config.set_dimension(2)

    from src.mpm.soft_particle.IPCULMPM import IPCULMPM
    from src.mpm.soft_particle.IPCTLMPM import IPCTLMPM

    _run_ipc_contact_case(IPCULMPM, "UL", tmp_path / "ul")
    _run_ipc_contact_case(IPCTLMPM, "TL", tmp_path / "tl")

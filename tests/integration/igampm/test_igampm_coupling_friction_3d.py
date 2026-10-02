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


def _build_mpm_particle(output_path):
    from src.mpm.generator.Body import Body
    from src.mpm.engines.direct.ImplicitULMPM import ImplicitULMPM

    body = Body()
    body.add_particles(
        [[0.5, 0.5, -0.031]],
        volume=1.0e-3,
        xmin=[-0.1, -0.1, -0.1],
        xmax=[1.1, 1.1, 0.3],
        boundary_ids=[0],
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
def test_igampm_coupling_friction_3d_opposes_tangential_motion(
    taichi_runtime,
    tmp_path,
):
    config.set_dimension(3)

    from src.igampm import IGAMPM

    iga = _build_iga_cube(tmp_path / "iga")
    mpm = _build_mpm_particle(tmp_path / "mpm")
    coupling = IGAMPM(iga, mpm, kappa=1.0e4, dhat=0.08, mu=0.5, epsv=1.0e-3, activate_friction=True, friction_nnz=50_000)

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
    mpm_force_x = float(np.sum(forces["mpm"][0::config.DIM]))
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

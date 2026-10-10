"""Two-dimensional IGA-MPM friction integration check."""

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


def _set_coupled_displacement(iga, mpm, value):
    iga.grid_disp.from_numpy(np.ascontiguousarray(value[: iga.degree_of_freedom], dtype=np.float64))
    mpm_value = np.zeros(mpm.degree_of_freedom, dtype=np.float64)
    mpm_value[: mpm.active_dof] = value[iga.degree_of_freedom : iga.degree_of_freedom + mpm.active_dof]
    mpm.grid_disp.from_numpy(mpm_value)


def _evaluate_friction(coupling, iga, mpm, displacement):
    _set_coupled_displacement(iga, mpm, displacement)
    coupling.update_particle_pos(mpm.grid_disp)
    coupling.assemble_friction_system()
    active_dof = iga.degree_of_freedom + mpm.active_dof
    matrix = coupling.friction_matrix()[:active_dof, :active_dof].toarray()
    forces = coupling.friction_contact_forces(mpm.active_dof)
    force = np.concatenate((forces["iga"], forces["mpm"]))
    return force, matrix


def _fd_force_jacobian(force_function, displacement, step):
    jacobian = np.zeros((displacement.size, displacement.size), dtype=np.float64)
    for column in range(displacement.size):
        perturbation = np.zeros_like(displacement)
        perturbation[column] = step
        jacobian[:, column] = (
            force_function(displacement + perturbation) - force_function(displacement - perturbation)
        ) / (2.0 * step)
    return jacobian


def _mixed_error(actual, expected):
    return np.linalg.norm(actual - expected, ord=np.inf) / max(
        np.linalg.norm(actual, ord=np.inf),
        np.linalg.norm(expected, ord=np.inf),
        1.0,
    )


def _build_iga_rectangle(output_path):
    from src.iga import ImplicitIGA, Primitives, Rectangle

    rectangle = Rectangle()
    rectangle.set_parameters(start_point=[0.0, 0.0], size=[1.0, 0.2])
    rectangle.generate_knot_u(degree=2, num_ctrlpts=3)
    rectangle.generate_knot_v(degree=2, num_ctrlpts=3)
    rectangle.generate_ctrlpts()
    rectangle.generate_weights()

    primitives = Primitives()
    primitives.append(rectangle, "rectangle")
    primitives.finialize()
    return ImplicitIGA(
        primitives=primitives,
        newmark=[1.0, 0.5, 1.0],
        young_modulus=1.0e5,
        poisson_ratio=0.3,
        density=1000.0,
        gravity=[0.0, 0.0],
        residual=1.0e-8,
        interval=1,
        step=1,
        degree=[2, 2],
        path=str(output_path),
    )


def _build_mpm_particle(output_path):
    from src.mpm.generator.Body import Body
    from src.mpm.engines.direct.ImplicitULMPM import ImplicitULMPM

    body = Body()
    body.add_particles(
        [[0.5, -0.031]],
        volume=1.0e-3,
        xmin=[-0.1, -0.1],
        xmax=[1.1, 0.3],
        boundary_ids=[0],
    )
    mpm = ImplicitULMPM(
        domain=[1.2, 0.5],
        dx=0.1,
        dt=1.0e-3,
        bodies=body,
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


def test_igampm_friction_production_assembly(
    taichi_runtime,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("GEOTAICHI_REAL_DTYPE", "float64")
    config.set_dimension(2)

    from src.igampm import IGAMPM

    iga = _build_iga_rectangle(tmp_path / "iga")
    mpm = _build_mpm_particle(tmp_path / "mpm")
    coupling = IGAMPM(
        iga, mpm, kappa=1.0e4, dhat=0.08, mu=0.5, epsv=1.0e-3, activate_friction=True, friction_nnz=20_000
    )

    coupling.initialize_friction()
    print(f"2D IGA-MPM friction contacts: {coupling.curr_friction_contact_num}")
    assert coupling.curr_friction_contact_num > 0
    lagged_energy = coupling.friction_potential_energy()
    assert np.isfinite(lagged_energy)
    assert lagged_energy >= 0.0

    total_dof = iga.degree_of_freedom + mpm.active_dof
    translation = np.zeros(total_dof, dtype=np.float64)
    translation[: iga.degree_of_freedom].reshape((-1, config.DIM))[:, 0] = -0.5
    translation[iga.degree_of_freedom :].reshape((-1, config.DIM))[:, 0] = 0.5

    for label, amplitude, fd_step in (
        ("dynamic", 2.0e-4, 2.0e-8),
        ("smoothed", 2.0e-7, 2.0e-10),
    ):
        displacement = amplitude * translation
        force, matrix = _evaluate_friction(coupling, iga, mpm, displacement)
        gradient = coupling.friction_grad.to_numpy().copy()
        raw_values = coupling.friction_hash_matrix.non_diag.blockH.to_numpy().copy()
        diagonal = coupling.friction_hash_matrix.diag.to_numpy().copy()
        raw_count = int(coupling.friction_hash_matrix.raw_non_diag_count[0])
        coupling.friction_grad.fill(float("nan"))
        coupling.assemble_friction_system(need_matrix=False)
        np.testing.assert_allclose(coupling.friction_grad.to_numpy(), gradient, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(coupling.friction_hash_matrix.non_diag.blockH.to_numpy(), raw_values)
        np.testing.assert_array_equal(coupling.friction_hash_matrix.diag.to_numpy(), diagonal)
        assert int(coupling.friction_hash_matrix.raw_non_diag_count[0]) == raw_count
        assert int(coupling.friction_nnz_count[0]) == 0
        jacobian = _fd_force_jacobian(
            lambda value: _evaluate_friction(coupling, iga, mpm, value)[0],
            displacement,
            fd_step,
        )
        eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        component_force = force.reshape((-1, config.DIM)).sum(axis=0)

        assert int(coupling.friction_nnz_count[0]) > 0, label
        assert np.all(np.isfinite(matrix)), label
        assert np.all(np.isfinite(force)), label
        assert np.linalg.norm(force) > 0.0, label
        assert _mixed_error(jacobian, -matrix) < 8.0e-5, label
        assert np.allclose(matrix, matrix.T, rtol=1.0e-10, atol=1.0e-10), label
        assert eigenvalues.min() >= -1.0e-12 * max(eigenvalues.max(), 1.0), label
        assert np.allclose(component_force, 0.0, rtol=1.0e-9, atol=1.0e-9), label
        assert float(np.dot(force, displacement)) < 0.0, label

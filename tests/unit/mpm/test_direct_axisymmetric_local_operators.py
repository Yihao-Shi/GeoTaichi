"""Analytic oracles for Direct MPM 3D-embedded 2D material pullbacks."""

import numpy as np
import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.cpu,
    pytest.mark.serial,
    pytest.mark.isolated_dimension(2),
]


def test_finite_strain_plastic_stiffness_projects_forward_but_not_exact_reverse():
    from src.mpm.engines.direct.ImplicitTLMPM import ImplicitTLMPM
    from src.mpm.engines.direct.ImplicitULMPM import ImplicitULMPM

    class Dummy:
        is_finite_strain_plastic = True

        def _assemble_stiffness_matrix_hash(self, active_dof, grid_disp, project_spd):
            calls.append((active_dof, grid_disp, project_spd))

    for solver_class in (ImplicitULMPM, ImplicitTLMPM):
        calls = []
        solver_class.assemble_stiffness_matrix_hash(Dummy(), 12, None, project_spd=False)
        solver_class.assemble_stiffness_matrix_hash(
            Dummy(),
            12,
            None,
            project_spd=False,
            exact_plastic_tangent=True,
        )
        assert calls == [(12, None, True), (12, None, False)]


def test_axisymmetric_and_plane_strain_derivatives_match_oracles(
    taichi_runtime,
):
    import taichi as ti

    from src.mpm.engines.direct.ImplicitMPM import ImplicitMPM

    support_count = 2
    engine = object.__new__(ImplicitMPM)
    engine.axis_offset = 0.25
    engine.shape = ti.field(ti.f64, shape=(1, support_count))
    engine.dshape = ti.Vector.field(2, ti.f64, shape=(1, support_count))
    engine.F0 = ti.Matrix.field(3, 3, ti.f64, shape=1)
    particle_type = ti.types.struct(x=ti.types.vector(2, ti.f64))
    engine.particle = particle_type.field(shape=1)

    tangent_field = ti.Matrix.field(9, 9, ti.f64, shape=())
    derivative_field = ti.field(ti.f64, shape=(2 * support_count, 9))
    hessian_field = ti.field(ti.f64, shape=(2 * support_count, 2 * support_count))
    plane_derivative_field = ti.field(ti.f64, shape=(2 * support_count, 9))
    plane_hessian_field = ti.field(ti.f64, shape=(2 * support_count, 2 * support_count))

    @ti.kernel
    def evaluate():
        for local_id, component in ti.static(ti.ndrange(support_count, 2)):
            derivative = engine.axisymmetric_dF_du(0, local_id, component)
            for column, row in ti.static(ti.ndrange(3, 3)):
                derivative_field[2 * local_id + component, row + 3 * column] = derivative[row, column]
            plane_derivative = engine.plane_strain_dF_du(0, local_id, component)
            for column, row in ti.static(ti.ndrange(3, 3)):
                plane_derivative_field[2 * local_id + component, row + 3 * column] = plane_derivative[row, column]
        for local_i, local_j in ti.static(ti.ndrange(support_count, support_count)):
            block = engine.axisymmetric_local_stiffness(0, local_i, local_j, tangent_field[None])
            for component_i, component_j in ti.static(ti.ndrange(2, 2)):
                hessian_field[
                    2 * local_i + component_i,
                    2 * local_j + component_j,
                ] = block[component_i, component_j]
            plane_block = engine.plane_strain_local_stiffness(0, local_i, local_j, tangent_field[None])
            for component_i, component_j in ti.static(ti.ndrange(2, 2)):
                plane_hessian_field[
                    2 * local_i + component_i,
                    2 * local_j + component_j,
                ] = plane_block[component_i, component_j]

    shape = np.array([0.4, 0.6], dtype=np.float64)
    shape_gradients = np.array([[-0.7, 0.25], [0.7, -0.25]], dtype=np.float64)
    previous_deformation = np.array(
        [[1.08, 0.04, 0.0], [-0.03, 0.94, 0.0], [0.0, 0.0, 1.12]],
        dtype=np.float64,
    )
    tangent = np.random.default_rng(17062026).normal(size=(9, 9))
    particle_radius = 1.35

    engine.shape.from_numpy(shape.reshape(1, support_count))
    engine.dshape.from_numpy(shape_gradients.reshape(1, support_count, 2))
    engine.F0.from_numpy(previous_deformation.reshape(1, 3, 3))
    engine.particle.x.from_numpy(np.array([[particle_radius, 0.4]], dtype=np.float64))
    tangent_field[None] = tangent
    evaluate()

    radius = particle_radius - engine.axis_offset
    expected_derivative = np.zeros((2 * support_count, 9))
    expected_plane_derivative = np.zeros((2 * support_count, 9))
    for local_id in range(support_count):
        for component in range(2):
            incremental_derivative = np.zeros((3, 3))
            incremental_derivative[component, :2] = shape_gradients[local_id]
            if component == 0:
                incremental_derivative[2, 2] = shape[local_id] / radius
            total_derivative = incremental_derivative @ previous_deformation
            expected_derivative[2 * local_id + component] = total_derivative.T.reshape(-1)
            incremental_derivative[2, 2] = 0.0
            plane_total_derivative = incremental_derivative @ previous_deformation
            expected_plane_derivative[2 * local_id + component] = plane_total_derivative.T.reshape(-1)

    np.testing.assert_allclose(
        derivative_field.to_numpy(),
        expected_derivative,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        hessian_field.to_numpy(),
        expected_derivative @ tangent @ expected_derivative.T,
        rtol=5.0e-14,
        atol=5.0e-14,
    )
    np.testing.assert_allclose(
        plane_derivative_field.to_numpy(),
        expected_plane_derivative,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        plane_hessian_field.to_numpy(),
        expected_plane_derivative @ tangent @ expected_plane_derivative.T,
        rtol=5.0e-14,
        atol=5.0e-14,
    )

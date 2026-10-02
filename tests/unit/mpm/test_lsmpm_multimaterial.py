from types import SimpleNamespace

import numpy as np


def test_neohookean_soft_material_table_uses_material_id(taichi_runtime):
    ti = taichi_runtime
    from src.mpm.MaterialManager import SoftParticleNeoHookeanMaterialTable

    models = {
        0: SimpleNamespace(shear=2.0, lame_lambda=3.0),
        1: SimpleNamespace(shear=7.0, lame_lambda=11.0),
    }
    table = SoftParticleNeoHookeanMaterialTable(2, models)
    stress = ti.Matrix.field(3, 3, dtype=ti.f64, shape=2)
    energy = ti.field(dtype=ti.f64, shape=2)
    energy_gradient = ti.Vector.field(9, dtype=ti.f64, shape=2)
    energy_hessian = ti.Matrix.field(9, 9, dtype=ti.f64, shape=2)

    @ti.kernel
    def evaluate():
        deformation = ti.Matrix([[1.1, 0.05, 0.0], [0.0, 0.95, 0.02], [0.0, 0.0, 1.03]])
        for material_id in range(2):
            stress[material_id] = table.soft_particle_pk1(material_id, deformation)
            energy[material_id] = table.Psi(material_id, deformation)
            energy_gradient[material_id] = table.dPsi_div_dF(material_id, deformation)
            energy_hessian[material_id] = table.d2Psi_div_d2F(material_id, deformation)

    evaluate()
    ti.sync()

    deformation = np.asarray(
        [[1.1, 0.05, 0.0], [0.0, 0.95, 0.02], [0.0, 0.0, 1.03]],
        dtype=np.float64,
    )
    det_f = np.linalg.det(deformation)
    inverse_transpose = np.linalg.inv(deformation).T
    i1 = np.sum(deformation * deformation)
    for material_id, model in models.items():
        expected_stress = (
            model.shear * (deformation - inverse_transpose) + model.lame_lambda * np.log(det_f) * inverse_transpose
        )
        expected_energy = (
            0.5 * model.shear * (i1 - 3.0) - model.shear * np.log(det_f) + 0.5 * model.lame_lambda * np.log(det_f) ** 2
        )
        expected_hessian = np.zeros((9, 9), dtype=np.float64)
        for row in range(9):
            a, i = divmod(row, 3)
            for column in range(9):
                b, j = divmod(column, 3)
                expected_hessian[row, column] = (
                    model.shear * float(i == j and a == b)
                    + model.lame_lambda * inverse_transpose[i, a] * inverse_transpose[j, b]
                    - (model.lame_lambda * np.log(det_f) - model.shear)
                    * inverse_transpose[i, b]
                    * inverse_transpose[j, a]
                )
        np.testing.assert_allclose(stress.to_numpy()[material_id], expected_stress, rtol=1.0e-12)
        np.testing.assert_allclose(energy.to_numpy()[material_id], expected_energy, rtol=1.0e-12)
        np.testing.assert_allclose(
            energy_gradient.to_numpy()[material_id],
            expected_stress.reshape(-1, order="F"),
            rtol=1.0e-12,
        )
        np.testing.assert_allclose(
            energy_hessian.to_numpy()[material_id],
            expected_hessian,
            rtol=1.0e-12,
        )

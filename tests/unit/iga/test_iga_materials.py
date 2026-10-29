"""IGA's supported Neo-Hookean material uses consistent energy, PK1 and tangent interfaces."""

import numpy as np
import pytest
import taichi as ti

pytestmark = [pytest.mark.materials, pytest.mark.cpu]


@pytest.mark.parametrize("dimension", [pytest.param(d, marks=pytest.mark.isolated_dimension(d)) for d in (2, 3)])
@pytest.mark.parametrize("implicit", [False, True], ids=["explicit", "implicit"])
def test_iga_neohookean_material_derivatives(taichi_runtime, dimension, implicit, tmp_path):
    from src.iga import Cube, Rectangle, Primitives, ExplicitIGA, ImplicitIGA

    primitive = Rectangle() if dimension == 2 else Cube()
    primitive.set_parameters(start_point=[0.0] * dimension, size=[1.0] * dimension)
    primitive.generate_knot_u(degree=1, num_ctrlpts=2)
    primitive.generate_knot_v(degree=1, num_ctrlpts=2)
    if dimension == 3:
        primitive.generate_knot_w(degree=1, num_ctrlpts=2)
    primitive.generate_ctrlpts()
    primitive.generate_weights()
    bodies = Primitives()
    bodies.append(primitive, "material-point-test")
    bodies.finialize()
    solver = (ImplicitIGA if implicit else ExplicitIGA)(
        primitives=bodies,
        degree=[1] * dimension,
        young_modulus=2e5,
        poisson_ratio=0.3,
        density=1800.0,
        gravity=[0.0] * dimension,
        dt=1e-4,
        step=0,
        path=str(tmp_path),
    )
    model = solver.material
    size = dimension**2
    energy = ti.field(ti.f64, shape=())
    stress = ti.Vector.field(size, ti.f64, shape=())
    tangent = ti.Matrix.field(size, size, ti.f64, shape=())

    @ti.kernel
    def evaluate(values: ti.types.ndarray(dtype=ti.f64, ndim=2)):
        deformation = ti.Matrix.zero(ti.f64, dimension, dimension)
        for i, j in ti.static(ti.ndrange(dimension, dimension)):
            deformation[i, j] = values[i, j]
        energy[None] = model.Psi(deformation)
        stress[None] = model.dPsi_div_dF(deformation)
        tangent[None] = model.d2Psi_div_d2F(deformation)

    def pk1(deformation):
        jacobian = np.linalg.det(deformation)
        inverse_transpose = np.linalg.inv(deformation).T
        mu = 2e5 / (2 * 1.3)
        lame = 2e5 * 0.3 / (1.3 * 0.4)
        return mu * (deformation - inverse_transpose) + lame * np.log(jacobian) * inverse_transpose

    for deformation in (np.eye(dimension), np.diag(np.linspace(0.93, 1.12, dimension))):
        deformation[0, 1] = 0.0 if np.array_equal(deformation, np.eye(dimension)) else 0.07
        evaluate(deformation)
        jacobian = np.linalg.det(deformation)
        expected_energy = 0.5 * model.shear * (np.sum(deformation**2) - dimension)
        expected_energy -= model.shear * np.log(jacobian)
        expected_energy += 0.5 * model.lame_lambda * np.log(jacobian) ** 2
        assert energy[None] == pytest.approx(expected_energy, rel=1e-10, abs=1e-9)
        np.testing.assert_allclose(
            stress.to_numpy().reshape((dimension, dimension), order="F"), pk1(deformation), atol=1e-8
        )
        direction = np.arange(1, size + 1, dtype=np.float64).reshape(dimension, dimension) / size
        epsilon = 1e-6
        oracle = (pk1(deformation + epsilon * direction) - pk1(deformation - epsilon * direction)) / (2 * epsilon)
        numerical = (tangent.to_numpy() @ direction.flatten(order="F")).reshape((dimension, dimension), order="F")
        np.testing.assert_allclose(numerical, oracle, rtol=1e-8, atol=1e-4)

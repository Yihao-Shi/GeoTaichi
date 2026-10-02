"""Device checks for tensor-product NURBS basis derivatives."""

import numpy as np
import pytest
import taichi as ti

from src.nurbs.core.NurbsGeometry import (
    NurbsBasisFunction2d,
    NurbsBasisFunction3d,
)


pytestmark = [pytest.mark.unit, pytest.mark.geometry]


def setup_module():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)


def teardown_module():
    ti.reset()


@ti.data_oriented
class _TensorBasisHarness:
    def __init__(self):
        self.basis2 = NurbsBasisFunction2d(3, 3)
        self.basis3 = NurbsBasisFunction3d(2, 2, 2)
        self.knots2 = ti.field(ti.f64, shape=8)
        self.knots3 = ti.field(ti.f64, shape=6)
        self.weights2 = ti.field(ti.f64, shape=16)
        self.weights3 = ti.field(ti.f64, shape=27)
        self.shape2 = ti.field(ti.f64, shape=16)
        self.first2 = ti.Vector.field(2, ti.f64, shape=16)
        self.second2 = ti.Vector.field(3, ti.f64, shape=16)
        self.shape3 = ti.field(ti.f64, shape=27)
        self.first3 = ti.Vector.field(3, ti.f64, shape=27)
        self.second3 = ti.Vector.field(6, ti.f64, shape=27)

        self.knots2.from_numpy(
            np.asarray(
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                dtype=np.float64,
            )
        )
        self.knots3.from_numpy(
            np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        )
        self.weights2.fill(1.0)
        self.weights3.fill(1.0)

    @ti.kernel
    def evaluate_2d(self, u: ti.f64, v: ti.f64):
        shape, first, second = self.basis2.NurbsBasis2ndDers2d(
            0,
            0,
            0,
            8,
            8,
            u,
            v,
            self.knots2,
            self.knots2,
            self.weights2,
        )
        for index in range(16):
            self.shape2[index] = shape[index]
            self.first2[index] = ti.Vector([first[index, 0], first[index, 1]])
            self.second2[index] = ti.Vector(
                [second[index, 0], second[index, 1], second[index, 2]]
            )

    @ti.kernel
    def evaluate_3d(self, u: ti.f64, v: ti.f64, w: ti.f64):
        shape, first, second = self.basis3.NurbsBasis2ndDers3d(
            0,
            0,
            0,
            0,
            6,
            6,
            6,
            u,
            v,
            w,
            self.knots3,
            self.knots3,
            self.knots3,
            self.weights3,
        )
        for index in range(27):
            self.shape3[index] = shape[index]
            self.first3[index] = ti.Vector(
                [first[index, 0], first[index, 1], first[index, 2]]
            )
            self.second3[index] = ti.Vector(
                [
                    second[index, 0],
                    second[index, 1],
                    second[index, 2],
                    second[index, 3],
                    second[index, 4],
                    second[index, 5],
                ]
            )


def _quadratic_bernstein(value):
    basis = np.asarray(
        [(1.0 - value) ** 2, 2.0 * value * (1.0 - value), value**2]
    )
    first = np.asarray([2.0 * (value - 1.0), 2.0 - 4.0 * value, 2.0 * value])
    second = np.asarray([2.0, -4.0, 2.0])
    return basis, first, second


def _cubic_bernstein(value):
    basis = np.asarray(
        [
            (1.0 - value) ** 3,
            3.0 * value * (1.0 - value) ** 2,
            3.0 * value**2 * (1.0 - value),
            value**3,
        ]
    )
    first = np.asarray(
        [
            -3.0 * (1.0 - value) ** 2,
            3.0 * (1.0 - value) * (1.0 - 3.0 * value),
            3.0 * value * (2.0 - 3.0 * value),
            3.0 * value**2,
        ]
    )
    second = np.asarray(
        [
            6.0 * (1.0 - value),
            18.0 * value - 12.0,
            6.0 - 18.0 * value,
            6.0 * value,
        ]
    )
    return basis, first, second


def test_tensor_second_derivatives_populate_every_support_entry():
    harness = _TensorBasisHarness()
    u, v, w = 0.23, 0.41, 0.67
    bu, du, ddu = _cubic_bernstein(u)
    bv, dv, ddv = _cubic_bernstein(v)
    bw, dw, ddw = _quadratic_bernstein(w)

    harness.evaluate_2d(u, v)
    expected_shape2 = np.einsum("j,k->jk", bv, bu).reshape(-1)
    expected_first2 = np.stack(
        [
            np.einsum("j,k->jk", bv, du).reshape(-1),
            np.einsum("j,k->jk", dv, bu).reshape(-1),
        ],
        axis=1,
    )
    expected_second2 = np.stack(
        [
            np.einsum("j,k->jk", bv, ddu).reshape(-1),
            np.einsum("j,k->jk", ddv, bu).reshape(-1),
            np.einsum("j,k->jk", dv, du).reshape(-1),
        ],
        axis=1,
    )
    np.testing.assert_allclose(harness.shape2.to_numpy(), expected_shape2, atol=2e-14)
    np.testing.assert_allclose(harness.first2.to_numpy(), expected_first2, atol=2e-13)
    np.testing.assert_allclose(
        harness.second2.to_numpy(), expected_second2, atol=2e-12
    )

    harness.evaluate_3d(u, v, w)
    bu, du, ddu = _quadratic_bernstein(u)
    bv, dv, ddv = _quadratic_bernstein(v)
    tensor = lambda a, b, c: np.einsum("i,j,k->ijk", c, b, a).reshape(-1)
    expected_shape3 = tensor(bu, bv, bw)
    expected_first3 = np.stack(
        [tensor(du, bv, bw), tensor(bu, dv, bw), tensor(bu, bv, dw)], axis=1
    )
    expected_second3 = np.stack(
        [
            tensor(ddu, bv, bw),
            tensor(bu, ddv, bw),
            tensor(bu, bv, ddw),
            tensor(du, dv, bw),
            tensor(bu, dv, dw),
            tensor(du, bv, dw),
        ],
        axis=1,
    )
    np.testing.assert_allclose(harness.shape3.to_numpy(), expected_shape3, atol=3e-14)
    np.testing.assert_allclose(harness.first3.to_numpy(), expected_first3, atol=3e-13)
    np.testing.assert_allclose(
        harness.second3.to_numpy(), expected_second3, atol=4e-12
    )

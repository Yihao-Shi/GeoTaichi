import numpy as np
import pytest
import taichi as ti

pytestmark = [pytest.mark.unit, pytest.mark.ipc, pytest.mark.contact]

from src.physics_model.contact_model.ipc.ContactAssembly import (
    project_to_psd_dense,
    psd_project_gershgorin_nd,
    psd_project_nd,
)
from src.physics_model.contact_model.ipc.ContactDistance import (
    point_triangle_distance_grad_hess,
)
from src.physics_model.contact_model.ipc.IPC import (
    ipc_toolkit_barrier_distance2_terms,
)


def setup_module():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)
    global _PROJECTION_HARNESSES
    _PROJECTION_HARNESSES = {dimension: _PSDProjectionHarness(dimension) for dimension in (2, 3, 4, 5, 6, 9, 12)}
    global _GERSHGORIN_HARNESS
    _GERSHGORIN_HARNESS = _GershgorinProjectionHarness(12)


def teardown_module():
    global _PROJECTION_HARNESSES
    _PROJECTION_HARNESSES = {}
    global _GERSHGORIN_HARNESS
    _GERSHGORIN_HARNESS = None
    ti.reset()


@ti.data_oriented
class _PSDProjectionHarness:
    def __init__(self, dimension):
        self.dimension = int(dimension)
        self.matrix = ti.Matrix.field(self.dimension, self.dimension, ti.f64, shape=())
        self.projected = ti.Matrix.field(self.dimension, self.dimension, ti.f64, shape=())

    @ti.kernel
    def run(self):
        self.projected[None] = psd_project_nd(self.matrix[None])


@ti.data_oriented
class _GershgorinProjectionHarness:
    def __init__(self, dimension):
        self.matrix = ti.Matrix.field(dimension, dimension, ti.f64, shape=())
        self.projected = ti.Matrix.field(dimension, dimension, ti.f64, shape=())

    @ti.kernel
    def run(self):
        self.projected[None] = psd_project_gershgorin_nd(self.matrix[None])


@ti.data_oriented
class _OfficialPTBarrierHarness:
    def __init__(self):
        self.positions = ti.Vector.field(3, ti.f64, shape=4)
        self.raw_hessian = ti.Matrix.field(12, 12, ti.f64, shape=())

    @ti.kernel
    def run(self, active_distance: ti.f64, kappa: ti.f64):
        distance2, gradient, distance_hessian, unused_type = point_triangle_distance_grad_hess(
            self.positions[0],
            self.positions[1],
            self.positions[2],
            self.positions[3],
        )
        unused_energy, first, second = ipc_toolkit_barrier_distance2_terms(
            distance2, active_distance * active_distance, kappa
        )
        hessian = ti.Matrix.zero(ti.f64, 12, 12)
        for row, column in ti.ndrange(12, 12):
            hessian[row, column] = second * gradient[row] * gradient[column] + first * distance_hessian[row, column]
        self.raw_hessian[None] = hessian


def _run_projection(dimension, matrix):
    harness = _PROJECTION_HARNESSES[dimension]
    harness.matrix.from_numpy(np.ascontiguousarray(matrix, dtype=np.float64))
    harness.run()
    return harness.projected.to_numpy()


def test_gershgorin_projection_is_psd_for_large_contact_block():
    rng = np.random.default_rng(9021)
    matrix = rng.normal(size=(12, 12))
    matrix = 0.5 * (matrix + matrix.T)
    _GERSHGORIN_HARNESS.matrix.from_numpy(matrix)
    _GERSHGORIN_HARNESS.run()
    projected = _GERSHGORIN_HARNESS.projected.to_numpy()
    np.testing.assert_allclose(projected, projected.T, atol=1.0e-14)
    assert np.linalg.eigvalsh(projected).min() >= -1.0e-12


@pytest.mark.parametrize("sample", range(12))
def test_device_2x2_psd_matches_numpy_eigh(sample):
    rng = np.random.default_rng(2100 + sample)
    matrix = rng.normal(size=(2, 2))
    # Exercise the same explicit symmetrization used by the production helper.
    matrix[0, 1] += 0.37
    actual = _run_projection(2, matrix)
    expected = project_to_psd_dense(matrix, method="clamp")
    np.testing.assert_allclose(actual, expected, rtol=8.0e-14, atol=8.0e-14)
    np.testing.assert_allclose(actual, actual.T, rtol=0.0, atol=2.0e-15)
    assert np.linalg.eigvalsh(actual).min() >= -2.0e-14


@pytest.mark.parametrize(
    "base",
    [
        np.asarray([[-2.0, 0.75], [0.75, 1.0]]),
        np.asarray([[1.0, 0.125], [0.125, -3.0]]),
        np.asarray([[-0.5, -1.25], [-1.25, 2.0]]),
        np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        np.asarray([[2.0, 0.0], [0.0, -1.0]]),
    ],
)
def test_device_2x2_psd_is_scale_invariant(base):
    expected = project_to_psd_dense(base, method="clamp")
    reference = _run_projection(2, base)
    np.testing.assert_allclose(reference, expected, rtol=8.0e-14, atol=8.0e-14)

    for scale in np.logspace(-20.0, 20.0, 17):
        projected = _run_projection(2, scale * base)
        normalized = projected / scale
        assert np.isfinite(projected).all()
        np.testing.assert_allclose(normalized, expected, rtol=1.2e-13, atol=1.2e-13)
        np.testing.assert_allclose(normalized, reference, rtol=1.2e-13, atol=1.2e-13)
        np.testing.assert_allclose(normalized, normalized.T, rtol=0.0, atol=2.0e-15)
        assert np.linalg.eigvalsh(normalized).min() >= -2.0e-14


@pytest.mark.parametrize("dimension", [2, 3, 4, 5, 6, 9, 12])
def test_device_psd_matches_numpy_eigh_random_blocks(dimension):
    rng = np.random.default_rng(4100 + dimension)
    sample_count = 2 if dimension in (9, 12) else 1
    for sample in range(sample_count):
        basis, unused_r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
        eigenvalues = np.linspace(-3.0, 4.0, dimension)
        rng.shuffle(eigenvalues)
        # Taichi's CPU backend can leave IEEE exception flags set after a
        # masked Jacobi pivot; suppress those stale flags around NumPy BLAS
        # while checking finiteness explicitly below.
        with np.errstate(all="ignore"):
            matrix = basis @ np.diag(eigenvalues) @ basis.T
        # Verify the same input symmetrization semantics as the small-matrix
        # implementation, rather than only feeding exactly symmetric arrays.
        matrix += 1.0e-9 * rng.normal(size=matrix.shape)
        assert np.isfinite(matrix).all()
        actual = _run_projection(dimension, matrix)
        with np.errstate(all="ignore"):
            expected = project_to_psd_dense(matrix, method="clamp")
        assert np.isfinite(actual).all(), sample
        np.testing.assert_allclose(actual, actual.T, atol=2.0e-11)
        assert np.linalg.eigvalsh(actual).min() >= -2.0e-9
        np.testing.assert_allclose(actual, expected, rtol=2.0e-9, atol=2.0e-9)


@pytest.mark.parametrize(
    "direction",
    [
        np.asarray([1.0, 2.0, -3.0]),
        np.asarray([-4.0, 0.25, 1.5]),
        np.asarray([0.125, -8.0, 2.0]),
    ],
)
def test_device_3x3_psd_preserves_repeated_coulomb_spectrum(direction):
    """Rank-two sliding friction must remain ``[0, a, a]`` on device."""

    direction = direction / np.linalg.norm(direction)
    projector = np.eye(3) - np.outer(direction, direction)
    for scale in np.logspace(-16.0, 16.0, 9):
        matrix = scale * projector
        actual = _run_projection(3, matrix)
        normalized = actual / scale

        assert np.isfinite(actual).all()
        np.testing.assert_allclose(normalized, projector, rtol=2.0e-12, atol=2.0e-12)
        eigenvalues = np.linalg.eigvalsh(0.5 * (normalized + normalized.T))
        np.testing.assert_allclose(
            eigenvalues,
            np.asarray([0.0, 1.0, 1.0]),
            rtol=2.0e-12,
            atol=2.0e-12,
        )


def test_device_3x3_psd_handles_one_ulp_split_repeated_eigenvalues():
    """Regression for Taichi 1.7 ``sym_eig`` manufacturing a huge mode."""

    rng = np.random.default_rng(7351)
    rotation, unused_r = np.linalg.qr(rng.normal(size=(3, 3)))
    for scale in np.logspace(-12.0, 12.0, 7):
        repeated = 476.738059 * scale
        eigenvalues = np.asarray(
            [
                0.0,
                repeated,
                np.nextafter(repeated, np.inf),
            ]
        )
        matrix = rotation @ np.diag(eigenvalues) @ rotation.T
        actual = _run_projection(3, matrix)

        assert np.isfinite(actual).all()
        np.testing.assert_allclose(
            actual,
            matrix,
            rtol=3.0e-12,
            atol=3.0e-12 * max(scale, 1.0e-12),
        )
        actual_eigenvalues = np.linalg.eigvalsh(0.5 * (actual + actual.T))
        assert actual_eigenvalues[-1] <= 1.0e-11 + 1.00000000001 * (eigenvalues[-1])


def test_device_jacobi_psd_matches_numpy_for_official_pt_barrier_block():
    harness = _OfficialPTBarrierHarness()
    # The closest feature is edge (a,b), producing the indefinite exact IPC
    # barrier block projected before global assembly.
    harness.positions.from_numpy(
        np.asarray(
            [
                [0.50, -0.05, 0.08],
                [0.00, 0.00, 0.00],
                [1.00, 0.00, 0.00],
                [0.00, 1.00, 0.00],
            ],
            dtype=np.float64,
        )
    )
    harness.run(active_distance=0.20, kappa=3.5)
    raw = harness.raw_hessian.to_numpy()
    projector = _PROJECTION_HARNESSES[12]
    projector.matrix.from_numpy(np.ascontiguousarray(raw))
    projector.run()
    actual = projector.projected.to_numpy()
    with np.errstate(all="ignore"):
        expected = project_to_psd_dense(raw, method="clamp")
    assert np.linalg.eigvalsh(0.5 * (raw + raw.T)).min() < -1.0e-8
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, actual.T, atol=2.0e-11)
    assert np.linalg.eigvalsh(actual).min() >= -2.0e-9
    np.testing.assert_allclose(actual, expected, rtol=2.0e-9, atol=2.0e-9)

"""Real-device smoke oracles for small IPC and sparse-assembly kernels.

These tests never accept Taichi's CPU fallback as a GPU result.  A backend is
skipped only when it cannot execute even the minimal capability probe below;
after that probe succeeds, target-kernel compilation and numerical failures
remain test failures.
"""

import numpy as np
import pytest
import taichi as ti

from src.linear_solver.HashReduction import HashReduction
from src.physics_model.contact_model.ipc.ContactAssembly import psd_project_nd
from src.physics_model.contact_model.ipc.IPC import (
    ipc_friction_f0,
    ipc_friction_f1_over_speed,
    ipc_friction_hessian_term,
    ipc_toolkit_barrier_distance_terms,
)


pytestmark = [
    pytest.mark.unit,
    pytest.mark.ipc,
    pytest.mark.contact,
]


@ti.kernel
def _device_capability_probe(value: ti.f32) -> ti.f32:
    return value + 1.0


def _initialize_real_device(requested_arch, default_fp):
    ti.reset()
    try:
        ti.init(
            arch=requested_arch,
            default_fp=default_fp,
            offline_cache=False,
            enable_fallback=False,
        )
    except Exception as error:
        ti.reset()
        pytest.skip(
            f"Taichi backend {requested_arch!s} is unavailable: {error}"
        )

    actual_arch = ti.lang.impl.current_cfg().arch
    if requested_arch == ti.cuda:
        is_requested_device = actual_arch == ti.cuda
    else:
        is_requested_device = actual_arch != ti.cpu
    if not is_requested_device:
        ti.reset()
        pytest.skip(
            f"Taichi selected {actual_arch!s}, not a real "
            f"{requested_arch!s} device"
        )
    try:
        probe_result = float(_device_capability_probe(2.0))
        ti.sync()
    except Exception as error:
        ti.reset()
        pytest.skip(
            f"Taichi selected {actual_arch!s}, but that backend cannot "
            f"execute a minimal device kernel: {error}"
        )
    if probe_result != pytest.approx(3.0):
        ti.reset()
        pytest.fail(
            f"Taichi {actual_arch!s} capability probe returned "
            f"{probe_result}, expected 3"
        )
    return actual_arch


@pytest.fixture
def real_gpu_f32():
    actual_arch = _initialize_real_device(ti.gpu, ti.f32)
    try:
        yield actual_arch
    finally:
        ti.sync()
        ti.reset()


@pytest.fixture
def real_cuda_f64():
    actual_arch = _initialize_real_device(ti.cuda, ti.f64)
    try:
        yield actual_arch
    finally:
        ti.sync()
        ti.reset()


@ti.data_oriented
class _IPCScalarHarness:
    def __init__(self):
        self.output = ti.Vector.field(6, dtype=ti.f32, shape=())

    @ti.kernel
    def evaluate(
        self,
        distance: ti.f32,
        active_distance: ti.f32,
        kappa: ti.f32,
        speed: ti.f32,
        epsv: ti.f32,
        timestep: ti.f32,
    ):
        energy, gradient, hessian = ipc_toolkit_barrier_distance_terms(
            distance, active_distance, kappa
        )
        self.output[None] = ti.Vector(
            [
                energy,
                gradient,
                hessian,
                ipc_friction_f0(speed, epsv, timestep),
                ipc_friction_f1_over_speed(speed, epsv),
                ipc_friction_hessian_term(speed, epsv),
            ]
        )


@ti.data_oriented
class _PSD2Harness:
    def __init__(self):
        self.matrix = ti.Matrix.field(2, 2, dtype=ti.f64, shape=())
        self.projected = ti.Matrix.field(2, 2, dtype=ti.f64, shape=())

    @ti.kernel
    def run(self):
        self.projected[None] = psd_project_nd(self.matrix[None])


def _barrier_oracle(distance, active_distance, kappa):
    distance2 = distance * distance
    active_distance2 = active_distance * active_distance
    if distance2 >= active_distance2:
        return np.zeros(3, dtype=np.float64)
    diff = distance2 - active_distance2
    log_term = np.log(distance2 / active_distance2)
    gradient_distance2 = -kappa * (
        2.0 * diff * log_term + diff * diff / distance2
    )
    hessian_distance2 = -kappa * (
        2.0 * log_term
        + 4.0 * diff / distance2
        - diff * diff / (distance2 * distance2)
    )
    return np.asarray(
        [
            -kappa * diff * diff * log_term,
            2.0 * distance * gradient_distance2,
            2.0 * gradient_distance2
            + 4.0 * distance2 * hessian_distance2,
        ],
        dtype=np.float64,
    )


def _friction_oracle(speed, epsv, timestep):
    if speed < epsv:
        displacement = speed * timestep
        threshold = epsv * timestep
        f0 = (
            displacement
            * displacement
            * (-displacement / 3.0 + threshold)
            / (threshold * threshold)
            + threshold / 3.0
        )
        f1_over_speed = (-speed + 2.0 * epsv) / (epsv * epsv)
        hessian_term = -1.0 / (epsv * epsv)
    else:
        f0 = speed * timestep
        f1_over_speed = 1.0 / speed
        hessian_term = -1.0 / (speed * speed)
    return np.asarray(
        [f0, f1_over_speed, hessian_term], dtype=np.float64
    )


@pytest.mark.gpu
def test_real_gpu_ipc_scalar_kernels_match_numpy_oracle(real_gpu_f32):
    assert real_gpu_f32 != ti.cpu
    harness = _IPCScalarHarness()
    active_distance = 0.2
    kappa = 3.5
    epsv = 0.1
    timestep = 0.01

    for distance, speed in (
        (0.08, 0.0),
        (0.16, 0.04),
        (0.25, 0.10),
        (0.30, 0.25),
    ):
        harness.evaluate(
            distance, active_distance, kappa, speed, epsv, timestep
        )
        ti.sync()
        actual = np.asarray(
            [float(harness.output[None][i]) for i in range(6)],
            dtype=np.float64,
        )
        expected = np.concatenate(
            (
                _barrier_oracle(distance, active_distance, kappa),
                _friction_oracle(speed, epsv, timestep),
            )
        )
        np.testing.assert_allclose(
            actual, expected, rtol=3.0e-5, atol=3.0e-5
        )


def _coordinate_block_sums(block_i, block_j, block_h):
    result = {}
    for row, column, value in zip(block_i, block_j, block_h):
        if row < 0 or column < 0:
            continue
        key = (int(row), int(column))
        if key not in result:
            result[key] = np.zeros(value.shape, dtype=np.float64)
        result[key] += value
    return result


@pytest.mark.gpu
@pytest.mark.cuda
@pytest.mark.assembly
@pytest.mark.hash_triplet
def test_real_cuda_psd_and_hash_reduction_match_numpy_oracles(
    real_cuda_f64,
):
    assert real_cuda_f64 == ti.cuda

    matrix = np.asarray([[-2.0, 0.75], [0.75, 1.0]], dtype=np.float64)
    psd = _PSD2Harness()
    psd.matrix.from_numpy(matrix)
    psd.run()
    ti.sync()
    actual_projection = psd.projected.to_numpy()
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    expected_projection = (
        eigenvectors @ np.diag(np.maximum(eigenvalues, 0.0))
        @ eigenvectors.T
    )
    np.testing.assert_allclose(
        actual_projection,
        expected_projection,
        rtol=2.0e-12,
        atol=2.0e-12,
    )

    block_i = np.asarray([0, 0, 1, 0, 1, 2, -1], dtype=np.int32)
    block_j = np.asarray([1, 1, 0, 2, 0, 1, -1], dtype=np.int32)
    block_h = np.arange(1.0, 29.0, dtype=np.float64).reshape(7, 4)
    reduction = HashReduction(
        max_pairs_num=block_i.size,
        dim=2,
        max_nnz=6,
        hessian_size=4,
        pattern_cache=True,
    )
    assert reduction.device_reduction

    reduction.set_triplets_from_numpy(block_i, block_j, block_h)
    reduction.go(block_i.size)
    out_i, out_j, out_h = reduction.get_reduced_triplets_numpy()
    expected = _coordinate_block_sums(block_i, block_j, block_h)
    actual = _coordinate_block_sums(out_i, out_j, out_h)
    assert actual.keys() == expected.keys()
    for coordinate in expected:
        np.testing.assert_allclose(
            actual[coordinate],
            expected[coordinate],
            rtol=2.0e-13,
            atol=2.0e-13,
        )

    first_statistics = reduction.pattern_cache_statistics()
    scaled_h = -0.25 * block_h
    reduction.set_triplets_from_numpy(block_i, block_j, scaled_h)
    reduction.go(block_i.size)
    second_statistics = reduction.pattern_cache_statistics()
    assert second_statistics["last_mapping_misses"] == 0
    assert second_statistics["pattern_hits"] == (
        first_statistics["pattern_hits"] + 1
    )
    out_i, out_j, out_h = reduction.get_reduced_triplets_numpy()
    expected = _coordinate_block_sums(block_i, block_j, scaled_h)
    actual = _coordinate_block_sums(out_i, out_j, out_h)
    assert actual.keys() == expected.keys()
    for coordinate in expected:
        np.testing.assert_allclose(
            actual[coordinate],
            expected[coordinate],
            rtol=2.0e-13,
            atol=2.0e-13,
        )

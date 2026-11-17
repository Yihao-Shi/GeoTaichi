import numpy as np
import taichi as ti

from examples.mmpm.GranularWaterLeakage3D.granular_water_leakage_3d import record_velocity_extrema


def test_velocity_extrema_keep_a_transient_peak_without_changing_particles():
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=2, offline_cache=False)
    try:
        particle = ti.types.struct(
            active=ti.u8, phase=ti.u8, vf=ti.types.vector(3, ti.f64), vs=ti.types.vector(3, ti.f64)
        ).field(shape=3)
        count = np.array([3], dtype=np.int32)
        extrema = ti.field(ti.f64, shape=4)
        particle.active.from_numpy(np.array([1, 1, 0], dtype=np.uint8))
        particle.phase.from_numpy(np.array([2, 1, 2], dtype=np.uint8))
        particle.vf.from_numpy(np.array([[6, 0, 0], [99, 0, 0], [99, 0, 0]], dtype=float))
        particle.vs.from_numpy(np.array([[99, 0, 0], [0, 3, 4], [99, 0, 0]], dtype=float))
        before = particle.to_numpy()
        record_velocity_extrema(int(count[0]), particle, extrema)
        for key, value in before.items():
            np.testing.assert_array_equal(particle.to_numpy()[key], value)
        particle.vf.fill(0)
        particle.vs.fill(0)
        record_velocity_extrema(int(count[0]), particle, extrema)
        np.testing.assert_array_equal(extrema.to_numpy(), [25, 36, 2, 0])
        for invalid in (float("nan"), float("inf")):
            extrema[3] = 0
            particle.vf[0] = [invalid, 0, 0]
            record_velocity_extrema(int(count[0]), particle, extrema)
            assert extrema[3] == 1
    finally:
        ti.reset()

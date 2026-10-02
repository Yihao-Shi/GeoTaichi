"""Local fixtures for deterministic constitutive-model unit tests."""

import pytest
import taichi as ti


@pytest.fixture
def taichi_material_cpu():
    """Give each constitutive test an isolated deterministic CPU runtime."""

    ti.reset()
    try:
        ti.init(
            arch=ti.cpu,
            default_fp=ti.f64,
            cpu_max_num_threads=1,
            offline_cache=False,
            debug=True,
        )
        yield ti
    finally:
        ti.reset()

import os

import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from src.fem.soft_particle.ContactManager import (
    FEMSoftParticleContactManager,
)


pytestmark = [
    pytest.mark.unit,
    pytest.mark.fedem,
    pytest.mark.contact,
    pytest.mark.cpu,
    pytest.mark.serial,
]


@pytest.fixture(autouse=True)
def taichi_runtime():
    ti.reset()
    arch = ti.cuda if os.environ.get("GEOTAICHI_TEST_ARCH") == "cuda" else ti.cpu
    ti.init(
        arch=arch,
        default_fp=ti.f64,
        cpu_max_num_threads=1,
        offline_cache=False,
        log_level=ti.ERROR,
    )
    yield
    ti.reset()


def test_sparse_contact_history_stores_only_nonzero_tangential_state():
    manager = object.__new__(FEMSoftParticleContactManager)
    pt_stencil = ti.Vector.field(4, dtype=ti.i32, shape=3)
    ee_stencil = ti.Vector.field(4, dtype=ti.i32, shape=2)
    pt_overlap = ti.Vector.field(3, dtype=ti.f64, shape=3)
    ee_overlap = ti.Vector.field(3, dtype=ti.f64, shape=2)
    history_state = ti.field(dtype=ti.i32, shape=16)
    history_kind = ti.field(dtype=ti.i32, shape=16)
    history_stencil = ti.Vector.field(4, dtype=ti.i32, shape=16)
    history_overlap = ti.Vector.field(3, dtype=ti.f64, shape=16)
    history_overflow = ti.field(dtype=ti.i32, shape=())
    entry_count = ti.field(dtype=ti.i32, shape=())

    pt_stencil.from_numpy(
        np.asarray(
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
            dtype=np.int32,
        )
    )
    ee_stencil.from_numpy(
        np.asarray([[12, 13, 14, 15], [16, 17, 18, 19]], dtype=np.int32)
    )
    pt_overlap.from_numpy(
        np.asarray(
            [[0.0, 0.0, 0.0], [0.1, -0.2, 0.3], [0.0, 0.0, 0.0]],
            dtype=np.float64,
        )
    )
    ee_overlap.from_numpy(
        np.asarray([[0.0, 0.0, 0.0], [-0.4, 0.5, 0.0]], dtype=np.float64)
    )

    manager._count_history_entries(
        3, 2, pt_overlap, ee_overlap, entry_count
    )
    assert entry_count[None] == 2

    manager._clear_history(16, history_state, history_overflow)
    manager._save_history(
        3,
        2,
        pt_stencil,
        ee_stencil,
        pt_overlap,
        ee_overlap,
        16,
        history_state,
        history_kind,
        history_stencil,
        history_overlap,
        history_overflow,
    )
    assert history_overflow[None] == 0
    assert np.count_nonzero(history_state.to_numpy() == 0) == 2

    loaded_pt = ti.Vector.field(3, dtype=ti.f64, shape=3)
    loaded_ee = ti.Vector.field(3, dtype=ti.f64, shape=2)
    manager._load_history(
        3,
        2,
        pt_stencil,
        ee_stencil,
        loaded_pt,
        loaded_ee,
        16,
        history_state,
        history_kind,
        history_stencil,
        history_overlap,
    )
    np.testing.assert_allclose(loaded_pt.to_numpy(), pt_overlap.to_numpy())
    np.testing.assert_allclose(loaded_ee.to_numpy(), ee_overlap.to_numpy())

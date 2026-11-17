import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from src.fedem.Patch import FEMSurfacePatch
from src.fempm.Patch import FEMPMSurfacePatch


@pytest.fixture(autouse=True)
def taichi_cpu_runtime():
    ti.reset()
    ti.init(
        arch=ti.cpu,
        default_fp=ti.f64,
        cpu_max_num_threads=1,
        offline_cache=False,
    )
    yield
    ti.reset()


@pytest.mark.parametrize(
    "patch_type",
    (FEMSurfacePatch, FEMPMSurfacePatch),
)
def test_zero_offset_patch_references_authoritative_fem_positions(patch_type):
    positions = ti.Vector.field(3, dtype=ti.f64, shape=5)
    values = np.zeros((5, 3), dtype=np.float64)
    values[1] = (1.0, 0.0, 0.0)
    values[2] = (0.0, 1.0, 0.0)
    positions.from_numpy(values)
    patch = patch_type(
        5,
        np.array([[0, 1, 2]], dtype=np.int32),
        np.array([0], dtype=np.int32),
    )

    patch.update(positions)
    patch.commit_search_positions()

    assert patch.nodes is positions
    assert patch.offset_nodes is None
    assert not hasattr(patch, "old_nodes")
    assert patch.search_positions.shape[0] == 3

    values[0, 2] = 0.25
    positions.from_numpy(values)
    patch.update(positions)
    assert float(patch.maximum_displacement[None]) == pytest.approx(0.25)


@pytest.mark.parametrize(
    "patch_type",
    (FEMSurfacePatch, FEMPMSurfacePatch),
)
def test_nonzero_offset_allocates_separate_contact_coordinates(patch_type):
    positions = ti.Vector.field(3, dtype=ti.f64, shape=3)
    positions.from_numpy(
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
    )
    patch = patch_type(
        3,
        np.array([[0, 1, 2]], dtype=np.int32),
        np.array([0], dtype=np.int32),
        {"Value": 0.1},
    )

    patch.update(positions)

    assert patch.nodes is patch.offset_nodes
    assert patch.nodes is not positions

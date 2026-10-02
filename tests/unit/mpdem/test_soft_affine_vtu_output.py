from pathlib import Path

import numpy as np
import pyvista as pv

from src.mpdem.engines.SoftAffineIPCOperator import write_soft_affine_surface_vtu


def test_write_soft_affine_surface_vtu_round_trip(tmp_path: Path):
    snapshot = {
        "vertices": np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        ),
        "faces": np.asarray([[0, 1, 2]], dtype=np.int32),
        "bodyID": np.asarray([4, 4, 4], dtype=np.int32),
        "groupID": np.asarray([7, 7, 7], dtype=np.int32),
        "faceBodyID": np.asarray([4], dtype=np.int32),
        "faceGroupID": np.asarray([7], dtype=np.int32),
        "faceMaterialID": np.asarray([2], dtype=np.int32),
    }
    output = tmp_path / "GraphicAffineBody000000"

    assert write_soft_affine_surface_vtu(output, snapshot)
    mesh = pv.read(output.with_suffix(".vtu"))

    assert mesh.n_points == 3
    assert mesh.n_cells == 1
    np.testing.assert_array_equal(mesh.point_data["bodyID"], [4, 4, 4])
    np.testing.assert_array_equal(mesh.point_data["groupID"], [7, 7, 7])
    np.testing.assert_array_equal(mesh.cell_data["bodyID"], [4])
    np.testing.assert_array_equal(mesh.cell_data["groupID"], [7])
    np.testing.assert_array_equal(mesh.cell_data["materialID"], [2])

import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from src.fem import FEM
from src.fem.engines.FEMSolver import _volume_weighted_cell_to_node


def test_volume_weighted_cell_tensor_projection():
    connectivity = np.asarray(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32
    )
    reference_weights = np.asarray([[1.0], [3.0]], dtype=np.float64)
    stress = np.zeros((2, 3, 3), dtype=np.float64)
    stress[0, 0, 0] = 2.0
    stress[1, 0, 0] = 10.0

    nodal = _volume_weighted_cell_to_node(
        stress, connectivity, reference_weights, node_count=5
    )

    assert nodal.shape == (5, 3, 3)
    assert nodal[0, 0, 0] == 2.0
    np.testing.assert_allclose(nodal[[1, 2, 3], 0, 0], 8.0)
    assert nodal[4, 0, 0] == 10.0


def test_fem_vtu_writes_only_nodal_von_mises(tmp_path):
    import meshio

    ti.init(
        arch=ti.cpu,
        default_fp=ti.f64,
        cpu_max_num_threads=1,
        offline_cache=False,
    )
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Explicit", log=False)
    mesh = fem.add_mesh(
        {
            "Geometry": "Box",
            "Size": (1.0, 1.0, 1.0),
            "Divisions": (1, 1, 1),
            "ElementType": "TET4",
        }
    )
    fem.add_material(
        "StVK",
        young_modulus=1.0e4,
        poisson_ratio=0.3,
        density=1000.0,
    )
    fem.set_solver(step=0, log=False)
    engine = fem.build()
    output = tmp_path / "nodal_stress.vtu"

    engine.record(output, log=False)

    archived = meshio.read(output)
    assert archived.point_data["von_mises"].shape == (
        mesh.number_of_nodes,
    )
    assert "von_mises" not in archived.cell_data_dict
    assert "strain_energy" not in archived.cell_data_dict

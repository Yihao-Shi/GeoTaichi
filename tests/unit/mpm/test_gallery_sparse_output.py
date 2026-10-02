from types import SimpleNamespace

import numpy as np
import pytest

from src.mpm.Recorder import WriteFile, build_sparse_block_mesh
from src.mpm.Simulation import Simulation
from src.mpm.mainMPM import MPM


def test_build_sparse_block_mesh_2d_maps_physical_ids_and_clips_boundary_blocks():
    mesh = build_sparse_block_mesh(
        dimension=2,
        active_block_ids=[0, 5],
        block_count=[3, 2],
        block_size=4,
        grid_size=[0.5, 0.25],
        gnum=[9, 7],
    )

    np.testing.assert_array_equal(mesh["active_block_id"], [0, 5])
    np.testing.assert_array_equal(mesh["compact_block_id"], [0, 1])
    np.testing.assert_array_equal(mesh["block_index"], [[0, 0], [2, 1]])
    np.testing.assert_array_equal(mesh["connectivity"], [[0, 1, 2, 3], [4, 5, 6, 7]])
    np.testing.assert_allclose(mesh["coords"][:4], [[0.0, 0.0], [1.75, 0.0], [1.75, 0.875], [0.0, 0.875]])
    np.testing.assert_allclose(mesh["coords"][4:], [[3.75, 0.875], [4.0, 0.875], [4.0, 1.5], [3.75, 1.5]])


def test_build_sparse_block_mesh_3d_uses_vtk_hexahedron_vertex_order():
    mesh = build_sparse_block_mesh(
        dimension=3,
        active_block_ids=[7],
        block_count=[2, 2, 2],
        block_size=4,
        grid_size=[0.1, 0.2, 0.3],
        gnum=[7, 7, 7],
    )

    np.testing.assert_array_equal(mesh["block_index"], [[1, 1, 1]])
    np.testing.assert_array_equal(mesh["connectivity"], [[0, 1, 2, 3, 4, 5, 6, 7]])
    np.testing.assert_allclose(mesh["coords"][0], [0.35, 0.7, 1.05])
    np.testing.assert_allclose(mesh["coords"][6], [0.6, 1.2, 1.8])


def test_build_sparse_block_mesh_rejects_out_of_range_physical_id():
    with pytest.raises(ValueError, match="outside the physical block grid"):
        build_sparse_block_mesh(2, [4], [2, 2], 4, [0.1, 0.1], [8, 8])


def test_sparse_block_mesh_writes_reviewable_vtu_cell_fields(tmp_path):
    mesh = build_sparse_block_mesh(2, [0, 3], [2, 2], 4, [0.1, 0.1], [8, 8])
    writer = WriteFile.__new__(WriteFile)
    writer.vtk_path = str(tmp_path)
    writer.VisualizeSparseGrid(SimpleNamespace(dimension=2, current_print=3), mesh)

    output = tmp_path / "GraphicMPMGrid000003.vtu"
    assert output.is_file()
    header = output.read_bytes()[:4096]
    assert b'Name="active_block_id"' in header
    assert b'Name="compact_block_id"' in header
    assert b'Name="block_index"' in header


def _bare_sparse_simulation():
    sims = Simulation.__new__(Simulation)
    sims.sparse_grid = False
    sims.sparse_grid_backend = "BlockScan"
    sims.sparse_grid_block_size = 4
    sims.sparse_grid_capacity_factor = 1.25
    sims.sparse_grid_max_blocks = 0
    sims.sparse_grid_visualize_active_blocks = False
    return sims


def test_sparse_active_block_vtk_is_opt_in():
    legacy = _bare_sparse_simulation()
    legacy.set_sparse_grid({"Enabled": True, "Backend": "BlockScan"})
    assert legacy.sparse_grid_visualize_active_blocks is False

    gallery = _bare_sparse_simulation()
    gallery.set_sparse_grid(
        {
            "Enabled": True,
            "Backend": "BlockScan",
            "BlockSize": 8,
            "MaxActiveBlocks": 32,
            "VisualizeActiveBlocks": True,
        }
    )
    assert gallery.sparse_grid is True
    assert gallery.sparse_grid_block_size == 8
    assert gallery.sparse_grid_max_blocks == 32
    assert gallery.sparse_grid_visualize_active_blocks is True

    gallery.set_sparse_grid(False)
    assert gallery.sparse_grid is False
    assert gallery.sparse_grid_visualize_active_blocks is False


class _SemiImplicitParameterStub:
    def __init__(self):
        self.solver_type = "SemiImplicit_u_p"
        self.multilevel = 4
        self.pre_and_post_smoothing = 2
        self.bottom_smoothing = 10
        self.values = {}
        self.validated = False

    def set_residual_tolerance(self, value):
        self.values["residual_tolerance"] = value

    def set_pressure_parameter(self, value):
        self.values["pressure_beta"] = value

    def set_max_iteration(self, value):
        self.values["max_iteration_number"] = value

    def set_linear_solver(self, value):
        self.values["linear_solver"] = value

    def set_assemble_type(self, value):
        self.values["assemble_type"] = value

    def set_pressure_solver(self, value):
        self.values["pressure_solver"] = value

    def use_mgpcg_pressure_solver(self):
        return False

    def validate_configuration(self):
        self.validated = True


def test_semiimplicit_up_accepts_public_pressure_solver_parameters():
    mpm = MPM.__new__(MPM)
    mpm.sims = _SemiImplicitParameterStub()

    mpm.set_semi_implicit_solver_parameters(
        {
            "assemble_type": "MatrixFree",
            "linear_solver": "PCG",
            "pressure_solver": "PCG",
            "max_iteration_number": 123,
            "residual_tolerance": 2.0e-7,
            "pressure_beta": 0.8,
        }
    )

    assert mpm.sims.values == {
        "residual_tolerance": 2.0e-7,
        "pressure_beta": 0.8,
        "max_iteration_number": 123,
        "linear_solver": "PCG",
        "assemble_type": "MatrixFree",
        "pressure_solver": "PCG",
    }
    assert mpm.sims.validated is True

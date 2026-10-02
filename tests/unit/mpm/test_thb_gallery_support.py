from types import SimpleNamespace

import numpy as np

from src.mpm.Recorder import WriteFile
from src.mpm.elements.ElementNodesTHB_Generate import ElemNodesGen
from src.mpm.elements.QuadrilateralElement4Nodes import QuadrilateralElement4Nodes


def test_thb_generator_uses_problem_sized_storage():
    node_count, element_count, elements, nodes = ElemNodesGen(
        1.0,
        np.asarray([[0.0, 4.0], [0.0, 3.0]], dtype=np.float64),
        1,
        [[[1.0, 3.0], [1.0, 3.0]]],
    )

    assert node_count == 36
    assert element_count == 24
    assert len(elements) == 49
    assert len(nodes) == 81
    assert max(elements[index].nbInfNode for index in range(1, element_count + 1)) <= 32


def test_thb_boundary_selection_uses_refined_node_coordinates():
    element = object.__new__(QuadrilateralElement4Nodes)
    element.Nlevel = np.zeros((5, 2), dtype=np.int32)
    element.nodal_coords = np.asarray(
        [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0], [0.5, 0.5], [1.0, 0.5]],
        dtype=np.float64,
    )

    selected = element.get_boundary_nodes([0.0, 0.0], [0.0, 1.0])

    assert selected.tolist() == [0, 1, 2]


def test_regular_boundary_selection_keeps_structured_numbering():
    element = object.__new__(QuadrilateralElement4Nodes)
    element.igrid_size = np.asarray([2.0, 2.0], dtype=np.float64)
    element.gnum = np.asarray([5, 5], dtype=np.int32)

    selected = element.get_boundary_nodes([0.0, 0.0], [0.0, 1.0])

    assert selected.tolist() == [0, 5, 10]


def test_thb_grid_writer_emits_gallery_fields(tmp_path):
    writer = object.__new__(WriteFile)
    writer.vtk_path = str(tmp_path)
    writer.grid_path = str(tmp_path)
    sims = SimpleNamespace(
        current_print=7,
        current_time=0.25,
        dimension=2,
        visualize=True,
    )
    element = SimpleNamespace(
        gnum=np.asarray([3, 2], dtype=np.int32),
        nodal_coords=np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
            dtype=np.float64,
        ),
        Nlevel=np.asarray([[0, 0], [0, 0], [1, 1], [1, 0]], dtype=np.int32),
        Ntype=np.asarray([[1, 1], [2, 1], [3, 3], [5, 4]], dtype=np.int32),
    )

    writer.MonitorTHBGrid(sims, SimpleNamespace(element=element))

    vtk_file = tmp_path / "GraphicMPMGrid000007.vtu"
    assert vtk_file.is_file()
    header = vtk_file.read_bytes().split(b"<AppendedData", 1)[0]
    for field in (
        b"grid_level",
        b"thb_level_x",
        b"thb_level_y",
        b"thb_type_x",
        b"thb_type_y",
    ):
        assert field in header
    archive = np.load(tmp_path / "MPMGrid000007.npz")
    assert archive["grid_level"].tolist() == [0, 0, 1, 1]

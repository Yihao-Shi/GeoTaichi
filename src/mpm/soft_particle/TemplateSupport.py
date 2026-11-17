from dataclasses import dataclass
import numpy as np

HEXAHEDRON = 0
TETRAHEDRON = 1


@dataclass(frozen=True)
class SoftTemplateSupport:
    grid_type: int
    point_node: np.ndarray
    point_shape: np.ndarray
    point_dshape: np.ndarray
    point_count: np.ndarray
    surface_node: np.ndarray
    surface_shape: np.ndarray
    surface_count: np.ndarray
    sdf_node: np.ndarray
    sdf_shape: np.ndarray
    sdf_count: np.ndarray
    grid_origin: np.ndarray
    grid_shape: np.ndarray
    grid_space: float
    grid_base_space: float = 0.0

    @property
    def material_point_number(self):
        return int(self.point_count.shape[0])

    @property
    def surface_node_number(self):
        return int(self.surface_count.shape[0])

    @property
    def levelset_node_number(self):
        return int(self.sdf_count.shape[0])


_CUBE_VERTEX_OFFSETS = np.asarray(
    (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ),
    dtype=np.int64,
)

_CELL_TETRAHEDRA = np.asarray(
    (
        (0, 1, 3, 7),
        (0, 1, 5, 7),
        (0, 2, 3, 7),
        (0, 2, 6, 7),
        (0, 4, 5, 7),
        (0, 4, 6, 7),
    ),
    dtype=np.int64,
)

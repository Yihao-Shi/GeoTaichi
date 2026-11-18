import numpy as np
from src.sdf.LevelSetGrid import LocalGrid
import trimesh as tm
from src.sdf.MeshBased import MeshBased


class arbitrarily(MeshBased):
    def __init__(self, start_point, region_size, distance_field, node_coordinate, vertices, faces) -> None:
        super().__init__()
        self.is_simple_shape = 1
        self.mesh = tm.Trimesh(vertices=vertices, faces=faces)
        self.prox = tm.proximity.ProximityQuery(self.mesh)
        self.grid = LocalGrid()
        self.grid.set_distance_field(
            np.array(start_point),
            np.array(region_size),
            np.array(node_coordinate),
            np.array(distance_field),
            self.mesh.center_mass,
        )

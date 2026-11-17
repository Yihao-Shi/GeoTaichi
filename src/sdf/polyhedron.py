import trimesh as tm
from src.sdf.MeshBased import MeshBased


class polyhedron(MeshBased):
    def __init__(self, file: str) -> None:
        super().__init__()
        self.is_simple_shape = 1
        self.mesh: tm.Trimesh = tm.load(file, force="mesh")
        self.prox = tm.proximity.ProximityQuery(self.mesh)

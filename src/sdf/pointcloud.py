import numpy as np
import trimesh as tm
from src.sdf.BasicShape import BasicShape


class pointcloud(BasicShape):
    def __init__(self, points) -> None:
        super().__init__(ray=False)
        meshs = self.construct_surface_mesh_from_point_cloud(points, estimate_normals=True)
        self.mesh = tm.Trimesh(vertices=np.asarray(meshs.vertices), faces=np.asarray(meshs.triangles))
        self.prox = tm.proximity.ProximityQuery(self.mesh)

    def generate(self, *args, **kwargs):
        self.generate_sdf()
        self.mesh_operation()

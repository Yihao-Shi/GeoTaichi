from src.sdf.BasicShape import BasicShape


class MeshBased(BasicShape):
    def __init__(self):
        super().__init__(ray=False)

    def _estimate_bounding_box(self):
        self._get_bounding_box()

    def side(self, p):
        p = self.transfer(p)
        return self.mesh.contains(p)

    def generate(self, *args, **kwargs):
        self.mesh_operation()

    def _normal(self, p):
        pass

from src.iga.generator.Primitives import Primitives
from src.nurbs.BasicSurface import Rectangle, Ring
from src.nurbs.BasicVolume import Cube, Tube, Cylinder


class IGAGenerateManager:
    def create_primitives(self):
        return Primitives()


__all__ = [
    "IGAGenerateManager",
    "Primitives",
    "Cube",
    "Tube",
    "Cylinder",
    "Rectangle",
    "Ring",
]

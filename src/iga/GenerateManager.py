from src.iga.generator.Primitives import Primitives
from src.nurbs.Rectangle import Rectangle
from src.nurbs.Ring import Ring
from src.nurbs.Cube import Cube
from src.nurbs.Cylinder import Cylinder
from src.nurbs.Tube import Tube


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

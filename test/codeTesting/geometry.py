import taichi as ti
ti.init()

from src.utils.GeometryFunction import *
from src.utils.TypeDefination import vec3f


@ti.dataclass
class Triangle:
    vertice1: vec3f
    vertice2: vec3f
    vertice3: vec3f


point = vec3f(-1, 0, 12)
patch = Triangle.field(shape=1)
patch[0].vertice1 = vec3f(0, 0, 0)
patch[0].vertice2 = vec3f(3, 0, 0)
patch[0].vertice2 = vec3f(0, 6, 0)

@ti.kernel
def execute():
    print(DistanceFromPointToTriangle2(point, patch[0].vertice1, patch[0].vertice2, patch[0].vertice3))


execute()
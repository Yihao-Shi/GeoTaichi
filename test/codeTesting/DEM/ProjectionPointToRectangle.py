import taichi as ti
ti.init(arch=ti.gpu)

from src.utils.GeometryFunction import PointProjectionToRectangle

center = ti.Vector([6, 3, 5])
rcenter = ti.Vector([6, 7, 9])
norm = ti.Vector([1, 0, 1])
length = 3
width = 3
height = 3

@ti.kernel
def launch(center: ti.types.vector(3, float), 
           rcenter: ti.types.vector(3, float), 
           norm: ti.types.vector(3, float), 
           length: float, width: float, height: float):
    print(PointProjectionToRectangle(center, rcenter, norm, length, width, height))

launch(center, rcenter, norm, length, width, height)

import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f64, debug=False, device_memory_fraction=0.75)

from src.mpm.BaseStruct import *   
from src.mpm.elements.HexahedronElement8Nodes import *

n = ti.field(int, shape=())
n[None] = 1
p = ParticleCloud.field(shape=n[None])
from random import random

@ti.kernel
def play():
    for i in range(n[None]):
        p[i].x = [20.75, 20.75, 20.75]
        p[i].psize=[0., 0., 0.]
play()

x=ti.Vector.field(3, float, shape=41*41*41)
for i in range(41):
    for j in range(41):
        for k in range(41):
            
            nodeid=i+j*41+k*41*41
            x[nodeid]=[i*1, j*1, k*1]

ele = HexahedronElement8Nodes()
ele.gnum = ti.Vector([41, 41, 41])
ele.gridSum = int(41*41*41)
ele.grid_size = ti.Vector([1., 1., 1.])
ele.igrid_size = 1. / ele.grid_size
ele.nodal_coords = x
ele.element_initialize("Linear", None, n[None], local_coordiates=True)

from time import time
a=time()
for _ in range(1):
    ele.calculate(n, p)
b=time()
print(b-a)
print(ele.shape_fn[4], ele.dshape_fn[4])

elem = HexahedronElement8Nodes()
elem.gnum = ti.Vector([41, 41, 41])
elem.gridSum = int(41*41*41)
elem.grid_size = ti.Vector([1., 1., 1.])
elem.igrid_size = 1. / elem.grid_size
elem.nodal_coords = x
elem.element_initialize("Linear", None, n[None])

a=time()
for _ in range(1):
    elem.calculate(n, p)
b=time()
print(b-a)                     
print(ele.shape_fn[4], ele.dshape_fn[4])
print(elem.shape_fn[4], elem.dshape_fn[4])

'''
for i in range(27):
    print(ele.shape_fn[i], elem.shape_fn[i])
    print(ele.dshape_fn[i], elem.dshape_fn[i])
    print(ele.dshape_fnc[i], elem.dshape_fnc[i])
'''


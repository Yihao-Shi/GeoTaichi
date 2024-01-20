import sys
sys.path.append('/home/eleven/work/GeoTaichi')

import taichi as ti
from src.utils.MatrixFree.MatrixFreePCG import MatrixFreePCG 
from src.utils.MatrixFree.LinearOperator import LinearOperator

ti.init(arch=ti.cpu, default_fp=ti.f64)

operator = MatrixFreePCG(64)
n = 64
x = ti.field(dtype=float, shape=(n,))
b = ti.field(dtype=float, shape=(n,))
M = ti.field(dtype=float, shape=(n,))

@ti.kernel
def init():
    for i in ti.grouped(b):
        x[i] = 0.0
    b[0] = 4
    b[1] = 2
    b[2] = 5
    M[0] = 1
    M[1] = 1
    M[2] = 1

@ti.kernel
def compute(v: ti.template(), mv: ti.template()):
    mv[0] = 16*v[0]+4*v[1]+8*v[2]
    mv[1] = 4*v[0]+5*v[1]-4*v[2]
    mv[2] = 8*v[0]-4*v[1]+22*v[2]
    

init()        
A = LinearOperator(compute)
print(b)
print(operator.solve(A, b, x, M, 3, maxiter=10*8, tol=1e-18))
print(x)
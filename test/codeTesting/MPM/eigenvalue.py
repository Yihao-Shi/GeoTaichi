import taichi as ti
ti.init(arch=ti.cpu, default_fp=ti.f64, default_ip=ti.i32, debug=False)

from src.utils.MatrixFunction import get_eigenvalue_3x3

a = ti.Matrix([[2,1,1],[0,2,0],[0,-1,1]])
@ti.kernel
def k(a:ti.types.matrix(3, 3, float)):
    q=get_eigenvalue_3x3(a)
    print(q, ti.sym_eig(a))
    
k(a)

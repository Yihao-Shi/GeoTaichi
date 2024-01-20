import sys
sys.path.append('/home/eleven/work/GeoTaichi')

import taichi as ti
ti.init(arch=ti.cpu, default_fp=ti.f64, default_ip=ti.i32, debug=False)

from src.utils.ScalarFunction import morton3d32


@ti.kernel
def k():
    print(morton3d32(3, 3, 3))
    
k()

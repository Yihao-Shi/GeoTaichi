import taichi as ti
import sys
ti.init(arch=ti.gpu, device_memory_GB=6, kernel_profiler=True)
from time import time

n = int(2<<27)
a = ti.field(int, n)
b = ti.field(int, n)
c = ti.field(int, n)
d = ti.field(int, n)
f = ti.field(int, ())

@ti.kernel
def setup(a: ti.template(), b: ti.template(), c: ti.template()):
    for i in range(n):
        a[i] = i
        b[i] = i
        if ti.random() > 0.5:
            c[i] = 1
            
setup(a, b, c)

@ti.kernel
def copy1(a: ti.template(), b: ti.template(), c: ti.template(), f: ti.template()):
    f[None] = 0
    for i in range(n):
        if c[i] == 1:
            offset = ti.atomic_add(f[None], 1)
            b[offset] = a[i]
    #ti.sync()
            
@ti.kernel
def copy2(a: ti.template(), c: ti.template(), f: ti.template()):
    f[None] = 0
    ti.loop_config(serialize=True)
    for i in range(n):
        if c[i] == 1:
            offset = ti.atomic_add(f[None], 1)
            a[offset] = a[i]
    #ti.sync()
            
copy1(a, d, c, f)
copy2(b, c, f) 
setup(a, b, c)
         
toc1 = time()
copy1(a, d, c, f)
tec1 = time()

toc2 = time()
copy2(b, c, f)
tec2 = time()
print(tec1-toc1, tec2-toc2)
#ti.profiler.print_kernel_profiler_info('trace')

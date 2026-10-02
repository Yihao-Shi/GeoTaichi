import taichi as ti
ti.init(arch=ti.gpu)

n = 12345678
a = ti.field(int, ())
b = ti.field(int, n)
c = ti.field(int, n)

@ti.kernel
def k1(a: ti.template()):
    a[None]=0
    for i in range(n):
        x = (ti.log(i) * ti.sin(float(i)) + 2)/ti.exp(-i)
        for j in range(2200):
            ti.atomic_add(a[None], 1)

        
@ti.kernel
def k2(a: ti.template()):
    a.fill(0)
    for i in range(n):
        x = (ti.log(i) * ti.sin(float(i)) + 2)/ti.exp(-i)
        offset=0
        for j in range(2200): offset+=1
        a[i]=offset
        
k1(a)
k2(b)

from time import time
toc1 = time()
k1(a)
ti.sync()
tec1 = time()

toc2 = time()
k2(b)
ti.sync()
tec2 = time()
print(tec1-toc1, tec2-toc2)

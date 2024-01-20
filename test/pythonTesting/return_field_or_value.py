import taichi as ti
ti.init(arch=ti.gpu)
from time import time
a = ti.field(int, ())

@ti.kernel
def k1(a: ti.template()):
    for i in range(12345):
        if i%7==0:
            a[None] = 1
            
            
@ti.kernel
def k2() -> int:
    a = 0
    for i in range(12345):
        if i%7==0:
            a = 1      
    return a
    
    
def test1(): 
    toc = time()
    k1(a)
    tec = time()
    print(a[None], tec-toc)
    
def test2():
    toc = time()
    a = k2()
    tec = time()
    print(a, tec-toc)
    
k1(a)
_=k2()
a[None] = 0


test1()
test2()

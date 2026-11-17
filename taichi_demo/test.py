import taichi as ti
ti.init(arch=ti.gpu)

@ti.data_oriented
class A:
    def __init__(self) -> None:
        self.para = ti.field(float, shape=())
        self.para[None] = 3.295842

a = A()

@ti.data_oriented
class B:
    def __init__(self, a: A) -> None:
        self.a = a

    @ti.func
    def f(self):
        pass
        

    @ti.kernel
    def k(self):
        ti.loop_config(serialize=True)
        for _ in range(100000000):
            '''p = self.a.para[None]
            p+=ti.exp(p)/p
            self.a.para[None] = p'''
            self.a.para[None] += ti.exp(self.a.para[None])/self.a.para[None]

import time as t
b = B(a)
#b.k()
start = t.time()
b.k()
end = t.time()
print(end-start)
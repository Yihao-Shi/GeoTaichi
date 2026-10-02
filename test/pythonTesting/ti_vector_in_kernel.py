import taichi as ti
ti.init()

a = ti.Vector([0, 0, 0])
b=0

@ti.kernel
def k(a:ti.template()):
    a=2
    
k(b)
print(b)

import taichi as ti
ti.init()

@ti.dataclass
class A: 
    a: float
    
    @ti.func
    def _add(self, val):
        self.a += val

@ti.kernel
def k1(a: ti.template()):
    for i in range(3125123):
        if i < 310000:
            a[0, 0]._add(1)
        else:
            a[0, 1]._add(2)
    print(a[0, 0].a, a[0, 0].a + 0.5 * a[0, 1].a)

a=A.field(shape=(125432, 2))

@ti.kernel
def k2(a: ti.template()):
    for i in range(3125123):
        if i < 310000:
            a[0]._add(1)
        else:
            a[1]._add(2)
    print(a[0].a, a[0].a + 0.5 * a[1].a)

b=A.field(shape=125432)


@ti.kernel
def k3(a: ti.template()):
    for i in range(11100000):
        a[i%3, 0].a+=1
    print(a[0, 0].a+a[1, 0].a+a[2, 0].a)

c=A.field(shape=(125432, 2))

k1(a)
k2(b)
k3(c)
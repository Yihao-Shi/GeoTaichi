import taichi as ti
ti.init()

@ti.dataclass
class particle:
    active: ti.u8
    position: ti.types.vector(3, float)
    radius: float
    ID: int

p = particle.field(shape=123)

@ti.kernel
def fill(p: ti.template()):
    for i in p:
        p[i].position=[1+ti.random(),1+ti.random(),1+ti.random()]
        p[i].radius=ti.random()
        p[i].ID=i
        if 40<i<60:
            p[i].active=ti.u8(1)

fill(p)
for i in range(p.shape[0]):
    print(p[i].ID)


@ti.kernel
def restore(p: ti.template()):
    ti.loop_config(serialize=True)
    remain_particle=0
    for i in p:
        if int(p[i].active)==1:
            p[remain_particle]=p[i]
            remain_particle+=1

restore(p)
for i in range(p.shape[0]):
    print(p[i].ID)

import taichi as ti
import sys
ti.init()

'''
@ti.dataclass
class Particle:
    r: ti.math.vec3
    
    @ti.func
    def add(self, vec):
        self.r+= vec

particle = Particle.field(shape=2)

@ti.kernel
def simulate():
    #ti.loop_config(serialize=True)
    for i in range(2):
        for j in range(5):
            for k in range(3):
                print('inside', particle[i].r)
            particle[i].add(ti.math.vec3(1))
            print('outside', particle[i].r)

simulate()

'''

'''
@ti.dataclass
class Material:
    m: ti.f64
 

@ti.dataclass
class Particle:
    mcx: ti.f64
    material: Material
    

    def set_material(self, matptr):
        self.material = matptr
        

    def calcu_mass(self, vol):
        self.mcx = self.material.m * vol
        print(self.mcx)
particle = Particle.field(shape=20000000)   
material = Material.field(shape=20000000) 
material[0].m=1000
for i in range(20):
    particle[i].set_material(material[0])
    particle[i].calcu_mass(10)     
'''    
     
'''     
@ti.data_oriented
class manager(object):
    def initialize(self):
        self.material = Material.field(shape=3)
        self.particle = Particle.field(shape=24)
        #self.set(self.material, self.particle, 1)
    
    @ti.kernel    
    def set(self, material: ti.template(), particle: ti.template(), materialID: int):
        material[0].density = 1000
        material[1].density = 2000
        material[2].density = 3000
        for i in range(20):
            particle[i].set_material(material[materialID])
            particle[i].calcu_mass(10)

    def get(self):
        for i in range(20):
            print(self.particle[i].m)
        print(sys.getsizeof(id(self.particle.m)), sys.getsizeof(id(self.particle.material)))

m = manager()
m.initialize()
m.get()'''
#ti.profiler.print_memory_profiler_info()


@ti.dataclass
class materials:
    density: float
    
    def print_message(self):
        return None
    
@ti.dataclass
class material:
    density: float
    
    def print_message(self):
        return self.density
    
    
m=material.field(shape=2)
m[0].density=12
m[1].density=120    
 
@ti.dataclass
class particle:
    m: float
    materialID: int
    mat: materials
    vol: float
    
    def set(self, mat):
        self.mat=mat
        print(mat.print_message(), self.mat.print_message())
    
    def cal_mass(self):
        self.m=self.mat.density*self.vol

    def set_vol(self,vol):
        self.vol=vol
    
    def check(self):
        print(self.m, self.vol, self.mat.print_message(), type(self.set))
    
from random import random    
p=particle.field(shape=123) 

for i in range(61): 
    p[i].materialID = 0 
    
for i in range(61, 123): 
    p[i].materialID = 1
    
for i in range(123):
    p[i].set(m[p[i].materialID])

for i in range(123):
    p[i].set_vol(int(random()*100))

for i in range(123):
    p[i].cal_mass()

#for i in range(123):
#    p[i].check()

import taichi as ti
from time import time

ti.init(arch=ti.gpu, device_memory_GB=6)

row = 20000
column = 2000
a = ti.field(int, shape=(row, column))
b = ti.field(int, shape=(column, row))

@ti.kernel
def kerns1(row: int, column: int, vars: ti.template()):
    for i in range(row):
        for j in range(column):
            vars[i, j] = i

@ti.kernel
def kerns2(row: int, column: int, vars: ti.template()):
    for i in range(row):
        for j in range(column):
            vars[j, i] = i

kerns1(row, column, a)
kerns2(row, column, b)

a.fill(0)
b.fill(0)

start = time()
kerns1(row, column, a)
end = time()
print(end-start)

start = time()
kerns2(row, column, b)
end = time()
print(end-start)
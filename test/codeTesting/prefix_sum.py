from src import *

init(arch="gpu", debug=False)

from src.utils.PrefixSum import PrefixSumExecutor

a = ti.field(int, shape=2695)

@ti.kernel
def k(a: ti.template()):
    for i in range(0,2695):
        a[i] = 12695-i
    a[265] = 19625
k(a)        
pse = PrefixSumExecutor(2695)

print(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9])
import time
x = time.time()
pse.run(a, 2695)
y = time.time()
print(y - x)
print(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9])

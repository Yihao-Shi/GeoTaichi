import numba as nb
import numpy as np
from math import acos

@nb.jit(nopython=True)
def myfunc1(a):
    s = 0
    for i in range(100000):
        s += acos(0.5)
        
@nb.jit(nopython=True)
def myfunc2(a):
    s = 0
    for i in range(100000):
        s += acos(0.75)
    
@nb.jit(nopython=True)
def execu(func):
    a = np.zeros(3)
    func(a)
    print(a.shape[0])


import time
tic = time.time()
execu(myfunc2)
toc = time.time()
print(toc-tic)

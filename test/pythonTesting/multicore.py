from pathos.pools import ProcessPool as Pool
from functools import partial
import numpy as np

class B:
    def __init__(self):
        self.array=np.zeros(12) + 2
        
    def calculate(self, i, j):
        return i+j, i

class A:
    b: B
    
    def __init__(self, b):
        self.pool=Pool()
        self.vec=np.zeros(12)
        self.b = b
    
    def func(self, i, j):
        print(i[1], j)
        return 1
    
    def func1(self, array, i):
        return i
    
    def run(self):
        arr=np.zeros(12)+2
        func=partial(self.func)
        self.vec=self.pool.map(func, [range(0, 12),arr], range(0, 12))
    
    def run1(self):
        arr=np.zeros(12)
        func1=partial(self.func1, arr)
        returnVal2=self.pool.map(func1, range(0, 12))
        

b=B()
a=A(b)
a.run()
a.run1()
#print(np.sum(np.array(a.vec), axis=0))

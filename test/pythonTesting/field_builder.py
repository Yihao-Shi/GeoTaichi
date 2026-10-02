import taichi as ti
ti.init()

@ti.data_oriented
class B(object):
    def __init__(self):
        self.x=ti.field(int)
        fb=ti.FieldsBuilder()
        fb.dense(ti.i,1).place(self.x)
        self.snode_tree=fb.finalize()
        
    def clear(self):
        self.snode_tree.destroy()

@ti.data_oriented
class A(object):
    def __init__(self):
        self.n=12345
        
    def init(self):
        self.b=B()
        self.x=ti.field(int)
        fb=ti.FieldsBuilder()
        fb.dense(ti.i,self.n).place(self.x)
        self.snode_tree=fb.finalize()
        
    def clear(self):
        self.snode_tree.destroy()
        self.b.clear()
        del self.b
        
    @ti.kernel
    def k(self, m: int):
        for i in range(self.n):
            self.x[i] = m * i
            self.b.x[0] += m
            
    def start(self):
        self.init()
        self.k(1)
        print(self.x[34], self.b.x[0]) # -> output: 34 12345 // expected output: 34 12345
        self.clear()
        del self.x
        
        self.init()
        self.k(2)
        print(self.x[34], self.b.x[0]) # -> output: 0 24690 // expected output: 68 24690
        self.clear()
        del self.x
        
a=A()
a.start()
'''
class B(object):
    def __init__(self):
        self.x=ti.field(int)
        fb=ti.FieldsBuilder()
        fb.dense(ti.i,1).place(self.x)
        self.snode_tree=fb.finalize()
        
    def clear(self):
        self.snode_tree.destroy()
        del self.snode_tree, self.x


class A(object):
    def __init__(self):
        self.n=12345
        
    def init(self):
        #self.b=B()
        self.x=ti.field(int)
        ti.root.dense(ti.i,self.n).place(self.x)
        self.y=ti.field(int)
        ti.root.dense(ti.i,self.n).place(self.y)
        
        
    def clear(self):
        #self.snode_tree1.destroy()
        #self.snode_tree2.destroy()
        del self.x, self.y
        
    
    def start(self, i):
        self.init()
        k(i, self.n, self.x, self.y)
        print(i, self.x[12]) # -> output: 34 12345 // expected output: 34 12345
        print(i, self.y[12])
        self.clear()
        ti.profiler.memory_profiler.print_memory_profiler_info()
        
        
@ti.kernel
def k(m: int, n:int, x:ti.template(), y: ti.template()):
    for i in range(n):
        x[i] = m * i  
        y[i] = m + i
a=A()
i=1
while i < 200:
    a.start(i)
    i+=1
i=1
while i < 200000:
    print(i)
    i+=1
i=1
while i < 200:
    a.start(i)
    i+=1'''

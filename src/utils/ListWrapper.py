import taichi as ti


@ti.data_oriented
class ListWrapper:
    def __init__(self):
        self.lst = []

    def append(self, element):
        self.lst.append(element)

    def insert(self, loc, var):
        self.lst.insert(loc, var)

    def extend(self, iterable):
        self.lst.extend(iterable)

    def pop(self, loc=-1):
        self.lst.pop(loc)

    def remove(self, var):
        self.lst.remove(var)

    def clear(self):
        self.lst.clear()

    def reverse(self):
        self.lst.reverse()

    def copy(self):
        return self.lst.copy()
    
    def size(self):
        return len(self.lst)
    
    def max(self):
        return max(self.lst)
    
    def min(self):
        return min(self.lst)
    
    def sum(self):
        return sum(self.lst)
    
    def get_shape(self, lst):
        if isinstance(lst, list):
            if len(lst) == 0:
                return [0]
            return (len(lst), *self.get_shape(lst[0]))
        return []
    
    @property
    def shape(self):
        return self.get_shape(self.lst)
    
    def get_ndim(self, lst):
        if isinstance(lst, list) and lst:
            return 1 + self.get_ndim(lst[0])
        return 0
    
    @property
    def ndim(self):
        return self.get_ndim(self.lst)
    
    def __call__(self):
        return self
    
    def __getitem__(self, index):
        return self.lst[index]
    
    def __setitem__(self, index, value):
        self.lst[index] = value
    
    def __add__(self, list):
        self.lst += list
        return self
    
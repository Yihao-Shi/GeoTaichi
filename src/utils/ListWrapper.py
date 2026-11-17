import taichi as ti


@ti.data_oriented
class ListWrapper:
    """Material slots with a weak-referenceable identity for Taichi templates."""

    def __init__(self):
        self.lst = []

    def size(self):
        return len(self.lst)

    @property
    def shape(self):
        return (len(self.lst),)

    def __getitem__(self, index):
        return self.lst[index]

    def __setitem__(self, index, value):
        self.lst[index] = value

    def __add__(self, values):
        self.lst += values
        return self

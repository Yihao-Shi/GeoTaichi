import taichi as ti
import numpy as np
from itertools import chain


@ti.data_oriented
class DirichletBoundary:
    def __init__(self):
        self.np_dofID = []
        self.np_dofVal = []
        self.dof = None
        self.value = None
        self.num = 0

    def append(self, dof_id, dof_val):
        self.np_dofID.append(list(chain.from_iterable(dof_id)))
        self.np_dofVal.append(dof_val)
        self.num += len(dof_val)

    def finalize(self, total_dof):
        pairs = list(zip(chain.from_iterable(self.np_dofID), chain.from_iterable(self.np_dofVal)))
        pairs = np.array(pairs, dtype=[("id", np.int32), ("val", np.float64)])
        pairs.sort(order="id")
        self.node = ti.field(ti.i32, shape=total_dof)
        self.value = ti.field(ti.f64, shape=total_dof)
        self.set_constraints(np.ascontiguousarray(pairs["id"]), np.ascontiguousarray(pairs["val"]))

    @ti.kernel
    def set_constraints(self, node: ti.types.ndarray(), value: ti.types.ndarray()):
        for i in range(node.shape[0]):
            dofID = node[i]
            self.node[dofID] = 1
            self.value[dofID] = value[i]


@ti.data_oriented
class NeumannBoundary:
    def __init__(self):
        self.np_dofID = []
        self.np_dofVal = []
        self.dof = None
        self.value = None
        self.num = 0

    def append(self, dof_id, dof_val):
        self.np_dofID.append(list(chain.from_iterable(dof_id)))
        self.np_dofVal.append(dof_val)
        self.num += len(dof_val)

    def finalize(self):
        pairs = list(zip(chain.from_iterable(self.np_dofID), chain.from_iterable(self.np_dofVal)))
        pairs = np.array(pairs, dtype=[("id", np.int32), ("val", np.float64)])
        pairs.sort(order="id")
        self.node = ti.field(ti.i32, shape=self.num)
        self.value = ti.field(ti.f64, shape=self.num)
        self.node.from_numpy(pairs["id"])
        self.value.from_numpy(pairs["val"])

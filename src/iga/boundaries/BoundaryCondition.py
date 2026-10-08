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
        self.velocity_dofs = np.empty(0, dtype=np.int32)
        self.velocities = np.empty(0, dtype=np.float64)

    def append(self, dof_id, dof_val):
        self.np_dofID.append(list(chain.from_iterable(dof_id)))
        self.np_dofVal.append(dof_val)
        self.num += len(dof_val)

    def append_velocity(self, dof_id, velocity):
        """Prescribe constant velocity for implicit IGA, including step retries."""
        ids = np.asarray(list(chain.from_iterable(dof_id)))
        values = np.asarray(velocity, dtype=np.float64)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0):
            raise ValueError("velocity constraint DOFs must be nonnegative integers")
        if values.shape != ids.shape or not np.all(np.isfinite(values)):
            raise ValueError("velocity constraints require one finite value per DOF")
        existing = list(chain.from_iterable(self.np_dofID))
        if len(np.unique(ids)) != len(ids) or np.intersect1d(ids, existing).size:
            raise ValueError("velocity constraint DOFs must be unique and unconstrained")
        if np.any(ids > np.iinfo(np.int32).max):
            raise ValueError("velocity constraint DOFs exceed int32 range")
        self.append([ids.tolist()], np.zeros(len(ids)).tolist())
        self.velocity_dofs = np.concatenate((self.velocity_dofs, ids.astype(np.int32)))
        self.velocities = np.concatenate((self.velocities, values))

    def update_timestep(self, timestep):
        if self.velocity_dofs.size:
            if not np.isfinite(timestep) or timestep <= 0:
                raise ValueError("velocity constraints require a positive finite timestep")
            self.set_constraints(self.velocity_dofs, self.velocities * timestep)

    def finalize(self, total_dof):
        if np.any(self.velocity_dofs >= total_dof):
            raise ValueError("velocity constraint DOF exceeds the IGA system size")
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

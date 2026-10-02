"""Persistent device storage for FEM boundary-condition time frames."""

import numpy as np
import taichi as ti


@ti.data_oriented
class DeviceBoundaryData:
    """Boundary topology and values uploaded once before time integration.

    Static boundaries own one frame.  Python callables are sampled during FEM
    construction into compact frame fields, so activating a transient frame is
    a device-to-device operation and never a NumPy/Taichi transfer.
    """

    def __init__(
        self,
        dofs,
        displacement_frames,
        force_frames,
        *,
        real_type,
        numpy_type,
        dirichlet_dynamic=False,
        force_dynamic=False,
    ):
        dofs = np.ascontiguousarray(dofs, dtype=np.int32).reshape(-1)
        displacement_frames = np.ascontiguousarray(displacement_frames, dtype=numpy_type)
        force_frames = np.ascontiguousarray(force_frames, dtype=numpy_type)
        if displacement_frames.ndim != 2:
            raise ValueError("FEM displacement boundary frames must be a matrix")
        if displacement_frames.shape[1] != dofs.size:
            raise ValueError("FEM displacement boundary frame width is invalid")
        if force_frames.ndim != 3 or force_frames.shape[2] != 3:
            raise ValueError("FEM force boundary frames must have shape (frames, nodes, 3)")

        self.constraint_count = int(dofs.size)
        self.node_count = int(force_frames.shape[1])
        self.dirichlet_frame_count = int(displacement_frames.shape[0])
        self.force_frame_count = int(force_frames.shape[0])
        self.dirichlet_dynamic = bool(dirichlet_dynamic)
        self.force_dynamic = bool(force_dynamic)
        if self.dirichlet_frame_count <= 0 or self.force_frame_count <= 0:
            raise ValueError("FEM boundary data requires at least one frame")

        constraint_capacity = max(self.constraint_count, 1)
        self.dofs = ti.field(dtype=ti.i32, shape=constraint_capacity)
        self.displacement = ti.field(
            dtype=real_type,
            shape=(self.dirichlet_frame_count, constraint_capacity),
        )
        self.force = ti.Vector.field(
            3,
            dtype=real_type,
            shape=(self.force_frame_count, max(self.node_count, 1)),
        )

        padded_dofs = np.zeros(constraint_capacity, dtype=np.int32)
        padded_dofs[: self.constraint_count] = dofs
        padded_displacement = np.zeros((self.dirichlet_frame_count, constraint_capacity), dtype=numpy_type)
        padded_displacement[:, : self.constraint_count] = displacement_frames
        padded_force = np.zeros(
            (self.force_frame_count, max(self.node_count, 1), 3),
            dtype=numpy_type,
        )
        padded_force[:, : self.node_count] = force_frames
        self.dofs.from_numpy(padded_dofs)
        self.displacement.from_numpy(padded_displacement)
        self.force.from_numpy(padded_force)

    @ti.kernel
    def initialize_constraints(
        self,
        constrained: ti.template(),
        prescribed_displacement: ti.template(),
    ):
        for dof in constrained:
            constrained[dof] = 0
            prescribed_displacement[dof] = 0.0
        for index in range(self.constraint_count):
            dof = self.dofs[index]
            constrained[dof] = 1
            prescribed_displacement[dof] = self.displacement[0, index]

    @ti.kernel
    def activate_dirichlet(
        self,
        frame: ti.i32,
        prescribed_displacement: ti.template(),
    ):
        for index in range(self.constraint_count):
            prescribed_displacement[self.dofs[index]] = self.displacement[frame, index]

    @ti.kernel
    def activate_force(self, frame: ti.i32, boundary_force: ti.template()):
        for node in range(self.node_count):
            boundary_force[node] = self.force[frame, node]

    def set_dirichlet_frame(self, frame, prescribed_displacement):
        frame = int(frame)
        if not 0 <= frame < self.dirichlet_frame_count:
            raise IndexError(
                f"FEM Dirichlet frame {frame} is outside the preallocated " f"range [0, {self.dirichlet_frame_count})"
            )
        self.activate_dirichlet(frame, prescribed_displacement)

    def set_force_frame(self, frame, boundary_force):
        frame = int(frame)
        if not 0 <= frame < self.force_frame_count:
            raise IndexError(
                f"FEM Neumann frame {frame} is outside the preallocated " f"range [0, {self.force_frame_count})"
            )
        self.activate_force(frame, boundary_force)


__all__ = ["DeviceBoundaryData"]

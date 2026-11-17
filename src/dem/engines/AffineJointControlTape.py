"""Device records of the motor targets used by affine trajectory replay."""

import numpy as np
import taichi as ti


@ti.data_oriented
class AffineJointControlTape:
    def __init__(self, operator, steps):
        self.operator = operator
        shape = (steps, max(operator.joint_num, 1))
        self.targets = ti.field(ti.f64, shape=shape)
        self.gradient = ti.field(ti.f64, shape=shape)
        self.saved_targets = ti.field(ti.f64, shape=max(operator.joint_num, 1))

    @ti.kernel
    def record(self, step: ti.i32):
        for joint in range(self.operator.joint_num):
            self.targets[step, joint] = self.operator.joint_target_angle[joint]

    @ti.kernel
    def load(self, step: ti.i32):
        for joint in range(self.operator.joint_num):
            self.operator.joint_target_angle[joint] = self.targets[step, joint]

    @ti.kernel
    def save_current(self):
        for joint in range(self.operator.joint_num):
            self.saved_targets[joint] = self.operator.joint_target_angle[joint]

    @ti.kernel
    def restore_current(self):
        for joint in range(self.operator.joint_num):
            self.operator.joint_target_angle[joint] = self.saved_targets[joint]

    @ti.kernel
    def accumulate(self, step: ti.i32):
        for joint in range(self.operator.joint_num):
            self.gradient[step, joint] += self.operator.joint_target_vjp[joint]

    def result(self, steps):
        radians = self.gradient.to_numpy()[:steps, : self.operator.joint_num].copy()
        return {
            "joint_target_angle_radians_per_step": radians,
            "joint_target_angle_degrees_per_step": radians * (np.pi / 180.0),
        }

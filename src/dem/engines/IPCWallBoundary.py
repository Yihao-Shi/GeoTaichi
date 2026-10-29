"""Device geometry and transactional motion for existing DEM facet servo walls."""

import numpy as np
import taichi as ti


@ti.data_oriented
class IPCWallBoundary:
    def __init__(self, scene, sims):
        if sims.wall_type != 1:
            raise ValueError("IPC servo boundaries require DEM Polygon/facet walls")
        self.scene, self.sims = scene, sims
        self.wall_count = int(scene.wallNum[0])
        self.servo_count = int(scene.servoNum[0])
        raw = scene.wall.to_numpy()
        if not np.all(raw["active"][: self.wall_count]):
            raise ValueError("Initialize IPC servo boundaries with active facet walls")
        self.callbacks = ()
        if self.wall_count <= 0:
            raise ValueError("IPC facet boundary requires DEM walls")
        self.stiffness_control = sims.servo_type == "StiffnessControl"
        if self.servo_count:
            controls = scene.servo.to_numpy()
            for index in range(self.servo_count):
                start, end = int(controls["startIndex"][index]), int(controls["endIndex"][index])
                if not 0 <= start < end <= self.wall_count:
                    raise ValueError("IPC servo facet range is outside the DEM scene")
                for key in ("target_stress", "max_velocity", "alpha"):
                    if not np.isfinite(controls[key][index]) or controls[key][index] <= 0:
                        raise ValueError(f"IPC servo {key} must be finite and positive")
        size = max(1, self.wall_count)
        self.vertex1 = ti.Vector.field(3, ti.f64, shape=size)
        self.vertex2 = ti.Vector.field(3, ti.f64, shape=size)
        self.vertex3 = ti.Vector.field(3, ti.f64, shape=size)
        self.normal = ti.Vector.field(3, ti.f64, shape=size)
        self.velocity = ti.Vector.field(3, ti.f64, shape=size)
        self.offset = ti.Vector.field(3, ti.f64, shape=size)
        self.start1 = ti.Vector.field(3, ti.f64, shape=size)
        self.start2 = ti.Vector.field(3, ti.f64, shape=size)
        self.start3 = ti.Vector.field(3, ti.f64, shape=size)
        self.abd_force = ti.Vector.field(3, ti.f64, shape=size)
        self.fem_force = ti.Vector.field(3, ti.f64, shape=size)
        self.abd_stiffness = ti.field(ti.f64, shape=size)
        self.fem_stiffness = ti.field(ti.f64, shape=size)
        self.motion_alpha = ti.field(ti.f64, shape=())
        for field, key in (
            (self.vertex1, "vertice1"),
            (self.vertex2, "vertice2"),
            (self.vertex3, "vertice3"),
            (self.normal, "norm"),
            (self.velocity, "v"),
        ):
            field.from_numpy(np.asarray(raw[key][: self.wall_count], dtype=np.float64))
        self.update_servo_device(float(sims.delta), int(sims.servo_status == "On"))

    @ti.kernel
    def begin_step_device(self, dt: ti.f64):
        self.motion_alpha[None] = 1.0
        for wall in range(self.wall_count):
            self.start1[wall] = self.vertex1[wall]
            self.start2[wall] = self.vertex2[wall]
            self.start3[wall] = self.vertex3[wall]
            self.velocity[wall] = self.scene.wall[wall].v
            self.offset[wall] = dt * self.velocity[wall]

    @ti.kernel
    def apply_motion_device(self):
        for wall in range(self.wall_count):
            self.offset[wall] *= self.motion_alpha[None]
            self.vertex1[wall] = self.start1[wall] + self.offset[wall]
            self.vertex2[wall] = self.start2[wall] + self.offset[wall]
            self.vertex3[wall] = self.start3[wall] + self.offset[wall]

    @ti.kernel
    def rollback_device(self):
        for wall in range(self.wall_count):
            self.vertex1[wall] = self.start1[wall]
            self.vertex2[wall] = self.start2[wall]
            self.vertex3[wall] = self.start3[wall]
            self.offset[wall] = ti.Vector.zero(ti.f64, 3)

    @ti.kernel
    def publish_reactions_device(self):
        for wall in range(self.wall_count):
            self.scene.wall[wall].contact_force = self.abd_force[wall] + self.fem_force[wall]
            self.scene.wall[wall].contact_stiffness = self.abd_stiffness[wall] + self.fem_stiffness[wall]

    @ti.kernel
    def update_servo_device(self, dt: ti.f64, enabled: ti.i32):
        for index in range(self.servo_count):
            if self.scene.servo[index].active != 0 and enabled != 0:
                area, force, stiffness = 0.0, ti.Vector.zero(ti.f64, 3), 0.0
                normal = ti.Vector.zero(ti.f64, 3)
                for wall in range(self.scene.servo[index].startIndex, self.scene.servo[index].endIndex):
                    area += (
                        0.5
                        * (self.vertex2[wall] - self.vertex1[wall])
                        .cross(self.vertex3[wall] - self.vertex1[wall])
                        .norm()
                    )
                    normal += self.normal[wall]
                    force += self.abd_force[wall] + self.fem_force[wall]
                    stiffness += self.abd_stiffness[wall] + self.fem_stiffness[wall]
                normal = normal.normalized()
                if self.scene.servo[index].area > 0.0:
                    area = self.scene.servo[index].area
                self.scene.servo[index].area = area
                self.scene.servo[index].current_force = ti.max(0.0, -force.dot(normal))
                if ti.static(self.stiffness_control):
                    if stiffness > 1.0e-12:
                        self.scene.servo[index].gain = self.scene.servo[index].alpha / (dt * stiffness)
                    else:
                        self.scene.servo[index].gain = self.scene.servo[index].max_velocity / (
                            self.scene.servo[index].target_stress * area
                        )
                velocity = self.scene.servo[index].calculate_velocity(self.scene.wall)
                for wall in range(self.scene.servo[index].startIndex, self.scene.servo[index].endIndex):
                    self.velocity[wall] = velocity
                    self.scene.wall[wall].v = velocity

    def commit_servo_step(self, dt):
        self.publish_reactions_device()
        for callback in self.callbacks:
            callback()
        self.update_servo_device(dt, int(self.sims.servo_status == "On"))
        self.commit_geometry_device()

    @ti.kernel
    def commit_geometry_device(self):
        for wall in range(self.wall_count):
            self.scene.wall[wall].vertice1 = self.vertex1[wall]
            self.scene.wall[wall].vertice2 = self.vertex2[wall]
            self.scene.wall[wall].vertice3 = self.vertex3[wall]
            self.scene.wall[wall].translation += self.offset[wall]
            self.scene.wall[wall].verletDisp += self.offset[wall]


def ipc_wall_boundary(scene, sims):
    boundary = getattr(scene, "ipc_wall_boundary", None)
    if boundary is None:
        boundary = IPCWallBoundary(scene, sims)
        scene.ipc_wall_boundary = boundary
    return boundary

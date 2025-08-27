import taichi as ti

from src.utils.linalg import square_norm
from src.utils.VectorFunction import Squared


@ti.data_oriented
class Geomotry:
    def __init__(self):
        self.num = 0
        self.start_index = []
        self.end_index = []
        self.rotate_center = []
        self.velocity = []
        self.angular_velocity = []
        self.translate = []
        self.rotate = []

    def append(self, start_index, end_index, rotate_center, velocity, angular_velocity, wall):
        self.start_index.append(start_index)
        self.end_index.append(end_index)
        self.settings(self.num, rotate_center, velocity, angular_velocity, wall)
        self.num += 1

    def settings(self, index, rotate_center, velocity, angular_velocity, wall):
        is_translate = 1 if square_norm(velocity) else 0
        is_rotate = 1 if square_norm(angular_velocity) else 0
        if index == self.num:
            self.rotate_center.append(rotate_center)
            self.velocity.append(velocity)
            self.angular_velocity.append(angular_velocity)
            self.translate.append(is_translate)
            self.rotate.append(is_rotate)
        else:
            self.rotate_center[index] = rotate_center
            self.velocity[index] = velocity
            self.angular_velocity[index] = angular_velocity
            self.translate[index] = is_translate
            self.rotate[index] = is_rotate
            self.reset(self.start_index[index], self.end_index[index])

        start_index, end_index = self.start_index[index], self.end_index[index]
        if self.translate[index]:
            velocity = self.velocity[index]
            self.translate_initialize(start_index, end_index, velocity, wall)
        if self.rotate[index]:
            rotate_center = self.rotate_center[index]
            angular_velocity = self.angular_velocity[index]
            self.rotate_initialize(start_index, end_index, rotate_center, angular_velocity, wall)

    def modify(self, geometryID, rotate_center=None, velocity=[0., 0., 0.], angular_velocity=[0., 0., 0.], wall=None):
        self.settings(geometryID, rotate_center, velocity, angular_velocity, wall)

    def move(self, geometryID, disp, wall):
        start_index, end_index = self.start_index[geometryID], self.end_index[geometryID]
        self.kernel_move_patch_wall(start_index, end_index, disp, wall)

    def delete(self, geometryID):
        pass

    def go(self, dt, wall):
        for i in range(self.num):
            start_index, end_index = self.start_index[i], self.end_index[i]
            if self.translate[i]:
                velocity = self.velocity[i]
                self.kernel_translate_patch_wall(start_index, end_index, velocity, dt, wall)
            if self.rotate[i]:
                rotate_center = self.rotate_center[i]
                angular_velocity = self.angular_velocity[i]
                self.kernel_rotate_patch_wall(start_index, end_index, rotate_center, angular_velocity, dt, wall)

    @ti.kernel
    def reset(self, start_index: int, end_index: int, wall: ti.template()):
        for nw in range(start_index, end_index):
            wall[nw].v = [0., 0., 0.]

    @ti.kernel
    def translate_initialize(self, start_index: int, end_index: int, velocity: ti.types.vector(3, float), wall: ti.template()):
        for nw in range(start_index, end_index):
            wall[nw].v += velocity

    @ti.kernel
    def rotate_initialize(self, start_index: int, end_index: int, rotate_center: ti.types.vector(3, float), angular_velocity: ti.types.vector(3, float), wall: ti.template()):
        for nw in range(start_index, end_index):
            vec= wall[nw]._get_center() - rotate_center
            omega_dir = angular_velocity.normalized()
            velocity = angular_velocity.cross(vec - vec.dot(omega_dir)*omega_dir)
            wall[nw].v += velocity

    @ti.kernel
    def kernel_move_patch_wall(self, start_index: int, end_index: int, disp: ti.types.vector(3, float), wall: ti.template()):
        for nw in range(start_index, end_index):
            wall[nw]._move(disp)

    @ti.kernel
    def kernel_translate_patch_wall(self, start_index: int, end_index: int, velocity: ti.types.vector(3, float), dt: ti.template(), wall: ti.template()):
        for nw in range(start_index, end_index):
            dx = velocity * dt[None]
            wall[nw]._move(dx)

    @ti.kernel
    def kernel_rotate_patch_wall(self, start_index: int, end_index: int, rotate_center: ti.types.vector(3, float), angular_velocity: ti.types.vector(3, float), dt: ti.template(), wall: ti.template()):
        for nw in range(start_index, end_index):
            omega_mag = angular_velocity.norm()
            theta = omega_mag * dt[None]
            u = angular_velocity / omega_mag  
            
            v1 = wall[nw].vertice1 - rotate_center
            v2 = wall[nw].vertice2 - rotate_center
            v3 = wall[nw].vertice3 - rotate_center
            norm = wall[nw].norm
            vel = wall[nw].v

            v1_rot = (v1 * ti.cos(theta) + u.cross(v1) * ti.sin(theta) + u * u.dot(v1) * (1 - ti.cos(theta)))
            v2_rot = (v2 * ti.cos(theta) + u.cross(v2) * ti.sin(theta) + u * u.dot(v2) * (1 - ti.cos(theta)))
            v3_rot = (v3 * ti.cos(theta) + u.cross(v3) * ti.sin(theta) + u * u.dot(v3) * (1 - ti.cos(theta)))
            norm_rot = (norm * ti.cos(theta) + u.cross(norm) * ti.sin(theta) + u * u.dot(norm) * (1 - ti.cos(theta)))
            vel_rot = (vel * ti.cos(theta) + u.cross(vel) * ti.sin(theta) + u * u.dot(vel) * (1 - ti.cos(theta)))

            disp1_norm = Squared(v1_rot - v1)
            disp2_norm = Squared(v2_rot - v2)
            disp3_norm = Squared(v3_rot - v3)
            disp = v1_rot - v1
            max_dist = disp1_norm
            if disp2_norm > max_dist:
                max_dist = disp2_norm
                disp = v2_rot - v2
            if disp3_norm > max_dist:
                disp = v3_rot - v3

            wall[nw].vertice1 = v1_rot + rotate_center
            wall[nw].vertice2 = v2_rot + rotate_center
            wall[nw].vertice3 = v3_rot + rotate_center
            wall[nw].verletDisp += disp
            wall[nw].v = vel_rot
            wall[nw].norm = norm_rot


import taichi as ti
from src.utils.constants import THRESHOLD, ZEROVEC3F, M_THRESHOLD
from src.utils.ScalarFunction import sgn, sgn


@ti.dataclass
class ServoWall:
    active: ti.u8
    startIndex: ti.u8
    endIndex: ti.u8
    alpha: float
    area: float
    current_force: float
    target_stress: float
    max_velocity: float
    pre_velocity: float
    gain: float

    @ti.func
    def _restart(self, active, start_index, end_index, alpha, gain, target_stress, max_velocity):
        self.active = ti.u8(active)
        self.startIndex = ti.u8(start_index)
        self.endIndex = ti.u8(end_index)
        self.alpha = float(alpha)
        self.gain = float(gain)
        self.target_stress = float(target_stress)
        self.max_velocity = float(max_velocity)

    @ti.pyfunc
    def add_servo_wall(self, start_index, end_index, alpha, gain, target_stress, max_velocity):
        self.active = 1
        self.startIndex = start_index
        self.endIndex = end_index
        self.alpha = alpha
        self.gain = gain
        self.target_stress = target_stress
        self.max_velocity = max_velocity

    @ti.pyfunc
    def calculate_gains(self, dt, wall):
        stiffness = self.get_geometry_stiffness(wall)
        if stiffness > THRESHOLD:
            self.gain = self.alpha / (dt[None] * stiffness)
        else:
            self.gain = M_THRESHOLD

    @ti.func
    def calculate_sole_gains(self, dt, stiffness):
        if stiffness > THRESHOLD:
            self.gain = self.alpha / (dt[None] * stiffness)
        else:
            self.gain = M_THRESHOLD

    @ti.pyfunc
    def calculate_delta(self):
        return self.target_stress * self.area - self.current_force

    @ti.pyfunc
    def calculate_velocity(self, wall):
        norm = ZEROVEC3F
        velocity = ZEROVEC3F
        inv_number = 1.0 / (self.endIndex - self.startIndex)
        for nwall in range(self.startIndex, self.endIndex):
            velocity += wall[nwall].v
            norm += wall[nwall].norm
        velocity *= inv_number
        norm = (norm * inv_number).normalized()
        err = self.calculate_delta()

        normal_v = sgn(err) * ti.min(self.max_velocity, ti.abs(self.gain * err))
        self.pre_velocity = normal_v
        tang_v = velocity - velocity.dot(norm) * norm
        return normal_v * norm + tang_v

    @ti.func
    def move(self, distance, wall):
        for nwall in range(self.startIndex, self.endIndex):
            wall[nwall].vertice1 += distance
            wall[nwall].vertice2 += distance
            wall[nwall].vertice3 += distance

    @ti.func
    def get_geometry_center(self, wall):
        center = ZEROVEC3F
        for nwall in range(self.startIndex, self.endIndex):
            center += 1.0 / 3.0 * (wall[nwall].vertice1 + wall[nwall].vertice2 + wall[nwall].vertice3)
        return center / (self.endIndex - self.startIndex)

    @ti.pyfunc
    def get_geometry_force(self, wall):
        force = ZEROVEC3F
        for nwall in range(self.startIndex, self.endIndex):
            force += wall[nwall].contact_force
        return force

    @ti.pyfunc
    def get_geometry_stiffness(self, wall):
        stiffness = 0.0
        for nwall in range(self.startIndex, self.endIndex):
            stiffness += wall[nwall].contact_stiffness
        return stiffness

    @ti.func
    def get_geometry_velocity(self):
        return self.velocity

    @ti.func
    def update_current_velocity(self, velocity):
        self.velocity = velocity

    @ti.pyfunc
    def update_current_force(self, force):
        self.current_force = force

    @ti.pyfunc
    def update_area(self, area):
        self.area = area

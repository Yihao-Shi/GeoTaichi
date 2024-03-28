from typing import Any
import taichi as ti

from src.utils.constants import PI, DBL_EPSILON, Threshold, ZEROVEC3f, ZEROMAT2x2
from src.utils.GeometryFunction import SphereTriangleIntersectionArea, DistanceFromPointToTriangle
from src.utils.ObjectIO import DictIO
from src.utils.ScalarFunction import sgn
from src.utils.TypeDefination import vec3f, vec3u8, vec4f, vec3i, vec2f
from src.utils.VectorFunction import Zero2OneVector, linear_id


@ti.dataclass
class Material:
    density: float
    fdamp: float
    tdamp: float

    def add_attribute(self, attribute):
        density = DictIO.GetEssential(attribute, 'Density')
        fdamp = DictIO.GetAlternative(attribute, 'ForceLocalDamping', 0.7)
        tdamp = DictIO.GetAlternative(attribute, 'TorqueLocalDamping', 0.7)
        self.add_density(density)
        self.add_local_damping_coefficient(fdamp, tdamp)

    def add_density(self, density):
        self.density = density

    def add_local_damping_coefficient(self, fdamp, tdamp):
        self.fdamp = fdamp
        self.tdamp = tdamp

    @ti.func
    def _get_density(self):
        return self.density
    
    @ti.func
    def _get_force_damping_coefficient(self):
        return self.fdamp
    
    @ti.func
    def _get_torque_damping_coefficient(self):
        return self.tdamp
    
    def print_info(self, matID,):
        print("Material ID = ", matID)
        print("Density = ", self.density)
        print("Local Damping (Force) = ", self.fdamp)
        print("Local Damping (Torque) = ", self.tdamp, '\n')


@ti.dataclass
class ParticleFamily:          # device memory: 84B
    active: ti.u8
    multisphereIndex: int
    materialID: ti.u8
    groupID: ti.u8
    rad: float
    m: float
    x: vec3f
    v: vec3f
    w: vec3f
    contact_force: vec3f
    contact_torque: vec3f
    verletDisp: vec3f

    @ti.func
    def _restart(self, active, multisphereIndex, groupID, materialID, m, rad, x, v, w):
        self.active = ti.u8(active)
        self.multisphereIndex = int(multisphereIndex)
        self.groupID = ti.u8(groupID)
        self.materialID = ti.u8(materialID)
        self.m = float(m)
        self.rad = float(rad)
        self.x = float(x)
        self.v = float(v)
        self.w = float(w)

    @ti.func
    def _scale(self, factor):
        self.m *= factor * factor * factor
        self.rad *= factor

    @ti.func
    def _pebble_scale(self, factor, centor_of_mass):
        self.m *= factor * factor * factor
        self.rad *= factor
        self.x = self.x * factor + (1 - factor) * centor_of_mass

    @ti.func
    def _move(self, disp):
        self.x += disp
        self.verletDisp += disp

    @ti.func
    def _renew_verlet(self):
        self.verletDisp = ZEROVEC3f

    @ti.func
    def _velocity_update(self, dcurr_v):
        self.v += dcurr_v

    @ti.func
    def _pebble_velocity_update(self, dcurr_v):
        self.v += dcurr_v

    @ti.func
    def _angular_velocity_update(self, dcurr_w):
        self.w += dcurr_w

    @ti.func
    def _pebble_angular_velocity_update(self, dcurr_w):
        self.w += dcurr_w

    @ti.func
    def calm(self):
        self.v = ZEROVEC3f
        self.w = ZEROVEC3f

    @ti.func
    def _add_index(self, multisphereIndex):
        self.multisphereIndex = int(multisphereIndex)

    @ti.func
    def _update_contact_interaction(self, cforce, ctorque): 
        self.contact_force += cforce
        self.contact_torque += ctorque

    @ti.func
    def _add_particle_proporities(self, materialID, groupID, radius, mass):
        self.active = ti.u8(1)
        self.materialID = ti.u8(materialID)
        self.groupID = ti.u8(groupID)
        self.rad = float(radius)
        self.m = float(mass)

    @ti.func
    def _add_particle_kinematics(self, x, v, w):
        self.x = x
        self.v = v
        self.w = w

    @ti.func
    def _get_multisphere_index(self): return self.multisphereIndex

    @ti.func
    def _get_material(self): return self.materialID

    @ti.func
    def _get_group(self): return self.groupID

    @ti.func
    def _get_volume(self): return 4./3. * PI * self.rad * self.rad * self.rad

    @ti.func
    def _get_radius(self): return self.rad

    @ti.func
    def _get_mass(self): return self.m

    @ti.func
    def _get_position(self): return self.x

    @ti.func
    def _get_velocity(self): return self.v

    @ti.func
    def _get_angular_velocity(self): return self.w

    @ti.func
    def _get_verlet_displacement(self): return self.verletDisp


@ti.dataclass
class SphereFamily:            # device memory: 48B
    sphereIndex: int
    inv_I: float
    a: vec3f
    angmoment: vec3f
    q: vec4f
    fix_v: vec3u8
    fix_w: vec3u8

    @ti.func
    def _restart(self, sphereIndex, inv_I, q, a, angmoment, fix_v, fix_w):
        self.sphereIndex = int(sphereIndex)
        self.inv_I = float(inv_I)
        self.q = float(q)
        self.a = a
        self.angmoment = angmoment
        self.fix_v = ti.cast(fix_v, ti.u8)
        self.fix_w = ti.cast(fix_w, ti.u8)

    @ti.func
    def _scale(self, factor):
        self.inv_I /= factor * factor * factor * factor * factor

    @ti.func
    def _add_index(self, sphere_index):
        self.sphereIndex = int(sphere_index)

    @ti.func
    def _add_sphere_attribute(self, init_w, inv_inertia, q, fix_v, fix_w):
        self.inv_I = float(inv_inertia)
        self.q = float(q)
        self.fix_v = ti.cast(Zero2OneVector(fix_v), ti.u8)
        self.fix_w = ti.cast(Zero2OneVector(fix_w), ti.u8)
        self.angmoment = init_w / self.inv_I * int(self.fix_w)

    @ti.func
    def _get_sphere_index(self): return self.sphereIndex

    @ti.func
    def _get_inverse_inertia(self): return self.inv_I

    @ti.func
    def _get_quaternion(self): return self.q

    @ti.func
    def _get_is_velocity_fixed(self): return self.fix_v

    @ti.func
    def _get_is_angular_velocity_fixed(self): return self.fix_w

    def deactivate(self, particle: ParticleFamily):
        particle[self.sphereIndex] = ti.u8(0)

@ti.dataclass
class ClumpFamily:            # device memory: 84B
    startIndex: int
    endIndex: int 
    m: float
    equi_r: float
    a: vec3f
    angmoment: vec3f
    mass_center: vec3f
    v: vec3f
    w: vec3f
    q: vec4f
    inv_I: vec3f

    @ti.func
    def _restart(self, startIndex, endIndex, m, equi_r, mass_center, v, w, a, angmoment, q, inv_I):
        self.startIndex = startIndex
        self.endIndex = endIndex
        self.m = m
        self.equi_r = equi_r
        self.mass_center = mass_center
        self.v = v
        self.w = w
        self.q = q
        self.a = a
        self.angmoment = angmoment
        self.inv_I = inv_I

    @ti.func
    def _scale(self, factor):
        self.m *= factor * factor * factor
        self.equi_r *= factor
        self.inv_I /= factor * factor * factor * factor * factor

    @ti.func
    def _move(self, disp):
        self.mass_center += disp

    @ti.func
    def _velocity_update(self, dcurr_v):
        self.v += dcurr_v

    @ti.func
    def _angular_velocity_update(self, dcurr_w):
        self.w += dcurr_w

    @ti.func
    def _add_index(self, start_index, end_index):
        self.startIndex = int(start_index)
        self.endIndex = int(end_index)

    @ti.func
    def _add_clump_kinematic(self, centor_of_mass, init_v, init_w, inv_i):
        self.mass_center = float(centor_of_mass)
        self.v = float(init_v)
        self.w = float(init_w)
        self.angmoment = self.w / inv_i

    @ti.func
    def _add_clump_attribute(self, mass, equi_r, inv_inertia, q):
        self.m = float(mass)
        self.equi_r = float(equi_r)
        self.inv_I = float(inv_inertia)
        self.q = float(q)

    @ti.func
    def _get_start_index(self): return self.startIndex

    @ti.func
    def _get_end_index(self): return self.endIndex

    @ti.func
    def _get_mass(self): return self.m

    @ti.func
    def _get_volume(self): return 4/3. * PI * self.equi_r * self.equi_r * self.equi_r

    @ti.func
    def _get_equivalent_radius(self): return self.equi_r

    @ti.func
    def _get_center_of_mass(self): return self.mass_center

    @ti.func
    def _get_velocity(self): return self.v

    @ti.func
    def _get_angular_velocity(self): return self.w

    @ti.func
    def _get_quanternion(self): return self.q

    @ti.func
    def _get_inverse_inertia(self): return self.inv_I

    def deactivate(self, particle: ParticleFamily):
        for npebble in range(self.startIndex, self.endIndex + 1):
            particle[npebble] = ti.u8(0)


@ti.dataclass
class PlaneFamily:
    active: ti.u8
    wallID: int
    materialID: ti.u8
    point: vec3f
    norm: vec3f
    
    @ti.pyfunc
    def add_materialID(self, matID):
        self.materialID = matID

    @ti.pyfunc
    def add_wall_geometry(self, wallID, point, norm):
        self.wallID = wallID
        self.active = 1
        self.point = vec3f([float(center) for center in point])
        self.norm = vec3f([float(normal) for normal in norm])

    def deactivate(self):
        self.active = 0

    @ti.func
    def _restart(self, active, wallID, materialID, point, norm):
        self.active = ti.u8(active)
        self.wallID = int(wallID)
        self.materialID = ti.u8(materialID)
        self.point = float(point)
        self.norm = float(norm)

    @ti.func
    def _get_status(self): return self.active

    @ti.func
    def _get_materialID(self): return self.materialID

    @ti.func
    def _get_center(self): return self.point
    
    @ti.func
    def _get_norm(self): return self.norm

    @ti.func
    def _get_velocity(self): return ZEROVEC3f

    @ti.func
    def _update_contact_stiffness(self, stiffess): pass

    @ti.func
    def _update_contact_interaction(self, cforce): pass

    @ti.func
    def _move(self, disp): pass

    @ti.func
    def _renew_verlet(self): pass
    
    @ti.func
    def _point_projection(self, point):
        center = self._get_center()
        norm = self._get_norm()
        distance = (point - center).dot(norm)
        return point - distance * norm
    
    @ti.func
    def _point_to_wall_distance(self, point):
        return (point - self._get_center()).dot(self.norm)
    
    @ti.func
    def _get_norm_distance(self, point):
        return (point - self._get_center()).dot(self.norm)

    @ti.func
    def _is_in_plane(self, point):
        return 1
    
    @ti.func
    def _is_sphere_intersect(self, position, contact_radius):
        distance = self._point_to_wall_distance(position)
        return ti.abs(distance) < contact_radius 
    
    @ti.func
    def processCircleShape(self, point, distance, criteria):
        return 1

@ti.dataclass
class FacetFamily:
    wallID: int
    active: ti.u8
    materialID: ti.u8
    contact_stiffness: float
    vertice1: vec3f
    vertice2: vec3f
    vertice3: vec3f
    verletDisp: vec3f
    norm: vec3f
    v: vec3f
    contact_force: vec3f
    bound_beg: vec3f
    bound_end: vec3f

    @ti.pyfunc
    def add_materialID(self, matID):
        self.materialID = matID

    @ti.pyfunc
    def add_wall_geometry(self, wallID, vertice1, vertice2, vertice3, norm, init_v):
        self.wallID = wallID
        self.active = 1
        self.vertice1 = vec3f([float(vertice) for vertice in vertice1])
        self.vertice2 = vec3f([float(vertice) for vertice in vertice2])
        self.vertice3 = vec3f([float(vertice) for vertice in vertice3])
        self.norm = norm
        self.v = init_v

    def deactivate(self):
        self.active = 0

    @ti.func
    def _restart(self, active, wallID, materialID, point1, point2, point3, norm, velocity):
        self.active = ti.u8(active)
        self.wallID = int(wallID)
        self.materialID = ti.u8(materialID)
        self.vertice1 = float(point1)
        self.vertice2 = float(point2)
        self.vertice3 = float(point3)
        self.norm = float(norm)
        self.v = float(velocity)

    @ti.func
    def _contact_force_reset(self): self.contact_force = ZEROVEC3f

    @ti.func
    def _contact_stiffness_reset(self): self.contact_stiffness = 0.

    @ti.func
    def _reset(self): 
        self.contact_force = ZEROVEC3f
        self.contact_stiffness = 0.

    @ti.func
    def _get_square(self):
        a = self.vertice2 - self.vertice1
        b = self.vertice3 - self.vertice1
        return 0.5 * ti.sqrt((a[1] * b[2] - b[1] * a[2]) * (a[1] * b[2] - b[1] * a[2]) + (a[0] * b[2] - b[0] * a[2]) * (a[0] * b[2] - b[0] * a[2]) + \
                           (a[0] * b[1] - b[0] * a[1]) * (a[0] * b[1] - b[0] * a[1]))
    
    @ti.func
    def _move(self, disp):
        self.vertice1 += disp
        self.vertice2 += disp
        self.vertice3 += disp
        self.verletDisp += disp

    @ti.func
    def _renew_verlet(self):
        self.verletDisp = ZEROVEC3f

    @ti.func
    def _get_status(self): return self.active

    @ti.func
    def _get_materialID(self): return self.materialID

    @ti.func
    def _get_vertice1(self): return self.vertice1

    @ti.func
    def _get_vertice2(self): return self.vertice2

    @ti.func
    def _get_vertice3(self): return self.vertice3

    @ti.func
    def _get_norm(self): return self.norm

    @ti.func
    def _get_bounding_box(self): return self.bound_beg, self.bound_end

    @ti.func
    def _get_center(self): return (self.vertice1 + self.vertice2 + self.vertice3) / 3.
    
    @ti.func
    def _get_velocity(self): return self.v

    @ti.func
    def _update_contact_stiffness(self, stiffess):
        self.contact_stiffness += stiffess

    @ti.func
    def _update_contact_interaction(self, cforce): 
        self.contact_force += cforce
    
    @ti.func
    def _point_projection(self, point):
        center = self._get_center()
        norm = self._get_norm()
        distance = (point - center).dot(norm)
        return point - distance * norm
    
    @ti.func
    def _point_projection_by_distance(self, point, distance):
        return point - distance * self.norm
    
    @ti.func
    def _point_to_wall_distance(self, point):
        return DistanceFromPointToTriangle(point, self.vertice1, self.vertice2, self.vertice3)
    
    @ti.func
    def _get_norm_distance(self, point):
        return (point - self._get_center()).dot(self.norm)

    @ti.func
    def _is_in_plane(self, point):
        p1 = self.vertice1
        p2 = self.vertice2
        p3 = self.vertice3
        u = (p1 - point).cross(p2 - point)
        v = (p2 - point).cross(p3 - point)
        w = (p3 - point).cross(p1 - point)
        return u.dot(v) >= 0. and u.dot(w) >= 0.
    
    @ti.func
    def _bounding_box(self):
        xmin, ymin, zmin = self._wall_boundary_min()
        xmax, ymax, zmax = self._wall_boundary_max()
        self.bound_beg = vec3f([xmin, ymin, zmin])
        self.bound_end = vec3f([xmax, ymax, zmax])

    @ti.func
    def _wall_boundary_min(self):
        return ti.min(self.vertice1[0], self.vertice2[0], self.vertice3[0]), \
               ti.min(self.vertice1[1], self.vertice2[1], self.vertice3[1]), \
               ti.min(self.vertice1[2], self.vertice2[2], self.vertice3[2])

    @ti.func
    def _wall_boundary_max(self):
        return ti.max(self.vertice1[0], self.vertice2[0], self.vertice3[0]), \
               ti.max(self.vertice1[1], self.vertice2[1], self.vertice3[1]), \
               ti.max(self.vertice1[2], self.vertice2[2], self.vertice3[2])

    '''
    @ti.func
    def _is_sphere_intersect(self, position, contact_radius):
        # return 0 - no contact, 1 - face contact, 2 - edge contact, 3 - vertices contact
        # see Su et. al (2011) Discrete element simulation of particle flow.
        
        valid = 0
        distance = self._point_projection(position)
        if ti.abs(distance) < contact_radius:
            projection_point = position - distance * self.norm
            u = (self.vertice1 - projection_point).cross(self.vertice2 - projection_point)
            v = (self.vertice2 - projection_point).cross(self.vertice3 - projection_point)
            w = (self.vertice3 - projection_point).cross(self.vertice1 - projection_point)
            if u.dot(v) > 0. and u.dot(w) > 0. and v.dot(w) > 0.:
                vaild = 1
            elif u.dot(v) == 0. and u.dot(w) == 0. and v.dot(w) == 0.:
                vaild = 3
            elif u.dot(v) > 0. or u.dot(w) > 0. or v.dot(w) > 0.:
                vaild = 2
        return vaild
    '''
    
    @ti.func
    def _is_sphere_intersect(self, position, contact_radius):
        distance = self._point_to_wall_distance(position)
        # in_plane = self._is_in_plane(self._point_projection_by_distance(position, distance))
        return ti.abs(distance) < contact_radius
    
    @ti.func
    def processCircleShape(self, point, distance, criteria):
        r = ti.sqrt(criteria)
        area0 = PI * r * r
        position = self._point_projection_by_distance(point, distance)
        area = SphereTriangleIntersectionArea(position, r, self.vertice1, self.vertice2, self.vertice3, self.norm)
        return area / area0


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
    gain: float

    @ti.func
    def _restart(self, active, start_index, end_index, alpha, target_stress, max_velocity):
        self.active = ti.u8(active)
        self.startIndex = ti.u8(start_index)
        self.endIndex = ti.u8(end_index)
        self.alpha = float(alpha)
        self.target_stress = float(target_stress)
        self.max_velocity = float(max_velocity)

    def add_servo_wall(self, start_index, end_index, alpha, target_stress, max_velocity):
        self.active = 1
        self.startIndex = start_index
        self.endIndex = end_index
        self.alpha = alpha
        self.target_stress = target_stress
        self.max_velocity = max_velocity   

    @ti.func
    def calculate_gains(self, dt, wall):
        stiffness = self.get_geometry_stiffness(wall)
        if stiffness > Threshold:
            self.gain = self.alpha * self.calculate_delta() / (dt[None] * stiffness)
        else:
            self.gain = self.max_velocity

    @ti.func
    def calculate_sole_gains(self, dt, stiffness):
        if stiffness > Threshold:
            self.gain = self.alpha * self.calculate_delta() / (dt[None] * stiffness)
        else:
            self.gain = self.max_velocity

    @ti.func
    def calculate_delta(self):
        return self.target_stress * self.area - self.current_force 
    
    @ti.func
    def calculate_velocity(self, wall):
        norm = ZEROVEC3f
        velocity = ZEROVEC3f
        inv_number = 1. / (self.endIndex - self.startIndex)
        for nwall in range(self.startIndex, self.endIndex):
            velocity += wall[nwall].v
            norm += wall[nwall].norm
        velocity *= inv_number
        norm = (norm * inv_number).normalized()

        normal_v = sgn(self.calculate_delta()) * ti.min(self.max_velocity, ti.abs(self.gain))
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
        center = ZEROVEC3f
        for nwall in range(self.startIndex, self.endIndex):
            center += 1./3. * (wall[nwall].vertice1 + wall[nwall].vertice2 + wall[nwall].vertice3)
        return center / (self.endIndex - self.startIndex)
    
    @ti.func
    def get_geometry_force(self, wall):
        force = ZEROVEC3f
        for nwall in range(self.startIndex, self.endIndex):
            force += wall[nwall].contact_force
        return force 
    
    @ti.func
    def get_geometry_stiffness(self, wall):
        stiffness = 0.
        for nwall in range(self.startIndex, self.endIndex):
            stiffness += wall[nwall].contact_stiffness
        return stiffness
    
    @ti.func
    def get_geometry_velocity(self):
        return self.velocity

    @ti.func
    def update_current_velocity(self, velocity):
        self.velocity = velocity 
    
    @ti.func
    def update_current_force(self, force):
        self.current_force = force 
    
    @ti.func
    def update_area(self, area):
        self.area = area

        
@ti.dataclass
class PatchFamily:    # memory usage: 64B
    wallID: int
    active: ti.u8
    materialID: ti.u8
    vertice1: vec3f
    vertice2: vec3f
    vertice3: vec3f
    verletDisp: vec3f
    norm: vec3f
    v: vec3f

    @ti.pyfunc
    def add_materialID(self, matID):
        self.materialID = matID

    @ti.pyfunc
    def add_wall_geometry(self, wall_id, vertice1, vertice2, vertice3, vel):
        self.active = 1
        self.wallID = int(wall_id)
        self.vertice1 = vec3f([float(vertice) for vertice in vertice1])
        self.vertice2 = vec3f([float(vertice) for vertice in vertice2])
        self.vertice3 = vec3f([float(vertice) for vertice in vertice3])
        self.norm = ((vertice2 - vertice1).cross(vertice3 - vertice1)).normalized()
        self.v = vel

    def deactivate(self):
        self.active = 0

    @ti.func
    def _restart(self, active, wallID, materialID, point1, point2, point3, norm, velocity):
        self.active = ti.u8(active)
        self.wallID = int(wallID)
        self.materialID = ti.u8(materialID)
        self.vertice1 = float(point1)
        self.vertice2 = float(point2)
        self.vertice3 = float(point3)
        self.norm = float(norm)
        self.v = float(velocity)

    @ti.func
    def _get_square(self):
        a = self.vertice2 - self.vertice1
        b = self.vertice3 - self.vertice1
        return 0.5 * ti.sqrt((a[1] * b[2] - b[1] * a[2]) * (a[1] * b[2] - b[1] * a[2]) + (a[0] * b[2] - b[0] * a[2]) * (a[0] * b[2] - b[0] * a[2]) + \
                           (a[0] * b[1] - b[0] * a[1]) * (a[0] * b[1] - b[0] * a[1]))

    @ti.func
    def _get_bounding_radius(self):
        a = self.vertice2 - self.vertice1
        b = self.vertice3 - self.vertice1
        c = self.vertice2 - self.vertice3
        S =  0.5 * ti.sqrt((a[1] * b[2] - b[1] * a[2]) * (a[1] * b[2] - b[1] * a[2]) + (a[0] * b[2] - b[0] * a[2]) * (a[0] * b[2] - b[0] * a[2]) + \
                           (a[0] * b[1] - b[0] * a[1]) * (a[0] * b[1] - b[0] * a[1]))

        length1 = a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
        length2 = b[0] * b[0] + b[1] * b[1] + b[2] * b[2]
        length3 = c[0] * c[0] + c[1] * c[1] + c[2] * c[2]
        return 0.25 * ti.sqrt(length1 * length2 * length3) / S
    
    @ti.func
    def _move(self, disp):
        self.vertice1 += disp
        self.vertice2 += disp
        self.vertice3 += disp
        self.verletDisp += disp

    @ti.func
    def _renew_verlet(self):
        self.verletDisp = ZEROVEC3f

    @ti.func
    def _get_status(self): return self.active

    @ti.func
    def _get_materialID(self): return self.materialID

    @ti.func
    def _get_vertice1(self): return self.vertice1

    @ti.func
    def _get_vertice2(self): return self.vertice2

    @ti.func
    def _get_vertice3(self): return self.vertice3

    @ti.func
    def _get_norm(self): return self.norm

    @ti.func
    def _get_velocity(self): return ZEROVEC3f

    @ti.func
    def _update_contact_stiffness(self, stiffess): pass

    @ti.func
    def _update_contact_interaction(self, cforce): pass

    @ti.func
    def _get_center(self):
        return (self.vertice1 + self.vertice2 + self.vertice3) / 3.
    
    @ti.func
    def _point_projection(self, point):
        center = self._get_center()
        norm = self._get_norm()
        distance = (point - center).dot(norm)
        return point - distance * norm
    
    @ti.func
    def _point_projection_by_distance(self, point, distance):
        return point - distance * self.norm
    
    @ti.func
    def _point_to_wall_distance(self, point):
        return DistanceFromPointToTriangle(point, self.vertice1, self.vertice2, self.vertice3)
    
    @ti.func
    def _get_norm_distance(self, point):
        return (point - self._get_center()).dot(self.norm)

    @ti.func
    def _is_in_plane(self, point):
        p1 = self.vertice1
        p2 = self.vertice2
        p3 = self.vertice3
        u = (p1 - point).cross(p2 - point)
        v = (p2 - point).cross(p3 - point)
        w = (p3 - point).cross(p1 - point)
        return u.dot(v) >= 0. and u.dot(w) >= 0.
    
    '''
    @ti.func
    def _is_sphere_intersect(self, position, contact_radius):
        # return 0 - no contact, 1 - face contact, 2 - edge contact, 3 - vertices contact
        # see Su et. al (2011) Discrete element simulation of particle flow.
        
        valid = 0
        distance = self._point_projection(position)
        if ti.abs(distance) < contact_radius:
            projection_point = position - distance * self.norm
            u = (self.vertice1 - projection_point).cross(self.vertice2 - projection_point)
            v = (self.vertice2 - projection_point).cross(self.vertice3 - projection_point)
            w = (self.vertice3 - projection_point).cross(self.vertice1 - projection_point)
            if u.dot(v) > 0. and u.dot(w) > 0. and v.dot(w) > 0.:
                vaild = 1
            elif u.dot(v) == 0. and u.dot(w) == 0. and v.dot(w) == 0.:
                vaild = 3
            elif u.dot(v) > 0. or u.dot(w) > 0. or v.dot(w) > 0.:
                vaild = 2
        return
    '''

    @ti.func
    def _is_sphere_intersect(self, position, contact_radius):
        distance = self._point_to_wall_distance(position)
        # in_plane = self._is_in_plane(self._point_projection_by_distance(position, distance))
        return ti.abs(distance) < contact_radius
    
    @ti.func
    def processCircleShape(self, point, distance, criteria):
        r = ti.sqrt(criteria)
        area0 = PI * r * r
        position = self._point_projection_by_distance(point, distance)
        area = SphereTriangleIntersectionArea(position, r, self.vertice1, self.vertice2, self.vertice3, self.norm)
        return area / area0


@ti.dataclass
class EnergyFamily:
    kinetic: float
    potential: float


@ti.dataclass
class ContactTable:
    endID1: int
    endID2: int
    cnforce: vec3f
    csforce: vec3f
    oldTangOverlap: vec3f

    @ti.func
    def _set_id(self, endID1, endID2):
        self.endID1 = endID1
        self.endID2 = endID2

    @ti.func
    def _set_contact(self, cnforce, csforce, overlap):
        self.cnforce = cnforce
        self.csforce = csforce
        self.oldTangOverlap = overlap

    @ti.func
    def _no_contact(self):
        self.cnforce = ZEROVEC3f
        self.csforce = ZEROVEC3f
        self.oldTangOverlap = ZEROVEC3f


@ti.dataclass
class CoupledContactTable:
    endID1: int
    endID2: int
    oldTangOverlap: vec3f

    @ti.func
    def _set_id(self, endID1, endID2):
        self.endID1 = endID1
        self.endID2 = endID2

    @ti.func
    def _set_contact(self, overlap):
        self.oldTangOverlap = overlap

    @ti.func
    def _no_contact(self):
        self.oldTangOverlap = ZEROVEC3f


@ti.dataclass
class SFContactTable:
    endID1: int
    endID2: int

    @ti.func
    def _set_id(self, endID1, endID2):
        self.endID1 = endID1
        self.endID2 = endID2


@ti.dataclass
class RollingContactTable:
    endID1: int
    endID2: int
    cnforce: vec3f
    csforce: vec3f
    oldTangOverlap: vec3f
    oldRollAngle: vec3f
    oldTwistAngle: vec3f

    @ti.func
    def _set_id(self, endID1, endID2):
        self.endID1 = endID1
        self.endID2 = endID2

    @ti.func
    def _set_contact(self, cnforce, csforce, tangential_overlap, rolling_overlap, twisting_overlap):
        self.cnforce = cnforce
        self.csforce = csforce
        self.oldTangOverlap = tangential_overlap
        self.oldRollAngle = rolling_overlap
        self.oldTwistAngle = twisting_overlap

    @ti.func
    def _no_contact(self):
        self.cnforce = ZEROVEC3f
        self.csforce = ZEROVEC3f
        self.oldTangOverlap = ZEROVEC3f
        self.oldRollAngle = ZEROVEC3f
        self.oldTwistAngle = ZEROVEC3f
    

@ti.dataclass
class HistoryContactTable:
    DstID: int
    oldTangOverlap: vec3f

    @ti.func
    def _copy(self, endID, overlap):
        self.DstID = endID
        self.oldTangOverlap = overlap


@ti.dataclass
class HistoryRollingContactTable:
    DstID: int
    oldTangOverlap: vec3f
    oldRollAngle: vec3f
    oldTwistAngle: vec3f

    @ti.func
    def _copy(self, endID, tangential_overlap, rolling_overlap, twisting_overlap):
        self.DstID = endID
        self.oldTangOverlap = tangential_overlap
        self.oldRollAngle = rolling_overlap
        self.oldTwistAngle = twisting_overlap


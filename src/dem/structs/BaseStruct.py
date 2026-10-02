import taichi as ti

from src.utils.constants import PI, Threshold, ZEROVEC3f, ZEROMAT2x2, ZEROMAT3x9, ZEROMAT9x9, DBL_EPSILON
from src.utils.GeometryFunction import SphereTriangleIntersectionArea, DistanceFromPointToTriangle
from src.utils.MatrixFunction import LUinverse, get_eigenvalue
from src.utils.ObjectIO import DictIO
from src.utils.ScalarFunction import sgn, biInterpolate, linearize3D, sgn
from src.utils.TypeDefination import vec3f, vec3u8, vec4f, vec3i, vec2f, vec2i, mat3x3, vec9f, vec6f
from src.utils.VectorFunction import vsign
from src.utils.BitFunction import Zero2OneVector


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
    def _get_multisphere_index1(self): return self.multisphereIndex

    @ti.func
    def _get_multisphere_index2(self): return self.multisphereIndex

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

    @ti.func
    def _get_contact_radius(self, contact_point): return (contact_point - self.x).norm()


@ti.dataclass
class SphereFamily:            # device memory: 48B
    grainIndex: int
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
    def _add_index(self, grainIndex, sphere_index):
        self.grainIndex = grainIndex
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
    grainIndex: int
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
    def _add_index(self, grainIndex, start_index, end_index):
        self.grainIndex = grainIndex
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
        return distance <= contact_radius 
    
    @ti.func
    def processCircleShape(self, point, distance, criteria):
        return 1.
    
    @ti.func
    def processImplicitSurfaceShape(self):
        return 1.

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
        self.wallID = int(wallID)
        self.active = 1
        self.vertice1 = vec3f([float(vertice) for vertice in vertice1])
        self.vertice2 = vec3f([float(vertice) for vertice in vertice2])
        self.vertice3 = vec3f([float(vertice) for vertice in vertice3])
        self.norm = norm
        self.v = init_v

    @ti.pyfunc
    def add_wall_geometry_(self, wall_id, vertice1, vertice2, vertice3, vel):
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
        return DistanceFromPointToTriangle(point, self.vertice1, self.vertice2, self.vertice3, -self.norm)
    
    @ti.func
    def _get_norm_distance(self, point):
        return (point - self._get_center()).dot(self.norm)
    
    @ti.func
    def _is_positive_direction(self, point):
        return (point - self.vertice1).dot(self.norm) > 0.

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
    def _get_contact_type(self, projection_point):
        contact_type = 0
        u = (self.vertice1 - projection_point).cross(self.vertice2 - projection_point)
        v = (self.vertice2 - projection_point).cross(self.vertice3 - projection_point)
        w = (self.vertice3 - projection_point).cross(self.vertice1 - projection_point)
        if u.dot(v) > 0. and u.dot(w) > 0. and v.dot(w) > 0.:
            contact_type = 1
        elif u.dot(v) == 0. and u.dot(w) == 0. and v.dot(w) == 0.:
            contact_type = 3
        elif u.dot(v) > 0. or u.dot(w) > 0. or v.dot(w) > 0.:
            contact_type = 2
        return contact_type
    
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
    
    @ti.func
    def _is_sphere_intersect(self, position, contact_radius):
        distance = self._point_to_wall_distance(position) # self._get_norm_distance(position) or self._point_to_wall_distance(position) ???
        # in_plane = self._is_in_plane(self._point_projection_by_distance(position, distance))
        return distance <= contact_radius
    
    @ti.func
    def processCircleShape(self, point, radius, distance):
        fraction = 0.
        if 0. < distance < radius:
            r = ti.sqrt(radius * radius - distance * distance)
            area0 = PI * r * r
            position = self._point_projection_by_distance(point, distance)
            area = SphereTriangleIntersectionArea(position, r, self.vertice1, self.vertice2, self.vertice3, self.norm)
            fraction = area / area0
        return fraction
    
    @ti.func
    def processImplicitSurfaceShape(self):
        return 1.


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

    @ti.pyfunc
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
    def add_wall_geometry(self, wall_id, vertice1, vertice2, vertice3):
        self.active = 1
        self.wallID = int(wall_id)
        self.vertice1 = vec3f([float(vertice) for vertice in vertice1])
        self.vertice2 = vec3f([float(vertice) for vertice in vertice2])
        self.vertice3 = vec3f([float(vertice) for vertice in vertice3])
        self.norm = ((vertice2 - vertice1).cross(vertice3 - vertice1)).normalized()

    def deactivate(self):
        self.active = 0

    @ti.func
    def _restart(self, active, wallID, materialID, point1, point2, point3, norm):
        self.active = ti.u8(active)
        self.wallID = int(wallID)
        self.materialID = ti.u8(materialID)
        self.vertice1 = float(point1)
        self.vertice2 = float(point2)
        self.vertice3 = float(point3)
        self.norm = float(norm)

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
        S = 0.5 * ti.sqrt((a[1] * b[2] - b[1] * a[2]) * (a[1] * b[2] - b[1] * a[2]) + (a[0] * b[2] - b[0] * a[2]) * (a[0] * b[2] - b[0] * a[2]) + \
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
    def _get_velocity(self): return self.v

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
        return DistanceFromPointToTriangle(point, self.vertice1, self.vertice2, self.vertice3, -self.norm)
    
    @ti.func
    def _get_norm_distance(self, point):
        return (point - self._get_center()).dot(self.norm)
    
    @ti.func
    def _is_positive_direction(self, point):
        return (point - self.vertice1).dot(self.norm) > 0.

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
    def _get_contact_type(self, projection_point):
        contact_type = 0
        u = (self.vertice1 - projection_point).cross(self.vertice2 - projection_point)
        v = (self.vertice2 - projection_point).cross(self.vertice3 - projection_point)
        w = (self.vertice3 - projection_point).cross(self.vertice1 - projection_point)
        if u.dot(v) > 0. and u.dot(w) > 0. and v.dot(w) > 0.:
            contact_type = 1
        elif u.dot(v) == 0. and u.dot(w) == 0. and v.dot(w) == 0.:
            contact_type = 3
        elif u.dot(v) > 0. or u.dot(w) > 0. or v.dot(w) > 0.:
            contact_type = 2
        return contact_type

    @ti.func
    def _is_sphere_intersect(self, position, contact_radius):
        distance = self._get_norm_distance(position)
        # in_plane = self._is_in_plane(self._point_projection_by_distance(position, distance))
        return distance <= contact_radius
    
    @ti.func
    def processCircleShape(self, point, radius, distance):
        fraction = 0.
        if 0. < distance < radius:
            r = ti.sqrt(radius * radius - distance * distance)
            area0 = PI * r * r
            position = self._point_projection_by_distance(point, distance)
            area = SphereTriangleIntersectionArea(position, r, self.vertice1, self.vertice2, self.vertice3, self.norm)
            fraction = area / area0
        return fraction
    
    @ti.func
    def processImplicitSurfaceShape(self):
        return 1.


@ti.dataclass
class LevelSetGrid:
    distance_field: float

    @ti.func
    def _set_grid(self, sdf):
        self.distance_field = float(sdf)

    @ti.func
    def _get_sdf(self):
        return self.distance_field
    
    @ti.func
    def _scale(self, scale):
        self.distance_field *= float(scale)

    @ti.func
    def distance(self, scale):
        return self.distance_field * scale
    

@ti.dataclass
class DeformableGrid:
    m: float
    f: vec3f
    v: vec3f

    @ti.func
    def _grid_reset(self):
        self.m = 0.
        self.momentum = ZEROVEC3f
        self.force = ZEROVEC3f

    @ti.func
    def _set_dofs(self, rowth):
        pass

    @ti.func
    def _update_nodal_mass(self, m):
        self.m += m

    @ti.func
    def _update_nodal_momentum(self, momentum):
        self.momentum += momentum

    @ti.func
    def _compute_nodal_velocity(self):
        self.momentum /= self.m

    @ti.func
    def _update_nodal_force(self, force):
        self.force += force

    @ti.func
    def _update_external_force(self, external_force):
        self.force += external_force

    @ti.func
    def _update_internal_force(self, internal_force):
        self.force += internal_force

    @ti.func
    def _compute_nodal_kinematic(self, damp, dt):
        unbalanced_force = self.force 
        velocity = self.momentum
        if velocity.dot(unbalanced_force) > 0.:
            unbalanced_force -= damp * unbalanced_force.norm() * vsign(velocity)
        acceleration = unbalanced_force / self.m
        self.momentum += acceleration * dt[None]
        self.force = acceleration 

    @ti.func
    def _update_nodal_kinematic(self):
        self.force /= self.m
        self.momentum /= self.m


@ti.dataclass
class VerticeNode:
    x: vec3f
    parameter: float

    @ti.func
    def _restart(self, parameter, x):
        self.parameter = float(parameter)
        self.x = float(x)

    @ti.func
    def _set_surface_node(self, x):
        self.x = float(x)

    @ti.func
    def _set_coefficient(self, coeff):
        self.parameter = coeff

    @ti.func
    def _scale(self, scale, centor_of_mass):
        self.x = float(scale * (self.x - centor_of_mass) + centor_of_mass)


@ti.dataclass
class TemplateSoftNode:
    parameter: float

    @ti.func
    def _set_coefficient(self, coeff):
        self.parameter = coeff


@ti.dataclass
class SoftSurfacePoint:
    pointID: int
    contact_force: vec3f

    @ti.func
    def _reset(self):
        self.contact_force = ZEROVEC3f

    @ti.func
    def _update_contact_interaction(self, cforce, ctorque):
        self.contact_force += cforce


@ti.dataclass
class SoftPoint:
    x: vec3f
    v: vec3f

    @ti.func
    def _set_surface_node(self, x):
        self.x = float(x)

    @ti.func
    def _scale(self, scale, centor_of_mass):
        self.x = float(scale * (self.x - centor_of_mass) + centor_of_mass)


class VerticeSoftNode:
    def __init__(self, surface_point, material_point) -> None:
        self.template_point = TemplateSoftNode.field(shape=material_point)
        self.surface_point = SoftPoint(shape=material_point)
        self.soft_point = SoftSurfacePoint(shape=surface_point)


@ti.dataclass
class RigidBody:   
    groupID: ti.u8
    materialID: ti.u8
    startNode: int
    endNode: int
    localNode: int
    m: float
    equi_r: float
    mass_center: vec3f
    a: vec3f
    v: vec3f
    w: vec3f
    angmoment: vec3f
    q: vec4f
    inv_I: vec3f
    contact_force: vec3f
    contact_torque: vec3f
    is_fix: vec3u8

    @ti.func
    def _restart(self, groupID, materialID, startNode, endNode, localNode, mass, equiv_rad, mass_center, a, angmoment, v, w, q, inv_I, is_fix):
        self.groupID = ti.u8(groupID)
        self.materialID = ti.u8(materialID)
        self.startNode = int(startNode)
        self.endNode = int(endNode)
        self.localNode = int(localNode)
        self.m = float(mass)
        self.equi_r = float(equiv_rad)
        self.mass_center = float(mass_center)
        self.a = float(a)
        self.v = float(v)
        self.w = float(w)
        self.angmoment = float(angmoment)
        self.q = float(q)
        self.inv_I = float(inv_I)
        self.is_fix = ti.cast(is_fix, ti.u8)

    @ti.func
    def _add_body_attribute(self, centor_of_mass, mass, equiv_rad, inv_inertia, q):
        self.mass_center = float(centor_of_mass)
        self.m = float(mass)
        self.equi_r = float(equiv_rad)
        self.inv_I = float(inv_inertia)
        self.q = float(q)

    @ti.func
    def _add_body_properties(self, materialID, groupID, density):
        self.materialID = ti.u8(materialID)
        self.groupID = ti.u8(groupID)
        self.m *= density
        self.inv_I /= density

    @ti.func
    def _add_surface_index(self, start_index, end_index, local_index):
        self.startNode = int(start_index)
        self.endNode = int(end_index)
        self.localNode = int(local_index)

    @ti.func
    def _add_body_kinematic(self, init_v, init_w, is_fix):
        self.v = float(init_v)
        self.w = float(init_w)
        self.angmoment = self.w / self.inv_I
        self.is_fix = ti.cast(Zero2OneVector(is_fix), ti.u8)

    @ti.func
    def _scale(self, factor):
        self.m *= float(factor * factor * factor)
        self.inv_I /= float(factor * factor * factor * factor * factor)
    
    @ti.func
    def _velocity_update(self, dcurr_v):
        self.v += dcurr_v

    @ti.func
    def _angular_velocity_update(self, dcurr_w):
        self.w += dcurr_w

    @ti.func
    def calm(self):
        self.v = ZEROVEC3f
        self.w = ZEROVEC3f

    @ti.func
    def _update_contact_interaction(self, cforce, ctorque): 
        self.contact_force += cforce
        self.contact_torque += ctorque

    @ti.func
    def _add_particle_kinematics(self, x, v, w):
        self.mass_center = x
        self.v = v
        self.w = w

    @ti.func
    def _get_vertice_number(self): return int(self.endNode - self.startNode)

    @ti.func
    def _start_node(self): return self.localNode

    @ti.func
    def _end_node(self): return int(self.localNode + self.endNode - self.startNode)

    @ti.func
    def local_node_to_global(self, node): return int(node - self.localNode + self.startNode)

    @ti.func
    def global_node_to_local(self, node): return int(node - self.startNode + self.localNode)

    @ti.func
    def _get_material(self): return self.materialID

    @ti.func
    def _get_group(self): return self.groupID

    @ti.func
    def _get_mass(self): return self.m

    @ti.func
    def _get_position(self): return self.mass_center

    @ti.func
    def _get_velocity(self): return self.v

    @ti.func
    def _get_angular_velocity(self): return self.w

    @ti.func
    def _get_volume(self): return 4./3. * PI * self.equi_r * self.equi_r * self.equi_r

    @ti.func
    def _get_contact_radius(self, contact_point): return (contact_point - self.mass_center).norm() 

    @ti.func
    def _get_radius(self): return self.equi_r


@ti.dataclass
class ImplicitSurfaceParticle:   
    groupID: ti.u8
    materialID: ti.u8
    templateID: ti.u8
    scale: float
    m: float
    equi_r: float
    mass_center: vec3f
    a: vec3f
    v: vec3f
    w: vec3f
    angmoment: vec3f
    q: vec4f
    inv_I: vec3f
    contact_force: vec3f
    contact_torque: vec3f
    is_fix: vec3u8

    @ti.func
    def _restart(self, groupID, materialID, templateID, scale, mass, equiv_rad, mass_center, a, angmoment, v, w, q, inv_I, is_fix):
        self.groupID = ti.u8(groupID)
        self.materialID = ti.u8(materialID)
        self.templateID = int(templateID)
        self.scale = float(scale)
        self.m = float(mass)
        self.equi_r = float(equiv_rad)
        self.mass_center = float(mass_center)
        self.a = float(a)
        self.v = float(v)
        self.w = float(w)
        self.angmoment = float(angmoment)
        self.q = float(q)
        self.inv_I = float(inv_I)
        self.is_fix = ti.cast(is_fix, ti.u8)

    @ti.func
    def _add_body_attribute(self, scale_factor, centor_of_mass, mass, equiv_rad, inv_inertia, q):
        self.scale = float(scale_factor)
        self.mass_center = float(centor_of_mass)
        self.m = float(mass)
        self.equi_r = float(equiv_rad)
        self.inv_I = float(inv_inertia)
        self.q = float(q)

    @ti.func
    def _add_body_properties(self, materialID, groupID, templateID, density):
        self.materialID = ti.u8(materialID)
        self.groupID = ti.u8(groupID)
        self.templateID = ti.u8(templateID)
        self.m *= density
        self.inv_I /= density

    @ti.func
    def _add_body_kinematic(self, init_v, init_w, is_fix):
        self.v = float(init_v)
        self.w = float(init_w)
        self.angmoment = self.w / self.inv_I
        self.is_fix = ti.cast(Zero2OneVector(is_fix), ti.u8)

    @ti.func
    def _scale(self, factor):
        self.m *= float(factor * factor * factor)
        self.inv_I /= float(factor * factor * factor * factor * factor)
        self.scale *= factor
    
    @ti.func
    def _velocity_update(self, dcurr_v):
        self.v += dcurr_v

    @ti.func
    def _angular_velocity_update(self, dcurr_w):
        self.w += dcurr_w

    @ti.func
    def calm(self):
        self.v = ZEROVEC3f
        self.w = ZEROVEC3f

    @ti.func
    def _update_contact_interaction(self, cforce, ctorque): 
        self.contact_force += cforce
        self.contact_torque += ctorque

    @ti.func
    def _add_particle_kinematics(self, x, v, w):
        self.mass_center = x
        self.v = v
        self.w = w

    @ti.func
    def _get_material(self): return self.materialID

    @ti.func
    def _get_group(self): return self.groupID

    @ti.func
    def _get_template(self): return self.templateID

    @ti.func
    def _get_scale(self): return self.scale

    @ti.func
    def _get_mass(self): return self.m

    @ti.func
    def _get_position(self): return self.mass_center

    @ti.func
    def _get_velocity(self): return self.v

    @ti.func
    def _get_angular_velocity(self): return self.w

    @ti.func
    def _get_volume(self): return 4./3. * PI * self.equi_r * self.equi_r * self.equi_r

    @ti.func
    def _get_contact_radius(self, contact_point): return (contact_point - self.mass_center).norm() 

    @ti.func
    def _get_radius(self): return self.equi_r

    @ti.func
    def _get_margin(self): return 0.05*self.equi_r


@ti.dataclass
class PolySuperEllipsoid: 
    xrad1: float
    yrad1: float
    zrad1: float
    epsilon1: float
    epsilon2: float
    xrad2: float
    yrad2: float
    zrad2: float

    @ti.func
    def _add_template_parameter(self, xrad1, yrad1, zrad1, epsilon_e, epsilon_n, xrad2, yrad2, zrad2):
        self.xrad1 = xrad1
        self.yrad1 = yrad1
        self.zrad1 = zrad1
        self.epsilon1 = 2. / epsilon_n
        self.epsilon2 = 2. / epsilon_e
        self.xrad2 = xrad2
        self.yrad2 = yrad2
        self.zrad2 = zrad2

    @ti.func
    def physical_parameters(self, scale):
        return ti.Vector([self.xrad1 * scale, self.yrad1 * scale, self.zrad1 * scale, 
                          self.xrad2 * scale, self.yrad2 * scale, self.zrad2 * scale, self.epsilon1, self.epsilon2])
    
    @ti.func
    def evolving_physical_parameters(self, fraction, bounding_rad, scale):
        return ti.Vector([bounding_rad + fraction * (scale * self.xrad1 - bounding_rad),
                          bounding_rad + fraction * (scale * self.yrad1 - bounding_rad),
                          bounding_rad + fraction * (scale * self.zrad1 - bounding_rad),
                          bounding_rad + fraction * (scale * self.xrad2 - bounding_rad),
                          bounding_rad + fraction * (scale * self.yrad2 - bounding_rad),
                          bounding_rad + fraction * (scale * self.zrad2 - bounding_rad), 
                          2. + fraction * (self.epsilon1 - 2.), 2. + fraction * (self.epsilon2 - 2.)])

    @ti.func
    def get_param(self, x, y, z, parameters):
        xrad = parameters[3] if x < 0. else parameters[0]
        yrad = parameters[4] if y < 0. else parameters[1]
        zrad = parameters[5] if z < 0. else parameters[2]
        return xrad, yrad, zrad

    @ti.func
    def fx(self, x, y, z, params):
        xrad, yrad, zrad = self.get_param(x, y, z, params)
        funcs = ti.pow(ti.pow(ti.abs(x / xrad), params[7]) + ti.pow(ti.abs(y / yrad), params[7]), params[6] / params[7]) \
                + ti.pow(ti.abs(z / zrad), params[6]) - 1.
        return funcs

    @ti.func
    def gradient(self, x, y, z, params):
        xrad, yrad, zrad = self.get_param(x, y, z, params)
        x0 = ti.abs(x / xrad)
        x1 = ti.abs(y / yrad)
        x2 = ti.abs(z / zrad)
        x3 = x0 ** (params[7])
        x4 = x1 ** (params[7])
        x5 = x0 ** (params[7] - 1.)
        x6 = x1 ** (params[7] - 1.)
        x7 = x2 ** (params[6] - 1.)
        mu = (x3 + x4) ** (params[6] / params[7] - 1.)
        return vec3f(params[6] / xrad * x5 * mu * sgn(x),
                     params[6] / yrad * x6 * mu * sgn(y),
                     params[6] / zrad * x7 * sgn(z))
    
    @ti.func
    def hessian(self, x, y, z, params):
        xrad, yrad, zrad = self.get_param(x, y, z, params)
        inv_xrad2 = 1. / (xrad * xrad)
        inv_yrad2 = 1. / (yrad * yrad)
        inv_zrad2 = 1. / (zrad * zrad)
        inv_xyrad = 1. / (xrad * yrad)
        x0 = ti.abs(x / xrad)
        x1 = ti.abs(y / yrad)
        x2 = ti.abs(z / zrad)
        x3 = x0 ** (params[7] - 2.)
        x4 = x1 ** (params[7] - 2.)
        x5 = x2 ** (params[6] - 2.)
        x6 = x0 ** (2. * params[7] - 2.)
        x7 = x1 ** (2. * params[7] - 2.)
        x8 = x0 ** (params[7])
        x9 = x1 ** (params[7])
        x10 = x0 ** (params[7] - 1.)
        x11 = x1 ** (params[7] - 1.)

        mu = x8 + x9
        mu0 = mu ** (params[6] / params[7] - 1.)
        mu1 = mu ** (params[6] / params[7] - 2.)

        n1n2 = params[6] * (params[7] - 1.)
        n1n1 = params[6] * (params[6] - 1.)
        n1n1n2 = params[6] * (params[6] - params[7])

        fxx = inv_xrad2 * (n1n2 * x3 * mu0 + n1n1n2 * x6 * mu1)
        fyy = inv_yrad2 * (n1n2 * x4 * mu0 + n1n1n2 * x7 * mu1)
        fxy = inv_xyrad * n1n1n2 * x10 * x11 * mu1 * sgn(x * y)
        fzz = inv_zrad2 * n1n1 * x5
        return mat3x3([fxx, fxy, 0.], [fxy, fyy, 0.], [0., 0., fzz])
    
    @ti.func
    def nearest_point(self, local_start_point, local_normal, params):
        t = 0.
        x = local_start_point[0] + t * local_normal[0]
        y = local_start_point[1] + t * local_normal[1]
        z = local_start_point[2] + t * local_normal[2]
            
        iter = 0
        fval = self.fx(x, y, z, params)
        while ti.abs(fval) > 1e-12 and iter < 50:
            t -= fval / self.gradient(x, y, z, params).dot(local_normal)

            x = local_start_point[0] + t * local_normal[0]
            y = local_start_point[1] + t * local_normal[1]
            z = local_start_point[2] + t * local_normal[2]
            fval = self.fx(x, y, z, params)
            iter += 1
        return t * local_normal + local_start_point

    @ti.func
    def support(self, local_normal, params):
        xrad, yrad, zrad = self.get_param(*local_normal, params)
        eps1, eps2 = 2. / self.epsilon2, 2. / self.epsilon1

        cos_phi0, sin_phi0, cos_phi1, sin_phi1 = 0.0, 0.0, 0.0, 0.0
        if ti.abs(local_normal[0]) < DBL_EPSILON:
            cos_phi0 = 0.0
            sin_phi0 = 1.0
            if ti.abs(local_normal[1]) < DBL_EPSILON:
                cos_phi1 = 0.0
                sin_phi1 = 1.0
            else:
                phi1_cos2 = 1.0 / (1.0 + pow(ti.abs(zrad * local_normal[2] * pow(sin_phi0, 2. - eps1) / yrad / local_normal[1]), 2 / (2. - eps2)))
                cos_phi1 = pow(phi1_cos2, 0.5)
                sin_phi1 = pow(1.0 - phi1_cos2, 0.5)
        else:
            phi0_cos2 = 1.0/(1.0 + pow((ti.abs(yrad*local_normal[1]/xrad/local_normal[0])),2/(2.-eps1)))
            cos_phi0 = pow(phi0_cos2,0.5)
            sin_phi0 = pow(1.0 - phi0_cos2, 0.5)
            phi1_cos2 = 1.0/(1.0 + pow(ti.abs(zrad*local_normal[2]* pow(cos_phi0,2.-eps1)/xrad/local_normal[0]),2/(2.-eps2)))
            cos_phi1 = pow(phi1_cos2,0.5)
            sin_phi1 = pow(1.0 - phi1_cos2, 0.5)

        x = sgn(local_normal[0]) * xrad * ti.pow(cos_phi0, eps1) * ti.pow(cos_phi1, eps2)
        y = sgn(local_normal[1]) * yrad * ti.pow(sin_phi0, eps1) * ti.pow(cos_phi1, eps2)
        z = sgn(local_normal[2]) * zrad * ti.pow(sin_phi1, eps2)
        return vec3f([x, y, z])
    
    @ti.func
    def mean_curvature(self, p, params):
        grad = self.gradient(p[0], p[1], p[2], params)
        hess = self.hessian(p[0], p[1], p[2], params)
        grad_norm = grad.norm()
        return 0.5 * (grad.T @ hess @ grad - grad_norm * grad_norm * (hess[0, 0] + hess[1, 1] + hess[2, 2])) / (grad_norm * grad_norm * grad_norm)
    
    @ti.func
    def gauss_curvature(self, p, params):
        grad = self.gradient(p[0], p[1], p[2], params)
        hess = self.hessian(p[0], p[1], p[2], params)
        
        A = grad[2] * (hess[0, 0] * grad[2] - 2. * grad[0] * hess[0, 2]) + grad[0] * grad[0] * hess[2, 2]
        B = grad[2] * (hess[1, 1] * grad[2] - 2. * grad[1] * hess[1, 2]) + grad[1] * grad[1] * hess[2, 2]
        C = grad[2] * (hess[0, 1] * grad[2] - grad[1] * hess[0, 2] - grad[0] * hess[1, 2]) + grad[0] * grad[1] * hess[2, 2]
        D = grad[2] * grad[2] * (grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2])
        return ti.sqrt(A * B - C * C) / D


@ti.dataclass
class PolySuperQuadrics: 
    xrad1: float
    yrad1: float
    zrad1: float
    epsilon1: float
    epsilon2: float
    epsilon3: float
    xrad2: float
    yrad2: float
    zrad2: float

    @ti.func
    def _add_template_parameter(self, xrad1, yrad1, zrad1, epsilon_x, epsilon_y, epsilon_z, xrad2, yrad2, zrad2):
        self.xrad1 = xrad1
        self.yrad1 = yrad1
        self.zrad1 = zrad1
        self.epsilon1 = 2. / epsilon_x
        self.epsilon2 = 2. / epsilon_y
        self.epsilon3 = 2. / epsilon_z
        self.xrad2 = xrad2
        self.yrad2 = yrad2
        self.zrad2 = zrad2

    @ti.func
    def physical_parameters(self, scale):
        return ti.Vector([self.xrad1 * scale, self.yrad1 * scale, self.zrad1 * scale, 
                          self.xrad2 * scale, self.yrad2 * scale, self.zrad2 * scale, self.epsilon1, self.epsilon2, self.epsilon3])
    
    @ti.func
    def evolving_physical_parameters(self, fraction, bounding_rad, scale):
        return ti.Vector([bounding_rad + fraction * (scale * self.xrad1 - bounding_rad),
                          bounding_rad + fraction * (scale * self.yrad1 - bounding_rad),
                          bounding_rad + fraction * (scale * self.zrad1 - bounding_rad),
                          bounding_rad + fraction * (scale * self.xrad2 - bounding_rad),
                          bounding_rad + fraction * (scale * self.yrad2 - bounding_rad),
                          bounding_rad + fraction * (scale * self.zrad2 - bounding_rad), 
                          2. + fraction * (self.epsilon1 - 2.), 2. + fraction * (self.epsilon2 - 2.), 2. + fraction * (self.epsilon3 - 2.)])

    @ti.func
    def get_param(self, x, y, z, parameters):
        xrad = parameters[3] if x < 0. else parameters[0]
        yrad = parameters[4] if y < 0. else parameters[1]
        zrad = parameters[5] if z < 0. else parameters[2]
        return xrad, yrad, zrad

    @ti.func
    def get_param(self, x, y, z):
        xrad, yrad, zrad = self.xrad1, self.yrad1, self.zrad1
        if x < 0.:
            xrad = self.xrad2
        if y < 0.:
            yrad = self.yrad2
        if z < 0.:
            zrad = self.zrad2
        return xrad, yrad, zrad

    @ti.func
    def fx(self, x, y, z, params):
        xrad, yrad, zrad = self.get_param(x, y, z, params)
        funcs = ti.pow(ti.abs(x / xrad), params[6]) + ti.pow(ti.abs(y / yrad), params[7]) + ti.pow(ti.abs(z / zrad), params[8]) - 1.
        return funcs

    @ti.func
    def gradient(self, x, y, z, params):
        xrad, yrad, zrad = self.get_param(x, y, z, params)
        x0 = ti.abs(x / xrad)
        x1 = ti.abs(y / yrad)
        x2 = ti.abs(z / zrad)
        x5 = x0 ** (params[6] - 1.)
        x6 = x1 ** (params[7] - 1.)
        x7 = x2 ** (params[8] - 1.)
        return vec3f(params[6] / xrad * x5 * sgn(x),
                     params[7] / yrad * x6 * sgn(y),
                     params[8] / zrad * x7 * sgn(z))
    
    @ti.func
    def hessian(self, x, y, z, params):
        xrad, yrad, zrad = self.get_param(x, y, z, params)
        inv_xrad2 = 1. / (xrad * xrad)
        inv_yrad2 = 1. / (yrad * yrad)
        inv_zrad2 = 1. / (zrad * zrad)
        x0 = ti.abs(x / xrad)
        x1 = ti.abs(y / yrad)
        x2 = ti.abs(z / zrad)
        x3 = x0 ** (params[6] - 2.)
        x4 = x1 ** (params[7] - 2.)
        x5 = x2 ** (params[8] - 2.)
        n1n1 = params[6] * (params[6] - 1.)
        n2n2 = params[7] * (params[7] - 1.)
        n3n3 = params[8] * (params[8] - 1.)
        fxx = inv_xrad2 * n1n1 * x3
        fyy = inv_yrad2 * n2n2 * x4
        fzz = inv_zrad2 * n3n3 * x5
        return mat3x3([fxx, 0., 0.], [0., fyy, 0.], [0., 0., fzz])
    
    @ti.func
    def nearest_point(self, local_start_point, local_normal, params):
        t = 0.
        x = local_start_point[0] + t * local_normal[0]
        y = local_start_point[1] + t * local_normal[1]
        z = local_start_point[2] + t * local_normal[2]
            
        iter = 0
        fval = self.fx(x, y, z, params)
        while ti.abs(fval) > 1e-12 and iter < 50:
            t -= fval / self.gradient(x, y, z, params).dot(local_normal)

            x = local_start_point[0] + t * local_normal[0]
            y = local_start_point[1] + t * local_normal[1]
            z = local_start_point[2] + t * local_normal[2]
            fval = self.fx(x, y, z, params)
            iter += 1
        return t * local_normal + local_start_point

    @ti.func
    def support(self, local_normal, params):
        pass
    
    @ti.func
    def mean_curvature(self, p, params):
        grad = self.gradient(p[0], p[1], p[2], params)
        hess = self.hessian(p[0], p[1], p[2], params)
        grad_norm = grad.norm()
        return 0.5 * (grad.T @ hess @ grad - grad_norm * grad_norm * (hess[0, 0] + hess[1, 1] + hess[2, 2])) / (grad_norm * grad_norm * grad_norm)
    
    @ti.func
    def gauss_curvature(self, p, params):
        grad = self.gradient(p[0], p[1], p[2], params)
        hess = self.hessian(p[0], p[1], p[2], params)
        
        A = grad[2] * (hess[0, 0] * grad[2] - 2. * grad[0] * hess[0, 2]) + grad[0] * grad[0] * hess[2, 2]
        B = grad[2] * (hess[1, 1] * grad[2] - 2. * grad[1] * hess[1, 2]) + grad[1] * grad[1] * hess[2, 2]
        C = grad[2] * (hess[0, 1] * grad[2] - grad[1] * hess[0, 2] - grad[0] * hess[1, 2]) + grad[0] * grad[1] * hess[2, 2]
        D = grad[2] * grad[2] * (grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2])
        return ti.sqrt(A * B - C * C) / D


@ti.dataclass
class SoftBody:   
    groupID: ti.u8
    materialID: ti.u8
    startNode: int
    endNode: int
    localNode: int

    @ti.func
    def _restart(self, startIndex, endIndex, groupID, materialID):
        self.startIndex = int(startIndex)
        self.endIndex = int(endIndex)
        self.groupID = ti.u8(groupID)
        self.materialID = ti.u8(materialID)

    @ti.func
    def _add_body_attribute(self, mass):
        self.m = float(mass)

    @ti.func
    def _add_body_properties(self, materialID, groupID):
        self.materialID = ti.u8(materialID)
        self.groupID = ti.u8(groupID)

    @ti.func
    def _add_surface_index(self, start_index, end_index, local_index):
        self.startNode = int(start_index)
        self.endNode = int(end_index)
        self.localNode = int(local_index)

    @ti.func
    def _get_vertice_number(self): return int(self.endNode - self.startNode)

    @ti.func
    def _start_node(self): return self.localNode

    @ti.func
    def _end_node(self): return int(self.localNode + self.endNode - self.startNode)

    @ti.func
    def local_node_to_global(self, node): return int(node - self.localNode + self.startNode)

    @ti.func
    def global_node_to_local(self, node): return int(node - self.startNode + self.localNode)

    @ti.func
    def _get_material(self): return self.materialID

    @ti.func
    def _get_group(self): return self.groupID


@ti.dataclass
class BoundingSphere:
    active: ti.u8
    rad: float
    x: vec3f
    verletDisp: vec3f

    @ti.func
    def _restart(self, active, bounding_center, bounding_radius):
        self.active = ti.u8(active)
        self.x = float(bounding_center)
        self.rad = float(bounding_radius)

    @ti.func
    def _add_bounding_sphere(self, bounding_center, bounding_radius):
        self.active = ti.u8(1)
        self.x = float(bounding_center)
        self.rad = float(bounding_radius)

    @ti.func
    def _move(self, disp):
        self.x += disp
        self.verletDisp += disp

    @ti.func
    def _scale(self, scale, centor_of_mass):
        self.rad *= float(scale)
        self.x = float(scale * (self.x - centor_of_mass) + centor_of_mass)

    @ti.func
    def _translate(self, offset):
        self.x += offset

    @ti.func
    def _renew_verlet(self):
        self.verletDisp = ZEROVEC3f

    @ti.func
    def _get_multisphere_index1(self): return -1

    @ti.func
    def _get_multisphere_index2(self): return 1

    @ti.func
    def _get_position(self): return self.x

    @ti.func
    def _get_radius(self): return self.rad

    @ti.func
    def _get_verlet_displacement(self): return self.verletDisp


@ti.dataclass
class DeformableBoundingSphere:
    active: ti.u8
    rad: float
    rad0: float
    mass_center0: vec3f
    x: vec3f
    x0: vec3f
    verletDisp: vec3f

    @ti.func
    def _restart(self, active, bounding_center, bounding_radius):
        self.active = ti.u8(active)
        self.x = float(bounding_center)
        self.rad = float(bounding_radius)

    @ti.func
    def _add_bounding_sphere(self, bounding_center, bounding_radius):
        self.active = ti.u8(1)
        self.x = float(bounding_center)
        self.rad = float(bounding_radius)

    @ti.func
    def _move(self, disp):
        self.x += disp
        self.verletDisp += disp

    @ti.func
    def _scale(self, scale, centor_of_mass):
        self.rad *= float(scale)
        self.x = float(scale * (self.x - centor_of_mass) + centor_of_mass)

    @ti.func
    def _translate(self, offset):
        self.x += offset

    @ti.func
    def _renew_verlet(self):
        self.verletDisp = ZEROVEC3f

    @ti.func
    def _get_multisphere_index1(self): return -1

    @ti.func
    def _get_multisphere_index2(self): return 1

    @ti.func
    def _get_position(self): return self.x

    @ti.func
    def _get_radius(self): return self.rad

    @ti.func
    def _get_verlet_displacement(self): return self.verletDisp

    @ti.func
    def _evolution(self, startIndex, endIndex, particle):
        # Efficient updates of bounding sphere hierarchies for geometrically deformable models
        # Meshless Deformations based on shape matching
        apq = mat3x3([0, 0, 0], [0, 0, 0], [0, 0, 0])
        aqq = mat3x3([0, 0, 0], [0, 0, 0], [0, 0, 0])
        mass_center = vec3f(0., 0., 0.)
        for np in range(startIndex, endIndex):
            mass_center += particle[np].x
        mass_center /= (endIndex - startIndex)
        for np in range(startIndex, endIndex):
            pp = particle[np].x - mass_center
            qq = particle[np].x0 - self.mass_center0
            apq += particle[np].m * pp.outer_product(qq)
            aqq += particle[np].m * qq.outer_product(qq)
        a = apq @ aqq.inverse()
        d = 0.
        for np in range(startIndex, endIndex):
            pp = particle[np].x - mass_center
            qq = particle[np].x0 - self.mass_center0
            d = max(d, (a @ qq - pp).norm())
        aTa = a.transpose() @ a
        val = get_eigenvalue(aTa)
        self.rad = ti.sqrt(abs(val[2])) * self.rad0 + d
        self.x = a @ (self.x0 - self.mass_center0) + mass_center


@ti.dataclass
class DeformableQuadraticBoundingSphere:
    active: ti.u8
    rad: float
    rad0: float
    mass_center0: vec3f
    x: vec3f
    x0: vec3f
    verletDisp: vec3f
    qmax: float
    mmax: float

    @ti.func
    def _restart(self, active, bounding_center, bounding_radius):
        self.active = ti.u8(active)
        self.x = float(bounding_center)
        self.rad = float(bounding_radius)

    @ti.func
    def _add_bounding_sphere(self, bounding_center, bounding_radius):
        self.active = ti.u8(1)
        self.x = float(bounding_center)
        self.rad = float(bounding_radius)

    @ti.func
    def _move(self, disp):
        self.x += disp
        self.verletDisp += disp

    @ti.func
    def _scale(self, scale, centor_of_mass):
        self.rad *= float(scale)
        self.x = float(scale * (self.x - centor_of_mass) + centor_of_mass)

    @ti.func
    def _translate(self, offset):
        self.x += offset

    @ti.func
    def _renew_verlet(self):
        self.verletDisp = ZEROVEC3f

    @ti.func
    def _compute_auxiliary_parameter(self, startIndex, endIndex, particle):
        center = self.x
        for np in range(startIndex, endIndex):
            position = particle[np].x
            self.qmax = max(self.qmax, vec3f(position[0] * position[0] - center[0] * center[0], 
                                             position[1] * position[1] - center[1] * center[1], 
                                             position[2] * position[2] - center[2] * center[2]).norm())
            self.mmax = max(self.mmax, vec3f(position[0] * position[1] - center[0] * center[1], 
                                             position[1] * position[2] - center[1] * center[2], 
                                             position[0] * position[2] - center[0] * center[2]).norm())

    @ti.func
    def _get_multisphere_index1(self): return -1

    @ti.func
    def _get_multisphere_index2(self): return 1

    @ti.func
    def _get_position(self): return self.x

    @ti.func
    def _get_radius(self): return self.rad

    @ti.func
    def _get_verlet_displacement(self): return self.verletDisp

    @ti.func
    def _evolution(self, startIndex, endIndex, mass_center, particle):
        # Efficient updates of bounding sphere hierarchies for geometrically deformable models
        # Meshless Deformations based on shape matching
        apq = ZEROMAT3x9
        aqq = ZEROMAT9x9
        mass_center = vec3f(0., 0., 0.)
        for np in range(startIndex, endIndex):
            mass_center += particle[np].x
        mass_center /= (endIndex - startIndex)
        for np in range(startIndex, endIndex):
            pp = particle[np].x - self.mass_center
            qq0 = particle[np].x0 - self.mass_center0
            qq = vec9f(qq0[0], qq0[1], qq0[2], qq0[0] * qq0[0], qq0[1] * qq0[1], qq0[2] * qq0[2], qq0[0] * qq0[1], qq0[1] * qq0[2], qq0[2] * qq0[0])
            apq += particle[np].m * pp.outer_product(qq)
            aqq += particle[np].m * qq.outer_product(qq)
        a_cap = apq @ LUinverse(aqq)
        d = 0., 0., 0.
        for np in range(startIndex, endIndex):
            pp = particle[np].x - mass_center
            qq0 = particle[np].x0 - self.mass_center0
            qq = vec9f(qq0[0], qq0[1], qq0[2], qq0[0] * qq0[0], qq0[1] * qq0[1], qq0[2] * qq0[2], qq0[0] * qq0[1], qq0[1] * qq0[2], qq0[2] * qq0[0])
            d = max(d, (a_cap @ qq - pp).norm())
        a = mat3x3([a_cap[0, 0], a_cap[0, 1], a_cap[0, 2]], [a_cap[1, 0], a_cap[1, 1], a_cap[1, 2]], [a_cap[2, 0], a_cap[2, 1], a_cap[2, 2]])
        q = mat3x3([a_cap[0, 3], a_cap[0, 4], a_cap[0, 5]], [a_cap[1, 3], a_cap[1, 4], a_cap[1, 5]], [a_cap[2, 3], a_cap[2, 4], a_cap[2, 5]])
        m = mat3x3([a_cap[0, 6], a_cap[0, 7], a_cap[0, 8]], [a_cap[1, 6], a_cap[1, 7], a_cap[1, 8]], [a_cap[2, 6], a_cap[2, 7], a_cap[2, 8]])
        aval = get_eigenvalue(a.transpose() @ a)
        qval = get_eigenvalue(q.transpose() @ q)
        mval = get_eigenvalue(m.transpose() @ m)
        self.rad = ti.sqrt(abs(aval[2])) * self.rad0 + ti.sqrt(abs(qval[2])) * self.qmax + ti.sqrt(abs(mval[2])) * self.mmax + d
        cc0 = self.x0 - self.mass_center0
        cc = vec9f(cc0[0], cc0[1], cc0[2], cc0[0] * cc0[0], cc0[1] * cc0[1], cc0[2] * cc0[2], cc0[0] * cc0[1], cc0[1] * cc0[2], cc0[2] * cc0[0])
        self.x = a_cap @ cc + mass_center


@ti.dataclass
class BoundingBox:
    xmin: vec3f
    xmax: vec3f
    startGrid: int
    gnum: vec3i
    grid_space: float
    scale: float
    extent: int

    @ti.func
    def _restart(self, gnum, xmin, xmax, startGrid, grid_space, scale, extent):
        self.gnum = int(gnum)
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.startGrid = float(startGrid)
        self.grid_space = float(grid_space)
        self.scale = float(scale)
        self.extent = float(extent)

    @ti.func
    def _set_bounding_box(self, xmin, xmax):
        self.xmin = float(xmin)
        self.xmax = float(xmax)

    @ti.func
    def _get_center(self):
        return 0.5 * (self.xmin + self.xmax)
    
    @ti.func
    def _scale(self, scale, centor_of_mass):
        self.xmin = float(scale * (self.xmin - centor_of_mass) + centor_of_mass)
        self.xmax = float(scale * (self.xmax - centor_of_mass) + centor_of_mass)

    @ti.func
    def _add_grid(self, start_grid, grid_space, gnum, scale, extent):
        self.startGrid = int(start_grid)
        self.grid_space = float(grid_space)
        self.gnum = float(gnum)
        self.scale = float(scale)
        self.extent = int(extent)
    
    @ti.func
    def _translate(self, offset):
        self.xmin += offset
        self.xmax += offset

    @ti.func
    def _get_dim(self):
        return self.xmax - self.xmin
    
    @ti.func
    def _in_box(self, point):
        in_box = 1
        if point[0] < self.xmin[0]: in_box = 0
        if point[0] > self.xmax[0]: in_box = 0
        if point[1] < self.xmin[1]: in_box = 0
        if point[1] > self.xmax[1]: in_box = 0
        if point[2] < self.xmin[2]: in_box = 0
        if point[2] > self.xmax[2]: in_box = 0
        return in_box
    
    @ti.func
    def closet_corner(self, point):
        retIndices = vec3i(0, 0, 0)
        for index in range(3):
            retIndices[index] = int((point[index] - self.xmin[index]) / self.grid_space)
        return retIndices
    
    @ti.func
    def distance(self, point, grid):
        space = self.grid_space
        indices = self.closet_corner(point)
        xInd, yInd, zInd = indices[0], indices[1], indices[2]

        Ind000 = linearize3D(xInd, yInd, zInd, self.gnum) + self.startGrid
        Ind001 = linearize3D(xInd, yInd, zInd + 1, self.gnum) + self.startGrid
        Ind010 = linearize3D(xInd, yInd + 1, zInd, self.gnum) + self.startGrid
        Ind011 = linearize3D(xInd, yInd + 1, zInd + 1, self.gnum) + self.startGrid
        Ind100 = linearize3D(xInd + 1, yInd, zInd, self.gnum) + self.startGrid
        Ind101 = linearize3D(xInd + 1, yInd, zInd + 1, self.gnum) + self.startGrid
        Ind110 = linearize3D(xInd + 1, yInd + 1, zInd, self.gnum) + self.startGrid
        Ind111 = linearize3D(xInd + 1, yInd + 1, zInd + 1, self.gnum) + self.startGrid

        temp1 = self.xmin + vec3i(xInd, yInd, zInd) * space
        temp2 = self.xmin + vec3i(xInd, yInd + 1, zInd) * space
        temp3 = self.xmin + vec3i(xInd, yInd, zInd + 1) * space
        yzCoord = vec2f(point[1], point[2])
        yExtr = vec2f(temp1[1], temp2[1])
        zExtr = vec2f(temp1[2], temp3[2])

        knownValx0, knownValx1 = ZEROMAT2x2, ZEROMAT2x2
        knownValx0[0, 0] = grid[Ind000].distance_field * self.scale
        knownValx0[0, 1] = grid[Ind001].distance_field * self.scale
        knownValx0[1, 0] = grid[Ind010].distance_field * self.scale
        knownValx0[1, 1] = grid[Ind011].distance_field * self.scale

        knownValx1[0, 0] = grid[Ind100].distance_field * self.scale
        knownValx1[0, 1] = grid[Ind101].distance_field * self.scale
        knownValx1[1, 0] = grid[Ind110].distance_field * self.scale
        knownValx1[1, 1] = grid[Ind111].distance_field * self.scale

        f0yz = biInterpolate(yzCoord, yExtr, zExtr, knownValx0)
        f1yz = biInterpolate(yzCoord, yExtr, zExtr, knownValx1)
        return (point[0] - temp1[0]) / space * (f1yz - f0yz) + f0yz
    
    @ti.func
    def calculate_gradient(self, point, grid):
        indices = self.closet_corner(point)
        xInd, yInd, zInd = indices[0], indices[1], indices[2]
        spacing = self.grid_space
        
        xRed = (point[0] - (self.xmin[0] + xInd * spacing)) / spacing
        yRed = (point[1] - (self.xmin[1] + yInd * spacing)) / spacing
        zRed = (point[2] - (self.xmin[2] + zInd * spacing)) / spacing

        normal = vec3f(0, 0, 0)
        for i in ti.static(range(2)):
            for j in ti.static(range(2)):
                for k in ti.static(range(2)):
                    Ind = linearize3D(xInd + i, yInd + j, zInd + k, self.gnum) + self.startGrid
                    lsVal = grid[Ind].distance_field * self.scale
                    normal[0] += lsVal * (2 * i - 1) * ((1 - j) * (1 - yRed) + j * yRed) * ((1 - k) * (1 - zRed) + k * zRed)
                    normal[1] += lsVal * (2 * j - 1) * ((1 - i) * (1 - xRed) + i * xRed) * ((1 - k) * (1 - zRed) + k * zRed)
                    normal[2] += lsVal * (2 * k - 1) * ((1 - i) * (1 - xRed) + i * xRed) * ((1 - j) * (1 - yRed) + j * yRed)
        return normal
    
    @ti.func
    def calculate_normal(self, point, grid):
        return self.calculate_gradient(point, grid).normalized()
    

@ti.dataclass
class HierarchicalBody:
    level: ti.u8
    max_potential_particle_pairs: int
    max_potential_wall_pairs: int

    @ti.func
    def _set(self, potential_particle_ratio, body_coordination_number, wall_coordination_number):
        self.max_potential_particle_pairs = ti.ceil(potential_particle_ratio * body_coordination_number)
        self.max_potential_wall_pairs = wall_coordination_number

    @ti.func
    def _set_level(self, level):
        self.level = ti.u8(level)

    @ti.func
    def potential_particle_num(self):
        return self.max_potential_particle_pairs
    
    @ti.func
    def potential_wall_num(self):
        return self.max_potential_wall_pairs


@ti.dataclass
class HierarchicalCell:
    pnum: int
    rad_min: float
    rad_max: float
    grid_size: float
    igrid_size: float
    factor: float
    cell_index: int
    cnum: vec3i
    wall_per_cell: int
    wall_cells: int

    @ti.func
    def _set(self, grid_size, factor, cnum, wall_per_cell):
        self.grid_size = float(grid_size)
        self.igrid_size = 1. / float(grid_size)
        self.factor = float(factor)
        self.cnum = cnum
        self.wall_per_cell = wall_per_cell

    @ti.func
    def _set_wall_cells(self, wall_cells):
        self.wall_cells = wall_cells

    @ti.func
    def _set_cell_index(self, csum):
        self.cell_index = float(csum)

    @ti.func
    def _cell_sum(self):
        cnum = self.cnum
        return cnum[0] * cnum[1] * cnum[2]

    @ti.func
    def _calculate(self):
        pass



@ti.dataclass
class EnergyFamily:
    kinetic: float
    potential: float


@ti.dataclass
class LSPotentialContact:
    end1: int
    end2: int

    @ti.func
    def _set(self, end1, end2):
        self.end1 = end1
        self.end2 = end2


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
class ISContactTable:
    endID1: int
    endID2: int
    cnforce: vec3f
    csforce: vec3f
    oldTangOverlap: vec3f
    contactSA: vec3f

    @ti.func
    def _set_id(self, endID1, endID2):
        self.endID1 = endID1
        self.endID2 = endID2

    @ti.func
    def _set_contact(self, cnforce, csforce, overlap, contactSA):
        self.cnforce = cnforce
        self.csforce = csforce
        self.oldTangOverlap = overlap
        self.contactSA = contactSA

    @ti.func
    def _no_contact(self, contactSA):
        self.cnforce = ZEROVEC3f
        self.csforce = ZEROVEC3f
        self.oldTangOverlap = ZEROVEC3f
        self.contactSA = contactSA


@ti.dataclass
class VerletContactTable:
    endID1: int
    endID2: int
    verletDisp: vec3f

    @ti.func
    def _renew_verlet(self):
        self.verletDisp = ZEROVEC3f


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
    def _set_contact(self, normal_force, tangential_force, overlap):
        self.oldTangOverlap = overlap

    @ti.func
    def _no_contact(self):
        self.oldTangOverlap = ZEROVEC3f


@ti.dataclass
class CoupledRollingContactTable:
    endID1: int
    endID2: int
    oldTangOverlap: vec3f
    oldRollAngle: vec3f
    oldTwistAngle: vec3f

    @ti.func
    def _set_id(self, endID1, endID2):
        self.endID1 = endID1
        self.endID2 = endID2

    @ti.func
    def _set_contact(self, normal_force, tangential_force, tangential_overlap, rolling_overlap, twisting_overlap):
        self.oldTangOverlap = tangential_overlap
        self.oldRollAngle = rolling_overlap
        self.oldTwistAngle = twisting_overlap

    @ti.func
    def _no_contact(self):
        self.oldTangOverlap = ZEROVEC3f
        self.oldRollAngle = ZEROVEC3f
        self.oldTwistAngle = ZEROVEC3f


@ti.dataclass
class DigitalContactTable:
    oldTangOverlap: vec3f

    @ti.func
    def _set_contact(self, var1, var2, tangential_overlap):
        self.oldTangOverlap = tangential_overlap

    @ti.func
    def _no_contact(self):
        self.oldTangOverlap = ZEROVEC3f


@ti.dataclass
class DigitalRollingContactTable:
    oldTangOverlap: vec3f
    oldRollAngle: vec3f
    oldTwistAngle: vec3f

    @ti.func
    def _set_contact(self, tangential_overlap, rolling_overlap, twisting_overlap):
        self.oldTangOverlap = tangential_overlap
        self.oldRollAngle = rolling_overlap
        self.oldTwistAngle = twisting_overlap

    @ti.func
    def _no_contact(self):
        self.oldTangOverlap = ZEROVEC3f
        self.oldRollAngle = ZEROVEC3f
        self.oldTwistAngle = ZEROVEC3f


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
class RollingISContactTable:
    endID1: int
    endID2: int
    cnforce: vec3f
    csforce: vec3f
    oldTangOverlap: vec3f
    oldRollAngle: vec3f
    oldTwistAngle: vec3f
    contactSA: vec3f

    @ti.func
    def _set_id(self, endID1, endID2):
        self.endID1 = endID1
        self.endID2 = endID2

    @ti.func
    def _set_contact(self, cnforce, csforce, tangential_overlap, rolling_overlap, twisting_overlap, contactSA):
        self.cnforce = cnforce
        self.csforce = csforce
        self.oldTangOverlap = tangential_overlap
        self.oldRollAngle = rolling_overlap
        self.oldTwistAngle = twisting_overlap
        self.contactSA = contactSA

    @ti.func
    def _no_contact(self, contactSA):
        self.cnforce = ZEROVEC3f
        self.csforce = ZEROVEC3f
        self.oldTangOverlap = ZEROVEC3f
        self.oldRollAngle = ZEROVEC3f
        self.oldTwistAngle = ZEROVEC3f
        self.contactSA = contactSA
    

@ti.dataclass
class HistoryISContactTable:
    DstID: int
    oldTangOverlap: vec3f
    contactSA: vec3f

    @ti.func
    def _copy(self, endID, overlap, contactSA):
        self.DstID = endID
        self.oldTangOverlap = overlap
        self.contactSA = contactSA
    

@ti.dataclass
class HistoryContactTable:
    DstID: int
    oldTangOverlap: vec3f

    @ti.func
    def _copy(self, endID, overlap):
        self.DstID = endID
        self.oldTangOverlap = overlap
        

@ti.dataclass
class HistoryRollingISContactTable:
    DstID: int
    oldTangOverlap: vec3f
    oldRollAngle: vec3f
    oldTwistAngle: vec3f
    contactSA: vec3f

    @ti.func
    def _copy(self, endID, tangential_overlap, rolling_overlap, twisting_overlap, contactSA):
        self.DstID = endID
        self.oldTangOverlap = tangential_overlap
        self.oldRollAngle = rolling_overlap
        self.oldTwistAngle = twisting_overlap
        self.contactSA = contactSA
        

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


class DigitalElevationModel(object):
    def __init__(self) -> None:
        self.materialID = 0
        self.digital_size = 0.
        self.idigital_size = 0.
        self.digital_dim = [0, 0]

    def set_digital_elevation(self, materialID, cell_size, cnum):
        self.materialID = materialID
        self.digital_size = cell_size
        self.idigital_size = 1. / cell_size
        self.digital_dim = vec2i(cnum)



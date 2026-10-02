import numpy as np

from src.sdf.SDFs import SDF3D
from src.sdf.MultiSDF import intersection
from src.sdf.utils import _min, _max, _vec, _dot, _length, _normalize
from src.utils.linalg import ndot

    
# Primitives
class sphere(SDF3D):
    def __init__(self, radius=1., center=(0., 0., 0.), ray=False):
        super().__init__(ray)
        self.radius = radius
        self.center = np.array(center)

    def _distance(self, p):
        return _length(p - self.center) - self.radius
    
class plane(SDF3D):
    def __init__(self, normal=(0., 0., 1.), center=(0., 0., 0.), ray=False):
        super().__init__(ray)
        self.center = np.array(center)
        self.normal = _normalize(np.array(normal))

    def _distance(self, p):
        return np.dot(self.center - p, self.normal)

class slab(SDF3D):
    def __init__(self, x0=None, y0=None, z0=None, x1=None, y1=None, z1=None, smooth=None, ray=False):
        super().__init__(ray)
        smooth = smooth or self._smooth
        fs = []
        if x0 is not None:
            fs.append(plane(normal=[1., 0., 0.], center=[x0, 0, 0]))
        if x1 is not None:
            fs.append(plane(normal=[-1., 0., 0.], center=[x1, 0, 0]))
        if y0 is not None:
            fs.append(plane(normal=[0., 1., 0.], center=[0, y0, 0]))
        if y1 is not None:
            fs.append(plane(normal=[0., -1., 0.], center=[0, y1, 0]))
        if z0 is not None:
            fs.append(plane(normal=[0., 0., 1.], center=[0, 0, z0]))
        if z1 is not None:
            fs.append(plane(normal=[0., 0., -1.], center=[0, 0, z1]))
        self.calculate_distance = intersection(*fs, k=smooth)

class box(SDF3D):
    def __init__(self, size=(1., 1., 1.), center=(0., 0., 0.), point1=None, point2=None, ray=False):
        super().__init__(ray )
        if point1 is not None and point2 is not None:
            point1 = np.array(point1)
            point2 = np.array(point2)
            size = point2 - point1
            center = point1 + size / 2
        self.size = np.array(size)
        self.center = np.array(center)

    def _distance(self, p):
        q = np.abs(p - self.center) - self.size / 2
        return _length(_max(q, 0)) + _min(np.amax(q, axis=1), 0)

class rounded_box(SDF3D):
    def __init__(self, size=(1., 1., 1.), center=(0., 0., 0.), point1=None, point2=None, smooth=None, ray=False):
        super().__init__(ray)
        if point1 is not None and point2 is not None:
            point1 = np.array(point1)
            point2 = np.array(point2)
            size = point2 - point1
            center = point1 + size / 2
        self.size = np.array(size)
        self.center = np.array(center)
        self._smooth = smooth or self._smooth

    def _distance(self, p):
        q = np.abs(p) - self.size / 2 + self._smooth
        return _length(_max(q, 0)) + _min(np.amax(q, axis=1), 0) - self._smooth
    
class box_frame(SDF3D):
    def __init__(self, thickness, size=(1., 1., 1.), center=(0., 0., 0.), point1=None, point2=None, ray=False):
        super().__init__(ray)
        if point1 is not None and point2 is not None:
            point1 = np.array(point1)
            point2 = np.array(point2)
            size = point2 - point1
            center = point1 + size / 2
        self.size = np.array(size)
        self.center = np.array(center)
        self.thickness = thickness

    def g(self, a, b, c):
        return _length(_max(_vec(a, b, c), 0)) + _min(_max(a, _max(b, c)), 0)
    
    def _distance(self, p):
        p = np.abs(p) - self.size / 2 - self.thickness / 2
        q = np.abs(p + self.thickness / 2) - self.thickness / 2
        px, py, pz = p[:,0], p[:,1], p[:,2]
        qx, qy, qz = q[:,0], q[:,1], q[:,2]
        return _min(_min(self.g(px, qy, qz), self.g(qx, py, qz)), self.g(qx, qy, pz))

class torus(SDF3D):
    def __init__(self, radius1, radius2, ray=False):
        super().__init__(ray)
        self.radius1 = radius1
        self.radius2 = radius2

    def _distance(self, p):
        xy = p[:,[0, 1]]
        z = p[:,2]
        a = _length(xy) - self.radius1
        b = _length(_vec(a, z)) - self.radius2
        return b
    
class capped_torus(SDF3D):
    def __init__(self, radius1, radius2, theta, ray=False):
        super().__init__(ray)
        self.radius1 = radius1
        self.radius2 = radius2
        self.theta = theta

    def _distance(self, p):
        sc = np.array([np.sin(self.theta), np.cos(self.theta)])
        p[:,0] = np.abs(p[:,0])
        k = _dot(p[:,[0,1]], sc) if sc[1] * p[0] > sc[0] * p[1] else _length(p[:,[0,1]])
        return np.sqrt(_dot(p, p) + self.radius1 * self.radius1 - 2. * self.radius1 * k) - self.radius2
    
class link(SDF3D):
    def __init__(self, radius1, radius2, length, ray=False):
        super().__init__(ray)
        self.radius1 = radius1
        self.radius2 = radius2
        self.length = length

    def _distance(self, p):
        q = np.array([p[0], max(abs(p[1]) - self.length, 0.), p[2]])
        return _length(np.array([_length(p[:,[0,1]]) - self.radius1, q[2]])) - self.radius2
    
class heart(SDF3D):
    def __init__(self, a=1., b=1., c=1., depth=1.0, thick=9./80., ray=False):
        super().__init__(ray)
        self.a = a
        self.b = b
        self.c = c
        self.depth = depth
        self.thick = thick

    def _distance(self, p):
        return (self.a * p[0] ** 2 + self.b * p[1] ** 2 + self.c * p[2] ** 2 - 1) ** 3 - self.depth * p[0] ** 2 * p[2] ** 3 - self.thick * p[1] ** 2 * p[2] ** 3

class hexagonal_prism(SDF3D):
    def __init__(self, height1, height2, ray=False):
        super().__init__(ray)
        self.height1 = height1
        self.height2 = height2

    def _distance(self, p):
        k = np.array([-0.5 * np.sqrt(2), 0.5, np.sqrt(3) / 3.])
        p = np.abs(p)
        scale = 2. * min(np.array([k[0], k[1]]).dot(np.array([p[0], p[1]])), 0.)
        p[0] -= scale * k
        p[1] -= scale * k
        d = np.array([np.linalg.norm(np.array([p[0], p[1]]) - np.array([np.clip(p[0], -k[2] * self.height1, k[1] * self.height1), self.height1])) * np.sign(p[1] - self.height1), p[2] - self.height2])
        return min(max(d[0], d[1]), 0.) + np.linalg.norm(np.maximum(d, 0.))

class capsule(SDF3D):
    def __init__(self, point1, point2, radius, ray=False):
        super().__init__(ray)
        self.radius = radius
        self.point1= np.array(point1)
        self.point2 = np.array(point2)

    def _distance(self, p):
        pa = p - self.point1
        ba = self.point2 - self.point1
        h = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0, 1).reshape((-1, 1))
        return _length(pa - np.multiply(ba, h)) - self.radius

class cylinder(SDF3D):
    def __init__(self, radius, ray=False):
        super().__init__(ray)
        self.radius = radius
        
    def _distance(self, p):
        return _length(p[:,[0,1]]) - self.radius

class capped_cylinder(SDF3D):
    def __init__(self, point1, point2, radius, ray=False):
        super().__init__(ray)
        self.radius = radius
        self.point1 = np.array(point1)
        self.point2 = np.array(point2)

    def _distance(self, p):
        ba = self.point2 - self.point1
        pa = p - self.point1
        baba = np.dot(ba, ba)
        paba = np.dot(pa, ba).reshape((-1, 1))
        x = _length(pa * baba - ba * paba) - self.radius * baba
        y = np.abs(paba - baba * 0.5) - baba * 0.5
        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))
        x2 = x * x
        y2 = y * y * baba
        d = np.where(_max(x, y) < 0, -_min(x2, y2), np.where(x > 0, x2, 0) + np.where(y > 0, y2, 0))
        return np.sign(d) * np.sqrt(np.abs(d)) / baba

class rounded_cylinder(SDF3D):
    def __init__(self, height, radius1, radius2, ray=False):
        super().__init__(ray)
        self.height = height
        self.radius1 = radius1
        self.radius2 = radius2

    def _distance(self, p):
        d = _vec(_length(p[:,[0, 1]]) - self.radius1 + self.radius2, np.abs(p[:,2]) - self.height / 2 + self.radius2)
        return (_min(_max(d[:,0], d[:,1]), 0) + _length(_max(d, 0)) - self.radius2)

class cone(SDF3D):
    def __init__(self, angle, height, ray=False):
        super().__init__(ray)
        self.angle = angle
        self.height = height

    def _distance(self, p):
        q = self.height * np.array([np.sin(self.angle) / np.cos(self.angle), -1.])
        w = np.array([np.sqrt(p[0] * p[0] + p[2] * p[2]), p[1]])
        a = w - q * np.clip(w.dot(q) / q.dot(q), 0., 1.)
        b = w - q * np.array([np.clip(w[0] / q[0], 0., 1.), 1.])
        k = np.sign(q[1])
        d = min(a.dot(a), b.dot(b))
        s = max(k * (w[0] * q[1] - w[1] - q[0]), k * (w[1] - q[1]))
        return np.sqrt(d) * np.sign(s)

class capped_cone(SDF3D):
    def __init__(self, point1, point2, radius1, radius2, ray=False):
        super().__init__(ray)
        self.point1 = np.array(point1)
        self.point2 = np.array(point2)
        self.radius1 = radius1
        self.radius2 = radius2

    def _distance(self, p):
        rba = self.radius2 - self.radius1
        baba = np.dot(self.point2 - self.point1, self.point2 - self.point1)
        papa = _dot(p - self.point1, p - self.point1)
        paba = np.dot(p - self.point1, self.point2 - self.point1) / baba
        x = np.sqrt(papa - paba * paba * baba)
        cax = _max(0, x - np.where(paba < 0.5, self.radius1, self.radius2))
        cay = np.abs(paba - 0.5) - 0.5
        k = rba * rba + baba
        f = np.clip((rba * (x - self.radius1) + paba * baba) / k, 0, 1)
        cbx = x - self.radius1 - f * rba
        cby = paba - f
        s = np.where(np.logical_and(cbx < 0, cay < 0), -1, 1)
        return s * np.sqrt(_min(
            cax * cax + cay * cay * baba,
            cbx * cbx + cby * cby * baba))

class rounded_cone(SDF3D):
    def __init__(self, radius1, radius2, height, ray=False):
        super().__init__(ray)
        self.radius1 = radius1
        self.radius2 = radius2
        self.height = height

    def _distance(self, p):
        q = _vec(_length(p[:,[0,1]]), p[:,2])
        b = (self.radius1 - self.radius2) / self.height
        a = np.sqrt(1 - b * b)
        k = np.dot(q, _vec(-b, a))
        c1 = _length(q) - self.radius1
        c2 = _length(q - _vec(0, self.height)) - self.radius2
        c3 = np.dot(q, _vec(a, b)) - self.radius1
        return np.where(k < 0, c1, np.where(k > a * self.height, c2, c3))
    
class revolved_vesica(SDF3D):
    def __init__(self, point1, point2, radius, ray=False):
        super().__init__(ray)
        self.point1 = np.array(point1)
        self.point2 = np.array(point2)
        self.radius = radius

    def _distance(self, p):
        c = 0.5 * (self.point1 + self.point2)
        l = np.linalg.norm(self.point2 - self.point1)
        v = (self.point2 - self.point1) / l
        y = (p - c).dot(v)
        q = np.array([np.linalg.norm(p - c - y * v), abs(y)])

        r = 0.5 * l
        d = 0.5 * (r * r - self.radius * self.radius) / self.radius
        h = np.array([0., r, 0.]) if r * q[0] < d* (q[1] - r) else np.array([-d, 0., d + self.radius])
        return np.linalg.norm(q - np.array([h[0], h[1]])) - h[2]
    
class octahedron(SDF3D):
    def __init__(self, length, ray=False):
        super().__init__(ray)
        self.length = length

    def _distance(self, p):
        p = abs(p)
        m = p[0] + p[1] + p[2] - self.length

        value = 0.
        if 3. * p[0] < m: 
            q = np.array(p)
            k = np.clip(0.5 * (q[2] - q[1] + self.length), 0., self.length)
            value = np.array([q[0], q[1] - self.length + k, q[2] - k])
        elif 3. * p[1] < m: 
            q = np.array([p[1], p[2], p[0]])
            k = np.clip(0.5 * (q[2] - q[1] + self.length), 0., self.length)
            value = np.array([q[0], q[1] - self.length + k, q[2] - k])
        elif 3. * p[2] < m: 
            q = np.array([p[2], p[0], p[1]])
            k = np.clip(0.5 * (q[2] - q[1] + self.length), 0., self.length)
            value = np.array([q[0], q[1] - self.length + k, q[2] - k])
        else: 
            value = np.sqrt(3) / 3. * m
        return value
    
class rhombus(SDF3D):
    def __init__(self, length1, length2, radius, height, ray=False):
        super().__init__(ray)
        self.length1 = length1
        self.length2 = length2
        self.radius = radius
        self.height = height

    def _distance(self, p):
        p = np.abs(p)
        b = np.array([self.length1, self.length2])
        f = np.clip((ndot(b, b - 2. * np.array([p[0], p[2]]))) / b.dot(b), -1., 1.)
        q = np.array([np.linalg.norm(np.array([p[0], p[2]]) - 0.5 * b * np.array([1.0 - f, 1.0 + f])) * np.sign(p[0] * b[1] + p[2] * b[0] - b[0] * b[1]) - self.radius, p[1] - self.height])
        return min(max(q[0],q[1]), 0.0) + np.linalg.norm(np.maximum(q, 0.0))

class pyramid(SDF3D):
    def __init__(self, height, ray=False):
        super().__init__(ray)
        self.height = height

    def _distance(self, p):
        a = np.abs(p[:,[0,1]]) - 0.5
        w = a[:,1] > a[:,0]
        a[w] = a[:,[1,0]][w]
        px = a[:,0]
        py = p[:,2]
        pz = a[:,1]
        m2 = self.height * self.height + 0.25
        qx = pz
        qy = self.height * py - 0.5 * px
        qz = self.height * px + 0.5 * py
        s = _max(-qx, 0)
        t = np.clip((qy - 0.5 * pz) / (m2 + 0.25), 0, 1)
        a = m2 * (qx + s) ** 2 + qy * qy
        b = m2 * (qx + 0.5 * t) ** 2 + (qy - m2 * t) ** 2
        d2 = np.where(
            _min(qy, -qx * m2 - qy * 0.5) > 0,
            0, _min(a, b))
        return np.sqrt((d2 + qz * qz) / m2) * np.sign(_max(qz, -py))


# Platonic Solids
class tetrahedron(SDF3D):
    def __init__(self, length, ray=False):
        super().__init__(ray)
        self.length = length

    def _distance(self, p):
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        return (_max(np.abs(x + y) - z, np.abs(x - y) + z) - self.length) / np.sqrt(3)

class dodecahedron(SDF3D):
    def __init__(self, length, ray=False):
        super().__init__(ray)
        self.x, self.y, self.z = _normalize(((1 + np.sqrt(5)) / 2, 1, 0))
        self.length = length

    def _distance(self, p):
        p = np.abs(p / self.length)
        a = np.dot(p, (self.x, self.y, self.z))
        b = np.dot(p, (self.z, self.x, self.y))
        c = np.dot(p, (self.y, self.z, self.x))
        q = (_max(_max(a, b), c) - self.x) * self.length
        return q

class icosahedron(SDF3D):
    def __init__(self, length, ray=False):
        super().__init__(ray)
        self.length = 0.8506507174597755 * length
        self.x, self.y, self.z = _normalize(((np.sqrt(5) + 3) / 2, 1, 0))
        self.w = np.sqrt(3) / 3

    def _distance(self, p):
        p = np.abs(p / self.length)
        a = np.dot(p, (self.x, self.y, self.z))
        b = np.dot(p, (self.z, self.x, self.y))
        c = np.dot(p, (self.y, self.z, self.x))
        d = np.dot(p, (self.w, self.w, self.w)) - self.x
        return _max(_max(_max(a, b), c) - self.x, d) * self.length



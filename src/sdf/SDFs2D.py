import numpy as np

from src.sdf.SDFs import SDF2D
from src.sdf.MultiSDF import intersection
from src.sdf.utils import _min, _max, _vec, _dot, _length, _normalize


# Primitives
class circle(SDF2D):
    def __init__(self, radius=1., center=(0., 0.), ray=False):
        super().__init__(ray)
        self.radius = radius
        self.center = center

    def _distance(self, p):
        return _length(p - self.center) - self.radius

class line(SDF2D):
    def __init__(self, normal=(0., 1.), center=(0., 0.), ray=False):
        super().__init__(ray)
        self.point = np.array(center)
        self.normal = _normalize(np.array(normal))

    def _distance(self, p):
        return np.dot(self.point - p, self.normal)

class slab(SDF2D):
    def __init__(self, x0=None, y0=None, x1=None, y1=None, smooth=None, ray=False):
        super().__init__(ray)
        smooth = smooth or self._smooth
        fs = []
        if x0 is not None:
            fs.append(line([1., 0.], [x0, 0]))
        if x1 is not None:
            fs.append(line([-1., 0.], [x1, 0]))
        if y0 is not None:
            fs.append(line([0., 1.], [0, y0]))
        if y1 is not None:
            fs.append(line([0., -1.], [0, y1]))
        self.calculate_distance = intersection(*fs, k=smooth)

class rectangle(SDF2D):
    def __init__(self, size=(1., 1.), center=(0., 0.), point1=None, point2=None, ray=False):
        super().__init__(ray)
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

class rounded_rectangle(SDF2D):
    def __init__(self, size=(1., 1.), center=(0., 0.), point1=None, point2=None, smooth=None, ray=False):
        super().__init__(ray)
        if point1 is not None and point2 is not None:
            point1 = np.array(point1)
            point2 = np.array(point2)
            size = point2 - point1
            center = point1 + size / 2
        self.size = np.array(size)
        self.center = np.array(center)
        
        radius = smooth or self._smooth
        try:
            self.r0, self.r1, self.r2, self.r3 = radius
        except TypeError:
            self.r0 = self.r1 = self.r2 = self.r3 = radius

    def _distance(self, p):
        x = p[:, 0]
        y = p[:, 1]
        r = np.zeros(len(p)).reshape((-1, 1))
        r[np.logical_and(x > 0, y > 0)] = self.r0
        r[np.logical_and(x > 0, y <= 0)] = self.r1
        r[np.logical_and(x <= 0, y <= 0)] = self.r2
        r[np.logical_and(x <= 0, y > 0)] = self.r3
        q = np.abs(p) - self.size / 2 + r
        return (_min(_max(q[:,0], q[:,1]), 0).reshape((-1, 1)) + _length(_max(q, 0)).reshape((-1, 1)) - r)

class equilateral_triangle(SDF2D):
    def __init__(self, ray=False):
        super().__init__(ray)
    
    def _distance(self, p):
        k = 3 ** 0.5
        p = _vec(
            np.abs(p[:,0]) - 1,
            p[:,1] + 1 / k)
        w = p[:,0] + k * p[:,1] > 0
        q = _vec(
            p[:,0] - k * p[:,1],
            -k * p[:,0] - p[:,1]) / 2
        p = np.where(w.reshape((-1, 1)), q, p)
        p = _vec(
            p[:,0] - np.clip(p[:,0], -2, 0),
            p[:,1])
        return -_length(p) * np.sign(p[:,1])

class hexagon(SDF2D):
    def __init__(self, size=1., ray=False):
        super().__init__(ray)
        self.r = size * 3 ** 0.5 / 2

    def _distance(self, p):
        k = np.array((3 ** 0.5 / -2, 0.5, np.tan(np.pi / 6)))
        p = np.abs(p)
        p -= 2 * k[:2] * _min(_dot(k[:2], p), 0).reshape((-1, 1))
        p -= _vec(np.clip(p[:,0], -k[2] * self.r, k[2] * self.r), np.zeros(len(p)) + self.r)
        return _length(p) * np.sign(p[:,1])
    
class rounded_x(SDF2D):
    def __init__(self, w, r, ray=False):
        super().__init__(ray)
        self.w = w
        self.r = r

    def _distance(self, p):
        p = np.abs(p)
        q = (_min(p[:,0] + p[:,1], self.w) * 0.5).reshape((-1, 1))
        return _length(p - q) - self.r
    
class cross(SDF2D):
    def __init__(self, length1=1., length2=1., smooth=None, ray=False):
        super().__init__(ray)
        self.length1 = length1
        self.length2 = length2
        self._smooth = smooth or self._smooth

    def _distance(self, p):
        p = np.abs(p) 
        p[np.where(p[:,1] > p[:,0]), :] = p[np.where(p[:,1] > p[:,0]), [1,0]] 
        q = p - np.array([self.length1, self.length2])
        k = _max(q[:,1], q[:,0])
        w = q if k > 0.0 else np.array([self.length2 - p[0], -k])
        return np.sign(k) * _length(_max(w, 0.0)) + self._smooth

class polygon(SDF2D):
    def __init__(self, points, ray=False):
        super().__init__(ray)
        self.points = [np.array(p) for p in points]

    def _distance(self, p):
        n = len(self.points)
        d = _dot(p - self.points[0], p - self.points[0])
        s = np.ones(len(p))
        for i in range(n):
            j = (i + n - 1) % n
            vi = self.points[i]
            vj = self.points[j]
            e = vj - vi
            w = p - vi
            b = w - e * np.clip(np.dot(w, e) / np.dot(e, e), 0, 1).reshape((-1, 1))
            d = _min(d, _dot(b, b))
            c1 = p[:,1] >= vi[1]
            c2 = p[:,1] < vj[1]
            c3 = e[0] * w[:,1] > e[1] * w[:,0]
            c = _vec(c1, c2, c3)
            s = np.where(np.all(c, axis=1) | np.all(~c, axis=1), -s, s)
        return s * np.sqrt(d)




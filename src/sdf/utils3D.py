import numpy as np
from src.sdf.utils import _min, _max, _vec, _normalize, _perpendicular


# Positioning
def translate(other, offset):
    offset = np.array(offset)
    def f(p):
        return other(p - offset)
    return f

def scale(other, factor):
    try:
        x, y, z = factor
    except TypeError:
        x = y = z = factor
    s = (x, y, z)
    m = min(x, min(y, z))
    def f(p):
        return other(p / s) * m
    return f

def rotate(other, angle, vector):
    x, y, z = _normalize(np.array(vector))
    s = np.sin(angle)
    c = np.cos(angle)
    m = 1 - c
    matrix = np.array([
        [m*x*x + c, m*x*y + z*s, m*z*x - y*s],
        [m*x*y - z*s, m*y*y + c, m*y*z + x*s],
        [m*z*x + y*s, m*y*z - x*s, m*z*z + c],
    ]).T
    def f(p):
        return other(np.dot(p, matrix))
    return f

def rotate_to(other, a, b):
    a = _normalize(np.array(a))
    b = _normalize(np.array(b))
    dot = np.dot(b, a)
    if dot == 1:
        return other
    if dot == -1:
        return rotate(other, np.pi, _perpendicular(a))
    angle = np.arccos(dot)
    v = _normalize(np.cross(b, a))
    return rotate(other, angle, v)

def orient(other, axis):
    return rotate_to(other, (0., 0., 1.), axis)

def circular_array(other, count, offset):
    other = other.translate(np.array([1., 0., 0.]) * offset)
    da = 2 * np.pi / count
    def f(p):
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        d = np.hypot(x, y)
        a = np.arctan2(y, x) % da
        d1 = other(_vec(np.cos(a - da) * d, np.sin(a - da) * d, z))
        d2 = other(_vec(np.cos(a) * d, np.sin(a) * d, z))
        return _min(d1, d2)
    return f


# Alterations
def elongate(other, size):
    def f(p):
        q = np.abs(p) - size
        x = q[:,0].reshape((-1, 1))
        y = q[:,1].reshape((-1, 1))
        z = q[:,2].reshape((-1, 1))
        w = _min(_max(x, _max(y, z)), 0)
        return other(_max(q, 0)) + w
    return f

def twist(other, k):
    def f(p):
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        c = np.cos(k * z)
        s = np.sin(k * z)
        x2 = c * x - s * y
        y2 = s * x + c * y
        z2 = z
        return other(_vec(x2, y2, z2))
    return f

def bend(other, k):
    def f(p):
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        c = np.cos(k * x)
        s = np.sin(k * x)
        x2 = c * x - s * y
        y2 = s * x + c * y
        z2 = z
        return other(_vec(x2, y2, z2))
    return f

def bend_linear(other, p0, p1, v, e):
    p0 = np.array(p0)
    p1 = np.array(p1)
    v = -np.array(v)
    ab = p1 - p0
    def f(p):
        t = np.clip(np.dot(p - p0, ab) / np.dot(ab, ab), 0, 1)
        t = e(t).reshape((-1, 1))
        return other(p + t * v)
    return f

def bend_radial(other, r0, r1, dz, e):
    def f(p):
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        r = np.hypot(x, y)
        t = np.clip((r - r0) / (r1 - r0), 0, 1)
        z = z - dz * e(t)
        return other(_vec(x, y, z))
    return f

def transition_linear(f0, f1, p0, p1, e):
    p0 = np.array(p0)
    p1 = np.array(p1)
    ab = p1 - p0
    def f(p):
        d1 = f0(p)
        d2 = f1(p)
        t = np.clip(np.dot(p - p0, ab) / np.dot(ab, ab), 0, 1)
        t = e(t).reshape((-1, 1))
        return t * d2 + (1 - t) * d1
    return f

def transition_radial(f0, f1, r0, r1, e):
    def f(p):
        d1 = f0(p)
        d2 = f1(p)
        r = np.hypot(p[:,0], p[:,1])
        t = np.clip((r - r0) / (r1 - r0), 0, 1)
        t = e(t).reshape((-1, 1))
        return t * d2 + (1 - t) * d1
    return f

def wrap_around(other, x0, x1, r, e):
    p0 = np.array([1., 0., 0.]) * x0
    p1 = np.array([1., 0., 0.]) * x1
    v = -np.array([0., 1., 0.])
    if r is None:
        r = np.linalg.norm(p1 - p0) / (2 * np.pi)
    def f(p):
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        d = np.hypot(x, y) - r
        d = d.reshape((-1, 1))
        a = np.arctan2(y, x)
        t = (a + np.pi) / (2 * np.pi)
        t = e(t).reshape((-1, 1))
        q = p0 + (p1 - p0) * t + v * d
        q[:,2] = z
        return other(q)
    return f



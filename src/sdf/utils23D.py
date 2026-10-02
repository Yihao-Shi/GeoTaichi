import numpy as np

from src.sdf.utils import _min, _max, _vec, _length


# 3D => 2D Operations
def slice(other, slab):
    a = other & slab
    b = other.negate() & slab
    def f(p):
        p = _vec(p[:,0], p[:,1], np.zeros(len(p)))
        A = a(p).reshape(-1)
        B = -b(p).reshape(-1)
        w = A <= 0
        A[w] = B[w]
        return A
    return f


# 2D => 3D Operations
def extrude(other, h):
    def f(p):
        d = other(p[:,[0,1]])
        w = _vec(d.reshape(-1), np.abs(p[:,2]) - h / 2)
        return _min(_max(w[:,0], w[:,1]), 0) + _length(_max(w, 0))
    return f

def extrude_to(a, b, h, e):
    def f(p):
        d1 = a(p[:,[0,1]])
        d2 = b(p[:,[0,1]])
        t = e(np.clip(p[:,2] / h, -0.5, 0.5) + 0.5)
        d = d1 + (d2 - d1) * t.reshape((-1, 1))
        w = _vec(d.reshape(-1), np.abs(p[:,2]) - h / 2)
        return _min(_max(w[:,0], w[:,1]), 0) + _length(_max(w, 0))
    return f

def revolve(other, offset=0):
    def f(p):
        xy = p[:,[0,1]]
        q = _vec(_length(xy) - offset, p[:,2])
        return other(q)
    return f
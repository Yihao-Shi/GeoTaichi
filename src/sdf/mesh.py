from skimage import measure

import itertools
import numpy as np

from src.utils.linalg import cartesian_product


def _marching_cubes(volume, level=0):
    verts, faces, _, _ = measure.marching_cubes(volume, level)
    return verts[faces].reshape((-1, 3))


def _skip(sdf, job):
    X, Y, Z = job
    x0, x1 = X[0], X[-1]
    y0, y1 = Y[0], Y[-1]
    z0, z1 = Z[0], Z[-1]
    x = (x0 + x1) / 2
    y = (y0 + y1) / 2
    z = (z0 + z1) / 2
    r = abs(sdf(np.array([(x, y, z)])).reshape(-1)[0])
    d = np.linalg.norm(np.array((x-x0, y-y0, z-z0)))
    if r <= d:
        return False
    corners = np.array(list(itertools.product((x0, x1), (y0, y1), (z0, z1))))
    values = sdf(corners).reshape(-1)
    same = np.all(values > 0) if values[0] > 0 else np.all(values < 0)
    return same


def _worker(sdf, job, sparse):
    X, Y, Z = job
    if sparse and _skip(sdf, job):
        return None
        #return _debug_triangles(X, Y, Z)
    P = cartesian_product(X, Y, Z)
    volume = sdf(P).reshape((len(X), len(Y), len(Z)))

    try:
        points = _marching_cubes(volume)
    except Exception:
        return []
        # return _debug_triangles(X, Y, Z)
    scale = np.array([X[1] - X[0], Y[1] - Y[0], Z[1] - Z[0]])
    offset = np.array([X[0], Y[0], Z[0]])
    return points * scale + offset


def _debug_triangles(X, Y, Z):
    x0, x1 = X[0], X[-1]
    y0, y1 = Y[0], Y[-1]
    z0, z1 = Z[0], Z[-1]

    p = 0.25
    x0, x1 = x0 + (x1 - x0) * p, x1 - (x1 - x0) * p
    y0, y1 = y0 + (y1 - y0) * p, y1 - (y1 - y0) * p
    z0, z1 = z0 + (z1 - z0) * p, z1 - (z1 - z0) * p

    v = [
        (x0, y0, z0),
        (x0, y0, z1),
        (x0, y1, z0),
        (x0, y1, z1),
        (x1, y0, z0),
        (x1, y0, z1),
        (x1, y1, z0),
        (x1, y1, z1),
    ]

    return [
        v[3], v[5], v[7],
        v[5], v[3], v[1],
        v[0], v[6], v[4],
        v[6], v[0], v[2],
        v[0], v[5], v[1],
        v[5], v[0], v[4],
        v[5], v[6], v[7],
        v[6], v[5], v[4],
        v[6], v[3], v[7],
        v[3], v[6], v[2],
        v[0], v[3], v[2],
        v[3], v[0], v[1],
    ]


def sample_slice(sdf, w, h, x, y, z):
    (x0, y0, z0) = sdf.lower_bound
    (x1, y1, z1) = sdf.upper_bound

    if x is not None:
        X = np.array([x])
        Y = np.linspace(y0, y1, w)
        Z = np.linspace(z0, z1, h)
        extent = (Z[0], Z[-1], Y[0], Y[-1])
        axes = 'ZY'
    elif y is not None:
        Y = np.array([y])
        X = np.linspace(x0, x1, w)
        Z = np.linspace(z0, z1, h)
        extent = (Z[0], Z[-1], X[0], X[-1])
        axes = 'ZX'
    elif z is not None:
        Z = np.array([z])
        X = np.linspace(x0, x1, w)
        Y = np.linspace(y0, y1, h)
        extent = (Y[0], Y[-1], X[0], X[-1])
        axes = 'YX'
    else:
        raise Exception('x, y, or z position must be specified')

    P = cartesian_product(X, Y, Z)
    return sdf(P).reshape((w, h)), extent, axes

import bootstrap
import tempfile, os
import numpy as np
from igakit.cad import *
from igakit.io  import *

def mktemp():
    fd, name = tempfile.mkstemp()
    os.close(fd)
    return name

def test_io_1d():
    rw = PetIGA()
    C1 = circle()
    fn = mktemp()
    try:
        for nsd in (2, 3):
            rw.write(fn, C1, nsd=nsd)
            C2 = rw.read(fn)
            for U1, U2 in zip(C1.knots, C2.knots):
                assert np.allclose(U1, U2)
            assert np.allclose(C1.control, C2.control)
    finally:
        os.remove(fn)

def test_io_2d():
    rw = PetIGA()
    L = line((1,0), (2,0))
    S = revolve(L, point=(0,0), axis=2)
    N1 = S
    fn = mktemp()
    try:
        for nsd in (2, 3):
            rw.write(fn, N1, nsd=nsd)
            N2 = rw.read(fn)
            for U1, U2 in zip(N1.knots, N2.knots):
                assert np.allclose(U1, U2)
            assert np.allclose(N1.control, N2.control)
    finally:
        os.remove(fn)

def test_io_3d():
    rw = PetIGA()
    L = line((1,0), (2,0))
    S = revolve(L, point=(0,0), axis=2)
    V = extrude(S, displ=1, axis=2)
    N1 = V
    fn = mktemp()
    try:
        rw.write(fn, N1)
        N2 = rw.read(fn)
        for U1, U2 in zip(N1.knots, N2.knots):
            assert np.allclose(U1, U2)
        assert np.allclose(N1.control, N2.control)
        rw.write_vec(fn, V.array.ravel(), V)
        vec = rw.read_vec(fn, V)
        assert V.array.shape, vec.shape
        assert np.allclose(V.array, vec)
        rw.write_vec(fn, V.weights.ravel(), V)
        vec = rw.read_vec(fn, V)
        assert V.weights.shape, vec.shape
        assert np.allclose(V.weights, vec)
    finally:
        os.remove(fn)

def test_io_vtk():
    rw = VTK()
    L = line((1,0), (2,0)).refine(0, [0.5])
    S = revolve(L, point=(0,0), axis=2, angle=3*Pi/2)
    V = extrude(S, displ=1, axis=2)
    uniform = lambda U: np.linspace(U[0], U[-1], 32)
    fn = mktemp()
    try:
        rw.write(fn, L)
        rw.write(fn, S)
        rw.write(fn, V)
        for N in (L, S, V):
            F = np.random.random(N.shape+(4,))
            for sampler in (None, uniform):
                for control in (False, True):
                    rw.write(fn, N, fields=None,
                             control=control, sampler=sampler)
                    rw.write(fn, N, fields=F,
                             control=control, sampler=sampler,
                             scalars=[(None,0),('',1),("my scalar",2)],
                             vectors=dict(u=[0],v=[0,1],w=[0,1,2]))

            N = NURBS(N.knots, N.control, fields=F)
            for sampler in (None, uniform):
                for control in (False, True):
                    for fields in (None, F):
                        rw.write(fn, N, fields=fields,
                                 control=control, sampler=sampler)
                        rw.write(fn, N, fields=fields,
                                 control=control, sampler=sampler,
                                 scalars=[(None,0),('',1),("my scalar",2)],
                                 vectors=dict(u=[0],v=[0,1],w=[0,1,2]))
                        rw.write(fn, N, fields=fields,
                                 control=control, sampler=sampler,
                                 scalars=[(None,0),('',1),("my scalar",2)],
                                 vectors=dict(u=[0],v=[0,1],w=[0,1,2]))

    finally:
        os.remove(fn)

if __name__ == '__main__':
    test_io_1d()
    test_io_2d()
    test_io_3d()
    test_io_vtk()

import bootstrap
import numpy as np
from igakit.igalib import bsp

def test_fks(VERB=0):
    if VERB: print(bsp.FindSpan.__doc__)
    p = 2
    U = np.asarray([0,0,0, 0.5, 1,1,1])
    for u, k in [ (0.00, 2),
                  (0.25, 2),
                  (0.50, 3),
                  (0.75, 3),
                  (1.00, 3),
                  ]:
        span = bsp.FindSpan(p,U,u)
        assert span == k
    U = np.asarray([-1,0,0, 0.5, 1,1, 2])
    for u, k in [ (0.00, 2),
                  (0.25, 2),
                  (0.50, 3),
                  (0.75, 3),
                  (1.00, 3),
                  ]:
        span = bsp.FindSpan(p,U,u)
        assert span == k
    U = np.asarray([-1,-1,0, 0.5, 1,2,2])
    for u, k in [ (0.00, 2),
                  (0.25, 2),
                  (0.50, 3),
                  (0.75, 3),
                  (1.00, 3),
                  ]:
        span = bsp.FindSpan(p,U,u)
        assert span == k
    U = np.asarray([-2,-1,0, 0.5, 1,2,3])
    for u, k in [ (0.00, 2),
                  (0.25, 2),
                  (0.50, 3),
                  (0.75, 3),
                  (1.00, 3),
                  ]:
        span = bsp.FindSpan(p,U,u)
        assert span == k
    U = np.asarray([0,0,0, 1,1,1], dtype='d')
    for u, k in [ (0.00, 2),
                  (0.50, 2),
                  (1.00, 2),
                  ]:
        span = bsp.FindSpan(p,U,u)
        assert span == k
    U = np.asarray([-1,0,0, 1,1,2], dtype='d')
    for u, k in [ (0.00, 2),
                  (0.50, 2),
                  (1.00, 2),
                  ]:
        span = bsp.FindSpan(p,U,u)
        assert span == k
    U = np.asarray([-1,-1,0, 1,2,2], dtype='d')
    for u, k in [ (0.00, 2),
                  (0.50, 2),
                  (1.00, 2),
                  ]:
        span = bsp.FindSpan(p,U,u)
        assert span == k
    U = np.asarray([-2,-1,0, 1,2,3], dtype='d')
    for u, k in [ (0.00, 2),
                  (0.50, 2),
                  (1.00, 2),
                  ]:
        span = bsp.FindSpan(p,U,u)
        assert span == k
    p = 3
    U = np.asarray([-2,-1,0,0,1,1,2,3], dtype='d')
    for u, k in [ (0.00, 3),
                  (0.50, 3),
                  (1.00, 3),
                  ]:
        span = bsp.FindSpan(p,U,u)
        assert span == k
    U = np.asarray([-2,-1,0,0,0.5,1,1,2,3], dtype='d')
    for u, k in [ (0.00, 3),
                  (0.50, 4),
                  (1.00, 4),
                  ]:
        span = bsp.FindSpan(p,U,u)
        assert span == k


def test_mult(VERB=0):
    if VERB: print(bsp.FindMult.__doc__)
    p = 2
    U = np.asarray([0,0,0, 0.25, 0.25, 0.5, 1,1,1])
    for u, s in [ (0.00, 3),
                  (0.25, 2),
                  (0.50, 1),
                  (0.75, 0),
                  (1.00, 3),
                  ]:
        span = bsp.FindSpan(p,U,u)
        mult1 = bsp.FindMult(p,U,u,span)
        mult2 = bsp.FindMult(p,U,u)
        assert mult2 == mult1
        assert mult2 == s
    U = np.asarray([-1,0,0,0.25, 0.25, 0.5, 1,1,2])
    for u, s in [ (0.00, 2),
                  (0.25, 2),
                  (0.50, 1),
                  (0.75, 0),
                  (1.00, 2),
                  ]:
        span = bsp.FindSpan(p,U,u)
        mult1 = bsp.FindMult(p,U,u,span)
        mult2 = bsp.FindMult(p,U,u)
        assert mult2 == mult1
        assert mult2 == s
    U = np.asarray([-2,-1,0,0.25, 0.25, 0.5, 1,2,3])
    for u, s in [ (0.00, 1),
                  (0.25, 2),
                  (0.50, 1),
                  (0.75, 0),
                  (1.00, 1),
                  ]:
        span = bsp.FindSpan(p,U,u)
        mult1 = bsp.FindMult(p,U,u,span)
        mult2 = bsp.FindMult(p,U,u)
        assert mult2 == mult1
        assert mult2 == s

def test_bf_val(VERB=0, PLOT=0):
    if VERB: print(bsp.EvalBasisFuns.__doc__)
    p = 2
    U = np.asarray([0,0,0, .25, .5, .75, 1,1,1])
    #U = np.asarray([0,0,0, 0.5, 1,1,1], dtype=float)
    #U = np.asarray([0,0,0, 1,1,1], dtype=float)
    X = np.linspace(U[0], U[-1], 200)
    Y = np.zeros((X.size, p+1))
    for i,u in enumerate(X):
        Y[i,:] = bsp.EvalBasisFuns(p,U,u)

    if not PLOT: return
    plt.figure()
    plt.title("Basis - Values")
    plt.plot(X,Y,'-')


def test_bf_der(VERB=0, PLOT=0):
    if VERB: print(bsp.EvalBasisFunsDers.__doc__)
    p = 3
    #U = np.asarray([0,0,0, .25, .5, .75, 1,1,1])
    U = np.zeros(2*(p+1)); U[:p+1]=0; U[p+1:]=1;
    X = np.linspace(U[0], U[-1], 200)
    d = p
    Y = np.zeros((X.size, d+1, p+1))
    for i,u in enumerate(X):
        Y[i,...] = bsp.EvalBasisFunsDers(p,U,u,d)

    if not PLOT: return
    plt.figure()
    plt.title("Basis - Derivatives")
    for i in range(d+1):
        plt.subplot(100*(d+1)+10+(i+1))
        plt.plot(X,Y[:,i,:],'-')

if __name__ == '__main__':
    VERB=1
    try:
        from matplotlib import pylab as plt
        PLOT=1
    except ImportError:
        PLOT=0
    if 1: test_fks(VERB=VERB)
    if 1: test_mult(VERB=VERB)
    if 1: test_bf_val(VERB=VERB,PLOT=PLOT)
    if 1: test_bf_der(VERB=VERB,PLOT=PLOT)
    if PLOT: plt.show()

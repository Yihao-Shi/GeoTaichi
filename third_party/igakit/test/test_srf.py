import bootstrap
import numpy as np
from igakit.nurbs  import NURBS
from igakit.igalib import bsp

def test_srf_ev(PLOT=0):
    p = 2
    q = 1
    U = np.asarray([0,0,0, 1,1,1], dtype=float)
    V = np.asarray([0,0,     1,1], dtype=float)
    n = len(U)-1-(p+1)
    m = len(V)-1-(q+1)
    Pw = np.zeros((n+1,m+1,3))
    Pw[0,0,:] = [0.0, 0.5, 1.0]
    Pw[1,0,:] = [0.5, 0.5, 1.0]
    Pw[2,0,:] = [0.5, 0.0, 1.0]
    Pw[0,1,:] = [0.0, 1.0, 1.0]
    Pw[1,1,:] = [1.0, 1.0, 1.0]
    Pw[2,1,:] = [1.0, 0.0, 1.0]
    Pw[1,:,:] *= np.sqrt(2)/2

    u = np.linspace(U[0], U[-1], 31)
    v = np.linspace(V[0], V[-1], 10)

    Cw = bsp.Evaluate2(p,U,q,V,Pw,u,v)
    Dw = bsp.Evaluate2(p,U,q,V,Pw,U,V)

    P = Pw[:,:,:2] / Pw[:,:,2, None]
    C = Cw[:,:,:2] / Cw[:,:,2, None]
    D = Dw[:,:,:2] / Dw[:,:,2, None]

    if not PLOT: return
    plt.figure()
    plt.title("Surface - Evaluation")

    x1 = C[:,:,0]
    y1 = C[:,:,1]
    plt.plot(x1,y1,'.b')

    x2 = D[:,:,0]
    y2 = D[:,:,1]
    plt.plot(x2,y2,'og')

    x0 = P[:,:,0]
    y0 = P[:,:,1]
    plt.plot(x0,y0,'sr')

    t = np.linspace(0,np.pi/2,100)
    a = np.cos(t)
    b = np.sin(t)
    plt.plot(1.0*a,1.0*b,'-k')
    plt.plot(0.5*a,0.5*b,'-k')

    plt.axis("equal")

def test_srf_ki(PLOT=0):
    p = 2
    q = 1
    U = np.asarray([0,0,0, 1,1,1], dtype=float)
    V = np.asarray([0,0,     1,1], dtype=float)
    n = len(U)-1-(p+1)
    m = len(V)-1-(q+1)
    Pw = np.zeros((n+1,m+1,4))
    Pw[0,0,:] = [0.0, 0.5, 0.0, 1.0]
    Pw[1,0,:] = [0.5, 0.5, 0.0, 1.0]
    Pw[2,0,:] = [0.5, 0.0, 0.0, 1.0]
    Pw[0,1,:] = [0.0, 1.0, 0.0, 1.0]
    Pw[1,1,:] = [1.0, 1.0, 0.0, 1.0]
    Pw[2,1,:] = [1.0, 0.0, 0.0, 1.0]
    Pw[1,:,:] *= np.sqrt(2)/2
    nurbs = NURBS([U,V],Pw)

    X = np.asarray([])
    Y = np.asarray([])
    nrb = nurbs.copy().refine(0,X).refine(1,Y)
    (Ubar, Vbar), Qw = nrb.knots, nrb.control
    assert np.allclose(U,  Ubar)
    assert np.allclose(V,  Vbar)
    assert np.allclose(Pw, Qw)

    X = np.asarray([.25, 0.5])#;X = np.asarray([])
    Y = np.asarray([0.5, .75])#;Y = np.asarray([])
    nrb = nurbs.refine(0,X).refine(1,Y)
    (Ubar, Vbar), Qw = nrb.knots, nrb.control

    u = np.linspace(Ubar[0], Ubar[-1], 31)
    v = np.linspace(Vbar[0], Vbar[-1], 10)
    Cw = bsp.Evaluate2(p,Ubar,q,Vbar,Qw,u,v)

    Q = Qw[:,:,:2] / Qw[:,:,3,None]
    C = Cw[:,:,:2] / Cw[:,:,3,None]

    if not PLOT: return
    plt.figure()
    plt.title("Surface - Knot Insertion")

    x = Q[:,:,0]
    y = Q[:,:,1]
    plt.plot(x,y,'sr')

    x = C[:,:,0]
    y = C[:,:,1]
    plt.plot(x,y,'.b')

    t = np.linspace(0,np.pi/2,100)
    a = np.cos(t)
    b = np.sin(t)
    plt.plot(1.0*a,1.0*b,'-k')
    plt.plot(0.5*a,0.5*b,'-k')

    plt.axis("equal")

def test_srf_de(PLOT=0):
    p = 2
    q = 1
    U = np.asarray([0,0,0, 1,1,1], dtype=float)
    V = np.asarray([0,0,     1,1], dtype=float)
    n = len(U)-1-(p+1)
    m = len(V)-1-(q+1)
    Pw = np.zeros((n+1,m+1,4))
    Pw[0,0,:] = [0.0, 0.5, 0.0, 1.0]
    Pw[1,0,:] = [0.5, 0.5, 0.0, 1.0]
    Pw[2,0,:] = [0.5, 0.0, 0.0, 1.0]
    Pw[0,1,:] = [0.0, 1.0, 0.0, 1.0]
    Pw[1,1,:] = [1.0, 1.0, 0.0, 1.0]
    Pw[2,1,:] = [1.0, 0.0, 0.0, 1.0]
    Pw[1,:,:] *= np.sqrt(2)/2
    nurbs = NURBS([U,V],Pw)

    nrb = nurbs.copy().elevate(0,0).elevate(1,0)
    (Uh, Vh), Qw = nrb.knots, nrb.control
    nrb = nurbs.copy().elevate(0,1).elevate(1,0)
    (Uh, Vh), Qw = nrb.knots, nrb.control
    nrb = nurbs.copy().elevate(0,0).elevate(1,1)
    (Uh, Vh), Qw = nrb.knots, nrb.control

    r, s = 1, 2
    nrb = nurbs.copy().elevate(0,r).elevate(1,s)
    (Uh, Vh), Qw = nrb.knots, nrb.control
    ph = p+r; qh = q+s;
    u = np.linspace(Uh[0], Uh[-1], 31)
    v = np.linspace(Vh[0], Vh[-1], 10)
    Cw = bsp.Evaluate2(ph,Uh,qh,Vh,Qw,u,v)

    Q = Qw[:,:,:2] / Qw[:,:,3, None]
    C = Cw[:,:,:2] / Cw[:,:,3, None]

    if not PLOT: return
    plt.figure()
    plt.title("Surface - Degree Elevation")

    x = Q[:,:,0]
    y = Q[:,:,1]
    plt.plot(x,y,'sr')

    x = C[:,:,0]
    y = C[:,:,1]
    plt.plot(x,y,'.b')

    t = np.linspace(0,np.pi/2,100)
    a = np.cos(t)
    b = np.sin(t)
    plt.plot(1.0*a,1.0*b,'-k')
    plt.plot(0.5*a,0.5*b,'-k')

    plt.axis("equal")

if __name__ == '__main__':
    try:
        from matplotlib import pylab as plt
        PLOT=1
    except ImportError:
        PLOT=0
    if 1: test_srf_ev(PLOT=PLOT)
    if 1: test_srf_ki(PLOT=PLOT)
    if 1: test_srf_de(PLOT=PLOT)
    if PLOT: plt.show()

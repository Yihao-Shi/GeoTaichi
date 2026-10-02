import bootstrap
import numpy as np
from igakit.nurbs  import NURBS
from igakit.igalib import bsp

def test_vol_ki(PLOT=0):

    px = 2
    py = 1
    pz = 1

    Ux = np.asarray([0,0,0, 1,1,1], dtype=float)
    Uy = np.asarray([0,0,     1,1], dtype=float)
    Uz = np.asarray([0,0,     1,1], dtype=float)

    nx = len(Ux)-1-(px+1)
    ny = len(Uy)-1-(py+1)
    nz = len(Uz)-1-(pz+1)

    Pw = np.zeros((nx+1,ny+1,nz+1,4), dtype=float)

    Pw[0,0,0,:] = [0.0, 0.5, 0.0, 1.0]
    Pw[1,0,0,:] = [0.5, 0.5, 0.0, 1.0]
    Pw[2,0,0,:] = [0.5, 0.0, 0.0, 1.0]
    Pw[0,1,0,:] = [0.0, 1.0, 0.0, 1.0]
    Pw[1,1,0,:] = [1.0, 1.0, 0.0, 1.0]
    Pw[2,1,0,:] = [1.0, 0.0, 0.0, 1.0]

    Pw[0,0,1,:] = [0.0, 0.5, 1.0, 1.0]
    Pw[1,0,1,:] = [0.5, 0.5, 1.0, 1.0]
    Pw[2,0,1,:] = [0.5, 0.0, 1.0, 1.0]
    Pw[0,1,1,:] = [0.0, 1.0, 1.0, 1.0]
    Pw[1,1,1,:] = [1.0, 1.0, 1.0, 1.0]
    Pw[2,1,1,:] = [1.0, 0.0, 1.0, 1.0]

    Pw[1,:,:,:] *= np.sqrt(2)/2

    nurbs = NURBS([Ux,Uy,Uz],Pw)

    X = np.asarray([], dtype=float)
    Y = np.asarray([], dtype=float)
    Z = np.asarray([], dtype=float)
    nrb = nurbs.copy().refine(0,X).refine(1,Y).refine(2,Z)
    (Vx, Vy, Vz), Qw = nrb.knots, nrb.control
    assert np.allclose(Ux, Vx)
    assert np.allclose(Uy, Vy)
    assert np.allclose(Uz, Vz)
    assert np.allclose(Pw, Qw)

    X = np.asarray([0.25, 0.50])
    Y = np.asarray([0.50, 0.75])
    Z = np.asarray([0.25, 0.75])
    nrb = nurbs.copy().refine(0,X).refine(1,Y).refine(2,Z)
    (Vx, Vy, Vz), Qw = nrb.knots, nrb.control
    x = np.linspace(Vx[0], Vx[-1], 15)
    y = np.linspace(Vy[0], Vy[-1], 5)
    z = np.linspace(Vy[0], Vy[-1], 4)
    Cw = bsp.Evaluate3(px,Vx,py,Vy,pz,Vz,Qw,x,y,z)

    Q = Qw[:,:,:,:3] / Qw[:,:,:,3,None]
    C = Cw[:,:,:,:3] / Cw[:,:,:,3,None]

    if not PLOT: return
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title("Volume - Knot Insertion")

    x = Q[:,:,:,0].flatten()
    y = Q[:,:,:,1].flatten()
    z = Q[:,:,:,2].flatten()
    ax.scatter(x,y,z, c='r')

    x = C[:,:,:,0].flatten()
    y = C[:,:,:,1].flatten()
    z = C[:,:,:,2].flatten()
    ax.scatter(x,y,z, c='b')

    t = np.linspace(0,np.pi/2,100)
    a = np.cos(t)
    b = np.sin(t)
    ax.plot(1.0*a,1.0*b,0,'-k')
    ax.plot(1.0*a,1.0*b,1,'-k')
    ax.plot(0.5*a,0.5*b,0,'-k')
    ax.plot(0.5*a,0.5*b,1,'-k')

    plt.axis("equal")

def test_vol_de(PLOT=0):

    px = 2
    py = 1
    pz = 1

    Ux = np.asarray([0,0,0, 1,1,1], dtype=float)
    Uy = np.asarray([0,0,     1,1], dtype=float)
    Uz = np.asarray([0,0,     1,1], dtype=float)

    nx = len(Ux)-1-(px+1)
    ny = len(Uy)-1-(py+1)
    nz = len(Uz)-1-(pz+1)

    Pw = np.zeros((nx+1,ny+1,nz+1,4), dtype=float)

    Pw[0,0,0,:] = [0.0, 0.5, 0.0, 1.0]
    Pw[1,0,0,:] = [0.5, 0.5, 0.0, 1.0]
    Pw[2,0,0,:] = [0.5, 0.0, 0.0, 1.0]
    Pw[0,1,0,:] = [0.0, 1.0, 0.0, 1.0]
    Pw[1,1,0,:] = [1.0, 1.0, 0.0, 1.0]
    Pw[2,1,0,:] = [1.0, 0.0, 0.0, 1.0]

    Pw[0,0,1,:] = [0.0, 0.5, 1.0, 1.0]
    Pw[1,0,1,:] = [0.5, 0.5, 1.0, 1.0]
    Pw[2,0,1,:] = [0.5, 0.0, 1.0, 1.0]
    Pw[0,1,1,:] = [0.0, 1.0, 1.0, 1.0]
    Pw[1,1,1,:] = [1.0, 1.0, 1.0, 1.0]
    Pw[2,1,1,:] = [1.0, 0.0, 1.0, 1.0]

    Pw[1,:,:,:] *= np.sqrt(2)/2

    nurbs = NURBS([Ux,Uy,Uz],Pw)

    nrb = nurbs.copy().elevate(0,0).elevate(1,0).elevate(2,0)
    (Vx, Vy, Vz), Qw = nrb.knots, nrb.control
    assert np.allclose(Ux, Vx)
    assert np.allclose(Uy, Vy)
    assert np.allclose(Uz, Vz)
    assert np.allclose(Pw, Qw)

    tx, ty, tz = 1, 2, 2
    nrb = nurbs.copy().elevate(0,tx).elevate(1,ty).elevate(2,tz)
    (Vx, Vy, Vz), Qw = nrb.knots, nrb.control
    qx = px + tx
    qy = py + ty
    qz = pz + tz

    x = np.linspace(Vx[0], Vx[-1], 15)
    y = np.linspace(Vy[0], Vy[-1], 5)
    z = np.linspace(Vz[0], Vz[-1], 4)
    Cw = bsp.Evaluate3(qx,Vx,qy,Vy,qz,Vz,Qw,x,y,z)

    Q = Qw[:,:,:,:3] / Qw[:,:,:,3,None]
    C = Cw[:,:,:,:3] / Cw[:,:,:,3,None]

    if not PLOT: return
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title("Volume - Degree Elevation")

    x = Q[:,:,:,0].flatten()
    y = Q[:,:,:,1].flatten()
    z = Q[:,:,:,2].flatten()
    ax.scatter(x,y,z, c='r')

    x = C[:,:,:,0].flatten()
    y = C[:,:,:,1].flatten()
    z = C[:,:,:,2].flatten()
    ax.scatter(x,y,z, c='b')

    t = np.linspace(0,np.pi/2,100)
    a = np.cos(t)
    b = np.sin(t)
    ax.plot(1.0*a,1.0*b,0,'-k')
    ax.plot(1.0*a,1.0*b,1,'-k')
    ax.plot(0.5*a,0.5*b,0,'-k')
    ax.plot(0.5*a,0.5*b,1,'-k')

    plt.axis("equal")


if __name__ == '__main__':
    try:
        from matplotlib import pylab as plt
        from mpl_toolkits.mplot3d import Axes3D
        PLOT=1
    except ImportError:
        PLOT=0
    if 1: test_vol_ki(PLOT=PLOT)
    if 1: test_vol_de(PLOT=PLOT)
    if PLOT: plt.show()

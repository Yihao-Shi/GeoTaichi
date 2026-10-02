import bootstrap
import numpy as np
from igakit.igalib import iga
from igakit.igalib import bsp

def test_iga_bd1(VERB=0, PLOT=0):
    if VERB: print(iga.BasisData.__doc__)
    p = 3
    U = np.asarray([0,0,0,0,
                    .2, .3, .4,
                    .5,.5,
                    .6, .7,
                    .8,.8,.8,
                    .9,
                    1,1,1,1])
    m = len(U)-1
    n = m-p-1

    for d in range(p+1):
        for q in range(1,10+1):
            O,J,W,X,N = iga.BasisData(p,U,d=d,q=q)
            assert O[ 0] == 0
            assert O[-1] == n-p
            assert O.shape == (9,)
            assert J.shape == (9,)
            assert W.shape == (q,)
            assert X.shape == (9, q)
            assert N.shape == (9, q, p+1, d+1)
    if not PLOT:return
    d=p
    q=10
    O,J,W,X,N = iga.BasisData(p,U,d=d,q=q)
    args = []
    for i in range(O.size):
        args.append(X[i])
        args.append(N[i,:,:,0])
        args.append('-o')
    fig = plt.figure()
    plt.plot(*args)


def test_iga_bd2(VERB=0, PLOT=0):
    if VERB: print(iga.BasisDataCol.__doc__)
    p = 3
    U = np.asarray([0,0,0,0,
                    .2, .3, .4,
                    .5,.5,
                    .6, .7,
                    .8,.8,.8,
                    .9,
                    1,1,1,1])
    m = len(U)-1
    n = m-p-1

    X = iga.Greville(p,U)
    assert X.shape == (n+1,)
    for d in range(p+1):
        O, N = iga.BasisDataCol(p,U,X,d)
        assert O.shape == (n+1,)
        assert N.shape == (n+1, p+1, d+1)
    if not PLOT:return
    fig = plt.figure()

    X = iga.Greville(p,U)
    O, N = iga.BasisDataCol(p,U,X)
    assert O.shape == (n+1,)
    assert N.shape == (n+1, p+1, p+1)
    args = []
    args.append(X)
    args.append(N[:,:,0])
    args.append('o')
    plt.plot(*args)

    X = np.linspace(0,1,200)
    O, N = iga.BasisDataCol(p,U,X)
    args.append(X)
    args.append(N[:,:,0])
    args.append('-')
    plt.plot(*args)


def test_fem_2d(VERB=0, PLOT=0):
    """
     -Laplacian(u(x,y)) = f(x,y)  in  0 < x,y < 1
                u(x,y)  = 0       on  x,y = 0,1
    """

    def forcing(x,y):
        return x**2 + y**2

    # geometry, order & continuity
    nelx = 5; px = 2; Cx = px-1
    nely = 5; py = 2; Cy = py-1
    Ux = iga.KnotVector(nelx,px,Cx)
    Uy = iga.KnotVector(nely,py,Cy)
    nx = len(Ux)-(px+1)
    ny = len(Uy)-(py+1)

    # quadrature and basis data
    nqpx = 3
    nqpy = 3
    Ox,Jx,Wx,X,Nx = iga.BasisData(px,Ux,d=1,q=nqpx)
    Oy,Jy,Wy,Y,Ny = iga.BasisData(py,Uy,d=1,q=nqpy)

    # global matrix and vector
    K = np.zeros((nx,ny,nx,ny))
    F = np.zeros((nx,ny))

    # element loop
    for ex in range(nelx):
        for ey in range(nely):
            Ke = np.zeros((px+1,py+1,px+1,py+1))
            Fe = np.zeros((px+1,py+1))
            # quadrature loop
            for qx in range(nqpx):
                for qy in range(nqpy):
                    # basis functions
                    N0 = np.zeros((px+1,py+1))
                    N1 = np.zeros((px+1,py+1,2))
                    for ax in range(px+1):
                        for ay in range(py+1):
                            N0[ax,ay]   = Nx[ex,qx,ax,0]*Ny[ey,qy,ay,0]
                            N1[ax,ay,0] = Nx[ex,qx,ax,1]*Ny[ey,qy,ay,0]
                            N1[ax,ay,1] = Nx[ex,qx,ax,0]*Ny[ey,qy,ay,1]
                    # stiffness matrix
                    Kab = np.zeros((px+1,py+1,px+1,py+1))
                    for ax in range(px+1):
                        for ay in range(py+1):
                            Na_x, Na_y = N1[ax,ay]
                            for bx in range(px+1):
                                for by in range(py+1):
                                    Nb_x, Nb_y = N1[bx,by]
                                    Kab[ax,ay,bx,by] = Na_x*Nb_x + Na_y*Nb_y
                    # force vector
                    x, y = X[ex,qx], Y[ey,qy]
                    Fa = np.zeros((px+1,py+1))
                    for ax in range(px+1):
                        for ay in range(py+1):
                            Na = N0[ax,ay]
                            Fa[ax,ay] = Na * forcing(x,y)
                    #
                    J = Jx[ex]*Jy[ey]
                    W = Wx[qx]*Wy[qy]
                    Ke += Kab * J*W
                    Fe += Fa  * J*W
            # global matrix and vector assembly
            ox = Ox[ex]
            oy = Oy[ey]
            for ax in range(px+1):
                for ay in range(py+1):
                    Ax = ox + ax
                    Ay = oy + ay
                    for bx in range(px+1):
                        for by in range(py+1):
                            Bx = ox + bx
                            By = oy + by
                            K[Ax,Ay,Bx,By] += Ke[ax,ay,bx,by]
                    F[Ax,Ay] += Fe[ax,ay]

    # boundary conditions (homogeneous)
    for Ax in (0, -1):
        for Ay in range(ny):
            K[Ax,Ay,  :, :] = 0
            K[ :, :, Ax,Ay] = 0
            K[Ax,Ay, Ax,Ay] = 1
            F[Ax,Ay] = 0
    for Ay in (0, -1):
        for Ax in range(nx):
            K[Ax,Ay,  :, :] = 0
            K[ :, :, Ax,Ay] = 0
            K[Ax,Ay, Ax,Ay] = 1
            F[Ax,Ay] = 0

    # solve linear system
    K.shape = (nx*ny,nx*ny)
    F.shape = (nx*ny,)
    X = np.linalg.solve(K,F)
    X.shape = (nx,ny)

    # interpolate solution
    x = np.linspace(Ux[0],Ux[-1],25)
    y = np.linspace(Uy[0],Uy[-1],25)
    z = bsp.Evaluate2(px,Ux,py,Uy,X,x,y)
    x, y = np.meshgrid(x,y)
    z.shape = z.shape[:-1]

    # surface plot solution
    if not PLOT: return
    from matplotlib import cm
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title("Poisson 2D (FEM)")
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.plot_surface(x,y,z, rstride=1, cstride=1,
                    cmap=cm.jet, antialiased=True)

def test_col_2d(VERB=0, PLOT=0):
    """
     -Laplacian(u(x,y)) = f(x,y)  in  0 < x,y < 1
                u(x,y)  = 0       on  x,y = 0,1
    """

    def forcing(x,y):
        #eturn 1
        return x**2 + y**2

    # geometry, order & continuity
    nelx = 5; px = 2; Cx = px-1
    nely = 5; py = 2; Cy = py-1
    Ux = iga.KnotVector(nelx,px,Cx,)
    Uy = iga.KnotVector(nely,py,Cy,)
    nx = len(Ux)-(px+1)
    ny = len(Uy)-(py+1)

    # basis functions
    X = iga.Greville(px,Ux)
    Y = iga.Greville(px,Uy)
    Ox, Nx = iga.BasisDataCol(px,Ux,X,d=2)
    Oy, Ny = iga.BasisDataCol(py,Uy,Y,d=2)

    # global matrix and vector
    K = np.zeros((nx,ny,nx,ny))
    F = np.zeros((nx,ny))

    # point loop
    for Ax in range(nx):
        for Ay in range(ny):
            ox = Ox[Ax]
            oy = Oy[Ay]
            for bx in range(px+1):
                for by in range(py+1):
                    N_xx = Nx[Ax,bx,2] * Ny[Ay,by,0]
                    N_yy = Nx[Ax,bx,0] * Ny[Ay,by,2]

                    Bx = ox + bx
                    By = oy + by
                    K[Ax,Ay,Bx,By] += -(N_xx + N_yy)


            x = X[Ax]
            y = Y[Ay]
            f = forcing(x,y)
            F[Ax,Ay] = f


    # boundary conditions (homogeneous)
    for Ax in (0, -1):
        for Ay in range(ny):
            K[Ax,Ay,  :, :] = 0
            K[ :, :, Ax,Ay] = 0
            K[Ax,Ay, Ax,Ay] = 1
            F[Ax,Ay] = 0
    for Ay in (0, -1):
        for Ax in range(nx):
            K[Ax,Ay,  :, :] = 0
            K[ :, :, Ax,Ay] = 0
            K[Ax,Ay, Ax,Ay] = 1
            F[Ax,Ay] = 0

    # solve linear system
    K.shape = (nx*ny,nx*ny)
    F.shape = (nx*ny,)
    X = np.linalg.solve(K,F)
    X.shape = (nx,ny)

    # interpolate solution
    x = np.linspace(Ux[0],Ux[-1],25)
    y = np.linspace(Uy[0],Uy[-1],25)
    z = bsp.Evaluate2(px,Ux,py,Uy,X,x,y)
    x, y = np.meshgrid(x,y)
    z.shape = z.shape[:-1]

    # surface plot solution
    if not PLOT: return
    from matplotlib import cm
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title("Poisson 2D (Collocation)")
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.plot_surface(x,y,z, rstride=1, cstride=1,
                    cmap=cm.jet, antialiased=True)


if __name__ == '__main__':
    VERB=1
    try:
        from matplotlib import pylab as plt
        from mpl_toolkits.mplot3d import Axes3D
        PLOT=1
    except ImportError:
        PLOT=0
    if 1: test_iga_bd1(VERB=VERB, PLOT=PLOT)
    if 1: test_iga_bd2(VERB=VERB, PLOT=PLOT)
    if 1: test_fem_2d (VERB=VERB, PLOT=PLOT)
    if 1: test_col_2d (VERB=VERB, PLOT=PLOT)
    if PLOT: plt.show()

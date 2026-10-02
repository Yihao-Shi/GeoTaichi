from igakit.nurbs import NURBS
from igakit.cad import ruled
from igakit.igalib import iga
import numpy as np

def ConformalProjectionError(srf):
    """
    Computes the integral of the squared sum of the Cauchy-Riemann
    equations, a measure of how conformal the map is:

    sqrt( int( (dx/du-dy/dv)^2 + (dx/dv+dy/du)^2 ) )

    Parameters
    ----------
    srf : two-dimensional NURBS object

    Returns
    -------
    error : float, value of the integral

    """
    W   = srf.control[:,:,-1]

    Ou,Ju,Wu,Xu,Nu = iga.BasisData(srf.degree[0],srf.knots[0])
    Ov,Jv,Wv,Xv,Nv = iga.BasisData(srf.degree[1],srf.knots[1])

    outer = np.outer
    error = 0.
    for eu in range(len(Ou)):
        for ev in range(len(Ov)):

            Lu,Ru = Ou[eu],Ou[eu]+srf.degree[0]+1
            Lv,Rv = Ov[ev],Ov[ev]+srf.degree[1]+1
            Wl    = W[Lu:Ru,Lv:Rv]

            for qu in range(len(Wu)):
                for qv in range(len(Wv)):

                    N   = outer(Nu[eu,qu,:,0],Nv[ev,qv,:,0])
                    dNu = outer(Nu[eu,qu,:,1],Nv[ev,qv,:,0])
                    dNv = outer(Nu[eu,qu,:,0],Nv[ev,qv,:,1])

                    w   = (  N*Wl).sum()
                    dwu = (dNu*Wl).sum()
                    dwv = (dNv*Wl).sum()

                    R   =  N*Wl / w
                    dRu = (-R*dwu + dNu*Wl) / w
                    dRv = (-R*dwv + dNv*Wl) / w

                    dXdu = (srf.control[Lu:Ru,Lv:Rv,0]*dRu).sum()
                    dXdv = (srf.control[Lu:Ru,Lv:Rv,0]*dRv).sum()
                    dYdu = (srf.control[Lu:Ru,Lv:Rv,1]*dRu).sum()
                    dYdv = (srf.control[Lu:Ru,Lv:Rv,1]*dRv).sum()

                    wgt = Ju[eu]*Ju[eu]*Wu[qu]*Wv[qv]
                    error += ( (dXdu-dYdv)**2 + (dXdv+dYdu)**2 )*wgt

    return np.sqrt(error)

def QuasiConformalProjection(srf):
    """
    Reparameterize a surface to be a near conformal. Solves for
    geometric coefficients which minimizes the squared sum of the
    Cauchy-Riemann equations,

    argmin ( int( (dx/du-dy/dv)^2 + (dx/dv+dy/du)^2 ) )

    subject to the condition that the boundary curves do not change.

    Parameters
    ----------
    srf : two-dimensional NURBS object

    Returns
    -------
    conf_srf : NURBS, the reparameterized surface

    """
    W   = srf.control[:,:,-1]

    Ou,Ju,Wu,Xu,Nu = iga.BasisData(srf.degree[0],srf.knots[0])
    Ov,Jv,Wv,Xv,Nv = iga.BasisData(srf.degree[1],srf.knots[1])

    shift = srf.shape[0]*srf.shape[1]
    M = np.zeros((2*shift,2*shift))
    F = np.zeros(2*shift)

    outer = np.outer
    for eu in range(len(Ou)):
        for ev in range(len(Ov)):

            Lu,Ru = Ou[eu],Ou[eu]+srf.degree[0]+1
            Lv,Rv = Ov[ev],Ov[ev]+srf.degree[1]+1
            Wl    = W[Lu:Ru,Lv:Rv]

            for qu in range(len(Wu)):
                for qv in range(len(Wv)):

                    N   = outer(Nu[eu,qu,:,0],Nv[ev,qv,:,0])
                    dNu = outer(Nu[eu,qu,:,1],Nv[ev,qv,:,0])
                    dNv = outer(Nu[eu,qu,:,0],Nv[ev,qv,:,1])

                    w   = (  N*Wl).sum()
                    dwu = (dNu*Wl).sum()
                    dwv = (dNv*Wl).sum()

                    R   =  N*Wl / w
                    dRu = (-R*dwu + dNu*Wl) / w
                    dRv = (-R*dwv + dNv*Wl) / w

                    dRu = dRu.reshape((-1,1),order='f')
                    dRv = dRv.reshape((-1,1),order='f')

                    wgt = Ju[eu]*Ju[eu]*Wu[qu]*Wv[qv]
                    m00 = ( outer(dRu,dRu)+outer(dRv,dRv))*wgt
                    m01 = (-outer(dRu,dRv)+outer(dRv,dRu))*wgt

                    ind  = outer(np.ones(Ru-Lu,dtype='int'),range(Lv,Rv))*srf.shape[0]
                    ind += outer(range(Lu,Ru),np.ones(Rv-Lv,dtype='int'))
                    ind  = ind.reshape((-1,1),order='f')

                    M[      ind,      ind[:,0]] += m00
                    M[shift+ind,      ind[:,0]] -= m01
                    M[      ind,shift+ind[:,0]] += m01
                    M[shift+ind,shift+ind[:,0]] += m00

    for side in [0,srf.shape[1]-1]:
        for dof in range(srf.shape[0]):
            for c in range(2):
                col = c*shift + side*srf.shape[0] + dof
                F -= M[:,col]*srf.control[dof,side,c]
                M[:,col] = M[col,:] = 0
                M[col,col] = 1
                F[col] = srf.control[dof,side,c]

    for side in [0,srf.shape[0]-1]:
        for dof in range(1,srf.shape[1]-1):
            for c in range(2):
                col = c*shift + dof*srf.shape[0] + side
                F -= M[:,col]*srf.control[side,dof,c]
                M[:,col] = M[col,:] = 0
                M[col,col] = 1
                F[col] = srf.control[side,dof,c]

    U = np.linalg.solve(M,F)

    conf_srf = srf.copy()
    conf_srf.control[:,:,0] = U[:shift].reshape((conf_srf.shape[0],conf_srf.shape[1]),order='f')
    conf_srf.control[:,:,1] = U[shift:].reshape((conf_srf.shape[0],conf_srf.shape[1]),order='f')

    return conf_srf

if __name__ == "__main__":

    # generates two random curves
    cp = np.outer([0,0.3,0.5,0.6,1.0],[1,1])
    cp[:,1] = np.random.rand(5)
    c1 = NURBS([[0,0,0,0,0.5,1,1,1,1]],cp)
    cp[:,0] += 0.3
    cp[:,1] = 1.+np.random.rand(5)
    c2 = NURBS([[0,0,0,0,0.5,1,1,1,1]],cp)

    # create a surface by linearly blending curves, then refine
    srf = ruled(c1,c2)
    srf.elevate(1,1)
    srf.refine(0,np.linspace(0,0.5,6)[1:-1])
    srf.refine(0,np.linspace(0.5,1.0,6)[1:-1])
    srf.refine(1,np.linspace(0,1,11)[1:-1])

    # make the projection
    err0 = ConformalProjectionError(srf)
    srf = QuasiConformalProjection(srf)
    print "Relative error: %.6e" % (ConformalProjectionError(srf)/err0)
    from igakit.io import VTK
    VTK().write("conformal.vtk", srf)

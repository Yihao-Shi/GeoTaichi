import numpy as np
from igakit.nurbs import NURBS, transform
from igakit.plot import plt

def CntRev(C, axis):
    dim = C.ndim - 1
    index = list(np.index_exp[:,:,:][:dim])
    index[axis] = slice(None, None, -1)
    return C[index]

def VolumifyInterior(nrb):
    """
    Creates volumes from closed surfaces.

    Given a tube-like, closed surface, return a nurbs volume of the
    interior. Actually returns 5 volumes, one of the core and 4 which
    connect the core to the outer hull. Assumes that the
    circumferential direction is 1st direction

    ..note: Lisandro, for now I am just making this work for the pipe
    example, but several things will need to be made more general.

    """
    assert nrb.dim == 2
    """
    Step 1: extract 4 C0 surfaces from the single input surface. I
    propose that we simple insert knots to extract at
    [0.25,0.5,0.75]*U[-1]. Note that in this case, these knots are
    already inserted and so I skip this part.
    """

    """
    Step 2: locate the control point id's which corresponds to the C0
    lines we just inserted. Create a trilinear solid which will form
    the core of the returned volume. Degree elevate to make equal to
    the 1st dim of the input surface.
    """
    fct = 0.75
    c0 = 0; c1 = 2; c2 = 4; c3 = 6; c4 = 8
    C = np.zeros((2,2,nrb.shape[1],4))
    C[0,0,:,:] = fct*nrb.control[c0,:,:] + (1.0-fct)*nrb.control[c2,:,:]
    C[1,1,:,:] = (1.0-fct)*nrb.control[c0,:,:] + fct*nrb.control[c2,:,:]
    C[1,0,:,:] = fct*nrb.control[c1,:,:] + (1.0-fct)*nrb.control[c3,:,:]
    C[0,1,:,:] = (1.0-fct)*nrb.control[c1,:,:] + fct*nrb.control[c3,:,:]
    core = NURBS(C,[[0,0,1,1],[0,0,1,1],nrb.knots[1]])
    t=max(0,nrb.degree[0]-core.degree[0])
    core.elevate(t,t,0)

    """
    Step 3: create 4 volumes around the core.
    """
    D1 = np.zeros((3,2,nrb.shape[1],4))
    D1[:,0,:,:] = core.control[:,0,:,:]
    D1[:,1,:,:] = nrb.control[c0:c1+1,:,:]
    v1 = NURBS(D1,[[0,0,0,1,1,1],[0,0,1,1],nrb.knots[1]])
    D2 = np.zeros((3,2,nrb.shape[1],4))
    D2[:,0,:,:] = core.control[2,:,:,:]
    D2[:,1,:,:] = nrb.control[c1:c2+1,:,:]
    v2 = NURBS(D2,[[0,0,0,1,1,1],[0,0,1,1],nrb.knots[1]])
    D3 = np.zeros((3,2,nrb.shape[1],4))
    D3[:,0,:,:] = core.control[:,2,:,:]
    D3[:,1,:,:] = CntRev(nrb.control[c2:c3+1,:,:],0)
    v3 = NURBS(D3,[[0,0,0,1,1,1],[0,0,1,1],nrb.knots[1]])
    D4 = np.zeros((3,2,nrb.shape[1],4))
    D4[:,0,:,:] = core.control[0,:,:,:]
    D4[:,1,:,:] = CntRev(nrb.control[c3:c4+1,:,:],0)
    v4 = NURBS(D4,[[0,0,0,1,1,1],[0,0,1,1],nrb.knots[1]])
    return core,v1,v2,v3,v4

L  = 3.0
R0 = 1.0
R1 = 2.0
Rb = R0+R1

circle = np.zeros((9,4), dtype='d')
circle[:,:2] = [[ 1, 0],
                [ 1, 1],
                [ 0, 1],
                [-1, 1],
                [-1, 0],
                [-1,-1],
                [ 0,-1],
                [ 1,-1],
                [ 1, 0]]
circle[:,-1] = 1
circle[1::2,:] *= np.sqrt(2)/2

bentpipe = np.zeros((9,5,4), dtype='d')
t0 = transform().move(L, 2)
t1 = transform().move(L/2, 2)
t2 = transform().move(0, 2)
t3 = transform()
t3.scale([np.sqrt(2), 1, np.sqrt(2)])
t3.rotate(-np.pi/4, 1).move(-Rb, 2)
t4 = transform()
t4.rotate(-np.pi/2, 1).move([Rb, 0, -Rb])
bentpipe[:,0,:] = t0(circle)
bentpipe[:,1,:] = t1(circle)
bentpipe[:,2,:] = t2(circle)
bentpipe[:,3,:] = t3(circle)
bentpipe[:,4,:] = t4(circle)

Pw = bentpipe
U = [0,0,0, 1,1, 2,2, 3,3, 4,4,4]
W = [0,0,0, 1,1, 2,2,2]
nrb = NURBS(Pw, [U,W])


import sys
try:
    backend = sys.argv[1]
    plt.use(backend)
except IndexError:
    pass

plt.figure()
plt.plot(nrb)

n1,n2,n3,n4,n5 = VolumifyInterior(nrb)
plt.figure()
for i in range(1,6):
    s = 'plt.plot(n%d)' % i
    eval(s)
    s = 'plt.kplot(n%d)' % i
    eval(s)
plt.show()

import numpy as np
from igakit.nurbs import NURBS
from igakit.plot  import plt

def InsertUniformly(U,n):
    """
    Find knots to uniformly add to U.

    Given a knot vector U and the number of uniform spans desired,
    find the knots which need to be inserted. 

    Parameters
    ----------
    U : numpy.ndarray
        original knot vector for a C^p-1 space
    n : int
        target number of uniformly-spaced knot spans

    Returns
    -------
    Knots to be inserted into U
    """
    # U, input knot vector
    # n, total number of uniform spans
    U0 = U
    dU=(U.max()-U.min())/float(n) # target dU in knot vector
    idone=0
    while idone == 0:
        # Add knots in middle of spans which are too large
        Uadd=[]
        for i in range(len(U)-1):
            if U[i+1]-U[i] > dU:
                Uadd.append(0.5*(U[i+1]+U[i]))
        # Now we add these knots (once only, assumes C^(p-1))
        if len(Uadd) > 0: 
            U = np.sort(np.concatenate([U,np.asarray(Uadd)]))
        else:
            idone=1
        # And now a little Laplacian smoothing
        for num_iterations in range(5):
            for i in range(len(U)-2):
                if abs(U0[U0.searchsorted(U[i+1])]-U[i+1]) > 1.0e-14:
                    U[i+1] = 0.5*(U[i]+U[i+2])
    return np.setdiff1d(U,U0)

# Initial data for venturi tube. Unfortunately I have lost my initial
# steps. This is as far back as I can find data for.
U=np.asarray([0.0, 0.0, 0.0, 
               0.0322, 0.05, 0.064, 0.085, 0.121, 0.126, 0.179, 0.21, 0.23, 0.297, 0.469, 
               1.0, 1.0, 1.0])
V=np.asarray([0.0, 0.0, 1.0, 1.0])
P = np.zeros((len(U)-3,len(V)-2,2))
P[0,0,:] = [-152.087, 0.000]
P[1,0,:] = [-131.589, 0.000]
P[2,0,:] = [-99.6673, 0.000]
P[3,0,:] = [-79.2520, 0.000]
P[4,0,:] = [-54.6776, 0.000]
P[5,0,:] = [-23.0025, 8.67713]
P[6,0,:] = [1.72793, 16.5735]
P[7,0,:] = [41.8260, 11.9746]
P[8,0,:] = [93.6736, 3.35605]
P[9,0,:] = [125.273, -2.09856]
P[10,0,:] = [179.309, -11.6477]
P[11,0,:] = [328.972, -40.1282]
P[12,0,:] = [774.725, -89.9008]
P[13,0,:] = [1120.1, -119.228]
P[0,1,:] = [-152.087, 50.0807]
P[1,1,:] = [-131.579, 50.0807]
P[2,1,:] = [-99.7317, 50.0812]
P[3,1,:] = [-79.4763, 50.0818]
P[4,1,:] = [-57.1821, 50.0826]
P[5,1,:] = [-20.8732, 50.0842]
P[6,1,:] = [5.24524, 50.0859]
P[7,1,:] = [42.1936, 50.0884]
P[8,1,:] = [95.7095, 50.0937]
P[9,1,:] = [128.203, 50.0974]
P[10,1,:] = [183.546, 47.168]
P[11,1,:] = [335.568, 38.9067]
P[12,1,:] = [782.636, 12.9908]
P[13,1,:] = [1120.1, -10.3521]

# Adjust control points such that the length scale is 1
delx=P[:,:,0].max()-P[:,:,0].min()
minx=P[:,:,0].min()
P -= minx
P /= delx

tube = NURBS(P,[U,V])
tube.elevate(0,1)

# Determine which knots to insert
res = 32
insert_U=InsertUniformly(U,4*res)
insert_V=InsertUniformly(V,res)
tube.refine(insert_U,insert_V)

plt.use('mayavi')

plt.figure()
plt.plot(tube)

plt.show()

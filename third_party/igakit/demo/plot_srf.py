import numpy as np
from igakit.nurbs import NURBS
from igakit.plot import plt

def make_srf():
    C = np.zeros((3,5,5))
    C[:,:,0] = [[ 0.0,  3.0,  5.0,  8.0, 10.0],
                [ 0.0,  0.0,  0.0,  0.0,  0.0],
                [ 2.0,  2.0,  7.0,  7.0,  8.0],]
    C[:,:,1] = [[ 0.0,  3.0,  5.0,  8.0, 10.0],
                [ 3.0,  3.0,  3.0,  3.0,  3.0],
                [ 0.0,  0.0,  5.0,  5.0,  7.0],]
    C[:,:,2] = [[ 0.0,  3.0,  5.0,  8.0, 10.0],
                [ 5.0,  5.0,  5.0,  5.0,  5.0],
                [ 0.0,  0.0,  5.0,  5.0,  7.0],]
    C[:,:,3] = [[ 0.0,  3.0,  5.0,  8.0, 10.0],
                [ 8.0,  8.0,  8.0,  8.0,  8.0],
                [ 5.0,  5.0,  8.0,  8.0, 10.0],]
    C[:,:,4] = [[ 0.0,  3.0,  5.0,  8.0, 10.0],
                [10.0, 10.0, 10.0, 10.0, 10.0],
                [ 5.0,  5.0,  8.0,  8.0, 10.0],]
    C = C.transpose()
    U = [0, 0, 0, 1/3., 2/3., 1, 1, 1,]
    V = [0, 0, 0, 1/3., 2/3., 1, 1, 1,]
    srf = NURBS([U,V], C)
    return srf

import sys
try:
    backend = sys.argv[1]
    plt.use(backend)
except IndexError:
    pass

srf = make_srf()

plt.figure()
plt.cpoint(srf)
plt.cwire(srf)

plt.figure()
plt.kpoint(srf)
plt.kwire(srf)

plt.figure()
plt.curve(srf)
plt.figure()
plt.surface(srf)

plt.figure()
plt.cplot(srf)
plt.kplot(srf)
plt.plot(srf)

plt.show()

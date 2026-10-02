import numpy as np
from igakit.nurbs import NURBS
from igakit.plot import plt

def make_vol():
    C = np.zeros((3,2,3,4), dtype='d')
    C[0,0,:,0:2] = [0.0, 0.5]
    C[1,0,:,0:2] = [0.5, 0.5]
    C[2,0,:,0:2] = [0.5, 0.0]
    C[0,1,:,0:2] = [0.0, 1.0]
    C[1,1,:,0:2] = [1.0, 1.0]
    C[2,1,:,0:2] = [1.0, 0.0]
    C[:,:,0,2] = 0.0
    C[:,:,1,2] = 0.5
    C[:,:,2,2] = 1.0
    C[:,:,:,3] = 1.0
    C[1,:,:,:] *= np.sqrt(2)/2
    U = [0,0,0,     1,1,1]
    V = [0,0,         1,1]
    W = [0,0,   0.5,  1,1]
    vol = NURBS([U,V,W], C)
    return vol

import sys
try:
    backend = sys.argv[1]
    plt.use(backend)
except IndexError:
    pass

vol = make_vol()

plt.figure()
plt.cpoint(vol)
plt.cwire(vol)

plt.figure()
plt.kpoint(vol)
plt.kwire(vol)

plt.figure()
plt.curve(vol)

plt.figure()
plt.surface(vol)

plt.figure()
plt.cplot(vol)
plt.kplot(vol)
plt.plot(vol)

plt.show()

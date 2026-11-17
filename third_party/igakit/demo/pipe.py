import numpy as np
from igakit.nurbs import NURBS
from igakit.plot import plt

circle = np.array(
    [[ 1, 0],
     [ 1, 1],
     [ 0, 1],
     [-1, 1],
     [-1, 0],
     [-1,-1],
     [ 0,-1],
     [ 1,-1],
     [ 1, 0]],
    dtype='d')

R0 = 0.5
R1 = 1.0
L  = 4.0

C = np.zeros((9,2,2,4), dtype='d')
C[:,0,0,:2] = circle*R0
C[:,1,0,:2] = circle*R1
C[:,:,0, 2] = 0
C[:,:,1,:2] = C[:,:,0,:2]
C[:,:,1, 2] = L

C[:,:,:,-1] = 1
C[1::2,:,:,:] *= np.sqrt(2)/2


U = [0,0,0, 1,1, 2,2, 3,3, 4,4,4]
V = [0,0, 1,1]
W = [0,0, 1,1]

nrb = NURBS([U,V,W], C)

#nrb1 = nrb.clone().elevate(0,1,1)
#nrb2 = nrb.clone().refine([],[0.5],[0.25, 0.5, 0.75])

import sys
try:
    backend = sys.argv[1]
    plt.use(backend)
except IndexError:
    pass
#plt.use('mayavi')
#plt.use('matplotlib')

plt.figure()
plt.cpoint(nrb)
plt.cwire(nrb)

plt.figure()
plt.kpoint(nrb)
plt.kwire(nrb)

plt.figure()
plt.curve(nrb)
plt.figure()
plt.surface(nrb)

plt.figure()
plt.cplot(nrb)
plt.kplot(nrb)
plt.plot(nrb)

plt.show()

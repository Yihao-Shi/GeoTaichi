import numpy as np
from igakit.nurbs import NURBS
from igakit.plot import plt

def make_crv():
    C = [[ 6.0, 0.0, 6.0],
         [-5.5, 0.5, 5.5],
         [-5.0, 1.0,-5.0],
         [ 4.5, 1.5,-4.5],
         [ 4.0, 2.0, 4.0],
         [-3.5, 2.5, 3.5],
         [-3.0, 3.0,-3.0],
         [ 2.5, 3.5,-2.5],
         [ 2.0, 4.0, 2.0],
         [-1.5, 4.5, 1.5],
         [-1.0, 5.0,-1.0],
         [ 0.5, 5.5,-0.5],
         [ 0.0, 6.0, 0.0],]
    U = [0, 0, 0, 0,
         .1, .2, .3, .4, .5, .6, .7, .8, .9,
         1, 1, 1, 1,]
    crv = NURBS([U], C)
    return crv

import sys
try:
    backend = sys.argv[1]
    plt.use(backend)
except IndexError:
    pass

crv = make_crv()

plt.figure()
plt.cpoint(crv)
plt.cwire(crv)

plt.figure()
plt.kpoint(crv)
plt.kwire(crv)

plt.figure()
plt.curve(crv)

plt.figure()
plt.cplot(crv)
plt.kplot(crv)
plt.plot(crv)

plt.show()

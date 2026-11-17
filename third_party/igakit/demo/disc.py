from igakit.cad import *
from math import sqrt

nrb = grid([1, 1], degree=2).scale(2).move([-1, -1])
w = sqrt(2)/2
Cw = nrb.control
Cw[-1, 1, 0] += 1; Cw[-1, 1, :] *= w
Cw[ 1,-1, 1] += 1; Cw[ 1,-1, :] *= w
Cw[ 0, 1, 0] -= 1; Cw[ 0, 1, :] *= w
Cw[ 1, 0, 1] -= 1; Cw[ 1, 0, :] *= w
nrb.scale(w)

nrb1 = nrb
nrb2 = refine(nrb, 2)
nrb3 = refine(nrb, 3)
nrb4 = refine(nrb, 4)

from igakit.plot import plt
import sys
try:
    backend = sys.argv[1]
    plt.use(backend)
except IndexError:
    pass
for nrb in [nrb1, nrb2, nrb3, nrb4]:
    plt.figure()
    plt.kplot(nrb)
    plt.cplot(nrb)
plt.show()

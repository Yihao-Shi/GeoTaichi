from igakit.cad import *

# create two quarter circles
R0 = 1.0
R1 = 2.0
C0 = circle(radius=R0, angle=Pi/2)
C1 = circle(radius=R1, angle=Pi/2)

# make a ruled surface out of the two arcs
srf = ruled(C0, C1)

# make the radial direction first
srf.transpose()

# refine the surface to have:
#  * 3x6 nonempty knot spans
#  * degree two in both directions
#  * maximum allowable continuity
srf1 = refine(srf, factor=[3,6], degree=2)

# refine the surface to have:
#  * 5x10 nonempty knot spans
#  * degree three in both directions
#  * C^0 continuity
srf2 = refine(srf, factor=[5,10], degree=3, continuity=0)

import sys
from igakit.plot import plt
try:
    backend = sys.argv[1]
    plt.use(backend)
except IndexError:
    pass
for nrb in (srf, srf1, srf2):
    plt.figure()
    plt.cplot(nrb)
    plt.kplot(nrb)
plt.show()

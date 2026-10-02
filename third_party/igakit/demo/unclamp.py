from igakit.cad import *

c1 = circle(radius=1, angle=Pi/2)
c2 = c1.copy().unclamp(0)

from igakit.plot import plt
import sys
try:
    backend = sys.argv[1]
    plt.use(backend)
except IndexError:
    pass

plt.figure()
plt.cplot(c1)
plt.kplot(c1)

plt.figure()
plt.cplot(c2)
plt.kplot(c2)

plt.show()

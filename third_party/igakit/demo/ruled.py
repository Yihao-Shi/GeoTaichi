from igakit.cad import *

c1 = line().reverse()
c2 = circle(center=(0,1), angle=Pi)
S = ruled(c1, c2)

# ---

from igakit.plot import plt
import sys
try:
    backend = sys.argv[1]
    plt.use(backend)
except IndexError:
    pass
#plt.use('mayavi')
#plt.use('matplotlib')

plt.figure()
plt.curve(c1, color=(1,0,0))
plt.curve(c2, color=(1,0,0))
plt.cplot(S)
plt.plot(S)

plt.show()

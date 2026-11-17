from igakit.cad import *

crv = line(1,2)
srf = revolve(crv, point=0, axis=2, angle=[Pi/2,2*Pi])
vol = revolve(srf, point=3, axis=1, angle=[-Pi,Pi/2])

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
plt.cplot(vol)
plt.kplot(vol)
plt.plot(vol)

plt.show()

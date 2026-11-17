from igakit.cad import *

H  = 3.0
R0 = 1.0
R1 = 2.0
Rb = 3.0

C0 = circle(radius=R0)
C1 = circle(radius=R1)
annulus  = ruled(C0, C1)
pipe     = extrude(annulus, displ=H, axis=2)
elbow    = revolve(annulus, point=(Rb,0,0),
                   axis=(0,-1,0), angle=Pi/2)
bentpipe = join(pipe.reverse(2), elbow, axis=2)

# ---

nrb = bentpipe

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
plt.cpoint(nrb)
plt.cwire(nrb, mode='tube')
plt.kpoint(nrb)
plt.kwire(nrb, mode='tube')
plt.plot(nrb, opacity=0.1)

plt.figure()
plt.kplot(nrb, color=(0,0,1))
plt.plot(nrb)

plt.show()

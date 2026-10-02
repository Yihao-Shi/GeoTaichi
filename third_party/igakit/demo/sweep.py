from igakit.cad import *

C = circle(center=-1, angle=(-Pi/2,Pi/2))
T = circle(radius=3, center=3, angle=(Pi,Pi/2)).rotate(Pi/2, 0)

S = sweep(C, T)

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

plt.plot(C, color=(1,0,0))
plt.plot(T, color=(0,0,1))
plt.plot(S, color=(1,1,0))

plt.show()

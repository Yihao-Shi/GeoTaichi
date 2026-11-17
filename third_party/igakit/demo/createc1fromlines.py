import numpy as np
import pylab as plt
from igakit.nurbs import NURBS
from cad import line

def seg_intersect(x1,x2,x3,x4):
    a = x2-x1
    b = x4-x3
    c = x3-x1
    num = np.dot( np.cross(c,b), np.cross(a,b) )
    denom = np.dot( np.cross(a,b), np.cross(a,b) )
    s = num / denom
    return x1 + a*s

def curve_length(c,res=1000):
    u=np.linspace((c.knots[0])[0],(c.knots[0])[-1],res,endpoint=True)
    C = c(u)
    L=0.
    for i in range(C.shape[0]-1):
        L += np.linalg.norm(C[i+1,:]-C[i,:])
    return L


ms=4
"""
Step -1: Read the data from the top and bottom
"""
x_b = np.fromfile('venturi_bottom.dat',dtype='float',sep=" ")
x_b = x_b.reshape((x_b.size/2,2))
x_t = np.fromfile('venturi_top.dat',dtype='float',sep=" ")
x_t = x_t.reshape((x_t.size/2,2))
Lx=max(x_b[:,0].max(),x_t[:,0].max())-min(x_b[:,0].min(),x_t[:,0].min())
Ly=max(x_b[:,1].max(),x_t[:,1].max())-min(x_b[:,1].min(),x_t[:,1].min())
ar=Lx/Ly
plt.figure(figsize=(ar,2))
plt.subplot(311)
plt.plot(x_b[:,0],x_b[:,1],'k-')
plt.plot(x_b[:,0],x_b[:,1],'bo',markersize=ms)
plt.plot(x_t[:,0],x_t[:,1],'k-')
plt.plot(x_t[:,0],x_t[:,1],'bo',markersize=ms)
"""
Step 0: Create the lines for the bottom curve (lines selected by hand)
"""
ind=[[0,1],
     [2,16],
     [22,40],
     [90,105],
     [111,119],
     [121,122]]
L=[]
plt.subplot(312)
plt.plot(x_b[:,0],x_b[:,1],'bo',markersize=ms)
plt.plot(x_t[:,0],x_t[:,1],'bo',markersize=ms)
for i in ind:
    L.append(line(x_b[i[0],:],x_b[i[1],:]))
    plt.plot(L[-1].control[:,0],L[-1].control[:,1],'r-')
"""
Step 1: Create a quadratic curve between each line. The extra control
point is determined as the point where the lines intersect. This way
the construction is C1.
"""
P = np.zeros((3,2))
C = []
C.append(L[0])
for i in range(len(L)-1):
    mid = seg_intersect(L[i  ].points[0,0:2],L[i  ].points[1,0:2],
                        L[i+1].points[0,0:2],L[i+1].points[1,0:2])
    plt.plot(mid[0],mid[1],'yo',markersize=ms)
    P[0,:] = L[i  ].points[1,0:2]
    P[1,:] = mid[0:2]
    P[2,:] = L[i+1].points[0,0:2]
    C.append(NURBS(P,[[0,0,0,1,1,1]]))
    C.append(L[i+1])
"""
Step 2-4: Degree elevate to make all curves p=2. Also compute each
curve's length and the total combined length.
"""
plt.subplot(313)
L=[]
for c in C:
    c.elevate(2-c.degree[0])
    L.append(curve_length(c))
L = np.asarray(L)
Lt = L.sum()
L /= Lt
"""
Step 5-6: Combine all curves into 1. This we do by juxtaposing each
knot vector, scaled by the ratio of curve length to total curve
length. This is to keep det(J) mostly uniform.
"""
nc=len(C)
U=np.zeros(2*(nc+2))
P=np.zeros((2*nc+1,3))
for i in range(nc):        
    U[1+2*(i+1):1+2*(i+2)]=((C[i].knots[0])[3:5])*L[i]+U[1+2*(i):1+2*(i+1)]
    P[i*2,  :] = C[i].points[0,:]
    P[i*2+1,:] = C[i].points[1,:]
P[-1,:] = C[-1].points[2,:]
U[-1] = 1.0
final = NURBS(P,[U])
"""
Step 7: Here we would knot remove the double knots since the curve is
C^1 by construction. I see the fortran routine but I do not think we
yet have the python interface.
"""
u=np.linspace(0,1,1000,endpoint=True)
c=final(u)
plt.plot(x_b[:,0],x_b[:,1],'bo',markersize=ms)
plt.plot(x_t[:,0],x_t[:,1],'bo',markersize=ms)
plt.plot(c[:,0],c[:,1],'-y')

plt.show()

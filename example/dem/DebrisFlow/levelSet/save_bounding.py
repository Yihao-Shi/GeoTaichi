import numpy as np

def PointSpawn(points, radius = 0.25, spawn = True):
    pos = list()
    par_num = 0
    for p in points:
        delta = [-radius, radius]
        par_num += 1
        if not spawn:
            pos.append(Vector3f(p[0],p[1],p[2]))
            #if(par_num > 100): break
        else:
            #
            #if(par_num > 10): break
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        pos.append([p[0]+delta[i],p[1]+delta[j],p[2]+delta[k]])
    return pos
    
aabb_centers = np.loadtxt("pos_list.txt")

aabb_centers += np.array([115.5, 130.5-1,-1597.4]) #shift data
pos = np.array(PointSpawn(aabb_centers, radius = 0.25))
par_num = pos.shape[0]
pos = pos[0:par_num,:]
radius = np.repeat(0.25, par_num)
orientation = np.random.random((par_num, 3))
orientation = (orientation.T / np.linalg.norm(orientation, axis=1)).T
np.savetxt('BoundingSphere.txt', np.column_stack((pos, radius, orientation)), header="     PositionX            PositionY                PositionZ            Radius            DirX            DirY            DirZ", delimiter=" ")

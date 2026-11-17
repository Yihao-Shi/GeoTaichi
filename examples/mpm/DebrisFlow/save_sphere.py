import numpy as np
from pathlib import Path

POSITION_FILE = Path(__file__).resolve().parents[3] / "assets/data/DebrisFlow/pos_list.txt"


def PointSpawn(points, radius=0.25, spawn=True):
    pos = list()
    par_num = 0
    for p in points:
        delta = [-radius, radius]
        par_num += 1
        if not spawn:
            pos.append(Vector3f(p[0], p[1], p[2]))
            # if(par_num > 100): break
        else:
            #
            # if(par_num > 10): break
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        pos.append([p[0] + delta[i], p[1] + delta[j], p[2] + delta[k]])
    return pos


aabb_centers = np.loadtxt(POSITION_FILE)

aabb_centers += np.array([115.5, 130.5 - 1, -1597.4])  # shift data
pos = np.array(PointSpawn(aabb_centers, radius=0.25))
pos = np.array(PointSpawn(pos, radius=0.125))
par_num = pos.shape[0]
pos = pos[0:par_num, :]
volume = np.repeat(4.0 / 3.0 * np.pi * 0.125**3, par_num)
psize = np.repeat(0.125, par_num)
np.savetxt(
    "SpherePacking.txt",
    np.column_stack((pos, volume, psize, psize, psize)),
    header="     PositionX            PositionY                PositionZ            Volume                PsizeX                PsizeY                PsizeZ",
    delimiter=" ",
)

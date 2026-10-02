import os
import sys

import math

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

params = {
             'backend': 'ps',
             'font.size': 26,
             'lines.linewidth': 4.5,
             'lines.markersize': 12,
             'xtick.labelsize': 26,
             'ytick.labelsize': 26,
             'xtick.major.pad': 12,
             'ytick.major.pad': 12,
             "axes.labelpad":   8,
             'legend.fontsize': 26,
             'figure.figsize': [12, 9],
             'font.family': 'serif',
             'text.usetex': True,
             'font.serif': 'Arial',
             'savefig.dpi': 300
         }
rcParams.update(params)

         
color = [(0/255, 0/255, 0/255), 
         (255/255, 0/255, 0/255), 
         (94/255, 114/255, 255/255), 
         (0/255, 128/255, 0/255)]
        


def get_disp(path, end=1):
    data0 = np.load(path+"/particles/MPMParticle{0:06d}.npz".format(0), allow_pickle=True)
    pos0 = data0['position']
    data = np.load(path+"/particles/MPMParticle{0:06d}.npz".format(end), allow_pickle=True)
    epdstrain = data['state_vars'].item()['epdstrain'][pos0[:,2]>6]
    disp = np.linalg.norm(data['position']-pos0, axis=1)[pos0[:,2]>6]
    return np.max(disp)
    
def get_volume(path, end=1):
    data0 = np.load(path+"/particles/MPMParticle{0:06d}.npz".format(0), allow_pickle=True)
    pos0 = data0['position']
    data = np.load(path+"/particles/MPMParticle{0:06d}.npz".format(end), allow_pickle=True)
    epdstrain = data['state_vars'].item()['epdstrain'][pos0[:,2]>6]
    disp = np.linalg.norm(data['position']-pos0, axis=1)[pos0[:,2]>6]
    volume = 0.5**3*np.sum((disp>0.15))
    return volume


total_num = 400  
def convergence(name):
    volume = []
    stdv = []
    disp = []
    stdd = []
    for i in range(1, total_num):
        volume.append(get_volume(name+f"{i}"))
        stdv.append(np.std(volume, ddof=1))
        disp.append(get_disp(name+f"{i}"))
        stdd.append(np.std(disp, ddof=1))
    return stdv, stdd
        
stdv0, stdd0 = convergence("concave/Concave480_0_r1.0_rand")
stdv1, stdd1 = convergence("convex/Convex210_0_r1.0_rand")
stdv2, stdd2 = convergence("flat/Flat332_0_r1.0_rand")
x = np.arange(1, total_num)
plt.plot(x, stdv0, color=color[0], label='G1-AR1-F1')
plt.plot(x, stdv1, color=color[1], label='G2-AR1-F1')
plt.plot(x, stdv2, color=color[2], label='G3-AR1-F1')
plt.xlabel("Number of runs")
plt.ylabel("Standard deviation of sliding volume (m$^3$)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('volume_convergence' + '.svg')
plt.close()

plt.plot(x, stdd0, color=color[0], label='G1-AR1-F1')
plt.plot(x, stdd1, color=color[1], label='G2-AR1-F1')
plt.plot(x, stdd2, color=color[2], label='G3-AR1-F1')
plt.xlabel("Number of runs")
plt.ylabel("Standard deviation of sliding distance (m)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('disp_convergence' + '.svg')
plt.close()



import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from third_party.tablelegend import tablelegend

params = {
             'backend': 'ps',
             'font.size': 26,
             'lines.linewidth': 4.5,
             'lines.markersize': 10,
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
         (0/255, 128/255, 0/255),
         (165/255, 42/255, 42/255)]
         
def get_analytical(mu):
    t, ana=[], []
    for i in range(0, 51):
        time = i * 0.04
        ana.append(math.sqrt(2)/4.*9.8*(1.-mu)*time*time)
        t.append(time)
    return t, ana

def get_data(path):
    t, x = [], []
    for i in range(30, 51):
        data = np.load(path+"/particles/LSDEMRigid{0:06d}.npz".format(i))
        mass_center = data["mass_center"]
        time = data["t_current"]-3.
        x.append(mass_center[0, 0]-1.5)
        t.append(time)
    return t, x

ta1, ana1 = get_analytical(0)
ta2, ana2 = get_analytical(0.2)
ta3, ana3 = get_analytical(0.4)

t1, x1 = get_data('mu=0')
t2, x2 = get_data('mu=0.2')
t3, x3 = get_data('mu=0.4')

fig, ax=plt.subplots()
ax.plot(ta1, ana1, color=color[0], label="Analytical")
ax.plot(ta2, ana2, color=color[0])
ax.plot(ta3, ana3, color=color[0])

ax.scatter(t1, x1, color=color[1], marker='o', s=180, label="$\mu$ = 0")
ax.scatter(t2, x2, color=color[2], marker='h', s=180, label="$\mu$ = 0.2")
ax.scatter(t3, x3, color=color[3], marker='*', s=180, label="$\mu$ = 0.4")

ax.set_xlim([0, 2])
ax.set_ylim([0., 14])
ax.set_xlabel("Time (s)")
ax.set_ylabel('Displacement [m]')
ax.legend(frameon=False)
#tablelegend(ax, ncol=2, frameon=False, row_labels=['$\mu$ = 0', '$\mu$ = 0.1', '$\mu$ = 0.3', '$\mu$ = 0.5'], col_labels=['Analytical', 'Simulation'], columnspacing=1, title_label='')
fig.tight_layout()
fig.savefig ("sliding.pdf")
plt.close()

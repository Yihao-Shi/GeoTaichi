import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
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
    for i in range(0, 100):
        time = 0.01*(i-0.)
        ana.append(0.5*9.8*(1.-mu)*math.sqrt(2)*time)
        t.append(time)
    return t, ana

def get_data(path):
    data = np.loadtxt(path)[0:][:,0]
    t = [0.01*i for i in range(data.shape[0])]
    return t, data

ta1, ana1 = get_analytical(0)
ta2, ana2 = get_analytical(0.2)
ta3, ana3 = get_analytical(0.5)
ta4, ana4 = get_analytical(1.0)

t1, x1 = get_data('mu=0.0.txt')
t2, x2 = get_data('mu=0.2.txt')
t3, x3 = get_data('mu=0.5.txt')
t4, x4 = get_data('mu=1.0.txt')

fig, ax=plt.subplots()
ax.plot(ta1, ana1, color=color[0], label="Analytical")
ax.plot(ta2, ana2, color=color[0])
ax.plot(ta3, ana3, color=color[0])
ax.plot(ta4, ana4, color=color[0])

ax.scatter(t1, x1, color=color[1], marker='o', s=180, label="$\mu$ = 0.0")
ax.scatter(t2, x2, color=color[2], marker='o', s=180, label="$\mu$ = 0.2")
ax.scatter(t3, x3, color=color[3], marker='o', s=180, label="$\mu$ = 0.5")
ax.scatter(t4, x4, color=color[4], marker='o', s=180, label="$\mu$ = 1.0")

ax.set_xlim([0, 1])
ax.set_ylim([0., 7])
ax.set_xlabel("Time (s)")
ax.set_ylabel('Velocity [m/s]')
ax.legend(frameon=False)
#tablelegend(ax, ncol=2, frameon=False, row_labels=['$\mu$ = 0', '$\mu$ = 0.1', '$\mu$ = 0.3', '$\mu$ = 0.5'], col_labels=['Analytical', 'Simulation'], columnspacing=1, title_label='')
fig.tight_layout()
fig.savefig ("sliding.pdf")
plt.close()



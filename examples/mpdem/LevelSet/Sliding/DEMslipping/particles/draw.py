import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams

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
         (0/255, 128/255, 0/255)]

t, x, ana = [], [], []
for i in range(30, 51):
    data = np.load("LSDEMRigid{0:06d}.npz".format(i))
    mass_center = np.mean(data["mass_center"][0, 0])
    time = data["t_current"]-3.
    x.append(mass_center-1.)
    ana.append(math.sqrt(2)/4.*9.8*0.5*time*time)
    t.append(time)

plt.plot(t, ana, color=color[0], label="Analytical")
plt.scatter(t, x, color=color[1], label="GeoTaichi")
#plt.xlim([0, 0.02])
#plt.ylim([0., 1.6])
plt.xlabel("Time (s)")
plt.ylabel('Displacement [kJ]')
plt.tight_layout()
plt.legend(frameon=False)
plt.savefig ("disp.eps")




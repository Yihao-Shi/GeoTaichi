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

t, translation, rotation, contact, total = [], [], [], [], []
inih = 0.
for i in range(1, 100):
    data = np.load("LSDEMRigid{0:06d}.npz".format(i))
    if i==0: inih = data["mass_center"][0][2]
    mass = data["mass"]
    v = data["velocity"]
    w = data["omega"]
    inertia = 1. / data["inverse_inertia"]
    translation.append(np.sum(0.5 * mass * (v[:,0]*v[:,0]+v[:,1]*v[:,1]+v[:,2]*v[:,2])))
    rotation.append(np.sum(0.5*(inertia[:,0]*w[:,0]*w[:,0]+inertia[:,1]*w[:,1]*w[:,1]+inertia[:,2]*w[:,2]*w[:,2])))
    total.append(np.sum(0.5 * mass * (v[:,0]*v[:,0]+v[:,1]*v[:,1]+v[:,2]*v[:,2]))+np.sum(0.5*(inertia[:,0]*w[:,0]*w[:,0]+inertia[:,1]*w[:,1]*w[:,1]+inertia[:,2]*w[:,2]*w[:,2]))+np.sum(data["elastic_energy"][:]))
    contact.append(np.sum(data["elastic_energy"][:]))
    t.append(data["t_current"])
    
total = [i for i in total]
translation = [i for i in translation]
rotation = [i for i in rotation]
contact = [i for i in contact]

plt.plot(t, total, color=color[0], label="Total")
plt.plot(t, translation, color=color[1], label="Translation")
plt.plot(t, rotation, color=color[2], label="Rotation")
plt.plot(t, contact, color=color[3], label="Contact")
#plt.xlim([0, 0.04])
#plt.ylim([0., 25])
plt.xlabel("Time (s)")
plt.ylabel('Particle energy [J]')
plt.tight_layout()
plt.legend(frameon=False)
plt.savefig ("energy.pdf")

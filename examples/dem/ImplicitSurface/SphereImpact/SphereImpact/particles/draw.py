import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.spatial.transform import Rotation as R

params = {
             'backend': 'ps',
             'font.size': 36,
             'lines.linewidth': 4.5,
             'lines.markersize': 10,
             'xtick.labelsize': 32,
             'ytick.labelsize': 32,
             'xtick.major.pad': 12,
             'ytick.major.pad': 12,
             "axes.labelpad":   8,
             'legend.fontsize': 32,
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

for i in range(0, 100):
    data = np.load("LSDEMRigid{0:06d}.npz".format(i))
    mass = data["mass"]
    v = data["velocity"]
    w = data["omega"]
    q = data["quanternion"]
    r = R.from_quat(q)
    wl = w.copy()
    for j in range(w.shape[0]):
        wl[j] = r.as_matrix()[j,...]@w[j]
    inertia = 1. / data["inverse_inertia"]
    translation.append(np.sum(0.5*mass*(v[:,0]*v[:,0]+v[:,1]*v[:,1]+v[:,2]*v[:,2])))
    rotation.append(np.sum(0.5*(inertia[:,0]*wl[:,0]*wl[:,0]+inertia[:,1]*wl[:,1]*wl[:,1]+inertia[:,2]*wl[:,2]*wl[:,2])))
    total.append(np.sum(0.5*mass*(v[:,0]*v[:,0]+v[:,1]*v[:,1]+v[:,2]*v[:,2]))+np.sum(0.5*(inertia[:,0]*wl[:,0]*wl[:,0]+inertia[:,1]*wl[:,1]*wl[:,1]+inertia[:,2]*wl[:,2]*wl[:,2]))+np.sum(data["elastic_energy"][:]))
    contact.append(np.sum(data["elastic_energy"][:]))
    t.append(data["t_current"])
    
total = [i / 1000 for i in total]
translation = [i / 1000 for i in translation]
rotation = [i / 1000 for i in rotation]
contact = [i / 1000 for i in contact]

plt.plot(t, total, color=color[0], label="Total")
plt.plot(t, translation, color=color[1], label="Translation")
plt.plot(t, rotation, color=color[2], label="Rotation")
plt.plot(t, contact, color=color[3], label="Contact")
plt.xlabel("Time (s)")
plt.ylabel('Particle energy [kJ]')
plt.tight_layout()
plt.legend(frameon=False)
plt.savefig ("energy.svg")

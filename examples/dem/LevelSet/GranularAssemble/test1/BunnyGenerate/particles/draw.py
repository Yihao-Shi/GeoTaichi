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

t, kinetic, contact, potential, friction, damp, total = [], [], [], [], [], [], []
inih = None
for i in range(0, 100):
    data = np.load("LSDEMRigid{0:06d}.npz".format(i))
    if i==0: inih = data["mass_center"][:,2]
    mass = data["mass"]
    v = data["velocity"]
    w = data["omega"]
    q = data["quanternion"]
    r = R.from_quat(q)
    
    wl = w.copy()
    for j in range(w.shape[0]):
        wl[j] = r.as_matrix()[j,...].transpose()@w[j]

    inertia = 1. / data["inverse_inertia"]
    kinetic.append(np.sum(0.5*mass*(v[:,0]*v[:,0]+v[:,1]*v[:,1]+v[:,2]*v[:,2]))+np.sum(0.5*(inertia[:,0]*wl[:,0]*wl[:,0]+inertia[:,1]*wl[:,1]*wl[:,1]+inertia[:,2]*wl[:,2]*wl[:,2])))
    total.append(np.sum(0.5*mass*(v[:,0]*v[:,0]+v[:,1]*v[:,1]+v[:,2]*v[:,2]))+np.sum(0.5*(inertia[:,0]*wl[:,0]*wl[:,0]+inertia[:,1]*wl[:,1]*wl[:,1]+inertia[:,2]*wl[:,2]*wl[:,2]))+np.sum(data["elastic_energy"][:])-np.sum(data["friction_energy"][:])-np.sum(data["damp_energy"][:])+np.sum(mass*9.8*(data["mass_center"][:,2])))
    potential.append(np.sum(mass*9.8*(data["mass_center"][:,2])))
    contact.append(np.sum(data["elastic_energy"][:]))
    friction.append(-np.sum(data["friction_energy"][:]))
    damp.append(-np.sum(data["damp_energy"][:]))
    t.append(data["t_current"])
    
total = [i / 1000 for i in total]
kinetic = [i / 1000 for i in kinetic]
potential = [i / 1000 for i in potential]
contact = [i / 1000 for i in contact]
friction = [i / 1000 for i in friction]
damp = [i / 1000 for i in damp]

plt.plot(t, kinetic, color=color[1], label="Kinetic energy")
plt.plot(t, potential, color=color[2], label="Gravitational energy")
plt.plot(t, contact, color=color[3], label="Contact energy")
plt.plot(t, friction, color='orange', label="Friction energy")
plt.plot(t, damp, color='grey', label="Damping energy")
plt.plot(t, total, color=color[0], label="Total energy")
plt.xlim([0, 3.0])
plt.ylim([0., 1800])
plt.xlabel("Time (s)")
plt.ylabel('Particle energy [kJ]')
plt.tight_layout()
plt.legend(bbox_to_anchor=(0.48, 0.25), frameon=False)
plt.savefig ("energy.svg")

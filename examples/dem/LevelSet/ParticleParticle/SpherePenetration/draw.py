import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit

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

def get_data(path):
    t, x, f = [], [], []
    start=0.
    for i in range(0, 51):
        data = np.load(path+"/particles/LSDEMRigid{0:06d}.npz".format(i))
        if i==0: start=data["mass_center"][1][2]
        mass_center = data["mass_center"]
        time = data["t_current"]
        x.append((start-data["mass_center"][1][2])*1000)
        f.append(np.linalg.norm(data["contact_force"][0]))

        t.append(time)
    return t, x, f

t1, x1, f1 = get_data('160N')
t2, x2, f2 = get_data('646N')
t3, x3, f3 = get_data('2578N')
t4, x4, f4 = get_data('10306N')

f1=[i/2 for i in f1]
f2=[i/2 for i in f2]
f3=[i/2 for i in f3]
f4=[i/2 for i in f4]

x=[(x1[i]+x2[i]+x3[i])/3. for i in range(len(x1))]
f=[(f1[i]+f2[i]+f3[i])/3. for i in range(len(f1))]

fig, ax=plt.subplots()

def target_func(x, a):
    return a*x**1.5
    
'''popt,pcov = curve_fit(target_func, x, f)
calc_fdata=[target_func(i,popt[0]) for i in x]
res_fdata=np.array(f)-np.array(calc_fdata)
ss_res=np.sum(res_fdata**2)
ss_tot=np.sum((f-np.mean(f))**2)
r_squared=1.-(ss_res/ss_tot)'''

para=0.046
calc_fdata=[target_func(i,para) for i in x]

#ax.plot(x, calc_fdata, color=color[0], linestyle='--', label=f" $f$={format(para, '.3f')}$d^{{1.5}}$")
ax.scatter(x1, f1, c='none', edgecolors=color[1], marker='o', s=180, label="160 vertices")
ax.scatter(x2, f2, c='none', edgecolors=color[2], marker='h', s=180, label="646 vertices")
ax.scatter(x3, f3, c='none', edgecolors=color[3], marker='^', s=180, label="2578 vertices")
ax.scatter(x4, f4, c='none', edgecolors='grey', marker='^', s=180, label="10306 vertices")

#ax.set_xlim([0, 20])
#ax.set_ylim([0., 5])
ax.set_xlabel("Penetration, $d$ (mm)")
ax.set_ylabel('Contact force, $f/k_n$ ($10^{-5}$$m^{0.5}$)')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig ("penalty.svg")
plt.close()




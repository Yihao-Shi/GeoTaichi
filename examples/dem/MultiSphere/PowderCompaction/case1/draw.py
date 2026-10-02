#!/usr/bin/env python
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

params = {
             'backend': 'ps',
             'font.size': 26,
             'lines.linewidth': 3,
             'lines.markersize': 10,
             'xtick.labelsize': 26,
             'ytick.labelsize': 26,
             'xtick.major.pad': 12,
             'ytick.major.pad': 12,
             "axes.labelpad":   8,
             'legend.fontsize': 26,
             'figure.figsize': [12, 9],
             'font.family': 'serif',
             'text.usetex': False,
             'font.serif': 'Arial',
             'savefig.dpi': 300
         }
rcParams.update(params)

         
color = [(0/255, 0/255, 0/255), 
         (255/255, 0/255, 0/255), 
         (94/255, 114/255, 255/255), 
         (0/255, 128/255, 0/255)]



def macro(path, start_num, end_num):
    stress = []
    epsilon=[]
    leng0 = 0.

    for printNum in range(start_num, end_num):
        if printNum==start_num:
            data = np.load(path+'contacts/DEMContactPW{0:06d}.npz'.format(printNum), allow_pickle=True)
            mask = np.linalg.norm(data["normal_force"], axis=1) > 0.
            end2=data["end2"][mask]
            fn=data["normal_force"][mask]

            up_force = -(np.sum(fn[end2==2],axis=0) + np.sum(fn[end2==3],axis=0))[2]
        
            data = np.load(path+'walls/DEMWall{0:06d}.npz'.format(printNum), allow_pickle=True)
            down_position = (data["point1"][0][2]+data["point2"][0][2]+data["point3"][0][2]+data["point1"][1][2]+data["point2"][1][2]+data["point3"][1][2])/6.
            up_position = (data["point1"][2][2]+data["point2"][2][2]+data["point3"][2][2]+data["point1"][3][2]+data["point2"][3][2]+data["point3"][3][2])/6.
            leng0 = up_position-down_position
            leng = up_position-down_position

            sigma = up_force/(1.296e-7)
        else:
            data = np.load(path+'contacts/DEMContactPW{0:06d}.npz'.format(printNum), allow_pickle=True)
            print(data["contact_num"][-1])
            mask = np.linalg.norm(data["normal_force"], axis=1) > 0.
            end2=data["end2"][mask]
            fn=data["normal_force"][mask]
        
            up_force = -(np.sum(fn[end2==2],axis=0) + np.sum(fn[end2==3],axis=0))[2]
        
            data = np.load(path+'walls/DEMWall{0:06d}.npz'.format(printNum), allow_pickle=True)
            down_position = (data["point1"][0][2]+data["point2"][0][2]+data["point3"][0][2]+data["point1"][1][2]+data["point2"][1][2]+data["point3"][1][2])/6.
            up_position = (data["point1"][2][2]+data["point2"][2][2]+data["point3"][2][2]+data["point1"][3][2]+data["point2"][3][2]+data["point3"][3][2])/6.
    
            sigma = up_force/(1.296e-7)
            leng = up_position - down_position
        
        target_stress = 245e3
        sigma = sigma / target_stress
        stress.append(sigma)
        epsilon.append(math.log(leng0/leng)*100)
    return stress, epsilon
    
stress, epsilon = macro('OutputData/', 0, 10)

plt.plot(epsilon, stress, linestyle='-', color=color[0], label="case 1")
# plt.xlim([0., 0.5])
# plt.ylim([0., 300e3])
plt.xlabel('Axial strain, $\\varepsilon_a$ (\%)')
plt.ylabel('$\\sigma$ / $\\sigma_{target}$')
plt.legend(frameon=False)
plt.savefig ("stress_strain.png")   
plt.close()

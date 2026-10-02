import numpy as np
import math
import xlrd
import matplotlib.pyplot as plt
from matplotlib import rcParams

from scipy.signal import savgol_filter, butter, filtfilt
from scipy.interpolate import interp1d

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
         
start=0
end=41

def load_disp(path):
    contact_stress = []
    epslion = []
    zpos0 = 0.
    for printNum in range(start, end):
        wall = np.load(path+'walls/DEMWall{0:06d}.npz'.format(printNum))
        contact = np.load(path+'DEMPMcontacts/DEMPMContactPW{0:06d}.npz'.format(printNum))
        contact_stress.append((np.linalg.norm(wall['contact_force'][12])+np.linalg.norm(wall['contact_force'][13])))
        epslion.append(wall['t_current'])
        
    settle_u = np.linspace(min(epslion), max(epslion), 500)
    f = interp1d(epslion, contact_stress, kind='linear')
    force_u = f(settle_u)
    b, a = butter(3, 0.1, btype='low')
    force_smooth = filtfilt(b, a, force_u)
    return force_smooth, settle_u
    
depth = xlrd.open_workbook("experiment.xls")
sheet = depth.sheet_by_index(0)

rows: list = sheet.row_values(0)
index_x1 = rows.index("Line1")
index_y1 = rows.index("data1")
index_x2 = rows.index("Line2")
index_y2 = rows.index("data2")
index_x3 = rows.index("Line3")
index_y3 = rows.index("data3")
index_x4 = rows.index("Line4")
index_y4 = rows.index("data4")
index_x5 = rows.index("Line5")
index_y5 = rows.index("data5")

x1 = sheet.col_values(index_x1)
y1 = sheet.col_values(index_y1)
x2 = sheet.col_values(index_x2)
y2 = sheet.col_values(index_y2)
x3 = sheet.col_values(index_x3)
y3 = sheet.col_values(index_y3)
x4 = sheet.col_values(index_x4)
y4 = sheet.col_values(index_y4)
x5 = sheet.col_values(index_x5)
y5 = sheet.col_values(index_y5)

x1 = x1[1:len(x1)-9] 
y1 = y1[1:len(y1)-9] 
x2 = x2[1:len(x2)-7]
y2 = y2[1:len(y2)-7] 
x3 = x3[1:len(x3)-7]
y3 = y3[1:len(y3)-7] 
x4 = x4[1:len(x4)-2]
y4 = y4[1:len(y4)-2] 
x5 = x5[1:len(x5)-0]
y5 = y5[1:len(y5)-0] 

contact_stress1, epslion1 = load_disp('ImpactForce45/')
contact_stress2, epslion2 = load_disp('ImpactForce50/')
contact_stress3, epslion3 = load_disp('ImpactForce55/')
contact_stress4, epslion4 = load_disp('ImpactForce60/')
contact_stress5, epslion5 = load_disp('ImpactForce65/')

plt.plot(x1, y1, linestyle='-.', color=color[0], label="$\\theta=45^{\\circ}$")
plt.plot(epslion1, contact_stress1, color=color[0], label="$\\theta=45^{\\circ}$")

plt.plot(x2, y2, linestyle='-.', color=color[1], label="$\\theta=50^{\\circ}$")
plt.plot(epslion2, contact_stress2, color=color[1], label="$\\theta=50^{\\circ}$")

plt.plot(x3, y3, linestyle='-.', color=color[2], label="$\\theta=55^{\\circ}$")
plt.plot(epslion3, contact_stress3, color=color[2], label="$\\theta=55^{\\circ}$")

plt.plot(x4, y4, linestyle='-.', color=color[3], label="$\\theta=60^{\\circ}$")
plt.plot(epslion4, contact_stress4, color=color[3], label="$\\theta=60^{\\circ}$")

plt.plot(x5, y5, linestyle='-.', color='grey', label="$\\theta=65^{\\circ}$")
plt.plot(epslion5, contact_stress5, color='grey', label="$\\theta=65^{\\circ}$")

plt.ylabel("Impact force (N)")
plt.xlabel("Time (s)")
#plt.xlim([0, 400.])
#plt.ylim([0., 50.])
plt.tight_layout()
plt.legend(frameon=False)
plt.savefig ("load_disp9d.svg")
plt.close()

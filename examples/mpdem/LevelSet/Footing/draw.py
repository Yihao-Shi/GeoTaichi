import numpy as np
import math
import xlrd
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
         
q0=1000
start=0
end=90
vel0=0.02
b = 2.0

def calculatempm(path):
    contact_stress = []
    epslion = []
    time = []
    zpos0 = 0.
    for printNum in range(start, end):
        grid = np.load(path+'grids/MPMGrid{0:06d}.npz'.format(printNum))
        particle = np.load(path+'particles/MPMParticle{0:06d}.npz'.format(printNum))
        contact_stress.append(np.sum(grid['contact_force'][:,1][:,2]) / 0.1 / q0)

        if printNum==start:
            zpos0 = np.mean(particle['position'][particle["bodyID"]==1][:,2])
        epslion.append((zpos0 - np.mean(particle['position'][particle["bodyID"]==1][:,2]))/b)
        time.append(grid['t_current'])
    return contact_stress, epslion, time

def calculate(path):
    contact_stress = []
    epslion = []
    time = []
    zpos0 = 0.
    for printNum in range(start, end):
        body = np.load(path+'particles/LSDEMRigid{0:06d}.npz'.format(printNum))
        mass_center = body["mass_center"][0]
        contact_stress.append(body["contact_force"][0][2] / 0.1 / q0)

        if printNum==start:
            zpos0 = mass_center[2]
        epslion.append((zpos0 - mass_center[2])/b)
        time.append(body['t_current'])
    return contact_stress, epslion, time
        
contact_stress1, epslion1, time1 = calculatempm('../../../mpm/Contact/Footing/FootingLargeStrain/')
contact_stress3, epslion3, time3 = calculate('MCFooting/')
bins = np.linspace(0, 1., 10)
val1 = np.repeat(5.14, 10)
val2 = np.repeat(8.28, 10)

depth = xlrd.open_workbook("comparison.xls")
sheet = depth.sheet_by_index(0)

rows = sheet.row_values(0)
index_x1 = rows.index("Line #1")
index_y1 = rows.index("data1")
index_x2 = rows.index("Line #2")
index_y2 = rows.index("data2")
index_x3 = rows.index("Line #3")
index_y3 = rows.index("data3")

x1 = sheet.col_values(index_x1)
y1 = sheet.col_values(index_y1)
x2 = sheet.col_values(index_x2)
y2 = sheet.col_values(index_y2)
x3 = sheet.col_values(index_x3)
y3 = sheet.col_values(index_y3)

x1 = x1[2:len(x1)-90]
y1 = y1[2:len(y1)-90]
x2 = x2[2:len(x2)-0]
y2 = y2[2:len(y2)-0]
x3 = x3[2:len(x3)-99]
y3 = y3[2:len(y3)-99]

#plt.plot(bins, val1, color='black', linestyle='--', label='Lower bound (Prandtl, 1921)')
#plt.plot(bins, val2, color='black', linestyle='-.', label='Upper bound (Meyerhof, 1951)')
plt.plot(x1, y1, color=color[0], linestyle='-.', label='RITSS (Wang $et\ al.$, 2013)')
plt.plot(x2, y2, color=color[1], linestyle='--', label='PFEM (Monforte $et\ al.$, 2017)')
plt.plot(x3, y3, color=color[2], linestyle=':', label='Moving mesh MPM (Bisht $et\ al.$, 2021)')
plt.plot(epslion1, contact_stress1, c='grey',label="GIMP (this study)")
plt.plot(epslion3, contact_stress3, c=color[3],label="Level-set DEM-MPM (this study)")
#plt.plot(epslion2, contact_stress2, linestyle='--', c=color[1],label="$\overline{B}$ GIMP [this study]")
#plt.plot(epslion1, contact_stress1, linestyle='-.', c=color[2],label="$\overline{F}$ GIMP [this study]")


plt.xlabel("Normalized settlement")
plt.ylabel("Normalized load")
plt.xlim([0, 1.0])
plt.ylim([0., 8.])
plt.tight_layout()
plt.legend(frameon=False)
plt.savefig ("devol.pdf")

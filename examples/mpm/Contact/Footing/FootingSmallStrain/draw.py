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

q0=100
start=0
end=41

def calculate(path):
    contact_stress = []
    epslion = []
    time = []
    for printNum in range(start, end):
        data = np.load(path+'/MPMGrid{0:06d}.npz'.format(printNum))
        contact_stress.append(np.sum(data['contact_force'][:,1][:,2]) / 0.6 / q0)
        epslion.append((data['t_current'])*0.00125)
        time.append(data['t_current'])
    return contact_stress, epslion, time
    
def calculate_stable(path):
    contact_stress = []
    epslion = []
    time = []
    for printNum in range(start, end):
        data = np.load(path+'/MPMGrid{0:06d}.npz'.format(printNum))
        contact_stress.append(np.sum(data['contact_force'][:,1][:,2]) / 0.018 / q0)
        epslion.append((data['t_current'])*0.00125)
        time.append(data['t_current'])
    return contact_stress, epslion, time
        
contact_stress1, epslion1, time1 = calculate_stable('fbar/grids')
contact_stress2, epslion2, time2 = calculate_stable('bbar/grids')
contact_stress3, epslion3, time3 = calculate('no/grids')
bins = np.linspace(0, 0.003, 10)
val = np.repeat(5.14, 10)


depth = xlrd.open_workbook("comparison.xls")
sheet = depth.sheet_by_index(0)

rows: list = sheet.row_values(0)
index_x1 = rows.index("Line #1")
index_y1 = rows.index("data1")
index_x2 = rows.index("Line #2")
index_y2 = rows.index("data2")
index_x3 = rows.index("Line #3")
index_y3 = rows.index("data3")
index_x4 = rows.index("Line #4")
index_y4 = rows.index("data4")

x1 = sheet.col_values(index_x1)
y1 = sheet.col_values(index_y1)
x2 = sheet.col_values(index_x2)
y2 = sheet.col_values(index_y2)
x3 = sheet.col_values(index_x3)
y3 = sheet.col_values(index_y3)
x4 = sheet.col_values(index_x4)
y4 = sheet.col_values(index_y4)

x1 = x1[1:len(x1)-25]
y1 = y1[1:len(y1)-25]
x2 = x2[1:len(x2)-25]
y2 = y2[1:len(y2)-25]
x3 = x3[1:len(x3)-11]
y3 = y3[1:len(y3)-11]
x4 = x4[1:len(x4)-0]
y4 = y4[1:len(y4)-0]

plt.plot(bins, val, color=color[3], label='Analytical solution')
plt.scatter(x4, y4, color=color[0], marker='x', label='Classical GIMP (Zhao et. al, 2023)')
plt.scatter(x2, y2, color=color[1], marker='o', label='$\overline{B}$ GIMP (Bisht et. al, 2021)')
plt.scatter(x3, y3, color=color[2], marker='*', label='$\overline{F}$ GIMP (Zhao et. al, 2023)')

plt.plot(epslion3, contact_stress3, linestyle='-', c=color[0],label="Classical GIMP ($GeoTaichi$)")
plt.plot(epslion2, contact_stress2, linestyle='--', c=color[1],label="$\overline{B}$ GIMP ($GeoTaichi$)")
plt.plot(epslion1, contact_stress1, linestyle='-.', c=color[2],label="$\overline{F}$ GIMP ($GeoTaichi$)")

#plt.plot(x1, y1, color='black', label='Classical GIMP [Bishe et. al (2021)]')

plt.xlabel("Normalized displacement $(d/B)$")
plt.ylabel("Normalized load $(q/s_u)$")
plt.xlim([0, 0.003])
plt.ylim([0., 7.])
plt.legend(frameon=False)
plt.savefig ("devol.svg")

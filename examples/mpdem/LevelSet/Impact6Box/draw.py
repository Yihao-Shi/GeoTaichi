import os
import sys

import math

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


import numpy as np
import xlrd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import rcParams
from third_party.tablelegend import tablelegend

params = {
             'backend': 'ps',
             'font.size': 26,
             'lines.linewidth': 4.5,
             'lines.markersize': 12,
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
         
def SetToRotate(q):
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)], 
                   [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)], 
                   [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])

def get_data(path, start, end):
    pos10x, pos30x = 0., 0.
    t, disp1, disp3, rot = [], [], [], []
    for i in range(start, end):
        data = np.load(path+"/LSDEMRigid{0:06d}.npz".format(i))
        if i==start:
            pos10x=data["mass_center"][2][0]
            pos30x=data["mass_center"][4][0]
        disp1.append(100*(data["mass_center"][2][0]-pos10x))
        disp3.append(100*(data["mass_center"][4][0]-pos30x))
        dirs = SetToRotate(data["quanternion"][5]).T@np.array([0,0,1])
        dirs = dirs / np.linalg.norm(dirs)
        theta=90-np.arccos(np.dot(dirs,np.array([0,0,1])))/math.pi*180
        rot.append(theta)
        time=data["t_current"]
        t.append(time)
    return t, disp1, disp3, rot

t, disp1, disp3, rot = get_data("OutputData/particles", 10, 21)

depth = xlrd.open_workbook("disp.xls")
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

x1 = x1[2:len(x1)] 
y1 = y1[2:len(y1)] 
x2 = x2[2:len(x2)]
y2 = y2[2:len(y2)] 
x3 = x3[2:len(x3)] 
y3 = y3[2:len(y3)] 
x4 = x4[2:len(x4)] 
y4 = y4[2:len(y4)] 

fig, ax=plt.subplots()


ax.scatter(x3, y3, marker='o', color=color[0], label='Experiment-block-2')
ax.scatter(x1, y1, marker='^', color=color[1], label='Experiment-block-4')
ax.plot(x4, y4, linestyle='--', color=color[0], label='Experiment-block-2')
ax.plot(x2, y2, linestyle='--', color=color[1], label='Simulation-block-4')

ax.plot(t, disp1, color=color[0], label='Simulation Results')
ax.plot(t, disp3, color=color[1], label='Simulation Results')

ax.set_xlabel("Time (s)")
ax.set_ylabel("Horizontal displacement (cm)")
ax.set_xlim([0.2, 0.4])
ax.set_ylim([0, 6])
ax.legend(frameon=False)
#ax.axes.imshow(np.array(Image.open('blocky.tiff')).astype(np.float32))
tablelegend(ax, ncol=3, frameon=False, row_labels=['Block2', 'Block4'], col_labels=['Exp.', 'CoSim', 'GeoTaichi'])
fig.tight_layout()
fig.savefig('displacement'+'.eps')
plt.close()
    
velocity = xlrd.open_workbook("rotation.xls")
sheet = velocity.sheet_by_index(0)

rows: list = sheet.row_values(0)
index_x1 = rows.index("Line #1")
index_y1 = rows.index("data1")
index_x2 = rows.index("Line #2")
index_y2 = rows.index("data2")

x1 = sheet.col_values(index_x1)
y1 = sheet.col_values(index_y1)
x2 = sheet.col_values(index_x2)
y2 = sheet.col_values(index_y2)

x1 = x1[2:len(x1)] 
y1 = y1[2:len(y1)] 
x2 = x2[2:len(x2)]
y2 = y2[2:len(y2)] 
   
fig, ax=plt.subplots()

ax.scatter(x1, y1, marker='o', color=color[0], label='Experiment')
ax.plot(x2, y2, linestyle='-.', color=color[1], label='CoSim')
ax.plot(t, rot, color=color[2], label='GeoTaichi')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Orientation $(^\\circ)$")
ax.set_xlim([0.2, 0.4])
ax.set_ylim([0, 100])
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig('rotation'+'.eps')
plt.close()

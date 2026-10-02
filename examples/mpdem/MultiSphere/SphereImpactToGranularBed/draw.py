import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


import numpy as np
import xlrd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from third_party.tablelegend import tablelegend

params = {
             'backend': 'ps',
             'font.size': 26,
             'lines.linewidth': 6,
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

def get_data(file, loops, a):
    t, pos, vel = [], [], []
    for i in range(loops):
        data = np.load(file.format(i))
        pos.append((data["position"][0][2]-0.3123)*100)
        vel.append(data["velocity"][0][2]-a)
        time=data["t_current"]
        t.append(time)
    return t, pos, vel

for i in range(100, 101):
    t1, pos1, vel1 = get_data("OutputData/velz112/particles/DEMParticle{0:06d}.npz", i, -1.5)
    t2, pos2, vel2 = get_data("OutputData/velz187/particles/DEMParticle{0:06d}.npz", i, -1.2)
    t3, pos3, vel3 = get_data("OutputData/velz303/particles/DEMParticle{0:06d}.npz", i, -0.6)
    t4, pos4, vel4 = get_data("OutputData/velz330/particles/DEMParticle{0:06d}.npz", i, -0.3)
    t5, pos5, vel5 = get_data("OutputData/velz363/particles/DEMParticle{0:06d}.npz", i, 0.)

    depth = xlrd.open_workbook("depth.xls")
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
    index_x5 = rows.index("Line #5")
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

    x1 = x1[1:len(x1)] 
    y1 = y1[1:len(y1)] 
    x2 = x2[1:len(x2) - 1]
    y2 = y2[1:len(y2) - 1] 
    x3 = x3[1:len(x3)] 
    y3 = y3[1:len(y3)] 
    x4 = x4[1:len(x4)] 
    y4 = y4[1:len(y4)] 
    x5 = x5[1:len(x5)] 
    y5 = y5[1:len(y5)] 

    y1 = np.divide(y1, -1)
    y2 = np.divide(y2, -1)
    y3 = np.divide(y3, -1)
    y4 = np.divide(y4, -1)
    y5 = np.divide(y5, -1)

    fig, ax=plt.subplots()
    ax.scatter(x1, y1, marker='o', color=color[0], label='Experiment')
    ax.scatter(x2, y2, marker='x', color=color[1], label='Experiment')
    ax.scatter(x3, y3, marker='*', color=color[2], label='Experiment')
    ax.scatter(x4, y4, marker='^', color=color[3], label='Experiment')
    ax.scatter(x5, y5, marker='p', color='grey', label='Experiment')
    ax.plot(t1, pos1, color=color[0], label='Simulation Results')
    ax.plot(t2, pos2, color=color[1], label='Simulation Results')
    ax.plot(t3, pos3, color=color[2], label='Simulation Results')
    ax.plot(t4, pos4, color=color[3], label='Simulation Results')
    ax.plot(t5, pos5, color='grey', label='Simulation Results')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (cm)")
    ax.set_xlim([0, 0.2])
    ax.set_ylim([-20, 5])
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    tablelegend(ax, ncol=2, frameon=False, row_labels=['$1.12$ m/s', '$1.87$ m/s', '$3.03$ m/s', '$3.30$ m/s', '$3.63$ m/s'], col_labels=['Exp.', 'Sim.'], columnspacing=1, title_label='Initial vel.')
    fig.savefig('intepretaion_depth'+str(i)+'.eps')
    plt.close()
    
    velocity = xlrd.open_workbook("velocity.xls")
    sheet = velocity.sheet_by_index(0)

    rows: list = sheet.row_values(0)
    index_x1 = rows.index("Line #1")
    index_y1 = rows.index("data1")
    index_x2 = rows.index("Line #2")
    index_y2 = rows.index("data2")
    index_x3 = rows.index("Line #3")
    index_y3 = rows.index("data3")
    index_x4 = rows.index("Line #4")
    index_y4 = rows.index("data4")
    index_x5 = rows.index("Line #5")
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

    x1 = x1[1:len(x1)] 
    y1 = y1[1:len(y1)] 
    x2 = x2[1:len(x2) - 1]
    y2 = y2[1:len(y2) - 1] 
    x3 = x3[1:len(x3)] 
    y3 = y3[1:len(y3)] 
    x4 = x4[1:len(x4)] 
    y4 = y4[1:len(y4)] 
    x5 = x5[1:len(x5) - 1] 
    y5 = y5[1:len(y5) - 1] 
    
    y1 = np.subtract(np.divide(y1, -100), -1.2)
    y2 = np.subtract(np.divide(y2, -100), -0.9)
    y3 = np.subtract(np.divide(y3, -100), -0.6)
    y4 = np.subtract(np.divide(y4, -100), -0.3)
    y5 = np.subtract(np.divide(y5, -100), 0.0)
    
    vel1 = np.subtract(vel1, 0.3)
    vel2 = np.subtract(vel2, 0.3)
    
    fig, ax=plt.subplots()
    ax.scatter(x1, y1, marker='o', color=color[0], label='Experiment')
    ax.scatter(x2, y2, marker='x', color=color[1], label='Experiment')
    ax.scatter(x3, y3, marker='*', color=color[2], label='Experiment')
    ax.scatter(x4, y4, marker='^', color=color[3], label='Experiment')
    ax.scatter(x5, y5, marker='p', color='grey', label='Experiment')
    ax.plot(t1, vel1, color=color[0], label='Simulation Results')
    ax.plot(t2, vel2, color=color[1], label='Simulation Results')
    ax.plot(t3, vel3, color=color[2], label='Simulation Results')
    ax.plot(t4, vel4, color=color[3], label='Simulation Results')
    ax.plot(t5, vel5, color='grey', label='Simulation Results')
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_xlim([0, 0.2])
    ax.set_ylim([-4, 2])
    tablelegend(ax, ncol=2, frameon=False, row_labels=['$1.12$ m/s', '$1.87$ m/s', '$3.03$ m/s', '$3.30$ m/s', '$3.63$ m/s'], col_labels=['Exp.', 'Sim.'], columnspacing=1, title_label='Initial vel.')
    fig.tight_layout()
    fig.savefig('velocity'+str(i)+'.eps')
    plt.close()

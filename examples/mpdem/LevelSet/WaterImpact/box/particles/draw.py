import numpy as np
import xlrd
import matplotlib.pyplot as plt
from matplotlib import rcParams

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
         

def get_data(path, start, end):
    t, disp = [], []
    for i in range(start, end):
        data = np.load(path+"LSDEMRigid{0:06d}.npz".format(i))
        disp.append(data["mass_center"][0][2]*100)
        time=data["t_current"]
        t.append(time)
    return t, disp

t, disp = get_data("", 0, 50)

depth = xlrd.open_workbook("data.xls")
sheet = depth.sheet_by_index(0)

rows: list = sheet.row_values(0)
index_x1 = rows.index("Line1")
index_y1 = rows.index("data1")
index_x2 = rows.index("Line2")
index_y2 = rows.index("data2")

x1 = sheet.col_values(index_x1)
y1 = sheet.col_values(index_y1)
x2 = sheet.col_values(index_x2)
y2 = sheet.col_values(index_y2)

x1 = x1[1:len(x1)-25] 
y1 = y1[1:len(y1)-25] 
x2 = x2[1:len(x2)]
y2 = y2[1:len(y2)] 
fig, ax=plt.subplots()

ax.scatter(x1, y1, marker='o', c='none', edgecolors=color[0], label='Experiment (Wu et al., 2014)')
ax.plot(x2, y2, linestyle='-.', color=color[2], label='SPH-DEM (Liu et al., 2022)')
ax.plot(t, disp, color=color[3], label='This study')

ax.set_xlabel("Time (s)")
ax.set_ylabel("Vertical position (cm)")
ax.set_xlim([0., 0.4])
ax.set_ylim([0., 14])
ax.legend(loc="best", frameon=False)
fig.tight_layout()
fig.savefig('displacement'+'.pdf')
plt.close()


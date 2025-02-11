import os
import numpy as np
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

plt.figure(1)
folder_path = 'particles'
file_names = sorted(os.listdir(folder_path))
for fl in range(1,11):
    file_name = file_names[fl]
    file_path = os.path.join(folder_path, file_name)
    data = np.load(file_path)
    position = data['position']
    pressure = data['pressure']
    lp = pressure.shape

    L = 0.05
    f = position[:, 0] < L/2
    p = pressure[f]
    x = position[f]
    if fl==1:
        plt.scatter(p/10000, x[:,1], color='blue', label="GeoTaichi")
    else:
        plt.scatter(p/10000, x[:,1], color='blue')

kk = 1e-3
e = 1e7
v = 0.3
Cv = kk * e * (1 - v) / ((1 + v) * (1 - 2 * v) * 9.8 * 1000)
Tv = np.arange(0.0, 1.1, 0.1).reshape(-1, 1)
h = 1.0
t = Tv * h**2 / Cv
H = np.arange(0, h + 0.005, 0.005).reshape(-1, 1)
H = h - H
k = 1
mm = np.arange(1, 10001, 2).reshape(-1, 1)
pp = []

for i in range(1, len(Tv)):
    for ii in range(len(H)):
        a = 1.0 / mm
        b = np.sin((mm * np.pi * H[ii]) * 0.5 / h)
        c = np.exp(-(mm**2) * np.pi**2 * Tv[i] / 4)
        pp.append([h - H[ii, 0], 4 / np.pi * np.sum(a * b * c)])

pp = np.array(pp)

#plt.figure(1)
k = 0
for i in range(1, len(Tv)):
    if i==1:
        plt.plot(pp[k:k+len(H), 1], pp[k:k+len(H), 0], '-', color='black', label="Terzaghi's theory")
    else:
        plt.plot(pp[k:k+len(H), 1], pp[k:k+len(H), 0], '-', color='black')
    k += len(H)

plt.show()
plt.xlim([0, 1])
plt.ylim([0., 1])
plt.xlabel('Normalized pore pressure')
plt.ylabel('Normalized height')
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig ("consolidation.svg")   
plt.close()




    

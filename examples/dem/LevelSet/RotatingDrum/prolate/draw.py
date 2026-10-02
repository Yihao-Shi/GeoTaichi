import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.spatial.transform import Rotation as R

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


def LaceyIndex(path):
    t, mixing = [], []
    bin_size = 0.02
    for i in range(0, 51):
        data = np.load(path+"LSDEMRigid{0:06d}.npz".format(i))
        positions = data["mass_center"]
        print(positions.shape)
        group_ids = data["groupID"]
        assert positions.shape[0] == group_ids.shape[0]

        mins = np.array([0,0,0])
        maxs = np.array([0.2,0.2,0.2])
        num_bins = np.ceil((maxs - mins) / bin_size).astype(int)
    
        bin_indices = ((positions - mins) / bin_size).astype(int)
        bin_keys = [tuple(idx) for idx in bin_indices]

        from collections import defaultdict
        bin_total = defaultdict(int)
        bin_group0 = defaultdict(int)

        for key, gid in zip(bin_keys, group_ids):
            bin_total[key] += 1
            if gid == 0:
                bin_group0[key] += 1

        x_list = []
        for key in bin_total:
            if bin_total[key]>0:
                frac = bin_group0[key] / bin_total[key]
                x_list.append(frac)

        x = np.array(x_list)
        x_bar = 0.5
        var_mixed = np.mean((x - x_bar)**2)
        var_unmixed = x_bar * (1 - x_bar)

        LMI = 1 - var_mixed / var_unmixed
        mixing.append(LMI)
        t.append(data["t_current"]/3.)
    return t, mixing

t1, mixing1 = LaceyIndex('20rpm/particles/')

t3 = [0., 0.497, 1.01, 1.51, 1.99, 2.52, 3., 3.49, 4.01, 4.5, 5.]
mixing3 = [0, 0.173, 0.398, 0.542, 0.787, 0.92, 0.92, 0.96, 0.98, 0.972, 0.96]
t4 = [0.00E+00,
2.02E-01,
2.48E-01,
4.04E-01,
6.83E-01,
7.76E-01,
9.78E-01,
1.09E+00,
1.30E+00,
1.48E+00,
1.63E+00,
1.75E+00,
2.05E+00,
2.31E+00,
2.44E+00,
2.50E+00,
2.62E+00,
2.80E+00,
3.00E+00,
3.09E+00,
3.39E+00,
3.68E+00,
3.79E+00,
3.91E+00,
4.15E+00,
4.33E+00,
4.38E+00,
4.53E+00,
4.70E+00,
4.97E+00,
]
mixing4 = [0.00E+00,
1.08E-01,
1.20E-01,
1.33E-01,
2.29E-01,
2.53E-01,
3.33E-01,
4.06E-01,
4.90E-01,
5.62E-01,
6.67E-01,
6.67E-01,
7.99E-01,
8.80E-01,
8.88E-01,
8.92E-01,
9.04E-01,
9.28E-01,
9.28E-01,
9.44E-01,
9.32E-01,
9.72E-01,
9.60E-01,
9.80E-01,
9.44E-01,
9.68E-01,
9.60E-01,
9.92E-01,
9.60E-01,
9.56E-01,
]
plt.scatter(t3 , mixing3, color=color[0], label="Experiment")
plt.plot(t4 , mixing4, color=color[2], label="Super-ellipsoid (Ma $et\ al$., 2017)")
plt.plot(t1, mixing1, color=color[3], label="This study")
plt.xlabel("Revolutions")
plt.ylabel('Mixing Index')
plt.xlim([0,5])
plt.ylim([0,1])
plt.tight_layout()
plt.legend(frameon=False)
plt.savefig ("index.pdf")
plt.close()

x1=[1.85E-02,
2.63E-02,
3.41E-02,
5.07E-02,
6.24E-02,
8.63E-02,
9.95E-02,
1.13E-01,
1.29E-01,
1.46E-01,
1.56E-01,
1.68E-01,
1.82E-01,
1.91E-01,
]
y1=[4.64E-02,
4.74E-02,
5.13E-02,
5.38E-02,
5.82E-02,
7.18E-02,
8.61E-02,
1.00E-01,
1.15E-01,
1.33E-01,
1.39E-01,
1.45E-01,
1.52E-01,
1.50E-01,
]
x2=[2.98E-02,
4.49E-02,
5.78E-02,
7.06E-02,
8.80E-02,
1.06E-01,
1.20E-01,
1.33E-01,
1.50E-01,
1.70E-01,
1.82E-01,
]
y2=[4.33E-02,
4.71E-02,
5.10E-02,
5.86E-02,
6.92E-02,
8.59E-02,
1.05E-01,
1.19E-01,
1.34E-01,
1.48E-01,
1.48E-01,
]
x3=[3.36E-02,
5.25E-02,
6.54E-02,
9.03E-02,
1.13E-01,
1.35E-01,
1.55E-01,
1.75E-01,
1.82E-01,
]
y3=[3.65E-02,
3.80E-02,
4.71E-02,
6.92E-02,
9.66E-02,
1.21E-01,
1.40E-01,
1.50E-01,
1.51E-01,
]
plt.plot(x2 , y2, color=color[0], linestyle='--', label="Experiment (Ma $et\ al$., 2017)")
plt.plot(x3 , y3, color=color[2], label="Super-ellipsoid (Ma $et\ al$., 2017)")
plt.plot(x1, y1, color=color[3], label="This study")
plt.xlabel("x (m)")
plt.ylabel('y (m)')
#plt.xlim([0,5])
#plt.ylim([0,1])
plt.tight_layout()
plt.legend(frameon=False)
plt.savefig ("profile.pdf")

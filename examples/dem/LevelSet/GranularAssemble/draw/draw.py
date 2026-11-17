#!/usr/bin/env python
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams

params = {
    "backend": "ps",
    "font.size": 26,
    "lines.linewidth": 4.5,
    "lines.markersize": 10,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "xtick.major.pad": 12,
    "ytick.major.pad": 12,
    "axes.labelpad": 8,
    "legend.fontsize": 26,
    "figure.figsize": [12, 9],
    "font.family": "serif",
    "text.usetex": True,
    "font.serif": "Arial",
    "savefig.dpi": 300,
}
rcParams.update(params)


color = [
    (0 / 255, 0 / 255, 0 / 255),
    (255 / 255, 0 / 255, 0 / 255),
    (94 / 255, 114 / 255, 255 / 255),
    (0 / 255, 128 / 255, 0 / 255),
]

x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
xT = [0.0, 0.05, 0.1, 0.15]
y1 = [
    1523.6244761943817,
    1523.6244761943817,
    1523.6244761943817,
    1523.6244761943817,
    1523.6244761943817,
    1523.6244761943817,
]
y2 = [1763.22470164299, 590.2318165302277, 524.6117603778839, 591.6413898468018, 652.2779026031494, 715.5190675258636]
y3 = [
    2032.3381659984589,
    610.5664749145508,
    469.0668578147888,
    455.7775876522064,
    501.09127831459045,
    559.5415236949921,
]
y4 = [2257.206956386566, 650.162957906723, 502.2999668121338, 474.2800860404968, 464.1654374599457, 511.8898801803589]
y = np.array([y1, y2, y3, y4])

z1 = [
    1736.9612028598785,
    1736.9612028598785,
    1736.9612028598785,
    1736.9612028598785,
    1736.9612028598785,
    1736.9612028598785,
]
z2 = [1932.7623410224915, 491.3294584751129, 504.59620428085327, 665.1574399471283, 773.508859872818, 874.2034590244293]
z3 = [2207.0326607227325, 517.7649660110474, 539.8581161499023, 553.449286699295, 663.4796271324158, 756.4779114723206]
z4 = [2469.231126308441, 577.3877367973328, 535.8891003131866, 632.4889814853668, 636.5228719711304, 721.999353647232]
z = np.array([z1, z2, z3, z4])

plt.plot(x, y[0, :], marker="o", markerfacecolor="none", markersize=15, color=color[0], label="$\Lambda$ = 0")
plt.plot(x, y[1, :], marker="h", markerfacecolor="none", markersize=15, color=color[1], label="$\Lambda$ = 0.05")
plt.plot(x, y[2, :], marker="v", markerfacecolor="none", markersize=15, color=color[2], label="$\Lambda$ = 0.1")
plt.plot(x, y[3, :], marker="D", markerfacecolor="none", markersize=15, color=color[3], label="$\Lambda$ = 0.15")
# plt.plot(x, y[4,:], marker='o', markerfacecolor='none', color=color[2], label="$\lambda$=0.4")
# plt.plot(x, y[5,:], marker='o', markerfacecolor='none', color=color[2], label="$\lambda$=0.5")
plt.xlim([0, 0.5])
plt.ylim([0.0, 2400])
plt.xlabel("$\lambda$")
plt.ylabel("Elapsed time (s)")
plt.tight_layout()
plt.legend(frameon=False)
plt.savefig("polysand.svg")
plt.close()

plt.plot(x, z[0, :], marker="o", markerfacecolor="none", markersize=15, color=color[0], label="$\Lambda$ = 0")
plt.plot(x, z[1, :], marker="h", markerfacecolor="none", markersize=15, color=color[1], label="$\Lambda$ = 0.05")
plt.plot(x, z[2, :], marker="v", markerfacecolor="none", markersize=15, color=color[2], label="$\Lambda$ = 0.1")
plt.plot(x, z[3, :], marker="D", markerfacecolor="none", markersize=15, color=color[3], label="$\Lambda$ = 0.15")
# plt.plot(x, z[4,:], marker='o', markerfacecolor='none', color=color[2], label="$\lambda$=0.4")
# plt.plot(x, z[5,:], marker='o', markerfacecolor='none', color=color[2], label="$\lambda$=0.5")
plt.xlim([0, 0.5])
plt.ylim([0.0, 2500])
plt.xlabel("$\lambda$")
plt.ylabel("Elapsed time (s)")
plt.tight_layout()
plt.legend(frameon=False)
plt.savefig("sand.svg")

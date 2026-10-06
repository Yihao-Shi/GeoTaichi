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

x = [(i + 1) * 0.15 for i in range(20)]
y1 = [
    0.0,
    1.7336816787719727,
    3.2856671810150146,
    5.229424476623535,
    7.829206705093384,
    11.405766248703003,
    16.435838222503662,
    23.235785245895386,
    31.813594818115234,
    42.77782607078552,
    57.75111508369446,
    76.01958107948303,
    100.27293825149536,
    121.48941421508789,
    154.35803151130676,
    197.07763385772705,
    230.31490564346313,
    253.04264116287231,
    271.72806692123413,
    289.2918004989624,
    306.7316060066223,
]
y2 = [
    0.0,
    2.9446356296539307,
    5.449858903884888,
    8.778921365737915,
    13.306620836257935,
    19.72439670562744,
    28.878723621368408,
    41.73024368286133,
    59.301183223724365,
    82.48548769950867,
    112.80185723304749,
    148.54392194747925,
    196.75419735908508,
    245.79856276512146,
    318.91233229637146,
    405.28996443748474,
    473.22322392463684,
    517.3718276023865,
    555.950835943222,
    596.2680490016937,
    633.0982933044434,
]
y3 = [
    0.0,
    3.609555721282959,
    7.252881050109863,
    11.937421798706055,
    18.5131254196167,
    28.09826159477234,
    41.616333961486816,
    60.8237521648407,
    87.44322299957275,
    122.63439393043518,
    168.16792154312134,
    227.5485680103302,
    300.1120934486389,
    376.0032260417938,
    487.0598256587982,
    616.6525478363037,
    721.0004432201385,
    795.1467893123627,
    859.661981344223,
    923.6308193206787,
    975.6770148277283,
]
y4 = [
    0.0,
    4.457944631576538,
    9.261386394500732,
    15.506669521331787,
    24.199403047561646,
    36.68672323226929,
    54.744601249694824,
    80.59905934333801,
    116.17204284667969,
    163.46731162071228,
    223.81858253479004,
    301.81655859947205,
    398.1513509750366,
    502.76288866996765,
    655.8980751037598,
    832.4270594120026,
    967.3126606941223,
    1058.998942375183,
    1147.06804728508,
    1233.3505618572235,
    1311.165939092636,
]
y5 = [
    0.0,
    5.412619113922119,
    11.592668056488037,
    20.012279510498047,
    31.89113163948059,
    49.206812143325806,
    72.83627104759216,
    105.62119817733765,
    150.11455726623535,
    209.5348515510559,
    286.5205397605896,
    389.6899013519287,
    511.98284578323364,
    670.6641998291016,
    862.7630391120911,
    1079.6192872524261,
    1262.6724863052368,
    1392.230715751648,
    1523.3479630947113,
    1657.9281992912292,
    1776.0751421451569,
]

y1 = [1500 / (y1[i + 1] - y1[i]) for i in range(len(y1) - 1)]
y2 = [1500 / (y2[i + 1] - y2[i]) for i in range(len(y2) - 1)]
y3 = [1500 / (y3[i + 1] - y3[i]) for i in range(len(y3) - 1)]
y4 = [1500 / (y4[i + 1] - y4[i]) for i in range(len(y4) - 1)]
y5 = [1500 / (y5[i + 1] - y5[i]) for i in range(len(y5) - 1)]

plt.plot(x, y1, marker="o", markerfacecolor="none", markersize=15, color=color[0], label="$N$ = 100000")
plt.plot(x, y2, marker="h", markerfacecolor="none", markersize=15, color=color[1], label="$N$ = 200000")
plt.plot(x, y3, marker="v", markerfacecolor="none", markersize=15, color=color[2], label="$N$ = 300000")
plt.plot(x, y4, marker="+", markerfacecolor="none", markersize=15, color=color[3], label="$N$ = 400000")
plt.plot(x, y5, marker="*", markerfacecolor="none", markersize=15, color="grey", label="$N$ = 540000")

plt.xlim([0, 3])
# plt.ylim([1., 1000])
plt.yscale("log")
plt.xlabel("Time (s)")
plt.ylabel("Simulation speed (steps/s)")
plt.tight_layout()
plt.legend(frameon=False)
plt.savefig("largesand.svg")
plt.close()

a = [100000, 200000, 300000, 400000, 540000]
b = [296.5343511104584, 564.4278094768524, 901.9137921333313, 1196.83811545372, 1776.0751421451569]
c = [1880 / 1024, 3160 / 1024, 4440 / 1024, 5720 / 1024, 7416 / 1024]
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(a, b, marker="o", markerfacecolor="none", markersize=15, color=color[0], label="Elapsed time")
ax2.plot(a, c, marker="^", markerfacecolor="none", markersize=15, color=color[1], label="Memory usage")
plt.xlim([100000, 600000])
ax1.set_ylim([200.0, 1800])
ax2.set_ylim([0.0, 8])
ax1.set_xlabel("Particle number")
ax1.set_ylabel("Elapsed time (s)")
ax2.set_ylabel("Memory usage (GB)", color=color[1])
ax2.tick_params(axis="y", labelcolor="red")
plt.tight_layout()
plt.savefig("totaltime.svg")
plt.close()

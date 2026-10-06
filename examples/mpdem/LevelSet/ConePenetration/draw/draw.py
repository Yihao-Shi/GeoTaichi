import matplotlib.pyplot as plt
from matplotlib import rcParams

# 配置风格
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

# 数据
drop_height = [0, 0.5, 1.0]
exp_pazouki = [3.28, 3.56, 3.85]
sim_hu = [3.25, 3.60, 3.81]
this_study = [3.22, 3.55, 3.80]

# 绘图
plt.plot(drop_height, exp_pazouki, marker="o", label="Exp. (Pazouki et al. 2017)")
plt.plot(drop_height, sim_hu, marker="s", label="Sim. (Hu et al. 2021)")
plt.plot(drop_height, this_study, marker="^", label="This study")

# 轴标签
plt.xlabel("Normalized drop height")
plt.ylabel("Penetration depth (cm)")

# 图例
plt.legend()

# 显示图形
plt.tight_layout()
plt.savefig("depth.svg")

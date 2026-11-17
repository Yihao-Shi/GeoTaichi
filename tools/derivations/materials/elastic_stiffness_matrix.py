import sys
from pathlib import Path

import numpy as np
repository_root = Path(__file__).resolve().parents[3]
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))

from tools.derivations.symfunc import *

lame1, lame2 = symbols('lambda mu')  # λ 和 μ
eps_ = make_vector('eps', 6, 1)
df_ds = make_vector('dfds', 6, 1)
dg_ds = make_vector('dgds', 6, 1)
eps = Matrix([[eps_[0], eps_[3], eps_[5]], [eps_[3], eps_[1], eps_[4]], [eps_[5], eps_[4], eps_[2]]])
dfds = Matrix([[df_ds[0], df_ds[3], df_ds[5]], [df_ds[3], df_ds[1], df_ds[4]], [df_ds[5], df_ds[4], df_ds[2]]])
dgds = Matrix([[dg_ds[0], dg_ds[3], dg_ds[5]], [dg_ds[3], dg_ds[1], dg_ds[4]], [dg_ds[5], dg_ds[4], dg_ds[2]]])
De_vigot = Matrix([[lame1+2*lame2, lame1, lame1, 0, 0, 0],[lame1, lame1+2*lame2, lame1, 0, 0, 0],[lame1, lame1, lame1+2*lame2, 0, 0, 0],[0, 0, 0, 2*lame2, 0, 0],[0, 0, 0, 0, 2*lame2, 0],[0, 0, 0, 0, 0, 2*lame2]])
dim = 3
k = eye(dim)  # 单位张量

# 初始化四阶张量
De = MutableDenseNDimArray.zeros(dim, dim, dim, dim)

# 构造 De[i,j,k,l] = λ δ_ij δ_kl + μ (δ_ik δ_jl + δ_il δ_jk)
for i in range(dim):
    for j in range(dim):
        for k_ in range(dim):
            for l in range(dim):
                De[i, j, k_, l] = (
                    lame1 * k[i, j] * k[k_, l] +
                    lame2 * (k[i, k_] * k[j, l] + k[i, l] * k[j, k_])
                )

eps_mat = Matrix(3, 3, lambda k, l: symbols(f'eps{k}{l}'))

# 计算 σ_{ij} = C_{ijkl} * ε_{kl}
sigma = MutableDenseNDimArray.zeros(3, 3)
for i in range(3):
    for j in range(3):
        s = 0
        for k in range(3):
            for l in range(3):
                s += De[i, j, k, l] * eps_mat[k, l]
        sigma[i, j] = simplify(s)

# 打印结果
for i in range(3):
    for j in range(3):
        print(f'sigma[{i},{j}] = {sigma[i,j]}')

s = 0
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                s += dfds[i,j] * De[i,j,k,l] * dgds[k,l]

temp = MutableDenseNDimArray.zeros(6)
for i in range(6):
    for j in range(6):
        temp[i] += De_vigot[i, j] * dg_ds[j]

t = 0
scale = [1, 1, 1, 2, 2, 2]
for i in range(6):
    t += scale[i] * temp[i] * df_ds[i]

print(simplify(s-t))

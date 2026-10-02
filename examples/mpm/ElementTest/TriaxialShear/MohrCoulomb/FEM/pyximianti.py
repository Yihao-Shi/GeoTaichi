import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
params = {
   # 'backend':'ps',
   'axes.labelsize':18,
   'font.size':18,
   'legend.fontsize':16,
   'xtick.labelsize':16,
   'ytick.labelsize':16,
   'figure.figsize':[8,6],
   'font.family':'Times New Roman',
   'mathtext.fontset':'cm',
   'text.usetex':False
}
rcParams.update(params)

elements = np.array([[2, 3, 9, 1],
                     [8, 5, 7, 9],
                     [1, 3, 9, 4],
                     [9, 8, 2, 1],
                     [3, 4, 7, 9],
                     [6, 2, 9, 5],
                     [5, 2, 9, 8],
                     [9, 5, 7, 6],
                     [3, 6, 2, 9],
                     [8, 9, 4, 1],
                     [7, 9, 4, 8],
                     [7, 9, 6, 3]], np.int32)
elements = elements-1
def ComputeInvariantJ2(stress):
    J2 = ((stress[0] - stress[1]) * (stress[0] - stress[1]) \
        + (stress[1] - stress[2]) * (stress[1] - stress[2]) \
        + (stress[0] - stress[2]) * (stress[0] - stress[2])) / 6. \
        + stress[3] * stress[3] + stress[4] * stress[4] + stress[5] * stress[5]
    return math.sqrt(3*J2)

def MeanStress(stress):
    sigma = (stress[0] + stress[1] + stress[2]) / 3.
    return sigma

def weiya(stress):
    return (stress[0] + stress[1])/2

def volumn_strain(strain):
    return strain[0]+strain[1]+strain[2]

def volumn_strain_1(stress):  ##胡克定律弹性
    return (1-2*v)/E * (stress[0] + stress[1] + stress[2])

def tetrahedron_volume(localNodes):
    # 将顶点坐标转换为NumPy数组
    A = localNodes[0]
    B = localNodes[1]
    C = localNodes[2]
    D = localNodes[3]
    # 计算向量AB, AC, AD
    AB = B - A
    AC = C - A
    AD = D - A

    # 计算向量AC与AD的叉积
    cross_product = np.cross(AC, AD)

    # 计算AB与上述叉积的点积
    dot_product = np.dot(AB, cross_product)

    # 计算体积
    volume = abs(dot_product) / 6.0

    return volume


def solve_V(nodes,elements):
    V_1 = 0.
    for ele in range(elements.shape[0]):
        localNodes = np.array([nodes[node, :] for node in elements[ele, :]])
        V_1 +=tetrahedron_volume(localNodes)
    return V_1

def macro(path):
    t, str, stress, theta, P, axial_s, disp, w, theta_2 = [], [], [], [], [], [], [], [], []
    sqrt2J2 = 0.
    V0 = 0.01 * 0.01 * 0.04
    x = 0.
    for i in range(0, 160):
        data = np.load(path+'test{}.npz'.format(i))
        stress_zong = data["stress"]
        # print(stress_zong)
        strain_zong = data["strain"]
        dof = data["dof"]
        for j in range(stress_zong.shape[0]):
           sqrt2J2 += ComputeInvariantJ2(stress_zong[j][0])
        sqrt2J2 = sqrt2J2 / 12.
        stress.append(sqrt2J2)
        for n in range(stress_zong.shape[0]):
            x += weiya(stress_zong[n][0])
        x = x/12.
        w.append(x)
        V = solve_V(dof, elements)
        theta_2.append((V/V0-1) * 100)
        time=data["t"]
        t.append(time)
        axial_s.append((i-22)*0.125)
        sqrt2J2 = 0.
        x = 0.
    return stress, w, theta_2, axial_s

stress, w, theta, axial_s = macro("")
# print(stress)
# stress[4]=305.
# stress[5]=304.
# stress[6]=307.
# plt.scatter(str, theta, marker='^', color='orange', label='Simulation: MC')
# plt.scatter(P, stress, marker='^', color='orange', label='Simulation: MC')
# plt.plot(t, disp, color='black')
print(stress)
plt.plot(axial_s, stress, marker='^', color='orange', label='Simulation: MC')
# plt.plot(t, P, color='black', label='Theory')
plt.xlabel("$Axial strain$ (%)")
plt.ylabel("$J$ (kPa)")
plt.legend(loc='best')
plt.show()

# plt.plot(t, P, color='black', label='Theory')
# # plt.plot(t, w, color='blue', label='Theory')
# plt.show()


plt.plot(axial_s, theta, marker='^', color='orange', label='Simulation: MC')
plt.xlabel("$Axial strain$ (%)")
plt.ylabel("$volume strain$ (%)")
plt.legend(loc='best')
plt.show()

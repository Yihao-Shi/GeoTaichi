import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import style

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
         

p0=100000
fai=30.5/180*math.pi
c=8500
start=10
end=61

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
    for i in range(0, 160, 4):
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
        axial_s.append((i-0)*0.125/100)
        sqrt2J2 = 0.
        x = 0.
    return stress, w, theta_2, axial_s

def calculate(path):
    q = []
    p = []
    q_p=[]
    epslion = []
    time = []
    vertical_stress0 = 0.
    for printNum in range(start, end):
        data = np.load(path+'MPMGrid{0:06d}.npz'.format(printNum))
        
        contact_force = data['contact_force']
        vertical_stress = np.sum(contact_force[:, 1], 0)[2] / 1.
       
        q.append((vertical_stress-p0)/1000)
        p.append((vertical_stress+2*p0)/3.)
        q_p.append((vertical_stress-p0)/((vertical_stress+2*p0)/3.))
        epslion.append((data['t_current']-0.1)*0.01)
        time.append(data['t_current'])
    return p, q, q_p, epslion, time

stress, w, theta, axial_s = macro("FEM/")
p1, q1, q_p1, epslion1, time1 = calculate('Biaxial/grids/')
p2, q2, q_p2, epslion2, time2 = calculate('Triaxial/grids/')
q0 = (0*np.array(epslion1)+p0*(1+math.sin(fai))/(1-math.sin(fai))+2*c*math.cos(fai)/(1-math.sin(fai))-p0)/1000
plt.plot(epslion1, q0, linestyle='--', color=color[0], label='Analytical')
plt.plot(epslion1, q1, marker='o', color=color[1], markerfacecolor='none', label='Biaxial test (MPM)')
plt.plot(epslion2, q2, marker='s', color=color[2], markerfacecolor='none', label='Triaxial test (MPM)')
plt.plot(axial_s, stress, marker='^', color='orange', label='FEM')
plt.xlabel("Axial strain, $\epsilon_a$ (\%)")
plt.ylabel("Equivalent stress, q (kPa)")
plt.legend(loc='best')
plt.xlim([0, 0.15])
plt.ylim([0, 250])
plt.tight_layout()
plt.savefig('material'+'.svg')
plt.close()

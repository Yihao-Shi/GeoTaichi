import argparse
import numpy as np
from pyevtk import hl
from pyevtk.vtk import VtkTetra
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Convert FEM stress snapshots to VTK.")
parser.add_argument("--input-dir", default=str(CASE_DIR), help="Directory containing testN.npz snapshots.")
parser.add_argument("--output-dir", default=str(CASE_DIR / "vtk"))
parser.add_argument("--frames", type=int, default=117)
arguments = parser.parse_args()
input_dir = Path(arguments.input_dir).expanduser().resolve()
output_dir = Path(arguments.output_dir).expanduser().resolve()
output_dir.mkdir(parents=True, exist_ok=True)

data = np.load(input_dir / "test{}.npz".format(0))
ele = data['ele']
# print(ele, ele.shape[0])
offsets = np.zeros(int(ele.shape[0]/4))
for i in range(int(ele.shape[0]/4)):
    offsets[i] = (i + 1) * 4
elements_type = np.zeros(int(ele.shape[0]/4))
for i in range(int(ele.shape[0]/4)):
    elements_type[i] = VtkTetra.tid
# print(offsets)
def Plotstreecloud(info, ele, offsets, elements_type, kinc):
    disp = info['disp']
    nodes = info['dof']
    stress = info['stress']
    stressxx = stress[:, :, 0]
    stressyy = stress[:, :, 1]
    stresszz = stress[:, :, 2]
    pointdata = disp.reshape(int(len(disp)/3),3)
    print(pointdata)
    x = np.ascontiguousarray(nodes[:, 0])
    y = np.ascontiguousarray(nodes[:, 1])
    z = np.ascontiguousarray(nodes[:, 2])
    stress_xx = stressxx.flatten()
    stress_yy = stressyy.flatten()
    stress_zz = stresszz.flatten()
    point_xx = pointdata[:, 0].flatten()
    point_yy = pointdata[:, 1].flatten()
    point_zz = pointdata[:, 2].flatten()
    stress_1 = {"sigmazz": stress_zz,
                "sigmaxx": stress_xx,
                "sigmayy": stress_yy}
    pointdata = {"displacementzz": point_zz,
                 "displacementxx": point_xx,
                 "displacementyy": point_yy
                 }
    hl.unstructuredGridToVTK(
        str(output_dir / "ph{}".format(kinc)),
        x, y, z, ele, offsets, elements_type, cellData=stress_1, pointData=pointdata) #

#

for i in range(arguments.frames):
    info = np.load(input_dir / "test{}.npz".format(i))
    stress = info['stress']
    stresszz = stress[:, :, 2]
    Plotstreecloud(info, ele, offsets,elements_type,i)

# n = []
# info_1 = np.load("test{}.npz".format(15))
# stress_2 = info_1['stress']
# stresszz_1 = stress_2[:, :, 2]
# stress_3 = stresszz_1.flatten()
# # print(len(stress_3))
# for j in range(len(stress_3)):
#     n.append(j)
# stress_3 = stress_3.tolist()
# plt.scatter(n, stress_3)
# plt.show()

#Plotstreecloud(info, ele, offsets,elements_type,0)

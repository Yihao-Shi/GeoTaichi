import gmsh
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[4] / "assets/mesh/Drums"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

gmsh.initialize()
gmsh.model.add("Drum")

# 参数
radius = 1.0
length = 0.2
lc = 0.05

# 创建几何
cyl = gmsh.model.occ.addCylinder(0, 0, -length / 2, 0, 0, length, radius)
front = gmsh.model.occ.addDisk(0, 0, -length / 2, radius, radius)
back = gmsh.model.occ.addDisk(0, 0, length / 2, radius, radius)

gmsh.model.occ.synchronize()

# 获取所有面
faces = gmsh.model.getEntities(2)
for f in faces:
    com = gmsh.model.occ.getCenterOfMass(f[0], f[1])
    if abs(com[2]) < 1e-6:
        side = f
    elif abs(com[2] + length / 2) < 1e-6:
        front_face = f
    elif abs(com[2] - length / 2) < 1e-6:
        back_face = f

# 设置网格
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)


def export_stl(face, filename):
    gmsh.model.addPhysicalGroup(2, [face[1]])
    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    gmsh.model.mesh.clear()
    gmsh.model.removePhysicalGroups()


export_stl(side, str(OUTPUT_DIR / "drum_side_raw.stl"))
export_stl(front_face, str(OUTPUT_DIR / "drum_front_raw.stl"))
export_stl(back_face, str(OUTPUT_DIR / "drum_back_raw.stl"))

gmsh.finalize()

import trimesh
import numpy as np
from trimesh.exchange.stl import export_stl_ascii


def inverse_normal(path):
    mesh = trimesh.load(path)

    faces = mesh.faces.copy()
    for idx in range(faces.shape[0]):
        faces[idx] = faces[idx][[0, 2, 1]]
    mesh.faces = faces

    ascii_data = export_stl_ascii(mesh)
    with open(path, "w") as f:
        f.write(ascii_data)


inverse_normal(OUTPUT_DIR / "drum_side_raw.stl")
inverse_normal(OUTPUT_DIR / "drum_back_raw.stl")

# 加载 STL
mesh = trimesh.load(OUTPUT_DIR / "drum_side_raw.stl")

# 可视化，并自动显示法向（鼠标交互）
mesh.show()

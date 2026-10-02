import argparse
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[4]

def particle_sphericity_from_stl(stl_path):
    mesh = trimesh.load_mesh(stl_path)

    if mesh.is_empty:
        raise ValueError("STL 为空或读取失败")

    # 如果是场景，合并成一个 mesh
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values()]
        )

    # 尝试修复
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()

    # 检查是否闭合
    if not mesh.is_watertight:
        raise ValueError("网格不是封闭的，体积可能不可靠，不能直接计算球度")

    V = mesh.volume
    A = mesh.area

    psi = (np.pi ** (1/3) * (6 * V) ** (2/3)) / A
    return psi, V, A

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--hamburg-mesh",
    default=str(ROOT / "assets" / "mesh" / "LSDEM" / "Hamburg_sand.stl"),
)
parser.add_argument(
    "--sand-mesh",
    default=str(ROOT / "assets" / "mesh" / "LSDEM" / "sand.stl"),
)
arguments = parser.parse_args()

psi, V, A = particle_sphericity_from_stl(arguments.hamburg_mesh)
print("体积 V =", V)
print("表面积 A =", A)
print("球度 ψ =", psi)

psi, V, A = particle_sphericity_from_stl(arguments.sand_mesh)
print("体积 V =", V)
print("表面积 A =", A)
print("球度 ψ =", psi)

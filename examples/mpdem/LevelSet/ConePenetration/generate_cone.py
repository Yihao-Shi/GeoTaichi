import gmsh
import math
from pathlib import Path

OUTPUT = Path(__file__).resolve().parents[4] / "assets/mesh/LSDEM/cone.stl"

gmsh.initialize()
gmsh.model.add("cone_example")

# === 参数定义 ===
r1 = 0.0  # 底半径
r2 = 0.01  # 顶半径（0 表示圆锥）
h = 0.022  # 高度

# === 创建圆锥体 ===
# 圆锥底部在 (0,0,0)，高度方向沿 z 轴
cone = gmsh.model.occ.addCone(0, 0, 0, 0, 0, h, r1, r2)

# 同步 CAD 与 Gmsh 模型
gmsh.model.occ.synchronize()

# === 设置网格密度（越小越密） ===
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.0005)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.001)

# === 网格生成 ===
gmsh.model.mesh.generate(3)

# === 导出 STL 文件 ===
gmsh.write(str(OUTPUT))

print(f"✅ 圆锥 {OUTPUT} 已成功生成！")

# 若需要查看可视化窗口：
# gmsh.fltk.run()

gmsh.finalize()

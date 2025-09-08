import trimesh

# 加载 STL
mesh = trimesh.load('drum_side_raw.stl')

# 可视化，并自动显示法向（鼠标交互）
mesh.show()

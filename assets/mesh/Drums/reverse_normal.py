import meshio
import numpy as np

def reverse_stl(input_file, output_file):
    mesh = meshio.read(input_file)
    
    if "triangle" not in mesh.cells_dict:
        raise RuntimeError(f"No triangle cells in {input_file}")
    
    cells = mesh.cells_dict["triangle"]
    reversed_cells = cells[:, [0, 2, 1]]  # 交换顶点顺序

    meshio.write_points_cells(
        output_file,
        mesh.points,
        [("triangle", reversed_cells)],
        file_format="stl"  # ✅ 修正：不再写错为 "stl-ascii"
    )

# 批量处理
reverse_stl("drum_side_raw.stl",  "drum_side_inward.stl")
reverse_stl("drum_front_raw.stl", "drum_front_inward.stl")
reverse_stl("drum_back_raw.stl",  "drum_back_inward.stl")

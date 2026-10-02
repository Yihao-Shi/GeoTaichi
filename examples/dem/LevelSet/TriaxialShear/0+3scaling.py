import argparse
import numpy as np
import os
from pathlib import Path
import trimesh  # 导入 trimesh 库

CASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a sphere packing to an irregular-particle packing.")
    parser.add_argument("--template-mesh", required=True, help="STL mesh used as the irregular-particle template.")
    parser.add_argument(
        "--input-dir",
        default=str(CASE_DIR / "OutputData" / "DEM_Generation"),
        help="DEM result directory containing particles/DEMParticleXXXXXX.npz.",
    )
    parser.add_argument("--output-file", default=None, help="Output TXT file (default: INPUT_DIR/BoundingSphere1.txt).")
    parser.add_argument("--frame", type=int, default=5, help="Particle frame number to convert.")
    return parser.parse_args()

def get_template_properties_from_stl(stl_file_path):
    """
    加载一个 STL 文件并计算其作为不规则颗粒模板所需的几何属性。

    参数:
    stl_file_path (str): STL 文件的路径。

    返回:
    dict: 包含模板几何属性的字典。
    """
    print("-" * 70)
    print(f"正在从STL文件加载模板: {os.path.basename(stl_file_path)}")
    
    try:
        # 1. 使用 trimesh 加载 STL 网格文件
        mesh = trimesh.load_mesh(stl_file_path)
    except Exception as e:
        print(f"错误: 无法加载或处理STL文件 '{stl_file_path}'.")
        print(f"具体错误: {e}")
        return None

    # 确保网格是“水密的”(watertight)，这样体积和质心计算才准确
    if not mesh.is_watertight:
        print("警告: STL网格不是水密的(not watertight)。体积和质心可能不准确。")
        # 尝试修复网格，但这并不总是成功
        mesh.fill_holes()

    # 2. 计算并提取所需的几何属性
    # a) 最小外接包围球
    bounding_sphere_center = mesh.bounding_sphere.primitive.center
    bounding_sphere_radius = mesh.bounding_sphere.primitive.radius
    
    # b) 质心
    center_of_mass = mesh.center_mass
    
    # c) 体积和等体积等效半径
    volume = mesh.volume
    # 公式 V = (4/3) * pi * r^3  =>  r = ((3 * V) / (4 * pi))^(1/3)
    equivalent_radius = ( (3 * volume) / (4 * np.pi) ) ** (1/3.0)

    # 3. 将结果存入字典并打印
    properties = {
        "radius_bounding_sphere": bounding_sphere_radius,
        "center_of_bounding_sphere": bounding_sphere_center,
        "center_of_mass": center_of_mass,
        "equivalent_radius": equivalent_radius,
        "volume": volume
    }
    
    print("\n--- 模板颗粒属性 ---")
    print(f"体积 (Volume) = {properties['volume']:.6e}")
    print(f"等效半径 (Equivalent Radius) = {properties['equivalent_radius']:.6f}")
    print(f"质心 (Center of Mass) = {np.array2string(properties['center_of_mass'], formatter={'float_kind':lambda x: f'{x:.6f}'})}")
    print(f"最小包围球球心 (BS Center) = {np.array2string(properties['center_of_bounding_sphere'], formatter={'float_kind':lambda x: f'{x:.6f}'})}")
    print(f"最小包圍球半徑 (BS Radius) = {properties['radius_bounding_sphere']:.6f}")
    print("-" * 70)
    
    return properties

def convert_sphere_packing_to_irregular(
    input_folder, 
    output_file, 
    p_num,
    template_properties
):
    """
    将球体堆积数据转换为不规则颗粒堆积数据。
    """
    print(f"开始转换文件: 'DEMParticle{p_num:06d}.npz'...")

    npz_file_path = os.path.join(input_folder, 'particles', f'DEMParticle{p_num:06d}.npz')

    try:
        # --- 1. 加载原始球体堆积数据 ---
        particle_data = np.load(npz_file_path, allow_pickle=True)
        original_positions = particle_data["position"]
        original_radii = particle_data["radius"]
        num_particles = len(original_radii)

        if num_particles == 0:
            print("警告：输入文件中未找到颗粒数据。")
            return
        print(f"在文件中检测到 {num_particles} 个颗粒。")
        
        # c) 确定要写入文件的半径（即每个颗粒的包围球半径）
        new_bounding_radii = original_radii

        print("已完成所有颗粒的位置和半径转换计算。")

        # --- 3. 生成随机方向 ---
        direction_vectors = 360.0 * np.random.rand(num_particles, 3)
        print("已生成所有随机方向。")

        # --- 4. 写入文件 ---
        with open(output_file, 'w') as f_out:
            print ("2")
            header = "#      PositionX            PositionY                PositionZ            Radius            DirX            DirY            DirZ\n"
            f_out.write(header)

            for i in range(num_particles):
                pos_x, pos_y, pos_z = original_positions[i]
                out_radius = new_bounding_radii[i]
                dir_x, dir_y, dir_z = direction_vectors[i]
                
                output_line = (
                    f"{pos_x:.18e} {pos_y:.18e} {pos_z:.18e} "
                    f"{out_radius:.18e} {dir_x:.18e} {dir_y:.18e} {dir_z:.18e}\n"
                )
                f_out.write(output_line)

        print(f"\n处理完成！新的颗粒文件已保存至 '{output_file}'")

    except FileNotFoundError:
        print(f"错误：找不到输入文件 '{npz_file_path}'")
    except Exception as e:
        print(f"处理文件时发生未知错误: {e}")

def calculate_sphere_radius_from_stl(
    stl_file_path: str, 
    target_equivalent_radius: float
) -> float:
    """
    根据目标等效半径和STL模板文件，反算初始堆积中所需球体的半径。

    参数:
    stl_file_path (str): 模板不规则颗粒的STL文件路径。
    target_equivalent_radius (float): 您最终想要得到的不规则颗粒的等效半径。

    返回:
    float: 在生成初始球体堆积时，您应该使用的球体半径。
    """
    # 1. 从STL文件获取模板的几何属性
    template_props = get_template_properties_from_stl(stl_file_path)
    
    if template_props is None:
        raise ValueError("无法从STL文件获取模板属性，计算中止。")
    
    template_equivalent_radius = template_props['equivalent_radius']
    template_bounding_sphere_radius = template_props['radius_bounding_sphere']
    
    if template_equivalent_radius <= 1e-9: # 增加一个小的容错
        raise ValueError("模板的等效半径为零或过小，无法进行计算。")
        
    # 2. 计算缩放系数 S
    scaling_factor = target_equivalent_radius / template_equivalent_radius
    
    # 3. 用缩放系数 S 计算所需的初始球体半径
    required_sphere_radius = template_bounding_sphere_radius * scaling_factor
    factor = required_sphere_radius / target_equivalent_radius
    return required_sphere_radius, factor

# --- 如何使用 ---

if __name__ == "__main__":
    arguments = parse_args()
    input_folder = Path(arguments.input_dir).expanduser().resolve()
    output_filename = (
        Path(arguments.output_file).expanduser().resolve()
        if arguments.output_file
        else input_folder / "BoundingSphere1.txt"
    )
    output_filename.parent.mkdir(parents=True, exist_ok=True)

    # 3. 从STL文件自动获取模板属性
    template_props = get_template_properties_from_stl(str(Path(arguments.template_mesh).expanduser().resolve()))

    # 4. 检查属性是否成功获取，然后运行转换
    if template_props:
        print("1")
        convert_sphere_packing_to_irregular(
            input_folder=str(input_folder),
            output_file=str(output_filename),
            p_num=arguments.frame,
            template_properties=template_props
        )

# if __name__ == "__main__":
#     try:
#         # 1. 指定您的不规则颗粒模板 STL 文件路径，例如 arguments.template_mesh。

#         # 2. 指定您想要生成的最终不规则颗粒的等效半径
#         target_size = 0.0005

#         # 3. 调用函数，一步完成计算
#         initial_sphere_radius, scaling_factor = calculate_sphere_radius_from_stl(
#             stl_file_path=stl_path,
#             target_equivalent_radius=target_size
#         )
        
#         print("\n--- 计算结果 ---")
#         print(f"为了得到一个等效半径为 {target_size:.6f} 的不规则颗粒,")
#         print(f"您需要在初始堆积中生成一个半径为: {initial_sphere_radius:.18e} 的球体。")
#         print(f"放大系数: {scaling_factor:.18e}")


#     except (ValueError, FileNotFoundError) as e:
#         print(f"\n计算失败: {e}")

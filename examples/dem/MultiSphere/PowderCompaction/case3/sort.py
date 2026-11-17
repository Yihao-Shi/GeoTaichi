import argparse
from pathlib import Path
import numpy as np

CASE_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Sort a sphere packing by particle radius.")
parser.add_argument("input_file", help="SpherePacking.txt to sort.")
parser.add_argument("--output-file", default=str(CASE_DIR / "SpherePacking_sorted.txt"))
arguments = parser.parse_args()
in_file = Path(arguments.input_file).expanduser().resolve()
destination_file = Path(arguments.output_file).expanduser().resolve()
destination_file.parent.mkdir(parents=True, exist_ok=True)
# 1. 读取数据
# ---------------------------------------------------------

# 实际读取代码：
# np.loadtxt 默认把 # 开头的行当作注释忽略，非常方便
data = np.loadtxt(in_file)

# 2. 获取排序的索引
# data[:, 3] 表示获取第4列 (即 Radius 列)
# argsort 返回的是排序后的索引数组
sorted_indices = np.argsort(data[:, 3])

# 3. 根据索引重排数据
sorted_data = data[sorted_indices]

print("排序后的数据 (NumPy 数组):")
print(sorted_data)

# 保存
np.savetxt(destination_file, sorted_data, header='PositionX PositionY PositionZ Radius')

import argparse
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = Path(__file__).resolve().parents[5] / "assets/data/DEM/PowderCompaction/input.txt"
scale_fac = 0.0010  # 缩放因子 (微米->毫米->米)


def generate_particle_file(input_filename, output_filename):
    try:
        with open(input_filename, "r", encoding="utf-8") as f_in:
            lines = f_in.readlines()
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{input_filename}'，请检查文件路径。")
        return

    k = 0.0
    count = 0
    radius_max = 0
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    with output_filename.open("w", encoding="utf-8") as f_out:
        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                d_microns = float(parts[0])
                mass_percent = float(parts[1])
            except ValueError:
                continue

            fraction = mass_percent / 100.0

            # 计算半径并乘以缩放因子
            # x/2000 是将微米直径转换为毫米半径
            radius_val = (d_microns / 2000.0) * scale_fac

            if radius_val > radius_max:
                radius_max = radius_val

            if fraction > 0.0:
                output_block = (
                    f"{{\n"
                    f'    "GroupID": 0, \n'
                    f'    "MaterialID": 0, \n'
                    f'    "MinRadius": {radius_val:.12f}, \n'
                    f'    "MaxRadius": {radius_val:.12f}, \n'
                    f'    "BodyOrientation": "uniform", \n'
                    f'    "Fraction":{fraction:.10f}}}, \n'
                )

                f_out.write(output_block)

                k += fraction
                count += 1

    print(f"共生成 {count} 个条目。")
    print(f"Fraction 总和 (k): {k:.6f}")
    print(radius_max)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a particle-size distribution to GeoTaichi fractions.")
    parser.add_argument(
        "input_file",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help="Text file containing diameter and mass-percent columns.",
    )
    parser.add_argument("--output-file", default=str(CASE_DIR / "fractions.txt"))
    arguments = parser.parse_args()
    generate_particle_file(
        Path(arguments.input_file).expanduser().resolve(),
        Path(arguments.output_file).expanduser().resolve(),
    )

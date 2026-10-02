#!/usr/bin/env pvpython
# -*- coding: utf-8 -*-

"""
使用 ParaView 渲染 MPM + DEM + FEM VTU 时间序列，并生成 GIF。

功能：
1. 自动查找和配对 MPM、DEM 的 VTU 文件；
2. MPM 按指定标量（默认 pressure）着色；
3. 仅裁剪 MPM，DEM 保持完整；
4. 默认固定颜色范围和相机，可选逐帧零点对称色标；
5. 输出 PNG 序列；
6. 调用 FFmpeg 合成高质量 GIF。

示例：

    pvpython paraview_mpm_dem_gif.py ./VTU \
        --mpm-glob "*MPM*.vtu" \
        --dem-glob "*DEM*.vtu" \
        --scalar pressure \
        --clip-normal 1 0 0 \
        --fps 15 \
        --width 1200 \
        --height 900 \
        -o result.gif

无桌面环境或服务器上：

    pvpython --force-offscreen-rendering \
        paraview_mpm_dem_gif.py ./VTU -o result.gif
"""

import os
import re
import sys
import math
import glob
import shutil
import argparse
import tempfile
import subprocess
from pathlib import Path

# ParaView 必须使用 pvpython 运行
try:
    from paraview.simple import (
        _DisableFirstRenderCameraReset,
        OpenDataFile,
        GetActiveViewOrCreate,
        GetAnimationScene,
        GetTimeKeeper,
        Show,
        Hide,
        Clip,
        Glyph,
        ColorBy,
        GetColorTransferFunction,
        GetOpacityTransferFunction,
        GetScalarBar,
        SaveScreenshot,
        Render,
        ResetCamera,
        Delete,
    )
except ImportError as exc:
    print(
        "\n错误：无法导入 paraview.simple。\n"
        "请不要使用普通 python 运行，应使用 ParaView 自带的 pvpython：\n\n"
        "    pvpython paraview_mpm_dem_gif.py 数据目录 -o result.gif\n",
        file=sys.stderr,
    )
    raise exc


# ----------------------------------------------------------------------
# 参数处理
# ----------------------------------------------------------------------


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="使用 ParaView 渲染 MPM + DEM VTU 序列并生成 GIF。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input_dir",
        help="包含 MPM、DEM、FEM VTU 文件的目录。",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="mpm_dem.gif",
        help="输出 GIF 文件路径。",
    )

    parser.add_argument(
        "--mpm-glob",
        default="*GraphicMPMParticle*.vtu",
        help="MPM 文件匹配模式，区分大小写。",
    )

    parser.add_argument(
        "--dem-glob",
        default="*GraphicAffineBody*.vtu",
        help="DEM 文件匹配模式，区分大小写。",
    )

    parser.add_argument(
        "--fem-glob",
        default="*FEM*.vtu",
        help="FEM 文件匹配模式，区分大小写；找不到时自动忽略 FEM。",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归搜索输入目录下的子目录。",
    )

    parser.add_argument(
        "--step-regex",
        default=r"(\d+)(?!.*\d)",
        help="从文件名提取时间步编号的正则表达式，默认提取最后一组数字。",
    )

    parser.add_argument(
        "--scalar",
        default="pressure",
        help="MPM 着色使用的数组名称，例如 pressure 或 velocity。",
    )

    parser.add_argument(
        "--scalar-mode",
        choices=("auto", "value", "magnitude"),
        default="auto",
        help=(
            "数组着色方式：auto 表示标量数组用自身数值、矢量数组用 magnitude；"
            "value 表示始终使用标量值/第一个分量；"
            "magnitude 表示对多分量数组使用模长。"
        ),
    )

    parser.add_argument(
        "--color-min",
        type=float,
        default=None,
        help="颜色映射最小值；不指定时自动扫描全部时间步。",
    )

    parser.add_argument(
        "--color-max",
        type=float,
        default=None,
        help="颜色映射最大值；不指定时自动扫描全部时间步。",
    )

    parser.add_argument(
        "--color-preset",
        default="Viridis (matplotlib)",
        help="ParaView 颜色预设名称。",
    )

    parser.add_argument(
        "--color-range-mode",
        choices=("fixed", "frame-symmetric"),
        default="fixed",
        help="fixed 使用全局色标；frame-symmetric 每帧按最大绝对值设置零点对称色标。",
    )

    parser.add_argument(
        "--log-color",
        action="store_true",
        help="使用对数颜色映射，数据和颜色范围必须大于零。",
    )

    parser.add_argument(
        "--hide-colorbar",
        action="store_true",
        help="隐藏 pressure 色标。",
    )

    parser.add_argument(
        "--colorbar-title",
        default=None,
        help="色标标题；默认使用标量名称。",
    )

    # 裁剪设置
    parser.add_argument(
        "--clip",
        action="store_true",
        help="启用 MPM 裁剪。默认不裁剪；只有显式指定 --clip 时才启用。",
    )

    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="不裁剪 MPM（兼容旧参数；当前默认本身就是不裁剪）。",
    )

    parser.add_argument(
        "--clip-origin",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="裁剪平面的原点；仅在使用 --clip 时生效。不指定时使用 MPM 包围盒中心。",
    )

    parser.add_argument(
        "--clip-normal",
        nargs=3,
        type=float,
        metavar=("NX", "NY", "NZ"),
        default=(1.0, 0.0, 0.0),
        help="裁剪平面的法向；仅在使用 --clip 时生效。",
    )

    parser.add_argument(
        "--clip-invert",
        action="store_true",
        help="反转 MPM 裁剪方向；仅在使用 --clip 时生效。",
    )

    # DEM 外观
    parser.add_argument(
        "--dem-representation",
        choices=("Points", "Surface", "Wireframe"),
        default="Surface",
        help="DEM 显示方式。",
    )

    parser.add_argument(
        "--dem-point-size",
        type=float,
        default=5.0,
        help="DEM 点大小，仅 Points 模式有效。",
    )

    parser.add_argument(
        "--dem-color",
        nargs=3,
        type=float,
        metavar=("R", "G", "B"),
        default=(0.20, 0.20, 0.20),
        help="DEM RGB 颜色，每个值范围为 0～1。",
    )

    parser.add_argument(
        "--dem-opacity",
        type=float,
        default=1.0,
        help="DEM 不透明度，范围为 0～1。",
    )

    parser.add_argument(
        "--dem-radius-array",
        default="radius",
        help="DEM 半径数组名称；若存在于 PointData，则自动用 Glyph(Sphere) 生成球体。",
    )

    parser.add_argument(
        "--no-dem-glyph",
        action="store_true",
        help="即使存在 radius 数组，也禁用 DEM 自动 Glyph。",
    )

    parser.add_argument(
        "--dem-glyph-theta-resolution",
        type=int,
        default=20,
        help="DEM Glyph 球体经度方向分辨率。",
    )

    parser.add_argument(
        "--dem-glyph-phi-resolution",
        type=int,
        default=20,
        help="DEM Glyph 球体纬度方向分辨率。",
    )

    # FEM 外观 / 标量
    parser.add_argument(
        "--fem-scalar",
        default="von_mesis",
        help="FEM 着色数组名称，默认 von_mesis。",
    )

    parser.add_argument(
        "--fem-representation",
        choices=("Surface", "Surface With Edges", "Wireframe", "Points"),
        default="Surface",
        help="FEM 显示方式。",
    )

    parser.add_argument(
        "--fem-opacity",
        type=float,
        default=1.0,
        help="FEM 不透明度，范围为 0～1。",
    )

    parser.add_argument(
        "--fem-point-size",
        type=float,
        default=5.0,
        help="FEM 点大小，仅 Points 模式有效。",
    )

    parser.add_argument(
        "--fem-color-min",
        type=float,
        default=None,
        help="FEM von_mesis 色标最小值；不指定时自动扫描全部时间步。",
    )

    parser.add_argument(
        "--fem-color-max",
        type=float,
        default=None,
        help="FEM von_mesis 色标最大值；不指定时自动扫描全部时间步。",
    )

    parser.add_argument(
        "--fem-color-preset",
        default="Viridis (matplotlib)",
        help="FEM 颜色预设名称。",
    )

    parser.add_argument(
        "--hide-fem-colorbar",
        action="store_true",
        help="隐藏 FEM 的 von_mesis 色标。",
    )

    # MPM 外观
    parser.add_argument(
        "--mpm-opacity",
        type=float,
        default=1.0,
        help="MPM 不透明度，范围为 0～1。",
    )

    parser.add_argument(
        "--mpm-representation",
        choices=("Surface", "Surface With Edges", "Points", "Wireframe"),
        default="Points",
        help="MPM 显示方式。",
    )

    parser.add_argument(
        "--mpm-point-size",
        type=float,
        default=5.0,
        help="MPM 点大小，仅 Points 模式有效。",
    )

    # 相机设置
    parser.add_argument(
        "--camera-view",
        choices=("+x", "-x", "+y", "-y", "+z", "-z", "xy+", "custom"),
        default="custom",
        help=(
            "预设相机观察方向。+x/-x/+y/-y/+z/-z 为标准轴向视图；"
            "xy+ 表示相机位于 +X/+Y 且略高于 +Z，朝模型中心斜看；"
            "custom 使用手动相机参数。"
        ),
    )

    parser.add_argument(
        "--camera-position",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="相机位置。",
    )

    parser.add_argument(
        "--camera-focal-point",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="相机焦点。",
    )

    parser.add_argument(
        "--camera-view-up",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 1.0),
        help="相机向上方向。",
    )

    parser.add_argument(
        "--parallel-scale",
        type=float,
        default=None,
        help="并行投影缩放值；不指定时由 ParaView 自动确定。",
    )

    parser.add_argument(
        "--camera-padding",
        type=float,
        default=1.05,
        help="自动相机视野留白倍数。",
    )

    parser.add_argument(
        "--perspective",
        action="store_true",
        help="使用透视投影；默认使用并行投影。",
    )

    parser.add_argument(
        "--interaction-mode",
        choices=("2D", "3D"),
        default="2D",
        help="ParaView 交互模式；2D 适合固定平面视图。",
    )

    # 图像和动画设置
    parser.add_argument(
        "--width",
        type=int,
        default=1200,
        help="输出图像宽度。",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=900,
        help="输出图像高度。",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="GIF 帧率。",
    )

    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="每隔多少个时间步输出一帧。",
    )

    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="开始渲染的时间步索引。",
    )

    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="结束渲染的时间步索引，包含该帧。",
    )

    parser.add_argument(
        "--background",
        nargs=3,
        type=float,
        metavar=("R", "G", "B"),
        default=(1.0, 1.0, 1.0),
        help="背景颜色，每个值范围为 0～1。",
    )

    parser.add_argument(
        "--transparent-background",
        action="store_true",
        help="PNG 使用透明背景；GIF 最终透明效果取决于 FFmpeg。",
    )

    parser.add_argument(
        "--show-axes",
        action="store_true",
        help="显示方向坐标轴。",
    )

    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="FFmpeg 可执行文件路径。",
    )

    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="保留中间 PNG 文件。",
    )

    parser.add_argument(
        "--frames-dir",
        default=None,
        help="中间 PNG 输出目录；不指定时使用临时目录。",
    )

    parser.add_argument(
        "--skip-gif",
        action="store_true",
        help="只输出 PNG 序列，不调用 FFmpeg。",
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
# 文件搜索和配对
# ----------------------------------------------------------------------


def natural_sort_key(path):
    """按文件名中的数字自然排序。"""
    text = str(path)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def find_files(input_dir, pattern, recursive=False):
    input_dir = os.path.abspath(input_dir)

    if recursive:
        search_pattern = os.path.join(input_dir, "**", pattern)
        files = glob.glob(search_pattern, recursive=True)
    else:
        search_pattern = os.path.join(input_dir, pattern)
        files = glob.glob(search_pattern)

    files = [os.path.abspath(path) for path in files if os.path.isfile(path)]

    return sorted(files, key=natural_sort_key)


def extract_step_number(filename, compiled_regex):
    """
    从文件名提取时间步。

    如果正则存在捕获组，使用第一个捕获组；
    否则使用完整匹配结果。
    """
    basename = os.path.basename(filename)
    match = compiled_regex.search(basename)

    if not match:
        return None

    value = match.group(1) if match.groups() else match.group(0)

    try:
        return int(value)
    except ValueError:
        return value


def build_step_map(files, compiled_regex, label):
    result = {}

    for filename in files:
        step = extract_step_number(filename, compiled_regex)

        if step is None:
            continue

        if step in result:
            raise RuntimeError(
                "{} 文件存在重复时间步编号 {}：\n  {}\n  {}".format(
                    label,
                    step,
                    result[step],
                    filename,
                )
            )

        result[step] = filename

    return result


def align_optional_file_sequences(source_files, step_regex):
    """
    对 MPM / DEM / FEM 任意可选时间序列进行对齐。

    - 某类文件为空：直接忽略该模块。
    - 只有一个模块存在：使用其全部时间步。
    - 多个模块都能从文件名提取 step：取公共 step。
    - 无法统一提取 step：按自然排序，以最短序列长度对齐。
    """
    active = {name: files for name, files in source_files.items() if files}

    if not active:
        raise RuntimeError("MPM、DEM、FEM 均未找到任何文件。")

    compiled_regex = re.compile(step_regex)
    step_maps = {}
    extractable = {}

    for name, files in active.items():
        mapping = build_step_map(files, compiled_regex, name)
        step_maps[name] = mapping
        extractable[name] = bool(mapping)

    # 只有一个模块
    if len(active) == 1:
        name, files = next(iter(active.items()))

        if extractable[name]:
            steps = sorted(step_maps[name].keys())
            aligned = {}
            for key in source_files:
                if key == name:
                    aligned[key] = [step_maps[name][step] for step in steps]
                else:
                    aligned[key] = [None] * len(steps)
        else:
            steps = list(range(len(files)))
            aligned = {}
            for key in source_files:
                aligned[key] = list(files) if key == name else [None] * len(steps)

        return steps, aligned

    # 多模块，且所有存在模块都能提取 step
    if all(extractable.values()):
        common_steps = None

        for name in active:
            current = set(step_maps[name].keys())
            common_steps = current if common_steps is None else common_steps & current

        common_steps = sorted(common_steps or [])

        if not common_steps:
            raise RuntimeError("已找到多个数据模块，但它们之间没有共同时间步。")

        aligned = {}
        for name in source_files:
            if name in active:
                aligned[name] = [step_maps[name][step] for step in common_steps]
            else:
                aligned[name] = [None] * len(common_steps)

        for name in active:
            ignored = sorted(set(step_maps[name].keys()) - set(common_steps))
            if ignored:
                print(
                    "警告：{} 的这些时间步未与其它模块对齐，将忽略：{}".format(
                        name,
                        ignored,
                    )
                )

        return common_steps, aligned

    # 部分模块无法提取 step：按文件顺序对齐
    print("警告：部分模块无法统一提取时间步；" "将按自然排序并以最短序列长度对齐。")

    count = min(len(files) for files in active.values())
    steps = list(range(count))

    aligned = {}
    for name in source_files:
        if name in active:
            aligned[name] = list(active[name][:count])
        else:
            aligned[name] = [None] * count

    return steps, aligned


def pair_mpm_dem_files(mpm_files, dem_files, step_regex):
    """
    优先根据时间步编号配对。

    如果所有文件都无法提取编号，则按自然排序顺序配对。
    """
    compiled_regex = re.compile(step_regex)

    mpm_map = build_step_map(mpm_files, compiled_regex, "MPM")
    dem_map = build_step_map(dem_files, compiled_regex, "DEM")

    if not mpm_map and not dem_map:
        if len(mpm_files) != len(dem_files):
            raise RuntimeError(
                "无法从文件名提取时间步，并且 MPM/DEM 文件数量不同：" "{} != {}".format(len(mpm_files), len(dem_files))
            )

        print("警告：未提取到时间步编号，将按自然排序顺序配对。")

        return [(index, mpm_file, dem_file) for index, (mpm_file, dem_file) in enumerate(zip(mpm_files, dem_files))]

    common_steps = sorted(set(mpm_map.keys()) & set(dem_map.keys()))

    missing_dem = sorted(set(mpm_map.keys()) - set(dem_map.keys()))
    missing_mpm = sorted(set(dem_map.keys()) - set(mpm_map.keys()))

    if missing_dem:
        print("警告：以下 MPM 时间步没有对应 DEM，将忽略：{}".format(missing_dem))

    if missing_mpm:
        print("警告：以下 DEM 时间步没有对应 MPM，将忽略：{}".format(missing_mpm))

    if not common_steps:
        raise RuntimeError("MPM 和 DEM 文件之间没有共同时间步。")

    return [(step, mpm_map[step], dem_map[step]) for step in common_steps]


# ----------------------------------------------------------------------
# ParaView 数据辅助函数
# ----------------------------------------------------------------------


def property_exists(proxy, property_name):
    try:
        return property_name in proxy.ListProperties()
    except Exception:
        return hasattr(proxy, property_name)


def set_property_if_exists(proxy, property_name, value):
    if property_exists(proxy, property_name):
        try:
            setattr(proxy, property_name, value)
            return True
        except Exception:
            return False
    return False


def valid_bounds(bounds):
    if bounds is None or len(bounds) != 6:
        return False

    return all(math.isfinite(value) for value in bounds) and (
        bounds[0] <= bounds[1] and bounds[2] <= bounds[3] and bounds[4] <= bounds[5]
    )


def bounds_center(bounds):
    return (
        0.5 * (bounds[0] + bounds[1]),
        0.5 * (bounds[2] + bounds[3]),
        0.5 * (bounds[4] + bounds[5]),
    )


def combine_bounds(*bounds_list):
    """合并多个有效包围盒，用于自动相机居中。"""
    valid = [bounds for bounds in bounds_list if valid_bounds(bounds)]

    if not valid:
        return None

    return (
        min(bounds[0] for bounds in valid),
        max(bounds[1] for bounds in valid),
        min(bounds[2] for bounds in valid),
        max(bounds[3] for bounds in valid),
        min(bounds[4] for bounds in valid),
        max(bounds[5] for bounds in valid),
    )


def camera_preset(view_name):
    """
    返回 (view_direction, view_up)。

    其中：
        view_direction = CameraFocalPoint - CameraPosition

    因此：
        +x 表示沿 +X 看
        -x 表示沿 -X 看
        +y 表示沿 +Y 看
        -y 表示沿 -Y 看
        +z 表示沿 +Z 看
        -z 表示沿 -Z 看
    """
    presets = {
        "+x": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        "-x": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        "+y": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "-y": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
        "+z": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        "-z": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
        # 斜视图：相机位于 +X、+Y，并稍微高于 +Z，
        # 朝模型中心看，因此观察方向约为 (-X, -Y, -0.3Z)。
        "xy+": ((-1.0, -1.0, -0.3), (0.0, 0.0, 1.0)),
    }

    return presets[view_name]


def has_point_array(proxy, array_name, time_value=None):
    """判断 PointData 中是否存在指定数组。"""
    if proxy is None:
        return False

    if time_value is None:
        proxy.UpdatePipeline()
    else:
        proxy.UpdatePipeline(time_value)

    point_info = proxy.GetPointDataInformation()
    return point_info.GetArray(array_name) is not None


def find_array_association(proxy, scalar_name, time_value=None):
    """
    判断数组属于点数据还是单元数据。

    返回：
        ("POINTS", array_information)
        ("CELLS", array_information)
    """
    if time_value is None:
        proxy.UpdatePipeline()
    else:
        proxy.UpdatePipeline(time_value)

    point_info = proxy.GetPointDataInformation()
    point_array = point_info.GetArray(scalar_name)

    if point_array is not None:
        return "POINTS", point_array

    cell_info = proxy.GetCellDataInformation()
    cell_array = cell_info.GetArray(scalar_name)

    if cell_array is not None:
        return "CELLS", cell_array

    point_names = [point_info.GetArray(index).GetName() for index in range(point_info.GetNumberOfArrays())]

    cell_names = [cell_info.GetArray(index).GetName() for index in range(cell_info.GetNumberOfArrays())]

    raise RuntimeError(
        "未找到标量数组 {!r}。\n"
        "可用点数组：{}\n"
        "可用单元数组：{}".format(
            scalar_name,
            point_names,
            cell_names,
        )
    )


def resolve_scalar_mode(array_info, requested_mode):
    """
    根据数组分量数决定最终着色模式。

    返回：
        effective_mode: "value" 或 "magnitude"
        number_of_components: 分量数
    """
    number_of_components = array_info.GetNumberOfComponents()

    if requested_mode == "auto":
        effective_mode = "magnitude" if number_of_components > 1 else "value"
    elif requested_mode == "magnitude" and number_of_components > 1:
        effective_mode = "magnitude"
    else:
        # 单分量数组即使请求 magnitude，也退化为 value
        effective_mode = "value"

    return effective_mode, number_of_components


def get_array_range(proxy, association, scalar_name, scalar_mode):
    if association == "POINTS":
        data_info = proxy.GetPointDataInformation()
    else:
        data_info = proxy.GetCellDataInformation()

    array_info = data_info.GetArray(scalar_name)

    if array_info is None:
        return None

    effective_mode, number_of_components = resolve_scalar_mode(
        array_info,
        scalar_mode,
    )

    if number_of_components < 1:
        return None

    if effective_mode == "magnitude" and number_of_components > 1:
        # ParaView/VTK 中 -1 通常表示向量模长
        value_range = array_info.GetComponentRange(-1)
    else:
        # 对标量或 value 模式，取第一个分量
        value_range = array_info.GetComponentRange(0)

    if value_range is None or len(value_range) != 2:
        return None

    value_min = float(value_range[0])
    value_max = float(value_range[1])

    if not math.isfinite(value_min) or not math.isfinite(value_max):
        return None

    return value_min, value_max


def compute_global_scalar_range(
    proxy,
    time_values,
    association,
    scalar_name,
    scalar_mode,
):
    global_min = math.inf
    global_max = -math.inf

    total = len(time_values)

    print("正在扫描全部时间步的 {} 范围……".format(scalar_name))

    for index, time_value in enumerate(time_values):
        proxy.UpdatePipeline(time_value)
        current_range = get_array_range(
            proxy,
            association,
            scalar_name,
            scalar_mode,
        )

        if current_range is not None:
            global_min = min(global_min, current_range[0])
            global_max = max(global_max, current_range[1])

        print(
            "\r  扫描进度：{}/{}".format(index + 1, total),
            end="",
            flush=True,
        )

    print()

    if not math.isfinite(global_min) or not math.isfinite(global_max):
        raise RuntimeError("无法计算标量 {!r} 的全局范围。".format(scalar_name))

    if global_min == global_max:
        # 避免零宽度颜色范围
        delta = abs(global_min) * 0.01
        if delta == 0.0:
            delta = 1.0

        global_min -= delta
        global_max += delta

    return global_min, global_max


def symmetric_color_range(value_range, fallback):
    if value_range is None or not all(math.isfinite(v) for v in value_range):
        raise ValueError("Missing or non-finite frame color range")
    extent = max(abs(v) for v in value_range)
    if extent == 0:
        extent = max(abs(v) for v in fallback)
    if extent <= 0 or not math.isfinite(extent):
        raise ValueError("Invalid zero-field color range fallback")
    return -extent, extent


def get_time_values(number_of_pairs):
    """
    从 ParaView TimeKeeper 获取时间值。

    如果获取不到，则使用 0, 1, 2, ...。
    """
    time_keeper = GetTimeKeeper()

    try:
        values = list(time_keeper.TimestepValues)
    except Exception:
        values = []

    if not values:
        return [float(index) for index in range(number_of_pairs)]

    # 两个输入文件序列经过配对后通常具有相同数量的时间步
    if len(values) < number_of_pairs:
        print("警告：ParaView 返回的时间值数量少于可渲染时间步数量，" "将仅渲染 {} 帧。".format(len(values)))
        return values

    return values[:number_of_pairs]


# ----------------------------------------------------------------------
# FFmpeg
# ----------------------------------------------------------------------


def resolve_ffmpeg(ffmpeg_command):
    """
    查找 FFmpeg。

    参数既可以是：
        ffmpeg
    也可以是：
        C:/ffmpeg/bin/ffmpeg.exe
    """
    if os.path.isfile(ffmpeg_command):
        return os.path.abspath(ffmpeg_command)

    executable = shutil.which(ffmpeg_command)

    if executable:
        return executable

    raise RuntimeError(
        "找不到 FFmpeg：{}\n" "请安装 FFmpeg，或使用 --ffmpeg 指定可执行文件路径。".format(ffmpeg_command)
    )


def make_gif(ffmpeg, frames_dir, output_file, fps):
    """
    使用 palettegen + paletteuse 合成高质量 GIF。
    """
    input_pattern = os.path.join(frames_dir, "frame_%06d.png")

    filter_graph = (
        "split[s0][s1];"
        "[s0]palettegen=stats_mode=diff:max_colors=256[p];"
        "[s1][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle"
    )

    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-framerate",
        str(fps),
        "-start_number",
        "0",
        "-i",
        input_pattern,
        "-vf",
        filter_graph,
        "-loop",
        "0",
        output_file,
    ]

    print("\n正在调用 FFmpeg 生成 GIF：")
    print(" ".join('"{}"'.format(item) if " " in item else item for item in command))

    subprocess.run(command, check=True)


# ----------------------------------------------------------------------
# 主程序
# ----------------------------------------------------------------------


def main():
    args = parse_arguments()

    input_dir = os.path.abspath(args.input_dir)
    output_file = os.path.abspath(args.output)

    if not os.path.isdir(input_dir):
        raise RuntimeError("输入目录不存在：{}".format(input_dir))

    if args.width <= 0 or args.height <= 0:
        raise RuntimeError("图像宽度和高度必须大于 0。")

    if args.fps <= 0:
        raise RuntimeError("FPS 必须大于 0。")

    if args.frame_stride <= 0:
        raise RuntimeError("--frame-stride 必须大于 0。")

    if args.log_color and args.color_range_mode == "frame-symmetric":
        raise RuntimeError("零点对称逐帧色标不能使用 --log-color。")

    if not (0.0 <= args.dem_opacity <= 1.0):
        raise RuntimeError("--dem-opacity 必须位于 0～1。")

    if not (0.0 <= args.mpm_opacity <= 1.0):
        raise RuntimeError("--mpm-opacity 必须位于 0～1。")

    if not (0.0 <= args.fem_opacity <= 1.0):
        raise RuntimeError("--fem-opacity 必须位于 0～1。")

    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------------
    # 查找和配对文件
    # --------------------------------------------------------------

    mpm_files = find_files(
        input_dir,
        args.mpm_glob,
        recursive=args.recursive,
    )

    dem_files = find_files(
        input_dir,
        args.dem_glob,
        recursive=args.recursive,
    )

    fem_files = find_files(
        input_dir,
        args.fem_glob,
        recursive=args.recursive,
    )

    print("找到 MPM 文件：{} 个".format(len(mpm_files)))
    print("找到 DEM 文件：{} 个".format(len(dem_files)))
    print("找到 FEM 文件：{} 个".format(len(fem_files)))

    if not mpm_files:
        print("未找到 MPM 文件，将忽略 MPM 模块。")

    if not dem_files:
        print("未找到 DEM 文件，将忽略 DEM 模块。")

    if not fem_files:
        print("未找到 FEM 文件，将忽略 FEM 模块。")

    steps, aligned_files = align_optional_file_sequences(
        {
            "MPM": mpm_files,
            "DEM": dem_files,
            "FEM": fem_files,
        },
        args.step_regex,
    )

    paired_mpm_files = aligned_files["MPM"]
    paired_dem_files = aligned_files["DEM"]
    paired_fem_files = aligned_files["FEM"]

    print("可渲染时间步：{} 个".format(len(steps)))
    print("第一个时间步：{}".format(steps[0]))
    print("最后一个时间步：{}".format(steps[-1]))

    if paired_mpm_files[0] is not None:
        print("第一个 MPM：{}".format(paired_mpm_files[0]))

    if paired_dem_files[0] is not None:
        print("第一个 DEM：{}".format(paired_dem_files[0]))

    if paired_fem_files[0] is not None:
        print("第一个 FEM：{}".format(paired_fem_files[0]))

    # --------------------------------------------------------------
    # 创建 ParaView 管线
    # --------------------------------------------------------------

    _DisableFirstRenderCameraReset()

    mpm_reader = None
    dem_reader = None
    fem_reader = None
    mpm_display = None
    dem_display = None
    fem_display = None
    dem_glyph = None
    mpm_clip = None
    mpm_output = None
    dem_output = None
    fem_output = None

    if paired_mpm_files and paired_mpm_files[0] is not None:
        print("\n正在读取 MPM 文件序列……")
        mpm_reader = OpenDataFile(paired_mpm_files)

        if mpm_reader is None:
            raise RuntimeError("ParaView 无法读取 MPM 文件序列。")
    else:
        print("\n未提供 MPM 文件，跳过 MPM 管线。")

    if paired_dem_files and paired_dem_files[0] is not None:
        print("正在读取 DEM 文件序列……")
        dem_reader = OpenDataFile(paired_dem_files)

        if dem_reader is None:
            raise RuntimeError("ParaView 无法读取 DEM 文件序列。")

        # IMPORTANT:
        # DEM 永远直接使用原始 reader，不经过任何 Clip/裁剪过滤器。
        # 所有几何裁剪只允许作用于 MPM。
        dem_output = dem_reader
    else:
        print("未提供 DEM 文件，跳过 DEM 管线。")

    if paired_fem_files and paired_fem_files[0] is not None:
        print("正在读取 FEM 文件序列……")
        fem_reader = OpenDataFile(paired_fem_files)

        if fem_reader is None:
            raise RuntimeError("ParaView 无法读取 FEM 文件序列。")

        fem_output = fem_reader
    else:
        print("未提供 FEM 文件，跳过 FEM 管线。")

    animation_scene = GetAnimationScene()

    try:
        animation_scene.UpdateAnimationUsingDataTimeSteps()
    except Exception:
        pass

    time_values = get_time_values(len(steps))

    if not time_values:
        raise RuntimeError("未获得任何 ParaView 时间步。")

    first_time = time_values[0]
    animation_scene.AnimationTime = first_time

    if mpm_reader is not None:
        mpm_reader.UpdatePipeline(first_time)

    if dem_output is not None:
        dem_output.UpdatePipeline(first_time)

    if fem_output is not None:
        fem_output.UpdatePipeline(first_time)

        # DEM 自动 Glyph：
        # 如果 PointData 中存在 radius（或 --dem-radius-array 指定的数组），
        # 则把 DEM 点自动转成 Sphere Glyph。
        if not args.no_dem_glyph and has_point_array(
            dem_output,
            args.dem_radius_array,
            first_time,
        ):
            print("检测到 DEM PointData 数组 {!r}，自动使用 Glyph(Sphere)。".format(args.dem_radius_array))

            dem_glyph = Glyph(
                registrationName="DEM_Radius_Glyph",
                Input=dem_output,
                GlyphType="Sphere",
            )

            # Sphere 的基础半径设为 1；
            # ScaleArray=radius 且 ScaleFactor=1，
            # 因此最终球半径 = radius。
            set_property_if_exists(
                dem_glyph,
                "ScaleArray",
                ["POINTS", args.dem_radius_array],
            )
            set_property_if_exists(
                dem_glyph,
                "ScaleFactor",
                1.0,
            )
            set_property_if_exists(
                dem_glyph,
                "GlyphMode",
                "All Points",
            )

            if hasattr(dem_glyph, "GlyphType"):
                set_property_if_exists(
                    dem_glyph.GlyphType,
                    "Radius",
                    1.0,
                )
                set_property_if_exists(
                    dem_glyph.GlyphType,
                    "ThetaResolution",
                    args.dem_glyph_theta_resolution,
                )
                set_property_if_exists(
                    dem_glyph.GlyphType,
                    "PhiResolution",
                    args.dem_glyph_phi_resolution,
                )

            dem_glyph.UpdatePipeline(first_time)
            dem_render_output = dem_glyph
        else:
            dem_render_output = dem_output

            if not args.no_dem_glyph:
                print("DEM PointData 中未找到 {!r}，保持原始 DEM 显示。".format(args.dem_radius_array))
    else:
        dem_render_output = None

    # --------------------------------------------------------------
    # 获取可见对象包围盒并设置裁剪面
    # --------------------------------------------------------------

    mpm_bounds = mpm_reader.GetDataInformation().GetBounds() if mpm_reader is not None else None
    dem_bounds = dem_render_output.GetDataInformation().GetBounds() if dem_render_output is not None else None
    fem_bounds = fem_output.GetDataInformation().GetBounds() if fem_output is not None else None
    combined_scene_bounds = combine_bounds(
        mpm_bounds,
        dem_bounds,
        fem_bounds,
    )

    if valid_bounds(mpm_bounds):
        automatic_clip_origin = bounds_center(mpm_bounds)
        print(automatic_clip_origin)
    elif valid_bounds(combined_scene_bounds):
        automatic_clip_origin = bounds_center(combined_scene_bounds)
    else:
        automatic_clip_origin = (0.0, 0.0, 0.0)

    clip_origin = tuple(args.clip_origin) if args.clip_origin is not None else automatic_clip_origin

    if mpm_reader is None:
        mpm_output = None
        mpm_clip = None
        print("未提供 MPM 数据，跳过 MPM 裁剪。")
    elif (not args.clip) or args.no_clip:
        # 默认不裁剪。只有显式传入 --clip 才启用。
        mpm_output = mpm_reader
        mpm_clip = None
        print("MPM 裁剪：关闭（默认不裁剪）")
    else:
        mpm_clip = Clip(
            registrationName="MPM_Clip",
            Input=mpm_reader,
        )

        # ParaView 5.11+ 常用设置
        mpm_clip.ClipType = "Plane"
        mpm_clip.ClipType.Origin = list(clip_origin)
        mpm_clip.ClipType.Normal = list(args.clip_normal)

        # 不同 ParaView 版本的属性名称可能不同
        if property_exists(mpm_clip, "Invert"):
            mpm_clip.Invert = int(args.clip_invert)
        elif property_exists(mpm_clip, "InsideOut"):
            mpm_clip.InsideOut = int(args.clip_invert)

        mpm_output = mpm_clip
        mpm_output.UpdatePipeline(first_time)

        print("MPM 裁剪：开启")
        print("裁剪原点：{}".format(clip_origin))
        print("裁剪法向：{}".format(tuple(args.clip_normal)))
        print("反转裁剪：{}".format(args.clip_invert))

    # --------------------------------------------------------------
    # 查找 pressure 数组 / 确定颜色范围（仅 MPM）
    # --------------------------------------------------------------

    association = None
    color_min = None
    color_max = None
    effective_scalar_mode = "value"
    scalar_num_components = 1

    scalar_bar = None
    if mpm_output is not None:
        association, array_info = find_array_association(
            mpm_output,
            args.scalar,
            first_time,
        )

        effective_scalar_mode, scalar_num_components = resolve_scalar_mode(
            array_info,
            args.scalar_mode,
        )

        print(
            "着色数组：{}，数据关联：{}，分量数：{}，着色模式：{}".format(
                args.scalar,
                association,
                scalar_num_components,
                effective_scalar_mode,
            )
        )

        # --------------------------------------------------------------
        # 确定全局颜色范围
        # --------------------------------------------------------------

        if args.color_min is None or args.color_max is None:
            automatic_min, automatic_max = compute_global_scalar_range(
                mpm_output,
                time_values,
                association,
                args.scalar,
                effective_scalar_mode,
            )
        else:
            automatic_min = args.color_min
            automatic_max = args.color_max

        color_min = args.color_min if args.color_min is not None else automatic_min

        color_max = args.color_max if args.color_max is not None else automatic_max

        if color_min >= color_max:
            raise RuntimeError(
                "颜色范围无效：color_min={}，color_max={}".format(
                    color_min,
                    color_max,
                )
            )

        if args.log_color and color_min <= 0:
            raise RuntimeError("使用 --log-color 时，颜色范围最小值必须大于 0，" "当前为 {}。".format(color_min))

        range_title = "初始颜色范围（逐帧更新）" if args.color_range_mode == "frame-symmetric" else "固定颜色范围"
        print("{}：[{:.8g}, {:.8g}]".format(range_title, color_min, color_max))

        # 扫描后回到第一帧
        animation_scene.AnimationTime = first_time

        if mpm_reader is not None:
            mpm_reader.UpdatePipeline(first_time)

        if dem_output is not None:
            dem_output.UpdatePipeline(first_time)

        if fem_output is not None:
            fem_output.UpdatePipeline(first_time)

        mpm_output.UpdatePipeline(first_time)
    else:
        print("未提供 MPM 文件，跳过 pressure 着色与 color bar 设置。")
        animation_scene.AnimationTime = first_time

        if dem_output is not None:
            dem_output.UpdatePipeline(first_time)

        if fem_output is not None:
            fem_output.UpdatePipeline(first_time)

    # --------------------------------------------------------------
    # FEM von_mesis 数组 / 全局颜色范围
    # --------------------------------------------------------------

    fem_association = None
    fem_color_min = None
    fem_color_max = None

    if fem_output is not None:
        fem_association, fem_array_info = find_array_association(
            fem_output,
            args.fem_scalar,
            first_time,
        )

        fem_mode, fem_components = resolve_scalar_mode(
            fem_array_info,
            "value",
        )

        print(
            "FEM 着色数组：{}，数据关联：{}，分量数：{}".format(
                args.fem_scalar,
                fem_association,
                fem_components,
            )
        )

        if args.fem_color_min is None or args.fem_color_max is None:
            fem_auto_min, fem_auto_max = compute_global_scalar_range(
                fem_output,
                time_values,
                fem_association,
                args.fem_scalar,
                "value",
            )
        else:
            fem_auto_min = args.fem_color_min
            fem_auto_max = args.fem_color_max

        fem_color_min = args.fem_color_min if args.fem_color_min is not None else fem_auto_min
        fem_color_max = args.fem_color_max if args.fem_color_max is not None else fem_auto_max

        if fem_color_min >= fem_color_max:
            raise RuntimeError(
                "FEM 颜色范围无效：fem_color_min={}，fem_color_max={}".format(
                    fem_color_min,
                    fem_color_max,
                )
            )

        print(
            "FEM 固定颜色范围：[{:.8g}, {:.8g}]".format(
                fem_color_min,
                fem_color_max,
            )
        )

        animation_scene.AnimationTime = first_time

        if mpm_reader is not None:
            mpm_reader.UpdatePipeline(first_time)
        if dem_output is not None:
            dem_output.UpdatePipeline(first_time)
        fem_output.UpdatePipeline(first_time)

    # --------------------------------------------------------------
    # 创建视图
    # --------------------------------------------------------------

    render_view = GetActiveViewOrCreate("RenderView")
    render_view.ViewSize = [args.width, args.height]
    render_view.Background = list(args.background)

    set_property_if_exists(
        render_view,
        "OrientationAxesVisibility",
        1 if args.show_axes else 0,
    )

    # 交互模式：2D / 3D
    set_property_if_exists(render_view, "InteractionMode", args.interaction_mode)
    print("InteractionMode = {}".format(args.interaction_mode))

    # 抗锯齿相关设置
    set_property_if_exists(render_view, "UseFXAA", 1)
    set_property_if_exists(render_view, "UseColorPaletteForBackground", 0)

    # --------------------------------------------------------------
    # 显示 MPM
    # --------------------------------------------------------------

    if mpm_output is not None:
        mpm_display = Show(mpm_output, render_view)
        mpm_display.Representation = args.mpm_representation
        mpm_display.Opacity = args.mpm_opacity

        if args.mpm_representation == "Points":
            set_property_if_exists(
                mpm_display,
                "PointSize",
                args.mpm_point_size,
            )
            set_property_if_exists(
                mpm_display,
                "RenderPointsAsSpheres",
                1,
            )

        if effective_scalar_mode == "magnitude" and scalar_num_components > 1:
            ColorBy(
                mpm_display,
                (association, args.scalar, "Magnitude"),
            )
        else:
            ColorBy(
                mpm_display,
                (association, args.scalar),
            )

        pressure_lut = GetColorTransferFunction(args.scalar)
        pressure_opacity = GetOpacityTransferFunction(args.scalar)

        try:
            pressure_lut.ApplyPreset(args.color_preset, True)
            print("颜色预设：{}".format(args.color_preset))
        except Exception:
            print("警告：找不到颜色预设 {!r}，改用 Cool to Warm。".format(args.color_preset))
            try:
                pressure_lut.ApplyPreset("Cool to Warm", True)
            except Exception:
                pass

        pressure_lut.RescaleTransferFunction(color_min, color_max)
        pressure_opacity.RescaleTransferFunction(color_min, color_max)

        if property_exists(pressure_lut, "UseLogScale"):
            pressure_lut.UseLogScale = 1 if args.log_color else 0

        # 禁用 ParaView 自动重缩放；逐帧模式由渲染循环显式设置范围。
        set_property_if_exists(
            pressure_lut,
            "AutomaticRescaleRangeMode",
            "Never",
        )

        if args.hide_colorbar:
            mpm_display.SetScalarBarVisibility(render_view, False)
        else:
            mpm_display.SetScalarBarVisibility(render_view, True)

            scalar_bar = GetScalarBar(pressure_lut, render_view)
            scalar_bar.Title = (
                args.colorbar_title
                if args.colorbar_title
                else (
                    "{} magnitude".format(args.scalar)
                    if effective_scalar_mode == "magnitude" and scalar_num_components > 1
                    else args.scalar
                )
            )
            scalar_bar.ComponentTitle = ""

            set_property_if_exists(scalar_bar, "Orientation", "Vertical")
            set_property_if_exists(scalar_bar, "WindowLocation", "Any Location")
            set_property_if_exists(scalar_bar, "Position", [0.86, 0.16])
            set_property_if_exists(scalar_bar, "ScalarBarLength", 0.68)
            set_property_if_exists(scalar_bar, "TitleFontSize", 18)
            set_property_if_exists(scalar_bar, "LabelFontSize", 15)
            # Color bar 字体：Times，文字颜色：黑色
            set_property_if_exists(scalar_bar, "TitleFontFamily", "Times")
            set_property_if_exists(scalar_bar, "LabelFontFamily", "Times")
            set_property_if_exists(scalar_bar, "TitleColor", [0.0, 0.0, 0.0])
            set_property_if_exists(scalar_bar, "LabelColor", [0.0, 0.0, 0.0])
            set_property_if_exists(scalar_bar, "DrawTickMarks", 1)
            set_property_if_exists(scalar_bar, "DrawSubTickMarks", 1)
            if args.color_range_mode == "frame-symmetric":
                # Small late-time pressures otherwise push the title outside the image.
                set_property_if_exists(scalar_bar, "AutomaticLabelFormat", 0)
                set_property_if_exists(scalar_bar, "LabelFormat", "%.2e")
                set_property_if_exists(scalar_bar, "RangeLabelFormat", "%.2e")

            # 强制 color bar 显示颜色范围端点，避免 ParaView 自动刻度
            # 看起来像 color_min/color_max 没有生效。
            set_property_if_exists(scalar_bar, "AddRangeLabels", 1)

            # 如果当前 ParaView 版本支持自定义刻度，则固定为 5 个线性刻度，
            # 包含 color_min 和 color_max。
            custom_labels = [
                color_min,
                color_min + 0.25 * (color_max - color_min),
                color_min + 0.50 * (color_max - color_min),
                color_min + 0.75 * (color_max - color_min),
                color_max,
            ]
            if set_property_if_exists(
                scalar_bar,
                "UseCustomLabels",
                1,
            ):
                set_property_if_exists(
                    scalar_bar,
                    "CustomLabels",
                    custom_labels,
                )

        # 隐藏未经裁剪的 MPM
        if mpm_clip is not None:
            Hide(mpm_reader, render_view)
    else:
        print("未显示 MPM。")

    # --------------------------------------------------------------
    # 显示 DEM —— 始终显示完整 DEM，绝不经过 MPM Clip
    # --------------------------------------------------------------

    if dem_render_output is not None:
        dem_display = Show(dem_render_output, render_view)
        dem_display.Representation = "Surface" if dem_glyph is not None else args.dem_representation
        dem_display.DiffuseColor = list(args.dem_color)
        dem_display.AmbientColor = list(args.dem_color)
        dem_display.Opacity = args.dem_opacity

        print("DEM 裁剪：关闭（始终显示完整 DEM 数据）")

        if dem_glyph is not None:
            print("DEM 显示模式：Glyph(Sphere)，半径数组={!r}".format(args.dem_radius_array))

        # 取消 DEM 标量着色，使用固定颜色
        try:
            ColorBy(dem_display, None)
        except Exception:
            pass

        if dem_glyph is None and args.dem_representation == "Points":
            set_property_if_exists(
                dem_display,
                "PointSize",
                args.dem_point_size,
            )
            set_property_if_exists(
                dem_display,
                "RenderPointsAsSpheres",
                1,
            )

        set_property_if_exists(
            dem_display,
            "RenderLinesAsTubes",
            1,
        )
    else:
        print("未显示 DEM。")

    # --------------------------------------------------------------
    # 显示 FEM —— 默认按 von_mesis 着色
    # --------------------------------------------------------------

    if fem_output is not None:
        fem_display = Show(fem_output, render_view)
        fem_display.Representation = args.fem_representation
        fem_display.Opacity = args.fem_opacity

        if args.fem_representation == "Points":
            set_property_if_exists(
                fem_display,
                "PointSize",
                args.fem_point_size,
            )
            set_property_if_exists(
                fem_display,
                "RenderPointsAsSpheres",
                1,
            )

        ColorBy(
            fem_display,
            (fem_association, args.fem_scalar),
        )

        fem_lut = GetColorTransferFunction(args.fem_scalar)
        fem_opacity_tf = GetOpacityTransferFunction(args.fem_scalar)

        try:
            fem_lut.ApplyPreset(args.fem_color_preset, True)
            print("FEM 颜色预设：{}".format(args.fem_color_preset))
        except Exception:
            print("警告：找不到 FEM 颜色预设 {!r}，改用 Cool to Warm。".format(args.fem_color_preset))
            try:
                fem_lut.ApplyPreset("Cool to Warm", True)
            except Exception:
                pass

        fem_lut.RescaleTransferFunction(
            fem_color_min,
            fem_color_max,
        )
        fem_opacity_tf.RescaleTransferFunction(
            fem_color_min,
            fem_color_max,
        )

        set_property_if_exists(
            fem_lut,
            "AutomaticRescaleRangeMode",
            "Never",
        )

        if args.hide_fem_colorbar:
            fem_display.SetScalarBarVisibility(
                render_view,
                False,
            )
        else:
            fem_display.SetScalarBarVisibility(
                render_view,
                True,
            )

            fem_scalar_bar = GetScalarBar(
                fem_lut,
                render_view,
            )
            fem_scalar_bar.Title = args.fem_scalar
            fem_scalar_bar.ComponentTitle = ""

            set_property_if_exists(
                fem_scalar_bar,
                "Orientation",
                "Vertical",
            )
            set_property_if_exists(
                fem_scalar_bar,
                "WindowLocation",
                "Any Location",
            )
            # 放左边，避免和 MPM colorbar 重叠
            set_property_if_exists(
                fem_scalar_bar,
                "Position",
                [0.04, 0.16],
            )
            set_property_if_exists(
                fem_scalar_bar,
                "ScalarBarLength",
                0.68,
            )
            set_property_if_exists(
                fem_scalar_bar,
                "TitleFontSize",
                18,
            )
            set_property_if_exists(
                fem_scalar_bar,
                "LabelFontSize",
                15,
            )
            set_property_if_exists(
                fem_scalar_bar,
                "TitleFontFamily",
                "Times",
            )
            set_property_if_exists(
                fem_scalar_bar,
                "LabelFontFamily",
                "Times",
            )
            set_property_if_exists(
                fem_scalar_bar,
                "TitleColor",
                [0.0, 0.0, 0.0],
            )
            set_property_if_exists(
                fem_scalar_bar,
                "LabelColor",
                [0.0, 0.0, 0.0],
            )
            set_property_if_exists(
                fem_scalar_bar,
                "DrawTickMarks",
                1,
            )
            set_property_if_exists(
                fem_scalar_bar,
                "DrawSubTickMarks",
                1,
            )
            set_property_if_exists(
                fem_scalar_bar,
                "AddRangeLabels",
                1,
            )

            fem_custom_labels = [
                fem_color_min,
                fem_color_min + 0.25 * (fem_color_max - fem_color_min),
                fem_color_min + 0.50 * (fem_color_max - fem_color_min),
                fem_color_min + 0.75 * (fem_color_max - fem_color_min),
                fem_color_max,
            ]
            if set_property_if_exists(
                fem_scalar_bar,
                "UseCustomLabels",
                1,
            ):
                set_property_if_exists(
                    fem_scalar_bar,
                    "CustomLabels",
                    fem_custom_labels,
                )

        set_property_if_exists(
            fem_display,
            "RenderLinesAsTubes",
            1,
        )

        print("FEM 显示：开启，标量={}".format(args.fem_scalar))
    else:
        print("未显示 FEM。")

    # --------------------------------------------------------------
    # 固定相机    # --------------------------------------------------------------
    # 固定相机
    # --------------------------------------------------------------

    render_view.CameraParallelProjection = 0 if args.perspective else 1

    if args.interaction_mode == "2D" and args.perspective:
        print("警告：InteractionMode=2D 但同时启用了 --perspective；" "已保留 perspective 设置。")

    Render(render_view)

    axis_views = {"+x", "-x", "+y", "-y", "+z", "-z", "xy+"}

    if args.camera_view in axis_views:
        # 预设视图（标准轴向 + xy+ 斜视图）：
        # 使用当前存在的 MPM + DEM + FEM 总体包围盒中心进行居中，
        # 然后用 ResetCamera 自动 fit，尽量减少白边。
        if valid_bounds(combined_scene_bounds):
            scene_center = bounds_center(combined_scene_bounds)

            dx = combined_scene_bounds[1] - combined_scene_bounds[0]
            dy = combined_scene_bounds[3] - combined_scene_bounds[2]
            dz = combined_scene_bounds[5] - combined_scene_bounds[4]

            scene_diag = math.sqrt(dx * dx + dy * dy + dz * dz)
        else:
            scene_center = automatic_clip_origin
            scene_diag = 1.0

        if scene_diag <= 0.0 or not math.isfinite(scene_diag):
            scene_diag = 1.0

        view_direction, view_up = camera_preset(args.camera_view)

        # 定义：
        # CameraFocalPoint - CameraPosition = view_direction
        camera_distance = 2.0 * scene_diag

        render_view.CameraFocalPoint = list(scene_center)

        render_view.CameraPosition = [
            scene_center[0] - camera_distance * view_direction[0],
            scene_center[1] - camera_distance * view_direction[1],
            scene_center[2] - camera_distance * view_direction[2],
        ]

        render_view.CameraViewUp = list(view_up)

        Render(render_view)

        # 自动 fit 到 MPM + DEM 可见范围
        ResetCamera(render_view)

        # 防止 ParaView 在 ResetCamera 后改变 roll
        render_view.CameraViewUp = list(view_up)

        if not args.perspective:
            render_view.CameraParallelScale *= args.camera_padding

        # 如果用户明确给了 parallel-scale，
        # 则手动值优先于自动 fit。
        if args.parallel_scale is not None:
            render_view.CameraParallelScale = args.parallel_scale

        Render(render_view)

        print(
            "相机预设：{}，观察方向={}".format(
                args.camera_view,
                view_direction,
            )
        )

    elif args.camera_position is not None:
        # custom 模式：继续完全支持原来的手动相机参数
        render_view.CameraPosition = list(args.camera_position)

        if args.camera_focal_point is not None:
            render_view.CameraFocalPoint = list(args.camera_focal_point)
        else:
            render_view.CameraFocalPoint = list(automatic_clip_origin)

        render_view.CameraViewUp = list(args.camera_view_up)

        if args.parallel_scale is not None:
            render_view.CameraParallelScale = args.parallel_scale

        Render(render_view)

        print("相机模式：custom")

    else:
        # custom 但未指定 camera-position：
        # 保持原来的自动相机行为
        ResetCamera(render_view)

        render_view.CameraViewUp = list(args.camera_view_up)

        if not args.perspective:
            render_view.CameraParallelScale *= args.camera_padding

        if args.parallel_scale is not None:
            render_view.CameraParallelScale = args.parallel_scale

        Render(render_view)

        print("相机模式：custom/auto-fit")

    print("\n固定相机参数：")
    print("  CameraPosition = {}".format(tuple(render_view.CameraPosition)))
    print("  CameraFocalPoint = {}".format(tuple(render_view.CameraFocalPoint)))
    print("  CameraViewUp = {}".format(tuple(render_view.CameraViewUp)))
    print("  CameraParallelScale = {}".format(render_view.CameraParallelScale))

    # --------------------------------------------------------------
    # 确定输出帧范围
    # --------------------------------------------------------------

    total_time_steps = len(time_values)

    start_index = max(0, args.start_frame)

    if args.end_frame is None:
        end_index = total_time_steps - 1
    else:
        end_index = min(args.end_frame, total_time_steps - 1)

    if start_index > end_index:
        raise RuntimeError(
            "无效帧范围：start-frame={}，end-frame={}".format(
                start_index,
                end_index,
            )
        )

    selected_indices = list(
        range(
            start_index,
            end_index + 1,
            args.frame_stride,
        )
    )

    if not selected_indices:
        raise RuntimeError("没有需要渲染的时间步。")

    print(
        "\n将渲染 {} 帧，时间步索引范围 {}～{}，stride={}。".format(
            len(selected_indices),
            start_index,
            end_index,
            args.frame_stride,
        )
    )

    # --------------------------------------------------------------
    # 准备 PNG 目录
    # --------------------------------------------------------------

    temporary_directory = None

    if args.frames_dir:
        frames_dir = os.path.abspath(args.frames_dir)
        os.makedirs(frames_dir, exist_ok=True)
    elif args.keep_frames or args.skip_gif:
        frames_dir = os.path.splitext(output_file)[0] + "_frames"
        os.makedirs(frames_dir, exist_ok=True)
    else:
        temporary_directory = tempfile.TemporaryDirectory(prefix="paraview_mpm_dem_")
        frames_dir = temporary_directory.name

    # 清理同名旧帧，避免 FFmpeg 读到多余文件
    for old_frame in glob.glob(os.path.join(frames_dir, "frame_*.png")):
        try:
            os.remove(old_frame)
        except OSError:
            pass

    print("PNG 目录：{}".format(frames_dir))

    # --------------------------------------------------------------
    # 渲染 PNG 序列
    # --------------------------------------------------------------

    total_output_frames = len(selected_indices)

    for output_index, time_index in enumerate(selected_indices):
        time_value = time_values[time_index]

        animation_scene.AnimationTime = time_value

        # 显式更新时间序列，避免批处理模式下未刷新
        if mpm_reader is not None:
            mpm_reader.UpdatePipeline(time_value)

        if dem_output is not None:
            dem_output.UpdatePipeline(time_value)

        if fem_output is not None:
            fem_output.UpdatePipeline(time_value)

        if dem_glyph is not None:
            dem_glyph.UpdatePipeline(time_value)

        if mpm_clip is not None:
            mpm_clip.UpdatePipeline(time_value)

        if mpm_output is not None and args.color_range_mode == "frame-symmetric":
            frame_min, frame_max = symmetric_color_range(
                get_array_range(mpm_output, association, args.scalar, effective_scalar_mode),
                (color_min, color_max),
            )
            pressure_lut.RescaleTransferFunction(frame_min, frame_max)
            pressure_opacity.RescaleTransferFunction(frame_min, frame_max)
            if scalar_bar is not None:
                set_property_if_exists(
                    scalar_bar, "CustomLabels", [frame_min, frame_min / 2, 0, frame_max / 2, frame_max]
                )
            print("\n当帧颜色范围：[{:.8g}, {:.8g}]".format(frame_min, frame_max))

        Render(render_view)

        frame_file = os.path.join(
            frames_dir,
            "frame_{:06d}.png".format(output_index),
        )

        SaveScreenshot(
            frame_file,
            render_view,
            ImageResolution=[args.width, args.height],
            TransparentBackground=(1 if args.transparent_background else 0),
        )

        original_step = steps[time_index] if time_index < len(steps) else time_index

        print(
            "\r渲染进度：{}/{}，序列索引={}，文件时间步={}".format(
                output_index + 1,
                total_output_frames,
                time_index,
                original_step,
            ),
            end="",
            flush=True,
        )

    print("\nPNG 序列渲染完成。")

    # --------------------------------------------------------------
    # 合成 GIF
    # --------------------------------------------------------------

    if args.skip_gif:
        print("已跳过 GIF 合成。")
        print("PNG 文件位于：{}".format(frames_dir))
    else:
        ffmpeg = resolve_ffmpeg(args.ffmpeg)
        make_gif(
            ffmpeg,
            frames_dir,
            output_file,
            args.fps,
        )

        print("\nGIF 生成完成：{}".format(output_file))

        if args.keep_frames or args.frames_dir:
            print("PNG 文件保留于：{}".format(frames_dir))

    # 显式清理 ParaView 对象
    if mpm_display is not None:
        try:
            Delete(mpm_display)
        except Exception:
            pass

    if dem_display is not None:
        try:
            Delete(dem_display)
        except Exception:
            pass

    if fem_display is not None:
        try:
            Delete(fem_display)
        except Exception:
            pass

    if dem_glyph is not None:
        try:
            Delete(dem_glyph)
        except Exception:
            pass

    if mpm_clip is not None:
        try:
            Delete(mpm_clip)
        except Exception:
            pass

    if mpm_reader is not None:
        try:
            Delete(mpm_reader)
        except Exception:
            pass

    if dem_reader is not None:
        try:
            Delete(dem_reader)
        except Exception:
            pass

    if fem_reader is not None:
        try:
            Delete(fem_reader)
        except Exception:
            pass

    # TemporaryDirectory 会在 cleanup 后删除
    if temporary_directory is not None:
        temporary_directory.cleanup()


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(
            "\nFFmpeg 执行失败，返回码：{}".format(exc.returncode),
            file=sys.stderr,
        )
        sys.exit(exc.returncode)
    except Exception as exc:
        print("\n运行失败：{}".format(exc), file=sys.stderr)
        sys.exit(1)

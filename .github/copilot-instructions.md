<!-- .github/copilot-instructions.md for GeoTaichi -->
# GeoTaichi — AI coding assistant guide

Purpose: 快速让 AI 编码代理在本仓库中高效、安全地工作，覆盖架构要点、常用命令、项目约定与示例引用。

- **大体架构**: 这是一个以 Taichi 为核心的数值模拟库。核心 Python 包有 `geotaichi/` 和 `src/`，演示/样例在 `examples/` 与 `taichi_demo/`，C++ 辅助实现位于 `src/utils/NURBS/`（由 `setup.py` 中的 `Extension` 列表编译）。

- **关键目录/文件**:
  - `pyproject.toml` — 包元数据与依赖（`taichi==1.6`, `numpy==1.26.4`）。
  - `setup.py` — 构建/编译 C++ 扩展（依赖 `pybind11` 与 Eigen include）。
  - `taichi_demo/` — 演示脚本（常用启动示例，例如 `taichi_demo/demo_mpm1.py`）。
  - `examples/`、`research/` — 研究与示例场景。
  - `assets/` 与 `geotaichi/assets` — 默认资源（网格、纹理等）。

- **构建 / 运行（可被复制的命令）**:
  - 安装 Python 依赖（通用）:

    ```bash
    python -m pip install --upgrade pip
    python -m pip install -e .
    # 或（requirements 脚本示例）
    python3 -m pip install taichi numpy scipy psutil scikit-image open3d rtree pynvml matplotlib pygalmesh meshio trimesh faiss-cpu
    ```

  - 若需要编译 C++ 扩展：确保安装 `pybind11` 和 Eigen（通过 `EIGEN_INCLUDE_DIR` 环境变量或系统路径），然后运行 `python setup.py build` 或 `pip install -e .`。

  - 运行 demo：`python taichi_demo/demo_mpm1.py`（或其它 `taichi_demo` 下脚本）。

- **测试**: 仓库含 `tests/` 目录；常规命令为 `python -m pytest tests`（需要先安装 `pytest`）。

- **项目约定与模式（不要凭空改变）**:
  - 包结构：代码分布在 `geotaichi/`（主包）和 `src/`（兼容包/扩展）；修改时保留这两个路径在 `pyproject.toml` 中的声明。
  - Taichi 版本固定：使用 `taichi==1.6` 进行兼容性开发与测试。
  - C++ 扩展位于 `src/utils/NURBS/` 并在 `setup.py` 中列出源文件；任何新增本地扩展都应在 `setup.py` 中注册并考虑 `pybind11` include。
  - 资源/示例应放在 `assets/` 或相应示例文件夹并通过相对路径引用。

- **集成点与外部依赖**:
  - `pybind11` + Eigen：用于 C++/Python 绑定（见 `setup.py` 的 `get_pybind_include` 和 `EIGEN_INCLUDE_DIR`）。
  - 第三方子模块/库位于 `third_party/`（例如 Pangolin、nanoflann 等），这些通常为可选依赖，仅在特定功能需要时编译或启用。

- **代码样式与工具**:
  - `pyproject.toml` 包含 `black` 配置（`line-length = 120`）。请遵循该格式化长度。

- **示例片段（在修改或生成代码时参考）**:
  - 若要引用主入口，查看 `pyproject.toml` 中定义的 script: `gs = geotaichi._main:main`。
  - 修改 NURBS 扩展时，编辑 `src/utils/NURBS/*.cpp` 并在 `setup.py` 中同步。

- **禁止与注意事项**:
  - 不要默认升级 Taichi 版本或 NumPy 至不兼容版本；若确实需要升级，先在 `taichi_demo/` 中运行全部演示脚本并修复兼容性问题。
  - 修改 `setup.py` 时注意 `load_requirements()` 假定存在 `requirements.txt`。

如果这份指引有遗漏或需要具体示例（例如常用 demo 的启动参数或特定扩展的编译步骤），请告诉我你想优先补充的部分，我会据此迭代更新。

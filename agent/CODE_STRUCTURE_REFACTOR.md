# 代码结构重构记录与约束

本文件集中保存本轮用户建议、实际迁移和修正。继续结构调整前先读本文件及
`CODING_GUIDELINES.md` 第 5 节；每次迁移说明实际归属问题，再更新本记录。
不要把“检查全部代码”理解为机械拆分、统一改名或搬迁全部文件。

## 2026-11-16 Utilities 清理与函数命名补齐

用户要求扫描全仓库、优先检查 Utilities、合并适合共享的实现，并继续统一函数、kernel、
device function 和变量命名。检查覆盖 1,802 个维护中的 Python 文件，包括迁移前 629 个
`src` 文件；第三方、缓存、结果和历史源码快照不参与重写。通过 AST、导入、调用和读写路径
检查小文件、包装函数及重复实现，没有仅凭文件长度或缺少静态引用删除动态入口。

### 归属和共享边界

31 个 utility 类模块中删除 10 个，保留 21 个。具体迁移如下：

| 原位置或重复实现 | 当前归属与理由 |
| --- | --- |
| `dem/structs/Utilities.py` 的材料名称包装 | 显示名称表随 `Material.py`，实际打印方法直接查表；只有一个生产消费者，无需新增参数化包装。严格的本构选择值继续由原验证路径处理 |
| FEDEM/FEMPM contact property 包装 | 复用 `DictIO.get_normalized`；保留别名优先级、大小写和连字符归一化、False/None、默认值及必填错误上下文 |
| IGAMPM contact property 包装 | 精确匹配契约归 `DEMContactModelBase.property_value`，不套用 FEDEM 的归一化规则；有限非负校验复用 `linalg.positive` |
| FEDEM neighbor、FEMPM linked-cell 的线性索引 | 直接调用已有 `ScalarFunction.linearize3d`；使用原标量参数和 cell 数组，保持整数表达式和设备内联边界 |
| `mpm/Utilities.py` | 后处理及 VTU 写出归 `PostPlot.py`；软材料点记录归 `WriteFile.monitor_soft_material_point`，耦合记录器直接调用其实际所有者 |
| `mpdem/engines/Utilities.py` | SoftAffine 表面 VTU 写出归 `SoftAffineIPCEngine.write_soft_affine_surface_vtu`；无生产消费者的能力字典函数删除，保留实际功能验证 |
| `MatrixFunction.Eig_3`、`get_eigenvalue` | 删除纯转发别名；消费者直接使用 `eig3_symmetric`、`eigenvalue` |
| soft-particle 的 `_pack_support`、`_logical_node_id` 和 NURBS 打印函数 | 删除无消费者的遗留实现 |

保留的 Utilities 具有实际职责：DEM 初始化/邻域支持、FEM 边界/cloth/contact/element 共享计算、
MPM 网格/生成/soft-particle/sparse-grid/struct 支持、线性求解控制、NURBS 几何辅助、IPC 几何操作、
SDF 构造和可视化。单函数的 BVH 重建间隔验证仍是多个消费者共享的有效输入边界。
SDF 的维度工具以及其他真实共享数学模块继续保留。

合计删除 21 个源文件，包含上述 10 个 utility 类模块，及以下未连接、重复或未完成的实现：

- `contact_detection/bounding_volume_hierarchy/WarpBVH.py`、`utils/Selector.py`；对应旧原型没有实际入口，
  前者未完成，后者还包含导入即执行的 GUI 演示。
- `nurbs/core/NurbsGeometry.py`、`fem/engines/SparseMatrix.py`、`igampm/contact/ContactModelBase.py`；
  分别是已删除旧投影的常量容器、仅剩文档的文件、未使用的旧占位基类。
- `mesh/StructuredMesh.py`、`mesh/GenerateMesh.py`、`mesh/TetraMesh.py`；未使用的重复或未完成网格原型。
  旧 `mesh/Utilities.py` 的 JAX 原型也无消费者，实际网格输入由现有 FEM generator 负责。
- `igampm/geometry/BoundingBox.py`、`mpm/elements/BoundingBox.py` 和前者目录下的空 `__init__.py`；
  无消费者的重复包围盒实现和仅含文档的包标记。

消费者、VTU 后处理工具、测试和 Mesh README 一并迁移，不保留旧路径转发文件。
没有为了复用合并状态和装配契约不同的求解循环，也没有增加设备侧开关、遍历、启动、分配、复制或同步。
真正为空的源目录检查结果为零；仅剩缓存的旧目录遵循保留缓存的约束。

### 命名覆盖与兼容边界

在上一轮文件、类、普通局部变量迁移基础上，继续统一普通函数、方法、`ti.kernel`、`ti.func` 和
`ti.pyfunc` 的声明、导入、方法引用、反射及嵌入的子进程源码；不通过旧名称别名维持两套接口。
内部 device 参数同步统一，冲突参数使用 `input_wall_id` 等不同名称；历史数组 `state_variables`
与单条 `state_vars` 记录保持不同绑定。公开 Python 关键字参数、构造入口、属性、状态字段、
Taichi struct 字段、模型值、配置和输出/重启键保留原契约。数学量保留可识别记号。
本节取代下文首轮记录中“共享本构方法名和现有所有函数签名不改”的阶段性边界。

例如 `ComputeStress` → `compute_stress`、`GetEssential` → `get_essential`、
`linearize3D` → `linearize3d`、`Solver` 方法 → `solve`，但 `Solver` 类不改。
有实际含义的尾下划线变体使用 `restrict_radius_weighted`、`usf_updating_drained` 等明确名称。
构造工厂 `DEM`/`MPM` 和状态属性 `gridSum` 等属于规范中明确列出的兼容例外。

标准库及第三方符号不随项目同名函数迁移。工具自检发现并恢复了误改的 `ast.Add`/`ast.Sub`；
另检查了导入外部模块的属性变化，未留下同类误改。能力索引只更新相关方法和 reader 元数据，
不顺带刷新无关历史条目；文档中的实现方法名同步更新，模型选择字符串保持原值。

命名检查继续放在既有 `test_source_layout.py` 并纳入 pre-commit，覆盖文件/目录、类、普通局部变量、
所有普通 callable 及 device 参数；新增小型索引回归核对 NumPy 的 x-fast 布局和关键字调用。

本次统一 1,516 个 `src` 函数声明：453 个 kernel、624 个 `ti.func`、32 个 `ti.pyfunc`、
407 个 host 函数；内部参数迁移覆盖 2,124 个源声明及 1 个测试替身声明。
模块变量收尾覆盖 126 个文件的声明或消费者，包括 `THRESHOLD`、`ZEROVEC3F`、`ZEROMAT3X3`、
SANISAND 常量及 THB 的 `element_list`、`node_list`、计数和粒子信息。

### 数值与性能证据

使用 `/opt/anaconda3/envs/geotaichi/bin/python`（Python 3.10.0、Taichi 1.7.0），数值回归设置
`GEOTAICHI_FORCE_CPU=1 GEOTAICHI_REAL_DTYPE=float64`，实际默认 arm64 CPU/f64。
参数和模块变量分别迁移后，对 608 个源文件做 AST 及符号表对照：数值表达式、常量值、默认值、
类型、控制结构、字符串/schema 均未改变；13,088 个词法作用域的符号种类、引用、赋值、闭包和
独立绑定保持一致。模块常量改名只跟随已解析的项目模块导入，不修改同名第三方属性。

| 检查 | 结果与范围 |
| --- | --- |
| 最终 pre-commit | 114 passed、59 deselected；包含文件/类/局部变量/callable/device 参数/模块变量命名检查 |
| 最终核心组 | 105 passed；字典与显示名称、索引/矩阵/四元数/求根、VTU/NPZ、soft particle、THB、shape functions、原生结构精度及 SDF mesh |
| 最终消费路径组 | 38 passed；FEDEM/FEMPM/IGAMPM 实际 surface/contact 路径、SANISAND 与有限应变 runtime kernels |
| device 参数迁移后的扩展组 | 213 passed；DEM 几何/邻域、两相压力、重启及失败回滚、device Krylov、multigrid、NURBS basis、cloth/FEM contact；在最后的常量改名前完成，未计作该阶段的再次执行 |
| 工具回归 | 34 passed；GPU memory estimator 与 suite hygiene，验证标准库 AST 拼写修复 |
| 模型构建工具 | `self_test.py` 的 8 项检查通过 |
| 静态检查 | 1,886 个维护中的 Python 源码（含当前研究脚本）无缓存语法编译通过；参数和模块变量的 AST/作用域检查通过；旧常量的直接导入残留为零 |
| 单元测试收集 | 3,163 项成功收集；不等于全部执行 |
| 差异 | `git diff --check` 通过 |

核心组首次运行的唯一失败是新命名测试漏允许 `_LAZY_EXPORTS` 等私有大写常量；修正测试后
完整核心组和 pre-commit 均通过。没有修改计算代码或放宽数值容差来通过检查。
最终检查中的标准命令为：

```bash
python tests/testing/run_partition.py pre-commit -q --taichi-arch cpu
python agent/geotaichi-model-builder/scripts/self_test.py
python -m pytest tests/unit --collect-only -q --taichi-arch cpu
python -m pytest tests/unit/tools/test_gpu_memory_estimator.py \
  tests/unit/testing/test_suite_hygiene.py -q --taichi-arch cpu
python -m pytest tests/unit/testing/test_source_layout.py \
  tests/unit/utils/test_scalar_functions.py tests/unit/utils/test_object_io.py \
  tests/unit/dem/test_material_console_output.py tests/unit/utils/test_matrix_functions.py \
  tests/unit/mpm/test_pressure_vtk_postprocess.py tests/unit/mpm/test_soft_point_recording.py \
  tests/unit/mpm/test_soft_particle_output.py tests/unit/mpm/test_lsmpm_template_support.py \
  tests/unit/mpdem/test_soft_affine_vtu_output.py \
  tests/unit/mpdem/test_ipc_soft_affine_official_flow.py::test_soft_affine_fully_implicit_capability_is_enabled \
  tests/unit/tools/test_export_twophase_vtu.py tests/unit/tools/test_reprocess_cfdem_vtk.py \
  tests/unit/mpm/test_thb_gallery_support.py tests/unit/utils/test_precision_transform.py \
  tests/unit/mpm/test_native_struct_precision.py tests/unit/mpm/test_shape_functions.py \
  tests/unit/utils/test_root_finding.py tests/unit/utils/test_quaternion.py \
  tests/unit/levelset/test_sdf_mesh.py -q --taichi-arch cpu
python -m pytest tests/unit/fedem/test_levelset_broadphase.py \
  tests/integration/fedem/test_explicit_surface_coupling.py \
  tests/integration/fempm/test_explicit_fempm_surface.py \
  tests/integration/igampm/test_igampm_explicit_dem_contact.py \
  tests/unit/physics_model/materials/test_sanisand_ms.py \
  tests/unit/physics_model/materials/test_finite_strain_runtime_kernels.py -q --taichi-arch cpu
```

pytest 各组使用独立的 `/private/tmp` basetemp；本次没有执行全部测试或重编译整份 LaTeX 手册。

共享线性索引使用 1,048,576 个 i32 坐标，尺寸 `(101, 103, 107)`；旧公式与共享函数的结果均与
NumPy 完全相同。预热 5 次后做 40 组成对交替、同步计时，最终代码的中位数如下：

| 实际后端 | 原公式 | 共享 `linearize3d` | 共享/原公式 |
| --- | --- | --- | --- |
| arm64 CPU | 0.2064 ms | 0.2037 ms | 0.987 |
| Metal GPU | 0.7028 ms | 0.6358 ms | 0.905 |

这是单个索引算子的等价与微基准证据，未观察到该测量范围内的回退，不据此宣称完整求解器加速。
Metal 在沙箱内不能上传 shader，已获授权在沙箱外完成同一临时脚本；没有清理用户 Taichi 缓存。
完整 GPU 求解器、GPU race 及生产轨迹仍未验证。

## 2026-11-15 命名规范与执行

用户单独要求统一命名，并明确以 `src` 内的类、变量和文件为主，同时更新其他位置的消费者。
正式规范位于 [CODING_GUIDELINES.md 第 3 节](CODING_GUIDELINES.md#3-naming-and-formatting)，
根 `AGENTS.md` 已链接到该规范。

本次迁移 62 个文件（其中 44 个属于本构目录更正），统一 25 个类名和 2,325 处局部变量绑定。
变量修改覆盖 149 个 `src` 文件，按实际作用域处理闭包和推导式；13 处命名冲突分别使用
`body_index`、`existing_particle_num`、`neighbor_cell_id`、`point_body_ids`、
`particle_body_id`、`support_node_id`、`element_level` 等不同名称，避免合并不同物理量。

| 原名称 | 当前名称 |
| --- | --- |
| `physics_model/consititutive_model/` | `physics_model/constitutive_model/` |
| `TypeDefination.py`、`BoundaryStrcut.py`、`Geomtry.py` | `TypeDefinition.py`、`BoundaryStruct.py`、`Geometry.py` |
| `BrustNeighbor.py`、`BrustNeighborKernel.py`、`BrustSearch.py` | `BruteNeighbor.py`、`BruteNeighborKernel.py`、`BruteSearch.py` |
| `neighbor_kernel.py`、`warp_bvh.py` | `NeighborKernel.py`、`WarpBVH.py` |
| `_GeometryDeviceState.py`、`_LevelSetBVH.py`、`_LevelSetLinkedCell.py` | 去掉文件和类名的前导下划线 |
| `MultiGridPCG_mixture.py`、`MultiGridPCG_mixture_Axi.py` | `MultiGridPCGMixture.py`、`MultiGridPCGMixtureAxisymmetric.py` |
| `ElementNodesTHB_Generate.py`、`ULSemiImplicitTwoPhaseEngine_u_p.py` | `ElementNodesTHBGenerate.py`、`ULSemiImplicitTwoPhaseEngineUP.py` |
| SDF `utils23D.py`、`utils2D.py`、`utils3D.py` | `utils_2d_3d.py`、`utils_2d.py`、`utils_3d.py` |
| DEM/MPM `myScene`、`set_orientation`、`ti_aabb`、`Fluid_Surface` | `SceneManager`、`Orientation`、`TaichiAABB`、`FluidSurface` |
| 其余带前导下划线的实现类、带 `_rowMajor`/`_Axi` 的类 | 使用无前导下划线的 PascalCase 名称 |
| 普通局部 `bodyID`、`particleNum`、`pointData` 等 | `body_id`、`particle_num`、`point_data` 等 |

同步更新实际导入、延迟导出字符串、测试及子进程源码、维护中的工具/示例、文档链接、能力索引构建器
和已有索引中的源码路径；没有增加旧路径转发文件。下文历史记录保留迁移发生时的原路径。
外部脚本如直接导入这些内部模块/类，需要采用新名称。

数学记号、现有函数签名与关键字参数、共享本构方法名、状态属性、Taichi struct 字段、模型名、配置键、
输出/重启字段，以及 SDF 公开工厂名称保持原契约。该边界是正式规范的一部分，并非宣称所有历史接口
都已改为 snake_case。第三方和实验数据不在本次迁移范围。

按用户随后提出的空目录清理要求，删除旧本构目录及只含过期字节码的子目录，另删除 5 个真正为空的
测试、绘图和资源目录，合计 15 个目录。旧目录的 80 个 `.pyc` 已先备份至
`/private/tmp/geotaichi_naming_obsolete_bytecode.tar.gz`；其他缓存和实验输出目录保留。

命名检查放入既有 `test_source_layout.py`，并纳入 CPU pre-commit 分区。
另同步 5 个被 Git 忽略的本地研究脚本的导入/模块缓存键，以及 3 份本地手册源码的路径；
THB 诊断的 `ElemNodesGen` 直接导入实际所有者 `elements/Utilities.py`。
这些本地文件保持原有 Git 忽略状态，未改历史源码快照、结果文件或既有 PDF。

验证使用 `/opt/anaconda3/envs/geotaichi/bin/python`（Python 3.10.0、Taichi 1.7.0）。
数值回归设置 `GEOTAICHI_FORCE_CPU=1 GEOTAICHI_REAL_DTYPE=float64`，实际为 arm64 CPU/f64。

| 检查 | 结果及范围 |
| --- | --- |
| source layout | 9 passed；文件/目录、类、局部变量、导入及原有结构约束 |
| CPU pre-commit | 111 passed、59 deselected；含新增命名检查 |
| 材料针对性回归 | 52 passed；有限应变 runtime kernels、cloth、流体、线性软化、Hencky 谱精度 |
| 消费路径回归 | 初次 261 passed、1 failed；唯一失败是材料打印测试仍读取 `BaseStruct.py`，原始源码快照同样失败。改为实际 `Material.py` 后该文件 4 passed；未因此改任何生产代码或容差 |
| 模型构建工具 | `self_test.py` 的 8 项检查通过；能力索引的源码路径随迁移更新，未顺带刷新其他历史条目 |
| 静态检查 | 1771 个维护中的 Python 文件无缓存语法编译通过；629 个 src 文件的签名、decorator、字符串/schema 和除名字之外的 AST 不变；局部/闭包/推导式的符号绑定无变化 |
| 本地脚本 | 5 个源码语法编译通过；THB 直接导入及已加载 f64 类型别名的精度锁检查通过 |
| 测试收集 | 3145 项单元测试成功收集；不等于全部执行 |
| 手册 | XeLaTeX 编译成功，564 页；产物仅在 `/private/tmp/geotaichi_naming_manual/` |
| 差异检查 | `git diff --check` 通过 |

主要验证命令（pytest 命令使用独立的临时 `--basetemp`）：

```bash
python -m pytest tests/unit/testing/test_source_layout.py -q --taichi-arch cpu
python tests/testing/run_partition.py pre-commit -q --taichi-arch cpu
python -m pytest tests/unit/physics_model/materials/test_finite_strain_runtime_kernels.py \
  tests/unit/physics_model/materials/test_cloth_models.py \
  tests/unit/physics_model/materials/test_strain_rate_fluids.py \
  tests/unit/physics_model/materials/test_linear_softening.py \
  tests/unit/physics_model/test_hencky_spectral_precision.py -q --taichi-arch cpu
python -m pytest tests/unit/dem/test_geometry_device_state.py \
  tests/unit/fedem/test_levelset_broadphase.py \
  tests/unit/mpm/test_twophase_single_layer_pressure_operator.py \
  tests/unit/linear_solver/test_device_krylov.py tests/unit/levelset/test_sdf_mesh.py \
  tests/unit/visualization/test_realtime_viewer.py tests/unit/contact_detection/test_lbvh.py \
  tests/unit/dem/neighbor/test_lsdem_facet_point_candidates.py \
  tests/unit/iga/geometry/test_nurbs_basis_and_refinement.py \
  tests/unit/mpm/test_constitutive_restart.py tests/unit/mpm/test_two_phase_restart.py \
  tests/unit/linear_solver/test_multigrid_pcg_runtime_stencils.py \
  tests/unit/dem/test_geometry_primitives.py tests/unit/dem/test_material_console_output.py \
  tests/unit/fem/test_cloth_fem.py tests/unit/fem/test_fem_contact.py -q --taichi-arch cpu
python -m pytest tests/unit/dem/test_material_console_output.py -q --taichi-arch cpu
python agent/geotaichi-model-builder/scripts/self_test.py
python -m pytest tests/unit --collect-only -q --taichi-arch cpu
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -outdir=/private/tmp/geotaichi_naming_manual docs/helper/geotaichi_user_theory_manual.tex
```

完整 `commit-materials` 曾启动，因耗时主动中断，已确认进程退出，不计作通过；以上材料组是另行完成的
针对性检查。未运行所有测试、GPU 回归、完整接触法则组或生产轨迹。

## 2026-11-15 复查问题修复

按本次确认的八项问题修复，不改第三方或实验数据。迁移前已核对实际消费者：

- 普通 MPM restart 保留现有恢复 kernel；校验、历史准备完成后再提交，并保护失败回滚。
  同查两相读取路径，覆盖提交阶段的粒子、历史、牵引、缓存和时钟恢复。
  公开 `read_restart` 入口失败时恢复输出序号与继续运行标志，防止读取前的标量修改残留。
- SDF 提取显式识别无等值面区域，其余错误传播；不把内存失败当作空网格。
- ABD 控制点循环改为 device 运行时循环，保持现有静态展开限制及 VJP 方程。
- IGA 上传显式 SciPy 解后，复用已有 device 范数 kernel。
- MPM/IGA 通用数学和字段向量操作直接使用 ScalarFunction、MatrixFunction、ShapeFunctions、
  Quaternion、VectorFunction。删除无调用重复公式和变空的原工具文件，不留 shim。
  原 Direct MPM 的重力旋转使用 f64，迁移不能随 native 默认精度变成 f32。
- 删除不存在实现的 gs 入口；保留实际 MCP/job 入口。
- 等向压实示例拥有自己的构造和参数状态，不调用另一可执行示例的建模函数或改其全局变量。
- IGA-MPM README 移除已删除文件路径并指向真实所有者。

本次实际归属与保留边界：

| 原位置 | 处理及理由 |
| --- | --- |
| `src/mpm/utils.py`、`src/iga/engines/EngineUtils.py` 中的 `linearize`、`vectorize_id` | 直接复用 `ScalarFunction.py` 中已有实现 |
| 同上两文件中的矩阵列、对称张量转换 | 归入 `MatrixFunction.py`；矩形矩阵列按行数构造，新增 2×3、3×2 检查 |
| IGA 参数域 Jacobian | 归入 `ShapeFunctions.py`，公式不变 |
| 字段复制、求和、取负和清零 kernel | 归入 `VectorFunction.py`；整场复制与活动自由度前缀复制契约不同，保留二者 |
| MPM Rodrigues 旋转 | 复用 `Quaternion.py`，显式传入 f64；其余调用保留 native 默认精度 |
| 旧工具文件中的无调用重复公式 | 确认消费者后删除；两个旧文件删除，不保留导入转发 |
| `isotropic_compaction.py` 对其他可执行示例的依赖 | 将实际依赖的 17 个函数及 32 个常量语句放入本示例，合计 840 行原源码；逐函数比较 AST，保留方程和默认值。遵守独立示例规则，未为减少示例重复而新增库抽象 |
| `ConstitutiveBase` 历史恢复 | 所属类分开校验准备与上传，使整个 checkpoint 在写入前完成校验；粒子恢复和事务仍由 `BodyReader` 负责 |

未搬迁求解器、模板或类，未改变模型名、配置键、材料及接触方程或收敛容差。
用户既有原则已在 `CODING_GUIDELINES.md` 中，本次不另建约束框架。

本次检查使用 `/opt/anaconda3/envs/geotaichi/bin/python`（Python 3.10.0 / Taichi 1.7.0），
数值测试显式选择 `ti.cpu`，实际后端为 `arm64`；保留原有精度和维度隔离。
下列组有交叠，不把数量相加当作全仓库覆盖。

| 检查 | 结果与范围 |
| --- | --- |
| 针对修复点的测试 | 81 项通过：SDF 异常传播、普通 restart、f32/f64 Rodrigues、矩形矩阵列、独立示例几何与组分、IGA device 范数及结构规则 |
| IGA 与 restart 消费路径 | 40 项通过：IGA 后端、局部/轴对称装配、两相与不可压流体 restart |
| Direct MPM 消费路径 | 63 项通过：材料用例及编译热点 |
| 摩擦及伴随消费路径 | 8 项通过：ABD 摩擦装配、实际缓存刷新 VJP、耦合轨迹混合摩擦及失败恢复 |
| 最终 restart 与结构回归 | 77 项通过：普通/两相 restart、压力 checkpoint、保存和新增牵引提交失败恢复、结构规则 |
| 公开入口与连续轨迹 | 22 项通过：普通 restart 回归、入口失败后输出序号/标志不变、2D/3D 双层及单层两相连续运行与恢复轨迹一致 |
| 最终 CPU pre-commit | 107 项通过，54 项按分区规则排除 |
| 单元测试收集 | 3139 项成功收集；不等于全部执行 |
| 静态检查 | 1765 个受维护 Python 文件无缓存语法编译通过；直接 `src` 导入无未解析模块或符号；`git diff --check` 通过 |
| 文档与示例入口 | API 方法和配置键覆盖检查通过；手册 XeLaTeX 编译成功，564 页；压实示例 `--help` 成功 |

最后一次 pre-commit 命令：

```bash
python tests/testing/run_partition.py pre-commit -q --taichi-arch cpu
```

直接消费路径的数值检查命令（各组使用独立临时目录）：

```bash
python -m pytest tests/unit/mpm/test_material_models.py \
  tests/integration/mpm/test_direct_mpm_compile_hotspots.py -q --taichi-arch cpu
python -m pytest tests/unit/iga/solver/test_iga_backend.py \
  tests/unit/iga/solver/test_iga_element_local_operators.py \
  tests/unit/iga/solver/test_iga_axisymmetric_local_operators.py \
  tests/unit/mpm/test_two_phase_restart.py \
  tests/verification/mpdem/test_incompressible_restart_pressure.py -q --taichi-arch cpu
python -m pytest tests/unit/dem/contact/test_ipc_affine_friction_assembly.py \
  tests/unit/physics_model/contact/ipc/test_ipc_friction_cache_vjp.py \
  tests/integration/mpdem/test_direct_affine_nonassociated.py::test_coupled_trajectory_mixed_friction_and_failure_restore \
  -q --taichi-arch cpu
python -m pytest tests/unit/mpm/test_constitutive_restart.py \
  tests/unit/mpm/test_two_phase_restart.py \
  tests/verification/mpdem/test_incompressible_restart_pressure.py \
  tests/verification/mpdem/test_two_phase_checkpoint_roundtrip.py \
  tests/unit/testing/test_source_layout.py tests/unit/testing/test_suite_hygiene.py \
  -q --taichi-arch cpu
python -m pytest tests/unit/mpm/test_constitutive_restart.py \
  tests/integration/mpm/test_two_phase_restart_continuation.py -q --taichi-arch cpu
```

同步修订 MPM、IGA、SDF、IGA-MPM 和 FEDEM 示例 README，以及 helper API 的 restart 说明。
材料、接触及控制方程没有变化，已复核其边界而不改写理论章节。手册产物放临时目录，未覆盖既有成果。
未执行全部测试、CUDA/Metal 回归或 125 颗粒压实的完整生产轨迹；这些结果不构成相应验证。

## 用户已明确的原则

1. 大型且独立的实现类应拆分。小型、职责紧密相关的并列类、数据结构、基类及小型变体可以合放。
   不按类数量或固定行数机械拆分，不要求文件名与类名一致。
2. 保留已经清楚的模块归属和文件名。`DEMBase.py` 改成 `Solver.py` 没有实际收益，应恢复。
   MPM 的 `ContactBase`、`MPMContact`、`GeoContact`、`DEMContact` 可以同放 `Contact.py`。
3. 类专属的 Python 配置、预处理等辅助函数放进所属类；不能仅为少量专属函数创建伴生的
   `xxxUtilities.py`。某函数有多个调用文件，并不自动说明它属于通用工具。
4. 同一子系统中真正共享、又不属于单一类的函数，可以放该子系统的 `Utilities.py`。
   `src/utils` 不再增加一层层 `xxxUtilities`，先查找已有分类位置。
5. 通用数学公式应归入 `src/utils` 中现有的对应分类，不因最初在 MPM、DEM、NURBS 或 IPC 中使用
   就留在这些求解器的工具模块里。迁移前检查依赖、精度、边界条件和调用契约，不能只看函数名字。
6. `src/utils/linalg.py` 放通用 host Python 数学函数，主要服务初始化、预处理等边界；
   device 数学函数按功能放入 `MatrixFunction.py`、`VectorFunction.py`、`ScalarFunction.py`、
   `ShapeFunctions.py` 等已有分类。求解器状态、散射装配和迭代流程仍归求解器。
7. `src` 中本构的能量、应力、切线及输出应力计算使用 device 实现；用于验证的 Python/NumPy
   参考公式和有限差分放 `tests`。Python 允许配置、初始化、调度、输出与标量控制。
8. 禁止只做 `from ... import ...` 中转的实现文件。迁移后更新实际调用者并删除废弃文件，
   不保留兼容中转层。具有实际初始化或延迟构造行为的公共入口按其真实职责保留。
9. 每个 model 只有一个正式、区分大小写的名字；输入错误明确报错，不增加大小写折叠、缩写等 model alias。
   参数键与非模型选项的现有约定另行检查，不能借此无差别改动。
10. 有用的自定义异常集中在 `src/utils/errors.py`，按失败含义选类型；标准输入错误使用
    `ValueError`/`TypeError`。删除未使用异常定义前检查 raise、catch、重试、回滚和报告。
    不能把所有失败换成同一种 `RuntimeError`，也不能因改异常类型破坏现有恢复逻辑。
11. 处理一个问题时检查其他受维护代码是否存在同类问题，同步修正真实同类情况；不改第三方、缓存和实验数据。
12. 命名采用 `CODING_GUIDELINES.md` 第 3 节：实现文件和类不加前导下划线，
    普通局部变量使用 snake_case。保留数学记号、现有关键字参数、共享数值方法和状态 schema。
    2026-11-15 的单独命名请求已授权本次迁移；其他结构调整不能顺带继续批量改名。
13. 函数不做过度包装。没有新增校验、状态、生命周期、数值专门化或必要接口边界的纯转发应删掉，
    直接调用实现；不能按“一行函数”或“类很小”机械删除实际接口。
14. 编码与重构应用已安装的 `karpathy-guidelines`，和 Ponytail 一起使用。
    先查真实调用，再用最少改动消除确认的问题，不为执行这些原则另建包装框架。
15. 每个 Python 文件的位置与实际功能一致；模板准备归 generator，字段序列化归 FieldIO，
    字典访问归 DictIO，通用 host 数学归 linalg。先检查能否复用现有实现；输入所有权、坐标系、
    几何表示或校验契约不同的模板不能为了减少文件而强行合并。

## 迁移前的检查

- 当前位置究竟有什么问题？保留现有文件是否已经满足上述原则？
- 函数属于某个类、子系统共享逻辑，还是无求解器依赖的数学公式？先查已有实现。
- 查清读者、写者、配置、测试、示例、文档、源码字符串和 monkeypatch 的实际目标。
- 保持 host/device 边界、导入时的维度和精度设置、数值公式、容差及失败检查。
- 不同时引入无关改名。迁移后不留转发文件，不创建只装一个专属小函数的工具模块。
- 更新记录，运行与所改路径有关的检查；测试通过后停止扩大改动。

## 已做的工作与修正

| 范围 | 实际操作 | 当前判断 |
| --- | --- | --- |
| Cloth 本构 | 将原 `finite_strain/Cloth.py` 的实现在 `cloth_model` 下按 ARAP、NeoHookean、bending 分开；删除原中转文件 | 符合用户要求；原实现叫 ARAP，不能无依据改成 ASAP |
| Cloth 验证 | Python 参考计算移到 `tests/helpers/cloth_reference.py`，生产输出应力走 device | 保持验证与生产边界 |
| 有限应变本构 | NeoHookean、StVK 的 Python 参考公式放到 `tests/helpers/finite_strain_reference.py`；FEM 输出应力走 device | 需要材料、装配和输出回归 |
| model 名字 | 对受维护的本构与接触模型选择器移除 model 名称 alias，更新调用者 | 需核对所有公开文档与无效输入测试 |
| 异常 | 集中已有异常及实际使用的失败类型，更新 raise/catch 的引用 | 需保持原继承关系和恢复语义 |
| 多类大文件 | 对 DEM/MPM 结构、NURBS、SDF、装配器、边界及显示模块做过拆分 | 早期按行数扫描过度，不能作为继续拆分的依据 |
| MPM Contact | 曾将几十行的 `GeoContact`、`DEMContact` 拆走 | 已合回 `Contact.py`，更新调用者，删除两个新文件 |
| DEMBase | 曾把 `Solver`、小型 `AffineBodySolver` 搬到新文件 | 恢复 `DEMBase.py`，不保留新中转文件 |
| 小型并列类 | 复核后合回部分 DEM 基础记录、MPM 网格/软粒子结构和可视化数据源/选项 | 避免把小型相关类机械拆散 |
| 专属 Utilities | 曾为多个类生成 `xxxUtilities.py` | 专属 Python 函数归类，真实共享函数合入子系统 `Utilities.py` |
| DeviceKrylov | 伴生工具文件只有一个被实际调用的执行策略检查 | 检查作为 Python 静态方法归回 `DeviceKrylov`，删除伴生文件 |
| 通用数学 | 从 MPM、DEM、NURBS、SDF、cloth、线性求解器、IPC 中归集通用 host/device 公式 | host 放 `linalg.py`，device 按向量、矩阵、标量分类；保留领域流程 |
| 重复 B 样条 | 软粒子准备阶段存在重复的 weight-only 与 weight/gradient 多项式 | 使用同一 host 实现，检查值与导数契约 |
| 根求解公式 | `linalg.py` 混有三个 `@ti.func` 根/复数运算函数 | 移到 `ScalarFunction.py`，使 `linalg.py` 无 device 函数 |
| `src/mpm/config.py` | 原有 `DIM`、`DYNAMIC` 被 Direct MPM/软粒子场及内核使用 | 保留，补充导入敏感说明；不能当作空中转文件删除 |

## 本轮发现的迁移风险

- 将 `@ti.func` 机械改为 `@staticmethod` 会改变 Taichi 参数专门化，已导致块 Jacobi 数值错误。
  device helper 应为模块函数或含 `self` 的实例方法；Python 静态方法规则不能直接套用。
- 迁移不能改变常量定义顺序，也不能把 Python `import` 塞进 Taichi device 函数解决循环依赖。
- `from __future__ import annotations` 会影响 Taichi 参数标注，不能随工具文件合并引入。
- AST 导入替换不会自动修复子进程中的源码字符串、源码路径断言或 monkeypatch；这些要查实际消费者。
- 辅助函数变为类方法后，函数体内对同组函数的调用也必须按新归属限定；例如
  `Timer.has_remaining_time()` 内调用 `Timer.time_tolerance()`。只改外部 import 不够。
- Taichi 测试运行期间不修改源代码，避免 JIT 从磁盘读取与已加载函数不一致的代码。

## 验证状态

测试环境为 GeoTaichi 依赖环境中的 Python 3.10 / Taichi 1.7.0，实际 CPU 后端 `arm64`。
测试中的显式精度隔离保持不变；没有 CUDA/Metal 验证。以下用例存在交叠，不相加为总覆盖数。

| 检查 | 结果与范围 |
| --- | --- |
| 单元测试收集 | 3068 项成功收集 |
| 原失败节点和基础回归 | 407 项通过；包括原先 24 个失败节点、utils/testing/MCP/Blender、DEM 回调、物理时钟、MPM Contact 和显示模块 |
| 数学、几何、装配与线性求解 | 298 项通过；2 项 NURBS 测试因 monkeypatch 仍指向旧模块失败，随后修正实际目标 |
| 最后修复回归 | 61 项通过；包含上述 2 项 NURBS、结构规则、Timer、公共信息输出和显示模块 |
| 工具与示例消费方 | 120 项通过，另有 5 项 subtest 通过 |
| 最终 CPU pre-commit | 107 项通过，54 项按分区规则排除 |
| 静态归属与导入检查 | 632 个 src Python 文件可解析；受维护代码的直接 src 导入无未解析目标；未发现纯转发实现文件、host 本构 evaluator 或裸 RuntimeError |
| Python 语法编译 | 工作区中 944 个变更的 Python 文件编译通过，不写入缓存 |
| 差异检查 | `git diff --check` 通过 |
| 文档 | 手册重新编译成功；API 方法覆盖仍有此前已有的 20 个缺口，未将该检查报告为通过 |

早期结构快照的材料组另有 378 项通过、MPM/DEM/耦合组 925 项通过，不能视为最后一次迁移后的
全量回归。当前工作未覆盖全部大规模耦合、积分测试或 GPU 并行行为。

最终提交门禁命令：

```bash
python tests/testing/run_partition.py pre-commit -q --taichi-arch cpu
```

最后修复回归命令：

```bash
python -m pytest \
  tests/unit/physics_model/contact/ipc/test_ipc_nurbs_contact_common.py::test_surface_projection_attempts_tensor_hint_seeds \
  tests/unit/physics_model/contact/ipc/test_ipc_nurbs_contact_common.py::test_surface_projection_uses_field_parameter_seed \
  tests/unit/testing/test_source_layout.py \
  tests/unit/utils/test_time_ticker.py \
  tests/unit/utils/test_solver_console.py \
  tests/unit/visualization/test_realtime_viewer.py \
  -q --taichi-arch cpu --maxfail=2 --tb=short
```

删除了导入改写遗留的 35 个多余 `pass`；对每个修改文件比较去除冗余 `pass` 后的 AST，
确认没有其他语法树节点变化。此后没有继续搬迁实现归属。

## 最终类归属清单

以下比较本轮全库检查前的工作区快照与当前实现，列出实际改变文件归属的类。
已合回原文件的类不计作迁移；上表另记其拆分和回退过程。清单用于追踪，不是进一步搬迁的任务列表。

| 原实现文件 | 类及当前实现文件 |
| --- | --- |
| `src/dem/engines/AffineDiffIPC.py` | `TaichiAffineDiffIPCProjector` → `src/dem/engines/TaichiAffineDiffIPCProjector.py`；`TaichiAffineMeshDiffIPCProjector` → `src/dem/engines/TaichiAffineMeshDiffIPCProjector.py` |
| `src/dem/generator/BodyGenerator.py` | `ParticleCreator` → `src/dem/generator/ParticleCreator.py`；`ParticleGenerator` → `src/dem/generator/ParticleGenerator.py` |
| `src/dem/neighbor/AffineBodyNeighbor.py` | `AffineBodyNeighborBase` → `src/dem/neighbor/AffineBodyNeighborBase.py`；`AffineBodyLinkedCell` → `src/dem/neighbor/AffineBodyLinkedCell.py`；`AffineBodyHierarchicalLinkedCell` → `src/dem/neighbor/AffineBodyHierarchicalLinkedCell.py`；`AffineBodyBoundingVolumeHierarchy` → `src/dem/neighbor/AffineBodyBoundingVolumeHierarchy.py` |
| `src/dem/structs/BaseStruct.py` | `Material` → `src/dem/structs/Material.py`；`ParticleFamily` → `src/dem/structs/ParticleFamily.py`；`ClumpFamily` → `src/dem/structs/ClumpFamily.py`；`PlaneFamily` → `src/dem/structs/PlaneFamily.py`；`FacetFamily` → `src/dem/structs/FacetFamily.py`；`ServoWall` → `src/dem/structs/ServoWall.py`；`PatchFamily` → `src/dem/structs/PatchFamily.py`；`RigidBody` → `src/dem/structs/RigidBody.py`；`ImplicitSurfaceParticle` → `src/dem/structs/ImplicitSurfaceParticle.py`；`PolySuperEllipsoid` → `src/dem/structs/PolySuperEllipsoid.py`；`PolySuperQuadrics` → `src/dem/structs/PolySuperQuadrics.py`；`DeformableBoundingSphere` → `src/dem/structs/DeformableBoundingSphere.py`；`DeformableQuadraticBoundingSphere` → `src/dem/structs/DeformableQuadraticBoundingSphere.py`；`BoundingBox` → `src/dem/structs/BoundingBox.py` |
| `src/dem/structs/Geometry.py` | `_GeometryDeviceState` → `src/dem/structs/_GeometryDeviceState.py` |
| `src/fedem/neighbor/LevelSet.py` | `_LevelSetLinkedCell` → `src/fedem/neighbor/_LevelSetLinkedCell.py`；`_LevelSetBVH` → `src/fedem/neighbor/_LevelSetBVH.py`；`FEMLevelSetBroadPhase` → `src/fedem/neighbor/FEMLevelSetBroadPhase.py` |
| `src/fem/boundaries/BoundaryCondition.py` | `DirichletBoundary` → `src/fem/boundaries/DirichletBoundary.py`；`NeumannBoundary` → `src/fem/boundaries/NeumannBoundary.py` |
| `src/fem/contact/ContactAssembler.py` | `FEMContactAssembler` → `src/fem/contact/FEMContactAssembler.py`；`FEMMultiContactAssembler` → `src/fem/contact/FEMMultiContactAssembler.py` |
| `src/fem/engines/SparseMatrix.py` | `FEMTripletContribution` → `src/fem/engines/FEMTripletContribution.py`；`FEMSparseMatrix` → `src/fem/engines/FEMSparseMatrix.py` |
| `src/igampm/contact/DEMContact.py` | `DEMContactModelBase` → `src/igampm/contact/DEMContactModelBase.py`；`LinearDEMContactModel` → `src/igampm/contact/LinearDEMContactModel.py`；`HertzMindlinDEMContactModel` → `src/igampm/contact/HertzMindlinDEMContactModel.py` |
| `src/levelset/SemiLagrangian.py` | `SemiLagrangianMultiGrid` → `src/levelset/SemiLagrangianMultiGrid.py` |
| `src/mesh/GaussPoint.py` | `GaussPointInTriangle` → `src/mesh/GaussPointInTriangle.py`；`GaussPointInRectangle` → `src/mesh/GaussPointInRectangle.py` |
| `src/mpdem/fluid_dynamics/IncompressibleCoupling.py` | `IncompressibleLSDEMCoupling` → `src/mpdem/fluid_dynamics/IncompressibleLSDEMCoupling.py`；`TwoPhaseDoubleLayerLSDEMCoupling` → `src/mpdem/fluid_dynamics/TwoPhaseDoubleLayerLSDEMCoupling.py`；`IncompressibleAffineBodyCoupling` → `src/mpdem/fluid_dynamics/IncompressibleAffineBodyCoupling.py` |
| `src/mpdem/generator/LSBodyGenerate.py` | `ParticleCreator` → `src/mpdem/generator/ParticleCreator.py`；`ParticleGenerator` → `src/mpdem/generator/ParticleGenerator.py` |
| `src/mpm/MaterialManager.py` | `SoftParticleNeoHookeanMaterialTable` → `src/mpm/SoftParticleNeoHookeanMaterialTable.py`；`MaterialHandle` → `src/mpm/MaterialHandle.py`；`SoftParticleMaterialManager` → `src/mpm/SoftParticleMaterialManager.py` |
| `src/mpm/engines/Operator.py` | `MomentBalanceDynamicOperator` → `src/mpm/engines/MomentBalanceDynamicOperator.py`；`PoissonEquationOperator` → `src/mpm/engines/PoissonEquationOperator.py` |
| `src/mpm/generator/BodyGenerator.py` | `SoftParticleCreatorMixin` → `src/mpm/generator/SoftParticleCreatorMixin.py`；`SoftParticleGeneratorMixin` → `src/mpm/generator/SoftParticleGeneratorMixin.py` |
| `src/mpm/soft_particle/Structs.py` | `SoftBody` → `src/mpm/soft_particle/SoftBody.py` |
| `src/mpm/structs/GridNode.py` | `ContactNodes` → `src/mpm/structs/ContactNodes.py`；`ContactNodes2D` → `src/mpm/structs/ContactNodes2D.py`；`NodeTwoPhase` → `src/mpm/structs/NodeTwoPhase.py`；`NodeTwoPhase2D` → `src/mpm/structs/NodeTwoPhase2D.py`；`Nodes` → `src/mpm/structs/Nodes.py`；`Nodes2D` → `src/mpm/structs/Nodes2D.py` |
| `src/mpm/structs/Particle.py` | `ParticleCloud2D` → `src/mpm/structs/ParticleCloud2D.py`；`ParticleCloudIncompressible2D` → `src/mpm/structs/ParticleCloudIncompressible2D.py`；`ParticleCloudTwoPhase2D` → `src/mpm/structs/ParticleCloudTwoPhase2D.py`；`ParticleCloudTwoPhase` → `src/mpm/structs/ParticleCloudTwoPhase.py`；`ParticleCloudTwoPhase2DAxisy` → `src/mpm/structs/ParticleCloudTwoPhase2DAxisy.py`；`ParticleCloud2DAxisy` → `src/mpm/structs/ParticleCloud2DAxisy.py`；`ParticleCloud` → `src/mpm/structs/ParticleCloud.py`；`ParticleCoupling` → `src/mpm/structs/ParticleCoupling.py`；`ImplicitParticleCoupling` → `src/mpm/structs/ImplicitParticleCoupling.py`；`ImplicitParticle` → `src/mpm/structs/ImplicitParticle.py`；`ImplicitParticle2D` → `src/mpm/structs/ImplicitParticle2D.py`；`ParticleCloudIncompressible3D` → `src/mpm/structs/ParticleCloudIncompressible3D.py` |
| `src/nurbs/BSplinePrimitives.py` | `BSplineCurve` → `src/nurbs/BSplineCurve.py`；`BSplineSurface` → `src/nurbs/BSplineSurface.py`；`BSplineVolume` → `src/nurbs/BSplineVolume.py` |
| `src/nurbs/BasicSurface.py` | `PrimitiveSurface` → `src/nurbs/PrimitiveSurface.py`；`Rectangle` → `src/nurbs/Rectangle.py`；`Ring` → `src/nurbs/Ring.py` |
| `src/nurbs/BasicVolume.py` | `PrimitiveVolume` → `src/nurbs/PrimitiveVolume.py`；`Cube` → `src/nurbs/Cube.py`；`Tube` → `src/nurbs/Tube.py`；`Cylinder` → `src/nurbs/Cylinder.py` |
| `src/nurbs/Energy.py` | `ASAP` → `src/nurbs/ASAP.py`；`MIPS` → `src/nurbs/MIPS.py` |
| `src/nurbs/Fitting.py` | `CurvePoints` → `src/nurbs/CurvePoints.py`；`SurfacePoints` → `src/nurbs/SurfacePoints.py` |
| `src/nurbs/NurbsPrimitives.py` | `NurbsCurve` → `src/nurbs/NurbsCurve.py`；`NurbsSurface` → `src/nurbs/NurbsSurface.py`；`NurbsVolume` → `src/nurbs/NurbsVolume.py` |
| `src/nurbs/SplinePrimitives.py` | `Spline` → `src/nurbs/Spline.py`；`SplineCurve` → `src/nurbs/SplineCurve.py`；`SplineSurface` → `src/nurbs/SplineSurface.py`；`SplineVolume` → `src/nurbs/SplineVolume.py` |
| `src/nurbs/core/NurbsGeometry.py` | `NurbsBasisFunction` → `src/nurbs/core/NurbsBasisFunction.py`；`NurbsBasisFunction1d` → `src/nurbs/core/NurbsBasisFunction1d.py`；`NurbsBasisFunction2d` → `src/nurbs/core/NurbsBasisFunction2d.py`；`NurbsBasisFunction3d` → `src/nurbs/core/NurbsBasisFunction3d.py` |
| `src/physics_model/consititutive_model/MaterialModel.py` | `Solid` → `src/physics_model/consititutive_model/Solid.py`；`Fluid` → `src/physics_model/consititutive_model/Fluid.py` |
| `src/physics_model/consititutive_model/finite_strain/FiniteStrainModel.py` | `PlasticModel` → `src/physics_model/consititutive_model/finite_strain/PlasticModel.py` |
| `src/physics_model/consititutive_model/infinitesimal_strain/ElasPlasticity.py` | `ElasticMaterial` → `src/physics_model/consititutive_model/infinitesimal_strain/ElasticMaterial.py`；`PlasticMaterial` → `src/physics_model/consititutive_model/infinitesimal_strain/PlasticMaterial.py` |
| `src/physics_model/contact_model/ipc/IPC.py` | `Barrier` → `src/physics_model/contact_model/ipc/Barrier.py`；`Friction` → `src/physics_model/contact_model/ipc/Friction.py` |
| `src/sdf/BasicShape.py` | `pointcloud` → `src/sdf/pointcloud.py`；`MeshBased` → `src/sdf/MeshBased.py`；`arbitrarily` → `src/sdf/arbitrarily.py`；`polyhedron` → `src/sdf/polyhedron.py`；`Surface` → `src/sdf/Surface.py`；`surfacefunction` → `src/sdf/surfacefunction.py`；`SuperSurface` → `src/sdf/SuperSurface.py`；`polysuperellipsoid` → `src/sdf/polysuperellipsoid.py`；`polysuperquadrics` → `src/sdf/polysuperquadrics.py`；`SDF` → `src/sdf/SDF.py` |
| `src/sdf/SDFs.py` | `SDF2D` → `src/sdf/SDF2D.py`；`SDF3D` → `src/sdf/SDF3D.py` |
| `src/visualization/realtime.py` | `RealtimeViewer` → `src/visualization/RealtimeViewer.py` |

## 反过度封装检查

本轮按用户要求检查维护中的库、公开入口、示例、工具、Blender 和两套 agent 集成，共扫描
1334 个 Python 文件。第三方、缓存和生成结果不在修改范围。短函数和重复代码只是线索，
不能直接等同于过度封装；Taichi 接口、不同物理假设、配置和异常契约须逐个核对。

### Skill 与使用依据

已通过 Codex 的 skill installer 安装并完整读取
[karpathy-guidelines](https://github.com/multica-ai/andrej-karpathy-skills/tree/main/skills/karpathy-guidelines)，
位置为 Codex skills 目录中的 `karpathy-guidelines/SKILL.md`。后续回合可自动发现，本轮直接按已读内容执行。
2026-11-14 查询 GitHub 原仓库元数据：Karpathy Guidelines 232900 stars，Ponytail 190399 stars。
这是所比较候选的仓库流行度依据，不是经过验证的全网 skill 使用量排行榜。没有向求解器添加运行时依赖。

### 两个被询问模块的实际职责

- `SolverDiagnostics.py` 为八类公共求解器和 MCP 提供无 Taichi/NumPy 依赖的进度、失败和最近一步快照，
  支持后端尚未初始化的情况；专用引擎已有诊断时直接委托它。这个真实共享契约保留。
- `SolverRuntime.py` 中 `StepSchedule` 按已接受步数决定诊断、Jacobian 检查和历史记录的频率，
  并限制历史容量。多个 FEM/IGA/Direct MPM/耦合求解器实际使用，保留调度数据及校验。
  回调和非线性线性化控制不属于这个调度记录，不能仅为了归入 class 就挂成它的静态方法。

### 已确认的简化

| 位置 | 原问题 | 最小修改 |
| --- | --- | --- |
| `SolverDiagnosticsMixin` | 公共诊断方法再转发到一个静态入口；最近历史读取只有一次使用 | 直接实现公共方法，内联单次历史读取；保留标量转换和共享状态读取 |
| 原生求解器诊断 | `time` 可能是目标时间，优先于真正进度 `current_time` | 优先读取 `current_time`，目标时间单独保留；新增回归验证旧实现输出 4.0、新实现输出 0.25 的案例 |
| `StepSchedule` | 用一个调度 class 收纳五个无关共享操作，另有 `_due` 单行中间层 | 回调归 `RuntimeHook.py`，求解器标量控制归 `linear_solver/Utilities.py`；三个频率方法直接判断取模 |
| `linear_solver/Utilities.py` | 有遗留矩阵函数导入；控制函数不应提早加载数值后端 | 删除未使用导入；Taichi/NumPy/SciPy 只在明确选择 host sparse solve 时导入 |
| `LinearOperator` | 保存 callable 后又经 `matvec_go` 转发 | 直接把 callable 绑定到 `matvec`；保持 Krylov 需要的对象协议 |
| `BuildTriplet` | 两个 group-Jacobi 方法只原样转发到现有 preconditioner | 调用已有 `build_triplet` / `apply` 实现，更新测试消费方 |
| `DEMPMBridge` | 18 个函数只原样调用 MPM 实现或 kernel | 后端接口名称直接绑定已有实现；保留真实的安装、状态准备、邻域和接触逻辑 |
| DEM / soft-affine IPC operator | 几何、barrier、friction 和 Hessian 散射被重复转发 | 直接调用原共享 device 函数/散射方法；方程、参数、精度及执行设备不变 |
| MPM | 无用途的两个输出别名函数；材料名验证和位移恢复的重复转发 | 删除无调用定义，消费方直接使用已有验证/恢复方法 |
| `ElementTest`、`GranularMaterial`、DEMWall assembler | override 只有一次同参 `super()` 调用 | 使用继承的方法，不另加一层转发 |
| `ObjectIO.py` | JSON 包装 class 没有调用者，和广泛使用的 `DictIO` 并存 | 删除未使用的 class 及其导入，保留 `DictIO` 和原文件名 |
| `ListWrapper` | 为实际材料 slots 提供了许多未使用的 list 便利方法 | 保留 Taichi 模板所需身份、索引、size、shape 和追加；删除无调用便利方法，新增真实 kernel 模板测试 |
| `MatrixFreeBICG.py` | 无调用兼容子类，将 BiCGSTAB 另起 BiCG 名称 | 删除这个空兼容文件，同步线性求解器 README |

上述操作没有重命名现有实现文件，也没有新建源代码包装类。初次处理涉及 62 个代码/测试文件，
移除或直接绑定 46 处包装/声明；五个共享操作保留原公式和校验，仅去掉错误的 class 归属。

### 检查后保留的短接口

- `ListWrapper` 不能直接换成普通 `list`：当前 Taichi 模板参数需要弱引用身份，普通 list 不满足。
- 公开求解器 facade、DEM 到 MPM 的初始化边界、MCP JSON/日志和 Blender 回调契约有实际消费者，
  不因其方法短就删除。桥接仍负责实际初始化与状态准备，没有改成纯中转文件。
- profiler 的直接调用适配器会去掉 stage 名称以匹配正常执行签名；运行时绑定可避免热循环重复分派。
- 物理材料/节点统一接口、Taichi 模板专门化、继承钩子和测试中用于比较的参考实现不机械合并。
- 容差、非有限值、capacity、feasibility、rollback 与失败重试不是冗余校验。

### 验证进度

3070 项单元测试收集成功。针对诊断、调度、回调、Taichi 材料 slots、耦合运行时绑定及结构约束的
56 项测试通过；第一轮一项旧 monkeypatch 目标的问题已通过使用正式 Python callback 接口修正。
后续 263 项运行时/MCP/输出/原生材料回归、201 项线性求解与 IGA/IGA--MPM 回归、161 项
摩擦装配/fully implicit/soft-affine/LSMPM 回归均通过。各组有交叠，不相加为独立覆盖数。

## 按 CODING_GUIDELINES 全仓库复查与功能归属

复查范围为 `src`、`geotaichi`、维护中的 examples/tools、Blender、MCP 和 model-builder Python。
本轮静态扫描覆盖 1333 个文件，结合调用者、最近测试和 owning README 决定实际修改。
第三方、缓存、生成结果、实验归档及保存于 OutputData 的旧源码路径不参与搬迁。
静态结果不等同于逐行数值证明，也不把所有短函数当成待删除项。

### 实际归属与简化

| 原位置或问题 | 当前处理 | 判断依据 |
| --- | --- | --- |
| `dem/affine/AffineBodyTemplate.py` | 实现移到 `dem/generator/AffineBodyTemplate.py`；更新三个生成消费方和测试，删除旧空目录入口，不留 shim | 实际职责是从表面构造模板，非 affine 求解或状态 |
| 是否直接改用 `GeneralShapeTemplate` | 保留两个模板，复用同一通用三角形节点面积求积 | affine 复制表面并保留 template/grid 变换，不分配 SDF；GeneralShapeTemplate 依赖/准备 level-set 网格。强并会新增开关、额外分配或改变输入所有权与坐标契约 |
| `ipc/ContactMeasure.py` 的七个 host 几何函数 | 归到 `utils/linalg.py`，调用者直接导入；GeneralShapeTemplate 使用同一 `lumped_vertex_measures` | 面积、长度、边和节点权重本身不依赖 IPC；IPC 的 fallback measure 和 device policy 仍留在原所有者 |
| `linalg.get_dataclass_to_dict` | 移到现有 `utils/FieldIO.py`，更新消费方 | 读取 Taichi 字段和组织输出属于字段 I/O，不能让纯 Python 数学模块依赖 Taichi |
| `linalg.read_dict_list` | 归入现有 `DictIO`，更新三个消费方 | 实际是字典/列表分派，不是数学公式 |
| linalg 复制的 namedtuple 和 Root 中专用 getter | 使用标准库 `collections.namedtuple`，删除复制工厂及不再使用的 getter | 保持 Newton full-output 数据契约，已有标准实现足够 |
| 11 个只有同参 `super().__init__` 的构造器 | 删除无行为 override，使用继承初始化 | 涉及 NURBS、BrustSearch、两个相邻双相引擎、ULExplicitTwoPhase、PlasticModel、SDF 三角形；真正改变默认配置的构造器保留 |
| 无行为 `ElasticModel` 中间类 | 删除，由 PlasticModel 直接继承真实 `FiniteStrainModel` | 原类仅被 PlasticModel 当作空父类；构造器清理曾留下空 class，语法检查发现后修正，没有补 `pass` 或兼容包装 |
| GeneralShapeTemplate 的空 clear/finalize/read 方法 | 删除声明和无操作调用；无效 VisualizeMode/RayPath 明确 ValueError | 没有子类覆盖或实际生命周期行为，不能把空方法当成必要接口 |
| 几何失败与异常层 | affine/零面积几何用 GeometryError；删除 linalg 五个原样重抛分支 | 保留原来的校验和异常传播，不新增错误包装 |
| 显式 SciPy 预条件器恢复 | 只捕获分解 RuntimeError，MemoryError/ValueError 不再被吞掉 | 奇异分解的原有可恢复分支保留；没有改成 device 到 host 的隐式回退 |

`lumped_vertex_measures` 使用向量化叉积和 `np.add.at`，避免共享后把原来的向量化初始化退化为
Python 面循环。5120 面固定输入的节点权重最大绝对差为 0；单次 host 诊断约为旧 0.0594 s、
新 0.00134 s。这个测量只说明该预处理例子，不是求解器加速比例。

### 规则复查暴露的实际行为问题

- 普通 MPM NPZ 原先只存可视化不变量，不能恢复完整 strain 历史。现在沿用双相已有的
  `constitutive_state` raw 字段约定，保留 `state_vars` 输出字段；reader 优先读取 raw 历史。
  ConstitutiveBase 在任何历史 upload 前校验所有字段名、粒子数、容量、分量、有限值和精度范围。
  旧展开的对称张量可恢复；丢失张量的标量不变量明确报错，不再 `except: continue`。
- 非 restart NPZ 插入曾引用仅在 restart 分支存在的 `particle_info`；现从实际 `particle_cloud`
  读取历史，并按插入起点恢复，保留已有粒子的历史。
- 普通 restart kernel 曾写死 vec3/mat3。新增真实往返测试发现二维编译失败，现按粒子字段的
  维度构造位置、速度、约束和速度梯度。二维/三维往返比较所有这些数组及 scalar/vector/matrix 历史。
- 本构等效应变输出从 Python/NumPy 改到现有材料 kernel 文件中的 device 导出，保持
  原 `sqrt(4/3 J2)` 约定。独立 NumPy 参考仅在 tests 中。
- MPM.add_contact 剩余的 IPC 字符串别名删除，只接受 `BarrierIPC` 及三个正式接触检测类型。
  FEM 错误提示和 helper 手册同步。双相 `DragModel` 使用 `Ergun`、`Darcy`、`Beetstra`；
  拼写错误不再静默回退 Ergun，device 内部整数编号不变。参数键与非模型选项的既有别名不批量改动。

### 验证证据

| 检查 | 结果及边界 |
| --- | --- |
| 模板、几何、NURBS、有限应变、双相构造回归 | 112 passed，1 skipped；跳过项为本机 CPU 请求被 GeoTaichi 映射到 Metal 后不支持的 BlockScan sparse_grid 用例 |
| 历史恢复、二维/三维粒子往返、模型名称、输出、数学和显式 SciPy 恢复 | 118 passed；首轮 1 failed/56 passed 暴露二维恢复维度问题，修正后包含原失败用例全过 |
| f32 补充回归 | `GEOTAICHI_REAL_DTYPE=float32` 配合 `--taichi-fp f32`，device 应变输出和二维粒子重启 2 passed |
| 本轮 CPU pre-commit | 107 passed，54 deselected；命令为 `python tests/testing/run_partition.py pre-commit -q --taichi-arch cpu` |
| 静态结构 | 630 个 src 文件：无纯导入中转文件、class/游离函数混放、多个大型独立实现、host 本构 evaluator、裸 RuntimeError 或 bare except；47 个相关小类组保留 |
| 导入与差异 | 受维护直接 src 导入无未解析目标；`git diff --check` 通过 |
| 单元测试收集 | 当前树 3104 项成功收集；收集成功不表示全部执行 |
| 文档与工具 | helper API 全类别方法/配置键覆盖无遗漏；model-builder self-test 8 passed/0 failed；更新手册成功编译为 564 页 |

分组原始结果保存在本机 `/private/tmp/geotaichi_simplification_*.log` 及
`/private/tmp/geotaichi_ownership_*.log`，不是仓库的生成依赖。手册构建使用
`latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=/private/tmp/geotaichi_ownership_manual docs/helper/geotaichi_user_theory_manual.tex`，
未覆盖已有实验产物。原有 20 项 API 方法缺口和随后发现的 35 项配置键缺口已按实际消费者补齐，
后者包含多个 facade 共用的条目；覆盖检查只是文档存在性证据。

本轮 118 项回归的实际命令（使用 `/opt/anaconda3/envs/geotaichi/bin/python`）：

```bash
python -m pytest tests/unit/mpm/test_constitutive_restart.py \
  tests/unit/mpm/test_static_two_phase_preconditioner.py tests/unit/mpm/test_model_names.py \
  tests/unit/mpm/test_two_phase_restart.py tests/unit/utils/test_linalg.py \
  tests/unit/utils/test_field_io_prefix.py \
  tests/unit/physics_model/materials/test_fluid_bulk_modulus.py \
  tests/unit/linear_solver/test_tolerance_options.py \
  tests/verification/mpdem/test_incompressible_restart_pressure.py \
  -q --taichi-arch cpu --maxfail=3 --tb=short
```

CPU 测试有效后端为 arm64，精度和维度隔离由现有 fixture 控制。没有新增 CUDA/Metal 数值验证，
没有运行所有大型例子或全部 solver trajectory；目录与接口检查不能替代这些数值证据。

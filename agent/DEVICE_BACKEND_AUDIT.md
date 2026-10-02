# `src` 设备后端约束与审计

审计日期：2026-09-14。

设备驻留不自动等于低开销：重复 kernel 遍历、候选重建、trial 状态装配和 field
复制同样属于热路径问题。本轮针对“为了复用而重复工作”的检查、修改与有意保留
项记录在 [HOT_PATH_REUSE_AUDIT.md](HOT_PATH_REUSE_AUDIT.md)。

## 运行时约束

GeoTaichi 的数值 backend 以 Taichi field 为状态唯一真值。一个时间步或一个
Newton/摩擦固定点迭代内，下列计算必须由 `ti.kernel`/`ti.func` 完成：

- 内力、外力、阻尼、残量、能量、质量和状态更新；
- 单元/粒子/接触局部矩阵和全局 COO/HashTriplet 装配；
- Dirichlet 消元、矩阵向量乘、预条件、PCG/BiCGSTAB；
- 接触宽相、窄相、CCD、lagged friction 缓存和 AL 乘子更新；
- trial/accepted/rollback 状态复制和向量范数归约。
- 可微轨迹 checkpoint/replay、精确残量 Jacobian 装配、伴随线性求解、
  状态 VJP 和材料/接触参数 VJP。

Python 只允许承担以下边界工作：

- 配置检查、网格/材质/拓扑预处理和用户回调调度；
- Newton、line search、CCD 的标量控制流以及读取少量计数/状态标量；
- Taichi 不能在 kernel 内完成的 field 容量增长；
- 用户请求的记录、可视化、checkpoint 和诊断快照；
- 用户明确选择 `linear_solver="Scipy"` 时的一次线性系统转换和求解；这也
  适用于可微流程，但 device solver 不能静默回退到 SciPy。

`to_numpy()/from_numpy()` 因而可以出现在输入、预处理、输出和显式 Scipy
线性求解边界，但不能出现在设备 backend 的时间步、Newton、line search、
CCD 或摩擦热循环里。NumPy 参考公式可以作为输出/验证 adapter 存在，但不能由
`run()`、`step()` 或默认 solver 分派到。

## 本轮检查结果

| 模块 | 生产路径 | 结果 |
| --- | --- | --- |
| FEM explicit/Newmark | Taichi state kernels | 阻尼力、残量、加速度、预测/校正、边界投影、trial/rollback 和范数均在 device；不存在 NumPy backend |
| FEM volume/TRI3 | Classical assembler + COO/HashTriplet | `TRI3/TET4/HEX8` 的 F、能量、力、切线和散射在 device |
| FEM cloth | Cloth assembler + COO/HashTriplet | `3x2 Ds Dm^-1`、膜、二次/二面角弯曲、garment stitch、冻结帧 SDF、定点 spring、解析 Hessian、PSD 和散射均在 device |
| FEM IPC/AL | DynamicLinkedCellBroadPhase / DynamicBVHBroadPhase + contact kernels | linked cell 每次 count → prefix sum → fill；BVH 每次 device refit 后对候选 count → prefix sum → fill；CCD 用所选后端重新搜索且不用 Verlet multiplier；PT/EE/plane、IPC/AL friction 和持久 AL hash 状态在 device |
| FEM/cloth 可微 | device tape + exact COO/HashTriplet adjoint | replay、未投影残量 Jacobian、PCG→BiCGSTAB、初态/材料/接触 VJP 在 device；只有显式 `linear_solver="Scipy"` 会下载并求解该线性系统 |
| IGA explicit/implicit | IGA kernels + COO/Hash | 默认 PCG 在所有 Taichi 架构可用；rest shape 在预处理装入 reference field；参考 Jacobian 和轴对称参考半径由 device 状态归约并在进入求解前统一拒绝非法构形 |
| direct MPM | MPM kernels + COO/Hash | explicit/implicit 热循环保持 field resident；Scipy 仅是显式线性求解选项 |
| static two-phase UL-MPM | HashTriplet BiCGSTAB | 压力投影、残量、Newton 增量、line search 和状态合法性归约在 device；旧 Python Poisson 调试求解器已删除 |
| semi-implicit two-phase MPM | MGPCG/COO device solver | COO Poisson 不再转 Scipy；旧带尾下划线的 CPU 调试实现已删除 |
| SoftParticle IPC-MPM | monolithic HashTriplet | 所有 Taichi 架构强制设备 PCG/BiCGSTAB；CPU Dirichlet/CSR backend 已删除 |
| Direct MPM 可微 | device particle/contact tape + monolithic adjoint | DP/VM return-map、lagged friction、跨步 VJP 与参数累加都在 device；初始 acceleration→gravity 的 O(N) 累加也已移出 NumPy |
| DEM AffineBody | MatrixFree/COO/HashTriplet device path | 所有 Taichi 架构使用设备 nonlinear/CCD/line-search 和持久状态；未初始化 Taichi 时生产 operator 明确报错 |
| AffineBody 可微 | device tape + exact HashTriplet adjoint | 多步 replay、关节/阻尼/lagged friction VJP 和 PCG→BiCGSTAB 在 device；level-set 初始消穿也不再调用 host optimizer |
| DEM movable patch wall | persistent geometry state + Taichi kernels | 接触合力/力矩归约、刚体速度与角速度、指数映射、SVD 姿态正交化和墙面更新均在 device；每步不再下载合力或调用 `numpy.linalg` |
| MPDEM SoftAffineIPC | monolithic HashTriplet device path | 所有 Taichi 架构使用相同设备热循环；步首/接受不再整向量上传下载，只有 recorder 同步输出 |
| FEM-MPM IPC | monolithic COO/HashTriplet | 当前/扫掠候选、PT/PE CCD/ACCD、材料与接触装配、Armijo 和状态合法性归约均在 device；失败步用 Taichi 快照事务性恢复 FEM 节点、MPM 粒子/网格、完整 `F0` 和塑性历史；Cartesian 2D 塑性采用 3D plane-strain 嵌入 |
| IGA-MPM IPC | monolithic COO/Hash + device closest point | 所有 Taichi 架构使用设备 contact/Newton/Krylov；预处理、求解和接受共用一个 Taichi 快照事务，失败时恢复 IGA/MPM 物理状态、步级位移、`F0` 和 Direct ULMPM 塑性历史；2D curve 最近点也在 kernel 中 |

## 仍保留的主机代码不是 backend

以下代码有意保留，但不参与默认数值推进：

- meshio/OBJ/Gmsh/VTK 输入输出、网格生成、边界选择和 NURBS fitting；
- recorder、post-plot、checkpoint 和结果 dataclass；
- FEM 的 `assemble_output()`/有限差分验证 adapter；
- FEM 接触中私有的 `_reference_broad_phase_candidates()` 二次复杂度实现仅用于把 linked-cell/BVH 结果与独立 oracle 对照；公共 `broad_phase_candidates()` 和生产接触均执行用户选择的 Taichi 宽相；
- 稀疏矩阵的 `to_scipy()`/`to_numpy()` 检查接口；
- adaptive MPM 改变网格层级时的拓扑平衡、稀疏节点容量扩展和 hanging-node 表重建；这些是离散拓扑重建边界，粒子标记、计数、拆分及力学推进仍在 kernel；
- 若干历史单元测试直接调用的 host oracle。生产 operator 没有 Taichi runtime
  时会报错，不能退回这些 oracle。

部分已有私有函数名仍含 `_cuda`，这是兼容旧调用者的历史名称；其分派条件已经
改成“任意已初始化 Taichi 架构”，不再表示 CUDA-only。新代码应使用
`device` 命名，后续公共 API 清理时可以在不改变数值路径的情况下移除旧别名。

`BuildTriplet`/`HashReduction` 现在在 CPU、Metal、CUDA 上都默认设备归并；SciPy
不会在这些对象构造时加载。只有显式 `Scipy` 求解、矩阵输出或验证接口会延迟
导入 SciPy。CPU/Metal 的显式 host-reduction 开关仅保留给独立数值 oracle，任何
生产 assembler 都固定传入 `device_reduction=True`。

## 后续提交检查

修改 backend 时至少搜索：

```text
to_numpy()  from_numpy(  np.  scipy  spsolve  csr_matrix  coo_matrix
```

每个命中都必须属于上面的允许边界，或者迁移为 kernel。性能优化不能以悄悄
改变本构、接触激活、CCD 安全步长、PSD 层级或边界条件语义为代价；这类数值
差异必须在相应模块的复现/验证文档中单独登记。

# GPU performance work in GeoTaichi

Use this workflow for a measured GPU bottleneck. Read the owning solver and
the [device boundaries](SOLVER_INTEGRATION_PRINCIPLES.md#5-architecture-scope) first. Profiling is an optional
local investigation; it adds no cloud service or commit-hook requirement.

## Establish the comparison

1. Preserve a passing numerical reference. Hold material/contact laws, geometry,
   timestep, precision, tolerances, assembly representation, and output cadence
   constant. Set import-sensitive dimension and `GEOTAICHI_REAL_DTYPE` before
   importing specialized modules; match field types and `ti.init(default_fp=...)`.
2. Record the actual runtime backend, GPU, driver, Taichi version, and revision.
   A requested `--arch` value or benchmark JSON label is insufficient. Inspect
   `ti.lang.impl.current_cfg().arch`, the startup output, and actual GPU activity.
   Use explicit `ti.cuda` and `enable_fallback=False` in a reduced CUDA driver.
   Adapt its existing initialization; initializing Taichi twice invalidates fields.
   The public `geotaichi.init` wrapper accepts `cpu`/`gpu`, not a `cuda` string.
3. Warm the selected kernels and specializations before timing. Keep compilation
   and allocation separate from steady-state work. For evolving scenes, compare
   matching physical states and record active particles/nodes/pairs and solver
   iterations; a faster empty contact set is a different workload.
4. Measure the affected complete step or iteration with `ti.sync()` immediately
   before and after the timed region. Reset inputs consistently outside that
   region when benchmarking repeated assembly/solves. Retain individual samples
   and compare repeated measurements, not a single asynchronous launch.

## Choose the tool

| Question | Tool and evidence |
| --- | --- |
| Is time spent compiling or repeatedly specializing? | Taichi scoped profiler: `ti.profiler.print_scoped_profiler_info()`. |
| Which kernels dominate CPU/CUDA execution, and how often do they run? | [Taichi kernel profiler](https://docs.taichi-lang.org/docs/profiler): aggregate counts and kernel time. |
| Where are CUDA launches, host stalls, transfers, and synchronization? | [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html): inspect the application timeline before tuning one kernel. |
| Why is a selected CUDA kernel expensive? | [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html): targeted hardware counters and launch/resource analysis. |
| How does a Metal run behave on a Mac? | [Xcode Instruments / Metal system trace](https://developer.apple.com/metal/tools/): correlate CPU, GPU, and memory activity. |

Run each profiler separately. Use ordinary synchronized, uninstrumented runs
for the final speed comparison. CUDA metrics do not establish Metal performance;
use a supported scene/precision on each device and identify differences explicitly.

## Reuse a benchmark that measures the touched path

| Existing driver | Scope |
| --- | --- |
| [IGA fixed assembly](../tools/benchmarks/igampm/benchmark_fixed_assembly.py) | Fixed-slot solid assembly against a raw-matrix numerical oracle; excludes compilation from its samples. |
| [FEM-MPM fixed assembly](../tools/benchmarks/fempm/benchmark_fixed_assembly.py) | TET4/HEX8 assembly comparison. |
| [Device Krylov](../tools/benchmarks/linear_solver/benchmark_device_krylov.py) | Host/device scalar recurrences, true residuals, and warmed linear-solve timings. |
| [MPM sparse grids](../tools/benchmarks/mpm/sparse_grid_benchmark.py) | Sparse-grid behavior; inspect its options and scenario before selecting it. |

These microbenchmarks do not measure a full coupled timestep. For a step-level
claim, also run the smallest representative solver scene. Existing drivers may
record the requested architecture; verify the effective backend before using
their results as GPU evidence.

## Start with Taichi profiling

Enable `kernel_profiler=True` in the driver's existing CPU/CUDA initialization.
The following is an adaptation fragment: `step`, `warmup`, and `steps` belong to
that driver. Preserve its state/reset policy and chosen precision.

```python
import time
import taichi as ti

for _ in range(warmup):
    step()
ti.sync()
ti.profiler.clear_kernel_profiler_info()
started = time.perf_counter()
for _ in range(steps):
    step()
ti.sync()
print("measured wall seconds:", time.perf_counter() - started)
ti.profiler.print_kernel_profiler_info("count")
```

Use `"trace"` for individual launches. Kernel totals omit host-side work; compare
them with the wall boundary above. Taichi's advanced CUPTI mode needs a compatible
CUDA build/toolkit; do not assume the installed wheel has it. Use Instruments
for Metal analysis rather than expecting CUDA counters from this API.

## Follow the CUDA timeline with Nsight Systems

On an available NVIDIA CUDA machine, use the repository Python environment.
The commands below use the existing IGA assembly driver and keep reports in a
new temporary directory. Keep baseline and candidate source snapshots separate.

```bash
export GEOTAICHI_REAL_DTYPE=float64
profile_dir=$(mktemp -d "${TMPDIR:-/tmp}/geotaichi-profile.XXXXXX")
python tools/benchmarks/igampm/benchmark_fixed_assembly.py \
  --arch cuda --elements 30 --repeats 7 --output "$profile_dir/baseline.json"
nsys profile --trace=cuda,nvtx --sample=none --output="$profile_dir/system" \
  python tools/benchmarks/igampm/benchmark_fixed_assembly.py \
  --arch cuda --elements 30 --repeats 7 --output "$profile_dir/nsys-run.json"
nsys stats --report=cuda_gpu_kern_sum,cuda_api_sum "$profile_dir/system.nsys-rep"
```

Locate warm execution after JIT/setup. Inspect GPU idle gaps, launch frequency,
copies, synchronization, and repeated allocation or rebuilds. The NVTX trace
option captures existing annotations; it does not add labels automatically.
If phase boundaries remain ambiguous, add temporary ranges in the diagnostic
driver around search, assembly, solve, and update, or use a focused capture range
as described in the [Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#focused-profiling).

## Inspect a hotspot with Nsight Compute

Use the [Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
on an observed kernel specialization, with a small launch count:

```bash
ncu --set basic --kernel-name 'regex:OBSERVED_FIXED_ASSEMBLY_KERNEL' \
  --launch-skip 2 --launch-count 1 --export "$profile_dir/kernel" \
  python tools/benchmarks/igampm/benchmark_fixed_assembly.py \
  --arch cuda --elements 30 --repeats 7 --output "$profile_dir/ncu-run.json"
```

Replace the regex with the actual kernel name. The example skip count assumes
one matching fixed-pattern kernel per warmup assembly and two warmup calls;
check the driver and trace, including the separate raw-matrix oracle. The count
applies to matching launches, not simulation steps. Names can change with JIT
specializations and revisions. Add `--target-processes all` only when the target
driver actually delegates GPU work to children, as dimension-isolated pytest does.

Start with `basic`; use `ncu --list-sections` to select additional memory,
occupancy, or scheduler analysis for the question at hand. Counter availability
depends on the device/tool installation. Record missing access or unsupported
metrics as a limit. [Replay](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#replay)
can add passes and memory save/restore overhead, so an instrumented application's
total runtime is not the final throughput measurement.

## Use the Metal path on a Mac

In Xcode Instruments, select Metal system trace and target the Python process
that runs the supported Metal scene. Locate warmed compute activity and inspect
CPU/GPU overlap, command submission gaps, and resource traffic. Correlate this
with synchronized wall timings from the same scene. The
[Apple tools guide](https://developer.apple.com/metal/tools/) describes the capture tools.

Record the effective backend and precision. If the scene's required precision
is unsupported, report that limit; a changed-precision run is a separate result.
An available Mac CPU/Metal run cannot replace missing NVIDIA counter evidence.

## Turn evidence into a small change

| Observed cost | Inspect in GeoTaichi |
| --- | --- |
| GPU gaps near scalar reads or copies | Python convergence dispatch, `to_numpy`/`from_numpy`, repeated field reads, and permitted output boundaries. Keep solver vectors and reductions on device. |
| Excessive launches, rebuilds, or resets | Workspace and sparse-pattern lifetimes, repeated candidate queries, unneeded Hessians, and trial/accepted state copies. |
| Heavy memory traffic or contention | Field access order, active indices, scatter/atomic patterns, and existing reduction/assembly paths under [linear solvers](../src/linear_solver/README.md). |
| Register pressure, spills, or poor latency hiding | Large local matrices, static unrolling, live temporaries, and measured block size. Occupancy alone is not an optimization target. |
| Repeated compilation | New field/template identities or static variants created inside time/Newton loops. Move ownership to setup while preserving reset and rollback. |

Share the smallest mathematical operation when shared orchestration would add
queries, restore passes, or unneeded Hessians. Prepared contacts may be reused
only for the geometry and direction they bound; preserve standalone validation.

Change one measured cause, then rerun the numerical reference and the same
uninstrumented benchmark. Preserve stress measures, tangents, symmetry rules,
convergence criteria, capacities, and accepted/trial history. Do not obtain a
speedup by reducing precision, work, or the requested physics without making
that a separate explicit change.

## Record the result

Report command/revision, actual backend/device, versions, precision, scene and
active counts, warmup and measured range, synchronization boundary, individual
samples/summary, iteration counts, numerical errors, and before/after results.
Distinguish kernel, assembly/solve, and complete-step gains. Keep JSON and profiler
reports as investigation artifacts outside tracked source; put durable findings
beside the owning module or in an existing audit.

Use the [local commit checks](../tests/README.md#local-commit-checks) for commits.
Profiling runs and performance benchmarks remain separate manual evidence.

# Diagnose a Taichi failure in GeoTaichi

Use this guide when a kernel crashes, an active set is corrupted, a result
depends on execution order, or CPU and GPU behavior disagree. For an incorrect
but finite numerical result, also use [numerical validation](NUMERICAL_VALIDATION.md).
For timing and counters, use [GPU optimization](GPU_OPTIMIZATION.md).

## Keep the failing execution identifiable

Record the command, interpreter, revision and local edits, input/seed,
dimension, field precision, actual backend, assembly representation, and first
failing step/iteration. Preserve the first traceback and the last accepted
state before reducing the scene. Keep output in a temporary directory.

The [public initialization](../geotaichi/__init__.py), a test fixture, and a
standalone `ti.init` call can select different configurations. Inspect
`ti.lang.impl.current_cfg().arch` in the running process. Startup messages,
`gpu` filenames, and internal method names containing `cuda` are insufficient.
The pytest `--taichi-arch` and `--taichi-fp` options configure only tests using
the shared `taichi_runtime` fixture; other tests can own or override initialization.

Set `GEOTAICHI_REAL_DTYPE` before importing [TypeDefination.py](../src/utils/TypeDefination.py).
Configure all dimension-sensitive imports before loading their annotations;
reuse [dimension isolation](../tests/README.md#runtime-and-dimension-isolation)
for a maintained test. A late `ti.reset()` does not re-import Python modules.

## Find the first stage that corrupts state

Reduce particle/control-point counts and steps while preserving the selected
law, geometry feature, constraint, and parallel assembly that trigger the defect.
Observe boundaries in this order:

```text
configuration and geometry -> candidate/active-index construction
  -> local law and scatter -> assembled operator and right-hand side
  -> linear/nonlinear solve -> trial update -> accepted state
```

Use the facade's `diagnostics_snapshot()` for the existing time/step and any
engine-provided failure/history fields. The [adapter](../src/utils/SolverDiagnostics.py)
omits unavailable fields; it is not a dump of every kernel buffer.
Add a temporary `ti.sync()` after a suspect phase to attribute asynchronous
errors. Inspect a small diagnostic field or an active prefix through
[FieldIO.py](../src/utils/FieldIO.py), rather than downloading every capacity-sized
array each iteration. Remove diagnostic barriers and dumps after locating the cause.

Check finite values where they are first produced: geometry metrics, local
stress/contact terms, matrix blocks, residuals, corrections, and step lengths.
Keep the invalid value and owning indices in the failure report. Replacing NaN
with zero or adding an unexplained epsilon changes the problem instead of
establishing a valid degeneracy policy.

## Use a controlled CPU probe

For a reduced standalone kernel harness, adapt its existing initialization:

```python
import taichi as ti

ti.init(
    arch=ti.cpu,
    default_fp=ti.f64,
    cpu_max_num_threads=1,
    debug=True,
    offline_cache=False,
    random_seed=0,
)
assert ti.lang.impl.current_cfg().arch == ti.cpu
```

Set import-sensitive GeoTaichi precision/dimension before importing the
implementation under test. This fragment owns its runtime; do not prepend it
to a script that also calls `geotaichi.init`, or retain fields across reinitialization.
Match the original precision for the final reproduction; an f64 probe of an
f32 failure is a separate diagnostic.

[Taichi debugging](https://docs.taichi-lang.org/docs/debugging) documents CPU/CUDA
bounds checking with `debug=True` and single-thread CPU execution. Compare a
fresh single-thread CPU process with a fresh parallel CPU process, then the
original device. A serial pass only narrows the investigation; inspect shared
write destinations, read/write ordering, and storage aliasing before concluding
that the GPU has a race. Debug timing is unsuitable for a performance claim.

The shared [runtime fixture](../tests/conftest.py) currently uses one CPU thread.
Passing `--taichi-arch cpu` therefore does not exercise parallel CPU assembly.
Use a reduced harness with the appropriate thread count for that comparison.
Keep caches intact; disable offline caching only in the diagnostic process.

## Audit active indices and workspace reuse

| Boundary | Check |
| --- | --- |
| Candidate/scatter allocation | Capacity includes the worst supported stencil and duplicate entries. Detect overflow before a reserved slot is written; retain the overflow failure. |
| Index conversion | Particle, body, node, control point, global DOF, compact row, and block-local component have distinct ranges. Apply each offset once. |
| Sentinel handling | Guard an established inactive sentinel before dereferencing it. Treat an unexpected invalid index as an error instead of discarding a physical contribution. |
| Reset/reuse | Reset active counts and accumulators, overwrite every consumed slot, and invalidate topology-dependent mappings when their inputs change. |
| Parallel scatter | Use disjoint destinations or the existing atomic/reduction path. Check repeated coordinates and coupled block offsets. |
| Small and boundary counts | Include empty, single-entry, capacity-edge, and changing active sets; verify that a smaller next call cannot reuse stale entries. |

Existing examples include [bucket reduction](../tests/unit/linear_solver/test_contact_bucket_reduction.py),
[large contact stencils](../tests/unit/physics_model/contact/ipc/test_contact_runtime_loop_bounds.py),
and [prefix-transfer checks](../tests/unit/utils/test_field_io_prefix.py).
These are starting points; retain a reproduction of the particular failing path.

## Check CUDA memory and synchronization on NVIDIA hardware

Use [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
on an available CUDA machine with a compatible toolkit. Start with `memcheck`.
The following targets an existing strict CUDA/f64 test from the repository root:

```bash
GEOTAICHI_REAL_DTYPE=float64 compute-sanitizer --tool memcheck \
  --error-exitcode 1 --target-processes all \
  python -m pytest -q -rs \
  tests/unit/physics_model/contact/ipc/test_ipc_real_gpu_kernels.py::test_real_cuda_psd_and_hash_reduction_match_numpy_oracles
```

Require actual execution of the test, not a capability skip. Its
[fixture](../tests/unit/physics_model/contact/ipc/test_ipc_real_gpu_kernels.py)
disables fallback and checks the selected backend. Replace the node with the
smallest strict CUDA reproduction of the affected operation; the example does
not validate every solver. Child tracing also covers dimension-isolated tests.

| Tool | Question it addresses |
| --- | --- |
| `memcheck` | Invalid or misaligned device memory accesses. |
| `initcheck` | Uninitialized device global-memory reads. |
| `racecheck` | Shared-memory access hazards; it does not establish that global-field scatter is race-free. |
| `synccheck` | Incorrect use of device synchronization primitives. |

After fixing memory errors, run the tool appropriate to the remaining symptom.
Taichi JIT symbols/source attribution and tool support depend on the installed
versions. A clean report covers only the executed paths. These CUDA tools do
not check a Metal run, and a missing tool/device remains unverified evidence.

## Verify the repair on the original path

Rerun the focused oracle, the original backend/parallel configuration, and the
smallest original scene that exposed the failure. Remove diagnostic serialization
and synchronization before claiming the defect is fixed. Valid floating-point
reduction-order differences need a scale-aware numerical comparison; they do
not alone establish an invalid parallel write.

Report the first bad stage, root cause, repaired invariant, actual backend,
passing commands, and any device-specific checks still missing. Follow the
[local commit policy](../tests/README.md#local-commit-checks); sanitizer runs are
manual investigations and do not expand the ordinary gate.

# Working in GeoTaichi

GeoTaichi is a Python/Taichi simulation library. Start here, then read only the
documents and source paths needed for the requested change.

## Find the owner

1. Check `git status --short` and preserve existing edits and experiment data.
2. Read the owning module's README, nearest maintained test, and consuming code.
   Follow an API or dictionary key from its public entry through validation to
   the engine/kernel that uses it. Current implementation and numerical evidence
   take precedence over an outdated summary.
3. Search callers before changing shared behavior. Include readers, writers,
   configuration, examples, and documentation in the change when they depend on it.
4. Reuse the local implementation and vocabulary. Keep the diff within the task;
   leave third-party code, dependencies, caches, and generated results alone.

| Task | Read next |
| --- | --- |
| Library implementation or review | [Code paths](agent/CODEMAP.md), relevant [workflow](agent/WORKFLOWS.md), and the owning `src/<module>/README.md`; use [coding guidelines](agent/CODING_GUIDELINES.md) for detailed rules |
| Model or public example | [Model-builder instructions](agent/geotaichi-model-builder/SKILL.md) and the nearest example using that solver path |
| Materials, contact, or linear algebra | [Physics models](src/physics_model/README.md), [contact detection](src/contact_detection/README.md), or [linear solvers](src/linear_solver/README.md) |
| Test selection and local commits | [Test guide](tests/README.md#local-commit-checks) and [partition definitions](tests/testing/test_partitions.json) |
| MCP or Blender integration | [MCP package](agent/geotaichi-mcp/README.md) or [Blender add-on](blender/README.md) |
| GPU profiling, device residency, or hot-path changes | [GPU optimization workflow](agent/GPU_OPTIMIZATION.md) and [device boundaries](agent/SOLVER_INTEGRATION_PRINCIPLES.md#5-architecture-scope) |
| Kernel crashes, invalid indices, or backend-dependent results | [Taichi debugging](agent/TAICHI_DEBUGGING.md) |
| Derivative, assembly, adjoint, or rollback evidence | [Numerical validation](agent/NUMERICAL_VALIDATION.md) |

## Numerical work

- Preserve units, coordinate frames, signs, precision, capacities, and accepted
  configuration keys. Derive changes to residuals, tangents, transfers, or contact
  from the governing equation or a testable invariant.
- Keep numerical time-step and nonlinear-iteration work in Taichi fields and
  kernels. Python owns setup, dispatch, output, and scalar control. An explicitly
  selected SciPy linear solve is a separate supported boundary, never a silent
  fallback for device computation.
- Preserve accepted/trial state and rollback, including constitutive history.
  Report invalid geometry, NaN, overflow, and non-convergence as failures.
- Set import-sensitive precision and dimension before loading specialized modules.
  Use the existing isolated test fixtures and record the effective backend;
  CPU evidence does not validate GPU races or GPU-only behavior.
- Keep API keys, state schemas, recorders, restart readers, and their tests in
  agreement. Do not change physics or loosen tolerances to obtain a passing test.

## Local validation

Use the Python environment containing GeoTaichi's dependencies. Run the smallest
test that demonstrates the change, then use the local commit checks:

```bash
python tests/testing/run_partition.py pre-commit -q --taichi-arch cpu
```

An ordinary commit runs this explicit small set, including VTU round trips.
A message containing `[full test]` also runs `commit-validation`: NURBS geometry,
and small linear systems. It additionally selects `commit-materials` and/or
`commit-contact` when the staged diff touches those laws or shared dependencies;
unrelated changes skip the larger law groups. The marker does not request every
test under `tests/`. Follow the [hook setup](tests/README.md#local-commit-checks);
let failures block the commit rather than bypassing the hooks. No cloud CI is required.

## Finish the change

Update documentation only where behavior, usage, or a stated limitation changed.
Keep durable decisions beside the owning implementation or in an existing audit;
keep experiment chronology out of the default reading path. Report what changed,
the exact checks and effective backend, and anything still unverified. A basic
commit gate is evidence for its listed cases, not a whole-solver validation claim.

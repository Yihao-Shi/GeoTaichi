# Development work in GeoTaichi

Choose the relevant procedure after locating the owner with [CODEMAP.md](CODEMAP.md).
The [coding guidelines](CODING_GUIDELINES.md) define the constraints; this page
helps turn a request into a small, verifiable change.

## Fix a failing result

1. Preserve the original command, first failure, input, effective backend,
   precision, dimension, and relevant iteration or step. Separate a reproduced
   failure from a hypothesis based only on reading code.
2. Reduce the failing mechanism while keeping its physical law and execution
   path. Choose an analytical value, invariant, finite-difference comparison,
   or independent small reference that can expose the defect.
3. Follow the value through initialization, assembly, solve, state update, and
   output. Locate the first incorrect stage rather than repairing the final
   visualization. Read every caller before changing a shared operation.
4. Make the smallest repair in the owning implementation. Preserve error
   handling and rollback; do not compensate by changing the requested problem.
5. Rerun the failing assertion and the original reproduction when practical.
   Record any larger reproduction that still needs verification. Add a focused
   regression when the behavior is not already covered.

For generated models, use the existing [validation loop](geotaichi-model-builder/references/agent-validation-loop.md)
and its contract/evidence records instead of inventing another diagnosis format.

For kernel crashes, invalid indices, or order/backend-dependent results, follow
[Taichi debugging](TAICHI_DEBUGGING.md). For an oracle that separates local laws,
assembly, physical/projected derivatives, and history, use
[numerical validation](NUMERICAL_VALIDATION.md).

## Extend a supported solver path

Trace one maintained sibling from its public key through validation, allocation,
dispatch, kernels, and output. Identify which native/direct, explicit/implicit,
dimension, and assembly branches actually support the requested behavior.
Reuse the shared law only where its measures and history match the consumer.

Before adding dispatch, define required parameters, state ownership and reset,
capacity growth/overflow behavior, and unsupported-combination errors. Wire the
facade/manager and actual consumer together; an accepted key with no consuming
branch is incomplete. Use the existing [coordination map](CODING_GUIDELINES.md#25-coordinated-changes)
for affected readers, tests, examples, and documentation. Keep unsupported
branches explicit rather than silently selecting another algorithm.

## Change a material

- Identify the consuming manager and branch first. Native MPM small-strain
  updates, Direct MPM finite-strain updates, FEM, and IGA do not automatically
  share stress measures, state layouts, or supported plasticity.
- Write down the stress/strain measures, sign convention, parameter units,
  state variables, return/flow rule, and meaning of the tangent being changed.
- Check zero increment/rest state, the affected elastic or inelastic branch,
  accepted-history update, and trial-state isolation. Differentiate the actual
  stress/residual measure in the numerical oracle; compare an energy Hessian
  only where the formulation has the corresponding potential.
- Follow history into checkpoint and restart readers when its shape or meaning
  changes. Keep rejected trials from updating accepted material history.
- Use the [material coverage matrix](../tests/unit/physics_model/materials/README.md)
  to choose targeted tests. A complete law-test run still does not substitute
  for a regression of a previously uncovered plastic or hardening branch.

## Change contact or coupling

Trace candidate generation, closest-feature geometry, active indices, local
force/residual/tangent, assembly, feasible step, and accepted state. Establish
coordinate frames and the sign of the gap before comparing quantities.

Check equal-and-opposite exchange and force/moment or work balance where the
formulation requires them. Test the touched explicit/implicit and assembly
paths separately; a shared facade does not prove matching implementations.
Keep conservative lagged-IPC potential checks distinct from the residual-merit
checks used for fully implicit friction. See the [test guide](../tests/README.md#test-layers-and-oracles).

Reject a failed step without retaining partial particle/grid/body or material
history changes. Keep capacity/feasibility failures visible rather than dropping
candidates or weakening convergence tests.

## Diagnose an implicit solve

Capture the first failing iteration, active DOFs, actual operator/solver route,
initial and final true residuals, stopping threshold, and breakdown/nonfinite
status. Use the [linear-solver contract](../src/linear_solver/README.md) and
[numerical validation](NUMERICAL_VALIDATION.md#separate-local-laws-from-global-assembly).

1. Check physical feasibility, finite local terms, index mappings, duplicate
   reduction, and nonzero Dirichlet elimination before tuning iterations.
2. Compare a small assembled operator and `A p` to an independent reference.
   Distinguish the physical Jacobian from the matrix projected or lagged for
   the solve; choose PCG or a nonsymmetric solver from that actual operator.
3. Recompute `b - A x` against the operator being solved. Check null modes,
   constraints, scale, and preconditioner validity when the linear solve fails.
4. If the linear solve passes but Newton/backtracking fails, check the direction
   sign, physical tangent product, feasible step, trial-state restoration, and
   the intended merit function. Lagged IPC and fully implicit friction use
   different globalization contracts.
5. Verify rollback after a forced failure. Preserve the failed reproduction;
   larger limits, relaxed tolerances, or a different backend do not by themselves
   establish a repair.

## Change output, restart, or a client contract

Follow the writer and reader together. Preserve field names, shapes, dtypes,
logical IDs, time/step meaning, units, and schema versions. A readable file is
not proof that it can continue a simulation.

- For VTU, read back connectivity, cell types, coordinates, and field values.
- For restart, check restored physical/history state and the next accepted step
  against an uninterrupted reference. Check capacity and topology compatibility.
- For Blender/MCP, trace shared assets through validation, staging, the job CLI,
  and result loading. Keep Blender data access on its main thread and numerical
  execution in the external worker; use the [existing integration guide](../blender/README.md).

## Measure a performance change

For GPU tool selection and concrete Taichi/Nsight/Metal procedures, follow
[GPU_OPTIMIZATION.md](GPU_OPTIMIZATION.md).

First reproduce a numerical baseline, then measure the complete affected step
or iteration before selecting a kernel to optimize. Inspect repeated traversals,
candidate rebuilds, field allocation, and host/device transfers alongside kernel
time. Preserve each caller's geometry and accepted/trial lifecycle when reusing
prepared work.

Use an existing driver under [tools/benchmarks](../tools/benchmarks/) when it
measures the requested path. Hold geometry, physics, tolerance, precision, and
output cadence constant. Separate first-use compilation from warmed execution
and synchronize at matching timing boundaries. Measure diagnostic/profiler runs
separately when their synchronization changes throughput.

Record the command, revision, device, effective backend, precision, problem
size, warmup, measured steps, timing boundary, correctness result, and before/
after values. Rerun without temporary diagnostics. Do not present CPU timing as
GPU evidence or valid atomic-order rounding as a proven race. Keep important
measurements in the owning document or an existing audit, not the startup guide.

## Review, document, and deliver

For a review, read the changed implementation with its callers and relevant
tests; preserve the working tree. Report the failing trigger, consequence,
source location, and evidence for each actionable finding. Distinguish a
confirmed bug from an untested risk, and prioritize physics/state loss and
incorrect public behavior over formatting preferences.

For a behavior change, check the owning README, public API/theory entries,
examples, and machine-consumed model knowledge. Update only affected surfaces.
Use the [documentation synchronization rules](CODING_GUIDELINES.md#121-documentation-synchronization-gate)
and keep one owner for each fact. Compile edited examples and check local links
and code fences; a Markdown check does not validate a numerical formula.

Finish with the concrete outcome, commands/results, and remaining limits. Use
the existing [model handoff](geotaichi-model-builder/references/handoff-contract.md)
for model deliverables. Perform commits, pushes, PR comments, tags, or package
publication only when they are included in the user's authorized task.

## Select local checks

Run the focused test that proves the change. For example, a small sparse-system
change can be checked with:

```bash
python -m pytest tests/unit/linear_solver/test_assembly_backends.py -q --taichi-arch cpu
```

Then follow the [local commit checks](../tests/README.md#local-commit-checks).
The ordinary gate includes VTU read/write; `[full test]` additionally selects
NURBS and small linear systems. Material/contact changes and shared dependencies
select the corresponding complete law groups; unrelated changes skip them. The
[partition JSON](../tests/testing/test_partitions.json) owns the exact selection.
Neither commit mode requests the entire test directory. A targeted regression
outside these lists remains necessary for a change to that behavior.

## First checks for misleading failures

| Symptom | Check before changing the numerical method |
| --- | --- |
| Imports fail in one shell or hook | Compare `python -c "import sys; print(sys.executable)"` with `git config --get geotaichi.python` and any `GEOTAICHI_PYTHON` override; use the existing environment before changing pins. |
| A marker-selected run fails during collection | Marker filtering does not prevent test-module imports. Check stale imports and duplicate module basenames; use the explicit commit partitions for the small gate, and fix collection defects separately rather than masking them. |
| A 3D test passes alone but fails in a group | Dimension-sensitive annotations may have been imported in the parent 2D process. Use [dimension isolation](../tests/README.md#runtime-and-dimension-isolation), not a late reset as a substitute for re-importing. |
| A precision change has no effect | `GEOTAICHI_REAL_DTYPE` is read by [TypeDefination.py](../src/utils/TypeDefination.py) at import. Check both stored field types and the initialized runtime. |
| A CPU/GPU claim disagrees with the result | Inspect the initialized backend with [runtime_architecture](../src/utils/SolverConsole.py) and the fixture/entry path; a requested label alone is insufficient. |
| The first step dominates a benchmark | Separate Taichi compilation/cache effects from steady-state work before changing algorithms. |
| A residual or friction test changes after another module | Check process-global settings, mutable solver state, and the [runtime isolation fixtures](../tests/conftest.py). Preserve the original failure and rerun the relevant sequence. |
| A solve looks stable but the trajectory is wrong | Inspect scaling, local/global/compact index conversions, load transfer, convergence diagnostics, and accepted-history updates before relaxing tolerances. |
| A VTU opens but restart diverges | Visualization fields do not establish a restart contract; check time, topology, constitutive history, and a continuation comparison. |

When a new failure produces a durable lesson, add the source/test-backed rule
to its owner. Keep machine-specific errors and one-off investigation history in
the task handoff rather than expanding the global rules.

# GeoTaichi Development Guidelines

This file is the operational guide for coding agents working in GeoTaichi. Read
`agent/CODING_GUIDELINES.md` for the full contribution standard. When guidance conflicts, current source code and
the closest maintained test are authoritative; report the conflict instead of guessing.

## Source of truth

Use sources in this order:

1. The current public facade and implementation in `geotaichi/` and `src/`.
2. The nearest maintained test under `tests/unit`, `integration`,
   `verification`, or `regression`.
3. One to three nearby examples using the same solver path.
4. The helper documentation specification and its theory/API sources under `docs/helper/`.
5. Repository metadata and older assistant instructions.

Do not invent an API name, nested-dictionary key, accepted value, compatibility rule, or numerical formula. Trace it
to the code that consumes it. Repository dependency and build metadata currently come from several files; compare
`pyproject.toml`, `requirements.txt`, `geotaichi_env.yml`, and the relevant setup file before changing any of them.

## Before editing

- Run `git status --short` and preserve all user changes. Do not clean, reset, or reformat unrelated files.
- Find the nearest sibling implementation, test, and example with `rg` or `rg --files` before introducing structure.
- Confirm whether the target is public API, solver orchestration, a Taichi hot path, research code, or generated output.
- Keep the change within the requested subsystem. Do not edit `third_party/`, nested repositories, `build/`, caches,
  or experiment results unless the task explicitly requires it.
- State assumptions that affect physics, compatibility, precision, backend support, or data formats.

## Repository map

- `geotaichi/`: public import surface, runtime initialization, and signed-distance-function helpers.
- `src/dem/`: discrete element method (DEM) facade, scene generation, contact, neighbor search, and engines.
- `src/mpm/`: material point method (MPM) facade, explicit/implicit engines, grids, generation, and soft particles.
- `src/mpdem/`: DEM-MPM and computational fluid dynamics-DEM (CFDEM) coupling.
- `src/fem/`, `src/fedem/`, and `src/fempm/`: finite elements, explicit
  FEM-DEM, and explicit or monolithic implicit-IPC FEM-MPM surface coupling.
- `src/iga/` and `src/igampm/`: isogeometric analysis (IGA) and IGA-MPM coupling.
- `src/physics_model/`, `src/contact_detection/`, `src/linear_solver/`, `src/sdf/`, and `src/utils/`: shared methods.
- `src/nurbs/cnurbs/`: native non-uniform rational B-spline (NURBS) sources and its local build script.
- `examples/`: user-facing workflows. Use public facades here.
- `tests/`: maintained unit, integration, verification, regression, and
  opt-in benchmark layers; see `tests/README.md`.
- `tools/diagnostics/`, `tools/derivations/`, `tools/benchmarks/`,
  `taichi_demo/`, and `research/`: explicit manual or experiment-specific
  code, never implicit pytest inputs.
- `docs/helper/`: the maintained user theory/API manual and its specification.
- `agent/geotaichi-model-builder/`: canonical model-building Skill, generated
  capability index, validation references, templates, and maintenance scripts.
- `agent/geotaichi-mcp/`: MCP source organized into shared `core`, read-only
  `knowledge`, task/live `execution`, versioned SolverJob resources/CLI, and
  safe/trusted public tool/server profiles.
- `blender/`: thin SolverJob/SceneManifest client. Keep Blender data access on
  the main thread and numerical execution in external GeoTaichi workers.

## Priorities

- Preserve physical correctness, conservation properties, units, frames, signs, and solver convergence behavior.
- Prefer clear host-side setup and validation during initialization.
- Prefer bulk Taichi kernels and bounded memory traffic in per-step hot paths.
- Keep public facade behavior and established configuration dictionaries stable unless an API change is requested.
- Make the smallest coherent change. Avoid speculative helpers, compatibility branches, and broad cleanup.

## Public API and configuration dictionaries

- Public examples normally import from `geotaichi`; do not expose internal engine classes without an established need.
- Follow the lifecycle used by nearby examples: initialize the runtime, configure the solver, allocate memory, add
  materials/elements/regions/bodies/boundaries, select output, then run.
- Preserve exact key spelling, capitalization, nesting, aliases, defaults, and accepted values in configuration dicts.
- Validate invalid combinations on the Python side with a precise `ValueError` or `RuntimeError` before launching a
  kernel. Reuse the subsystem's existing `validate_configuration` flow when available.
- A new public key requires implementation, validation, a focused test, an example when user-visible, and matching
  API documentation.

## Taichi runtime and kernels

- `geotaichi.init()` configures process-global Taichi state. Use the isolated
  fixtures in `tests/conftest.py`; use a separate process for incompatible
  dimensions, backends, or import-time global settings.
- `GEOTAICHI_REAL_DTYPE` is read while `src/utils/TypeDefination.py` is imported. Set it before importing GeoTaichi.
- Isolate tests that require different dimensions, backends, default types, or import-time environment settings in
  separate processes unless an existing fixture safely resets the runtime.
- On Apple Silicon, `geotaichi.init(arch="cpu")` selects Metal. Do not claim that such a run validates the CPU backend.
- Match the local pattern for free `@ti.kernel` functions versus methods on `@ti.data_oriented` classes.
- Keep Python orchestration outside `@ti.kernel` and `@ti.func`; use explicit kernel argument types and use
  `ti.template()` or `ti.static()` only for compile-time behavior.
- Assume the outermost kernel loop is parallel. Every shared write must be disjoint, atomic, or intentionally
  serialized. Never accept a CPU-only pass as evidence that a race is safe.
- Do not allocate fields or perform full field-to-NumPy transfers inside a simulation-step loop. Use bulk transfers;
  for one-dimensional subsets, prefer `field_to_numpy_slice` or `field_to_numpy_prefix` from `src/utils/FieldIO.py`.
- Preserve backend restrictions and test a backend-specific change on that backend when it is available.

## Numerical and data rules

- Treat array shape, dtype, index space, coordinate frame, and units as part of every interface.
- Keep transfers between Taichi fields and NumPy explicit and outside tight loops. Avoid scalar host-device traffic.
- Preserve the configured capacity model used by `memory_allocate`; validate overflow rather than dropping entries or
  silently changing a user's capacity. Traverse and transfer the active prefix instead of the unused capacity tail.
- When a user asks how many GB/GiB of GPU memory a Python example reserves, automatically run
  `python agent/geotaichi-model-builder/scripts/estimate_gpu_memory.py <script>`. Report the target platform,
  `estimated_preallocated_pool_gib`, confidence, and notes from its JSON output; pass relevant `--env` overrides and
  `--platform linux` for a Linux/CUDA target. Do not execute the model or mistake CUDA/graphics context overhead for
  the configured Taichi device-memory pool.
- Changes to residuals, tangents, barriers, contact forces, transfer operators, or time integration require a
  mathematical invariant, an analytical result, or a finite-difference/reference comparison.
- Never hide NaN, infinity, capacity overflow, failed line search, or non-convergence with a warning and continuation
  when the resulting state is physically invalid.
- Do not alter a physical parameter, tolerance, precision, time step, or iteration limit merely to make a test pass.

## Style

- Match the nearest maintained sibling. Historical GeoTaichi code mixes naming styles; do not perform mass renames.
- Use clear `snake_case` for new local variables and functions unless the surrounding public API establishes another
  convention. Preserve public class names, file names, dictionary keys, and serialized field names.
- Python lines use the configured Black limit of 120 characters. Format only touched files and avoid unrelated churn.
- Put imports at module scope unless lazy loading or a circular dependency requires a local import.
- Comments and docstrings describe current behavior and explain non-obvious invariants, formulas, units, or backend
  constraints. Change history belongs in Git, not source comments.
- Remove debug prints and dead commented-out code from the edited path. Use the established logging mechanism for
  durable diagnostics.
- Never add credentials, tokens, private keys, passwords, or machine-specific absolute paths. Read secrets from the
  environment and build paths relative to the repository or an explicit user-provided location.

## Testing

Start with the smallest relevant test node, then run its named partition.
Maintained test modules are side-effect-free at import.

```bash
python -m py_compile path/to/changed_file.py
python -m pytest tests/unit/<component>/test_feature.py::test_specific_behavior
python tests/testing/run_partition.py <component> -q
```

- Some maintained tests honor `GEOTAICHI_TEST_ARCH=cpu|cuda`; inspect the test before relying on it.
- Prefer CPU and float64 for small analytical checks when the implementation supports them. Add a real GPU run for
  CUDA/Metal-specific kernels, atomics, sparse fields, or memory behavior.
- A bug fix needs a regression test that fails before the fix and passes after it.
- Assert physics or numerics: conservation, analytical motion, monotonic energy, contact separation, derivative and
  Hessian checks, matrix assembly equivalence, or a justified reference result. "Runs without error" is insufficient.
- Choose tolerances from precision, scale, conditioning, and discretization error. Investigate failures before
  widening a tolerance or reducing coverage.
- Keep scenes, particle counts, and step counts minimal. Write outputs to pytest temporary paths or `/private/tmp`.
- Never pipe pytest through `tail` or `grep`; that can hide failures and the exit status. Capture complete output first
  if a log is needed.

If Black is installed, check only the changed Python paths:

```bash
python -m black --check path/to/changed_file.py
```

## Examples and documentation

For example scripts, public configuration dictionaries, or helper API documentation, read
`agent/geotaichi-model-builder/SKILL.md` first. It identifies the facade call order, compatibility checks, and source
files to inspect.

For MCP tools, transports, resource loading, or persistent task behavior, read
`agent/geotaichi-mcp/README.md`. Keep model knowledge and templates in the
model-builder Skill; the MCP package consumes them and must not duplicate them.

- Read one to three neighboring examples and preserve their import, setup, output-path, and call-order conventions.
- Compile every edited example with `python -m py_compile`; run a short smoke case when practical.
- Keep the helper manual's theory and API entries synchronized. Follow
  `docs/helper/geotaichi_documentation_spec.md` and document each dictionary key at its full nesting path.
- Build the helper manual with its documented `latexmk -xelatex` command when the required toolchain is available.
- Do not commit incidental logs, caches, VTK/NPZ output, or LaTeX intermediates.

## Dependencies, native code, and tooling

- Do not upgrade or reconcile dependency pins unless explicitly requested. Report disagreements between manifests.
- Treat root packaging metadata and subsystem-local native build scripts separately; verify the exact path and command
  before claiming that a build works.
- Changes under `src/nurbs/cnurbs/` must follow neighboring C++ style and be validated with that directory's actual
  build path. Do not copy obsolete paths from older setup metadata.
- Remote, cluster, and long-running GPU commands require task-specific context. Reuse an existing script under the
  relevant `research/*/scripts/` directory only after reading it and confirming its inputs and outputs.

## Completion checklist

- Review the diff and confirm that only task-scoped files changed.
- Run syntax checks plus the smallest relevant tests; state exactly what ran and what could not run.
- Apply the mandatory documentation synchronization gate in `agent/CODING_GUIDELINES.md` whenever behavior,
  configuration, numerical methods, backends, outputs, or limitations change. Update every affected module README,
  `docs/helper` API/theory entry, example, and agent instruction in the same change; explicitly report reviewed but
  unaffected surfaces.
- Report numerical or backend limitations and any pre-existing failures separately from the implemented change.
- Follow `agent/CODING_GUIDELINES.md` for naming, review, tests, documentation, and contribution details.

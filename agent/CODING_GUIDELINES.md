# GeoTaichi Coding Guidelines

This document defines how code is designed, implemented, tested, documented, and reviewed in GeoTaichi. It is the
detailed companion to `agent/CLAUDE.md` and applies to contributors and coding agents.

Three rules resolve most uncertainty:

- **Current code is authoritative.** Trace public behavior from the facade to validation and the implementation that
  consumes it. Do not derive an API from prose alone.
- **Match the closest maintained sibling.** Before adding a helper, kernel, option, test, fixture, or file layout,
  inspect the same capability in the same subsystem and reuse its vocabulary and structure.
- **Documentation ships with behavior.** A code change is incomplete until every affected module README, helper
  manual entry, example, and agent-facing instruction is updated and validated in the same change.

## 0. Coding-agent prerequisite: Ponytail

Coding agents must install and enable [Ponytail](https://github.com/DietrichGebert/ponytail) before implementing or
reviewing GeoTaichi code. The repository checkout belongs at `agent/ponytail`; do not copy Ponytail rules or scripts
into solver packages. If the checkout or plugin is missing, run:

```bash
git clone https://github.com/DietrichGebert/ponytail.git agent/ponytail
node agent/ponytail/scripts/check-rule-copies.js
codex plugin marketplace add DietrichGebert/ponytail
codex plugin add ponytail@ponytail
```

Verify installation with `codex plugin list`. In Codex, open `/hooks`, review and trust Ponytail's two lifecycle
hooks, then restart the Codex application and begin a new thread so the plugin is active. An agent working in the
current thread before hooks can be reloaded must read `agent/ponytail/AGENTS.md` completely and apply its rules
directly rather than postponing the review.

Ponytail supplements these project-specific guidelines; it does not override numerical correctness, public API,
restart compatibility, or trust-boundary validation. Apply its YAGNI, single-owner, direct-import, and deletion-first
rules when reviewing every change, while retaining checks that protect physical validity, finite state, capacity,
strict IPC feasibility, accepted-step rollback, external inputs, and persisted data.

## 1. Philosophy and priorities

- Physical and numerical correctness comes first. Preserve governing equations, sign conventions, units, frames,
  conservation laws, boundary conditions, and convergence criteria.
- Reuse before adding. Search the owning subsystem and its maintained siblings for an existing model, kernel,
  topology, sparse assembly path, solver, validation rule, or test fixture before writing a new one. Extend or compose
  the owning implementation when its mathematical assumptions match; duplicate code only when the assumptions differ,
  and document that difference at the new ownership boundary.
- Hot-path simplicity and cost take priority over reuse. Do not reuse an interface when doing so adds avoidable field
  traversals, kernel launches, host-device synchronization, allocations, format conversion, candidate rebuilds, or
  full-vector copies to a per-step, Newton, line-search, CCD, or friction path. In that case, extract the smallest shared
  mathematical operation into a local `ti.func`, kernel, or narrow helper, or write a specialized implementation whose
  inputs and state match the consuming backend. Record the ownership boundary and cover equivalence with tests.
- Initialization and scene construction favor clarity, explicit validation, and maintainability.
- Per-step solver paths favor bounded allocations, bulk operations, and minimal host-device synchronization.
- A contribution should improve correctness, robustness, performance, usability, or maintainability in a way that can
  be demonstrated by tests, benchmarks, or simpler public behavior.
- Keep changes narrow. Do not combine a feature or fix with unrelated renaming, formatting, generated-output changes,
  dependency upgrades, or research cleanup.
- Delete dead code introduced or exposed by the task when ownership is clear. Do not add speculative abstraction or
  compatibility logic for a hypothetical future caller.
- Preserve user changes and local experiments. A dirty worktree is normal and is never permission to reset files.

## 2. Architecture and ownership

### 2.1 Public surface

`geotaichi/` is the public import surface. It initializes Taichi and exposes solver constructors and signed-distance
helpers. User-facing examples should use this facade unless a low-level example in the same directory intentionally
demonstrates an internal method.

Public behavior includes more than Python signatures:

- nested configuration-dictionary keys and their accepted values;
- call order and initialization requirements;
- default precision, backend, and allocation behavior;
- output file names and serialized field names;
- documented exceptions and compatibility restrictions.

Changing any of these is an API change and requires explicit review, validation, tests, and documentation.

### 2.2 Solver implementations

- `src/dem/` owns DEM state, bodies, walls, clumps, contact, neighbor search, and time integration.
- `src/mpm/` owns MPM particles, grids, elements, materials, transfers, explicit/implicit solvers, and soft particles.
- `src/mpdem/` owns coupled DEM-MPM and CFDEM orchestration and exchange terms.
- `src/fem/`, `src/fedem/`, and `src/fempm/` own FEM, explicit FEM-DEM, and
  explicit DEM-law plus monolithic implicit-IPC FEM-MPM surface coupling.
- `src/iga/` and `src/igampm/` own IGA and IGA-MPM basis, assembly, solve, and coupling behavior.
- Shared numerical code belongs in the narrowest relevant shared package under `src/`; do not create a generic helper
  merely to avoid a small local expression.

Keep orchestration, state ownership, and numerical kernels in their existing layers. A consumer should use the
owning subsystem's method rather than reconstructing derived state from several private fields.

### 2.3 Examples, tests, research, and third-party code

- `examples/` is user-facing and must favor supported public APIs and reproducible relative paths.
- `tests/` contains maintained unit, integration, verification, regression,
  and opt-in benchmark layers described in `tests/README.md`.
- `tools/diagnostics/`, `tools/derivations/`, `tools/benchmarks/`,
  `taichi_demo/`, and `research/` contain explicit manual or expensive
  workflows. Do not promote local shortcuts into library conventions without
  verification.
- Vendored or copied external libraries must live under
  `third_party/<upstream-project>/`, never under `src/`, `geotaichi/`, or an
  owning solver package. Keep the upstream license, notices, and required
  runtime assets with the vendored library; project-specific adapters and
  integration code remain in the owning GeoTaichi package.
- `third_party/`, nested repositories, build directories, and generated
  experiment data are outside normal edits unless the task explicitly adds,
  updates, or removes a vendored dependency.

### 2.4 Agent knowledge and MCP integration

- `agent/geotaichi-model-builder/` owns model-building instructions,
  references, the generated capability index, model templates, and their
  maintenance scripts.
- `agent/geotaichi-mcp/src/geotaichi_mcp/` owns MCP integration code: `core`
  contains shared configuration/contracts/resources, `knowledge` contains
  read-only browsing/inspection handlers, `execution` contains persistent task,
  isolated diagnostic, and cooperative live-queue handlers, and root `tools.py`
  is only the stable registration facade. Numerical loops expose only the
  dependency-free `src.utils.RuntimeHook` safe point; they must not import MCP
  or transport code.
- Keep root `pyproject.toml` as the single build manifest and keep MCP tests in
  `tests/unit/mcp` and `tests/integration/mcp`; do not add a second package
  manifest or a parallel solver-wide logging subsystem. MCP execution may own
  only its per-task stdout/stderr capture and must continue reusing existing
  solver recorders.
- Every public MPM, DEM, MPDEM/CFDEM, FEM, FEDEM, FEMPM, IGA, and IGAMPM
  facade exposes the dependency-free common diagnostics contract from
  `src.utils.SolverDiagnostics`. Specialized engines may extend the snapshot;
  do not fork facade-specific task schemas.
- Shared model assets remain canonical in the model-builder Skill and are
  installed as package data. Do not duplicate generated indexes or templates
  inside the MCP Python package.
- `SolverJob` and `SceneManifest` templates are shared model-builder assets.
  MCP and the repository-root `blender/` add-on consume the same versioned
  documents; task submission stages immutable copies before worker launch.
- Network transports default to the safe tool profile and require bearer
  authentication. Trusted script/code tools remain explicit, and Blender must
  move blocking job I/O off its main thread without accessing Blender data in
  the worker.

## 3. Naming and formatting

- Match naming in the owning subsystem. GeoTaichi has established CamelCase module/class names and snake_case methods;
  do not launch broad renames to enforce a new global style.
- New local variables and free functions should use descriptive `snake_case` unless a neighboring numerical kernel
  has a well-established mathematical convention.
- Reuse domain terms accurately: particle, grid/node, material, body, wall, clump, element, control point, contact,
  degree of freedom, residual, gradient, Hessian, tangent, and increment are not interchangeable.
- Boolean names should read as conditions (`is_active`, `has_contact`, `enable_friction`) and preserve established
  public option spelling.
- Index names must reveal the indexed space when ambiguity is possible. Distinguish particle, node, body, material,
  contact, local, global, active, and compact indices.
- Include units or frames in names when the surrounding type cannot make them clear.
- Preserve exact spelling and capitalization of public dictionary keys, enumerated strings, environment variables,
  output fields, and restart data.
- Python formatting follows Black's configured 120-character line length. Do not reformat untouched regions merely
  because an old file differs from Black.
- Wrap a long call consistently: prefer one argument per continuation line instead of mixed packing.
- C++ and shell changes follow the closest file because the repository defines no global formatter for them.
- Use English and ASCII in new source comments and identifiers when practical, while preserving surrounding files and
  externally defined notation. Never rewrite unrelated non-ASCII historical content as cleanup.

## 4. Comments and docstrings

- Comments explain a non-obvious invariant, numerical choice, ordering constraint, backend restriction, or failure
  mode. They do not narrate obvious assignments or loops.
- Describe current behavior in the present tense. Put bug history and implementation chronology in commits and pull
  requests rather than comments such as "previously" or "fix for".
- A formula comment states symbols, units, sign/frame conventions, and the source equation when these are not clear
  from the code. Keep the implementation and referenced derivation synchronized.
- Put a comment immediately above the code it explains. Avoid fragile line-number references.
- Keep each fact at one source location. Other sites may point to the owning symbol or document without duplicating a
  long explanation.
- Function-level descriptions belong in docstrings. Inline comments belong beside local implementation details.
- Public docstrings describe parameters, accepted values, array shapes, units, side effects, errors, and returns.
- Option documentation must explain the user-visible tradeoff: benefit, cost, and when to choose each mode.
- Remove debugging prints and commented-out code from completed work. Use the subsystem's existing logging style for
  intentional diagnostics.

## 5. Imports, functions, and structure

- Keep imports at module scope. A local import is appropriate only for an established lazy-loading boundary, optional
  dependency, or real circular dependency; `src/__init__.py` contains intentional lazy constructors.
- Preserve the public facade boundary. Examples should not depend on an internal class when a facade operation exists.
- A helper should encode a reusable concept, not merely shorten one call site. Keep linear one-use logic local when it
  remains readable.
- Prefer module-level numerical helpers when no object state is required. Use methods when the owning state and
  invariants belong to the class.
- Avoid duplicate implementations of formulas or validation. Move shared logic only when callers truly share the same
  assumptions, dtype, shape, and failure behavior.
- Review reuse over the complete high-level operation, not one call at a time. A shared routine with many mode flags or
  repeated preparation/restoration passes is usually the wrong boundary for a solver hot path; share the narrow
  invariant operation and keep the caller-specific orchestration local.
- Do not attach attributes dynamically to foreign objects. Declare owned state during initialization so lifecycle and
  memory behavior stay visible.
- Keep optional state explicit with `None` or the established sentinel. Do not use broad `getattr`/`hasattr` probes to
  hide an unclear ownership boundary.
- Remove unused imports, temporary debug branches, and unreachable code touched by the change when safe to do so.

## 6. Public APIs and configuration dictionaries

Configuration dictionaries are a core GeoTaichi API. They are intentionally supported and must be treated as typed
schemas even when represented by plain Python dictionaries.

- Find every accepted key in the consuming facade, `Simulation`, manager, generator, contact model, or constitutive
  model. Do not infer keys from a similar project.
- Preserve full nesting, spelling, casing, aliases, defaults, units, and accepted value sets.
- Use the existing dictionary access and configuration-validation utilities in that subsystem.
- Validate required keys, ranges, mutually exclusive settings, backend restrictions, and solver compatibility before
  memory allocation or kernel launch whenever possible.
- Error messages identify the full key path, invalid value, and accepted condition. Use `ValueError` for invalid user
  values and `RuntimeError` for an unavailable runtime state or unsupported execution path.
- Do not silently coerce a materially different algorithm, precision, backend, or contact law. Established benign
  aliases may be normalized at the existing validation boundary.
- Add defaults only when they are safe across supported scenes. A new tuning knob must have a documented physical or
  numerical tradeoff and a robust default.
- A new or changed public key requires focused tests and updates to user examples and `docs/helper/` as appropriate.
- Maintain the established call lifecycle. Reject an operation invoked before its prerequisites rather than creating
  partially initialized state.

## 7. Types, arrays, memory, and data transfer

- Array shape, dtype, index space, coordinate frame, ownership, and lifetime are part of an interface. Document or
  assert them at the boundary where ambiguity would cause incorrect physics.
- Use the dtype definitions and precision flow already owned by the subsystem. Do not introduce isolated hard-coded
  `float32`/`float64` choices into a generic solver path.
- `GEOTAICHI_REAL_DTYPE` is evaluated when `src/utils/TypeDefination.py` is imported. Any process that changes it must
  set the environment variable before importing GeoTaichi.
- Cast only at a real boundary: user input normalization, Taichi kernel arguments, serialization, native libraries,
  or an explicitly mixed-precision algorithm. Repeated casts can hide precision errors and allocate copies.
- Keep NumPy work vectorized. Move per-particle, per-node, and per-contact hot loops into an appropriate Taichi kernel.
- Production differentiable simulation keeps contact search/assembly, forward
  replay, nonlinear updates, linear solves, and state/material VJPs in Taichi
  fields and kernels. NumPy/SciPy is limited to preprocessing, seed upload,
  scalar diagnostics, result download, and explicitly labelled test oracles;
  an explicitly selected `linear_solver="Scipy"` may transfer and solve only
  that assembled linear system. Device solvers must never fall back to it
  silently or move any other core work to the host.
- Hoist Taichi field-to-NumPy and NumPy-to-field transfers outside step loops. Transfer complete useful batches rather
  than individual entries.
- When only a one-dimensional field subset is needed, use `field_to_numpy_slice` or `field_to_numpy_prefix` from
  `src/utils/FieldIO.py` instead of materializing the full field.
- The public `memory_allocate` workflow uses maximum capacities by design. Compute or request justified capacities,
  validate overflow, and never discard data silently. Hot paths should traverse and transfer the active prefix rather
  than the unused capacity tail. Do not change capacity semantics as a micro-optimization.
- For a user question about the GPU memory reserved by a Python model, run
  `python agent/geotaichi-model-builder/scripts/estimate_gpu_memory.py <script>` automatically. Use the target
  platform and environment overrides requested by the user, and report the JSON result's
  `estimated_preallocated_pool_gib`, confidence, and notes. The estimate is the configured Taichi allocator pool; do
  not present it as CUDA/graphics context overhead, external-library memory, host RAM, or active bytes within the pool.
- Allocate persistent buffers during the established build/allocation phase. Avoid per-step Python or device
  allocations and preserve reuse of scratch fields where ownership is clear.
- Treat restart/NPZ/VTK field names, ordering, shapes, and dtypes as compatibility contracts. Update readers, writers,
  tests, and docs together when an intentional format change is requested.
- Use repository-relative paths or `pathlib.Path` composition. Machine-specific absolute paths belong only in local
  experiment configuration and must not enter library code, examples, or tests.

## 8. Taichi runtime and kernel code

### 8.1 Runtime lifecycle

- `geotaichi.init()` owns global Taichi initialization and GeoTaichi's dimension/backend configuration.
- Initialize once per process. Tests that require different dimensions, backends, precision, or import-time settings
  should use separate subprocesses unless a proven fixture performs `ti.reset()` at a safe boundary.
- Apple Silicon maps the facade's `arch="cpu"` request to Metal. Record the effective backend in test reports and do
  not treat that path as CPU validation.
- Environment variables that alter imports, precision, or dispatch must be set before importing their owning module.
- Do not delete global Taichi caches as a diagnostic. Disable caching for the one test process when necessary.

### 8.2 Kernels and functions

- Production numerical computation belongs in Taichi.  Per-particle,
  per-node, per-element, per-contact, and per-candidate work; force, energy,
  residual, gradient, tangent/Hessian, assembly, CCD/ACCD, broad-phase,
  culling, state update, reduction, line-search merit, and convergence-norm
  evaluation must execute in `@ti.kernel` or `@ti.func`.  Python may validate
  configuration, allocate fields, dispatch kernels, coordinate scalar
  iterations, perform output, and read scalar diagnostics.  It must not
  provide a NumPy/SciPy numerical-backend fallback for these operations.
- The sole production exception is an explicitly user-selected SciPy linear
  system solve.  Device assembly must be complete before that boundary; only
  the sparse matrix/right-hand side and solved increment may cross it.  The
  residual, contact search, CCD/ACCD, line search, state update, and subsequent
  assembly remain Taichi operations.  NumPy reference formulas and finite-
  difference oracles belong under `tests/`, never in a production solver
  package.
- Treat device residency as an invariant of every production time-step and
  nonlinear-iteration path.  Outside one-time preprocessing, output/postprocessing,
  objective/adjoint seed input and final-gradient output, scalar diagnostics,
  and the explicit SciPy exception above, do not call
  `to_numpy()`/`from_numpy()`, use NumPy sorting or deduplication for active sets,
  or make host decisions from particle/node/contact arrays.  Search, active-set
  marking and compaction, pair grouping, assembly, CCD, solve, and accepted-state
  updates must remain device-resident.
- This IPC rule applies equally to BarrierIPC and SemiIPC in standalone and
  coupled solvers; SemiIPC active-set projection and multiplier updates are not
  preprocessing and therefore must never use a host deduplication/sort path.
- Match the subsystem's existing choice between free `@ti.kernel` functions and methods on `@ti.data_oriented`
  classes. Both patterns are established in GeoTaichi.
- `@ti.func` and `@ti.kernel` bodies use Taichi-compatible data and control flow. Keep file I/O, NumPy operations,
  dynamic Python containers, logging, and exception construction on the host side.
- Annotate kernel scalar and ndarray arguments. Use `ti.template()` only for genuine compile-time polymorphism and
  `ti.static()` only when the branch or loop is intended to specialize at compilation.
- Treat every template value and static branch as a potential new compiled specialization. Avoid unnecessary variants
  in frequently constructed scenes.
- The outermost kernel loop is parallel by default. Writes from distinct iterations must target disjoint locations or
  use the appropriate atomic operation. Serialize only when the algorithm requires ordering and document why.
- Never rely on a non-atomic read-modify-write that happened to pass with one CPU thread. Validate race-sensitive code
  on the backend whose parallel execution matters.
- Do not pass the same mutable field under multiple semantic roles when the kernel assumes independent storage.
- Keep index conversions explicit. Distinguish compact, physical, local, global, active, and prefix-offset indices.
- Bound loops and field accesses from validated counts. Capacity overflow must stop before an out-of-bounds write.
- Avoid kernel `print` in completed code. For backend debugging, write minimal diagnostic state to a dedicated field
  and inspect it on the host.

### 8.3 Performance

- Allocate field-owning sparse solvers and numerical workspaces once, then reset
  and reuse them across Newton iterations, time steps, and reverse replay.  Do
  not construct new Taichi fields in a production hot loop; template identity
  changes trigger avoidable compilation and break persistent device residency.
- Contact hot paths must consume compact broad-phase candidates or active-pair
  lists; do not traverse particle/body, vertex/body, or body/body Cartesian
  products after a search has already produced a smaller set.  A brute-force
  implementation is allowed only as an explicit small-scene/capacity-overflow
  fallback or one-time preprocessing path, and must be labeled with its cost
  and upgrade condition.
- Measure before changing kernel layout, block size, sparse topology, atomics, or precision for performance.
- Report the device, backend, precision, scene size, warmup, step count, and synchronization points with benchmarks.
- Exclude first-use compilation from steady-state timing unless compilation time is the quantity under study.
- A speedup cannot trade away physical checks, supported backends, determinism requirements, or memory safety without
  an explicit design decision and documentation.

## 9. Numerical methods and physics

- Start from the governing equation or discrete invariant. Derive the implementation-level residual, gradient,
  tangent/Hessian, update, or transfer before changing the kernel.
- State coordinate frames, sign conventions, normalization, integration weights, and units at the ownership boundary.
- Preserve mathematical pairs together: energy/gradient/Hessian, residual/tangent, P2G/G2P, restriction/prolongation,
  predictor/corrector, contact/friction, and writer/reader.
- For analytical derivatives, add a finite-difference test at non-singular representative states. Check symmetry or
  positive-definiteness only when the formulation mathematically guarantees it.
- Handle degeneracy explicitly: zero distance, repeated nodes, empty active sets, near-incompressibility, contact
  cutoffs, singular matrices, and line-search limits must follow a documented policy.
- Do not label a method conservative, symmetric, differentiable, stable, or backend-independent without a derivation
  or test that supports the claim.
- Keep changes aligned with the helper theory manual when a user-visible numerical method changes.
- Reference implementations are evidence, not authority. Reconcile their units, conventions, discretization, and
  precision before comparing values.

## 10. Error handling and diagnostics

- Reject invalid user configuration before expensive allocation or compilation when possible.
- Never catch all exceptions merely to continue a simulation. Catch a specific exception only when the caller can
  recover or when adding context and re-raising.
- A physically invalid state, NaN/infinity, capacity overflow, failed required factorization, or exhausted nonlinear
  solve must produce a clear failure rather than a plausible-looking output.
- Error messages include the subsystem, operation, relevant key/count/value, and condition that was violated without
  dumping large arrays or sensitive data.
- Warnings are for recoverable behavior with a well-defined result. They are not substitutes for failed physics.
- Logging must be optional in tests (`log=False` where supported) and must not create repository artifacts as a side
  effect of import or collection.

## 11. Testing

### 11.1 Scope and location

- Put a test in the narrowest appropriate layer under `tests/unit`,
  `integration`, `verification`, `regression`, or `benchmarks`, mirroring the
  owning physics/solver capability. Create a new file only for a genuinely new
  capability.
- Start with a pure host-level unit test when it can validate parsing, formulas, or dispatch without initializing
  Taichi. Add the smallest kernel/integration test needed for execution behavior.
- Use a separate process for incompatible Taichi runtime configurations or when module imports freeze global settings.
- Mark or gate tests that truly require CUDA, Metal, a native extension, external data, or large memory. A skip message
  states the missing capability.

### 11.2 What to assert

- A bug fix includes a regression test that fails on the previous behavior and passes after the change.
- Assert meaningful results: conservation, analytical motion, equilibrium, no penetration, dissipative friction,
  monotonic energy, derivative accuracy, matrix assembly equivalence, convergence, or a justified reference output.
- "The scene ran" is a smoke check, not a correctness test.
- Every assertion is a maintained contract. Avoid redundant shape/None assertions already implied by a complete value
  comparison, but make dtype/shape explicit when that is the behavior being specified.
- Use `numpy.testing.assert_allclose` or `assert_array_equal` for array comparisons unless a nearby test has a more
  suitable established helper.
- Base tolerances on floating-point precision, nondimensional scale, conditioning, discretization order, and solver
  stopping criteria. Record why a loose tolerance is mathematically necessary.
- Investigate failures before widening tolerances, reducing steps, removing backends, or weakening assertions.

### 11.3 Cost and reproducibility

- Use the minimum particles, nodes, contacts, bodies, and time steps that exercise the mechanism.
- Set deterministic seeds and explicit precision/backend when the test depends on them.
- Disable log creation and offline cache in focused tests when those features are unrelated.
- Write temporary meshes, logs, NPZ/VTK files, and subprocess scripts to pytest's temporary directory or an explicit
  location under `/private/tmp`, then clean them through the fixture lifecycle.
- Never filter pytest output in a pipeline. Preserve the complete output and exit status.
- Run the exact failing assertion path after a fix; a smaller diagnostic reproduction is supplementary evidence.

### 11.4 Recommended validation sequence

```bash
python -m py_compile path/to/changed_file.py
python -m pytest tests/unit/<component>/test_feature.py::test_specific_behavior
python tests/testing/run_partition.py <component> -q
```

Expand to a component or backend suite only when the environment supports it.
All maintained test modules must remain safe to collect; benchmarks still
require the explicit `--run-benchmarks` option.

If Black is installed:

```bash
python -m black --check path/to/changed_file.py
```

## 12. Examples and documentation

### 12.1 Documentation synchronization gate

Every feature addition, behavior change, bug fix that changes observable behavior, public API change, configuration
change, numerical-model change, backend change, output-format change, or newly discovered limitation must include a
documentation impact review before the task is considered complete. Coding agents must perform this review
automatically; the user does not need to request documentation separately.

Review all of the following surfaces and update every affected one in the same change:

- `src/<owning-module>/README.md`: capabilities, package layout, supported models, solver/backend paths, usage,
  limitations, and relevant tests.
- `docs/helper/geotaichi_api_reference.tex`: public methods, complete nested configuration-key paths, accepted values,
  defaults, units, call order, errors, outputs, and compatibility restrictions.
- `docs/helper/geotaichi_theory_reference.tex` and `docs/helper/geotaichi_user_theory_manual.tex`: governing equations,
  discretizations, constitutive/contact laws, algorithms, assumptions, and API-to-theory cross-links when the
  numerical method changes.
- User-facing examples and example indexes when the supported workflow, defaults, required arguments, or recommended
  setup changes.
- `agent/` guidance, model-builder knowledge/templates, and MCP documentation when architecture, ownership, coding
  policy, agent workflow, validation commands, or machine-consumed public knowledge changes.

Documentation synchronization is impact-based: not every code edit requires changing every document, but every item
above must be considered. If an item is not affected, leave it unchanged and state that it was reviewed and why no
update was necessary in the final handoff. Never update prose to describe planned behavior that the code does not yet
implement, and never leave newly implemented public behavior documented only in tests or source comments.

Before completion:

1. Trace the final behavior from the public facade through validation to the consuming implementation; current code is
   authoritative when older documentation disagrees.
2. Search the repository for the changed API name, configuration key, model name, default, equation, and former
   limitation so stale copies are not left behind.
3. Update affected documentation and examples together with the implementation, using identical terminology and
   defaults across all surfaces.
4. Validate edited Python examples, Markdown links/code fences, and the helper manual according to its specification.
5. Report which documentation surfaces changed and which were reviewed but unaffected. A task with unresolved
   documentation drift is not complete.

For a literature-derived constitutive model, documentation must also identify
the exact stress/strain measures, sign convention, parameter units, flow rule,
hardening law, apex or tensile treatment, state variables, and tangent type.
Maintain one explicit ledger of deliberate deviations from the documented
equations. If a requested restriction (for example associated flow only)
removes an algorithmic loop, remove the unused runtime scaffolding
instead of retaining a dormant compatibility layer.

- Read `agent/geotaichi-model-builder/SKILL.md` before changing examples, public dictionary documentation, or helper
  API/theory material.
- Inspect one to three adjacent examples and follow their facade, call order, naming, and output style.
- Examples use public APIs, only necessary non-default options, repository-relative assets, and a bounded problem size.
- Each module owns a self-contained example script. Do not import another example or dispatch to it via `runpy`,
  `exec`, or dynamic loading. Keep scene geometry, initialization, boundaries, solver setup, output, and run logic
  in that script; ordinary library imports are allowed. Shared solver algorithms remain in `src/`.
- Keep solver initialization before configuration/allocation and keep generation, boundary, output, and run calls in
  the lifecycle expected by that facade.
- Check feature combinations against `Simulation.validate_configuration` and the actual consuming source. Do not
  combine switches because their names appear compatible.
- Run `python -m py_compile` for every edited example and a reduced smoke run when practical. Do not alter the
  published physics solely to make validation faster; use a temporary reduced copy.
- Follow `docs/helper/geotaichi_documentation_spec.md` for manual updates. Theory and API entries cross-link, and every
  dictionary key is documented separately at its full nesting path.
- User documentation states accepted values, units, behavior, tradeoffs, restrictions, and related theory. Internal
  kernel mechanics belong in developer comments or the theory implementation notes.
- Use the documented `latexmk -xelatex` command to validate the helper manual when the toolchain is available. Do not
  commit incidental LaTeX intermediates.

## 13. Dependencies, packaging, and native code

- Treat `pyproject.toml` as the primary packaging metadata, then compare `requirements.txt`, `geotaichi_env.yml`, and
  relevant setup scripts. Report contradictory pins or paths instead of silently choosing or reconciling them.
- Do not upgrade Taichi, NumPy, native toolchains, or broad dependency sets as part of an unrelated feature.
- A dependency change explains backend/platform impact and is tested in a clean environment representative of users.
- Verify console entry points and source paths exist before documenting or invoking them.
- Native NURBS sources live under `src/nurbs/cnurbs/` with a local setup script. Use that actual directory as the
  starting point; root build metadata may contain historical paths.
- C++ changes preserve Python binding signatures, dtype/shape expectations, ownership, and exception behavior. Add a
  focused Python-level test for a user-visible native change.
- Do not edit vendored or submodule code unless the task specifically targets it. Record the upstream revision and
  reason for any authorized vendor patch.

## 14. Git, generated files, and security

- Review `git status` before and after work. Change only task-scoped files and never discard another contributor's
  modifications or untracked experiments.
- Do not use destructive cleanup or history-rewriting commands as an implementation shortcut.
- Do not commit caches, logs, screenshots, large result directories, VTK/NPZ dumps, compiled objects, or LaTeX
  intermediates unless the artifact is an explicit deliverable.
- Never commit or print API keys, tokens, passwords, private keys, cookies, or other credentials. Use environment
  variables or an approved secret store, and redact secrets from diagnostics.
- Never commit machine-specific absolute paths, usernames, hostnames, or private dataset locations to reusable code.
- Keep commits focused. A commit title states the user-visible or engineering outcome; implementation detail belongs
  in the review description.
- Pull requests state the problem, the change, numerical/backend implications, and exact validation performed. Include
  benchmark methodology for performance claims and before/after evidence for bug fixes.
- Do not add automated-tool or AI attribution to source, commits, or pull-request metadata.

## 15. Review checklist

- The implementation follows the owning subsystem and nearest maintained sibling.
- Public keys, defaults, call order, outputs, and compatibility rules remain correct and documented.
- Taichi lifecycle, compile-time specialization, parallel writes, capacity bounds, and transfers are safe.
- Physics changes have a derivation or invariant and focused regression coverage.
- Tests cover the exact path on every claimed backend without weakened tolerances or assertions.
- Examples and helper docs are updated when users can observe the change.
- Dependency, native build, generated-file, path, and credential rules are satisfied.
- The final diff contains no unrelated edits and the handoff lists every validation command and limitation.

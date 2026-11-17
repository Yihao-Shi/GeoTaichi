# GeoTaichi Coding Guidelines

This document defines how code is designed, implemented, tested, documented, and reviewed in GeoTaichi. It is the
detailed companion to the repository-root `AGENTS.md` and applies to contributors
and coding agents. Use the entry document to select the sections relevant to
the task; module READMEs and maintained tests provide the local context.

Three rules resolve most uncertainty:

- **Current code is authoritative.** Trace public behavior from the facade to validation and the implementation that
  consumes it. Do not derive an API from prose alone.
- **Match the closest maintained sibling.** Before adding a helper, kernel, option, test, fixture, or file layout,
  inspect the same capability in the same subsystem and reuse its vocabulary and structure.
- **Documentation ships with behavior.** A code change is incomplete until every affected module README, helper
  manual entry, example, and agent-facing instruction is updated and validated in the same change.

## 0. Coding-agent prerequisites

Use [Karpathy Guidelines](https://github.com/multica-ai/andrej-karpathy-skills/tree/main/skills/karpathy-guidelines)
for coding and refactoring, alongside Ponytail below. Install `skills/karpathy-guidelines` from that repository with
the Codex skill installer and read its installed `SKILL.md` before writing code. Keep the upstream skill outside
solver packages; do not add a project-specific wrapper framework to enforce it.

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

### 2.5 Coordinated changes

Follow the behavior through its consumers before editing. The following map
identifies what must be checked together; it does not require changing every
listed surface for every task.

| Change | Trace and check together |
| --- | --- |
| Public option or default | Facade, configuration validator, actual consumer, invalid-input test, supported example, API documentation |
| Constitutive law | Parameter initialization, state layout, stress/energy/tangent, branch and history tests, restart compatibility, material documentation |
| Contact or coupling | Candidate generation, geometry/frame conventions, force/residual/tangent, feasibility and rollback, coupled invariant test |
| Output or checkpoint | Recorder/writer, loader/restart path, visualization consumer, field names/shapes/dtypes, round-trip test |
| Solver optimization | Callers across the complete step/iteration, workspace lifetime, transfers and synchronization, numerical equivalence, measured cost |
| MCP or Blender boundary | Shared contract, staging, CLI/tool consumer, resources/packaging, lifecycle tests |

Record a non-obvious design choice where the affected subsystem is documented:
state the constraint, chosen approach, rejected alternative if it matters, and
the supporting test or measurement. Update an existing README or audit before
creating another document. Historical timings and proposals are not current
capability guarantees.

## 3. Naming and formatting

### 3.1 Source naming

| Object | Convention | Examples |
| --- | --- | --- |
| Python package/directory | `snake_case`, with established solver abbreviations | `physics_model/constitutive_model`, `mpm/soft_particle` |
| Solver implementation module | `PascalCase.py`; preserve established acronym spelling | `SceneManager.py`, `TypeDefinition.py`, `LevelSetBVH.py`, `NeighborKernel.py` |
| Class, record, mixin, exception | `PascalCase`, without a leading underscore | `SceneManager`, `GeometryDeviceState`, `ClassicalAssemblyMixin`, `CapacityError` |
| Function, method, argument, ordinary variable/attribute | Descriptive `snake_case` | `find_body`, `body_id`, `particle_num`, `point_body_ids` |
| Fixed module constant / environment variable | `UPPER_SNAKE_CASE` | `THB_MAX_INFLUENCE_NODES`, `GEOTAICHI_REAL_DTYPE` |
| Maintained test / standalone tool | `test_<behavior>.py` / `snake_case.py` | `test_source_layout.py`, `run_partition.py` |

Use the implementation-module style for solver classes, kernel collections and shared numerical categories; do not
rename a file merely to match one class. Keep the established `mainDEM.py`, `mainMPM.py`, etc. facade entry names.
Lightweight support modules (`config.py`, `constants.py`, `errors.py`, `linalg.py`), the SDF factory modules,
visualization adapters and MCP/Blender packages retain their existing `snake_case.py` convention. Spell out a variant
in implementation names (`MultiGridPCGMixtureAxisymmetric`); do not mix CamelCase with `_mixture_Axi` suffixes.

Do not add leading underscores to implementation files or classes as a visibility marker. Internal callables may
use a leading underscore (`_get_position`). Remove trailing implementation underscores; distinguish real variants
by their operation (`restrict_radius_weighted`, `usf_updating_drained`). Python protocol names such as `__init__`
are unchanged. Host functions, kernels, `ti.func` and `ti.pyfunc` all use `snake_case`: `compute_stress`,
`d_psi_div_d_f`, `get_essential`, `linearize3d`, `kernel_normalize_nodal_dpressure_2d`. Acronyms in callables are
lowercase; dimension tags use `2d`/`3d`. Decorators identify the execution boundary, so do not add wrappers or
runtime dispatch merely to give kernels and functions separate naming prefixes.
SDF's exported callable classes (`sphere`, `rounded_box`, `text`, etc.) keep their public `snake_case` factory names;
this exception is limited to the existing factory modules, not other classes in `src/sdf`.

### 3.2 Numerical notation and compatibility

- Keep recognizable mathematical notation in variables naming an equation quantity: `F`, `J`, `K`, `Ax`, `rTr`,
  `dPsi_dF`, `dNdxi`. Short algebraic symbols and derivative notation are not prose variables to lowercase mechanically.
- Use `body_id`/`node_id` for ordinary local IDs and `particle_num`/`particle_count` for counts, matching the closest
  maintained caller. Distinguish a body index from a body count and a point-body ID array from one scalar ID.
- Existing public Python keyword-argument spellings, callable constructors (`DEM`, `MPM`, etc.), accessible state attributes and
  Taichi struct fields remain compatibility
  contracts. For example, a local `body_id` may read `particle.bodyID`; `bodyID` in output/restart data stays `bodyID`.
  New internal names follow the table. Changing one of these contracts needs its own complete consumer/schema
  migration; never fix only the declaration or add aliases to hide a partial rename.
- Internal kernel, `ti.func` and `ti.pyfunc` arguments follow `snake_case`, with the same mathematical-notation
  exceptions as locals. Rename their launch/call keywords together. Keep separate argument and local bindings
  distinct (`input_wall_id` versus `wall_id`, whole `state_variables` versus one local `state_vars` record).
- Fixed module constants use uppercase (`THRESHOLD`, `L_THRESHOLD`, `ZEROVEC3F`, `ZEROMAT3X3`); keep the
  established tensor dimensions and type suffixes readable. Internal mutable module variables use `snake_case`,
  with imports and `global` declarations migrated together. Type aliases retain `PascalCase`; runtime configuration
  attributes such as `GlobalVariable.Taylor` and `GlobalVariable.Affine` retain their compatibility spelling.
- Numerical callables follow the same rule as other functions (`ComputeStress` → `compute_stress`, `Psi` → `psi`).
  Property getter/setter names identify state attributes, so existing properties such as `gridSum` and
  `materialID_numpy` retain their state-interface spelling. Source imports, method references, reflection and
  embedded subprocess code migrate together; model-selection strings and serialized keys keep their exact values.
- A source-module/class rename updates imports, exports, lazy-import strings, monkeypatch targets, subprocess code,
  maintained examples/tools/tests, capability-index source paths, and documentation links together. Do not keep
  forwarding implementation files. Historical migration records can retain the paths that existed at the time.
- Before renaming locals, inspect the complete scope, nested references and comprehensions. Check collisions with
  arguments and other locals; `bodyNum` and `body_num` can denote different values. Preserve signatures, field names,
  dictionary keys, string payloads, decorators, annotation precision and every numerical expression.
- `tests/unit/testing/test_source_layout.py` enforces source file/class naming, local binding style and resolvable
  `src` import paths. These checks are included in the CPU pre-commit partition.

### 3.3 Vocabulary and formatting

- Reuse domain terms accurately: particle, grid/node, material, body, wall, clump, element, control point, contact,
  degree of freedom, residual, gradient, Hessian, tangent, and increment are not interchangeable.
- Boolean names should read as conditions (`is_active`, `has_contact`, `enable_friction`) and preserve established
  public option spelling.
- Index names must reveal the indexed space when ambiguity is possible. Distinguish particle, node, body, material,
  contact, local, global, active, and compact indices.
- Include units or frames in names when the surrounding type cannot make them clear.
- Preserve exact spelling and capitalization of public dictionary keys, enumerated strings, environment variables,
  output fields, and restart data.
- Give each model one canonical, case-sensitive name. Reject unknown names and misspellings explicitly; do not
  add aliases, abbreviations, punctuation stripping, or case folding to make invalid model names work.
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

- A Python file's location follows its actual responsibility. Geometry/template preparation belongs to the
  generator, field serialization to field I/O, and generic host mathematics to `linalg.py`; engine state and
  solver algorithms stay with their engine. Inspect callers and an existing candidate before moving code.
  Reuse a template only when its input ownership, coordinate transforms, geometry representation, and validation
  match. Do not force surface-only affine preparation through an SDF template by adding flags or unused allocations.
  Record each actual move, reuse decision, and correction in the existing structure refactor record.
- Split a module when it contains multiple large implementations with independent responsibilities. Small,
  closely related parallel classes, records, and a base class with small variants may stay together.
  Class count or line count alone does not justify splitting. Preserve established filenames and import paths
  when ownership is already clear; a filename need not match its class name. Before moving code, record the
  concrete ownership or reuse problem and check whether keeping the existing module already solves it.
  Follow the agreed boundaries and migration ledger in [the structure refactor record](CODE_STRUCTURE_REFACTOR.md).
- Shared custom exception types belong in the dependency-free `src/utils/errors.py`, with small exception
  declarations grouped there by failure meaning. Import them directly from that module; do not define duplicate
  errors inside solvers or add forwarding modules. Before introducing or removing an error, trace its raise,
  catch, retry, rollback, and reporting sites. Keep useful errors; delete unused declarations without removing
  failure checks. Reuse a specific existing type when it matches the failure instead of raising bare
  `RuntimeError` for every case. Use standard `ValueError`/`TypeError` for invalid inputs, and catch specific
  failure types when recovery depends on their meaning. Preserve exception inheritance and retry/rollback
  behavior when consolidating definitions; do not collapse distinct failures into one generic error.
- Do not create or retain forwarding modules that only import and re-export symbols without implementation.
  When moving a class, update callers to import its actual implementation module and delete the obsolete module;
  do not replace it with a compatibility shim or add a package-level forwarding chain.
- Keep imports at module scope. A local import is appropriate only for an established lazy-loading boundary, optional
  dependency, or real circular dependency; `src/__init__.py` contains intentional lazy constructors.
- Preserve the public facade boundary. Examples should not depend on an internal class when a facade operation exists.
- A helper should encode a reusable concept, not merely shorten one call site. Keep linear one-use logic local when it
  remains readable.
- Remove forwarding functions that add no validation, state, lifecycle, numerical specialization, or necessary API
  boundary. Call the implementation directly. A one-line function is not automatically redundant: retain real
  polymorphic interfaces, device template adapters, profiler/callback contracts, and synchronization or failure checks.
- Do not turn unrelated shared functions into static methods on a configuration class just to give them a namespace.
  `StepSchedule` owns accepted-step scheduling; runtime callbacks belong in `RuntimeHook.py`, and shared numerical
  solver-control operations belong in `linear_solver/Utilities.py`. Keep actual formulas and validation centralized.
- Put model-specific helpers, especially Python configuration and preprocessing functions, on the owning
  class. For shared stateless helpers, search `src/utils/` first and reuse the existing owner: general Python
  numerical/scalar helpers such as `positive()` belong in `src/utils/linalg.py`.
- Decide helper ownership from its responsibility and callers, not from the source filename. A class-specific
  helper belongs on that class even if several consumers call it. Functions genuinely shared by multiple files
  within one subsystem belong in that subsystem's existing `Utilities.py`, or one such module if none exists.
  Do not mechanically split each class module into a companion `xxxUtilities.py`, or create a utility
  module merely to hold one or two class-specific helpers. In `src/utils/`, put class-specific helpers on the corresponding class
  and general Python numerical helpers in `linalg.py`; do not add another `xxxUtilities.py` layer.
  Check for an existing implementation before moving or adding a helper, and consolidate compatible duplicates
  without merging different validation contracts or numerical assumptions.
- Generic mathematics belongs to the existing shared numerical owner even if its first caller is a single solver.
  `src/utils/linalg.py` contains host Python mathematics for setup and preprocessing, with no Taichi functions
  or kernels. Shared device mathematics belongs in the existing functional categories under `src/utils/`,
  such as `MatrixFunction.py`, `VectorFunction.py`, `ScalarFunction.py`, and `ShapeFunctions.py`.
  Solver state, assembly, and algorithm orchestration stay in the owning solver; do not move them to shared
  utilities merely because they contain formulas. Preserve the execution boundary, precision, and validation.
  Taichi device helpers are module-level `@ti.func` functions or instance methods with `self`; do not wrap
  them in Python `@staticmethod`, which changes Taichi's argument specialization.
- Constitutive numerical computation in `src/` belongs in device functions and kernels, including energy,
  stress, tangent, and output-stress evaluation. Do not keep Python/NumPy reference evaluators such as host
  `evaluate()` implementations in production model files or move them into production utilities. Put independent
  Python formulas and finite-difference oracles under `tests/` and compare them with the device implementation.
  Python may validate configuration, prepare immutable inputs, dispatch kernels, and download output;
  downloading output must not recompute the constitutive response on the host.
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
- Preserve full nesting, spelling, casing, parameter-key aliases, defaults, units, and accepted value sets.
  Model selection follows the canonical-name rule above; do not retain model-name aliases when consolidating selectors.
- Use the existing dictionary access and configuration-validation utilities in that subsystem.
- Validate required keys, ranges, mutually exclusive settings, backend restrictions, and solver compatibility before
  memory allocation or kernel launch whenever possible.
- Error messages identify the full key path, invalid value, and accepted condition. Use `ValueError` for invalid user
  values and the matching shared exception type for an unavailable runtime state or unsupported execution path.
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
- `GEOTAICHI_REAL_DTYPE` is evaluated when `src/utils/TypeDefinition.py` is imported. Any process that changes it must
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
- Verify the initialized backend instead of inferring it from a facade request, `ti.gpu`, or startup messages.
  Record the effective backend in test reports; use [TAICHI_DEBUGGING.md](TAICHI_DEBUGGING.md) for runtime diagnosis.
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
- IPC uses BarrierIPC in standalone and coupled solvers. BVH sorting, tree
  construction and refitting are device work, including the initial build.
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

- Follow [GPU_OPTIMIZATION.md](GPU_OPTIMIZATION.md) for Taichi profiling,
  Nsight Systems/Compute, Metal tools, and reproducible before/after evidence.
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

- Use [NUMERICAL_VALIDATION.md](NUMERICAL_VALIDATION.md) to select derivative, assembly, VJP, and state-transition
  evidence for the actual solver branch.
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

For local commits, use the versioned hooks described in `tests/README.md`.
Ordinary messages select `pre-commit`, including VTU round trips. The exact
marker `[full test]` additionally selects `commit-validation` for NURBS geometry,
and small linear systems. Affected staged changes also select `commit-materials`
and/or `commit-contact`; shared implementation or dependency changes select both.
The JSON lists the complete law-test directories/globs, and unrelated changes
skip those larger groups. None of these selections scans the entire test tree.
Run the test that proves a specific fix separately when it is
outside these selections, and report backend or constitutive branches not tested.

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

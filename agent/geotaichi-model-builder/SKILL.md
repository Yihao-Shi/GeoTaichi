---
name: geotaichi-model-builder
description: Build, edit, diagnose, validate, document, execute, and monitor GeoTaichi simulation models through the supported geotaichi facades and optional MCP server. Use for natural-language-to-model requests, example scripts, MPM/DEM/MPDEM/CFDEM/FEM/IGA/IGAMPM workflow selection, configuration dictionaries, capacity planning, backend and compatibility checks, MCP capability discovery, isolated diagnostics, cooperative live task execution, logs/status/interruption, reduced smoke runs, production-run handoff, and docs/helper API synchronization.
---

# GeoTaichi Model Builder

Turn a physical problem into a source-backed GeoTaichi script and validation
record. Treat public methods, nested dictionaries, units, capacities, solver
compatibility, and numerical acceptance criteria as one model contract.

## Required outcome

Deliver:

1. a completed model contract with assumptions and unresolved physical choices;
2. a repository-relative Python script using the public `geotaichi` facade;
3. an evidence ledger for every non-default API choice;
4. syntax, configuration, reduced-run, and physics-validation results as available;
5. a handoff separating verified behavior from production-only checks.

Do not call a script complete merely because it parses or advances without an
exception.

## Source authority

Resolve conflicts in this order:

1. current public facade and consuming implementation under `geotaichi/` and
   `src/`;
2. the nearest maintained unit, integration, verification, or regression test;
3. one to three maintained examples using the same branch;
4. `docs/helper/geotaichi_api_reference.tex` and the linked theory section;
5. this skill and older prose.

Record a conflict instead of merging incompatible claims. Read
[knowledge-system.md](references/knowledge-system.md) before a broad API or
capability investigation.

## Workflow

### 1. Classify the request

Choose one primary operation:

- create a new model;
- modify an existing example;
- diagnose a failed or nonphysical model;
- update API/theory documentation;
- prepare or monitor a long production run.

For a new or materially changed model, create the contract before composing
the script. Read [model-contract.md](references/model-contract.md), copy
`assets/model-contract.json`, and validate it with:

```bash
python agent/geotaichi-model-builder/scripts/validate_model_contract.py model-contract.json
```

When handing the model to Blender, MCP, or the local job CLI, also read
[solver-job.md](references/solver-job.md) and copy the versioned
`assets/solver-job.json`/`assets/scene-manifest.json` boundaries.

Use `[AGENT]` for discovery, implementation, validation, and ordinary
diagnosis. Use `[USER ACTION REQUIRED]` only when a missing choice changes the
physical problem or requires authority outside the request.

### 2. Select the owning facade

Use the narrowest supported owner:

- continuum solid, fluid, or porous media: `MPM()`;
- spheres, clumps, rigid level sets, AffineBody, or LSMPM soft bodies: `DEM()`;
- Lagrangian DEM--MPM or fluid--particle coupling: `DEMPM(dem, mpm)`;
- volume or TRI3 membrane/cloth finite elements, including explicit DEM-law
  contact between FEM volume soft particles: `FEM()`;
- explicit DEM/LSDEM--deforming FEM surface coupling or elastic
  FEM--AffineBody IPC with pair-local lagged friction: `FEDEM(dem, fem)`;
- explicit DEM-law or elastic-FEM monolithic IPC MPM point--deforming FEM
  surface coupling, with supported elastic/associated-plastic Direct ULMPM:
  `FEMPM(fem, mpm)`;
- NURBS analysis: `IGA()`;
- IGA--MPM contact: `IGAMPM(iga, mpm)`.

Read exactly one primary workflow reference, then add coupling references only
when needed:

- [workflow-mpm.md](references/workflow-mpm.md)
- [workflow-dem.md](references/workflow-dem.md)
- [workflow-coupling.md](references/workflow-coupling.md)
- [workflow-fem.md](references/workflow-fem.md)
- [workflow-iga.md](references/workflow-iga.md)

Read [compatibility.md](references/compatibility.md) for any non-default solver,
backend, sparse/adaptive grid, direct backend, LSMPM, AffineBody, or coupling
choice.

### 3. Discover capabilities hierarchically

Use browse when the module or path is known:

```bash
python agent/geotaichi-model-builder/scripts/query_capabilities.py browse
python agent/geotaichi-model-builder/scripts/query_capabilities.py browse mpm/methods
python agent/geotaichi-model-builder/scripts/query_capabilities.py browse dem/keys/soft_levelset_advection_scheme
```

Use query when only a physical concept or approximate name is known:

```bash
python agent/geotaichi-model-builder/scripts/query_capabilities.py query "soft level set weno"
python agent/geotaichi-model-builder/scripts/query_capabilities.py query "incompressible pressure projection"
```

Regenerate the index after public API changes:

```bash
python agent/geotaichi-model-builder/scripts/build_capability_index.py \
  --output agent/geotaichi-model-builder/references/capability-index.json
```

After locating a candidate, trace it from facade to validator to consumer.
Never treat a fuzzy match as proof of support.

When the GeoTaichi MCP server is registered, prefer its equivalent tools so
responses stay in the active model-building session:

- `geotaichi_browse_capabilities` for a known category/path;
- `geotaichi_query_capabilities` when only a concept is known;
- `geotaichi_get_model_template` for the selected facade scaffold;
- `geotaichi_inspect_model` before runtime;
- `geotaichi_score_physics` for evidence-based solver/invariant scoring;
- `geotaichi_review_model` for the current validation-loop state and next action;
- `geotaichi_audit_api_docs` after public API changes.

The CLI scripts remain the dependency-free fallback and use the same shared
browse/query/inspection implementation.

### 4. Build an evidence ledger

For each non-default feature, record:

```text
physical intent -> facade method/full key path -> accepted value/default
-> prerequisites -> conflicts/backend limits -> source/test/example
```

Distinguish four evidence levels:

- `verified-runtime`: exercised in the current environment;
- `verified-source`: traced through current source and a maintained test;
- `documented`: present in helper docs or examples but not yet source-traced;
- `assumed`: unresolved and prohibited from the final production model.

Use exact aliases only when normalization is confirmed in source. Prefer the
canonical spelling in newly generated scripts.

### 5. Compose in dependency order

Build the smallest complete scene:

1. set import-time environment variables;
2. import `geotaichi` and call `init` once;
3. configure the owning facade or both sub-solvers;
4. configure implicit/semi-implicit or contact solver controls;
5. set timestep/output and allocate justified capacities;
6. add materials, elements/templates, regions, bodies, walls, and boundaries;
7. add every active contact/coupling pair property;
8. select output and run.

If the script creates or remeshes a FEM volume mesh, insert a preflight before
the production run. Record finite coordinates, connectivity bounds, consistent
orientation, positive element measures, minimum/mean element quality, the
explicit critical timestep when applicable, and one zero-load solver step.
Declare these as a required `mesh_quality` invariant in the model contract and
reject the candidate when the predeclared quality bound or first-step check
fails.

Use nearby examples for style and call order. Use the templates under `assets/`
only as lifecycle scaffolds; populate their dictionaries from the contract and
current source.

Select the asset matching the facade rather than relying on a generic name:

- `mpm_model_template.py` for `MPM`;
- `dem_model_template.py` for `DEM`, including LSDEM/LSMPM/AffineBody;
- `mpdem_model_template.py` for Lagrangian `DEMPM`/MPDEM;
- `fem_model_template.py` for volume or cloth `FEM`;
- `fedem_model_template.py` for explicit `FEDEM`/`DEMFEM`; adapt it to the
  documented implicit lifecycle for FEM--AffineBody IPC;
- `fempm_model_template.py` for `FEMPM`/`MPMFEM` (the canonical asset is
  explicit; use `examples/fempm/implicit_ipc_elastic_contact.py` for the
  Direct-MPM implicit IPC lifecycle);
- `iga_model_template.py` for pure `IGA` with a Python problem factory;
- `igampm_model_template.py` for coupled IGA--MPM;
- `coupled_model_template.py` only as the compatibility scaffold for older
  MPDEM/CFDEM generation flows.

Treat capacity as correctness. Estimate particles, bodies, grid/surface nodes,
constraints, and contact candidates with a stated margin. Never handle overflow
by silently dropping entities. Interpret `nParticlesPerCell` per coordinate
axis, so one full cell contains `nParticlesPerCell ** dimension` particles.

When the user asks how much GPU memory a Python model reserves, do not infer the
answer by reading one line manually and do not execute the simulation. Run:

```bash
python agent/geotaichi-model-builder/scripts/estimate_gpu_memory.py path/to/model.py
```

The command statically resolves the reachable `geotaichi.init`/`taichi.init`
call, project-style environment helper functions, Taichi's pinned default pool,
CPU/GPU backend, and `device_memory_GB` or `device_memory_fraction`. Pass the
target environment with `--env NAME=VALUE`; use `--platform linux` when the
script will run on a Linux GPU host rather than the current machine. A fraction
requires the target GPU capacity, supplied by `--gpu-total-gib` or detected from
NVML on the current host. Report `estimated_preallocated_pool_gib`, its
confidence, target platform, and all notes. This is the Taichi allocator pool,
not CUDA/graphics context overhead, external-library allocations, host RAM, or
the active bytes used inside the pool. If the expression or backend remains
unresolved, report that uncertainty instead of guessing.

### 6. Inspect before execution

Run the static inspector:

```bash
python agent/geotaichi-model-builder/scripts/inspect_example.py path/to/model.py \
  --contract path/to/model-contract.json
```

Resolve errors before runtime. Treat warnings as items requiring evidence, not
automatic failures. Run the API documentation audit when public methods or keys
change:

```bash
python agent/geotaichi-model-builder/scripts/audit_api_docs.py
```

### 7. Validate progressively

Read [validation-and-runs.md](references/validation-and-runs.md). Apply gates in
order:

1. parse/compile;
2. contract and API inspection;
3. GPU-pool estimation when capacity or device fit is in scope;
4. facade construction and configuration validation;
5. reduced run in `/private/tmp`;
6. numerical invariants and model-specific acceptance observable;
7. production resolution and duration.

After static inspection, follow
[agent-validation-loop.md](references/agent-validation-loop.md). Write a
`validation-evidence.json`, score it without changing the contract, and use the
returned decision to select the next bounded action. A SolverJob with a staged
contract automatically writes `physics-score.json`; a completed process with
missing evidence remains `insufficient_evidence`.

Use synchronous foreground commands for fast diagnostics. Use a tracked
background process or the repository's established cluster launcher for long
runs; preserve the command, process/job identity, log path, output path, start
time, and termination procedure. Do not infer completion from submission.

For a prepared model/contract/scene bundle, prefer
`geotaichi_validate_solver_job` and `geotaichi_submit_solver_job` with explicit
trusted-execution confirmation. Poll with
`geotaichi_check_task_status`, and continue each stdout/stderr stream from its
returned byte offset. Use `geotaichi_list_tasks` for persistent history and
`geotaichi_list_task_artifacts` for bounded artifact discovery; use
`geotaichi_interrupt_task` for process-group interruption. Use raw
`geotaichi_execute_task` only when no SolverJob is available. The task worker
sets internal task identity and diagnostics context in the process
environment. SolverJobs pass staged model-contract, SceneManifest, and output
paths explicitly through `--contract`, `--scene-manifest`, and `--output-dir`;
their task-path placeholders are resolved after staging. Bundled templates use
these named arguments and reuse `--output-dir` as `SavePath` only when the
model contract did not select one explicitly. Raw tasks and legacy SolverJobs
without path placeholders retain environment-path compatibility.

Use `geotaichi_execute_code` without `task_id` for short isolated diagnostics.
Pass a running task's ID to execute against its shared `__main__` namespace at
the next cooperative solver checkpoint. Bundled templates publish `model` and
`contract`; FEM/IGA also publish `problem`, and IGAMPM publishes `iga` and `mpm`.
For a custom loop, call `geotaichi_mcp.execution.live.checkpoint()` after a
complete numerical step and use `publish(...)` for function-local objects.
Never claim mid-kernel inspection: a live request waits until Taichi/native
code returns to Python. Treat live snippets as trusted state mutations and
revalidate the model after failures or timeouts.

Report effective architecture, precision, reduced parameters, skipped
backend-specific checks, and first causal failure. A Metal run reached through
`arch="cpu"` on Apple Silicon is not CPU validation.

### 8. Diagnose with bounded fallbacks

Classify before modifying:

- schema/call-order error;
- unsupported compatibility combination;
- insufficient capacity;
- units or scale error;
- stability/timestep error;
- nonlinear/linear convergence failure;
- backend or compilation restriction;
- nonphysical result or failed invariant.

Preserve the exact error and try the next evidence-backed fallback. Do not
change constitutive behavior, friction, damping, precision, tolerance,
resolution, or boundary conditions merely to make a run finish.

### 9. Hand off structurally

Follow [handoff-contract.md](references/handoff-contract.md). Report:

- `status`: `validated`, `partially_validated`, or `blocked`;
- artifacts and entry script;
- model assumptions and unresolved user choices;
- evidence ledger and validation commands;
- effective backend/precision;
- reduced versus production settings;
- warnings, skipped checks, and next action.

## Resource routing

- Read [knowledge-system.md](references/knowledge-system.md) for browse/query,
  source hierarchy, confidence, and index maintenance.
- Read [model-contract.md](references/model-contract.md) for required physical
  and numerical fields.
- Read [solver-job.md](references/solver-job.md) for Blender/CLI/MCP job and
  scene interchange.
- Read the one workflow file matching the selected facade.
- Read [compatibility.md](references/compatibility.md) whenever combining
  non-default features.
- Read [validation-and-runs.md](references/validation-and-runs.md) before any
  runtime or long task.
- Read [agent-validation-loop.md](references/agent-validation-loop.md) before
  repairing or accepting an agent-generated model.
- Read [handoff-contract.md](references/handoff-contract.md) before finalizing.
- Execute `scripts/self_test.py` after changing bundled Python tools.
- Read [`../geotaichi-mcp/README.md`](../geotaichi-mcp/README.md) when configuring MCP
  clients or maintaining execution/task lifecycle behavior.

## Guardrails

- Use public facades in user examples; import internals only for a documented
  narrow diagnostic.
- Keep units, frames, signs, array shapes, material IDs, body IDs, and pair
  properties explicit.
- Keep environment variables before the first GeoTaichi import.
- Separate configuration validity, numerical stability, and physical
  validation; passing one does not imply the others.
- Write smoke outputs and generated diagnostics outside tracked example
  directories.
- Preserve existing user changes and avoid unrelated source edits.

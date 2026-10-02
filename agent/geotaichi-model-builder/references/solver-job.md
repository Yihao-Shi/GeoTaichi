# SolverJob and SceneManifest

Use `assets/solver-job.json` as the versioned boundary between model-building,
Blender, the local CLI, and MCP. Use `assets/scene-manifest.json` for a
Blender-neutral scene description. Both documents currently use
`schema_version: 1`.

`SolverJob.model.entry_script` names the trusted Python entry point. Optional
`contract_path` and `scene.manifest_path` inputs are resolved relative to the
job file first, then the configured repository root. `execution.profile` must
be `trusted`, and `execution.requires_confirmation` must be `true`; neither the
document nor validation grants permission to execute it.

Use `model.arguments` for the entry script's explicit command-line interface.
Task-owned paths use whole-argument placeholders that are resolved only after
the immutable task directory has been created:

- `{model_contract}` for the staged `model-contract.json`;
- `{scene_manifest}` for the staged `scene-manifest.json`;
- `{output_directory}` for the task output directory;
- `{solver_job}` for the staged `solver-job.json`.

For example:

```json
"arguments": [
  "--contract", "{model_contract}",
  "--scene-manifest", "{scene_manifest}",
  "--output-dir", "{output_directory}",
  "--steps", "20"
]
```

Each placeholder must occupy one complete argument. Submission replaces the
placeholders with staged absolute paths and records those effective arguments
in the task-owned SolverJob. Put material, contact, discretization, and solver
configuration in the model contract rather than flattening the complete model
into CLI flags; reserve additional arguments for intentional run-time
overrides.

Validate before submission:

```bash
geotaichi-job --repo-root /path/to/GeoTaichi validate solver-job.json
```

Submit only after reviewing the resolved entry script and inputs:

```bash
geotaichi-job --repo-root /path/to/GeoTaichi submit \
  solver-job.json --confirm-trusted-execution
```

Submission copies the decoded documents into the task directory. Standard
contract-driven entry scripts accept `--contract`, `--scene-manifest`, and
`--output-dir`; the bundled templates do not read those paths from the
environment. `GEOTAICHI_MCP_TASK_ID`, `GEOTAICHI_MCP_TASK_DIR`,
`GEOTAICHI_DIAGNOSTICS_PATH`, and `GEOTAICHI_SOLVER_JOB` remain process/task
context. Raw tasks and older SolverJobs that omit a path placeholder continue
to receive the corresponding path environment variable for compatibility.
Terminal task and solver diagnostics are written to `diagnostics.json` and
exposed by the status/artifact APIs.

`SceneManifest` records a stable scene/object identity, coordinate convention,
unit scale, frame range, world transform, role, geometry counts, and a
deterministic geometry fingerprint. It intentionally does not encode solver
physics. Keep material laws, boundary conditions, timestep, capacities, and
validation observables in the model contract and entry script.

The repository-root `blender/` add-on is a thin client for this boundary. It
must not embed or initialize Taichi. Blender data stays on the main thread;
blocking CLI and filesystem work uses the add-on's single-flight background
service.

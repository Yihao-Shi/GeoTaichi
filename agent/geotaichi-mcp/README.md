# GeoTaichi MCP

GeoTaichi MCP exposes the repository's model-building knowledge and local
execution lifecycle through the Model Context Protocol. Its structure follows a clean Python subprocess per GeoTaichi model plus cooperative safe points in the Python solve loops instead of a proprietary GUI bridge. Numerical kernels and physical update rules are unchanged.

## Project layout

The MCP implementation now lives entirely under `agent/geotaichi-mcp`; the
repository root `pyproject.toml` remains the single packaging and dependency
definition for GeoTaichi and the MCP extra.

```text
agent/geotaichi-mcp/
├── README.md
└── src/geotaichi_mcp/
    ├── core/          # configuration, response contracts, shared resources
    ├── knowledge/     # browsing, API audit, inspection, read-only tool handlers
    ├── execution/     # task manager, worker, live queue, and tool handlers
    ├── tools.py       # safe/trusted tool profiles and FastMCP registration
    ├── job_cli.py     # Blender/MCP-shared SolverJob lifecycle CLI
    ├── server.py      # transport and command-line entry point
    └── __main__.py
```

Model-building instructions, generated capability data, templates, and helper
scripts stay in the sibling `agent/geotaichi-model-builder` Skill. Both the
Skill and MCP package consume those canonical resources, so there is no second
copy to drift. Repository-wide MCP tests stay in `tests/unit/mcp` with the
dependency-free layer and `tests/integration/mcp` for optional FastMCP/Taichi
runtime gates.

## Installation and startup

FastMCP requires Python 3.10 or newer. Install the optional integration from
the repository root:

```bash
python -m pip install -e '.[mcp]'
geotaichi-mcp --repo-root /path/to/GeoTaichi
```

The default stdio transport uses the trusted tool profile. HTTP and SSE default
to the safe profile and require a bearer token:

```bash
export GEOTAICHI_MCP_AUTH_TOKEN='replace-with-a-long-random-token'
geotaichi-mcp --transport http --host 127.0.0.1 --port 8000
```

Pass `--tool-profile trusted` only when the network client is allowed to launch
local Python. `--allow-unauthenticated-http` exists for explicit loopback-only
development. Keep network transports on localhost unless deployment adds its
own TLS and access-control boundary.

## Client configuration

Codex and Claude Code can register the same stdio command:

```bash
codex mcp add geotaichi -- geotaichi-mcp --repo-root /path/to/GeoTaichi
claude mcp add geotaichi -- geotaichi-mcp --repo-root /path/to/GeoTaichi
```

Generic MCP clients can use:

```json
{
  "mcpServers": {
    "geotaichi": {
      "command": "geotaichi-mcp",
      "args": ["--repo-root", "/path/to/GeoTaichi"]
    }
  }
}
```

Use the absolute executable path from the environment where GeoTaichi was
installed if the client does not inherit that environment's `PATH`.

## Public tools

| Tool | Purpose |
|---|---|
| `geotaichi_browse_capabilities` | Browse a known facade, methods, keys, or examples path. |
| `geotaichi_query_capabilities` | Search an unknown physical/API term and return browse paths. |
| `geotaichi_get_model_template` | Load the canonical MPM, DEM, MPDEM, CFDEM, FEM, FEDEM, FEMPM, IGA, or IGAMPM scaffold. |
| `geotaichi_inspect_model` | Statically check import order, lifecycle, public methods, keys, and contract alignment. |
| `geotaichi_score_physics` | Score run evidence against the contract and solver-specific physical invariants. |
| `geotaichi_review_model` | Combine static and physical review into the next bounded agent-loop action. |
| `geotaichi_audit_api_docs` | Check the helper API reference against the generated capability index. |
| `geotaichi_validate_solver_job` | Validate and resolve a versioned SolverJob and optional SceneManifest without execution. |
| `geotaichi_check_task_status` | Read persistent metadata and bounded stdout/stderr chunks. |
| `geotaichi_list_tasks` | Browse persistent task history with pagination. |
| `geotaichi_list_task_artifacts` | List task-owned contracts, diagnostics, and outputs with pagination. |
| `geotaichi_interrupt_task` | Signal the verified worker process group, with optional forced termination. |
| `geotaichi_submit_solver_job` | Submit a validated trusted SolverJob after explicit confirmation. |
| `geotaichi_execute_task` | Submit a raw Python model as an isolated background task (trusted profile). |
| `geotaichi_execute_code` | Run arbitrary code in a live task or clean process (trusted profile). |

All tools return either `{ "ok": true, "data": ... }` or
`{ "ok": false, "error": ... }`. Large responses are bounded to protect the
model context; capability paths and log byte offsets provide continuation.

The safe profile contains the eleven read-only tools plus interruption of an
already running task. The trusted profile additionally exposes SolverJob, raw
script, and arbitrary-code submission.

## MCP resources

Start with `geotaichi://index`, then load only the relevant resource:

- `geotaichi://capabilities/summary`
- `geotaichi://contracts/model-contract`
- `geotaichi://contracts/solver-job`
- `geotaichi://contracts/scene-manifest`
- `geotaichi://validation/rubric`
- `geotaichi://workflow/{name}`, where `name` is `mpm`, `dem`, `fem`, `iga`,
  or `coupling`

The capability summary is intentionally small; detailed lookup remains in the
paginated browse/query tools instead of placing the generated index into agent
context.

## SolverJob boundary

`SolverJob` is the shared execution envelope for CLI, Blender, and MCP. It
points to one Python entry script, optional model contract and SceneManifest,
declares requested artifacts, and always marks local script execution as
trusted and confirmation-required. Validate and submit from a shell with:

```bash
geotaichi-job --repo-root /path/to/GeoTaichi validate solver-job.json
geotaichi-job --repo-root /path/to/GeoTaichi submit \
  solver-job.json --confirm-trusted-execution
geotaichi-job --repo-root /path/to/GeoTaichi status TASK_ID
geotaichi-job --repo-root /path/to/GeoTaichi artifacts TASK_ID
```

At submission the decoded job, scene manifest, and model contract are copied
into the task directory. The worker receives only those staged paths, so later
edits to the source documents cannot change the running job.

Entry-script paths are explicit command-line arguments. `model.arguments`
supports `{model_contract}`, `{scene_manifest}`, `{output_directory}`, and
`{solver_job}` as whole-argument placeholders. The task manager resolves them
after staging and writes the effective absolute arguments into the task-owned
SolverJob. The canonical templates accept `--contract`, `--scene-manifest`, and
`--output-dir`; detailed physical and numerical configuration stays in the
structured model contract.

## Task workspace and logs

The default workspace is:

```text
.geotaichi-mcp/tasks/<task_id>/
├── task.json
├── solver-job.json       # when submitted through the canonical job boundary
├── scene-manifest.json   # optional staged input
├── model-contract.json   # optional staged input
├── diagnostics.json      # terminal task and solver failure summary
├── physics-score.json    # contract/evidence score for SolverJob tasks
├── stdout.log
├── stderr.log
├── control/
│   ├── requests/
│   ├── results/
│   └── cancelled/
└── output/
```

Override it with `--workspace` or `GEOTAICHI_MCP_WORKSPACE`. A worker always
exports this internal process context:

- `GEOTAICHI_MCP_TASK_ID`
- `GEOTAICHI_MCP_TASK_DIR`
- `GEOTAICHI_DIAGNOSTICS_PATH`
- `GEOTAICHI_SOLVER_JOB` (SolverJob submissions)

Raw tasks and legacy SolverJobs also receive `GEOTAICHI_MCP_OUTPUT_DIR`,
`GEOTAICHI_SCENE_MANIFEST`, and `GEOTAICHI_MODEL_CONTRACT` when the
corresponding path was not requested through a SolverJob placeholder. The
bundled templates do not consume these compatibility variables: they preserve
an explicit solver `SavePath`, otherwise use `--output-dir`, and require
`--contract`. In a task they still default `init(log=False)` from the internal
task identity so the worker owns stdout/stderr instead of the legacy daily
global log.

Task status is written by the worker itself, so completed/failed/interrupted
state survives an MCP server restart. Before interrupting a recovered task, the
manager verifies that the PID command line still contains the matching worker
and task ID.

## Execution boundary

Every background model has a separate Taichi runtime. This is required for
different dimensions, backends, precision, import-time environment variables,
and concurrent models.

`geotaichi_execute_code(code, task_id=...)` is a cooperative live REPL. The
worker executes the task script in a persistent `__main__` dictionary; queued
snippets run on that same thread and dictionary at the next complete solver
step. Bundled templates publish `model` and `contract`; IGA also publishes
`problem`, and IGAMPM publishes `iga` and `mpm`. An expression is returned in
`result`; an `exec` snippet can assign `_geotaichi_mcp_result` to return a
structured value. Snippet stdout/stderr is returned and is also preserved in
the task logs.

The built-in headless MPM, DEM, MPDEM/CFDEM, FEM, IGA, IGAMPM, FEMPM IPC,
FEM--AffineBody IPC, AffineBody, soft-particle, and real-time visualization
loops call the dependency-free
`src.utils.RuntimeHook.runtime_checkpoint` after a complete step. A custom
Python loop must call `geotaichi_mcp.execution.live.checkpoint()` itself.
Function-local objects can be exposed with
`geotaichi_mcp.execution.live.publish(model=model)`.

Every public solver facade exposes `diagnostics_snapshot()`. The worker checks
the conventional `model`, `solver`, `engine`, `mpm`, `dem`, `fem`, `iga`, and
coupling variable names and writes the bounded JSON-safe snapshot into the
task's `diagnostics.json`. Specialized implicit engines add retry, nonlinear,
linear, CCD, friction, and contact state; explicit/native engines retain the
common time/step/timestep/target schema.

When a staged model contract exists, the worker also evaluates a public
`validation_evidence` dictionary or `output/validation-evidence.json` against
the canonical solver-specific rubric and writes `physics-score.json`. Missing
measurements produce `insufficient_evidence`; task completion alone never
produces physical acceptance. `geotaichi_review_model` combines this score with
static inspection and returns a bounded repair/validation/handoff stage.

Without `task_id`, `geotaichi_execute_code` retains the original isolated
subprocess behavior. This is still the right mode for import-time diagnostics
or a different Taichi dimension/backend/precision.

The live boundary is deliberately cooperative. It cannot inspect a model in
the middle of a Taichi/native kernel, and an uninstrumented loop may time out
without executing the request. Expired queued requests are cancelled so they
do not mutate state later. A Python trace guard stops long pure-Python snippets;
native extension calls cannot be preempted safely and may return only after the
native call completes. Live code is trusted arbitrary code and can leave a
partially modified model if it fails.

Graceful interruption raises a termination request in the worker. A long
native or Taichi kernel may not observe it until control returns to Python.
Forced interruption can stop the process group but may leave partial output.

## Development validation

Run the dependency-free core tests and the existing Skill self-test:

```bash
python -m pytest --confcutdir=tests/unit/mcp tests/unit/mcp -q
python -m pytest --confcutdir=tests/integration/mcp tests/integration/mcp -q
python agent/geotaichi-model-builder/scripts/self_test.py
python /path/to/skill-creator/scripts/quick_validate.py agent/geotaichi-model-builder
```

The core documentation, inspection, task, worker, and cooperative-live tests do
not require importing Taichi or installing FastMCP. Transport integration and
real Taichi checkpoint smoke tests are separate optional-environment gates.

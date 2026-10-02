# Validation and Run Lifecycle

## Contents

1. Validation gates
2. Static and configuration checks
3. Reduced runs
4. Physical acceptance
5. Long-running tasks
6. Failure and interruption
7. Evidence record

## 1. Validation gates

Advance through gates in order:

1. `syntax`: Python parses and imports are structurally ordered.
2. `contract`: required physics, units, capacity, and acceptance fields exist.
3. `api`: facade methods/keys and compatibility are source-backed.
4. `configuration`: facade validation accepts the reduced scene.
5. `smoke`: a few reduced steps finish with finite state and expected counts.
6. `numerical`: invariants and benchmark observable pass.
7. `production`: intended backend, resolution, duration, and output pass.

Record each gate as pass, fail, skip, or not-run with a command and reason.

## 2. Static and configuration checks

Run:

```bash
python -m py_compile path/to/model.py
python agent/geotaichi-model-builder/scripts/validate_model_contract.py contract.json
python agent/geotaichi-model-builder/scripts/inspect_example.py model.py --contract contract.json
```

Then construct the smallest scene that reaches facade validation. Keep runtime
output out of the repository and record effective backend and precision.

## 3. Reduced runs

Reduce duration, output cadence, geometry repetition, and resolution only in a
separate validation copy or parameter set. Preserve the same physical branch,
material/contact law, units, boundary types, and solver family. State every
reduction in the handoff.

Check:

- finite particle/grid/body/contact state;
- expected entity counts and no capacity truncation;
- boundary and contact activation;
- monotonic time and output cadence;
- no hidden timestep replacement;
- no failed nonlinear solve or line search.

## 4. Physical acceptance

Passing without an exception is only smoke evidence. Validate the contract's
observable and at least one invariant:

- mass/volume or momentum balance;
- force/moment equilibrium;
- energy conservation/dissipation as appropriate;
- contact separation/gap;
- pressure-domain connectivity and fluid volume;
- convergence history or refinement trend;
- analytical/benchmark value.

Use tolerances derived from precision, scale, discretization, and solver
criteria. Do not widen a tolerance before diagnosing the discrepancy.

Encode these results using the evidence contract in
`agent-validation-loop.md`, then run:

```bash
python agent/geotaichi-model-builder/scripts/score_physics.py \
  model-contract.json --evidence validation-evidence.json
```

The score is diagnostic and gated. Required failures, non-finite state,
capacity overflow, or failed convergence cannot be offset by other points.
`insufficient_evidence` means the run has not established physical validity;
it does not mean the model passed or failed.

## 5. Long-running tasks

Use a foreground command for fast diagnostics. For a local run started by an
MCP client, submit with `geotaichi_execute_task`; use an established cluster
launcher when the run belongs to an external scheduler. Record a task
descriptor:

```json
{
  "entry_script": "relative/path.py",
  "command": "exact command",
  "process_or_job_id": "...",
  "start_time": "ISO-8601",
  "log_path": "/private/tmp/...",
  "output_path": "/private/tmp/...",
  "status": "pending|running|completed|failed|interrupted",
  "production_parameters": true
}
```

Poll bounded recent output without hiding the command's exit status. Do not
claim completion from successful submission. Check produced artifacts and the
final numerical acceptance result.

GeoTaichi MCP stores this descriptor as
`.geotaichi-mcp/tasks/<task_id>/task.json` by default, with `stdout.log`,
`stderr.log`, and `output/` beside it. Continue logs using the returned byte
offsets instead of repeatedly loading the complete files. The bundled model
templates set `SavePath` to `output/` only when the contract leaves it unset.
Set `GEOTAICHI_MCP_WORKSPACE` or pass `--workspace` to move the persistent task
root.

### Live inspection and control

Call `geotaichi_execute_code` with `task_id` to inspect or modify a running
model in its shared `__main__` namespace. The request runs only when the task
reaches a complete-step checkpoint. Bundled templates publish their configured
model objects before `run`; custom function-based scripts should call
`publish(model=model)` and custom loops should call `checkpoint()` explicitly.
Use the no-`task_id` form for an isolated diagnostic process.

Keep live snippets short and make state changes explicit. Python execution has
a deadline guard, but a Taichi or native extension call cannot be preempted
safely. A client timeout before a checkpoint cancels the queued request so it
does not execute later. A timeout after execution began may leave partial state;
check invariants and task logs before continuing.

## 6. Failure and interruption

Preserve the first causal error, exact command, backend, precision, and current
model contract. Classify the failure before changing anything. If interruption
is requested, use the run mechanism's graceful stop first, then confirm the
terminal state and artifact integrity. Report whether restart output is usable.

After a timeout or forced termination, assume state/output may be partial until
verified. Never resume blindly from the newest file without checking its time,
shape, fields, and completion marker.

The task manager validates that a recorded PID still belongs to the matching
GeoTaichi worker before signaling its process group. Graceful termination can
still be delayed inside a long native/Taichi call. Use forced interruption only
after accepting that the current output may be incomplete.

## 7. Evidence record

For every command record:

- purpose and gate;
- working directory and environment overrides;
- exact command;
- exit status;
- concise result plus full log path when large;
- effective backend/precision;
- skipped checks and why;
- artifact paths.

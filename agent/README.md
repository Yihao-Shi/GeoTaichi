# GeoTaichi Agent Workspace

Start with the repository [AGENTS.md](../AGENTS.md). This directory holds model
knowledge, MCP integration, and the detailed guides used when a task needs them:

| Path | Responsibility |
|---|---|
| [`AGENTS.md`](AGENTS.md) | Ownership rules for the integrations in this directory. |
| [`CLAUDE.md`](CLAUDE.md) | Claude entry pointing to the same integration instructions. |
| [`CODING_GUIDELINES.md`](CODING_GUIDELINES.md) | Detailed implementation, testing, documentation, and review standard. |
| [`CODEMAP.md`](CODEMAP.md) | Source paths connecting public solvers, shared numerical code, I/O, and integrations. |
| [`WORKFLOWS.md`](WORKFLOWS.md) | Task procedures and first checks for numerical, runtime, and performance failures. |
| [`SOLVER_INTEGRATION_PRINCIPLES.md`](SOLVER_INTEGRATION_PRINCIPLES.md) | Shared solver APIs, configuration, output lifecycle, and device/host boundaries. |
| [`GPU_OPTIMIZATION.md`](GPU_OPTIMIZATION.md) | Taichi profiling, Nsight Systems/Compute, Metal tools, and numerical/performance comparisons. |
| [`TAICHI_DEBUGGING.md`](TAICHI_DEBUGGING.md) | Runtime isolation, first-invalid-state diagnosis, active indices, parallel writes, and CUDA sanitizers. |
| [`NUMERICAL_VALIDATION.md`](NUMERICAL_VALIDATION.md) | Derivative oracles, physical/projected tangents, sparse assembly, VJPs, rollback, and restart evidence. |
| [`EXAMPLE_WRITER.md`](EXAMPLE_WRITER.md) | Compatibility entry point that routes older configurations to the model-builder Skill. |
| [`geotaichi-model-builder/`](geotaichi-model-builder/SKILL.md) | Canonical model-building Skill, references, capability index, templates, and maintenance scripts. |
| [`geotaichi-mcp/`](geotaichi-mcp/README.md) | Installable MCP source, versioned SolverJob boundary, resources, safe/trusted transports, persistent tasks, and cooperative live execution. |

The model-builder owns knowledge and reusable model assets. The MCP package
exposes that knowledge and execution lifecycle to trusted clients, reusing the
same assets instead of copying them. Root `pyproject.toml` remains the single
Python build definition; MCP tests remain with repository tests under
`tests/unit/mcp` and `tests/integration/mcp`.

Use the model-builder for model/example work and the MCP README for transport,
resource, or task-lifecycle maintenance. Keep one owner for each rule and asset.
For code maintenance, locate the owner with `CODEMAP.md`, then read only the
relevant procedure in `WORKFLOWS.md` and the detailed rules it links.

The Blender add-on lives directly in the repository-root `blender/` directory.
It exports the same `SceneManifest` and `SolverJob` documents and invokes the
shared `geotaichi-job` CLI instead of embedding a numerical runtime in Blender.

## Find the relevant evidence

| Question | Existing entry |
| --- | --- |
| Which material branches have maintained checks? | [Material coverage](../tests/unit/physics_model/materials/README.md) |
| Did a device path acquire host computation or redundant passes? | [Backend boundaries](SOLVER_INTEGRATION_PRINCIPLES.md#5-architecture-scope) and [performance workflow](GPU_OPTIMIZATION.md#turn-evidence-into-a-small-change) |
| What assumptions apply to finite-strain plasticity? | [Formulation and restrictions](FINITE_STRAIN_ASSOCIATED_PLASTICITY.md) |
| Why do dimension/precision-dependent tests interfere? | [Runtime isolation](../tests/README.md#runtime-and-dimension-isolation) and `tests/helpers/dimension_isolation.py` |
| What runs before a local commit? | [Commit checks](../tests/README.md#local-commit-checks) and the canonical partition JSON |

Keep dated audit and experiment reports under `research/`; this directory holds
reusable instructions. Audits describe the code and measurements at their
recorded date. Recheck the
owning source and test before applying an old conclusion to a new backend,
scene, or configuration. Keep handoffs short: changed behavior, reproducible
command/result, and unresolved assumptions; move durable facts into their owner.

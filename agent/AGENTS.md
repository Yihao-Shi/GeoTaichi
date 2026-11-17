# Working on GeoTaichi agent integrations

Read the [repository entry](../AGENTS.md) first. The detailed implementation and
documentation rules remain in [CODING_GUIDELINES.md](CODING_GUIDELINES.md);
[README.md](README.md) routes integration and diagnostic tasks.

Within this directory:

- `geotaichi-model-builder/` owns modeling knowledge, contracts, templates,
  generated capability indexes, and their maintenance scripts. Read its
  [instructions](geotaichi-model-builder/SKILL.md) before model/example work.
- `geotaichi-mcp/` exposes those shared resources and the task/live-execution
  lifecycle. Read its [README](geotaichi-mcp/README.md) before changing tools,
  transports, resources, or task persistence. Keep numerical loops independent
  of MCP; their common safe point is `src.utils.RuntimeHook`.
- Root `pyproject.toml` owns packaging. MCP tests remain under repository
  `tests/unit/mcp` and `tests/integration/mcp`.
- The root `blender/` add-on consumes the same versioned SolverJob/SceneManifest
  contracts. Keep Blender data access on its main thread and solver execution
  in external workers.

Compatibility entry files should route to their owner, not grow another copy
of the rules or model assets. Update the owning document when a durable rule
changes; use dated audits only as evidence for their recorded scope.

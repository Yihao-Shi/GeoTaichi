# GeoTaichi Agent Workspace

The `agent` directory contains two related but separately owned systems plus
repository-wide guidance:

| Path | Responsibility |
|---|---|
| [`AGENTS.md`](AGENTS.md) | Concise development-agent operating rules and repository map. |
| [`CLAUDE.md`](CLAUDE.md) | Concise development-agent operating rules and repository map for Claude Code. |
| [`CODING_GUIDELINES.md`](CODING_GUIDELINES.md) | Detailed implementation, testing, documentation, and review standard. |
| [`EXAMPLE_WRITER.md`](EXAMPLE_WRITER.md) | Compatibility entry point that routes older configurations to the model-builder Skill. |
| [`geotaichi-model-builder/`](geotaichi-model-builder/SKILL.md) | Canonical model-building Skill, references, capability index, templates, and maintenance scripts. |
| [`geotaichi-mcp/`](geotaichi-mcp/README.md) | Installable MCP source, versioned SolverJob boundary, resources, safe/trusted transports, persistent tasks, and cooperative live execution. |

The model-builder owns knowledge and reusable model assets. The MCP package
exposes that knowledge and execution lifecycle to trusted clients, reusing the
same assets instead of copying them. Root `pyproject.toml` remains the single
Python build definition; MCP tests remain with repository tests under
`tests/unit/mcp` and `tests/integration/mcp`.

Start with `CLAUDE.md` for repository code changes, with
`geotaichi-model-builder/SKILL.md` for model/example work, and with
`geotaichi-mcp/README.md` for MCP installation or maintenance.

The Blender add-on lives directly in the repository-root `blender/` directory.
It exports the same `SceneManifest` and `SolverJob` documents and invokes the
shared `geotaichi-job` CLI instead of embedding a numerical runtime in Blender.

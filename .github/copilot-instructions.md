# GeoTaichi coding instructions

Follow the repository [AGENTS.md](../AGENTS.md). It routes tasks to the owning
module, numerical constraints, and local validation commands.

Use the relevant sections of [coding guidelines](../agent/CODING_GUIDELINES.md)
for implementation and review, and the [model-builder instructions](../agent/geotaichi-model-builder/SKILL.md)
for models and examples. Read dependency versions and build configuration from
`pyproject.toml` and the actual native build directory.

Local commit checks are documented in the [test guide](../tests/README.md#local-commit-checks).
`[full test]` adds NURBS and small linear systems, plus complete material/contact
law groups when the staged diff affects their code, tests, or shared dependencies.
It does not run the entire test tree. Keep this entry short and update the owning
guide instead of duplicating its rules here.

---
name: geotaichi-example-writer
description: Compatibility entry point for agents that create GeoTaichi models, examples, or helper API documentation.
---

# GeoTaichi Model and Example Writer

This file is the stable entry point for older agent configurations. The
complete, executable project now lives at
[`agent/geotaichi-model-builder/SKILL.md`](geotaichi-model-builder/SKILL.md).
Read that file before creating or modifying a model, public example,
configuration dictionary, or `docs/helper` API entry.

The project adopts useful agent-oriented ideas: hierarchical browsing,
search when the path is unknown, explicit task ownership, bounded fallbacks,
structured results, and staged validation. GeoTaichi now also provides an
optional MCP server with five documentation/model tools and five diagnostic,
live-execution, and task-lifecycle tools.
GeoTaichi source and maintained tests remain authoritative.

For incompressible MPM examples, remember that `nParticlesPerCell` is specified
per coordinate axis. In this project's free-surface cases it commonly needs to
be about 5--10 per axis: 25--100 particles per full 2D cell and 125--1000 per
full 3D cell. This high occupancy helps avoid transiently unsupported pressure
cells that look like artificial gas pockets.
Validate occupancy and pressure connectivity for the actual geometry instead
of silently lowering PPC for speed. High 3D PPC can increase global atomic P2G
contention, but the measured implementation cost still decides the algorithm;
see `geotaichi-model-builder/references/workflow-mpm.md`. The L40S float64 tests
at 36 PPC in 2D and 216 PPC in 3D found the existing particle-centric atomic
P2G faster than both direct-shared and warp-reduced shared implementations.
Use atomic P2G for incompressible examples; do not enable a shared P2G mode or
allocate particle-binning workspace unless a new target-specific benchmark
first demonstrates an end-to-end improvement.

## Required model-building sequence

1. Convert the physical request into a model contract. Record dimension,
   coordinate assumption, units, geometry, discretization, materials,
   initial/boundary conditions, contact or coupling, time, capacities, outputs,
   and a quantitative validation observable.
2. Select the owning public facade: `MPM`, `DEM`, `DEMPM`, `FEM`, `FEDEM`,
   `FEMPM`, `IGA`, or `IGAMPM`.
3. Browse a known capability path or query an unknown term with the bundled
   capability tool.
4. Trace every non-default method/key through the facade, validator, consumer,
   nearest maintained test, and a nearby example.
5. Compose through the public `geotaichi` API in dependency order.
6. Compile, statically inspect, configure, run a reduced case, and check a
   physical invariant before restoring production size.
7. Hand off verified evidence separately from assumptions, skipped backends,
   unresolved physical choices, and production-only validation.

Use `[AGENT]` for discovery, implementation, ordinary diagnosis, and
validation. Use `[USER ACTION REQUIRED]` only when a missing choice changes the
physical problem or requires new authority.

## Fast entry commands

Copy and validate the contract:

```bash
cp agent/geotaichi-model-builder/assets/model-contract.json model-contract.json
python agent/geotaichi-model-builder/scripts/validate_model_contract.py model-contract.json
```

Browse or query current working-tree capabilities:

```bash
python agent/geotaichi-model-builder/scripts/query_capabilities.py browse
python agent/geotaichi-model-builder/scripts/query_capabilities.py browse mpm/methods
python agent/geotaichi-model-builder/scripts/query_capabilities.py query "soft levelset WENO"
```

Inspect a generated example and audit the helper API reference:

```bash
python agent/geotaichi-model-builder/scripts/inspect_example.py model.py --contract model-contract.json
python agent/geotaichi-model-builder/scripts/audit_api_docs.py --kind methods
```

Regenerate the capability index after API changes and run the project tests:

```bash
python agent/geotaichi-model-builder/scripts/build_capability_index.py \
  --output agent/geotaichi-model-builder/references/capability-index.json
python agent/geotaichi-model-builder/scripts/self_test.py
```

Run the same discovery and execution workflows through MCP after installing
the optional dependency:

```bash
python -m pip install -e '.[mcp]'
geotaichi-mcp --repo-root /path/to/GeoTaichi
```

Read [`geotaichi-mcp/README.md`](geotaichi-mcp/README.md) for client
registration, all tool contracts, per-task logs, and interruption behavior. Prefer
`geotaichi_execute_task` over ad hoc shell backgrounding for runs started by an
MCP client.

## Source authority

Resolve conflicts in this order:

1. current public facade and consuming implementation under `geotaichi/` and
   `src/`;
2. nearest maintained unit, integration, verification, or regression test;
3. one to three maintained examples using the same branch;
4. `docs/helper/geotaichi_api_reference.tex` and linked theory;
5. agent guidance and historical prose.

Never invent an API name, nested key, accepted value, compatibility rule, or
physical parameter. Do not declare a model validated merely because it parses
or advances without an exception.

## Project map

- `geotaichi-model-builder/SKILL.md`: canonical workflow and guardrails.
- `geotaichi-model-builder/references/`: model contract, module workflows,
  compatibility, validation, run lifecycle, and handoff schemas.
- `geotaichi-model-builder/references/capability-index.json`: generated facade,
  key, example, and source locator.
- `geotaichi-model-builder/scripts/`: index builder/query, contract validator,
  example inspector, documentation audit, and self-test.
- `geotaichi-model-builder/assets/`: contract plus explicit MPM, DEM, MPDEM,
  pure IGA, IGA-MPM, and legacy generic coupled lifecycle templates.
- `geotaichi-mcp/src/geotaichi_mcp/`: MCP entry layer, shared core,
  documentation/inspection knowledge layer, and persistent execution lifecycle.

The templates are lifecycle scaffolds, not ready-made physical models. Replace
their dictionaries using the contract and current source, justify capacities,
and attach a model-specific acceptance check.

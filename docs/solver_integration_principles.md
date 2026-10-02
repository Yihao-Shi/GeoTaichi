# GeoTaichi Solver Integration Principles

This document is the implementation contract for solver APIs, examples,
terminal output, file output, Blender integration, and agent/MCP integration.
It applies to every public solver exported by `src`: `DEM`, `MPM`, `FEM`,
`IGA`, `MPDEM`/`DEMPM`, `FEDEM`, `FEMPM`, and `IGAMPM`.
Special backends and schemes such as Direct MPM, implicit MPM, AffineBody,
LSDEM, LSMPM, explicit coupling, and IPC coupling follow the same contract.

## 1. Public configuration uses arguments first

- User-controlled example values use `argparse` command-line arguments.
- Python APIs receive values as explicit function arguments or dictionaries.
- A command-line argument takes precedence over any compatibility fallback.
- Environment variables are reserved for process-wide bootstrap state that must
  exist before Python imports, secrets, or CI/container integration. They must
  not be the default interface for ordinary model, solver, path, time-step,
  material, or output options.
- If a legacy environment fallback must remain, expose the same setting as an
  argument, document the fallback, and do not let it silently override the
  argument.
- Blender properties and operators pass explicit values to the runner. The
  add-on must not require users to prepare shell environment variables.

## 2. Paths are independent of the working directory

- Input assets are resolved from `Path(__file__).resolve()` or from an explicit
  user argument.
- Output directories are exposed as arguments such as `--output-dir`.
- Examples must not assume that the terminal was opened in the repository root.
- Blender add-on changes stay under `blender/`; shared source changes belong in
  `src/` only when they are required by every caller.

## 3. Every solver follows one terminal-output contract

The standard lifecycle is:

1. basic configuration;
2. engine/backend information when applicable;
3. material information;
4. solver information;
5. memory information;
6. neighbor-search information when the configured path uses it;
7. simulation start;
8. save events, including frame zero;
9. compilation time, per-module accumulated timings, progress, and completion
   diagnostics.

The shared formatting functions in `src/utils/SolverConsole.py` are the source
of truth. Information headings use `<solver> <section>`, and runtime events use
the canonical public solver name rather than a private engine class name.

Each public facade provides these callable methods:

- `print_basic_simulation_info()`
- `print_solver_info()`
- `print_memory_info()`
- `print_neighbor_search_info()`

`log=False` may suppress optional welcome/configuration descriptions when a
caller explicitly requests a quiet setup. It does not suppress the runtime
start line or save-event lines. With setup logging enabled, `Save Path` appears
once in Solver Information; it is not repeated by runtime save events.

### Materials

- The first data line in a material section is the readable constitutive-model
  name; the material ID follows it.
- Do not expose an internal numeric model ID as the user-facing model name.
- The heading identifies the owning subsystem, for example `FEM Constitutive
  Model Information`, so two genuinely different materials are not presented
  as accidental duplicates.
- Rigid DEM contact material, AffineBody rigidity, and LSMPM finite-strain
  material are distinct concepts and must be labelled accordingly.
- Printed values are effective runtime values, not unrelated dataclass defaults.

### Neighbor search

- A solver prints a neighbor-search section only when the configured algorithm
  actually uses neighbor/contact search. Do not print placeholder fields for
  FEM, IGA, or grid-based MPM paths that do not use it.
- Contact solvers print the selected method, runtime search object when built,
  Verlet settings, coordination limits, and pair capacities that apply.

### Coupled save events

- A coupled output instant prints exactly one top-level save event (`MPDEM`,
  `FEDEM`, `FEMPM`, or `IGAMPM`).
- Child recorders still write their DEM/MPM/FEM/IGA files, but do so quietly.
- The top-level save line contains only the step, save number, and simulation
  time, using the same field order as MPM. `Save Path` is printed once in
  Solver Information and is not repeated for every frame. Private branch names
  such as `Direct`, `AffineBody IPC`, or an engine class must not create a
  second public save format.
- All VTK/VTU files are stored under `<Save Path>/vtks/`, including FEM,
  AffineBody, Direct MPM, and IGA outputs. Do not introduce solver-specific
  VTK directories such as `fems/`.

### Compilation and module timings

- The first physical step prints `Compiling first ... ...` and
  `Compiling time = ...`, matching the native MPM lifecycle.
- Timers cover named solver modules such as reset/setup, neighbor search,
  contact resolve, system assembly, linear solve, line search, integration,
  postprocessing, and output when those modules are present.
- Accumulated time records are printed with output events and use the shared
  `Timer` implementation. Do not print fake or permanently zero modules.
- Coupled solvers own the top-level timer; child recorders remain quiet so the
  same work is not reported twice.

## 4. Frame zero is a real output frame

- A fresh run writes frame zero before the first state update when output is
  enabled.
- Standalone and coupled solvers both follow this rule, including implicit FEM,
  AffineBody IPC, Direct MPM, IGA, and all coupling variants.
- Coupled frame zero includes every enabled child representation. For example,
  a FEM coupling writes `FEM000000.vtu` as part of the top-level frame-zero
  event.
- Restart logic must preserve counters and must not accidentally overwrite an
  existing restart frame.
- Tests check both scheduling and the actual `000000` output file.

## 5. Architecture scope

Reusable architecture learned from external contact-solver projects may be
adopted when it fits GeoTaichi's public APIs and all solver families. The
production device contract is:

- time-step physics, contact search/assembly, nonlinear updates, trajectory
  replay, linear solves, and parameter/state VJPs stay in Taichi fields and
  kernels;
- NumPy/SciPy may be used only for one-time preprocessing, input/seed upload,
  scalar diagnostics, output download, explicitly labelled test oracles, and
  a linear solve explicitly configured as `linear_solver="Scipy"`;
- selecting SciPy transfers only that assembled system and solution; contact,
  material, replay, state-update, and VJP work remains on device, and no
  device solver may silently fall back to NumPy or SciPy.

The current architecture plan deliberately excludes the following until a
new, explicit design decision is made:

- Cubic interpolation plus dynamic stiffness;
- Schwarz solvers;

Do not introduce those features indirectly while refactoring neighboring code.

## 6. Blender and agent/MCP integration

- Blender UI code, operators, manifests, and packaging live in `blender/`.
- Blender exposes explicit scene properties and operator arguments and produces
  the same model contract used by terminal workflows.
- MCP/agent tools are adapters around documented public operations; they do not
  become a second solver configuration system.
- CLI, Blender, and MCP invocations must converge on the same solver arguments,
  path rules, validation, console lifecycle, and output layout.

## 7. Verification for every change

- Audit the complete public-solver matrix, not only the solver that revealed a
  bug.
- Add or update regression tests for the shared console contract.
- Test special branches separately when they own scheduling or recording logic.
- Check frame zero, canonical save lines, quiet child recorders, readable
  material names, memory information, and neighbor-search information.
- Run `git diff --check`, syntax compilation, focused unit tests, and relevant
  integration tests before handoff.

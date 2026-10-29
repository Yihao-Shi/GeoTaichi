# GeoTaichi tests

GeoTaichi's maintained tests are split by purpose so a change can be checked
without launching unrelated simulations.

## Layout

```text
tests/
├── unit/          # one component or numerical law
├── integration/   # two or more production components
├── verification/  # analytical or published-reference comparisons
├── regression/    # minimal reproductions of fixed defects
├── benchmarks/    # opt-in performance/scalability runs
├── helpers/       # shared test-only drivers
├── data/          # small immutable fixtures
└── testing/       # named partition definitions and runner
```

The former `codeTesting`, `pythonTesting`, `demTesting`, `debug`, and
`sympy_generate` buckets have been removed. Their useful behavior is now one
of:

- a deterministic test under the five maintained layers above;
- an explicit diagnostic or derivation under `tools/`;
- a manually launched physical scenario under `examples/`;
- deleted when it was broken, duplicated, GUI-only scratch code, or depended
  on missing machine-local data.

Large examples are not correctness tests. A feature needs a small unit oracle
for its local laws and, when appropriate, a separate integration or
verification case for the complete solver path.

## Reproducibility rules

Collected test modules must be safe to import. They may not initialize Taichi,
allocate Taichi fields, launch a solver or GUI, write output, delete paths, or
embed a machine-specific absolute path at module scope. Runtime construction
belongs in a fixture or test function. New and modified tests put output under
`tmp_path` (or a `tmp_path_factory` directory shared deliberately by a fixture)
and random cases use an explicit local seed.

### Runtime and dimension isolation

The ordinary parent pytest process is configured as the canonical 2D process
before test modules are imported. This is necessary because legacy Taichi
annotations read GeoTaichi's dimension globals at import time; resetting the
Taichi runtime later cannot change an already specialized annotation.

A 3D test whose imports depend on those globals uses
`@pytest.mark.isolated_dimension(3)`. Its selected pytest node is executed in a
clean child interpreter that sets the classic MPM, direct MPM, IGA, and IGA-MPM
dimension sources before importing the module. The child inherits the
environment and receives the selected `--taichi-arch`, `--taichi-fp`, and
`--run-benchmarks` options. Other third-party pytest command-line options are
not forwarded automatically.

The module-scoped autouse fixture resets Taichi at each test-module boundary,
allowing kernels within one module to share compilation while preventing their
runtime from leaking into the next module. A function-scoped autouse fixture
restores mutable scalar GeoTaichi configuration and `GEOTAICHI_*`/`GT_*`
environment variables after every test. Tests that request `taichi_runtime`
additionally receive a fresh runtime for that test. The material suite has a
dedicated function-scoped CPU/f64 fixture because constitutive kernels are
specialized at compile time.

`tests/unit/testing/test_suite_hygiene.py` enforces these structural rules,
rejects zero-node `test_*.py` files, prevents `xfail` from masking
implementation defects, and rejects tracked AppleDouble, `.DS_Store`,
`__pycache__`, and bytecode artifacts. Generated untracked `__pycache__`
directories are allowed because Python creates them during an ordinary test
run.

## Test layers and oracles

Use the narrowest layer that proves the behavior:

| Layer | Expected evidence |
| --- | --- |
| Unit | exact scalar/vector law, invariant, analytical result, finite-difference derivative, or small test oracle |
| Integration | agreement between production components, assembly backends, lifecycle stages, or coupled solvers |
| Verification | analytical trajectory/load, published reference, bundled IPC data, or convergence behavior |
| Regression | the smallest input that reproduces a fixed bug |
| Benchmark | timing, allocation, or scalability only; never a correctness gate |

Material tests use a per-model contract: construction and
validation, state schema, zero increment, elastic and every inelastic branch,
history evolution, analytical or finite-difference tangent, invariants, and
batch-kernel agreement. Contact tests separately cover primitives, barrier
laws, PSD projection, friction gradients/Hessians, stick-slip transitions,
assembly, line search, and end-to-end motion.

See the [material coverage matrix](unit/physics_model/materials/README.md) for
the supported FEM, IGA and MPM models, their solver-entry tests, and remaining
constitutive branch coverage limits.

Lagged IPC is a conservative projected-Newton solve and uses a
monotone-potential backtracking rule. Fully implicit friction is
nonconservative and instead uses residual-merit Armijo backtracking; tests
keep these two globalization contracts separate.

## Markers and named partitions

### Local commit checks

Enable the versioned Git hooks once per checkout, using the Python environment
that already has GeoTaichi's dependencies, pytest, and Black installed:

```bash
chmod +x .githooks/pre-commit .githooks/commit-msg
git config --local core.hooksPath .githooks
git config --local geotaichi.python "$(command -v python)"
```

Every commit formats staged Python files, checks their syntax, and tests an
isolated copy of the staged content using CPU/f64. Partially staged Python files are rejected
before formatting; unstaged work and the active environment are preserved.

- Ordinary messages run the `pre-commit` partition: test-runner contracts,
  analytical quadrature, ENO/WENO accuracy, and two CPU IPC barrier/friction
  checks, plus small VTU write/read round trips. This gate has a 60-second
  test timeout.
- A message containing the exact text `[full test]` also runs the explicit
  `commit-validation` partition: NURBS basis/refinement/projection checks,
  and 4-by-4 linear-system assembly/solution checks. The two small partitions
  each have a 60-second test timeout. Comment lines do not request the extended run.
- For `[full test]`, the staged diff against `HEAD` also selects complete law
  groups: `commit-materials` covers shared constitutive oracles, material VJPs,
  and the solver material-entry checks; `commit-contact` covers shared IPC/DEM
  laws, unit contact consumers, and the two listed IPC derivative/reference files.
  These larger groups have no 60-second limit.
  Their directories and globs are maintained in `test_partitions.json` and
  include new tests placed in those locations automatically.

Law groups are selected independently. Constitutive-law source
and material-test changes trigger `commit-materials`; contact-law, contact
detection, contact-test, and bundled IPC-fixture changes trigger `commit-contact`.
Material managers, other `src/`, `geotaichi/`, test infrastructure, vendored
code, and dependency/build configuration changes conservatively trigger both. Markdown/RST-only
changes and unrelated tools do not trigger either law group. Renames are compared
as deletion plus addition so both old and new owners are considered. Logs name
every selected or skipped group. This is conservative path-based selection,
not a complete dynamic import/dependency graph; update the selection rules when
moving source ownership or adding another external dependency manifest.

The gate validates the index relative to `HEAD`; if amending or rewriting an
existing commit, run the required law groups explicitly for changes inherited
from that commit as well. Available-device tests keep their capability skips;
a CPU-only run does not establish GPU correctness.

Any failed check blocks the commit. `GEOTAICHI_PYTHON` can override the locally
configured interpreter. To run a suite manually:

```bash
python tests/testing/run_partition.py pre-commit -q --taichi-arch cpu
python tests/testing/run_partition.py commit-validation -q --taichi-arch cpu
python tests/testing/run_partition.py commit-materials -q --taichi-arch cpu
python tests/testing/run_partition.py commit-contact -q --taichi-arch cpu
git commit -m "Fix contact regression [full test]"
```

The commit hooks use local resources and do not configure CI/CD or install
dependencies. Keep the ordinary gate small. Long simulations, benchmarks, and
the full offline suite remain separate manual checks. Full box-sliding trajectories
and solver line-search scenarios stay in the manual verification suite.
Research-result audits
that import ignored `research/` scripts, including `test_lsmpm_contact_work_audit.py`,
remain manual as well; neither law group means
running every test under `tests/`.

Layer markers are inferred from the maintained directory. Stable domain and
solver markers are inferred from paths and filenames. Backend markers are
attached only when a path, test node, or `assemble_type` parameter identifies
the representation unambiguously; a filename containing `gpu` does not prove
which Taichi backend actually ran. Unknown markers are errors.

The `serial` marker is a selection contract, not an xdist scheduler. Run the
`serial` partition without concurrent pytest workers. Capability partitions
such as `metal`, `vulkan`, `pinn`, or `requires-network` may select no nodes on
a checkout that does not yet provide such a maintained test.

List and run partitions with:

```bash
python tests/testing/run_partition.py --list
python tests/testing/run_partition.py required -q
python tests/testing/run_partition.py materials -q
python tests/testing/run_partition.py ipc -q
python tests/testing/run_partition.py iga-mpm -k friction -x
python tests/testing/run_partition.py hash-triplet -q
python tests/testing/run_partition.py coupling -q
python -m pytest -m fedem
python -m pytest -m fempm
python tests/testing/run_partition.py gallery -q
python tests/testing/run_partition.py serial -q
```

For a long unattended final check, use the offline gate.  It validates the
local IPC data package before running any test, saves one log per
step, stops on the first failure, and always writes `summary.tsv`:

```bash
tests/testing/run_offline_validation.sh \
    --clean-cache
```

The default mode runs the remaining IPC and required gates.  Add `--targeted`
to repeat the affected SoftParticle/MPM, bounded-static-expansion, and
assembly regressions.  Use `--full` instead for one complete correctness run;
it already contains the smaller partitions and therefore does not repeat
them.  The script never downloads anything.  The IPC fixture is
bundled under `tests/data/ipc_toolkit`; `--ipc-data DIR` is only needed to
audit an alternate package containing the same frozen archive and provenance.

Direct pytest expressions also work:

```bash
python -m pytest -m "unit and materials"
python -m pytest -m "regression and ipc and not slow"
python -m pytest -m "mpm and assembly"
```

Scalability tests are part of default collection so collection regressions are
visible, but they receive an execution-time skip unless `--run-benchmarks` is
passed. Run them explicitly with:

```bash
python tests/testing/run_partition.py benchmarks
# or
python -m pytest tests/benchmarks --run-benchmarks
```

Tests requiring optional external data or optional tooling state the exact
environment variable or extra dependency in their skip reason. Symbolic
derivation tools use the `gen` extra (`pip install -e '.[gen]'`).

## Implicit assembly contracts

All implicit backends represent the full coupled Jacobian/operator. “Jacobi”
describes the preconditioner, not a matrix containing only its diagonal.

| Backend | Representation | Duplicate handling | Current preconditioner |
| --- | --- | --- | --- |
| MatrixFree | no global matrix; particle-local blocks are evaluated by `matvec` | gather/scatter on every product | scalar diagonal Jacobi |
| COO | scalar `(row, column, value)` entries | sum during COO product or sparse conversion | scalar diagonal Jacobi |
| HashTriplet | dense node-block triplets reduced to a unique block pattern | device hash reduction on CUDA | full node-block Jacobi for `dim > 1` |

`BuildTriplet(dim=1)` naturally reduces block Jacobi to scalar Jacobi. IGA
supports HashTriplet and COO but not MatrixFree. Affine's legacy
`MatrixFree` selection currently routes through its COO implementation.
Affine production solves use Taichi fields on CPU, Metal, and CUDA. Lagged
contact uses PCG; fully implicit friction uses nonsymmetric BiCGSTAB. The
mass-whitened dense eigensolve is a test-only oracle in
`tests/helpers/abd_engine_reference.py`, not a selectable production backend.
Affine body principal blocks retain all 12 DOFs in the device preconditioner;
coupled systems use those blocks for the affine prefix and node blocks for
other unknowns. See `src/linear_solver/README.md` for the current solver contract.
`tests/unit/linear_solver/test_assembly_backends.py` guards operator,
solution, duplicate-reduction, and preconditioner equivalence.

## Adding a test

Keep particle/grid counts and step counts minimal. Assert the physical or
mathematical result rather than only “runs without error.” Choose tolerances
from precision, conditioning, and discretization error; do not widen them to
hide a defect. Register a new marker only when it defines a reusable
partition, then update `pyproject.toml` and
`tests/testing/test_partitions.json` together.

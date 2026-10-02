# GeoTaichi Model Contract

## Contents

1. Contract rule
2. Required fields
3. Units and dimensional assumptions
4. Discretization and capacity
5. Validation observable
6. Unknowns and user decisions

## 1. Contract rule

Create the contract before writing a materially new model. It captures the
physical problem independently of the final API dictionaries and prevents an
agent from silently selecting material, contact, drainage, restraint, or units.

Copy `assets/model-contract.json` and validate it with
`scripts/validate_model_contract.py`.

## 2. Required fields

- `schema_version`: currently `1`.
- `title`: human-readable model name.
- `module`: `mpm`, `dem`, `mpdem`, `cfdem`, `fem`, `fedem`, `fempm`, `iga`,
  or `igampm`.
- `dimension`: `2` or `3`.
- `coordinate_assumption`: `3d`, `plane_strain`, `plane_stress`, or another
  explicitly supported assumption.
- `units`: base length, mass, and time units.
- `physics`: requested processes and governing assumptions.
- `geometry`: dimensions, coordinate frame, and source geometry.
- `discretization`: cell/element/particle/template resolution.
- `materials`: every material model and calibrated parameter with units.
- `initial_conditions`: stress, pressure, velocity, state, and gravity.
- `boundary_conditions`: selections, constrained components, loads, drainage,
  and time dependence.
- `contact_or_coupling`: law, active pairs, parameters, and transfer model.
- `time`: timestep policy, duration, save interval, and adaptivity.
- `capacities`: estimated active counts, allocation values, and safety margin.
- `outputs`: fields, cadence, path policy, and restart needs.
- `validation`: observable, expected result/range, tolerance, evidence, and
  pre-run invariant criteria.
- `assumptions`: explicit agent assumptions.
- `unresolved`: choices that prevent a production claim.
- `api`: the source-backed public dictionaries used by a template/driver.

Use `api.environment` only for values that must exist before GeoTaichi modules
are imported, such as `GEOTAICHI_REAL_DTYPE`. Put ordinary runtime, solver,
material, contact, output, and capacity values in their structured `api`
dictionaries. SolverJob named arguments select the staged contract, scene, and
output location; they do not replace the model schema.

For FEM or pure IGA, `api.problem_factory` names a Python module and optional
function that constructs non-JSON mesh/primitive, material, and boundary
objects. The default function names used by the templates are
`build_fem_problem(contract)` and `build_iga_problem(contract)`.

For explicit IGAMPM, set `api.configuration.contact_model` and
`api.contact_model.contact_model` to `Linear` or `HertzMindlin`, and put each
material/patch law in `api.properties` with `MPMmaterial`, `IGAbody`, and a
`property` dictionary. The template constructs the coupling before MPM memory
allocation so it can select Lagrangian coupling storage.

## 3. Units and dimensional assumptions

Use one consistent unit system. For every dimensional parameter record either
the unit or a derivation from the base units. Check especially:

- density and body forces;
- Young/bulk/shear moduli and pressure;
- viscosity, permeability, and drag coefficients;
- contact stiffness, damping, and activation distance;
- geometry, grid spacing, particle radius, velocity, and timestep.

Do not infer plane strain versus plane stress. GeoTaichi's supported branch and
constitutive representation must agree with the contract.

## 4. Discretization and capacity

Record both physical resolution and allocated maximums. Explain estimates for:

- MPM particles (`nParticlesPerCell` is per axis);
- active background or soft-grid nodes;
- DEM bodies, spheres, clumps, templates, and surface/SDF nodes;
- constraints and contact coordination/list sizes;
- coupled particle/body and wall pair capacities;
- adaptive particle splitting or compact template support.

Capacity must exceed the justified active estimate by a stated margin. A large
unexplained number is not a capacity plan.

## 5. Validation observable

Choose at least one result that can falsify the model, for example:

- analytical displacement, frequency, pressure, or velocity;
- static force/moment balance;
- mass, momentum, or energy balance;
- contact gap and complementarity behavior;
- convergence order under refinement;
- benchmark runout, settlement, pore pressure, or drag;
- finite-difference residual/tangent agreement.

Specify an expected value or bounded trend and a tolerance justified by
precision, scale, conditioning, and discretization error.

Before execution, declare every solver-specific invariant used by the scorer in
`validation.invariants`. Each entry contains a rubric `kind`, machine-readable
`expectation`, machine-readable `tolerance`, and the physical/numerical `basis`
for that tolerance. Evidence files must repeat the same expected value and
tolerance exactly. This binding prevents an agent from changing acceptance
criteria after seeing a failed run.

When the agent generates or remeshes a FEM volume mesh, it must also declare a
required `mesh_quality` invariant. Evidence must contain finite coordinates,
valid connectivity, consistent positive element measures, the minimum and mean
element-quality metric with its predeclared lower bound, and a successful
zero-load first-step stability check. A readable mesh or a visually plausible
surface is not sufficient. A failed mesh-quality check is a hard failure: the
agent must regenerate, smooth, or locally refine the mesh before a production
run and may not compensate by widening a physics tolerance.

## 6. Unknowns and user decisions

Put unresolved physical choices in `unresolved`. Stop for user input when two
plausible values change the modeled problem, including:

- material calibration or constitutive family;
- contact law/friction;
- drained versus undrained behavior;
- load versus displacement control;
- plane strain versus plane stress;
- geometry or unit ambiguity.

API discovery, file lookup, capacity arithmetic, syntax errors, and ordinary
debugging remain agent-owned work.

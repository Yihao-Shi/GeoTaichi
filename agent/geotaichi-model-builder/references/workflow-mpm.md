# MPM Workflow Profile

## Contents

1. Ownership and call order
2. Branch selection
3. Incompressible fluid
4. Implicit solid
5. Resolution and capacity
6. Validation

## 1. Ownership and call order

Use `geotaichi.MPM`. Follow this order unless the closest maintained example
for the selected branch proves otherwise:

```text
environment -> import/init -> MPM -> set_configuration
-> implicit/semi-implicit parameters -> set_solver -> memory_allocate
-> add_material -> add_element -> add_region -> add_body
-> add_boundary_condition -> select_save_data -> run
```

Use `create_body`, `create_ground`, and `add_ground` only with the direct MPM
backend. Native MPM rejects those operations.

## 2. Branch selection

Trace these choices together:

- `solver_type`: explicit, implicit, or semi-implicit family;
- `material_type`: solid, fluid, or two-phase family;
- `discretization`: FEM or supported FDM branch;
- `configuration`: ULMPM or TLMPM where supported;
- `ElementType` and shape function;
- sparse/adaptive grid and stabilization switches;
- assembly type and linear solver.

Do not combine similarly named branches by analogy. Run
`Simulation.validate_configuration` through facade construction before a long
run.

## 3. Incompressible fluid

The current incompressible FDM route uses implicit fluid configuration and a
staggered grid. Treat these as a coupled set, not independent toggles. Important
checks include:

- free-surface `fluid_level_set` and optional density projection;
- `fluid_domain_volume_fraction` occupancy threshold;
- solid SDF cut-cell settings and external SDF/IBM callbacks;
- pressure solver, multilevel controls, and pressure boundary conditions;
- delayed advection and sufficient particles per cell.

Incompressible MPM uses the particle-centric atomic P2G path on every backend;
there is no shared-P2G runtime switch. L40S float64 tests measured atomic P2G
at 0.305 ms for 19,293
particles in 2D (36 PPC) and 0.980 ms for 110,592 particles in 3D (216 PPC).
The corresponding warp-shared totals, including particle binning, were 0.992
ms and 2.556 ms and required additional workspace. Do not trade the atomic
path for a shared reduction based only on atomic-contention intuition: require
both a representative high-PPC benchmark and an end-to-end speedup first.

## 4. Implicit solid

Verify:

- Newmark parameters, quasi-static choice, nonlinear tolerances, and maximum
  iterations;
- `MatrixFree`, `COO`, or `HashTriplet` assembly support;
- strong versus penalty Dirichlet behavior for the selected assembly path;
- matrix symmetry only when the material/contact tangent is symmetric;
- relative versus absolute Krylov stopping criteria;
- elastic or symmetrized matrix-free tangent approximations are deliberate.

Do not use a tangent approximation to hide a constitutive derivative error.

## 5. Resolution and capacity

Interpret `nParticlesPerCell` per coordinate axis. A value `q` creates up to
`q ** dimension` particles in a full cell. For free-surface pressure projection,
validate cell occupancy, volume, pressure continuity, and surface behavior;
particle shifting does not replace adequate quadrature.

In incompressible MPM, too few particles can leave transiently unsupported
pressure cells that behave like artificial gas pockets. For this project's
free-surface cases, start validation around `q=5--10` per coordinate axis,
then adjust it using volume-fraction and pressure-connectivity diagnostics.
That means 25--100 particles per full 2D cell and 125--1000 per full 3D cell.
High 3D PPC is also the regime where particle-centric global-atomic P2G
contention is strongest. Keep the validated atomic path unless a future
implementation demonstrates an end-to-end speedup at the actual production
PPC.

Size `max_particle_number` for initial generation plus adaptive splitting or
insertion. Size constraint capacities for every boundary type used.

## 6. Validation

Check mass/volume, force balance, energy where applicable, boundary activation,
finite state, pressure-domain connectivity for fluids, convergence histories
for implicit solves, and one analytical or benchmark observable.

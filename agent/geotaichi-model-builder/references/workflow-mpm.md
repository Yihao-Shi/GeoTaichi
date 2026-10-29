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

Native adaptive refinement uses device-resident level fields, incremental node
allocation and interface constraints shared by 2D/3D in
`src/mpm/elements/AdaptiveTopologyKernel.py`. Do not introduce NumPy topology or
particle regrouping into runtime refinement. Host node-map array helpers are
setup/reference/inspection interfaces; only scalar capacity/status counts and
per-material launch bounds return during updates. Preserve stable compact IDs,
independent hanging masters and particle history when optimizing this path.

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

Direct implicit ULMPM optionally accepts `MPM.set_solver.solver.small_mass_ratio`
(finite `[0, 1)`, default zero). A positive value removes grid nodes below a
fraction of the all-node mean mass and compacts every particle connection before
assembly. Check reported discarded mass/momentum and compare thresholds; this
changes interpolation and does not conserve transferred momentum exactly.
Native, explicit and TLMPM paths do not implement this option.

Direct `MPM.add_element({"ShapeFunction": "QuadBSpline", ...})` uses three
nodes per axis, with the existing first/second/penultimate/last-node boundary
polynomials. Each grid axis requires at least four nodes. Particles inside
the grid keep partition of unity and linear reproduction near its boundary;
particles outside still fail. Values, gradients and IPC shape Hessians must
use the same boundary type. In axisymmetry, check an affine radial field
against both radial and hoop stretches and retain the revolved measures.

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

The independent finite-strain `StateDependentDruckerPrager` model is supported
only on a Direct implicit ULMPM child of FEM--MPM, IGA--MPM, or 3D MPM--ABD IPC.
It requires the physical residual/BiCGSTAB path; do not select standalone
Direct stepping, Native MPM, TLMPM, or LSMPM by analogy. See
`src/mpm/README.md` for the SDMC evolution parameters. Ordinary DP is unchanged.
Validation should check the evolving-angle Jacobian by finite differences,
void ratio against total volume, all 14 history entries on rollback/restart,
and independent lagged-friction convergence. Parameter/history adjoints for
this model are unsupported.

Check mass/volume, force balance, energy where applicable, boundary activation,
finite state, pressure-domain connectivity for fluids, convergence histories
for implicit solves, and one analytical or benchmark observable.

For adaptive contact comparisons, match the finest grid spacing, initial
particle quadrature, material, contact law and loading before comparing force
against measured settlement. Use a saved numerical history for historical
controls, with explicit 3D thickness normalization when reducing to plane
strain; a gallery GIF cannot establish force agreement. The maintained
`examples/mpm/Footing/strip_footing_adaptive` workflow compares uniform/L1/L2 separately
and reports a failed or missing historical control as a failed validation.
Current levels share one refinement threshold: another qualifying check
advances an original coarse cell again, rather than applying a separate
second-level threshold.


For the current sensitive-clay paper validation, use the fixed-grid DP case
contract and README. Do not configure AdaptiveGrid while validating this
baseline. Section 4.3 prescribes cohesion 20/4 kPa and slope -50 kPa; the
zero-friction/dilation Inscribed DP control has sqrt(J2)=c and accumulated
strain endpoint 0.32. This is a DP approximation to the paper's Tresca model.
Do not carry the native MC tensor-history sqrt(2) conversion into DP. Keep
elastic initial-stress relaxation and distinguish the fixed original base
from the frictional runout ground. Boundary Norm points out of the domain.
The same self-contained script supports `--material mc` for a fair uniform-grid
control using the corrected MC implementation; use endpoint 0.32/sqrt(2) for
its native tensor-history invariant. Judge approximate agreement by the major
failure sequence, block positions, distances and friction trend, rather than
pixel identity. Obsolete pre-fix MC/adaptive results are not current evidence.

For small-strain softening checks, verify both the strength law and its local
history update. Yield distances must remain signed inside the elastic domain;
refresh local parameters before checking the returned residual. Compare the
plastic modulus against a finite difference along the actual history rate per
plastic multiplier. MC/Mises retain their native tensor-history normalization;
do not silently replace it with DP's accumulated scalar convention. MCC bond
and subloading checks use current flow rates. NorSand is excluded from this
audit's changes at the user's request; retain its original implementation.

# IGA and IGAMPM Workflow Profile

## Contents

1. Pure IGA
2. IGAMPM construction
3. Contact manager lifecycle
4. Explicit DEM-law contact
5. Implicit IPC contact
6. Validation

## 1. Pure IGA

Use `geotaichi.IGA`. Configure dimension, solver family, basis/geometry,
materials, boundary conditions, integration, and run through the current
facade. Trace the selected implicit or explicit backend rather than importing
an older standalone IGA-MPM implementation.

Start from `assets/iga_model_template.py`. Because NURBS primitives and
Dirichlet/Neumann objects cannot be represented in JSON, set
`api.problem_factory.module` and optionally `api.problem_factory.function` in
the contract. The factory receives the full contract and returns
`primitives`, `degree`, `material`, and an optional `boundary` mapping. Build
those objects using the nearest maintained IGA example for the selected
dimension and solver family.

## 2. IGAMPM construction

Configure ordinary `IGA` and `MPM` sub-solvers, then construct
`IGAMPM(iga, mpm)`. The public wrapper can also create missing sub-solvers, but
they still require complete setup before `build()` or `run()`. Select the
contact family first: IPC needs implicit/implicit children, while Linear or
Hertz--Mindlin needs explicit/explicit children. Mixed pairs are rejected
because they do not share a consistent time discretization or unknown vector.

For finite-strain plastic MPM, select the Direct backend, implicit solver,
and `configuration="ULMPM"`. The supported coupling materials are
`DruckerPrager` and associated `VonMises`, alongside elastic `NeoHookean`.
Keep the IGA child elastic. Direct TLMPM plasticity is rejected because it does
not yet own explicit plastic-gradient state.
For Cartesian 2D plastic MPM, Direct ULMPM automatically uses a 3D
plane-strain material map with incremental `F_33=1`; do not request
`plane_strain=False`. This is not an intrinsic-2D yield model.

Cartesian IGA/IGAMPM supports dimensions two and three. For a no-swirl
axisymmetric meridian, configure IGA, Direct implicit ULMPM, and IGAMPM with
`dimension=2`, `axisymmetric=True`, and one shared `axis_offset`. Component
zero is radius and all material/contact quadrature points must satisfy
`R > 0`. Both material blocks use the 3D map with `F_theta_theta=r/R` and
`2*pi*R` reference weights; NURBS closest-point geometry, ACCD, and friction
remain two-dimensional in `(r,z)`.
Require a finite positive reference-patch Jacobian at every quadrature point.
`NeumannBoundary.append` takes resultant control-point forces rather than a
traction density, so an axisymmetric problem must include the `2*pi*R`
boundary measure before appending those values.

## 3. Contact manager lifecycle

Choose the contact mode before building. `freeze_configuration()` prevents
later changes to settings captured by the built contact path.
IPC parameters configure barrier/friction and sparse assembly. Explicit
`Linear`/`HertzMindlin` uses `add_property(MPMmaterial, IGAbody, property)`;
register every active material/patch pair before build. Properties and model
selection are frozen with the built contact path.

## 4. Explicit DEM-law contact

Construct `IGAMPM(..., contact_model="Linear")` or `HertzMindlin` before MPM
particle allocation so the facade selects native `ParticleCoupling` storage
and `coupling="Lagrangian"`. Configure both children as explicit, use 3D
Cartesian geometry, use one positive shared timestep, and mark the intended
MPM points as coupling points through the normal MPM body setup.

Linear accepts fixed `NormalStiffness`/`TangentialStiffness` or the adaptive
`EffectiveModulus`/`NormalToShearRatio` pair, plus static/dynamic friction and
normal/tangential damping. Hertz--Mindlin accepts `ShearModulus`, `Poisson`,
`Restitution`, and friction. Both call the shared production DEM Taichi force
functions. Nonzero rolling friction is invalid because NURBS control points
have translational but no rotational DOFs.

Every explicit step rebuilds current control-hull AABBs, performs all-span
projected Gauss--Newton point--NURBS projection in Taichi, retains tangential
history in a stable particle/surface table, and scatters equal/opposite force
through the rational basis. There is no Verlet multiplier: the usual small
number of whole NURBS boundaries makes AABB culling cheaper than rebuilding
and remapping a neighbor list. The CFL gate covers native MPM wave speed and
the contact law, but IGA currently has no independent spectral estimate.

## 5. Implicit IPC contact

Current solver-family routing is:

- implicit IGA + implicit MPM -> monolithic IPC contact;
- for `contact_model="IPC"`, explicit or mixed solver families -> rejected.

For IPC verify barrier activation distance/stiffness, friction, closest-point
geometry, matrix block assembly, line search, CCD/ACCD, convergence tolerances,
and contact buffer/NNZ capacities. Do not assume matrix symmetry when friction
or another nonsymmetric tangent is active.

IPC control-hull AABB culling keeps every pair but stores a conservative
distance lower bound for far inactive pairs. Active pairs and positive SemiIPC
multipliers retain full projection geometry. A reported inactive minimum is
therefore a lower bound, not necessarily the exact closest distance. ACCD can
skip a moving query only when its relative-motion bound certifies the complete
segment; near/approaching pairs still use the accumulated closest-point loop.
Three-dimensional IPC queries share refreshed knot-span hulls and fixed
Greville coordinates across particles. A fixed-topology span BVH is refitted
after deformation and used to prune span subtrees and locate the exact nearest
control-point seed. Queries also reuse previous closest parameters
as extra seeds. They retain span multistart and boundary searches; seed reuse
does not certify a global minimum by itself. Moving ACCD queries do not reuse
stationary span bounds.

The implicit Newton path checks the residual before assembling a Hessian.
At one unchanged trial, MPM force and tangent share prepared material data;
IGA reference quadrature and fixed topology are cached. Coupled HashTriplet
material blocks scatter into permanent reduced slots, while contact and MPM
remain dynamic. Reference geometry changes require IGA `precompute()` before
reuse. The CPT example's `--resume latest_state.npz` initializes reference data
before loading accepted physical fields and preserves frame numbering in the
same output directory; newer diagnostics are separated from the resumed branch.

Plastic MPM uses the same point--NURBS ACCD, material CCD, Armijo, and sparse
assembly paths as elastic MPM. Its history variables and total `F0` must be
committed only after coupled acceptance and restored together on an acceptance
failure. `IGAMPM.build()` preserves a valid preinitialized `F0`; it initializes
only an all-zero field.

Nonassociated DP accepts independent constant `DilationAngle`. FEM--MPM and
IGA--MPM directly assemble its physical force and complete nonsymmetric
Jacobian, including the derivative of the trial plastic volume. They select
device BiCGSTAB automatically, retain both matrix triangles, and use residual
Armijo with contact/material CCD. No material-consistency outer loop runs for
nonassociated DP. Associated DP retains PCG and its existing plastic-volume
consistency iteration; lagged friction still owns its separate outer loop.

Set the coupled `assemble_type` to `"COO"` or `"HashTriplet"`. Point--NURBS
ACCD must use both point and control-point trial directions and rerun the
closest-point query after each accumulated increment. Screen whole segments
first, then advance active pairs in bounded kernel launches. Pair state stays
on the device; only the active count is read on the host. Reduce final TOCs
only, never intermediate increments. Keep virtual `P_i0 + alpha*dP_i` geometry
without mutating shared control points. This split avoids excessive Taichi 1.7
CFG compilation of the nested ACCD/closest-point loops. Share immutable basis
objects by degree across surfaces to avoid duplicate kernel specializations.
The moving surface evaluator accumulates homogeneous geometry derivatives
and applies the quotient rule directly, avoiding support-sized rational
derivative matrices in the nested closest-point kernel.

## 6. Validation

Check NURBS partition of unity and mapping, assembled residual/tangent,
boundary conditions, closest-point/contact gap, barrier energy-gradient-
Hessian consistency, CCD-limited steps, monolithic convergence, and a known IGA
or contact benchmark. Explicit validation additionally checks pair-property
lookup, closest-point bounds, persistent friction history, equal-and-opposite
force transfer, normal-force moment balance, the finite-radius frictional
couple, and both material/contact timestep estimates.
Axisymmetric validation additionally checks revolved volume/contact measures
and the hoop stretch/tangent.

For implicit prescribed motion at constant speed, use
`DirichletBoundary.append_velocity(dof_id, velocity)`. Do not freeze `v * dt`
in `append` when adaptive retries or output-aligned steps can change `dt`.

For nonassociated DP IPC, `monolithic_inexact_newton=True` optionally adapts the
BiCGSTAB relative tolerance from `0.01` down to the configured
`monolithic_linear_solver_relative_tolerance` floor (which must be at most
`0.01`). It defaults to false. When enabled, acceptance requires the nonlinear
force and Dirichlet criteria; a small correction alone is insufficient. CCD
and residual Armijo remain active. Compare accepted states and residuals from
the same checkpoint before interpreting short-run timing as an acceleration.

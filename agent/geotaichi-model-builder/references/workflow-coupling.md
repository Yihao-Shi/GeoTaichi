# MPDEM, CFDEM, and Coupled Workflow Profile

## Contents

1. Ownership
2. Construction order
3. Lagrangian DEM--MPM
4. CFDEM
5. LSMPM soft--rigid routing
6. Validation
7. FEM--DEM, FEM--LSDEM, and FEM--AffineBody

## 1. Ownership

Use `geotaichi.DEMPM` for current DEM--MPM and CFDEM orchestration. The public
wrapper supports both `DEMPM(dem, mpm)` with existing sub-solvers and
`DEMPM()` followed by configuration through `model.dem` and `model.mpm`. The
coupled facade owns exchange terms, pair capacities, coupled contact/drag
properties, and the combined run lifecycle; it does not make an unsupported
sub-solver branch valid.

## 2. Construction order

```text
construct or receive DEM and MPM sub-solvers
-> construct DEMPM(dem, mpm), or use DEMPM() to create both
-> set coupling and sub-solver configuration
-> set the shared solver controls
-> allocate both sub-scenes and coupling buffers
-> add contact or drag properties
-> choose coupling/contact models
-> select output
-> coupled run
```

The exact order of coupling versus sub-solver `set_configuration` depends on
which owner supplies the domain; maintained examples commonly construct
`DEMPM()` first. In all forms, both sub-solvers must have coupling enabled and
must be fully configured before allocation/build/run. Check whether a facade
call forwards to both sub-solvers or belongs only to the coupling layer. Avoid
duplicated allocation or setup.

Use `assets/mpdem_model_template.py` for a new Lagrangian MPDEM model. The
older `assets/coupled_model_template.py` remains a generic compatibility
scaffold and must not be treated as evidence that MPDEM and CFDEM dictionaries
are interchangeable.

## 3. Lagrangian DEM--MPM

Verify:

- coupling scheme and which particles/bodies participate;
- MPM polygons/regions used by contact exchange;
- DEM particle/body and wall interactions;
- contact model and every material pair;
- normal/tangential force, friction, and history ownership;
- sparse MPM compatibility for the selected Lagrangian route;
- coupled explicit timestep bound.

Arbitrary implicit solid MPM coupling is not implied by the existence of an
implicit MPM solver. Follow current validation restrictions.

## 4. CFDEM

For fluid--particle coupling, distinguish unresolved/explicit drag paths from
semi-resolved incompressible sphere--fluid coupling. Verify:

- `coupling_scheme="CFDEM"` routing;
- supported fluid MPM branch and DEM particle type;
- drag, buoyancy, pressure-gradient, and virtual-mass models;
- porosity/void-fraction or solid-fraction mapping;
- coupling cadence and force units;
- domain overlap and boundary behavior;
- particle and wall coupling contact-list capacities.

Do not copy a drag coefficient between closures without checking its Reynolds,
porosity, diameter/radius, and unit conventions.

The current implicit 3D incompressible FDM route has two geometry-specific
branches. Ordinary DEM spheres use dense semi-resolved coupling: Gaussian
solid fraction enters the porosity continuity projection, and drag reaction is
applied with porosity-weighted MAC mass. Rigid LSDEM bodies use the fully
resolved SDF volume-fraction IBM with fictitious interior fluid, mixed density,
and the Eq. (28) pressure/viscous/IBM resultant. A moving LSDEM body is not a
cut-cell solid; `solid_sdf_cut_cell=True` remains meaningful for separately
declared fixed `SolidCell` walls. Start LSDEM force studies near ten cells per
particle diameter and require a refinement check.

## 5. LSMPM soft--rigid routing

The coupled facade can detect a DEM-owned LSMPM soft/rigid scene and route to
its dedicated runner. This query is orchestration, not configuration. Build the
DEM LSMPM scene correctly, keep the ordinary MPM allocation empty where the
route requires it, install recorder/output through the public workflow, and use
the critical-timestep check. Mixed soft--AffineBody IPC has separate implicit
solver/contact ownership and backend restrictions.

## 6. Validation

In addition to each sub-solver's checks, validate equal-and-opposite exchange,
total mixture momentum, drag/contact power, porosity bounds, no duplicate pair
forces, coupled timestep, output synchronization, and one coupled benchmark.

## 7. FEM--DEM, FEM--LSDEM, and FEM--AffineBody

Use `geotaichi.FEDEM(dem, fem)` (alias `DEMFEM`) for three supported routes:

- ordinary DEM spheres contacting a deforming explicit FEM boundary;
- rigid LSDEM signed-distance bodies contacting an explicit FEM boundary;
- AffineBody surface meshes contacting elastic implicit FEM through a
  monolithic IPC solve.

Configure and populate DEM and FEM first, then set coupling
configuration/solver, select `add_surface`, allocate
`contact_coordination_number`, `max_contact_pairs`, and the linked-cell
`max_facet_cell_pairs` or LSDEM `max_levelset_cell_pairs` where applicable.
The explicit routes choose `Linear` or `HertzMindlin` and add every active
`DEMmaterial`/`FEMbody` property.
The AffineBody route chooses `IPC` and may add pair-local
`AffineBody`/`FEMbody` barrier plus `friction_coefficient`/`epsv` parameters.

The FEM is not converted to an SDF. Its soft-particle volume remains TET4 or
HEX8 and its contact representation is a TRI3 boundary. In the LSDEM route,
only the rigid LSDEM child owns an SDF; current FEM boundary nodes query that
rigid field using current lumped nodal area and scatter opposite force/torque.
In the AffineBody route neither side needs an SDF because contact uses analytic
PT/EE distances.

`search="LinkedCell"` and `search="BVH"` are supported. The sphere route
retains particle/face tangential history across list rebuilds. LSDEM rebuilds
its broad phase each explicit step because rigid rotation can change the SDF
surface without center displacement. Affine IPC rebuilds current and swept
candidates for Newton and CCD/ACCD queries and has no Verlet multiplier.

The explicit routes require explicit FEM and must not simultaneously enable
internal IPC/AL. Explicit HEX8 elastoplastic FEM is compatible because penalty
contact is separate. Affine IPC requires elastic implicit FEM, uses analytic
barrier Hessians with PSD projection and Armijo, and supports `COO` or
`HashTriplet` with device `PCG` or explicit host `Scipy`. Mixed
FEM--AffineBody contact supports lagged IPC friction and positive coupled
`friction_iterations`; PT/EE frames and barrier normal forces refresh between
fixed-point solves while the step-start positions stay frozen. Fully implicit
mixed friction is unavailable, and AffineBody self-friction must remain lagged
with its own single friction iteration.

For an executable volume-soft-particle pair, start from
`examples/fedem/ExplicitLevelSetSoftParticle/explicit_levelset_soft_particle.py` for the LSDEM `gapn`
penalty route and `examples/fedem/ImplicitAffineIPCSoftParticle/implicit_affine_ipc_soft_particle.py` for
the AffineBody IPC route. They intentionally share the low-poly sphere asset
and TET4 body geometry while selecting the scheme-appropriate contact
representation. The implicit example's first step includes Taichi JIT of the
analytic FEM/contact Hessians; do not diagnose that preparation time as a
per-step Newton or CCD stall.

Validation must check the circular sphere/triangle area fraction,
equal-and-opposite nodal transfer, torque convention, history across a list
rebuild, rigid-SDF nodal action/reaction, PT/EE IPC barrier/friction energy and
symmetric PSD Hessian assembly, mixed CCD, pair-local friction and fixed-point
refresh, capacity overflow, and the appropriate explicit critical
timestep or implicit Newton/line-search transaction. The Affine IPC regression
must include a TET4/HEX8 volume body rather than relying only on a TRI3
membrane.

For the executable FEM soft-particle--rigid LSDEM route, encode at minimum one
contract observable, a signed-gap/penetration check, and an action--reaction,
momentum, energy, or force-balance check. The `fedem` policy in
`physics-validation-rubric.json` scores these as separate requirements; a good
benchmark match cannot conceal penetration or an unbalanced exchange force.

## 8. FEM--MPM

Use `geotaichi.FEMPM(fem, mpm)` (alias `MPMFEM`) for MPM material points
contacting a deforming explicit FEM boundary. Construct the coupling before
MPM allocates particle fields so the public facade can select
`ParticleCoupling`. Configure and populate MPM and FEM, select `add_surface`,
allocate `contact_coordination_number` and `max_contact_pairs` (plus
`max_facet_cell_pairs` for `LinkedCell`), choose `Linear` or `HertzMindlin`,
and add every active `MPMmaterial`/`FEMbody` property.

`search="LinkedCell"` uses per-rebuild triangle/cell counting and a prefix
sum. `search="BVH"` rebuilds a Morton-code LBVH over current triangle AABBs.
Both paths feed the same exact point--triangle cull, history storage, and DEM
contact law. Contact force is accumulated in the MPM point's
`external_force` before P2G and with opposite sign on FEM nodes.

Validation must exercise both broad phases, equal-and-opposite transfer,
history across rebuild, material-id/body-id property lookup, overflow, and the
minimum MPM/FEM/contact timestep.

For implicit IPC, configure both children as implicit, select
`mpm_backend="Direct"` with `configuration="ULMPM"`, and choose Neo-Hookean
elasticity, associated finite-strain Drucker--Prager, or associated
finite-strain von Mises on the MPM side. Then call
`choose_contact_model("IPC", dhat=..., kappa=...,
friction_coefficient=..., friction_mode="lagged")`. The coupled solver owns a
single FEM-node/active-MPM-grid Newton system, analytic PT Hessian pullback,
PSD projection, CCD/ACCD and Armijo. `assemble_type` is `COO` or
`HashTriplet`; `linear_solver="PCG"` stays on the Taichi device and `Scipy` is
the explicit host option. Current/swept candidates are rebuilt per query, so
implicit IPC has no Verlet multiplier. FEM remains elastic. The DP path
requires `DilationAngle == FrictionAngle` and is perfect plastic; von Mises
accepts optional `HardeningModulus`. Both plastic models require Direct ULMPM,
default to PSD projection, and cannot be combined with standalone child IPC.
The FEM internal formulation remains Total Lagrangian with a fixed rest shape;
Direct plastic MPM is Updated Lagrangian, and IPC contact is evaluated in the
current spatial configuration.
Cartesian 2D DP/von-Mises MPM uses a 3D plane-strain material embedding with
incremental `F_33=1`; `plane_strain=False` is rejected for those laws. The
coupled step snapshots FEM and MPM physical/plastic state on the device and
restores it if nonlinear solution or post-solve commit fails.

Implicit IPC additionally supports Cartesian 2D and no-swirl axisymmetry.
Set `dimension=2` on both children. For axisymmetry also set
`axisymmetric=True` and the same `axis_offset` on FEM, Direct MPM, and FEMPM;
component zero is radius and all reference quadrature/contact radii must be
positive. The 2D contact primitive is point--edge with analytic derivatives
and CCD/ACCD. Axisymmetric FEM/MPM material maps include
`F_theta_theta=r/R`, and reference volume/contact measures include `2*pi*R`.
The DEM-style explicit branch remains 3D only.

## 9. Explicit IGA--MPM

Use `geotaichi.IGAMPM(iga, mpm, contact_model="Linear")` or
`HertzMindlin` for MPM material points contacting deforming NURBS boundaries.
Construct the coupling before MPM particle allocation, configure both children
as explicit, use the native MPM backend in Lagrangian coupling mode, and add
every active `MPMmaterial`/`IGAbody` pair property before build. The current
explicit route is 3D Cartesian; 2D and axisymmetry remain on monolithic IPC.

The broad phase rebuilds one control-hull AABB per selected whole NURBS
boundary each step. A stable particle/surface table carries tangential history,
so no Verlet multiplier or neighbor-list remap is needed for the usual small
surface count. All-span projected Gauss--Newton geometry and the shared DEM
Linear/Hertz--Mindlin force functions run in Taichi. Rational-basis scatter
puts equal and opposite generalized force on the IGA control points.

Validation checks closest-point coverage across knot spans, material/patch
property lookup, tangential-history persistence, total force, normal-force
moment balance, the unresolved finite-radius tangential couple, moving-surface
relative velocity, and the MPM/contact CFL estimates. The current IGA explicit
backend has no separate spectral timestep estimator, so an IGA
refinement/stability study remains required.
